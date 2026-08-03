"""
tests/test_sequence_parallel_v03.py
=====================================

Tests for Sequence Parallelism's fused projection path and non-divisible
sequence length support (ADR-007).

History (see ADR-007 for the full writeup)
--------------------------------------------
The original v0.3 fused path called ``dist.all_reduce`` after projecting a
per-rank sequence shard through a *replicated* weight. That's wrong: each
rank's shard covers a *different* slice of tokens, so all-reducing sums
different tokens' projections together, which reconstructs nothing. The
one test that would have caught this (``test_sp_fused_2rank_numerically_correct``)
crashed with a shape-mismatch ``RuntimeError`` rather than a numerical
failure -- and had never actually run in CI, since every job filters out
``2rank`` tests by name. Fixed in ``pkg/distributed/sequence_parallel.py``:
the fused path now does the (correct) zero-collective local projection, and
this test's reference comparison was corrected to compare each rank's shard
against the matching *slice* of the full reference, not the whole thing.

Tests cover:
  * ``_sp_shard_layout`` padding arithmetic in isolation (pure function, no
    dist needed) across evenly-divisible, non-divisible, and edge cases
  * Fused path returns correct shape/values at tp_size=1 (no dist)
  * Original scatter-only path is unchanged (next_weight=None)
  * Round-trip: scatter(next_weight=None) -> gather recovers original
  * 2-rank mp.spawn: fused path matches the correct per-rank reference slice
  * 2-rank mp.spawn: fused path issues zero collectives (regression guard
    for the bug above -- this is the test that would have caught it fastest)
  * 2-rank mp.spawn: non-divisible sequence length round-trips exactly
    through scatter -> gather(orig_seq_len=...), plain and fused
"""

from __future__ import annotations

import os
import socket
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pkg.distributed.parallel_mesh import (  # noqa: E402
    build_topology,
    gather_from_sequence_parallel,
    scatter_to_sequence_parallel,
)
from pkg.distributed.sequence_parallel import _sp_shard_layout  # noqa: E402

pytestmark = pytest.mark.cpu

# ---------------------------------------------------------------------------
# _sp_shard_layout: pure padding arithmetic, no dist needed
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "seq_len,tp_size,expected_chunk,expected_pad",
    [
        (16, 4, 4, 0),  # evenly divisible
        (100, 10, 10, 0),  # evenly divisible
        (8, 1, 8, 0),  # tp_size=1: chunk=seq_len, no padding
        (7, 2, 4, 1),  # non-divisible: 7 -> ceil(7/2)=4 per rank, pad 1
        (1001, 8, 126, 7),  # non-divisible: 1001 -> ceil=126, 126*8=1008, pad=7
        (3, 8, 1, 5),  # fewer tokens than ranks: each rank gets 1, 5 are pure padding
        (1, 1, 1, 0),  # smallest possible, no padding
    ],
)
def test_shard_layout(seq_len, tp_size, expected_chunk, expected_pad):
    chunk, pad = _sp_shard_layout(seq_len, tp_size)
    assert chunk == expected_chunk
    assert pad == expected_pad
    # Invariant that must hold regardless of the specific numbers: padding
    # to `chunk * tp_size` must always land on exactly `seq_len + pad`.
    assert chunk * tp_size == seq_len + pad
    # And padding must never be enough to add a whole extra chunk (that
    # would mean chunk was computed too large).
    assert 0 <= pad < tp_size


@pytest.mark.parametrize("bad_tp_size", [0, -1])
def test_shard_layout_rejects_invalid_tp_size(bad_tp_size):
    with pytest.raises(ValueError):
        _sp_shard_layout(16, bad_tp_size)


@pytest.mark.parametrize("bad_seq_len", [0, -1])
def test_shard_layout_rejects_invalid_seq_len(bad_seq_len):
    with pytest.raises(ValueError):
        _sp_shard_layout(bad_seq_len, 4)


# ---------------------------------------------------------------------------
# Single-process tests (tp_size=1, no dist) — unaffected by ADR-007
# ---------------------------------------------------------------------------


def test_scatter_fused_shape_tp1():
    """Fused path returns [B, S, out_features] at tp=1 (no shard)."""
    topo = build_topology(dp_size=1, tp_size=1, ep_size=1, device_type="cpu")
    B, S, H, F = 2, 16, 32, 64
    x = torch.randn(B, S, H)
    w = torch.randn(F, H)
    out = scatter_to_sequence_parallel(x, topo, next_weight=w)
    assert out.shape == (B, S, F), f"Expected ({B},{S},{F}), got {out.shape}"


def test_scatter_fused_matches_linear_tp1():
    """Fused path output must equal nn.functional.linear(x, w) at tp=1."""
    torch.manual_seed(0)
    topo = build_topology(dp_size=1, tp_size=1, ep_size=1, device_type="cpu")
    B, S, H, F = 2, 8, 32, 64
    x = torch.randn(B, S, H)
    w = torch.randn(F, H)
    fused = scatter_to_sequence_parallel(x, topo, next_weight=w)
    ref = torch.nn.functional.linear(x, w)
    assert torch.allclose(fused, ref, atol=1e-6), (
        f"Fused SP output diverges from reference at tp=1. max_diff={abs(fused - ref).max():.2e}"
    )


def test_scatter_identity_tp1():
    """Without next_weight, scatter returns the original tensor at tp=1."""
    topo = build_topology(dp_size=1, tp_size=1, ep_size=1, device_type="cpu")
    x = torch.randn(4, 16, 32)
    out = scatter_to_sequence_parallel(x, topo, next_weight=None)
    assert torch.equal(out, x)


def test_gather_identity_tp1():
    topo = build_topology(dp_size=1, tp_size=1, ep_size=1, device_type="cpu")
    x = torch.randn(4, 64, 32)
    assert torch.equal(gather_from_sequence_parallel(x, topo), x)


def test_scatter_gather_round_trip_tp1():
    topo = build_topology(dp_size=1, tp_size=1, ep_size=1, device_type="cpu")
    x = torch.randn(2, 64, 32)
    recovered = gather_from_sequence_parallel(
        scatter_to_sequence_parallel(x, topo, next_weight=None), topo
    )
    assert torch.equal(recovered, x)


def test_gather_orig_seq_len_mismatch_at_tp1_raises():
    """At tp_size=1, scatter never pads (nothing to pad for). Passing an
    orig_seq_len that doesn't match x's actual length can only be a caller
    bug -- this must raise loudly, not silently return the wrong shape."""
    topo = build_topology(dp_size=1, tp_size=1, ep_size=1, device_type="cpu")
    x = torch.randn(2, 10, 32)
    with pytest.raises(ValueError):
        gather_from_sequence_parallel(x, topo, orig_seq_len=7)


def test_scatter_fused_no_nan():
    """Fused path must never produce NaN."""
    topo = build_topology(dp_size=1, tp_size=1, ep_size=1, device_type="cpu")
    x = torch.randn(4, 32, 64)
    w = torch.randn(128, 64)
    out = scatter_to_sequence_parallel(x, topo, next_weight=w)
    assert not torch.isnan(out).any()


@pytest.mark.parametrize(
    "B,S,H,F",
    [
        (1, 4, 16, 32),
        (2, 8, 64, 128),
        (4, 16, 32, 64),
    ],
)
def test_scatter_fused_parametrised(B, S, H, F):
    topo = build_topology(dp_size=1, tp_size=1, ep_size=1, device_type="cpu")
    x = torch.randn(B, S, H)
    w = torch.randn(F, H)
    out = scatter_to_sequence_parallel(x, topo, next_weight=w)
    ref = torch.nn.functional.linear(x, w)
    assert torch.allclose(out, ref, atol=1e-6)
    assert out.shape == (B, S, F)


# ---------------------------------------------------------------------------
# Multi-rank SP (2-rank mp.spawn)
# ---------------------------------------------------------------------------


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _run_2rank(worker, *args, timeout=60):
    """Shared 2-process mp.spawn harness. Returns {rank: result_dict}."""
    port = _free_port()
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [ctx.Process(target=worker, args=(r, 2, port, q, *args)) for r in range(2)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=timeout)
        assert p.exitcode == 0, f"Worker exited {p.exitcode}"
    results = {}
    while not q.empty():
        r = q.get_nowait()
        results[r["rank"]] = r
    assert len(results) == 2
    return results


def _sp_fused_worker(
    rank: int, world_size: int, port: int, result_queue, B: int, S: int, H: int, F: int, seed: int
):
    """
    Each rank holds a shard of the sequence [B, ceil(S/tp), H] (evenly
    divisible here, so no padding). Fused path:
    scatter_to_sequence_parallel(x_full, topo, next_weight=w).

    Correct comparison (post-ADR-007): each rank's fused_out is a SHARD, so
    it's compared against the matching SLICE of the full-sequence reference
    -- not the whole reference, which is what the original, broken version
    of this test did.
    """
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    torch.manual_seed(seed)
    topo = build_topology(dp_size=1, tp_size=world_size, ep_size=1, device_type="cpu")

    x_full = torch.randn(B, S, H)
    w = torch.randn(F, H)

    with torch.no_grad():
        fused_out = scatter_to_sequence_parallel(x_full, topo, next_weight=w)
        ref_full = torch.nn.functional.linear(x_full, w)
        chunk = S // world_size
        ref_shard = ref_full[:, rank * chunk : (rank + 1) * chunk, :]

    max_diff = float((fused_out - ref_shard).abs().max().item())
    result_queue.put(
        {
            "rank": rank,
            "max_diff": max_diff,
            "shape_ok": fused_out.shape == ref_shard.shape,
            "error": None,
        }
    )
    dist.destroy_process_group()


@pytest.mark.skipif(sys.platform == "darwin", reason="mp.spawn fork-safety on macOS CI")
def test_sp_fused_2rank_numerically_correct():
    """2-rank TP: fused SP path must match the corresponding slice of the
    full-sequence reference to atol=1e-5. (Previously compared against the
    *whole* reference and crashed on a shape mismatch before it could even
    get to a numerical comparison -- see ADR-007.)"""
    results = _run_2rank(_sp_fused_worker, 2, 8, 32, 64, 13)
    for rank, res in results.items():
        assert res["error"] is None, f"rank {rank}: {res['error']}"
        assert res["shape_ok"], f"rank {rank}: fused_out shape didn't match the reference slice"
        assert res["max_diff"] < 1e-5, (
            f"rank {rank}: fused SP diverges from reference by {res['max_diff']:.2e}"
        )


def _sp_no_collective_worker(
    rank: int, world_size: int, port: int, result_queue, B: int, S: int, H: int, F: int, seed: int
):
    """Regression guard for the actual bug in ADR-007: the fused path must
    issue ZERO collectives. Patches dist.all_reduce/all_gather* to detect
    any use during the fused scatter call -- this is the test that would
    have caught the original bug immediately (a wrong collective call),
    rather than downstream via a shape crash."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    torch.manual_seed(seed)
    topo = build_topology(dp_size=1, tp_size=world_size, ep_size=1, device_type="cpu")
    x_full = torch.randn(B, S, H)
    w = torch.randn(F, H)

    with (
        patch.object(dist, "all_reduce", wraps=dist.all_reduce) as spy_all_reduce,
        patch.object(
            dist, "all_gather_into_tensor", wraps=dist.all_gather_into_tensor
        ) as spy_gather,
        patch.object(dist, "all_gather", wraps=dist.all_gather) as spy_gather_list,
    ):
        with torch.no_grad():
            scatter_to_sequence_parallel(x_full, topo, next_weight=w)
        collectives_called = (
            spy_all_reduce.call_count + spy_gather.call_count + spy_gather_list.call_count
        )

    result_queue.put({"rank": rank, "collectives_called": collectives_called, "error": None})
    dist.destroy_process_group()


@pytest.mark.skipif(sys.platform == "darwin", reason="mp.spawn fork-safety on macOS CI")
def test_sp_fused_2rank_no_collective_used():
    """The fused path (replicated weight) must use zero collectives -- see
    module docstring and ADR-007. This is the fastest possible regression
    guard against reintroducing the original bug: if anyone adds back an
    all_reduce/all_gather here, this fails immediately and explicitly,
    rather than surfacing later as a shape crash or (worse) a silent wrong
    answer."""
    results = _run_2rank(_sp_no_collective_worker, 2, 8, 32, 64, 21)
    for rank, res in results.items():
        assert res["error"] is None, f"rank {rank}: {res['error']}"
        assert res["collectives_called"] == 0, (
            f"rank {rank}: fused SP path used {res['collectives_called']} "
            "collective(s) -- expected 0. See ADR-007."
        )


def _sp_nondivisible_worker(
    rank: int, world_size: int, port: int, result_queue, B: int, S: int, H: int, F: int, seed: int
):
    """S is deliberately NOT divisible by world_size. Verifies both the
    plain (next_weight=None) and fused round-trips recover the exact
    original sequence once orig_seq_len is passed to gather."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    torch.manual_seed(seed)
    topo = build_topology(dp_size=1, tp_size=world_size, ep_size=1, device_type="cpu")
    x_full = torch.randn(B, S, H)
    w = torch.randn(F, H)

    with torch.no_grad():
        # Plain round-trip.
        shard = scatter_to_sequence_parallel(x_full, topo)
        recovered = gather_from_sequence_parallel(shard, topo, orig_seq_len=S)
        plain_max_diff = float((recovered - x_full).abs().max().item())
        plain_shape_ok = recovered.shape == x_full.shape

        # Fused round-trip: project every (real + padding) token locally,
        # gather, strip padding, and compare against projecting the full
        # original sequence directly -- this is the strongest end-to-end
        # check, since it validates padding + fused projection + gather +
        # strip all agree with the un-sharded ground truth simultaneously.
        fused_shard = scatter_to_sequence_parallel(x_full, topo, next_weight=w)
        fused_recovered = gather_from_sequence_parallel(fused_shard, topo, orig_seq_len=S)
        ref = torch.nn.functional.linear(x_full, w)
        fused_max_diff = float((fused_recovered - ref).abs().max().item())
        fused_shape_ok = fused_recovered.shape == ref.shape

    result_queue.put(
        {
            "rank": rank,
            "plain_max_diff": plain_max_diff,
            "plain_shape_ok": plain_shape_ok,
            "fused_max_diff": fused_max_diff,
            "fused_shape_ok": fused_shape_ok,
            "error": None,
        }
    )
    dist.destroy_process_group()


@pytest.mark.skipif(sys.platform == "darwin", reason="mp.spawn fork-safety on macOS CI")
def test_sp_nondivisible_2rank_round_trip():
    """S=7 is not divisible by world_size=2 (the exact class of input the
    original scatter_to_sequence_parallel raised ValueError for). Both the
    plain and fused paths must round-trip exactly back to the un-padded
    original once orig_seq_len is supplied to gather."""
    results = _run_2rank(_sp_nondivisible_worker, 2, 7, 16, 32, 7)
    for rank, res in results.items():
        assert res["error"] is None, f"rank {rank}: {res['error']}"
        assert res["plain_shape_ok"], f"rank {rank}: plain round-trip shape mismatch"
        assert res["plain_max_diff"] < 1e-6, (
            f"rank {rank}: plain round-trip diverged by {res['plain_max_diff']:.2e}"
        )
        assert res["fused_shape_ok"], f"rank {rank}: fused round-trip shape mismatch"
        assert res["fused_max_diff"] < 1e-5, (
            f"rank {rank}: fused round-trip diverged by {res['fused_max_diff']:.2e}"
        )


@pytest.mark.skipif(sys.platform == "darwin", reason="mp.spawn fork-safety on macOS CI")
def test_sp_fewer_tokens_than_ranks_2rank():
    """Extreme non-divisible case: S=3 tokens across world_size=2 ranks (one
    rank ends up with a shard that's half real tokens, half padding, but
    neither rank is purely padding at world_size=2). Must not crash, and
    must still round-trip exactly."""
    results = _run_2rank(_sp_nondivisible_worker, 2, 3, 8, 16, 3)
    for rank, res in results.items():
        assert res["error"] is None, f"rank {rank}: {res['error']}"
        assert res["plain_shape_ok"] and res["plain_max_diff"] < 1e-6
        assert res["fused_shape_ok"] and res["fused_max_diff"] < 1e-5


def _sp_gather_batch_worker(
    rank: int, world_size: int, port: int, result_queue, B: int, S: int, H: int, seed: int
):
    """Regression guard for the gather_from_sequence_parallel batch-
    corruption bug (ADR-007): dist.all_gather_into_tensor concatenates
    along dim 0 only, but the output buffer used to be shaped as if
    concatenation happened along the sequence dimension. That's wrong for
    any B > 1 -- and B=1 is not a realistic training batch size, so this
    is deliberately the *first* test for gather_from_sequence_parallel at
    real multi-rank, checked independently of the non-divisible-length
    tests so a future regression here isn't only caught incidentally."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    torch.manual_seed(seed)
    topo = build_topology(dp_size=1, tp_size=world_size, ep_size=1, device_type="cpu")
    # Evenly divisible S -- deliberately not the non-divisible case, so this
    # test isolates the batch-dimension bug from the padding logic.
    x_full = torch.randn(B, S, H)

    with torch.no_grad():
        shard = scatter_to_sequence_parallel(x_full, topo)
        recovered = gather_from_sequence_parallel(shard, topo)

    # Check each batch element independently -- the original bug corrupted
    # data *across* the batch dimension, so an aggregate max-diff could in
    # principle mask a per-batch-element swap that happens to cancel out in
    # magnitude. It doesn't here, but checking per-element is the honest
    # version of this test regardless.
    per_batch_max_diff = [float((recovered[b] - x_full[b]).abs().max().item()) for b in range(B)]
    result_queue.put(
        {
            "rank": rank,
            "shape_ok": recovered.shape == x_full.shape,
            "per_batch_max_diff": per_batch_max_diff,
            "error": None,
        }
    )
    dist.destroy_process_group()


@pytest.mark.skipif(sys.platform == "darwin", reason="mp.spawn fork-safety on macOS CI")
def test_sp_gather_2rank_batch_dimension_not_corrupted():
    """B=4 (a realistic-ish batch size), S evenly divisible by tp_size=2.
    Every batch element must round-trip through scatter -> gather exactly,
    independently of every other batch element."""
    results = _run_2rank(_sp_gather_batch_worker, 4, 16, 32, 11)
    for rank, res in results.items():
        assert res["error"] is None, f"rank {rank}: {res['error']}"
        assert res["shape_ok"], f"rank {rank}: gather output shape mismatch"
        for b, diff in enumerate(res["per_batch_max_diff"]):
            assert diff < 1e-6, f"rank {rank}, batch element {b}: diverged by {diff:.2e}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
