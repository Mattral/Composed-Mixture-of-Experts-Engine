"""
tests/test_sequence_parallel_v03.py
=====================================

v0.3 tests for the fused Sequence Parallelism path.

The v0.3 SP upgrade adds a ``next_weight`` parameter to
``scatter_to_sequence_parallel``.  When provided, instead of returning the
scattered shard [B, S//tp, H] and requiring a separate all_gather downstream,
it fuses the backward all-gather with the output projection matmul:

    out = all_reduce(shard @ next_weight.T)

This halves the number of SP collectives per layer at tp_size > 1.

Tests cover:
  * Fused path returns correct shape at tp_size=1 (no dist)
  * Fused path output matches the non-fused reference at tp_size=1
  * Original scatter-only path is unchanged (next_weight=None)
  * Round-trip: scatter(next_weight=None) → gather recovers original
  * 2-rank mp.spawn test: fused path matches single-rank reference matmul
"""

from __future__ import annotations

import os
import socket
import sys
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pkg.distributed.parallel_mesh import (
    build_topology,
    gather_from_sequence_parallel,
    scatter_to_sequence_parallel,
)


# ---------------------------------------------------------------------------
# Single-process tests (tp_size=1, no dist)
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
        f"Fused SP output diverges from reference at tp=1. max_diff={abs(fused-ref).max():.2e}"
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


def test_scatter_fused_no_nan():
    """Fused path must never produce NaN."""
    topo = build_topology(dp_size=1, tp_size=1, ep_size=1, device_type="cpu")
    x = torch.randn(4, 32, 64)
    w = torch.randn(128, 64)
    out = scatter_to_sequence_parallel(x, topo, next_weight=w)
    assert not torch.isnan(out).any()


@pytest.mark.parametrize("B,S,H,F", [
    (1, 4, 16, 32),
    (2, 8, 64, 128),
    (4, 16, 32, 64),
])
def test_scatter_fused_parametrised(B, S, H, F):
    topo = build_topology(dp_size=1, tp_size=1, ep_size=1, device_type="cpu")
    x = torch.randn(B, S, H)
    w = torch.randn(F, H)
    out = scatter_to_sequence_parallel(x, topo, next_weight=w)
    ref = torch.nn.functional.linear(x, w)
    assert torch.allclose(out, ref, atol=1e-6)
    assert out.shape == (B, S, F)


# ---------------------------------------------------------------------------
# Multi-rank SP fused path (2-rank mp.spawn)
# ---------------------------------------------------------------------------

def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _sp_fused_worker(rank: int, world_size: int, port: int, result_queue, B: int, S: int, H: int, F: int, seed: int):
    """
    Each rank holds a shard of the sequence [B, S//tp, H].
    Fused path: scatter_to_sequence_parallel(x_full, topo, next_weight=w)
    Reference:  nn.functional.linear(x_full, w)  (run on every rank with full x)

    The fused path must match the reference output to atol=1e-5.
    """
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    torch.manual_seed(seed)
    topo = build_topology(dp_size=1, tp_size=world_size, ep_size=1, device_type="cpu")

    x_full = torch.randn(B, S, H)
    w = torch.randn(F, H)

    with torch.no_grad():
        # Fused SP path: scatter shards x by sequence, compute local proj, all_reduce
        fused_out = scatter_to_sequence_parallel(x_full, topo, next_weight=w)
        # Reference: plain matmul on full sequence
        ref_out = torch.nn.functional.linear(x_full, w)

    max_diff = float((fused_out - ref_out).abs().max().item())
    result_queue.put({"rank": rank, "max_diff": max_diff, "error": None})
    dist.destroy_process_group()


@pytest.mark.skipif(sys.platform == "darwin", reason="mp.spawn fork-safety on macOS CI")
def test_sp_fused_2rank_numerically_correct():
    """2-rank TP: fused SP path must match full-sequence reference to atol=1e-5."""
    B, S, H, F, seed = 2, 8, 32, 64, 13
    port = _free_port()
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [
        ctx.Process(target=_sp_fused_worker, args=(r, 2, port, q, B, S, H, F, seed))
        for r in range(2)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=60)
        assert p.exitcode == 0, f"Worker exited {p.exitcode}"

    results = {}
    while not q.empty():
        r = q.get_nowait()
        results[r["rank"]] = r

    assert len(results) == 2
    for rank, res in results.items():
        assert res["error"] is None, f"rank {rank}: {res['error']}"
        assert res["max_diff"] < 1e-5, (
            f"rank {rank}: fused SP diverges from reference by {res['max_diff']:.2e}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
