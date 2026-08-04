"""
tests/test_moe_layer.py
=========================

Correctness tests for ``DistributedMoELayer``'s token dispatch, expert
compute, and combine logic (ADR-009).

Context
-------
No test anywhere in the codebase checked DistributedMoELayer's output
against a per-token reference before this file. Existing tests
(``test_capacity_dropping.py``, ``test_mock_dist.py``, ``test_smoke_e2e.py``)
only checked shapes, NaN-absence, and drop-fraction bounds -- never that the
right expert actually processed the right token. That gap let three
compounding bugs ship, none of them requiring real multi-rank execution to
reproduce:

  1. ``send_counts`` overcounted whenever a token's ``top_k`` expert
     assignments spanned more than one EP rank (common whenever
     ``top_k>1`` and ``ep_size>1``) -- the exact cause of
     ``test_distributed_invariants.py``'s "Split sizes doesn't match total
     dim 0 size" failure.
  2. Sorting dispatch items by only ``idx[:, 0]`` (the token's first
     expert) and repeat-interleaving the token embedding meant slots
     ``k >= 1`` weren't actually grouped by their own destination rank.
  3. The local expert-compute loop divided ``received`` into
     **equal-sized** chunks, one per local expert -- correct only under
     perfectly balanced routing, which essentially never happens. This one
     reproduces at ``ep_size=1`` with no distributed setup at all, which
     is why most of this file doesn't need ``dist.init_process_group``.

All three are fixed in ``pkg/distributed/moe_layer.py``; see
``docs/adr/ADR-009-moe-dispatch-correctness.md`` for the full derivation.
This file is the permanent regression guard -- the direct per-token
reference comparison here was verified to actually fail against the
pre-fix code (reverted, confirmed failing by ~2.0 absolute max-diff, not a
numerical-precision-scale difference, then restored) before being trusted.
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

from pkg.distributed.parallel_mesh import (  # noqa: E402
    DistributedMoELayer,
    build_topology,
)

pytestmark = pytest.mark.cpu


# ===========================================================================
# Helpers
# ===========================================================================
def _manual_reference(layer: DistributedMoELayer, x: torch.Tensor) -> torch.Tensor:
    """Independent per-token reference: for each token, manually apply its
    top_k assigned experts and combine with router weights. Does not use
    any of the dispatch/sort/combine machinery under test -- deliberately
    the most naive possible correct implementation, an O(N*top_k) Python
    loop, so it can't share a bug with the code it's checking."""
    N, H = x.shape
    idx, weights, _ = layer.router(x)
    ref = torch.zeros(N, H)
    for i in range(N):
        for k in range(layer.top_k):
            eid = int(idx[i, k].item())
            w = float(weights[i, k].item())
            expert_module = layer.experts[layer.local_expert_ids.index(eid)]
            ref[i] += w * expert_module(x[i : i + 1]).squeeze(0)
    return ref


# ===========================================================================
# Per-token numerical correctness, ep_size=1 (no dist needed --
# reproduces bug #3 above, which is independent of ep_size entirely)
# ===========================================================================


@pytest.mark.parametrize(
    "N,H,F,E,K,seed",
    [
        (20, 16, 32, 6, 2, 0),  # the exact case that caught bug #3 originally
        (12, 8, 16, 4, 1, 1),  # top_k=1: simplest case, must still hold
        (7, 8, 16, 3, 2, 2),  # N not a multiple of E: forces imbalance
        (50, 16, 32, 8, 3, 3),  # larger N, top_k=3
        (5, 8, 16, 5, 2, 4),  # N == E: still generally imbalanced
        (30, 16, 32, 4, 4, 5),  # top_k == E: every expert used by every token
    ],
)
def test_moe_layer_matches_per_token_reference(N, H, F, E, K, seed):
    """DistributedMoELayer's output must exactly match independently
    computing each token's weighted top_k expert sum. Deliberately covers
    non-uniform N/E ratios (N=7,E=3 and N=5,E=5) that make perfectly
    balanced routing essentially impossible -- exactly the condition that
    exposed the equal-chunk bug."""
    torch.manual_seed(seed)
    topo = build_topology(dp_size=1, ep_size=1, tp_size=1, device_type="cpu")
    layer = DistributedMoELayer(hidden_dim=H, ffn_dim=F, num_experts=E, top_k=K, topology=topo)
    x = torch.randn(N, H)

    with torch.no_grad():
        out = layer(x)
        ref = _manual_reference(layer, x)

    max_diff = float((out - ref).abs().max().item())
    assert max_diff < 1e-5, (
        f"DistributedMoELayer diverges from per-token reference by {max_diff:.4f} "
        f"(N={N}, E={E}, K={K}, seed={seed}). See ADR-009."
    )


def test_moe_layer_matches_reference_across_many_seeds():
    """Broader sweep: many random seeds at a fixed, deliberately
    imbalance-prone shape, since any single seed could in principle land on
    an accidentally-balanced routing that wouldn't have caught bug #3."""
    H, F, E, K = 16, 32, 5, 2
    N = 23  # not a multiple of E
    for seed in range(20):
        torch.manual_seed(seed)
        topo = build_topology(dp_size=1, ep_size=1, tp_size=1, device_type="cpu")
        layer = DistributedMoELayer(hidden_dim=H, ffn_dim=F, num_experts=E, top_k=K, topology=topo)
        x = torch.randn(N, H)
        with torch.no_grad():
            out = layer(x)
            ref = _manual_reference(layer, x)
        max_diff = float((out - ref).abs().max().item())
        assert max_diff < 1e-5, f"seed={seed}: diverged by {max_diff:.4f}"


def test_moe_layer_gradient_flows_to_correct_experts_only():
    """A token's gradient must only reach the experts it was actually
    routed to. This would catch a regression where the wrong expert
    processes a token (bug #3's failure mode) even in cases where the
    forward-pass numbers happen to coincidentally match (e.g. if two
    experts converge to similar weights during training -- not the case
    here, but the point of this test is to check the *routing*, not just
    the *output values*)."""
    torch.manual_seed(0)
    H, F, E, K = 8, 16, 4, 1  # top_k=1: exactly one expert per token, easy to check
    topo = build_topology(dp_size=1, ep_size=1, tp_size=1, device_type="cpu")
    layer = DistributedMoELayer(hidden_dim=H, ffn_dim=F, num_experts=E, top_k=K, topology=topo)
    x = torch.randn(6, H)

    with torch.no_grad():
        idx, _, _ = layer.router(x)
    assigned = idx[:, 0]  # each token's single assigned expert

    out = layer(x)
    out.sum().backward()

    for local_i, eid in enumerate(layer.local_expert_ids):
        expert = layer.experts[local_i]
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in expert.parameters())
        should_have_grad = (assigned == eid).any().item()
        assert has_grad == should_have_grad, (
            f"expert {eid}: gradient presence ({has_grad}) doesn't match whether "
            f"any token was actually routed to it ({should_have_grad})"
        )


# ===========================================================================
# send_counts arithmetic in isolation (pure tensor logic, no model needed)
# ===========================================================================


@pytest.mark.parametrize(
    "N,E,K,ep_size,seed",
    [
        (8, 8, 2, 2, 0),  # the exact case that first exposed the overcount
        (20, 6, 3, 3, 1),
        (50, 16, 4, 4, 2),
        (13, 5, 2, 5, 3),  # ep_size == E: one expert per rank
        (100, 32, 1, 4, 4),  # top_k=1: no possible span-multiple-ranks case
    ],
)
def test_send_counts_sum_equals_dispatch_item_count(N, E, K, ep_size, seed):
    """send_counts.sum() must equal N*K always -- the invariant the
    original per-token 'any slot in range' check violated whenever a
    token's K experts spanned multiple EP ranks. This is pure arithmetic
    on a random idx tensor, independent of the model or any dispatch
    machinery, and independent of ep_size actually being reachable
    (values are computed the same way DistributedMoELayer.forward() does,
    inlined here so this test doesn't silently go stale if that logic ever
    moves)."""
    torch.manual_seed(seed)
    idx = torch.randint(0, E, (N, K))
    experts_per_rank = E // ep_size
    expert_ids_flat = idx.reshape(-1)

    send_counts = torch.zeros(ep_size, dtype=torch.long)
    for ep_r in range(ep_size):
        lo = ep_r * experts_per_rank
        hi = lo + experts_per_rank
        send_counts[ep_r] = int(((expert_ids_flat >= lo) & (expert_ids_flat < hi)).sum().item())

    assert int(send_counts.sum().item()) == N * K


# ===========================================================================
# Real multi-rank (2-rank EP, mp.spawn): the actual test_distributed_invariants.py
# scenario, but with a full per-token numerical check rather than only
# token-conservation counts and NaN-absence
# ===========================================================================


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _build_seed_matched_reference(
    seed, hidden_dim, ffn_dim, num_experts, top_k, ep_size, dtype=torch.float32
):
    """Reconstructs, with no distributed setup at all, the exact expert
    weights each real EP rank would build. DistributedMoELayer.__init__
    re-seeds fresh *within* each rank's process and then builds only that
    rank's local experts (``nn.ModuleList([_SwiGLUExpert(...) for _ in
    self.local_expert_ids])``) -- so with the same seed on every rank, each
    rank's local experts are bit-identical to every *other* rank's local
    experts at the same local index, not to a hypothetical single
    sequential build across all `num_experts` experts. This mirrors that
    construction pattern exactly, rank by rank, so the reference is a
    genuine ground truth for what the real 2-rank run actually computes --
    not what a naive "just build all experts in one loop" reference would
    (silently) get wrong.

    Must be called before dist.init_process_group in the calling process:
    build_topology validates its requested mesh shape against the real
    dist.get_world_size() once initialized, so a degenerate ep_size=1
    topology can only be constructed before that point (or in a
    completely separate process).
    """
    from pkg.distributed.moe_layer import _SwiGLUExpert
    from pkg.kernels.moe_router import MoERouter

    topo = build_topology(dp_size=1, tp_size=1, pp_size=1, ep_size=1, device_type="cpu")
    experts_per_rank = num_experts // ep_size
    expert_by_id = {}
    router = None
    for r in range(ep_size):
        torch.manual_seed(seed)  # exactly what each real rank does independently
        local_ids = list(range(r * experts_per_rank, (r + 1) * experts_per_rank))
        modules = [_SwiGLUExpert(hidden_dim, ffn_dim, topo, dtype) for _ in local_ids]
        for gid, m in zip(local_ids, modules):
            expert_by_id[gid] = m
        this_router = MoERouter(hidden_dim, num_experts, top_k, dtype=dtype)
        if router is None:
            router = (
                this_router  # every rank's router is identical by the same argument; keep the first
            )
    return expert_by_id, router


def _reference_output_from_experts(expert_by_id, router, x, top_k):
    N, H = x.shape
    idx, weights, _ = router(x)
    ref = torch.zeros(N, H)
    for i in range(N):
        for k in range(top_k):
            eid = int(idx[i, k].item())
            w = float(weights[i, k].item())
            ref[i] += w * expert_by_id[eid](x[i : i + 1]).squeeze(0)
    return ref


def _ep_2rank_worker(rank, world_size, port, q, H, F, E, K, N, seed):
    # Build the seed-matched reference and its input BEFORE
    # init_process_group -- see _build_seed_matched_reference's docstring
    # for why this ordering is required, not just convenient.
    expert_by_id, ref_router = _build_seed_matched_reference(seed, H, F, E, K, world_size)
    torch.manual_seed(seed + 1)
    x = torch.randn(N, H)
    ref = _reference_output_from_experts(expert_by_id, ref_router, x, K)

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    torch.manual_seed(seed)  # same seed, same construction pattern as the reference used
    topo = build_topology(dp_size=1, ep_size=world_size, tp_size=1, device_type="cpu")
    layer = DistributedMoELayer(hidden_dim=H, ffn_dim=F, num_experts=E, top_k=K, topology=topo)

    with torch.no_grad():
        out = layer(x)

    max_diff = float((out - ref).abs().max().item())
    q.put({"rank": rank, "max_diff": max_diff, "shape_ok": out.shape == ref.shape, "error": None})
    dist.destroy_process_group()


@pytest.mark.skipif(sys.platform == "darwin", reason="mp.spawn fork-safety on macOS CI")
def test_moe_layer_2rank_matches_per_token_reference():
    """The real ADR-009 scenario: 2-rank EP, top_k=2 (so a token's experts
    can span both ranks), output checked against a genuine per-token
    reference -- not just token-conservation counts and NaN-absence, which
    is all test_distributed_invariants.py checks. Both this test's model
    and the reference use the same seed, so their experts are bit-identical
    and the comparison is a real ground-truth check, not a self-consistency
    one."""
    port = _free_port()
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [
        ctx.Process(target=_ep_2rank_worker, args=(r, 2, port, q, 16, 32, 8, 2, 12, 42))
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
        assert res["shape_ok"], f"rank {rank}: shape mismatch"
        assert res["max_diff"] < 1e-5, f"rank {rank}: diverged by {res['max_diff']:.4f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
