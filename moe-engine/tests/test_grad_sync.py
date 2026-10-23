"""
tests/test_grad_sync.py
=======================

Group-membership tests for ``pkg/distributed/grad_sync.py``:

* ``sync_replicated_grads`` must sum the gradients of replicated (non-expert)
  parameters over exactly the EP group -- the ranks that hold different
  tokens -- and over the TP group only for parameters marked
  sequence-parallel. It must never reduce over the DP group (FSDP2 owns
  that), so at ep = tp = 1 it is a no-op whatever dp is (F21).
* ``broadcast_replicated_params`` must make every replicated parameter
  identical across the ranks that replicate it, without overwriting EP
  expert shards or TP weight shards (F14-init).
* The global-mean objective (F22, ADR-011) needs ``apply_fsdp2`` to SUM over
  DP and ``global_token_count`` to count each data rank's tokens once.

Gradients are set by hand to ``rank + 1`` so each expected sum names exactly
which ranks contributed. Multi-process Gloo; runs in the CI test-differential
job, not Tier-0.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pkg.distributed.mesh import build_topology  # noqa: E402
from pkg.distributed.moe_layer import DistributedMoELayer  # noqa: E402
from pkg.distributed.tensor_parallel import ColumnParallelLinear  # noqa: E402
from tests._differential import check, compare, failures_for, run_multiprocess  # noqa: E402

pytestmark = pytest.mark.cpu


class _Model(nn.Module):
    def __init__(self, topo):
        super().__init__()
        self.inp = nn.Linear(4, 4)
        self.moe = DistributedMoELayer(4, 8, 4, 1, topo)
        self.col = ColumnParallelLinear(4, 4, bias=True, topology=topo, gather_output=True)


def _is_expert(name: str) -> bool:
    return name.startswith("moe.experts.")


def _assert_ok(results, kind):
    fails = failures_for(results, kind)
    if fails:
        pytest.fail("\n".join(fails), pytrace=False)


# ===========================================================================
# sync_replicated_grads (F21)
# ===========================================================================


def _sync_worker(rank: int, ws: int, cfg: dict) -> list:
    from pkg.distributed.grad_sync import mark_sequence_parallel, sync_replicated_grads

    topo = build_topology(
        dp_size=cfg["dp"], ep_size=cfg["ep"], tp_size=cfg["tp"], device_type="cpu"
    )
    torch.manual_seed(rank)
    model = _Model(topo)
    sp_param = nn.Parameter(torch.zeros(3))
    model.register_parameter("sp_weight", sp_param)
    mark_sequence_parallel(sp_param)
    for p in model.parameters():
        p.grad = torch.full_like(p, float(rank + 1))

    sync_replicated_grads(model, topo)

    # Ranks sharing this rank's dp and tp coordinates form its EP group;
    # those sharing dp and ep coordinates form its TP group.
    coords = [None] * ws
    dist.all_gather_object(coords, (topo.dp_rank, topo.ep_rank, topo.tp_rank))
    me = coords[rank]
    ep_group = [r for r, c in enumerate(coords) if (c[0], c[2]) == (me[0], me[2])]
    same_dp = [r for r, c in enumerate(coords) if c[0] == me[0]]

    checks = []
    for name, p in model.named_parameters():
        if _is_expert(name):
            members = [rank]  # EP-sharded: never reduced here
        elif name == "sp_weight":
            # Non-expert (EP sum) and sequence-parallel (TP sum): together,
            # every rank with this rank's DP coordinate.
            members = same_dp
        else:
            members = ep_group  # TP ranks see identical data: no TP sum
        expected = float(sum(r + 1 for r in members))
        got = p.grad.unique()
        ok = got.numel() == 1 and float(got) == expected
        checks.append(
            check("sync", f"{name}.grad", ok, f"got {got.tolist()}, expected {expected} {members}")
        )
    return checks


@pytest.mark.parametrize(
    "dp,ep,tp,ws",
    [(2, 2, 1, 4), (1, 2, 1, 2), (1, 1, 2, 2)],
    ids=["dp2-ep2", "ep2", "tp2-sp"],
)
def test_sync_reduces_over_exactly_the_right_group(dp, ep, tp, ws):
    _assert_ok(run_multiprocess(_sync_worker, ws, dict(dp=dp, ep=ep, tp=tp)), "sync")


def test_sync_is_noop_at_dp_only():
    """dp=2, ep=tp=1: FSDP2 owns the DP reduction, so sync must not touch
    any gradient -- it can never double-reduce."""
    _assert_ok(run_multiprocess(_sync_worker, 2, dict(dp=2, ep=1, tp=1)), "sync")


# ===========================================================================
# broadcast_replicated_params (F14-init)
# ===========================================================================


def _init_worker(rank: int, ws: int, cfg: dict) -> list:
    from pkg.distributed.grad_sync import broadcast_replicated_params

    topo = build_topology(
        dp_size=cfg["dp"], ep_size=cfg["ep"], tp_size=cfg["tp"], device_type="cpu"
    )
    torch.manual_seed(1000 + rank)  # every parameter starts different on every rank
    model = _Model(topo)
    broadcast_replicated_params(model, topo)

    coords = [None] * ws
    dist.all_gather_object(coords, (topo.dp_rank, topo.ep_rank, topo.tp_rank))
    values = [None] * ws
    dist.all_gather_object(values, {n: p.detach().clone() for n, p in model.named_parameters()})
    me = coords[rank]

    checks = []
    for name, p in model.named_parameters():
        if _is_expert(name):
            # Replicated across DP only: same (ep, tp) position must match,
            # different ep positions must stay distinct.
            same = [r for r, c in enumerate(coords) if (c[1], c[2]) == (me[1], me[2])]
            diff = [r for r, c in enumerate(coords) if c[1] != me[1]]
        elif name == "col.weight":
            # TP shard: replicated across DP and EP, distinct across TP.
            same = [r for r, c in enumerate(coords) if c[2] == me[2]]
            diff = [r for r, c in enumerate(coords) if c[2] != me[2]]
        else:
            same, diff = list(range(ws)), []
        eq = all(torch.equal(values[r][name], p) for r in same)
        ne = all(not torch.equal(values[r][name], p) for r in diff)
        checks.append(check("init", name, eq and ne, f"same={same} distinct={diff}"))
    return checks


@pytest.mark.parametrize(
    "dp,ep,tp,ws",
    [(2, 2, 1, 4), (1, 1, 2, 2)],
    ids=["dp2-ep2", "tp2"],
)
def test_broadcast_makes_replicas_identical_and_keeps_shards(dp, ep, tp, ws):
    _assert_ok(run_multiprocess(_init_worker, ws, dict(dp=dp, ep=ep, tp=tp)), "init")


# ===========================================================================
# FSDP2 at dp > 1 (F24, F25) and the global-mean objective (F22): DP SUM via
# FSDP2, global token count
# ===========================================================================


def _fsdp_fp32_worker(rank: int, ws: int, cfg: dict) -> list:
    from pkg.distributed.data_parallel import apply_fsdp2

    topo = build_topology(dp_size=ws, ep_size=1, tp_size=1, device_type="cpu")
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 2))
    x = torch.randn(3, 4)
    ref = model(x).detach()
    model = apply_fsdp2(model, topo)  # fp32: no mixed-precision dtype requested
    out = model(x)
    out.sum().backward()
    has_grads = all(p.grad is not None for p in model.parameters())
    return [
        compare("fsdp", "output", out, ref),
        check("fsdp", "every parameter has a gradient", has_grads, ""),
    ]


def _fsdp_moe_worker(rank: int, ws: int, cfg: dict) -> list:
    """dp=2, E=2, top-1: rank 0's tokens all pick expert 0 and rank 1's all
    pick expert 1, so each rank leaves a different expert unused."""
    from pkg.distributed.data_parallel import apply_fsdp2
    from tests._differential import dense_moe

    H, Fd, E, N = 4, 8, 2, 6
    topo = build_topology(dp_size=ws, ep_size=1, tp_size=1, device_type="cpu")
    torch.manual_seed(0)  # identical replicas on both DP ranks
    layer = DistributedMoELayer(H, Fd, E, 1, topo)
    with torch.no_grad():
        layer.router.gate_w.zero_()
        layer.router.gate_w[0] = torch.tensor([3.0, -3.0])
    ref_experts = [
        [
            w.detach().clone().requires_grad_(True)
            for w in (m.w_gate.weight, m.w_up.weight, m.w_down.weight)
        ]
        for m in layer.experts
    ]
    gate_w = layer.router.gate_w.detach().clone()

    def batch(r):
        g = torch.Generator().manual_seed(100 + r)
        x = torch.randn(N, H, generator=g)
        x[:, 0] = 5.0 if r == 0 else -5.0
        return x, torch.randn(N, H, generator=g)

    # Dense reference: sum over DP ranks of each rank's loss. Only directions
    # are compared below, so this test does not depend on the DP mean/sum.
    sum(
        (dense_moe(x, gate_w, ref_experts, 1) * c).sum() for x, c in map(batch, range(ws))
    ).backward()

    layer = apply_fsdp2(layer, topo)
    x, cot = batch(rank)
    (layer(x) * cot).sum().backward()
    checks = []
    for i, m in enumerate(layer.experts):
        for j, pname in enumerate(("w_gate", "w_up", "w_down")):
            exp = ref_experts[i][j].grad
            got = getattr(m, pname).weight.grad
            got = got.full_tensor() if got is not None else torch.zeros_like(exp)
            cos = float(torch.nn.functional.cosine_similarity(got.flatten(), exp.flatten(), dim=0))
            checks.append(
                check(
                    "fsdp", f"experts.{i}.{pname}.grad direction", cos > 1 - 1e-5, f"cos={cos:.6f}"
                )
            )
    return checks


def test_fsdp2_moe_grads_correct_when_ranks_use_different_experts():
    """Each expert's DP-reduced gradient must come from that expert alone
    (F25: FSDP2 skipped unused parameters, so DP ranks that left different
    experts unused reduce-scattered different parameter sets -- a hang, or,
    with equal-sized experts as here, one expert's gradient mixed into
    another's)."""
    _assert_ok(run_multiprocess(_fsdp_moe_worker, 2, {}), "fsdp")


def test_apply_fsdp2_fp32_trains_at_dp2():
    """apply_fsdp2 without mixed precision must give a working FSDP2 model
    (F24: it passed mp_policy=None, and FSDP2 raised at the first forward)."""
    _assert_ok(run_multiprocess(_fsdp_fp32_worker, 2, {}), "fsdp")


def test_fsdp2_force_sum_api_present():
    """apply_fsdp2 relies on FSDP2 methods whose availability depends on the
    torch version (ADR-011): the DP SUM of the global-mean convention (F22)
    and the reduction of unused expert parameters (F25). Pin the
    requirement: fail here, not in the middle of a training run."""
    from torch.distributed.fsdp import FSDPModule

    for name in (
        "set_force_sum_reduction_for_comms",
        "set_gradient_divide_factor",
        "set_reduce_scatter_unused_params",
    ):
        assert hasattr(FSDPModule, name), f"torch {torch.__version__}: FSDPModule.{name} missing"


def _fsdp_sum_worker(rank: int, ws: int, cfg: dict) -> list:
    from pkg.distributed.data_parallel import apply_fsdp2

    topo = build_topology(dp_size=ws, ep_size=1, tp_size=1, device_type="cpu")
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 2))
    ref = [p.detach().clone().requires_grad_(True) for p in model.parameters()]
    model = apply_fsdp2(model, topo)
    model(torch.full((1, 4), float(rank + 1))).sum().backward()

    def dense(x):
        return nn.functional.linear(nn.functional.linear(x, ref[0], ref[1]), ref[2], ref[3])

    # DP ranks hold different data; the global objective sums over them.
    sum(dense(torch.full((1, 4), float(r + 1))).sum() for r in range(ws)).backward()
    checks = []
    for (name, p), r in zip(model.named_parameters(), ref):
        got = p.grad.full_tensor()
        ok = torch.allclose(got, r.grad, atol=1e-6, rtol=1e-5)
        ratio = float(got.norm() / r.grad.norm())
        checks.append(check("fsdp", f"{name}.grad", ok, f"|got| / |DP sum| = {ratio:.4f}"))
    return checks


def test_fsdp2_reduces_dp_grads_as_sum():
    """apply_fsdp2 must SUM gradients over DP (CPU/Gloo, dp=2), not average
    (F22: FSDP2's default DP mean)."""
    _assert_ok(run_multiprocess(_fsdp_sum_worker, 2, {}), "fsdp")


def _marks_worker(rank: int, ws: int, cfg: dict) -> list:
    from pkg.distributed.data_parallel import apply_fsdp2
    from pkg.distributed.grad_sync import (
        SEQUENCE_PARALLEL_ATTR,
        TENSOR_PARALLEL_SHARD_ATTR,
        mark_sequence_parallel,
        mark_tensor_parallel_shard,
    )

    topo = build_topology(dp_size=ws, ep_size=1, tp_size=1, device_type="cpu")
    model = nn.Module()
    model.lin = nn.Linear(4, 4)
    model.sp_w = nn.Parameter(torch.zeros(4, 4))
    mark_sequence_parallel(model.sp_w)
    mark_tensor_parallel_shard(model.lin.weight)
    model = apply_fsdp2(model, topo)
    return [
        check(
            "marks",
            "sp_w sequence-parallel",
            getattr(model.sp_w, SEQUENCE_PARALLEL_ATTR, False),
            "",
        ),
        check(
            "marks",
            "lin.weight TP shard",
            getattr(model.lin.weight, TENSOR_PARALLEL_SHARD_ATTR, False),
            "",
        ),
    ]


def test_apply_fsdp2_keeps_grad_sync_marks():
    """sync_replicated_grads finds sequence-parallel params by their mark,
    and runs after FSDP2 has wrapped the model: the marks must survive
    (F26: FSDP2's new DTensor parameters dropped them)."""
    _assert_ok(run_multiprocess(_marks_worker, 2, {}), "marks")


def _count_worker(rank: int, ws: int, cfg: dict) -> list:
    from pkg.distributed.grad_sync import global_token_count

    topo = build_topology(
        dp_size=cfg["dp"], ep_size=cfg["ep"], tp_size=cfg["tp"], device_type="cpu"
    )
    # Same count on the TP ranks of a (dp, ep) position: they hold the same tokens.
    data_rank = topo.dp_rank * topo.ep_size + topo.ep_rank
    got = global_token_count(10 * data_rank + 1, topo)
    expected = sum(10 * d + 1 for d in range(topo.dp_size * topo.ep_size))
    return [
        check("count", "global_token_count", got == expected, f"got {got}, expected {expected}")
    ]


@pytest.mark.parametrize(
    "dp,ep,tp,ws",
    [(2, 2, 1, 4), (1, 2, 2, 4), (2, 1, 2, 4)],
    ids=["dp2-ep2", "ep2-tp2", "dp2-tp2"],
)
def test_global_token_count_sums_over_data_ranks_only(dp, ep, tp, ws):
    """Summed over DP and EP (different tokens), never over TP (same tokens)."""
    _assert_ok(run_multiprocess(_count_worker, ws, dict(dp=dp, ep=ep, tp=tp)), "count")


# ===========================================================================
# FSDP2 root wrap: experts are expert-data-parallel over DP (F14-fsdp)
# ===========================================================================


def _edp_worker(rank: int, ws: int, cfg: dict) -> list:
    """dp=2 x ep=2: three full engine steps (broadcast, apply_fsdp2,
    backward, finalize_grads, clip_grad_norm_, SGD) on distinct per-rank data."""
    from pkg.distributed.data_parallel import apply_fsdp2
    from pkg.distributed.grad_sync import (
        broadcast_replicated_params,
        clip_grad_norm_,
        finalize_grads,
        global_token_count,
    )

    try:
        from torch.distributed.tensor import DTensor
    except ImportError:  # pragma: no cover
        from torch.distributed._tensor import DTensor

    topo = build_topology(dp_size=2, ep_size=2, tp_size=1, device_type="cpu")
    torch.manual_seed(100 + rank)  # every parameter starts different on every rank
    model = nn.Sequential(nn.Linear(4, 4), DistributedMoELayer(4, 8, 4, 2, topo))
    broadcast_replicated_params(model, topo)
    model = apply_fsdp2(model, topo)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)

    checks = []
    for name, p in model.named_parameters():
        ok = isinstance(p, DTensor) and p.device_mesh.mesh_dim_names == ("dp",)
        mesh = getattr(p, "device_mesh", None)
        checks.append(check("edp", f"{name} sharded over DP", ok, f"mesh={mesh}"))

    g = torch.Generator().manual_seed(7 + rank)  # different tokens on every rank
    for _ in range(3):
        x = torch.randn(6, 4, generator=g)
        opt.zero_grad()
        model(x).pow(2).sum().backward()
        finalize_grads(model, topo, global_token_count(x.shape[0], topo))
        clip_grad_norm_(model, 1.0, topo)
        opt.step()

    full = {n: p.full_tensor().detach() for n, p in model.named_parameters()}
    everyone = [None] * ws
    dist.all_gather_object(everyone, (topo.dp_rank, topo.ep_rank, full))
    for name, t in full.items():
        if name.startswith("1.experts."):
            same = [f for dpr, epr, f in everyone if epr == topo.ep_rank]
            diff = [f for dpr, epr, f in everyone if epr != topo.ep_rank]
            ok = all(torch.equal(f[name], t) for f in same) and all(
                not torch.equal(f[name], t) for f in diff
            )
            note = "identical across DP, distinct across EP"
        else:
            ok = all(torch.equal(f[name], t) for _, _, f in everyone)
            note = "identical on every rank after 3 steps"
        checks.append(check("edp", f"{name}: {note}", ok, ""))
    return checks


def test_fsdp2_root_wrap_makes_experts_expert_data_parallel():
    """apply_fsdp2's root fully_shard manages the expert parameters too: they
    are sharded over DP (expert-data-parallel; each DP group holds one EP
    rank's experts), stay distinct across EP, and after real training steps
    every replicated parameter -- router included -- is identical on every
    rank (M2 done-when (c))."""
    _assert_ok(run_multiprocess(_edp_worker, 4, {}), "edp")
