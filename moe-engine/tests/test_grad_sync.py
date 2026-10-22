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
# Global-mean objective (F22): DP SUM via FSDP2, global token count
# ===========================================================================

_F22 = "F22: the training objective is not a mean over the global batch"
_F24 = (
    "F24: apply_fsdp2 passes mp_policy=None to fully_shard when mixed precision "
    "is off, so FSDP2 raises at the first forward"
)


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


@pytest.mark.xfail(strict=True, reason=_F24)
def test_apply_fsdp2_fp32_trains_at_dp2():
    """apply_fsdp2 without mixed precision must give a working FSDP2 model."""
    _assert_ok(run_multiprocess(_fsdp_fp32_worker, 2, {}), "fsdp")


def test_fsdp2_force_sum_api_present():
    """The DP SUM that the global-mean convention needs relies on two FSDP2
    methods whose availability depends on the torch version (ADR-011).
    Pin the requirement: fail here, not in the middle of a training run."""
    from torch.distributed.fsdp import FSDPModule

    for name in ("set_force_sum_reduction_for_comms", "set_gradient_divide_factor"):
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


@pytest.mark.xfail(strict=True, reason=f"{_F24}; {_F22}")
def test_fsdp2_reduces_dp_grads_as_sum():
    """apply_fsdp2 must SUM gradients over DP (CPU/Gloo, dp=2), not average."""
    _assert_ok(run_multiprocess(_fsdp_sum_worker, 2, {}), "fsdp")


def _count_worker(rank: int, ws: int, cfg: dict) -> list:
    topo = build_topology(
        dp_size=cfg["dp"], ep_size=cfg["ep"], tp_size=cfg["tp"], device_type="cpu"
    )
    try:
        from pkg.distributed.grad_sync import global_token_count
    except ImportError as exc:
        return [check("count", "global_token_count", False, repr(exc))]
    # Same count on the TP ranks of a (dp, ep) position: they hold the same tokens.
    data_rank = topo.dp_rank * topo.ep_size + topo.ep_rank
    got = global_token_count(10 * data_rank + 1, topo)
    expected = sum(10 * d + 1 for d in range(topo.dp_size * topo.ep_size))
    return [
        check("count", "global_token_count", got == expected, f"got {got}, expected {expected}")
    ]


@pytest.mark.xfail(strict=True, reason=_F22)
@pytest.mark.parametrize(
    "dp,ep,tp,ws",
    [(2, 2, 1, 4), (1, 2, 2, 4), (2, 1, 2, 4)],
    ids=["dp2-ep2", "ep2-tp2", "dp2-tp2"],
)
def test_global_token_count_sums_over_data_ranks_only(dp, ep, tp, ws):
    """Summed over DP and EP (different tokens), never over TP (same tokens)."""
    _assert_ok(run_multiprocess(_count_worker, ws, dict(dp=dp, ep=ep, tp=tp)), "count")
