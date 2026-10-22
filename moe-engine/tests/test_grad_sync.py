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
from tests._differential import check, failures_for, run_multiprocess  # noqa: E402

pytestmark = pytest.mark.cpu

F14_INIT = "F14-init: replicated params are not broadcast at model construction"


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


@pytest.mark.xfail(strict=True, reason=F14_INIT)
@pytest.mark.parametrize(
    "dp,ep,tp,ws",
    [(2, 2, 1, 4), (1, 1, 2, 2)],
    ids=["dp2-ep2", "tp2"],
)
def test_broadcast_makes_replicas_identical_and_keeps_shards(dp, ep, tp, ws):
    _assert_ok(run_multiprocess(_init_worker, ws, dict(dp=dp, ep=ep, tp=tp)), "init")
