"""
tests/test_comm_ops.py
======================

Unit tests for the autograd-aware collectives in ``pkg/distributed/comm_ops.py``,
each checked on its own at 2 Gloo ranks in float64, before (and independently
of) being wired into the EP/TP/SP layers.

Every case checks the forward against values assembled by hand from the
ranks' inputs, and the backward with ``distributed_fd_check``: a
finite-difference check of the vector-Jacobian product over one global loss,
with every rank perturbing the same global input element in lockstep. (A
per-rank ``torch.autograd.gradcheck`` cannot see that a rank's output depends
on its peers' inputs.)

Multi-process; runs in the CI test-differential job, not Tier-0.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.distributed as dist

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pkg.distributed.comm_ops import all_to_all  # noqa: E402
from tests._differential import (  # noqa: E402
    compare,
    distributed_fd_check,
    failures_for,
    run_multiprocess,
)

pytestmark = pytest.mark.cpu

WS = 2


def _all_gather_obj(obj):
    out = [None] * dist.get_world_size()
    dist.all_gather_object(out, obj)
    return out


def _assert_ok(results):
    fails = failures_for(results, "fwd") + failures_for(results, "grad")
    if fails:
        pytest.fail("\n".join(fails), pytrace=False)


# ===========================================================================
# all_to_all (EP dispatch / combine)
# ===========================================================================


def _all_to_all_worker(rank: int, ws: int, cfg: dict) -> list:
    S = cfg["splits"]  # S[src][dst] = rows rank src sends to rank dst
    in_splits = list(S[rank])
    out_splits = [S[src][rank] for src in range(ws)]
    g = torch.Generator().manual_seed(100 + rank)
    x = torch.randn(sum(in_splits), 3, generator=g, dtype=torch.float64)
    cot = torch.randn(sum(out_splits), 3, generator=g, dtype=torch.float64)

    xs = _all_gather_obj(x)
    expected = torch.cat(
        [xs[src][sum(S[src][:rank]) : sum(S[src][: rank + 1])] for src in range(ws)]
    )

    def fn(t):
        return all_to_all(t, out_splits, in_splits)

    return [
        compare("fwd", "all_to_all output", fn(x), expected),
        distributed_fd_check(
            "all_to_all", fn, x, cot, input_replicated=False, output_replicated=False
        ),
    ]


@pytest.mark.parametrize(
    "splits",
    [
        [[1, 3], [2, 2]],  # uneven, different totals per rank
        [[0, 4], [3, 0]],  # zero-size splits in both directions
        [[2, 2], [2, 2]],  # even
    ],
    ids=["uneven", "zero-splits", "even"],
)
def test_all_to_all_forward_and_vjp(splits):
    _assert_ok(run_multiprocess(_all_to_all_worker, WS, {"splits": splits}))
