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

from pkg.distributed.comm_ops import (  # noqa: E402
    all_to_all,
    copy_to_group,
    gather_along_dim,
    reduce_from_group,
    split_along_dim,
)
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


# ===========================================================================
# Tensor / sequence parallel conjugate pairs
# ===========================================================================

# name -> (input_replicated, output_replicated). A replicated tensor is the
# same on every rank; a sharded/partitioned one differs per rank.
_PAIR_KINDS = {
    "copy_to_group": (True, False),
    "reduce_from_group": (False, True),
    "gather_along_dim": (False, True),
    "split_along_dim": (True, False),
}


def _pair_worker(rank: int, ws: int, cfg: dict) -> list:
    op, dim = cfg["op"], cfg["dim"]
    in_rep, out_rep = _PAIR_KINDS[op]
    shape = (2, 4, 6)
    g_in = torch.Generator().manual_seed(200 if in_rep else 300 + rank)
    x = torch.randn(*shape, generator=g_in, dtype=torch.float64)
    xs = _all_gather_obj(x)

    if op == "copy_to_group":
        fn, expected = (lambda t: copy_to_group(t)), x
    elif op == "reduce_from_group":
        fn, expected = (lambda t: reduce_from_group(t)), sum(xs)
    elif op == "gather_along_dim":
        fn, expected = (lambda t: gather_along_dim(t, dim)), torch.cat(xs, dim=dim)
    else:
        fn, expected = (lambda t: split_along_dim(t, dim)), x.chunk(ws, dim=dim)[rank]

    g_cot = torch.Generator().manual_seed(400 if out_rep else 500 + rank)
    cot = torch.randn(*expected.shape, generator=g_cot, dtype=torch.float64)
    return [
        compare("fwd", f"{op} output", fn(x), expected),
        distributed_fd_check(op, fn, x, cot, input_replicated=in_rep, output_replicated=out_rep),
    ]


@pytest.mark.parametrize(
    "op,dim",
    [
        ("copy_to_group", None),
        ("reduce_from_group", None),
        ("gather_along_dim", -1),
        ("gather_along_dim", 1),
        ("split_along_dim", -1),
        ("split_along_dim", 1),
    ],
    ids=["copy", "reduce", "gather-last", "gather-dim1", "split-last", "split-dim1"],
)
def test_conjugate_pair_forward_and_vjp(op, dim):
    _assert_ok(run_multiprocess(_pair_worker, WS, {"op": op, "dim": dim}))
