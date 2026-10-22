"""
tests/test_differential.py
==========================

Differential tests: every distributed primitive (EP, TP, SP) is compared
against a dense single-device reference, for the forward output, for every
parameter gradient (mapped to the rank's shard), and for the input gradient.
Multi-process Gloo on CPU via ``torch.multiprocessing`` spawn.

Why this file exists
--------------------
Before it, the distributed tests checked forward only, often under
``torch.no_grad()``, with the same seed on every rank. With identical seeds,
expert ``j`` on rank 0 has the same weights as expert ``j`` on rank 1, so a
token sent to the right local index on the wrong rank still passes; and with
no backward check, collectives that autograd cannot see (raw
``torch.distributed`` calls into fresh ``torch.empty`` buffers) leave
experts frozen while the suite stays green. See the October 2026 audit
handoff, findings F1-F4 and F15.

Ground rules
------------
* Distinct per-rank seeds for every sharded parameter and every input.
  Replicated parameters are broadcast from rank 0 *by this harness*: the
  engine has no init-time broadcast of its own (F14).
* The reference is dense and single-device (``tests/_differential.py``,
  which imports nothing from ``pkg``). A distributed result is never compared
  against another distributed result.
* fp32, ``atol=1e-5, rtol=1e-4``. Failure messages report max abs and
  relative error per tensor.
* Gradient conventions (replicated vs partitioned losses, shard vs full
  gradients) are spelled out in ``tests/_differential.py``.

Cases that fail today are marked ``xfail(strict=True)`` with their finding
IDs. Strict means a fix that makes one pass turns it into an XPASS failure,
so the marker must be removed in the same commit as the fix.

Running
-------
    pytest tests/test_differential.py -v

Each case spawns its own 2 or 4 Gloo processes, so a crash or hang in one
case (F4 can raise on one rank and leave the other blocked in a collective)
cannot poison the next.
"""

from __future__ import annotations

import functools
import math
import sys
import time
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pkg.distributed.data_parallel import apply_fsdp2  # noqa: E402
from pkg.distributed.expert_parallel import all_to_all_dispatch  # noqa: E402
from pkg.distributed.grad_sync import (  # noqa: E402
    broadcast_replicated_params,
    finalize_grads,
    global_token_count,
    mark_sequence_parallel,
)
from pkg.distributed.mesh import build_topology  # noqa: E402
from pkg.distributed.moe_layer import DistributedMoELayer  # noqa: E402
from pkg.distributed.sequence_parallel import (  # noqa: E402
    gather_from_sequence_parallel,
    scatter_to_sequence_parallel,
)
from pkg.distributed.tensor_parallel import (  # noqa: E402
    ColumnParallelLinear,
    RowParallelLinear,
)
from tests._differential import (  # noqa: E402
    check,
    compare,
    dense_moe,
    failures_for,
    run_multiprocess,
)

pytestmark = pytest.mark.cpu

SEED = 1234

# Forced imbalance: expert 0 (always owned by EP rank 0) gets a gate column
# aligned with a direction the inputs are shifted along, so most tokens pick
# it first and rank 0 receives far more dispatch items than its peers.
HOT_EXPERT = 0
IMB_GATE = 4.0
IMB_INPUT = 1.5


def _hot_direction(h: int) -> torch.Tensor:
    return torch.ones(h) / math.sqrt(h)


def _all_gather_obj(obj):
    out = [None] * dist.get_world_size()
    dist.all_gather_object(out, obj)
    return out


def _detached(t: torch.Tensor) -> torch.Tensor:
    return t.detach().clone()


# ===========================================================================
# Expert parallelism
# ===========================================================================


def _ep_worker(rank: int, ws: int, cfg: dict) -> list:
    H, Fd, E, K, N = cfg["H"], cfg["F"], cfg["E"], cfg["K"], cfg["N"]
    capacity_factor = cfg["capacity_factor"]

    topo = build_topology(dp_size=1, ep_size=ws, tp_size=1, device_type="cpu")
    torch.manual_seed(SEED + rank)  # distinct expert weights per rank
    layer = DistributedMoELayer(
        H,
        Fd,
        E,
        K,
        topo,
        capacity_factor=capacity_factor,
        capacity_dropping=cfg["drop"],
    )
    with torch.no_grad():
        for p in layer.router.parameters():
            dist.broadcast(p, src=0)
        if cfg["imbalanced"]:
            layer.router.gate_w[:, HOT_EXPERT] += IMB_GATE * _hot_direction(H)

    g = torch.Generator().manual_seed(SEED + 100 + rank)
    x = torch.randn(N, H, generator=g)
    if cfg["imbalanced"]:
        x = x + IMB_INPUT * _hot_direction(H)
    cot = torch.randn(N, H, generator=g)

    # ---- Dense reference: all ranks' inputs and all experts on one device ----
    local = {
        "x": x,
        "cot": cot,
        "experts": {
            eid: (
                _detached(m.w_gate.weight),
                _detached(m.w_up.weight),
                _detached(m.w_down.weight),
            )
            for eid, m in zip(layer.local_expert_ids, layer.experts)
        },
    }
    shards = _all_gather_obj(local)
    expert_weights = {}
    for s in shards:
        expert_weights.update(s["experts"])
    assert sorted(expert_weights) == list(range(E)), "expert ownership does not cover 0..E-1"
    ref_experts = [[w.clone().requires_grad_(True) for w in expert_weights[e]] for e in range(E)]
    gate = _detached(layer.router.gate_w)
    capacity = math.ceil(capacity_factor * N * K / E) if cfg["drop"] else None

    # The global loss is the sum of every rank's local loss. Run it block by
    # block so this rank's router gradient (its local-token contribution) is
    # available separately; expert gradients accumulate across all blocks.
    for s, blk in enumerate(shards):
        xs = blk["x"].clone().requires_grad_(True)
        gate_s = gate.clone().requires_grad_(True)
        ys = dense_moe(xs, gate_s, ref_experts, K, capacity)
        (ys * blk["cot"]).sum().backward()
        if s == rank:
            y_ref, x_grad_ref, gate_grad_ref = ys.detach(), xs.grad, gate_s.grad

    ref_param_grads = {"router.gate_w": gate_grad_ref}
    for i, eid in enumerate(layer.local_expert_ids):
        for j, pname in enumerate(("w_gate", "w_up", "w_down")):
            rg = ref_experts[eid][j].grad
            ref_param_grads[f"experts.{i}.{pname}.weight"] = (
                rg if rg is not None else torch.zeros_like(ref_experts[eid][j])
            )

    # ---- Distributed run ----
    xd = x.clone().requires_grad_(True)
    y = layer(xd)
    checks = [compare("fwd", "output", y, y_ref)]
    if cfg["drop"]:
        frac = layer.last_dropped_token_fraction
        checks.append(
            check("fwd", "harness: capacity case drops something", frac > 0, f"dropped={frac}")
        )
    (y * cot).sum().backward()

    checks.append(compare("grad", "input x.grad", xd.grad, x_grad_ref))
    for name, p in layer.named_parameters():
        if name not in ref_param_grads:
            checks.append(check("grad", f"{name}.grad", False, "no reference for parameter"))
            continue
        label = name
        if name.startswith("experts."):
            label = f"{name} (expert {layer.local_expert_ids[int(name.split('.')[1])]})"
        checks.append(compare("grad", f"{label}.grad", p.grad, ref_param_grads[name]))
    return checks


def _ep_cases():
    """(id, world_size, cfg, findings-that-fail-fwd, findings-that-fail-grad)."""
    cases = []
    # E=5 (ep=2, ep=4) and E=10 (ep=4) are the non-divisible cases that
    # crashed or deadlocked before the F4 fix; they are now plain regressions.
    for ws, e_values in ((2, (8, 5, 10)), (4, (8, 5, 10))):
        for E in e_values:
            for K in (1, 2):
                for imbalanced in (False, True):
                    for drop in (False, True):
                        cfg = dict(
                            H=16,
                            F=32,
                            E=E,
                            K=K,
                            N=24,
                            imbalanced=imbalanced,
                            drop=drop,
                            capacity_factor=0.5 if drop else 1.25,
                        )
                        cid = (
                            f"ep{ws}-E{E}-k{K}-{'imbal' if imbalanced else 'bal'}"
                            f"-{'drop' if drop else 'nodrop'}"
                        )
                        cases.append((cid, ws, cfg, (), ()))
    return cases


_EP_CASES = _ep_cases()


@functools.lru_cache(maxsize=None)
def _ep_result(cid: str):
    _, ws, cfg, _, _ = next(c for c in _EP_CASES if c[0] == cid)
    return run_multiprocess(_ep_worker, ws, cfg)


# Finding ID -> xfail reason, for cases that currently fail. A new failing
# case must add its finding here first.
_FINDING_TEXT: dict = {
    "F26": (
        "F26: FSDP2 replaces parameters with new DTensor parameters, dropping the "
        "mark_sequence_parallel attribute, so at dp > 1 an SP-region replicated "
        "weight's gradient is never summed over TP"
    ),
}


def _params(cases, which):
    out = []
    for c in cases:
        findings = c[which]
        marks = ()
        if findings:
            reason = "; ".join(_FINDING_TEXT[f] for f in findings)
            marks = (pytest.mark.xfail(strict=True, reason=reason),)
        out.append(pytest.param(c[0], id=c[0], marks=marks))
    return out


def _assert_kind(results, kind):
    fails = failures_for(results, kind)
    if fails:
        pytest.fail("\n".join(fails), pytrace=False)


@pytest.mark.parametrize("cid", _params(_EP_CASES, 3))
def test_ep_forward_matches_dense(cid):
    _assert_kind(_ep_result(cid), "fwd")


@pytest.mark.parametrize("cid", _params(_EP_CASES, 4))
def test_ep_gradients_match_dense(cid):
    _assert_kind(_ep_result(cid), "grad")


# ---------------------------------------------------------------------------
# EP dispatch must fail on every rank together, never hang (F4)
# ---------------------------------------------------------------------------


def _dispatch_bad_counts_worker(rank: int, ws: int, cfg: dict) -> list:
    """Rank ``cfg["bad_rank"]`` passes send_counts that do not sum to its row
    count. Before F4 that rank raised Gloo's local split-size error while its
    peers entered the all-to-all and blocked until the collective timeout."""
    topo = build_topology(dp_size=1, ep_size=ws, tp_size=1, device_type="cpu")
    rows_per_peer = 2
    tokens = torch.randn(rows_per_peer * ws, 4)
    send_counts = torch.full((ws,), rows_per_peer, dtype=torch.long)
    if rank == cfg["bad_rank"]:
        send_counts[0] += 1
    t0 = time.monotonic()
    try:
        all_to_all_dispatch(tokens, send_counts, topo)
    except RuntimeError as exc:
        msg = str(exc)
        named = "invalid send_counts" in msg and f"[{cfg['bad_rank']}]" in msg
        return [
            check(
                "abort",
                "every rank aborts with the same error naming the bad rank",
                named,
                f"after {time.monotonic() - t0:.1f}s: {msg[:300]}",
            )
        ]
    return [check("abort", "dispatch must not succeed", False, "returned normally")]


@pytest.mark.parametrize("ws,bad_rank", [(2, 1), (4, 2)], ids=["ep2-bad1", "ep4-bad2"])
def test_ep_dispatch_bad_counts_abort_on_all_ranks(ws, bad_rank):
    _assert_kind(run_multiprocess(_dispatch_bad_counts_worker, ws, {"bad_rank": bad_rank}), "abort")


# ===========================================================================
# Tensor parallelism (tp=2)
# ===========================================================================


def _tp_worker(rank: int, ws: int, cfg: dict) -> list:
    H, Fd = 8, 16
    topo = build_topology(dp_size=1, ep_size=1, tp_size=ws, device_type="cpu")
    torch.manual_seed(SEED + rank)  # distinct weight shards per rank

    # Input and cotangent are replicated across the TP group (same seed);
    # leading dims [3, 5] exercise the ADR-008 movedim path.
    g = torch.Generator().manual_seed(SEED + 500)
    x = torch.randn(3, 5, H, generator=g)

    if cfg["case"] == "col_gather":
        col = ColumnParallelLinear(H, Fd, bias=cfg["bias"], topology=topo, gather_output=True)
        if col.bias is not None:
            with torch.no_grad():
                col.bias.normal_()
                dist.broadcast(col.bias, src=0)
        cot = torch.randn(3, 5, Fd, generator=g)

        w_full = torch.cat(_all_gather_obj(_detached(col.weight)), dim=0)  # [F, H]
        xr = x.clone().requires_grad_(True)
        wr = w_full.clone().requires_grad_(True)
        br = _detached(col.bias).requires_grad_(True) if col.bias is not None else None
        yr = F.linear(xr, wr, br)
        (yr * cot).sum().backward()
        ref = {"weight": wr.grad.chunk(ws, dim=0)[rank]}
        if br is not None:
            ref["bias"] = br.grad
        modules = {"": col}

        xd = x.clone().requires_grad_(True)
        y = col(xd)
    else:  # "mlp": Column(gather_output=False) -> GELU -> Row
        col = ColumnParallelLinear(H, Fd, bias=False, topology=topo, gather_output=False)
        row = RowParallelLinear(Fd, H, bias=cfg["bias"], topology=topo)
        if row.bias is not None:
            with torch.no_grad():
                row.bias.normal_()
                dist.broadcast(row.bias, src=0)
        cot = torch.randn(3, 5, H, generator=g)

        w1 = torch.cat(_all_gather_obj(_detached(col.weight)), dim=0)  # [F, H]
        w2 = torch.cat(_all_gather_obj(_detached(row.weight)), dim=1)  # [H, F]
        xr = x.clone().requires_grad_(True)
        w1r = w1.clone().requires_grad_(True)
        w2r = w2.clone().requires_grad_(True)
        br = _detached(row.bias).requires_grad_(True) if row.bias is not None else None
        yr = F.linear(F.gelu(F.linear(xr, w1r)), w2r, br)
        (yr * cot).sum().backward()
        ref = {
            "col.weight": w1r.grad.chunk(ws, dim=0)[rank],
            "row.weight": w2r.grad.chunk(ws, dim=1)[rank],
        }
        if br is not None:
            ref["row.bias"] = br.grad
        modules = {"col.": col, "row.": row}

        xd = x.clone().requires_grad_(True)
        y = row(F.gelu(col(xd)))

    checks = [compare("fwd", "output", y, yr)]
    y.backward(cot)
    checks.append(compare("grad", "input x.grad", xd.grad, xr.grad))
    for prefix, mod in modules.items():
        for name, p in mod.named_parameters():
            key = prefix + name
            if key not in ref:
                checks.append(check("grad", f"{key}.grad", False, "no reference for parameter"))
                continue
            checks.append(compare("grad", f"{key}.grad", p.grad, ref[key]))
    return checks


_TP_CASES = [
    ("tp2-col_gather-nobias", dict(case="col_gather", bias=False), (), ()),
    ("tp2-col_gather-bias", dict(case="col_gather", bias=True), (), ()),
    # ColumnParallelLinear(gather_output=False, bias=True) raises
    # NotImplementedError by design (ADR-008), so only the Row bias varies.
    ("tp2-mlp-nobias", dict(case="mlp", bias=False), (), ()),
    ("tp2-mlp-rowbias", dict(case="mlp", bias=True), (), ()),
]


@functools.lru_cache(maxsize=None)
def _tp_result(cid: str):
    cfg = next(c for c in _TP_CASES if c[0] == cid)[1]
    return run_multiprocess(_tp_worker, 2, cfg)


@pytest.mark.parametrize("cid", _params(_TP_CASES, 2))
def test_tp_forward_matches_dense(cid):
    _assert_kind(_tp_result(cid), "fwd")


@pytest.mark.parametrize("cid", _params(_TP_CASES, 3))
def test_tp_gradients_match_dense(cid):
    _assert_kind(_tp_result(cid), "grad")


# ===========================================================================
# Sequence parallelism (tp=2)
# ===========================================================================


def _sp_worker(rank: int, ws: int, cfg: dict) -> list:
    B, H, OUT = 2, 8, 6
    S = cfg["S"]
    topo = build_topology(dp_size=1, ep_size=1, tp_size=ws, device_type="cpu")
    chunk = -(-S // ws)
    g_rep = torch.Generator().manual_seed(SEED + 700)  # replicated tensors
    checks = []

    if cfg["case"] == "gather":
        # Distinct per-rank shards in; full sequence (replicated) out.
        g_loc = torch.Generator().manual_seed(SEED + 800 + rank)
        xs = torch.randn(B, chunk, H, generator=g_loc)
        if (rank + 1) * chunk > S:  # zero the padding rows scatter would have added
            xs[:, S - rank * chunk :, :] = 0.0
        cot = torch.randn(B, S, H, generator=g_rep)

        full = torch.cat(_all_gather_obj(xs), dim=1)
        fr = full.clone().requires_grad_(True)
        yr = fr[:, :S, :]
        (yr * cot).sum().backward()
        x_grad_ref = fr.grad[:, rank * chunk : (rank + 1) * chunk, :]

        xd = xs.clone().requires_grad_(True)
        y = gather_from_sequence_parallel(xd, topo, orig_seq_len=S)
        checks.append(compare("fwd", "output", y, yr))
        y.backward(cot)
        checks.append(compare("grad", "input shard x.grad", xd.grad, x_grad_ref))
        return checks

    # "roundtrip": replicated x -> scatter (optionally fused projection) ->
    # gather -> replicated output.
    x = torch.randn(B, S, H, generator=g_rep)
    w = None
    if cfg["fused"]:
        torch.manual_seed(SEED + rank)
        w = torch.randn(OUT, H) / math.sqrt(H)
        dist.broadcast(w, src=0)  # next_weight must be replicated
    cot = torch.randn(B, S, OUT if cfg["fused"] else H, generator=g_rep)

    xr = x.clone().requires_grad_(True)
    wr = w.clone().requires_grad_(True) if w is not None else None
    yr = F.linear(xr, wr) if wr is not None else xr * 1.0
    (yr * cot).sum().backward()

    xd = x.clone().requires_grad_(True)
    wd = w.clone().requires_grad_(True) if w is not None else None
    y = gather_from_sequence_parallel(
        scatter_to_sequence_parallel(xd, topo, next_weight=wd), topo, orig_seq_len=S
    )
    checks.append(compare("fwd", "output", y, yr))
    y.backward(cot)
    checks.append(compare("grad", "input x.grad (replicated input)", xd.grad, xr.grad))
    if wd is not None and cfg.get("engine_sync"):
        # next_weight is replicated but applied to a sequence shard, so each
        # rank holds only its tokens' partial gradient. The engine's
        # replicated-gradient sync must sum it over the TP group for a param
        # marked sequence-parallel (F21); compare each rank's grad directly.
        try:
            from pkg.distributed.grad_sync import mark_sequence_parallel, sync_replicated_grads
        except ImportError as exc:
            checks.append(check("grad", "engine sync_replicated_grads", False, repr(exc)))
        else:
            holder = nn.Module()
            holder.next_weight = nn.Parameter(wd.detach().clone())
            holder.next_weight.grad = wd.grad.detach().clone()
            mark_sequence_parallel(holder.next_weight)
            sync_replicated_grads(holder, topo)
            checks.append(
                compare(
                    "grad", "next_weight.grad (engine-synced)", holder.next_weight.grad, wr.grad
                )
            )
    elif wd is not None:
        # Harness stand-in for the engine's sync (F21): compare the TP-group sum.
        wg = wd.grad.clone() if wd.grad is not None else torch.zeros_like(wd)
        dist.all_reduce(wg)
        checks.append(compare("grad", "next_weight.grad (TP-group sum)", wg, wr.grad))
    return checks


# Before F3, every SP case raised in forward: gather_from_sequence_parallel
# wrote in place into a fresh buffer while its input required grad.
_SP_CASES = []
for _S, _tag in ((6, "div"), (7, "nondiv")):
    _SP_CASES.append((f"sp2-gather-S{_S}{_tag}", dict(case="gather", S=_S), (), ()))
    for _fused in (False, True):
        _SP_CASES.append(
            (
                f"sp2-roundtrip-{'fused' if _fused else 'unfused'}-S{_S}{_tag}",
                dict(case="roundtrip", S=_S, fused=_fused),
                (),
                (),
            )
        )
    # Same fused round trip with no harness-side TP-group sum (F21).
    _SP_CASES.append(
        (
            f"sp2-roundtrip-fused-engine-sync-S{_S}{_tag}",
            dict(case="roundtrip", S=_S, fused=True, engine_sync=True),
            (),
            (),
        )
    )


@functools.lru_cache(maxsize=None)
def _sp_result(cid: str):
    cfg = next(c for c in _SP_CASES if c[0] == cid)[1]
    return run_multiprocess(_sp_worker, 2, cfg)


@pytest.mark.parametrize("cid", _params(_SP_CASES, 2))
def test_sp_forward_matches_dense(cid):
    _assert_kind(_sp_result(cid), "fwd")


@pytest.mark.parametrize("cid", _params(_SP_CASES, 3))
def test_sp_gradients_match_dense(cid):
    _assert_kind(_sp_result(cid), "grad")


# ===========================================================================
# Composite: ep=2 MoE layer inside a tiny model, one optimizer step
# ===========================================================================


class _TinyMoEModel(nn.Module):
    def __init__(self, H, Fd, E, K, topo):
        super().__init__()
        self.inp = nn.Linear(H, H)
        self.moe = DistributedMoELayer(H, Fd, E, K, topo)
        self.out = nn.Linear(H, H)

    def forward(self, x):
        h = torch.tanh(self.inp(x))
        return self.out(h + self.moe(h))


def _composite_worker(rank: int, ws: int, cfg: dict) -> list:
    H, Fd, E, K, N, LR = 16, 32, cfg["E"], 2, 24, 0.1
    topo = build_topology(dp_size=1, ep_size=ws, tp_size=1, device_type="cpu")
    torch.manual_seed(SEED + rank)
    model = _TinyMoEModel(H, Fd, E, K, topo)
    replicated = {n for n, _ in model.named_parameters() if not n.startswith("moe.experts.")}
    engine_errors = []
    if cfg["engine_init"]:
        # The engine must make replicated params identical across ranks (F14-init).
        try:
            from pkg.distributed.grad_sync import broadcast_replicated_params
        except ImportError as exc:
            engine_errors.append(("fwd", "engine broadcast_replicated_params", repr(exc)))
        else:
            broadcast_replicated_params(model, topo)
    else:
        # Harness stand-in for the init broadcast the engine lacked (F14).
        with torch.no_grad():
            for n, p in model.named_parameters():
                if n in replicated:
                    dist.broadcast(p, src=0)

    g = torch.Generator().manual_seed(SEED + 900 + rank)
    x = torch.randn(N, H, generator=g)
    cot = torch.randn(N, H, generator=g)

    # ---- Dense reference ----
    experts = {
        eid: (_detached(m.w_gate.weight), _detached(m.w_up.weight), _detached(m.w_down.weight))
        for eid, m in zip(model.moe.local_expert_ids, model.moe.experts)
    }
    shards = _all_gather_obj({"x": x, "cot": cot, "experts": experts})
    all_experts = {}
    for s in shards:
        all_experts.update(s["experts"])
    ref_experts = [[w.clone().requires_grad_(True) for w in all_experts[e]] for e in range(E)]
    # Rank 0's copy of each replicated parameter is the reference value.
    rank0 = _all_gather_obj(
        {n: _detached(p) for n, p in model.named_parameters() if n in replicated}
    )[0]
    rp = {n: t.clone().requires_grad_(True) for n, t in rank0.items()}

    def dense_model(xx):
        h = torch.tanh(F.linear(xx, rp["inp.weight"], rp["inp.bias"]))
        moe = dense_moe(h, rp["moe.router.gate_w"], ref_experts, K)
        return F.linear(h + moe, rp["out.weight"], rp["out.bias"])

    loss_ref = 0.0
    for s, blk in enumerate(shards):
        ys = dense_model(blk["x"])
        if s == rank:
            y_ref = ys.detach()
        loss_ref = loss_ref + (ys * blk["cot"]).sum()
    loss_ref.backward()
    ref_grads = {n: t.grad for n, t in rp.items()}
    for i, eid in enumerate(model.moe.local_expert_ids):
        for j, pname in enumerate(("w_gate", "w_up", "w_down")):
            rg = ref_experts[eid][j].grad
            ref_grads[f"moe.experts.{i}.{pname}.weight"] = (
                rg if rg is not None else torch.zeros_like(ref_experts[eid][j])
            )
    ref_after = {}
    with torch.no_grad():
        for n, t in rp.items():
            ref_after[n] = t - LR * t.grad
        for i, eid in enumerate(model.moe.local_expert_ids):
            for j, pname in enumerate(("w_gate", "w_up", "w_down")):
                t = ref_experts[eid][j]
                gt = t.grad if t.grad is not None else torch.zeros_like(t)
                ref_after[f"moe.experts.{i}.{pname}.weight"] = t - LR * gt

    # ---- Distributed step ----
    checks = [check(kind, name, False, err) for kind, name, err in engine_errors]
    for n, p in model.named_parameters():
        if n in replicated:
            checks.append(compare("fwd", f"init: {n} equals rank 0's", p, rank0[n]))
    y = model(x)
    checks.append(compare("fwd", "output", y, y_ref))
    (y * cot).sum().backward()
    if cfg["engine_sync"]:
        # The engine must sum replicated params' grads across the EP ranks,
        # which hold disjoint tokens (F21).
        try:
            from pkg.distributed.grad_sync import sync_replicated_grads
        except ImportError as exc:
            for kind in ("grad", "step"):
                checks.append(check(kind, "engine sync_replicated_grads", False, repr(exc)))
        else:
            sync_replicated_grads(model, topo)
    else:
        # Harness stand-in for the replicated-gradient sync the engine lacked
        # (F21): EP ranks hold disjoint tokens, so sum replicated grads.
        for n, p in model.named_parameters():
            if n in replicated and p.grad is not None:
                dist.all_reduce(p.grad)
    for n, p in model.named_parameters():
        checks.append(compare("grad", f"{n}.grad", p.grad, ref_grads[n]))
    opt = torch.optim.SGD(model.parameters(), lr=LR)
    opt.step()
    for n, p in model.named_parameters():
        checks.append(compare("step", f"{n} after SGD step", p, ref_after[n]))
    return checks


# (id, cfg, findings failing fwd, grad, step). "harness" keeps the test-side
# stand-ins (rank-0 broadcast, summed replicated grads); the engine-* variants
# switch them off and rely on the engine instead.
_COMPOSITE_CASES = [
    ("harness", dict(E=8, engine_init=False, engine_sync=False), (), (), ()),
    ("engine-sync", dict(E=8, engine_init=False, engine_sync=True), (), (), ()),
    (
        "engine-init",
        dict(E=8, engine_init=True, engine_sync=False),
        (),
        (),
        (),
    ),
    (
        "engine-all",
        dict(E=8, engine_init=True, engine_sync=True),
        (),
        (),
        (),
    ),
]


@functools.lru_cache(maxsize=None)
def _composite_result(variant: str = "harness"):
    cfg = next(c for c in _COMPOSITE_CASES if c[0] == variant)[1]
    return run_multiprocess(_composite_worker, 2, cfg)


def test_composite_ep2_forward_matches_dense():
    _assert_kind(_composite_result(), "fwd")


def test_composite_ep2_gradients_match_dense():
    _assert_kind(_composite_result(), "grad")


def test_composite_ep2_sgd_step_matches_dense():
    _assert_kind(_composite_result(), "step")


_ENGINE_CASES = [c for c in _COMPOSITE_CASES if c[0] != "harness"]


@pytest.mark.parametrize("variant", _params(_ENGINE_CASES, 2))
def test_composite_engine_forward_matches_dense(variant):
    _assert_kind(_composite_result(variant), "fwd")


@pytest.mark.parametrize("variant", _params(_ENGINE_CASES, 3))
def test_composite_engine_gradients_match_dense(variant):
    _assert_kind(_composite_result(variant), "grad")


@pytest.mark.parametrize("variant", _params(_ENGINE_CASES, 4))
def test_composite_engine_sgd_step_matches_dense(variant):
    _assert_kind(_composite_result(variant), "step")


# ===========================================================================
# Layout independence: one global batch under many (dp, ep, sp) layouts
# ===========================================================================

# Global batch: _LG sequences of length _LS per micro-batch, _LMB micro-batches
# (gradient accumulation). Sequence s of micro-batch mb has a loss mask over
# its first _LMASK[mb][s] tokens, so data ranks hold unequal numbers of
# loss-bearing tokens (in some layouts one holds none) and the micro-batches
# differ too. A per-rank or per-micro-batch mean then weights tokens
# unequally; only a mean over the global batch matches the dense reference.
_LG, _LS, _LMB = 4, 7, 2
_LMASK = ((7, 2, 5, 1), (3, 6, 0, 4))


class _LayoutModel(nn.Module):
    """Replicated input projection -> sequence-parallel block (fused
    projection of each sequence shard with a replicated weight) -> MoE ->
    output projection, with residuals. At tp = 1 the SP block is a plain
    projection; dp and ep split the batch; ep shards the experts."""

    def __init__(self, H, Fd, E, K, topo):
        super().__init__()
        self.topo = topo
        self.inp = nn.Linear(H, H)
        self.sp_w = nn.Parameter(torch.randn(H, H) / math.sqrt(H))
        self.moe = DistributedMoELayer(H, Fd, E, K, topo)
        self.out = nn.Linear(H, H)

    def forward(self, x):
        h = torch.tanh(self.inp(x))
        s = torch.tanh(scatter_to_sequence_parallel(h, self.topo, next_weight=self.sp_w))
        g = gather_from_sequence_parallel(s, self.topo, orig_seq_len=x.shape[1])
        return self.out(g + self.moe(g))


def _token_loss_sum(y, t, m):
    """Sum over loss-bearing tokens of 0.5 * ||y - t||^2."""
    return (0.5 * ((y - t) ** 2).sum(-1) * m).sum()


def _full(t):
    """Full tensor of an FSDP2 DTensor (a DP-group collective), else ``t``."""
    return t.full_tensor() if hasattr(t, "full_tensor") else t


def _layout_worker(rank: int, ws: int, cfg: dict) -> list:
    H, Fd, E, K, LR = 16, 32, 8, 2, 0.1
    dp, ep, tp = cfg["dp"], cfg["ep"], cfg["tp"]
    topo = build_topology(dp_size=dp, ep_size=ep, tp_size=tp, device_type="cpu")
    torch.manual_seed(SEED + rank)
    model = _LayoutModel(H, Fd, E, K, topo)
    mark_sequence_parallel(model.sp_w)
    broadcast_replicated_params(model, topo)

    # ---- The global batch (identical on every rank) and this rank's part.
    #      TP ranks of one (dp, ep) position hold the same sequences. ----
    g = torch.Generator().manual_seed(SEED + 4242)
    X = torch.randn(_LMB, _LG, _LS, H, generator=g)
    T = torch.randn(_LMB, _LG, _LS, H, generator=g)
    M = torch.zeros(_LMB, _LG, _LS)
    for mb, lengths in enumerate(_LMASK):
        for s, n in enumerate(lengths):
            M[mb, s, :n] = 1.0
    n_global = int(M.sum())
    per = _LG // (dp * ep)
    d = topo.dp_rank * ep + topo.ep_rank
    mine = slice(d * per, (d + 1) * per)

    # ---- Dense reference: full experts from every (expert, TP shard),
    #      replicated params from rank 0; loss = global-batch mean. ----
    shards = _all_gather_obj(
        {
            "experts": {
                (eid, topo.tp_rank): tuple(
                    _detached(w) for w in (m.w_gate.weight, m.w_up.weight, m.w_down.weight)
                )
                for eid, m in zip(model.moe.local_expert_ids, model.moe.experts)
            },
            "rep": {
                n: _detached(p)
                for n, p in model.named_parameters()
                if not n.startswith("moe.experts.")
            },
        }
    )
    parts = {}
    for s in shards:
        parts.update(s["experts"])
    ref_experts = [
        [
            torch.cat([parts[(e, r)][0] for r in range(tp)], 0).requires_grad_(True),
            torch.cat([parts[(e, r)][1] for r in range(tp)], 0).requires_grad_(True),
            torch.cat([parts[(e, r)][2] for r in range(tp)], 1).requires_grad_(True),
        ]
        for e in range(E)
    ]
    rp = {n: t.clone().requires_grad_(True) for n, t in shards[0]["rep"].items()}

    def dense_model(xx):
        h = torch.tanh(F.linear(xx, rp["inp.weight"], rp["inp.bias"]))
        gg = torch.tanh(F.linear(h, rp["sp_w"]))
        moe = dense_moe(gg.reshape(-1, H), rp["moe.router.gate_w"], ref_experts, K)
        return F.linear(gg + moe.reshape(gg.shape), rp["out.weight"], rp["out.bias"])

    loss_ref = 0.0
    y_ref = []
    for mb in range(_LMB):
        ys = dense_model(X[mb])
        y_ref.append(ys[mine].detach())
        loss_ref = loss_ref + _token_loss_sum(ys, T[mb], M[mb])
    (loss_ref / n_global).backward()

    chunk = Fd // tp
    rows = slice(topo.tp_rank * chunk, (topo.tp_rank + 1) * chunk)

    def ref_of(name, tensors_of):
        if not name.startswith("moe.experts."):
            return tensors_of(rp[name])
        i, pname = name.split(".")[2], name.split(".")[3]
        eid = model.moe.local_expert_ids[int(i)]
        j = ("w_gate", "w_up", "w_down").index(pname)
        full = tensors_of(ref_experts[eid][j])
        return full[:, rows] if pname == "w_down" else full[rows]

    def grad_of(t):
        return t.grad if t.grad is not None else torch.zeros_like(t)

    with torch.no_grad():
        ref_grads = {n: ref_of(n, grad_of) for n, _ in model.named_parameters()}
        ref_after = {
            n: ref_of(n, lambda t: t - LR * grad_of(t)) for n, _ in model.named_parameters()
        }

    # ---- Engine path: FSDP2 over DP, then train.py's step structure. ----
    # Each micro-batch backpropagates its local loss SUM; finalize_grads
    # divides by the global token count once per step (ADR-011).
    model = apply_fsdp2(model, topo)
    checks = []
    local_tokens = 0
    for mb in range(_LMB):
        y = model(X[mb][mine])
        checks.append(compare("fwd", f"micro-batch {mb} output", y, y_ref[mb]))
        _token_loss_sum(y, T[mb][mine], M[mb][mine]).backward()
        local_tokens += int(M[mb][mine].sum())
    n_engine = global_token_count(local_tokens, topo)
    checks.append(
        check("grad", "global_token_count", n_engine == n_global, f"{n_engine} vs {n_global}")
    )
    finalize_grads(model, topo, n_engine)
    for n, p in model.named_parameters():
        grad = _full(p.grad) if p.grad is not None else None
        checks.append(compare("grad", f"{n}.grad", grad, ref_grads[n]))
    torch.optim.SGD(model.parameters(), lr=LR).step()
    for n, p in model.named_parameters():
        checks.append(compare("step", f"{n} after SGD step", _full(p.detach()), ref_after[n]))
    return checks


# (id, cfg, findings failing fwd, grad, step); world size = dp * ep * tp.
_LAYOUT_CASES = [
    ("single", dict(dp=1, ep=1, tp=1), (), (), ()),
    ("dp2", dict(dp=2, ep=1, tp=1), (), (), ()),
    ("ep2", dict(dp=1, ep=2, tp=1), (), (), ()),
    ("sp2", dict(dp=1, ep=1, tp=2), (), (), ()),
    ("dp2-ep2", dict(dp=2, ep=2, tp=1), (), (), ()),
    ("dp2-sp2", dict(dp=2, ep=1, tp=2), (), ("F26",), ("F26",)),
    ("ep2-sp2", dict(dp=1, ep=2, tp=2), (), (), ()),
]


@functools.lru_cache(maxsize=None)
def _layout_result(cid: str):
    cfg = next(c for c in _LAYOUT_CASES if c[0] == cid)[1]
    return run_multiprocess(_layout_worker, cfg["dp"] * cfg["ep"] * cfg["tp"], cfg)


@pytest.mark.parametrize("cid", _params(_LAYOUT_CASES, 2))
def test_layout_forward_matches_dense(cid):
    _assert_kind(_layout_result(cid), "fwd")


@pytest.mark.parametrize("cid", _params(_LAYOUT_CASES, 3))
def test_layout_gradients_match_dense_global_mean(cid):
    _assert_kind(_layout_result(cid), "grad")


@pytest.mark.parametrize("cid", _params(_LAYOUT_CASES, 4))
def test_layout_sgd_step_matches_dense_global_mean(cid):
    _assert_kind(_layout_result(cid), "step")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
