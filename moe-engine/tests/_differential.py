"""
tests/_differential.py
======================

Support code for ``tests/test_differential.py``: dense single-device
references, a tolerance-checked tensor comparison, and a Gloo multi-process
runner.

The references below deliberately import nothing from ``pkg``. They are
written from the math (softmax top-k routing, SwiGLU experts, plain
``F.linear``), not from the engine, so a bug shared between the engine and
its reference cannot hide. Every reference runs on a single device with no
collectives.

Gradient conventions used by the differential tests
---------------------------------------------------
Each distributed run is compared against one global scalar loss evaluated
on the dense model:

* Outputs that are **replicated** across a group (TP, SP gather): every rank
  evaluates the identical loss term; the loss is not summed over replicas.
* Outputs that are **partitioned** across a group (EP ranks hold different
  tokens): the global loss is the sum of the rank-local loss terms.

Each loss term is ``(output * cotangent).sum()`` with a fixed random
cotangent, so a gradient check is a vector-Jacobian-product check that does
not depend on the forward output being right.

Expected local gradients:

* A **sharded** tensor (expert weights under EP, Column/Row weight shards
  under TP, sequence shards under SP) must hold its slice of the dense
  gradient.
* A **replicated** tensor must hold the full dense gradient, except where a
  test says it compares the group-summed gradient instead (replicated
  parameters whose cross-rank gradient reduction belongs to the data-parallel
  or optimizer layer, not to the op under test).
"""

from __future__ import annotations

import datetime
import os
import queue
import socket
import time
import traceback
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

# fp32 tolerance from the audit handoff (M0). Do not loosen.
ATOL = 1e-5
RTOL = 1e-4

# Gloo collective timeout inside workers. A rank that raises before a
# collective (e.g. F4's split-size error on one rank only) would otherwise
# leave its peers blocked for the default 30 minutes.
PG_TIMEOUT_S = 30
# Wall-clock budget for one multi-process case, spawn included.
RUN_TIMEOUT_S = 150


# ===========================================================================
# Comparison
# ===========================================================================


def compare(kind: str, name: str, actual: Optional[torch.Tensor], expected: torch.Tensor) -> dict:
    """Compare ``actual`` to ``expected`` at (ATOL, RTOL).

    ``actual is None`` (a gradient autograd never produced) is compared as
    zeros, and noted, so a parameter that legitimately received no tokens
    passes while one that should have trained fails with a clear message.

    Returns a plain picklable dict: kind, name, ok, max_abs, rel, note.
    """
    expected = expected.detach().to(torch.float64)
    note = ""
    if actual is None:
        note = "grad is None"
        actual64 = torch.zeros_like(expected)
    else:
        actual64 = actual.detach().to(torch.float64)
    if actual64.shape != expected.shape:
        return dict(
            kind=kind,
            name=name,
            ok=False,
            max_abs=float("inf"),
            rel=float("inf"),
            note=f"shape {tuple(actual64.shape)} != expected {tuple(expected.shape)}",
        )
    diff = actual64 - expected
    max_abs = float(diff.abs().max()) if diff.numel() else 0.0
    ref_norm = float(expected.norm())
    rel = float(diff.norm()) / ref_norm if ref_norm > 0 else float(diff.norm())
    ok = bool(torch.allclose(actual64, expected, atol=ATOL, rtol=RTOL))
    return dict(kind=kind, name=name, ok=ok, max_abs=max_abs, rel=rel, note=note)


def check(kind: str, name: str, ok: bool, note: str) -> dict:
    """A non-tensor harness assertion, reported like a comparison."""
    return dict(kind=kind, name=name, ok=bool(ok), max_abs=0.0, rel=0.0, note=note)


def distributed_fd_check(
    name: str,
    fn: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    cot: torch.Tensor,
    *,
    input_replicated: bool,
    output_replicated: bool,
    eps: float = 1e-6,
) -> dict:
    """Finite-difference check of a collective's VJP across all ranks.

    ``torch.autograd.gradcheck`` cannot be used per rank: each rank would
    perturb only its own input while its output also depends on its peers'.
    Here every rank evaluates one *global* scalar loss and all ranks perturb
    the same global input element in lockstep (so every rank issues the same
    number of collectives):

    * ``input_replicated``: ``x`` is one tensor held identically by every
      rank, so a perturbation is applied on all ranks at once and every rank
      must end up with the full gradient. Otherwise ``x`` is sharded: element
      ``(r, i)`` is perturbed on rank ``r`` only.
    * ``output_replicated``: every rank holds the same output, so the loss is
      ``<cot, fn(x)>`` counted once (``cot`` must then be identical on all
      ranks). Otherwise outputs are partitioned and the loss is the sum of
      every rank's ``<cot_r, fn_r(x)>``.

    The analytical gradient is each rank's ``x.grad`` after backward of its
    local ``<cot, fn(x)>``. Use float64; ``fn`` must be deterministic.
    """
    x = x.detach()
    xa = x.clone().requires_grad_(True)
    (fn(xa) * cot).sum().backward()
    analytical = xa.grad if xa.grad is not None else torch.zeros_like(x)

    def global_loss(xp: torch.Tensor) -> float:
        with torch.no_grad():
            val = (fn(xp) * cot).sum().reshape(1)
            if not output_replicated:
                dist.all_reduce(val)
        return float(val)

    rank = dist.get_rank()
    if input_replicated:
        elements = [(None, i) for i in range(x.numel())]
    else:
        sizes = [None] * dist.get_world_size()
        dist.all_gather_object(sizes, x.numel())
        elements = [(r, i) for r, n in enumerate(sizes) for i in range(n)]

    numeric = torch.zeros_like(x)
    for owner, i in elements:
        mine = owner is None or owner == rank
        losses = []
        for sign in (1.0, -1.0):
            xp = x.clone()
            if mine:
                xp.view(-1)[i] += sign * eps
            losses.append(global_loss(xp))
        if mine:
            numeric.view(-1)[i] = (losses[0] - losses[1]) / (2 * eps)
    return compare("grad", f"{name}: VJP vs finite differences", analytical, numeric)


# ===========================================================================
# Dense references (no pkg imports, no collectives)
# ===========================================================================


def dense_router(
    x: torch.Tensor, gate_w: torch.Tensor, k: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Softmax top-k router with renormalized weights, computed in fp64."""
    probs = torch.softmax(x.to(torch.float64) @ gate_w.to(torch.float64), dim=-1)
    vals, idx = torch.topk(probs, k=k, dim=-1)
    weights = (vals / vals.sum(dim=-1, keepdim=True)).to(x.dtype)
    return idx, weights


def per_slot_capacity_drop_mask(idx: torch.Tensor, capacity: int) -> torch.Tensor:
    """First-come-first-served capacity per expert, applied independently to
    each top-k slot column.

    This mirrors the engine's existing per-slot semantics (pinned by
    ``test_capacity_dropping.py::test_multi_k_independent_per_slot``); it is
    written as a plain Python loop so it shares no code with the engine.
    """
    N, K = idx.shape
    mask = torch.zeros(N, K, dtype=torch.bool)
    for k in range(K):
        seen: Dict[int, int] = {}
        for n in range(N):
            e = int(idx[n, k])
            mask[n, k] = seen.get(e, 0) >= capacity
            seen[e] = seen.get(e, 0) + 1
    return mask


def dense_aux_sums(
    x: torch.Tensor, gate_w: torch.Tensor, k: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """One MoE layer's auxiliary-loss sums over *all* tokens ``x`` of a
    micro-batch (fp64): router z-loss ``sum_t logsumexp(l_t)^2`` and the
    Switch load-balancing sum ``E * sum_e f_e * sum_t p_te``, where ``f_e``
    is the fraction of the ``N * k`` top-k assignments that go to expert
    ``e`` (not differentiable)."""
    logits = x.to(torch.float64) @ gate_w.to(torch.float64)
    n, e = logits.shape
    probs = torch.softmax(logits, dim=-1)
    idx = torch.topk(probs, k=k, dim=-1).indices
    f = torch.bincount(idx.reshape(-1), minlength=e).to(torch.float64) / (n * k)
    z = (torch.logsumexp(logits, dim=-1) ** 2).sum()
    lb = e * (f * probs.sum(0)).sum()
    return z, lb


def swiglu(x: torch.Tensor, w_gate: torch.Tensor, w_up: torch.Tensor, w_down: torch.Tensor):
    return F.linear(F.silu(F.linear(x, w_gate)) * F.linear(x, w_up), w_down)


def dense_moe(
    x: torch.Tensor,
    gate_w: torch.Tensor,
    experts: Sequence[Sequence[torch.Tensor]],
    k: int,
    capacity: Optional[int] = None,
) -> torch.Tensor:
    """Dense top-k MoE on one device.

    ``experts[e]`` is ``(w_gate [F, H], w_up [F, H], w_down [H, F])`` for
    global expert ``e``. Dropped (token, slot) pairs get zero combine weight.
    """
    idx, weights = dense_router(x, gate_w, k)
    if capacity is not None:
        weights = weights.masked_fill(per_slot_capacity_drop_mask(idx, capacity), 0.0)
    out = torch.zeros_like(x)
    for e, (wg, wu, wd) in enumerate(experts):
        rows, slots = (idx == e).nonzero(as_tuple=True)
        if rows.numel() == 0:
            continue
        contrib = swiglu(x[rows], wg, wu, wd) * weights[rows, slots].unsqueeze(-1)
        out = out.index_add(0, rows, contrib)
    return out


# ===========================================================================
# Multi-process runner
# ===========================================================================


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _entry(worker: Callable, rank: int, world_size: int, port: int, cfg: dict, q) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    result: dict
    try:
        dist.init_process_group(
            "gloo",
            rank=rank,
            world_size=world_size,
            timeout=datetime.timedelta(seconds=PG_TIMEOUT_S),
        )
        result = {"rank": rank, "error": None, "checks": worker(rank, world_size, cfg)}
    except Exception:
        result = {"rank": rank, "error": traceback.format_exc()[-2500:], "checks": []}
    q.put(result)
    try:
        dist.destroy_process_group()
    except Exception:
        pass


def run_multiprocess(worker: Callable, world_size: int, cfg: dict) -> Dict[int, Optional[dict]]:
    """Run ``worker(rank, world_size, cfg) -> List[check dict]`` on
    ``world_size`` Gloo CPU processes. Returns ``{rank: result or None}``;
    ``None`` means the rank never reported (hung past the timeout, or died).
    """
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    port = _free_port()
    procs = [
        ctx.Process(target=_entry, args=(worker, r, world_size, port, cfg, q), daemon=True)
        for r in range(world_size)
    ]
    for p in procs:
        p.start()
    results: Dict[int, Optional[dict]] = {r: None for r in range(world_size)}
    deadline = time.monotonic() + RUN_TIMEOUT_S
    received = 0
    while received < world_size:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        try:
            res = q.get(timeout=min(remaining, 5.0))
        except queue.Empty:
            if not any(p.is_alive() for p in procs) and q.empty():
                break
            continue
        results[res["rank"]] = res
        received += 1
    for p in procs:
        p.join(timeout=10)
        if p.is_alive():
            p.terminate()
            p.join(timeout=5)
    return results


def failures_for(results: Dict[int, Optional[dict]], kind: str) -> List[str]:
    """Collect human-readable failures for checks of ``kind``.

    A rank that raised or never reported is a failure for every kind. A run
    that produced no check of ``kind`` at all is also a failure, so a case
    can never pass vacuously.
    """
    out: List[str] = []
    n_checks = 0
    for rank in sorted(results):
        res = results[rank]
        if res is None:
            out.append(f"rank {rank}: no result (hung past {RUN_TIMEOUT_S}s or crashed)")
            continue
        if res["error"]:
            out.append(f"rank {rank}: raised\n{res['error']}")
            continue
        for c in res["checks"]:
            if c["kind"] != kind:
                continue
            n_checks += 1
            if not c["ok"]:
                note = f" ({c['note']})" if c["note"] else ""
                out.append(
                    f"rank {rank} {c['name']}: max_abs={c['max_abs']:.3e} rel={c['rel']:.3e}{note}"
                )
    if not out and n_checks == 0:
        out.append(f"no '{kind}' checks ran")
    return out
