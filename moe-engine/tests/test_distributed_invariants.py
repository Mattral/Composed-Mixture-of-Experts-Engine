"""
tests/test_distributed_invariants.py
=====================================

Distributed invariant tests — 4-process Gloo world (CPU-only, no GPU).

Two invariants verified across real multi-process collective execution:

  1. **Token conservation** — across a 4-rank EP world, the total dispatched
     token count aggregated via all_reduce equals N_local × K × world_size
     exactly.  No token is dropped, duplicated, or miscounted by the routing
     or dispatch path.

  2. **No NaN gradients** — a forward → loss → backward pass through
     DistributedMoELayer produces finite gradients on every named parameter
     across every rank.  This catches silent NaN propagation through the
     SwiGLU expert or the combine weighted-sum.

Design notes
------------
* Both tests share the same `_run_worker` function, which avoids duplicating
  the process-group bootstrap and layer construction.
* Ports are allocated dynamically (passed as arguments) to eliminate the
  fixed-port collision that caused flakiness in earlier CI runs.
* The `_SimpleMesh` shim has been replaced by a real `build_topology` call
  so the test exercises the same code path as production.
* Workers communicate results back to the test process via a shared
  `mp.Queue` so assertion failures surface cleanly rather than as
  mysterious non-zero exit codes.
"""

from __future__ import annotations

import os
import socket
from typing import Optional

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from pkg.distributed.parallel_mesh import (
    DistributedMoELayer,
    ParallelTopology,
    build_topology,
)


# ---------------------------------------------------------------------------
# Free-port helper (also in conftest.py, duplicated here for standalone use)
# ---------------------------------------------------------------------------
def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# ---------------------------------------------------------------------------
# Shared worker
# ---------------------------------------------------------------------------
def _run_worker(
    rank: int,
    world_size: int,
    port: int,
    result_queue: mp.Queue,
) -> None:
    """
    Bootstrap a Gloo PG, build DistributedMoELayer on a 2-rank EP topology,
    run forward + backward, then push per-rank results to result_queue.

    Pushes a dict:
        {"rank": int, "token_conservation": bool, "no_nan_grads": bool,
         "error": Optional[str]}
    """
    error: Optional[str] = None
    token_ok = False
    nan_ok = False

    try:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(port)
        dist.init_process_group(
            backend="gloo", rank=rank, world_size=world_size,
        )

        # Build a real topology via build_topology (not a hand-rolled shim).
        # ep_size=2 so two pairs of (dp,ep) ranks exist in world_size=4.
        topo = build_topology(
            dp_size=world_size // 2,
            ep_size=2,
            device_type="cpu",
        )

        H, F, E, K = 64, 128, 8, 2
        B, S = 2, 4

        layer = DistributedMoELayer(
            hidden_dim=H, ffn_dim=F, num_experts=E, top_k=K,
            topology=topo, dtype=torch.float32,
        )

        tokens = torch.randn(B, S, H, dtype=torch.float32)

        # ── Forward ──────────────────────────────────────────────────────
        out = layer(tokens)

        # NaN guard on output
        assert not torch.isnan(out).any(), f"rank {rank}: NaN in layer output"

        # ── Token conservation ───────────────────────────────────────────
        flat = tokens.reshape(B * S, H)
        _, _, dispatch_cnt = layer.router(flat)
        local_dispatched = dispatch_cnt.sum().long()

        total_tensor = local_dispatched.clone()
        dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
        total_dispatched = int(total_tensor.item())
        expected = B * S * K * world_size

        token_ok = (total_dispatched == expected)
        assert token_ok, (
            f"rank {rank}: token conservation broken — "
            f"dispatched={total_dispatched}, expected={expected}"
        )

        # ── Backward ─────────────────────────────────────────────────────
        out.abs().sum().backward()

        nan_ok = True
        for name, p in layer.named_parameters():
            if p.grad is not None and not torch.isfinite(p.grad).all():
                nan_ok = False
                error = f"rank {rank}: non-finite grad in {name}"
                break

    except Exception as exc:
        error = f"rank {rank}: {type(exc).__name__}: {exc}"
    finally:
        try:
            if dist.is_initialized():
                dist.destroy_process_group()
        except Exception:
            pass

    result_queue.put({
        "rank": rank,
        "token_conservation": token_ok,
        "no_nan_grads": nan_ok,
        "error": error,
    })


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------
def _spawn_and_collect(port: int) -> list[dict]:
    """Spawn 4 workers, collect their result dicts, return the list."""
    ctx = mp.get_context("spawn")
    q: mp.Queue = ctx.Queue()
    world = 4
    procs = [
        ctx.Process(target=_run_worker, args=(r, world, port, q))
        for r in range(world)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=120)
        assert p.exitcode == 0, (
            f"Worker rank {procs.index(p)} exited with code {p.exitcode}"
        )
    results = []
    while not q.empty():
        results.append(q.get_nowait())
    return results


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_token_conservation_distributed() -> None:
    """Total dispatched tokens across all ranks must equal N_local × K × world."""
    port = _free_port()
    results = _spawn_and_collect(port)

    assert len(results) == 4, f"Expected 4 rank results, got {len(results)}"
    for r in results:
        if r["error"] and not r["token_conservation"]:
            pytest.fail(r["error"])
        assert r["token_conservation"], (
            f"rank {r['rank']}: token conservation failed. error={r['error']}"
        )


def test_distributed_backward_no_nan() -> None:
    """No parameter gradient may be NaN or Inf after a forward+backward pass."""
    port = _free_port()
    results = _spawn_and_collect(port)

    assert len(results) == 4, f"Expected 4 rank results, got {len(results)}"
    for r in results:
        if r["error"] and not r["no_nan_grads"]:
            pytest.fail(r["error"])
        assert r["no_nan_grads"], (
            f"rank {r['rank']}: non-finite gradient detected. error={r['error']}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
