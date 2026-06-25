"""
tests/test_tensor_parallel.py
==============================

Unit tests for Tensor Parallelism layers.

Coverage
--------
* **ColumnParallelLinear** — shape, dtype, grad flow, bias/no-bias,
  numerical equivalence to nn.Linear at tp_size=1 (single-process).
* **RowParallelLinear** — shape, dtype, grad flow, bias/no-bias,
  numerical equivalence to nn.Linear at tp_size=1.
* **_SwiGLUExpert** — both w_gate and w_up are now ColumnParallel;
  verify output shape and grad flow; verify w_gate is ColumnParallelLinear.
* **scatter/gather_sequence_parallel** — no-op identity at tp_size=1;
  round-trip scatter→gather recovers original tensor.
* **Multi-rank equivalence (mp.spawn)** — 2-rank TP group: Column+Row
  combined produce identical output to single-rank nn.Linear. This is the
  critical test that actually exercises the collectives end-to-end.
* **RowParallelLinear all_reduce correctness** — verify that the correct
  collective (all_reduce sum) is used, not reduce_scatter+all_gather.

All multi-process tests are marked ``@pytest.mark.skipif`` unless
``torch.cuda.is_available()`` or the test explicitly spawns CPU workers.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pkg.distributed.parallel_mesh import (  # noqa: E402
    ColumnParallelLinear,
    RowParallelLinear,
    _SwiGLUExpert,
    build_topology,
    gather_from_sequence_parallel,
    scatter_to_sequence_parallel,
)

pytestmark = pytest.mark.cpu

# ---------------------------------------------------------------------------
# ColumnParallelLinear — single-process (tp_size=1)
# ---------------------------------------------------------------------------


def test_column_parallel_forward_shape():
    layer = ColumnParallelLinear(128, 256, bias=True)
    y = layer(torch.randn(32, 128))
    assert y.shape == (32, 256)
    assert not torch.isnan(y).any()


def test_column_parallel_backward():
    layer = ColumnParallelLinear(64, 128, bias=True)
    x = torch.randn(16, 64, requires_grad=True)
    layer(x).sum().backward()
    assert x.grad is not None and x.grad.shape == x.shape
    assert layer.weight.grad is not None and layer.weight.grad.shape == layer.weight.shape
    assert layer.bias.grad is not None and layer.bias.grad.shape == layer.bias.shape


def test_column_parallel_no_bias():
    layer = ColumnParallelLinear(128, 256, bias=False)
    assert layer.bias is None
    assert layer(torch.randn(4, 128)).shape == (4, 256)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_column_parallel_dtype(dtype):
    layer = ColumnParallelLinear(32, 64, dtype=dtype)
    y = layer(torch.randn(4, 32, dtype=dtype))
    assert y.dtype == dtype


def test_column_parallel_numerically_correct_tp1():
    """At tp_size=1, ColumnParallel must be identical to nn.Linear."""
    torch.manual_seed(0)
    B, H, F = 16, 64, 128
    ref = nn.Linear(H, F, bias=True)
    col = ColumnParallelLinear(H, F, bias=True)
    col.weight.data.copy_(ref.weight.data)
    col.bias.data.copy_(ref.bias.data)
    x = torch.randn(B, H)
    with torch.no_grad():
        assert torch.allclose(col(x), ref(x), atol=1e-6), (
            "ColumnParallel@tp=1 diverges from nn.Linear"
        )


# ---------------------------------------------------------------------------
# RowParallelLinear — single-process (tp_size=1)
# ---------------------------------------------------------------------------


def test_row_parallel_forward_shape():
    layer = RowParallelLinear(128, 256, bias=True)
    y = layer(torch.randn(32, 128))
    assert y.shape == (32, 256)
    assert not torch.isnan(y).any()


def test_row_parallel_backward():
    layer = RowParallelLinear(64, 128, bias=True)
    x = torch.randn(16, 64, requires_grad=True)
    layer(x).sum().backward()
    assert x.grad is not None and x.grad.shape == x.shape
    assert layer.weight.grad is not None
    assert layer.bias.grad is not None


def test_row_parallel_no_bias():
    layer = RowParallelLinear(128, 256, bias=False)
    assert layer.bias is None
    assert layer(torch.randn(4, 128)).shape == (4, 256)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_row_parallel_dtype(dtype):
    layer = RowParallelLinear(32, 64, dtype=dtype)
    y = layer(torch.randn(4, 32, dtype=dtype))
    assert y.dtype == dtype


def test_row_parallel_numerically_correct_tp1():
    """At tp_size=1, RowParallel must be identical to nn.Linear."""
    torch.manual_seed(1)
    B, H, F = 16, 128, 64
    ref = nn.Linear(H, F, bias=True)
    row = RowParallelLinear(H, F, bias=True)
    row.weight.data.copy_(ref.weight.data)
    row.bias.data.copy_(ref.bias.data)
    x = torch.randn(B, H)
    with torch.no_grad():
        assert torch.allclose(row(x), ref(x), atol=1e-6), "RowParallel@tp=1 diverges from nn.Linear"


def test_row_parallel_uses_all_reduce_not_reduce_scatter():
    """Verify the correct collective is used: RowParallel must call all_reduce,
    NOT reduce_scatter_tensor + all_gather_into_tensor."""
    import inspect

    src = inspect.getsource(RowParallelLinear.forward)
    assert "all_reduce" in src, (
        "RowParallelLinear.forward must use dist.all_reduce for sum-reduction"
    )
    assert "reduce_scatter_tensor" not in src, (
        "RowParallelLinear.forward must NOT use reduce_scatter_tensor (wrong collective)"
    )


# ---------------------------------------------------------------------------
# _SwiGLUExpert — both gate and up must be ColumnParallel
# ---------------------------------------------------------------------------


def test_swiglu_expert_shape():
    topo = build_topology(dp_size=1, ep_size=1, tp_size=1, device_type="cpu")
    expert = _SwiGLUExpert(hidden_dim=64, ffn_dim=128, topology=topo)
    x = torch.randn(8, 64)
    y = expert(x)
    assert y.shape == (8, 64), f"SwiGLU output shape wrong: {y.shape}"
    assert not torch.isnan(y).any()


def test_swiglu_expert_w_gate_is_column_parallel():
    """w_gate must be ColumnParallelLinear, not nn.Linear.
    This ensures consistent TP sharding: both gate and up are Column-sharded,
    their element-wise product stays in shard space [F//tp], then w_down
    (RowParallel) all_reduces to reconstruct the full hidden dimension.
    """
    topo = build_topology(dp_size=1, ep_size=1, tp_size=1, device_type="cpu")
    expert = _SwiGLUExpert(hidden_dim=64, ffn_dim=128, topology=topo)
    assert isinstance(expert.w_gate, ColumnParallelLinear), (
        f"w_gate must be ColumnParallelLinear, got {type(expert.w_gate).__name__}. "
        "Using plain nn.Linear for w_gate while w_up is ColumnParallel breaks TP "
        "sharding consistency at tp_size>1: gate and up projections would have "
        "different collective semantics."
    )
    assert isinstance(expert.w_up, ColumnParallelLinear)
    assert isinstance(expert.w_down, RowParallelLinear)


def test_swiglu_expert_backward():
    topo = build_topology(dp_size=1, ep_size=1, tp_size=1, device_type="cpu")
    expert = _SwiGLUExpert(hidden_dim=32, ffn_dim=64, topology=topo)
    x = torch.randn(4, 32, requires_grad=True)
    expert(x).sum().backward()
    assert x.grad is not None and x.grad.shape == x.shape
    for name, p in expert.named_parameters():
        assert p.grad is not None, f"No gradient for {name}"


def test_swiglu_expert_numerically_correct_tp1():
    """At tp_size=1, SwiGLU output must equal manual reference computation."""
    torch.manual_seed(42)
    topo = build_topology(dp_size=1, ep_size=1, tp_size=1, device_type="cpu")
    H, F = 32, 64
    expert = _SwiGLUExpert(hidden_dim=H, ffn_dim=F, topology=topo)
    x = torch.randn(4, H)

    with torch.no_grad():
        # Manual SwiGLU: down( silu(gate(x)) * up(x) )
        # Since tp_size=1, ColumnParallel and RowParallel are identity wraps
        gate_out = torch.nn.functional.linear(x, expert.w_gate.weight)
        up_out = torch.nn.functional.linear(x, expert.w_up.weight)
        hidden = torch.nn.functional.silu(gate_out) * up_out
        ref_out = torch.nn.functional.linear(hidden, expert.w_down.weight)
        got_out = expert(x)

    assert torch.allclose(got_out, ref_out, atol=1e-5), (
        f"SwiGLU output diverges from reference: max_diff={abs(got_out - ref_out).max():.2e}"
    )


# ---------------------------------------------------------------------------
# Sequence parallelism helpers — single-rank no-op
# ---------------------------------------------------------------------------


def test_scatter_sequence_parallel_identity_tp1():
    topo = build_topology(dp_size=1, tp_size=1, ep_size=1, device_type="cpu")
    x = torch.randn(4, 128, 64)
    assert torch.equal(scatter_to_sequence_parallel(x, topo), x)


def test_gather_sequence_parallel_identity_tp1():
    topo = build_topology(dp_size=1, tp_size=1, ep_size=1, device_type="cpu")
    x = torch.randn(4, 128, 64)
    assert torch.equal(gather_from_sequence_parallel(x, topo), x)


def test_scatter_gather_round_trip_tp1():
    """scatter then gather must recover the original tensor at tp_size=1."""
    topo = build_topology(dp_size=1, tp_size=1, ep_size=1, device_type="cpu")
    x = torch.randn(2, 64, 32)
    recovered = gather_from_sequence_parallel(scatter_to_sequence_parallel(x, topo), topo)
    assert torch.equal(recovered, x)


# ---------------------------------------------------------------------------
# Multi-rank TP correctness (2-rank CPU, mp.spawn)
# ---------------------------------------------------------------------------


def _tp2_worker(rank: int, world_size: int, result_queue, H: int, F: int, seed: int):
    """Worker: builds a 2-rank TP group, runs ColumnParallel→RowParallel,
    and verifies the combined output matches a single-rank nn.Linear reference."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29510"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    torch.manual_seed(seed)
    topo = build_topology(dp_size=1, tp_size=world_size, ep_size=1, device_type="cpu")

    # ---- Build sharded layers ----
    col = ColumnParallelLinear(H, F, bias=False, topology=topo)
    row = RowParallelLinear(F, H, bias=False, topology=topo)

    # ---- Build single-rank reference (same weight) ----
    # Gather weights from both ranks so every rank sees the full weight matrix
    col_full_w = torch.empty(F, H)
    # Each rank holds col.weight: [F//2, H]. Stack across ranks.
    gathered_col = [torch.empty_like(col.weight) for _ in range(world_size)]
    dist.all_gather(gathered_col, col.weight)
    col_full_w = torch.cat(gathered_col, dim=0)  # [F, H]

    row_full_w = torch.empty(H, F)
    gathered_row = [torch.empty_like(row.weight) for _ in range(world_size)]
    dist.all_gather(gathered_row, row.weight)
    row_full_w = torch.cat(gathered_row, dim=1)  # [H, F]

    torch.manual_seed(seed + 10)
    x = torch.randn(4, H)

    with torch.no_grad():
        # Sharded path (ColumnParallel all-gathers internally, RowParallel all_reduces)
        mid = col(x)  # [4, F] — all-gathered by ColumnParallel
        out_sharded = row(mid)  # [4, H] — all_reduced by RowParallel

        # Reference path (full matmul on every rank with reconstructed weight)
        ref_mid = torch.nn.functional.linear(x, col_full_w)  # [4, F]
        out_ref = torch.nn.functional.linear(ref_mid, row_full_w)  # [4, H]

    max_diff = float((out_sharded - out_ref).abs().max().item())
    result_queue.put((rank, max_diff))
    dist.destroy_process_group()


@pytest.mark.skipif(
    sys.platform == "darwin",
    reason="mp.spawn fork-safety issues on macOS CI",
)
def test_column_row_parallel_2rank_numerically_correct():
    """Two-rank TP: Column→Row combined output must match single-rank nn.Linear.

    This is the definitive correctness test for the TP implementation. It:
    1. Spawns 2 CPU Gloo workers.
    2. Builds ColumnParallel(H→F) + RowParallel(F→H) with sharded weights.
    3. Runs the sharded forward (Column all-gathers, Row all_reduces).
    4. Reconstructs the full weight matrices via all_gather.
    5. Runs the full (non-sharded) matmul as reference.
    6. Asserts max absolute difference < 1e-5.
    """
    H, F, seed = 64, 128, 7
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = []
    for rank in range(2):
        p = ctx.Process(
            target=_tp2_worker,
            args=(rank, 2, q, H, F, seed),
        )
        p.start()
        procs.append(p)
    for p in procs:
        p.join(timeout=60)
        assert p.exitcode == 0, f"Worker exited with code {p.exitcode}"

    results = {}
    while not q.empty():
        r, diff = q.get_nowait()
        results[r] = diff

    assert len(results) == 2, f"Expected results from 2 ranks, got {len(results)}"
    for rank, diff in results.items():
        assert diff < 1e-5, (
            f"Rank {rank}: TP output diverges from reference by {diff:.2e} "
            f"(tolerance 1e-5). Check ColumnParallel all-gather and RowParallel all_reduce."
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
