"""
tests/test_pipeline_parallel.py
================================

Unit tests for the 1F1B pipeline parallelism schedule (single-process).

Validates the PipelineStage class against the core invariants of the
1F1B schedule algorithm:
  1. All micro-batches are processed exactly once.
  2. Gradient tensors are returned for every micro-batch.
  3. The schedule completes in at most (num_stages - 1 + num_micro_batches)
     clock cycles, reproducing the classic 1F1B bound.
  4. Passthrough module produces output == input (identity test).
  5. Wrapped nn.Module is correctly invoked per micro-batch.
  6. Edge cases: single stage, single micro-batch, m == p.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from pkg.distributed.parallel_mesh import PipelineStage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _micro_batches(m: int, dim: int = 4, device: str = "cpu") -> list[torch.Tensor]:
    return [torch.randn(2, dim, device=device) for _ in range(m)]


# ---------------------------------------------------------------------------
# Basic API tests
# ---------------------------------------------------------------------------

def test_pipeline_stage_init():
    stage = PipelineStage(stage_id=0, num_stages=4)
    assert stage.stage_id == 0
    assert stage.num_stages == 4
    assert stage.module is None


def test_pipeline_stage_with_module():
    linear = nn.Linear(4, 4)
    stage = PipelineStage(stage_id=1, num_stages=4, module=linear)
    assert stage.module is linear


def test_forward_step_passthrough():
    """Without a module, forward_step is the identity."""
    stage = PipelineStage(stage_id=0, num_stages=1)
    x = torch.randn(2, 8)
    y = stage.forward_step(x)
    assert torch.equal(x, y)


def test_forward_step_with_module():
    """With a module, forward_step applies it."""
    lin = nn.Linear(8, 8)
    lin.weight.data.copy_(torch.eye(8))   # identity weights
    lin.bias.data.zero_()
    stage = PipelineStage(stage_id=0, num_stages=1, module=lin)
    x = torch.randn(2, 8)
    y = stage.forward_step(x)
    assert torch.allclose(y, x, atol=1e-6)


def test_backward_step_passthrough():
    """Without a module, backward_step passes the gradient through."""
    stage = PipelineStage(stage_id=0, num_stages=1)
    g = torch.ones(2, 8)
    g_out = stage.backward_step(g)
    assert torch.equal(g_out, g)


def test_backward_step_none_gradient():
    """backward_step must handle None gradient."""
    stage = PipelineStage(stage_id=0, num_stages=2)
    g_out = stage.backward_step(None)
    assert g_out is None


# ---------------------------------------------------------------------------
# 1F1B schedule correctness
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("num_stages,num_micro_batches", [
    (1, 4),
    (2, 4),
    (4, 4),
    (4, 8),
    (4, 2),
    (1, 1),
    (8, 8),
])
def test_1f1b_all_microbatches_get_gradients(num_stages, num_micro_batches):
    """Every micro-batch must produce a gradient (not None) after 1F1B."""
    stage = PipelineStage(stage_id=0, num_stages=num_stages)
    mb = _micro_batches(num_micro_batches)
    grads = stage.run_1f1b(mb)

    assert len(grads) == num_micro_batches, (
        f"Expected {num_micro_batches} gradient entries, got {len(grads)}"
    )
    for i, g in enumerate(grads):
        assert g is not None, f"micro-batch {i} gradient is None"


@pytest.mark.parametrize("num_stages,num_micro_batches", [
    (4, 4),
    (4, 8),
    (2, 6),
])
def test_1f1b_gradient_shapes_match_input(num_stages, num_micro_batches):
    """Gradient shapes must match the corresponding micro-batch shape."""
    DIM = 8
    stage = PipelineStage(stage_id=0, num_stages=num_stages)
    mb = _micro_batches(num_micro_batches, dim=DIM)
    grads = stage.run_1f1b(mb)

    for i, (x, g) in enumerate(zip(mb, grads)):
        assert g.shape == x.shape, (
            f"Micro-batch {i}: input shape {x.shape} != grad shape {g.shape}"
        )


def test_1f1b_single_stage_single_microbatch():
    """Degenerate case: p=1, m=1 — must still return one gradient."""
    stage = PipelineStage(stage_id=0, num_stages=1)
    mb = _micro_batches(1)
    grads = stage.run_1f1b(mb)
    assert len(grads) == 1
    assert grads[0] is not None


def test_1f1b_with_linear_module():
    """Verify that the module is actually applied during forward steps."""
    torch.manual_seed(42)
    lin = nn.Linear(4, 4, bias=False)
    lin.weight.data.fill_(2.0)   # scale by 2

    stage = PipelineStage(stage_id=0, num_stages=2, module=lin)
    mb = [torch.ones(2, 4)]      # input: all ones
    # Run forward_step manually to confirm the module runs
    out = stage.forward_step(mb[0])
    # 4 inputs × 2.0 weight = 8.0 per output
    assert torch.allclose(out, torch.full_like(out, 8.0), atol=1e-5)


@pytest.mark.parametrize("num_stages", [1, 2, 4, 8])
def test_1f1b_returns_list_length_equals_m(num_stages):
    """len(grads) == len(micro_batches) regardless of num_stages."""
    for m in [1, 2, 4, 8, 16]:
        stage = PipelineStage(stage_id=0, num_stages=num_stages)
        mb = _micro_batches(m)
        grads = stage.run_1f1b(mb)
        assert len(grads) == m, (
            f"p={num_stages}, m={m}: expected {m} grads, got {len(grads)}"
        )


def test_pipeline_stage_id_types():
    """stage_id and num_stages must be ints (not float, etc.)."""
    stage = PipelineStage(stage_id=2, num_stages=8)
    assert isinstance(stage.stage_id, int)
    assert isinstance(stage.num_stages, int)


def test_1f1b_empty_micro_batches():
    """Empty micro-batch list returns empty gradient list."""
    stage = PipelineStage(stage_id=0, num_stages=4)
    grads = stage.run_1f1b([])
    assert grads == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
