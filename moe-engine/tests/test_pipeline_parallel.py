"""
tests/test_pipeline_parallel.py
================================

Unit and integration tests for PipelineStage 1F1B schedule.

v0.3 additions
--------------
* ``test_pp_multiprocess_*`` — 2-rank mp.spawn tests that exercise the real
  dist.send/recv inter-stage communication path introduced in v0.3.
* ``test_run_1f1b_distributed_raises_single_process`` — verifies that
  run_1f1b_distributed raises a clear error when called without pp_size > 1.
* Existing single-process tests are unchanged (they use the fast-path).
"""

from __future__ import annotations

import os
import socket
import sys
from pathlib import Path
from typing import Optional

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pkg.distributed.parallel_mesh import PipelineStage, build_topology


# ---------------------------------------------------------------------------
# Single-process tests (unchanged from v0.2)
# ---------------------------------------------------------------------------

def _micro_batches(m: int, dim: int = 4, device: str = "cpu") -> list:
    return [torch.randn(2, dim, device=device) for _ in range(m)]


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
    stage = PipelineStage(stage_id=0, num_stages=1)
    x = torch.randn(2, 8)
    y = stage.forward_step(x)
    assert torch.equal(x, y)


def test_forward_step_with_module():
    lin = nn.Linear(8, 8)
    lin.weight.data.copy_(torch.eye(8))
    lin.bias.data.zero_()
    stage = PipelineStage(stage_id=0, num_stages=1, module=lin)
    x = torch.randn(2, 8)
    y = stage.forward_step(x)
    assert torch.allclose(y, x, atol=1e-6)


def test_backward_step_passthrough():
    stage = PipelineStage(stage_id=0, num_stages=1)
    g = torch.ones(2, 8)
    assert torch.equal(stage.backward_step(g), g)


def test_backward_step_none_gradient():
    stage = PipelineStage(stage_id=0, num_stages=2)
    assert stage.backward_step(None) is None


@pytest.mark.parametrize("num_stages,num_micro_batches", [
    (1, 4), (2, 4), (4, 4), (4, 8), (4, 2), (1, 1), (8, 8),
])
def test_1f1b_all_microbatches_get_gradients(num_stages, num_micro_batches):
    stage = PipelineStage(stage_id=0, num_stages=num_stages)
    mb = _micro_batches(num_micro_batches)
    grads = stage.run_1f1b(mb)
    assert len(grads) == num_micro_batches
    for i, g in enumerate(grads):
        assert g is not None, f"micro-batch {i} gradient is None"


@pytest.mark.parametrize("num_stages,num_micro_batches", [
    (4, 4), (4, 8), (2, 6),
])
def test_1f1b_gradient_shapes_match_input(num_stages, num_micro_batches):
    DIM = 8
    stage = PipelineStage(stage_id=0, num_stages=num_stages)
    mb = _micro_batches(num_micro_batches, dim=DIM)
    grads = stage.run_1f1b(mb)
    for i, (x, g) in enumerate(zip(mb, grads)):
        assert g.shape == x.shape, f"mb {i}: input {x.shape} != grad {g.shape}"


def test_1f1b_single_stage_single_microbatch():
    stage = PipelineStage(stage_id=0, num_stages=1)
    grads = stage.run_1f1b(_micro_batches(1))
    assert len(grads) == 1 and grads[0] is not None


def test_1f1b_with_linear_module():
    torch.manual_seed(42)
    lin = nn.Linear(4, 4, bias=False)
    lin.weight.data.fill_(2.0)
    stage = PipelineStage(stage_id=0, num_stages=2, module=lin)
    out = stage.forward_step(torch.ones(2, 4))
    assert torch.allclose(out, torch.full_like(out, 8.0), atol=1e-5)


@pytest.mark.parametrize("num_stages", [1, 2, 4, 8])
def test_1f1b_returns_list_length_equals_m(num_stages):
    for m in [1, 2, 4, 8, 16]:
        stage = PipelineStage(stage_id=0, num_stages=num_stages)
        grads = stage.run_1f1b(_micro_batches(m))
        assert len(grads) == m


def test_pipeline_stage_id_types():
    stage = PipelineStage(stage_id=2, num_stages=8)
    assert isinstance(stage.stage_id, int)
    assert isinstance(stage.num_stages, int)


def test_1f1b_empty_micro_batches():
    stage = PipelineStage(stage_id=0, num_stages=4)
    assert stage.run_1f1b([]) == []


def test_run_1f1b_distributed_raises_single_process():
    """run_1f1b_distributed must raise RuntimeError without dist + pp>1."""
    stage = PipelineStage(stage_id=0, num_stages=2)
    with pytest.raises(RuntimeError, match="run_1f1b_distributed requires pp_size"):
        stage.run_1f1b_distributed([torch.randn(2, 4)])


# ---------------------------------------------------------------------------
# Multi-process PP tests (v0.3) — real dist.send/recv
# ---------------------------------------------------------------------------

def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _pp2_worker(rank: int, world_size: int, port: int, result_queue, H: int, m: int):
    """Two-stage pipeline: stage 0 (rank 0) → stage 1 (rank 1)."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    topo = build_topology(dp_size=1, ep_size=1, tp_size=1, pp_size=world_size, device_type="cpu")

    # Stage 0: identity (passthrough). Stage 1: scale by 2.
    if rank == 0:
        module = None
    else:
        lin = nn.Linear(H, H, bias=False)
        with torch.no_grad():
            lin.weight.copy_(2.0 * torch.eye(H))
        module = lin

    stage = PipelineStage(stage_id=rank, num_stages=world_size, module=module, topology=topo)

    error = None
    outputs = []
    try:
        micro_batches = [torch.ones(2, H) for _ in range(m)] if rank == 0 else []
        out_list, loss_list = stage.run_1f1b_distributed(
            micro_batches,
            loss_fn=lambda o: o.sum() if rank == world_size - 1 else None,
        )
        # Last stage: verify outputs are 2× the input (scale-by-2 module)
        if rank == world_size - 1:
            for out in out_list:
                assert torch.allclose(out, torch.full_like(out, 2.0), atol=1e-5), (
                    f"Expected 2.0 everywhere, got {out}"
                )
            outputs = [float(o.mean().item()) for o in out_list]
    except Exception as exc:
        error = f"rank {rank}: {type(exc).__name__}: {exc}"

    result_queue.put({"rank": rank, "error": error, "outputs": outputs})
    dist.destroy_process_group()


@pytest.mark.skipif(sys.platform == "darwin", reason="mp.spawn fork-safety on macOS CI")
def test_pp_multiprocess_2stage_activation_flow():
    """2-rank PP: verify activations flow from stage 0 → stage 1 correctly."""
    H, m = 8, 4
    port = _free_port()
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [ctx.Process(target=_pp2_worker, args=(r, 2, port, q, H, m)) for r in range(2)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=60)
        assert p.exitcode == 0, f"Worker {procs.index(p)} exited {p.exitcode}"

    results = {}
    while not q.empty():
        r = q.get_nowait()
        results[r["rank"]] = r

    assert len(results) == 2
    for rank, res in results.items():
        assert res["error"] is None, f"rank {rank} error: {res['error']}"

    # Last stage should have m outputs, each ≈ 2.0 (scale-by-2 on ones input)
    last = results[1]
    assert len(last["outputs"]) == m
    for v in last["outputs"]:
        assert abs(v - 2.0) < 1e-4, f"Expected 2.0, got {v}"


@pytest.mark.skipif(sys.platform == "darwin", reason="mp.spawn fork-safety on macOS CI")
def test_pp_multiprocess_correct_micro_batch_count():
    """2-rank PP: last stage must produce exactly m outputs."""
    H, m = 4, 6
    port = _free_port()
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [ctx.Process(target=_pp2_worker, args=(r, 2, port, q, H, m)) for r in range(2)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=60)
        assert p.exitcode == 0

    results = {}
    while not q.empty():
        r = q.get_nowait()
        results[r["rank"]] = r

    assert len(results[1]["outputs"]) == m


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
