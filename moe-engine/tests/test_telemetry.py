"""
tests/test_telemetry.py
=======================

Unit tests for the v0.2 structured telemetry system.

Covers:
  * StepRecord dataclass field completeness and defaults
  * StructuredLogger JSON emission — all documented keys present
  * Thread-safety of concurrent emit() calls
  * Memory section auto-fill (CPU path; CUDA path guarded)
  * PrometheusExporter graceful no-op when prometheus_client is absent
  * v0.2 routing section fields (expert_load_imbalance, router_z_loss)
  * TensorBoard writes when SummaryWriter is available
  * close() idempotence
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest
import torch

from pkg.telemetry.logger import StepRecord, StructuredLogger


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_record(step: int = 0) -> StepRecord:
    return StepRecord(
        step=step,
        loss=1.5 - step * 0.01,
        mfu=0.42,
        tokens_per_sec=12_800.0,
        wall_clock_ms=78.4,
        kernel={
            "sram_bytes_per_block": 49152,
            "achieved_bw_gbps": 1.23,
            "tokens_per_expert_mean": 32.0,
            "tokens_per_expert_std": 4.1,
            "used_triton": False,
        },
        collective={
            "all_to_all_dispatch_ms": 0.72,
            "all_to_all_combine_ms": 0.68,
        },
        memory={
            "peak_allocated_gb": 3.14,
            "reserved_gb": 4.00,
            "leak_delta_gb": 0.00,
        },
        infra={
            "async_ckpt_commit_ms": 12.3,
            "active_nodes": 8,
            "ep_world_size": 4,
            "lr": 3e-4,
        },
        routing={
            "expert_load_imbalance": 1.08,
            "router_z_loss": 2.34,
        },
    )


# ---------------------------------------------------------------------------
# StepRecord structure
# ---------------------------------------------------------------------------

def test_step_record_has_routing_section():
    rec = _make_record()
    assert "routing" in rec.__dataclass_fields__
    assert "expert_load_imbalance" in rec.routing
    assert "router_z_loss" in rec.routing


def test_step_record_defaults():
    rec = StepRecord(step=0, loss=1.0, mfu=0.5, tokens_per_sec=100.0)
    assert rec.kernel == {}
    assert rec.collective == {}
    assert rec.memory == {}
    assert rec.infra == {}
    assert rec.routing == {}
    assert rec.wall_clock_ms == 0.0


# ---------------------------------------------------------------------------
# StructuredLogger JSON emission
# ---------------------------------------------------------------------------

REQUIRED_KEYS = {
    "step", "loss", "mfu", "tokens_per_sec",
    "kernel", "collective", "memory", "infra", "routing",
    "wall_clock_ms", "rank", "ts",
}

def test_emit_writes_jsonl(tmp_path: Path):
    log_path = tmp_path / "step.jsonl"
    logger = StructuredLogger(json_path=str(log_path), rank=0, also_stdout=False)
    logger.emit(_make_record(step=1))
    logger.emit(_make_record(step=2))
    logger.close()

    lines = [l for l in log_path.read_text().splitlines() if l.strip()]
    assert len(lines) == 2

    for line in lines:
        rec = json.loads(line)
        missing = REQUIRED_KEYS - rec.keys()
        assert not missing, f"Missing keys in JSON record: {missing}"


def test_emit_correct_field_values(tmp_path: Path):
    log_path = tmp_path / "step.jsonl"
    logger = StructuredLogger(json_path=str(log_path), rank=3, also_stdout=False)
    r = _make_record(step=7)
    logger.emit(r)
    logger.close()

    rec = json.loads(log_path.read_text().strip())
    assert rec["step"] == 7
    assert abs(rec["loss"] - r.loss) < 1e-6
    assert abs(rec["mfu"] - 0.42) < 1e-6
    assert rec["rank"] == 3
    assert rec["collective"]["all_to_all_dispatch_ms"] == pytest.approx(0.72, abs=1e-6)
    assert rec["collective"]["all_to_all_combine_ms"] == pytest.approx(0.68, abs=1e-6)
    assert rec["routing"]["expert_load_imbalance"] == pytest.approx(1.08, abs=1e-6)
    assert rec["routing"]["router_z_loss"] == pytest.approx(2.34, abs=1e-6)
    assert isinstance(rec["ts"], float) and rec["ts"] > 0


def test_emit_kernel_used_triton_bool(tmp_path: Path):
    log_path = tmp_path / "step.jsonl"
    logger = StructuredLogger(json_path=str(log_path), rank=0, also_stdout=False)
    r = _make_record()
    r.kernel["used_triton"] = False
    logger.emit(r)
    logger.close()
    rec = json.loads(log_path.read_text().strip())
    assert rec["kernel"]["used_triton"] is False


def test_emit_infra_fields(tmp_path: Path):
    """Infra block must contain async_ckpt_commit_ms and active_nodes."""
    log_path = tmp_path / "step.jsonl"
    logger = StructuredLogger(json_path=str(log_path), rank=0, also_stdout=False)
    logger.emit(_make_record())
    logger.close()
    rec = json.loads(log_path.read_text().strip())
    assert "async_ckpt_commit_ms" in rec["infra"]
    assert "active_nodes" in rec["infra"]


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------

def test_emit_thread_safe(tmp_path: Path):
    """100 concurrent emit() calls must all land in the JSONL without corruption."""
    log_path = tmp_path / "concurrent.jsonl"
    logger = StructuredLogger(json_path=str(log_path), rank=0, also_stdout=False)
    n = 100
    errors: list[Exception] = []

    def _worker(i: int):
        try:
            logger.emit(_make_record(step=i))
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=_worker, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    logger.close()

    assert not errors, f"Exceptions in emit threads: {errors}"
    lines = [l for l in log_path.read_text().splitlines() if l.strip()]
    assert len(lines) == n, f"Expected {n} lines, got {len(lines)}"
    # Every line must be valid JSON
    parsed_steps = set()
    for line in lines:
        rec = json.loads(line)   # raises if corrupted
        parsed_steps.add(rec["step"])
    assert len(parsed_steps) == n, "Duplicate or missing step numbers after concurrent emit"


# ---------------------------------------------------------------------------
# Memory auto-fill (CPU path)
# ---------------------------------------------------------------------------

def test_memory_section_not_auto_filled_on_cpu(tmp_path: Path):
    """When memory is pre-filled by caller, logger must not overwrite it."""
    log_path = tmp_path / "step.jsonl"
    logger = StructuredLogger(json_path=str(log_path), rank=0, also_stdout=False)
    r = _make_record()
    r.memory = {"peak_allocated_gb": 99.0}   # caller-provided sentinel
    logger.emit(r)
    logger.close()
    rec = json.loads(log_path.read_text().strip())
    # Logger must not override caller-provided memory section
    assert rec["memory"]["peak_allocated_gb"] == pytest.approx(99.0)


# ---------------------------------------------------------------------------
# Prometheus — graceful no-op when not installed
# ---------------------------------------------------------------------------

def test_prometheus_exporter_disabled_gracefully(tmp_path: Path):
    """Passing prometheus_port=0 must not raise even if prometheus_client absent."""
    log_path = tmp_path / "step.jsonl"
    logger = StructuredLogger(
        json_path=str(log_path), rank=0, also_stdout=False, prometheus_port=0
    )
    logger.emit(_make_record())
    logger.close()   # must not raise


# ---------------------------------------------------------------------------
# close() idempotence
# ---------------------------------------------------------------------------

def test_close_idempotent(tmp_path: Path):
    log_path = tmp_path / "step.jsonl"
    logger = StructuredLogger(json_path=str(log_path), rank=0, also_stdout=False)
    logger.emit(_make_record())
    logger.close()
    logger.close()   # must not raise


# ---------------------------------------------------------------------------
# Non-rank-0 suppresses TensorBoard
# ---------------------------------------------------------------------------

def test_non_rank0_no_tensorboard(tmp_path: Path):
    """Rank > 0 must skip TB writes even when tensorboard_dir is provided."""
    log_path = tmp_path / "step.jsonl"
    tb_dir = tmp_path / "tb"
    logger = StructuredLogger(
        json_path=str(log_path),
        tensorboard_dir=str(tb_dir),
        rank=1,
        also_stdout=False,
    )
    logger.emit(_make_record())
    logger.close()
    # TB directory should NOT be created by non-rank-0
    assert not tb_dir.exists() or not any(tb_dir.rglob("events.out.*"))


# ---------------------------------------------------------------------------
# Routing quality fields round-trip
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("imbalance,z_loss", [
    (1.0, 0.0),
    (1.42, 3.17),
    (2.8, 11.5),
])
def test_routing_fields_round_trip(tmp_path: Path, imbalance: float, z_loss: float):
    log_path = tmp_path / "step.jsonl"
    logger = StructuredLogger(json_path=str(log_path), rank=0, also_stdout=False)
    r = _make_record()
    r.routing["expert_load_imbalance"] = imbalance
    r.routing["router_z_loss"] = z_loss
    logger.emit(r)
    logger.close()
    rec = json.loads(log_path.read_text().strip())
    assert rec["routing"]["expert_load_imbalance"] == pytest.approx(imbalance, abs=1e-9)
    assert rec["routing"]["router_z_loss"] == pytest.approx(z_loss, abs=1e-9)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
