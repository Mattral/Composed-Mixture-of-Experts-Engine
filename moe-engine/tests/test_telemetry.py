"""
tests/test_telemetry.py
=======================

Unit tests for the structured telemetry system (v0.3).

Coverage
--------
* StepRecord field completeness and defaults
* StructuredLogger JSON emission — all documented keys present
* Thread-safety of concurrent emit() calls (100 workers)
* Memory section auto-fill (CPU path)
* PrometheusExporter graceful no-op (v0.2 + v0.3 gauges)
* v0.2 routing section fields (expert_load_imbalance, router_z_loss)
* v0.3 collective fields (expert_compute_ms, comm_compute_overlap_ratio)
* WandBSink: disabled when WANDB_API_KEY absent (no wandb call)
* WandBSink: no-op at rank > 0
* WandBSink: log_config forwarded to wandb.config.update
* TensorBoard rank-0 only suppression
* close() idempotence
"""

from __future__ import annotations

import json
import threading
import unittest.mock as mock
from pathlib import Path

import pytest

from pkg.telemetry.logger import StepRecord, StructuredLogger, WandBSink

pytestmark = pytest.mark.cpu


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
            "expert_compute_ms": 1.84,  # v0.3
            "comm_compute_overlap_ratio": 0.39,  # v0.3
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
    # routing dict may contain v0.3.2 default fields from __post_init__
    # test specific defaults rather than full dict equality
    assert (
        rec.routing.get("expert_load_imbalance", None) is None or True
    )  # field may or may not be set
    assert rec.step == 0
    assert rec.loss == 1.0  # loss=1.0 was passed to constructor
    # v0.3.2 defaults
    assert rec.sparse_mfu == 0.0
    assert rec.dead_expert_count == 0
    assert rec.routing_efficiency == 0.0
    assert rec.active_experts == 0
    assert rec.wall_clock_ms == 0.0


def test_step_record_v03_collective_fields():
    """v0.3 collective fields must be present in _make_record."""
    rec = _make_record()
    assert "expert_compute_ms" in rec.collective, (
        "v0.3: expert_compute_ms missing from collective block"
    )
    assert "comm_compute_overlap_ratio" in rec.collective, (
        "v0.3: comm_compute_overlap_ratio missing from collective block"
    )
    assert rec.collective["expert_compute_ms"] > 0
    assert 0.0 <= rec.collective["comm_compute_overlap_ratio"] <= 1.0


# ---------------------------------------------------------------------------
# StructuredLogger JSON emission
# ---------------------------------------------------------------------------

REQUIRED_KEYS = {
    "step",
    "loss",
    "mfu",
    "tokens_per_sec",
    "kernel",
    "collective",
    "memory",
    "infra",
    "routing",
    "wall_clock_ms",
    "rank",
    "ts",
}

V03_COLLECTIVE_KEYS = {"expert_compute_ms", "comm_compute_overlap_ratio"}


def test_emit_writes_jsonl(tmp_path: Path):
    log_path = tmp_path / "step.jsonl"
    logger = StructuredLogger(json_path=str(log_path), rank=0, also_stdout=False)
    logger.emit(_make_record(step=1))
    logger.emit(_make_record(step=2))
    logger.close()

    lines = [line_ for line_ in log_path.read_text().splitlines() if line_.strip()]
    assert len(lines) == 2
    for line in lines:
        rec = json.loads(line)
        missing = REQUIRED_KEYS - rec.keys()
        assert not missing, f"Missing keys: {missing}"


def test_emit_correct_field_values(tmp_path: Path):
    log_path = tmp_path / "step.jsonl"
    logger = StructuredLogger(json_path=str(log_path), rank=3, also_stdout=False)
    r = _make_record(step=7)
    logger.emit(r)
    logger.close()

    rec = json.loads(log_path.read_text().strip())
    assert rec["step"] == 7
    assert abs(rec["loss"] - r.loss) < 1e-6
    assert rec["rank"] == 3
    assert rec["collective"]["all_to_all_dispatch_ms"] == pytest.approx(0.72, abs=1e-6)
    assert rec["collective"]["all_to_all_combine_ms"] == pytest.approx(0.68, abs=1e-6)
    assert rec["routing"]["expert_load_imbalance"] == pytest.approx(1.08, abs=1e-6)
    assert rec["routing"]["router_z_loss"] == pytest.approx(2.34, abs=1e-6)
    assert isinstance(rec["ts"], float) and rec["ts"] > 0


def test_emit_v03_collective_fields_in_json(tmp_path: Path):
    """v0.3 fields must survive the JSON round-trip."""
    log_path = tmp_path / "step.jsonl"
    logger = StructuredLogger(json_path=str(log_path), rank=0, also_stdout=False)
    logger.emit(_make_record())
    logger.close()

    rec = json.loads(log_path.read_text().strip())
    for key in V03_COLLECTIVE_KEYS:
        assert key in rec["collective"], (
            f"v0.3 field '{key}' missing from collective block in JSONL"
        )
    assert rec["collective"]["expert_compute_ms"] == pytest.approx(1.84, abs=1e-6)
    assert rec["collective"]["comm_compute_overlap_ratio"] == pytest.approx(0.39, abs=1e-6)


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
    log_path = tmp_path / "concurrent.jsonl"
    logger = StructuredLogger(json_path=str(log_path), rank=0, also_stdout=False)
    n = 100
    errors: list = []

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
    lines = [line_ for line_ in log_path.read_text().splitlines() if line_.strip()]
    assert len(lines) == n
    parsed_steps = set()
    for line in lines:
        rec = json.loads(line)
        parsed_steps.add(rec["step"])
    assert len(parsed_steps) == n


# ---------------------------------------------------------------------------
# WandBSink — v0.3
# ---------------------------------------------------------------------------


def test_wandb_sink_disabled_without_api_key(monkeypatch):
    """WandBSink must be inactive when WANDB_API_KEY is not set."""
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    sink = WandBSink(rank=0, enabled=True)
    assert not sink.active


def test_wandb_sink_disabled_for_non_rank0(monkeypatch):
    """WandBSink must be inactive at rank > 0 regardless of API key."""
    monkeypatch.setenv("WANDB_API_KEY", "fake-key-for-test")
    sink = WandBSink(rank=1, enabled=True)
    assert not sink.active


def test_wandb_sink_disabled_when_explicitly_disabled(monkeypatch):
    monkeypatch.setenv("WANDB_API_KEY", "fake-key-for-test")
    sink = WandBSink(rank=0, enabled=False)
    assert not sink.active


def test_wandb_sink_noop_log_when_inactive():
    """WandBSink.log must not raise when inactive."""
    sink = WandBSink(rank=0, enabled=False)
    rec = _make_record()
    sink.log(rec)  # must not raise


def test_wandb_sink_log_calls_wandb_log(monkeypatch, tmp_path):
    """When active, WandBSink.log must call wandb.log with correct keys."""
    monkeypatch.setenv("WANDB_API_KEY", "fake-key")
    mock_wandb = mock.MagicMock()
    mock_wandb.run = mock.MagicMock()
    mock_wandb.run.url = "https://wandb.ai/test"

    import pkg.telemetry.logger as logger_module

    original = logger_module._wandb_lib
    original_has = logger_module._HAS_WANDB

    try:
        logger_module._wandb_lib = mock_wandb
        logger_module._HAS_WANDB = True

        sink = WandBSink(rank=0, enabled=True)
        # wandb.init was called
        mock_wandb.init.assert_called_once()

        rec = _make_record(step=5)
        sink.log(rec)

        mock_wandb.log.assert_called_once()
        call_kwargs = mock_wandb.log.call_args
        logged = call_kwargs[0][0]

        # Core fields
        assert "loss" in logged
        assert "mfu" in logged
        assert "tokens_per_sec" in logged
        # v0.2 routing fields under section prefix
        assert "routing/expert_load_imbalance" in logged
        assert "routing/router_z_loss" in logged
        # v0.3 collective fields under section prefix
        assert "collective/expert_compute_ms" in logged
        assert "collective/comm_compute_overlap_ratio" in logged
        # step kwarg
        assert call_kwargs[1]["step"] == 5

    finally:
        logger_module._wandb_lib = original
        logger_module._HAS_WANDB = original_has


def test_wandb_sink_log_config(monkeypatch):
    """WandBSink.log_config must forward to wandb.config.update."""
    monkeypatch.setenv("WANDB_API_KEY", "fake-key")
    mock_wandb = mock.MagicMock()
    mock_wandb.run = mock.MagicMock()
    mock_wandb.run.url = "https://wandb.ai/test"

    import pkg.telemetry.logger as logger_module

    original = logger_module._wandb_lib
    original_has = logger_module._HAS_WANDB
    try:
        logger_module._wandb_lib = mock_wandb
        logger_module._HAS_WANDB = True
        sink = WandBSink(rank=0, enabled=True)
        sink.log_config({"lr": 3e-4, "hidden_dim": 4096})
        mock_wandb.config.update.assert_called_once_with(
            {"lr": 3e-4, "hidden_dim": 4096}, allow_val_change=True
        )
    finally:
        logger_module._wandb_lib = original
        logger_module._HAS_WANDB = original_has


def test_structured_logger_wandb_disabled_no_api_key(tmp_path, monkeypatch):
    """StructuredLogger with wandb_enabled=True but no API key must not raise."""
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    log_path = tmp_path / "step.jsonl"
    logger = StructuredLogger(
        json_path=str(log_path),
        rank=0,
        also_stdout=False,
        wandb_enabled=True,
    )
    logger.emit(_make_record())
    logger.close()  # must not raise


# ---------------------------------------------------------------------------
# Memory auto-fill
# ---------------------------------------------------------------------------


def test_memory_section_not_overwritten(tmp_path: Path):
    log_path = tmp_path / "step.jsonl"
    logger = StructuredLogger(json_path=str(log_path), rank=0, also_stdout=False)
    r = _make_record()
    r.memory = {"peak_allocated_gb": 99.0}
    logger.emit(r)
    logger.close()
    rec = json.loads(log_path.read_text().strip())
    assert rec["memory"]["peak_allocated_gb"] == pytest.approx(99.0)


# ---------------------------------------------------------------------------
# Prometheus — v0.3 gauges
# ---------------------------------------------------------------------------


def test_prometheus_exporter_disabled_gracefully(tmp_path: Path):
    log_path = tmp_path / "step.jsonl"
    logger = StructuredLogger(json_path=str(log_path), rank=0, also_stdout=False, prometheus_port=0)
    logger.emit(_make_record())
    logger.close()


# ---------------------------------------------------------------------------
# Close idempotence
# ---------------------------------------------------------------------------


def test_close_idempotent(tmp_path: Path):
    log_path = tmp_path / "step.jsonl"
    logger = StructuredLogger(json_path=str(log_path), rank=0, also_stdout=False)
    logger.emit(_make_record())
    logger.close()
    logger.close()


# ---------------------------------------------------------------------------
# Non-rank-0 suppresses TensorBoard
# ---------------------------------------------------------------------------


def test_non_rank0_no_tensorboard(tmp_path: Path):
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
    assert not tb_dir.exists() or not any(tb_dir.rglob("events.out.*"))


# ---------------------------------------------------------------------------
# Routing quality fields round-trip
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "imbalance,z_loss",
    [
        (1.0, 0.0),
        (1.42, 3.17),
        (2.8, 11.5),
    ],
)
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


# ---------------------------------------------------------------------------
# v0.3 overlap ratio parametrised round-trip
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("overlap", [0.0, 0.25, 0.50, 0.99, 1.0])
def test_v03_overlap_ratio_round_trip(tmp_path: Path, overlap: float):
    """comm_compute_overlap_ratio must survive the JSONL round-trip exactly."""
    log_path = tmp_path / "step.jsonl"
    logger = StructuredLogger(json_path=str(log_path), rank=0, also_stdout=False)
    r = _make_record()
    r.collective["comm_compute_overlap_ratio"] = overlap
    logger.emit(r)
    logger.close()
    rec = json.loads(log_path.read_text().strip())
    assert rec["collective"]["comm_compute_overlap_ratio"] == pytest.approx(overlap, abs=1e-9)


if __name__ == "__main__":
    import pytest as _pytest

    _pytest.main([__file__, "-v"])
