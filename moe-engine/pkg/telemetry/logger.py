"""
pkg/telemetry/logger.py
=======================

Structured per-step telemetry.  Emits both:

  * **Newline-delimited JSON** to ``json_path`` for Prometheus / ELK / Loki.
  * **TensorBoard scalars** for human inspection.
  * **Prometheus exposition format** (optional) via an in-process HTTP server
    on ``/metrics`` at a configurable port.

v0.2 additions
--------------
* PrometheusExporter: optional in-process metrics endpoint.
* Richer routing-quality fields: ``expert_load_imbalance``, ``router_z_loss``.
* ``collective`` block now includes both dispatch and combine latencies.
* ``memory`` uses real ``torch.cuda.memory_stats()`` when available.
* Thread-safe emit() via a reentrant lock (replaces comment-only guarantee).
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Optional

import torch

log = logging.getLogger("moe_engine.telemetry")

try:
    from torch.utils.tensorboard import SummaryWriter
    _HAS_TB = True
except Exception:
    SummaryWriter = None    # type: ignore
    _HAS_TB = False

# Optional prometheus_client — gracefully degrade when not installed.
try:
    from prometheus_client import (  # type: ignore
        CollectorRegistry, Gauge, Counter, start_http_server,
    )
    _HAS_PROM = True
except Exception:
    _HAS_PROM = False


@dataclass
class StepRecord:
    step: int
    loss: float
    mfu: float
    tokens_per_sec: float
    kernel: dict = field(default_factory=dict)
    collective: dict = field(default_factory=dict)
    memory: dict = field(default_factory=dict)
    infra: dict = field(default_factory=dict)
    routing: dict = field(default_factory=dict)   # v0.2: routing quality
    wall_clock_ms: float = 0.0


# ==========================================================================
# Prometheus exporter (optional — only active when prometheus_client is
# installed and a non-zero port is given).
# ==========================================================================
class PrometheusExporter:
    """In-process Prometheus metrics endpoint.

    Exposes the following gauges on ``/metrics``:

        moe_step_loss
        moe_mfu
        moe_tokens_per_sec
        moe_all_to_all_dispatch_ms
        moe_all_to_all_combine_ms
        moe_peak_memory_gb
        moe_expert_load_imbalance
        moe_router_z_loss
    """

    def __init__(self, port: int = 9102, rank: int = 0):
        if not _HAS_PROM or rank != 0:
            self._enabled = False
            return
        self._enabled = True
        self._reg = CollectorRegistry()
        self._g: Dict[str, Gauge] = {}
        for name, doc in [
            ("moe_step_loss",            "Training loss per step"),
            ("moe_mfu",                  "Model FLOPs Utilization"),
            ("moe_tokens_per_sec",       "Training throughput (tokens/sec)"),
            ("moe_all_to_all_dispatch_ms", "EP dispatch all-to-all latency (ms)"),
            ("moe_all_to_all_combine_ms",  "EP combine all-to-all latency (ms)"),
            ("moe_peak_memory_gb",       "Peak CUDA memory allocated (GB)"),
            ("moe_expert_load_imbalance","Router load imbalance (max/mean)"),
            ("moe_router_z_loss",        "Router auxiliary z-loss"),
        ]:
            self._g[name] = Gauge(name, doc, registry=self._reg)
        try:
            start_http_server(port, registry=self._reg)
            log.info("Prometheus metrics listening on :%d/metrics", port)
        except Exception as exc:
            log.warning("Could not start Prometheus server: %s", exc)
            self._enabled = False

    def update(self, rec: StepRecord) -> None:
        if not self._enabled:
            return
        self._g["moe_step_loss"].set(rec.loss)
        self._g["moe_mfu"].set(rec.mfu)
        self._g["moe_tokens_per_sec"].set(rec.tokens_per_sec)
        self._g["moe_all_to_all_dispatch_ms"].set(
            rec.collective.get("all_to_all_dispatch_ms", 0.0))
        self._g["moe_all_to_all_combine_ms"].set(
            rec.collective.get("all_to_all_combine_ms", 0.0))
        if rec.memory:
            self._g["moe_peak_memory_gb"].set(
                rec.memory.get("peak_allocated_gb", 0.0))
        if rec.routing:
            self._g["moe_expert_load_imbalance"].set(
                rec.routing.get("expert_load_imbalance", 1.0))
            self._g["moe_router_z_loss"].set(
                rec.routing.get("router_z_loss", 0.0))


# ==========================================================================
# Main structured logger
# ==========================================================================
class StructuredLogger:
    """Sink for structured per-step telemetry.

    Thread-safe via a reentrant lock protecting the file handle and
    TensorBoard SummaryWriter.

    Parameters
    ----------
    json_path : str
        Path for newline-delimited JSON output.
    tensorboard_dir : Optional[str]
        Directory for TensorBoard events (only written by rank 0).
    rank : int
        Current process rank.  Non-zero ranks skip TensorBoard.
    also_stdout : bool
        Mirror rank-0 records to stdout (default True).
    prometheus_port : int
        Port for the Prometheus metrics endpoint (0 = disabled).
    """

    def __init__(
        self,
        json_path: str,
        tensorboard_dir: Optional[str] = None,
        rank: int = 0,
        also_stdout: bool = True,
        prometheus_port: int = 0,
    ):
        self.rank = rank
        self.also_stdout = also_stdout
        self.json_path = Path(json_path)
        self.json_path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.json_path.open("a", buffering=1)
        self._lock = threading.RLock()
        self._tb: Optional[SummaryWriter] = None

        if _HAS_TB and tensorboard_dir and rank == 0:
            Path(tensorboard_dir).mkdir(parents=True, exist_ok=True)
            self._tb = SummaryWriter(log_dir=tensorboard_dir)

        self._peak_mem_prev: float = 0.0
        self._prom = PrometheusExporter(port=prometheus_port, rank=rank) \
            if prometheus_port > 0 else None

    # ------------------------------------------------------------------
    def emit(self, record: StepRecord) -> None:
        """Emit one step record to all configured sinks."""
        # Auto-fill memory section from CUDA stats when not provided.
        if not record.memory and torch.cuda.is_available():
            stats = torch.cuda.memory_stats()
            alloc_gb = stats.get("allocated_bytes.all.peak", 0) / (1024 ** 3)
            reserv_gb = stats.get("reserved_bytes.all.current", 0) / (1024 ** 3)
            record.memory = {
                "peak_allocated_gb": round(alloc_gb, 4),
                "reserved_gb": round(reserv_gb, 4),
                "leak_delta_gb": round(alloc_gb - self._peak_mem_prev, 4),
            }
            self._peak_mem_prev = alloc_gb

        payload = asdict(record)
        payload["rank"] = self.rank
        payload["ts"] = time.time()
        line = json.dumps(payload, separators=(",", ":"))

        with self._lock:
            self._fh.write(line + "\n")
            if self.also_stdout and self.rank == 0:
                sys.stdout.write(line + "\n")
                sys.stdout.flush()

            if self._tb is not None:
                self._tb.add_scalar("loss", record.loss, record.step)
                self._tb.add_scalar("mfu", record.mfu, record.step)
                self._tb.add_scalar("tokens_per_sec", record.tokens_per_sec, record.step)
                for k, v in record.collective.items():
                    if isinstance(v, (int, float)):
                        self._tb.add_scalar(f"collective/{k}", v, record.step)
                for k, v in record.memory.items():
                    if isinstance(v, (int, float)):
                        self._tb.add_scalar(f"memory/{k}", v, record.step)
                for k, v in record.kernel.items():
                    if isinstance(v, (int, float)):
                        self._tb.add_scalar(f"kernel/{k}", v, record.step)
                for k, v in record.infra.items():
                    if isinstance(v, (int, float)):
                        self._tb.add_scalar(f"infra/{k}", v, record.step)
                for k, v in record.routing.items():
                    if isinstance(v, (int, float)):
                        self._tb.add_scalar(f"routing/{k}", v, record.step)

        if self._prom is not None:
            self._prom.update(record)

    def close(self) -> None:
        with self._lock:
            try:
                self._fh.close()
            except Exception:
                pass
            if self._tb is not None:
                self._tb.close()
