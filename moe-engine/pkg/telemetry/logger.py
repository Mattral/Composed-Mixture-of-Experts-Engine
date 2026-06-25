"""
pkg/telemetry/logger.py
=======================

Structured per-step telemetry.  Emits both:

  * **Newline-delimited JSON** to ``json_path`` for Prometheus / ELK / Loki.
  * **TensorBoard scalars** for human inspection.
  * **Prometheus exposition format** (optional) via an in-process HTTP server
    on ``/metrics`` at a configurable port.
  * **Weights & Biases** (optional) via ``wandb.log()`` — active when
    ``WANDB_API_KEY`` is set and ``wandb`` is installed.

v0.2 additions
--------------
* PrometheusExporter: optional in-process metrics endpoint.
* Richer routing-quality fields: ``expert_load_imbalance``, ``router_z_loss``.
* ``collective`` block now includes both dispatch and combine latencies.
* ``memory`` uses real ``torch.cuda.memory_stats()`` when available.
* Thread-safe emit() via a reentrant lock (replaces comment-only guarantee).

v0.3 additions
--------------
* WandB sink: ``WandBSink`` wraps ``wandb.log()`` under a single interface.
  Activated when ``WANDB_API_KEY`` is set in the environment and
  ``wandb`` is installed; otherwise degrades gracefully to a no-op.
* ``collective`` block adds ``expert_compute_ms`` and
  ``comm_compute_overlap_ratio`` (dispatch_ms / expert_compute_ms).
* ``StepRecord`` version marker bumped to reflect v0.3 field additions.
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
from typing import Any, Dict, Optional

import torch

log = logging.getLogger("moe_engine.telemetry")

try:
    from torch.utils.tensorboard import SummaryWriter

    _HAS_TB = True
except Exception:
    SummaryWriter = None  # type: ignore
    _HAS_TB = False

try:
    from prometheus_client import (  # type: ignore
        CollectorRegistry,
        Gauge,
        start_http_server,
    )

    _HAS_PROM = True
except Exception:
    _HAS_PROM = False

try:
    import wandb as _wandb_lib  # type: ignore

    _HAS_WANDB = True
except Exception:
    _wandb_lib = None  # type: ignore
    _HAS_WANDB = False


# ==========================================================================
# StepRecord — the canonical per-step telemetry envelope (v0.3)
# ==========================================================================
@dataclass
class StepRecord:
    """Per-step telemetry record — emitted to JSONL, TensorBoard, Prometheus, WandB.

    Version history
    ---------------
    v0.1 : step, loss, mfu, tokens_per_sec, kernel, collective, memory, infra
    v0.2 : routing block (expert_load_imbalance, router_z_loss)
    v0.3 : collective.expert_compute_ms, collective.comm_compute_overlap_ratio,
           WandBSink, Prometheus gauges
    v0.3.2 : sparse_mfu, dead_expert_count, routing_efficiency (MoE-specific)

    On-wire format (JSONL)
    ----------------------
    All dict fields are flattened into the JSON record. String keys are
    prefixed by section name in WandB (e.g. ``routing/z_loss``).

    MoE-specific fields (v0.3.2)
    ----------------------------
    sparse_mfu : float
        MFU accounting for the fact that only K/E experts fire per token.
        ``sparse_mfu = mfu * (K / E)`` where K=top_k, E=num_experts.
        This is the correct MFU denominator for sparse models.
        A dense MoE with E=64, K=2 has sparse_mfu ≈ 0.031 × mfu.

    dead_expert_count : int
        Number of experts that received zero tokens this step.
        Non-zero values indicate early-stage routing collapse.
        Alert threshold: > 0 for more than 5 consecutive steps.

    routing_efficiency : float
        Fraction of expert capacity actually used this step.
        ``routing_efficiency = actual_tokens / (capacity_budget * E)``.
        1.0 = capacity perfectly matched to load. < 0.7 = over-provisioned.
        > 1.0 = tokens dropped (capacity overflow).

    active_experts : int
        Number of distinct experts that received at least one token this step.
        Useful for detecting routing collapse (active_experts → small number).
    """

    step: int
    loss: float
    mfu: float
    tokens_per_sec: float
    kernel: dict = field(default_factory=dict)
    collective: dict = field(default_factory=dict)
    memory: dict = field(default_factory=dict)
    infra: dict = field(default_factory=dict)
    routing: dict = field(default_factory=dict)  # v0.2: load_imbalance, z_loss
    wall_clock_ms: float = 0.0

    # ------------------------------------------------------------------
    # v0.3.2 MoE-specific telemetry fields
    # These live as first-class dataclass fields (typed access) AND
    # are injected into the ``routing`` dict for backward-compat JSON.
    # ------------------------------------------------------------------
    sparse_mfu: float = 0.0  # mfu * (K/E) — correct for sparse activation
    dead_expert_count: int = 0  # experts receiving zero tokens this step
    routing_efficiency: float = 0.0  # actual_tokens / (capacity_budget * E)
    active_experts: int = 0  # distinct experts receiving ≥ 1 token

    def __post_init__(self) -> None:
        """Inject v0.3.2 fields into routing dict for backward-compat JSON."""
        self.routing.setdefault("sparse_mfu", self.sparse_mfu)
        self.routing.setdefault("dead_expert_count", self.dead_expert_count)
        self.routing.setdefault("routing_efficiency", self.routing_efficiency)
        self.routing.setdefault("active_experts", self.active_experts)


# ==========================================================================
# WandB sink (v0.3)
# ==========================================================================
class WandBSink:
    """Thin wrapper around ``wandb.log()`` for moe-engine telemetry.

    Activation
    ----------
    WandBSink is active when all of the following are true:
      1. ``wandb`` package is installed.
      2. ``WANDB_API_KEY`` environment variable is set.
      3. ``rank == 0`` (only the primary process logs to WandB).
      4. The sink was not explicitly disabled (``enabled=False``).

    When inactive, every method is a no-op so callers never need to check.

    Configuration
    -------------
    ``init_kwargs`` are forwarded to ``wandb.init()``.  Typical keys:
      * ``project``  — WandB project name (default: ``"moe-engine"``)
      * ``name``     — run name (default: auto-generated by WandB)
      * ``config``   — dict of hyperparameters to log as run config
      * ``resume``   — ``"allow"`` or ``"must"`` for resuming a run

    All standard moe-engine telemetry fields are logged under their
    section prefix, e.g. ``collective/dispatch_ms``, ``routing/z_loss``.
    """

    def __init__(
        self,
        rank: int = 0,
        enabled: bool = True,
        init_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self._active = False
        if not enabled or rank != 0:
            return
        if not _HAS_WANDB:
            log.debug("wandb not installed; WandBSink disabled")
            return
        if not os.environ.get("WANDB_API_KEY"):
            log.debug("WANDB_API_KEY not set; WandBSink disabled")
            return

        try:
            kwargs: Dict[str, Any] = {"project": "moe-engine"}
            if init_kwargs:
                kwargs.update(init_kwargs)
            _wandb_lib.init(**kwargs)
            self._active = True
            log.info(
                "WandB run initialised: %s", _wandb_lib.run.url if _wandb_lib.run else "unknown"
            )
        except Exception as exc:
            log.warning("WandB init failed (%s); WandBSink disabled", exc)

    @property
    def active(self) -> bool:
        return self._active

    def log(self, record: "StepRecord") -> None:
        """Log all numeric fields from a StepRecord to WandB."""
        if not self._active:
            return
        payload: Dict[str, Any] = {
            "loss": record.loss,
            "mfu": record.mfu,
            "tokens_per_sec": record.tokens_per_sec,
            "wall_clock_ms": record.wall_clock_ms,
        }
        for section, subdict in [
            ("kernel", record.kernel),
            ("collective", record.collective),
            ("memory", record.memory),
            ("infra", record.infra),
            ("routing", record.routing),
        ]:
            for k, v in subdict.items():
                if isinstance(v, (int, float)):
                    payload[f"{section}/{k}"] = v
        try:
            _wandb_lib.log(payload, step=record.step)
        except Exception as exc:
            log.warning("wandb.log failed at step %d: %s", record.step, exc)

    def log_config(self, cfg: Dict[str, Any]) -> None:
        """Update WandB run config with hyperparameters."""
        if not self._active:
            return
        try:
            _wandb_lib.config.update(cfg, allow_val_change=True)
        except Exception as exc:
            log.warning("wandb.config.update failed: %s", exc)

    def finish(self) -> None:
        """Mark the WandB run as finished (called on clean shutdown)."""
        if not self._active:
            return
        try:
            _wandb_lib.finish()
        except Exception:
            pass
        self._active = False


# ==========================================================================
# Prometheus exporter (unchanged from v0.2, extended for v0.3 fields)
# ==========================================================================
class PrometheusExporter:
    """In-process Prometheus metrics endpoint.

    Exposes the following gauges on ``/metrics``:

        moe_step_loss
        moe_mfu
        moe_tokens_per_sec
        moe_all_to_all_dispatch_ms
        moe_all_to_all_combine_ms
        moe_expert_compute_ms          (v0.3)
        moe_comm_compute_overlap_ratio (v0.3)
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
            ("moe_step_loss", "Training loss per step"),
            ("moe_mfu", "Model FLOPs Utilization"),
            ("moe_tokens_per_sec", "Training throughput (tokens/sec)"),
            ("moe_all_to_all_dispatch_ms", "EP dispatch all-to-all latency (ms)"),
            ("moe_all_to_all_combine_ms", "EP combine all-to-all latency (ms)"),
            ("moe_expert_compute_ms", "Expert FFN compute latency (ms) [v0.3]"),
            ("moe_comm_compute_overlap_ratio", "Dispatch latency / expert compute latency [v0.3]"),
            ("moe_peak_memory_gb", "Peak CUDA memory allocated (GB)"),
            ("moe_sparse_mfu", "MFU accounting for K/E sparse activation (v0.3.2)"),
            ("moe_dead_expert_count", "Experts receiving zero tokens this step (v0.3.2)"),
            ("moe_routing_efficiency", "Fraction of expert capacity used (v0.3.2)"),
            ("moe_active_experts", "Distinct experts receiving >= 1 token (v0.3.2)"),
            ("moe_expert_load_imbalance", "Router load imbalance (max/mean)"),
            ("moe_router_z_loss", "Router auxiliary z-loss"),
        ]:
            self._g[name] = Gauge(name, doc, registry=self._reg)
        try:
            start_http_server(port, registry=self._reg)
            log.info("Prometheus metrics listening on :%d/metrics", port)
        except Exception as exc:
            log.warning("Could not start Prometheus server: %s", exc)
            self._enabled = False

    def update(self, rec: "StepRecord") -> None:
        if not self._enabled:
            return
        self._g["moe_step_loss"].set(rec.loss)
        self._g["moe_mfu"].set(rec.mfu)
        self._g["moe_tokens_per_sec"].set(rec.tokens_per_sec)
        self._g["moe_all_to_all_dispatch_ms"].set(rec.collective.get("all_to_all_dispatch_ms", 0.0))
        self._g["moe_all_to_all_combine_ms"].set(rec.collective.get("all_to_all_combine_ms", 0.0))
        self._g["moe_expert_compute_ms"].set(rec.collective.get("expert_compute_ms", 0.0))
        self._g["moe_comm_compute_overlap_ratio"].set(
            rec.collective.get("comm_compute_overlap_ratio", 0.0)
        )
        if rec.memory:
            self._g["moe_peak_memory_gb"].set(rec.memory.get("peak_allocated_gb", 0.0))
        if rec.routing:
            if "sparse_mfu" in rec.routing:
                self._g["moe_sparse_mfu"].set(rec.routing["sparse_mfu"])
            if "dead_expert_count" in rec.routing:
                self._g["moe_dead_expert_count"].set(rec.routing["dead_expert_count"])
            if "routing_efficiency" in rec.routing:
                self._g["moe_routing_efficiency"].set(rec.routing["routing_efficiency"])
            if "active_experts" in rec.routing:
                self._g["moe_active_experts"].set(rec.routing["active_experts"])
        if rec.routing:
            self._g["moe_expert_load_imbalance"].set(rec.routing.get("expert_load_imbalance", 1.0))
            self._g["moe_router_z_loss"].set(rec.routing.get("router_z_loss", 0.0))


# ==========================================================================
# Main structured logger (all sinks unified)
# ==========================================================================
class StructuredLogger:
    """Unified sink for structured per-step telemetry.

    Emits to four sinks simultaneously:
      * JSONL file (always)
      * TensorBoard (rank 0 only, when tensorboard installed)
      * Prometheus /metrics endpoint (optional, rank 0 only)
      * WandB (rank 0 only, when WANDB_API_KEY set and wandb installed) [v0.3]

    All sinks are thread-safe via a reentrant lock on the JSONL file handle
    and TensorBoard writer.  WandB and Prometheus have their own internal
    serialisation.

    Parameters
    ----------
    json_path : str
    tensorboard_dir : Optional[str]
    rank : int
    also_stdout : bool
    prometheus_port : int
        0 = disabled.
    wandb_enabled : bool
        If True and conditions are met, WandB sink is activated.  [v0.3]
    wandb_init_kwargs : Optional[dict]
        Forwarded to ``wandb.init()``.  [v0.3]
    """

    def __init__(
        self,
        json_path: str,
        tensorboard_dir: Optional[str] = None,
        rank: int = 0,
        also_stdout: bool = True,
        prometheus_port: int = 0,
        wandb_enabled: bool = True,
        wandb_init_kwargs: Optional[Dict[str, Any]] = None,
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

        self._prom = (
            PrometheusExporter(port=prometheus_port, rank=rank) if prometheus_port > 0 else None
        )

        # v0.3: WandB sink
        self._wandb = WandBSink(
            rank=rank,
            enabled=wandb_enabled,
            init_kwargs=wandb_init_kwargs,
        )

    # ------------------------------------------------------------------
    def emit(self, record: StepRecord) -> None:
        """Emit one step record to all configured sinks."""
        if not record.memory and torch.cuda.is_available():
            stats = torch.cuda.memory_stats()
            alloc_gb = stats.get("allocated_bytes.all.peak", 0) / (1024**3)
            reserv_gb = stats.get("reserved_bytes.all.current", 0) / (1024**3)
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
                for section, subdict in [
                    ("collective", record.collective),
                    ("memory", record.memory),
                    ("kernel", record.kernel),
                    ("infra", record.infra),
                    ("routing", record.routing),
                ]:
                    for k, v in subdict.items():
                        if isinstance(v, (int, float)):
                            self._tb.add_scalar(f"{section}/{k}", v, record.step)

        # WandB and Prometheus have their own thread-safety
        self._wandb.log(record)
        if self._prom is not None:
            self._prom.update(record)

    def log_config(self, cfg: Dict[str, Any]) -> None:
        """Log hyperparameters to WandB run config (no-op if WandB inactive)."""
        self._wandb.log_config(cfg)

    def close(self) -> None:
        with self._lock:
            try:
                self._fh.close()
            except Exception:
                pass
            if self._tb is not None:
                self._tb.close()
        self._wandb.finish()
