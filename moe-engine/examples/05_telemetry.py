#!/usr/bin/env python3
"""
examples/05_telemetry.py
=========================

Structured logging and step record usage.

Shows:
- Creating and emitting StepRecord instances
- v0.3.2 MoE-specific fields: sparse_mfu, dead_expert_count,
  routing_efficiency, active_experts
- Reading emitted JSONL output
- WandBSink no-op behaviour when WANDB_API_KEY is absent

Run from moe-engine/:
    python examples/05_telemetry.py
"""

import json
import pathlib
import tempfile

from pkg.telemetry.logger import StepRecord, StructuredLogger

print("=" * 60)
print("moe-engine Example 05: Telemetry")
print("=" * 60)
print()

# ── Build a StepRecord ────────────────────────────────────────────
print("1. Constructing a StepRecord (v0.3.2)")
rec = StepRecord(
    step=42,
    loss=4.217,
    mfu=0.431,
    tokens_per_sec=79203.0,
    wall_clock_ms=412.3,
    kernel={
        "sram_bytes_per_block": 49152,
        "achieved_bw_gbps": 312.4,
        "tokens_per_expert_mean": 64.0,
        "tokens_per_expert_std": 8.3,
        "used_triton": False,
    },
    collective={
        "all_to_all_dispatch_ms": 0.38,
        "all_to_all_combine_ms": 0.41,
        "expert_compute_ms": 1.24,
        "comm_compute_overlap_ratio": 0.31,
    },
    routing={
        "expert_load_imbalance": 1.08,
        "router_z_loss": 2.87,
    },
    memory={"peak_allocated_gb": 42.1},
    infra={"lr": 3e-4, "active_nodes": 8},
    # v0.3.2 MoE-specific fields
    sparse_mfu=0.431 * (2 / 64),  # mfu * K/E
    dead_expert_count=3,  # 3 experts idle this step
    routing_efficiency=0.94,
    active_experts=61,
)
print(f"   step                : {rec.step}")
print(f"   loss                : {rec.loss}")
print(f"   sparse_mfu          : {rec.sparse_mfu:.5f}  (mfu * K/E = 0.431 * 2/64)")
print(f"   dead_expert_count   : {rec.dead_expert_count}  (alert if > 0 sustained)")
print(f"   routing_efficiency  : {rec.routing_efficiency:.2f}  (1.0 = perfect fit)")
print(f"   active_experts      : {rec.active_experts}/64")
print()

# Verify backward-compat: fields also in routing dict
assert "sparse_mfu" in rec.routing
assert "dead_expert_count" in rec.routing
print("   ✓ v0.3.2 fields also present in rec.routing dict (backward-compat JSON)")
print()

# ── Emit to JSONL ─────────────────────────────────────────────────
print("2. Emit to JSONL and read back")
with tempfile.TemporaryDirectory() as tmpdir:
    json_path = f"{tmpdir}/step.jsonl"
    logger = StructuredLogger(
        json_path=json_path,
        tensorboard_dir=f"{tmpdir}/tb",
        rank=0,
    )
    for s in range(3):
        r = StepRecord(
            step=s,
            loss=4.5 - s * 0.1,
            mfu=0.01 * (s + 1),
            tokens_per_sec=1000.0 * (s + 1),
            wall_clock_ms=100.0,
            sparse_mfu=0.001 * (s + 1),
            dead_expert_count=max(0, 2 - s),
            routing_efficiency=0.9 + s * 0.02,
            active_experts=60 + s * 2,
        )
        logger.emit(r)
    logger.close()

    lines = pathlib.Path(json_path).read_text().strip().split("\n")
    print(f"   Emitted {len(lines)} records")
    last = json.loads(lines[-1])
    print(f"   Last record step={last['step']}  loss={last['loss']:.2f}")
    print(f"   routing.sparse_mfu         = {last['routing'].get('sparse_mfu', 'MISSING')}")
    print(f"   routing.dead_expert_count  = {last['routing'].get('dead_expert_count', 'MISSING')}")
    print(f"   routing.routing_efficiency = {last['routing'].get('routing_efficiency', 'MISSING')}")
    print()

# ── Interpret the MoE-specific fields ────────────────────────────
print("3. Interpreting MoE-specific telemetry")
print()
print("   sparse_mfu:")
print("     Dense MFU of 43% with E=64, K=2 → sparse_mfu = 43% × 2/64 = 1.3%")
print("     This is the correct MFU for sparse models. Dense MFU overstates by K/E.")
print()
print("   dead_expert_count:")
print("     Experts receiving zero tokens in a step. During training this should")
print("     trend toward 0 as routing stabilises. Sustained non-zero → routing")
print("     collapse. Fix: enable z_loss_weight=1e-3 in config.")
print()
print("   routing_efficiency:")
print("     actual_tokens / (capacity_budget × E). Values:")
print("     0.7–1.0 = healthy   < 0.7 = over-provisioned (waste)")
print("     > 1.0   = tokens dropped (reduce capacity_factor or increase batch size)")
print()
print("Example 05 complete ✓")
