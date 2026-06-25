#!/usr/bin/env python3
"""
examples/06_checkpoint.py
==========================

Async two-tier checkpointing with schema versioning.

Shows:
- Constructing AsyncCheckpointer with local NVMe + remote tiers
- Non-blocking save (training loop not stalled)
- Schema versioning: v0.3.2 enriched meta dict
- Loading checkpoint and reading metadata
- Compatibility checking for older/newer schema versions
- Retention pruning

Run from moe-engine/:
    python examples/06_checkpoint.py
"""

import pathlib
import tempfile

import torch
import torch.nn as nn

from pkg.elastic.fault_monitor import (
    CHECKPOINT_SCHEMA_VERSION,
    AsyncCheckpointer,
    LocalNVMeAdapter,
    _check_schema_compatibility,
)

print("=" * 60)
print("moe-engine Example 06: Async Checkpointing")
print("=" * 60)
print()

# ── Setup in temp directory ──────────────────────────────────────
with tempfile.TemporaryDirectory() as tmpdir:
    local_dir = str(pathlib.Path(tmpdir) / "nvme")
    remote_uri = f"file://{pathlib.Path(tmpdir) / 'remote'}"
    pathlib.Path(local_dir).mkdir()

    local_adapter = LocalNVMeAdapter(local_dir)
    remote_adapter = LocalNVMeAdapter(pathlib.Path(tmpdir) / "remote")
    ckpt = AsyncCheckpointer(
        local_adapter=local_adapter,
        remote_adapter=remote_adapter,
        retention=3,
        workers=2,
    )

    # ── Simple model for testing ──────────────────────────────────
    model = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 32))
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    print(f"1. Schema version: {CHECKPOINT_SCHEMA_VERSION}")
    print(f"   local_dir  : {local_dir}")
    print(f"   remote_uri : {remote_uri}")
    print()

    # ── Save three checkpoints ────────────────────────────────────
    print("2. Save 3 checkpoints (non-blocking)")
    for step in [100, 200, 300]:
        # Simulate a training step
        x = torch.randn(4, 64)
        loss = model(x).sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        ckpt.save(model, optimizer, step=step, rank=0, extra_meta={"world_size": 1, "ep_size": 1})
        print(f"   step={step}: save() returned in {ckpt.last_commit_ms:.2f}ms (non-blocking)")

    # Drain the async queue
    ckpt._q.join()
    print()

    # ── List available checkpoints ────────────────────────────────
    latest = ckpt.latest_step()
    print(f"3. Latest checkpoint step: {latest}")
    print()

    # ── Load and inspect metadata ─────────────────────────────────
    print("4. Load checkpoint and inspect metadata")
    model2 = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 32))
    optimizer2 = torch.optim.AdamW(model2.parameters(), lr=3e-4)
    meta = ckpt.load(model2, optimizer2, step=latest, rank=0)

    print(
        f"   schema_version    : {meta.get('schema_version')}  (expected {CHECKPOINT_SCHEMA_VERSION})"
    )
    print(f"   step              : {meta.get('step')}")
    print(f"   moe_engine_version: {meta.get('moe_engine_version')}")
    print(f"   torch_version     : {meta.get('torch_version')}")
    print(f"   hostname          : {meta.get('hostname')}")
    print(f"   extra world_size  : {meta.get('world_size')}")
    print()

    # Verify weights were restored
    for (n1, p1), (n2, p2) in zip(model.named_parameters(), model2.named_parameters()):
        assert torch.allclose(p1, p2, atol=1e-6), f"Weight mismatch in {n1}"
    print(
        f"   ✓ All weights restored correctly ({sum(p.numel() for p in model.parameters()):,} params)"
    )
    print()

    # ── Schema compatibility warning ──────────────────────────────
    print("5. Schema compatibility")
    import io
    import logging

    # Capture log output
    handler = logging.StreamHandler(io.StringIO())
    logging.getLogger("moe_engine.elastic").addHandler(handler)

    _check_schema_compatibility({"schema_version": 1})  # older
    out = handler.stream.getvalue()
    print(f"   Schema v1 (older)  → logs: {'forward-compatible' in out or 'older' in out}")

    handler.stream.truncate(0)
    handler.stream.seek(0)
    _check_schema_compatibility({"schema_version": 99})  # newer
    out = handler.stream.getvalue()
    print(f"   Schema v99 (newer) → warns: {'newer' in out}")
    print()

    # ── Retention: only last 3 kept ───────────────────────────────
    print("6. Retention pruning")
    # Save a 4th checkpoint — should prune step=100
    ckpt.save(model, optimizer, step=400, rank=0)
    ckpt._q.join()
    oldest = ckpt.latest_step()
    print(f"   After 4 saves (retention=3), latest step: {ckpt.latest_step()}")
    print()

    ckpt.shutdown()

print("Example 06 complete ✓")
