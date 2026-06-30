#!/usr/bin/env python3
"""
examples/03_config_system.py
=============================

Pydantic MoEConfig usage patterns.

Shows:
- Loading from YAML (smoke.yaml and default.yaml)
- Direct field access (type-safe, no dict lookups)
- Cross-field validation (top_k > num_experts → error)
- Environment variable overrides
- Round-trip YAML serialisation
- Legacy load_config() shim
- z_loss_weight for aux load-balancing loss

Run from moe-engine/:
    python examples/03_config_system.py
"""

import os
import pathlib
import tempfile

from pkg.utils.config import ConfigValidationError, MoEConfig, load_config

print("=" * 60)
print("moe-engine Example 03: Config System")
print("=" * 60)
print()

# ── Load smoke config ────────────────────────────────────────────
print("1. Load smoke.yaml")
cfg = MoEConfig.from_yaml("configs/smoke.yaml")
print(f"   hidden_dim      : {cfg.model.hidden_dim}")
print(f"   num_experts     : {cfg.model.num_experts}")
print(f"   top_k           : {cfg.model.top_k}")
print(f"   world_size      : {cfg.parallelism.world_size}")
print(f"   dtype           : {cfg.model.dtype}")
print(f"   z_loss_weight   : {cfg.training.z_loss_weight}  (0.0 = disabled)")
print()

# ── Load production config ───────────────────────────────────────
print("2. Load default.yaml (production scale)")
cfg2 = MoEConfig.from_yaml("configs/default.yaml")
print(f"   hidden_dim      : {cfg2.model.hidden_dim}")
print(f"   num_experts     : {cfg2.model.num_experts}")
print(f"   ffn_dim         : {cfg2.model.ffn_dim}")
print(f"   dtype           : {cfg2.model.dtype}")
print(f"   world_size      : {cfg2.parallelism.world_size}  (64 GPUs)")
print()

# ── Validation catches bad configs ───────────────────────────────
print("3. Validation: bad configs are caught at load time")
try:
    MoEConfig.from_dict(
        {
            "model": {
                "hidden_dim": 64,
                "num_layers": 1,
                "num_experts": 4,
                "top_k": 8,
                "capacity_factor": 1.25,
                "ffn_dim": 128,
                "vocab_size": 256,
                "sequence_length": 8,
                "dtype": "float32",
            }
        }
    )
except ConfigValidationError as e:
    print(f"   top_k=8 > num_experts=4  → {str(e)[:80]}")

try:
    MoEConfig.from_dict(
        {
            "model": {
                "hidden_dim": 64,
                "num_layers": 1,
                "num_experts": 4,
                "top_k": 2,
                "capacity_factor": 1.25,
                "ffn_dim": 128,
                "vocab_size": 256,
                "sequence_length": 8,
                "dtype": "float_128",  # bad dtype
            }
        }
    )
except ConfigValidationError as e:
    print(f"   dtype='float_128'        → {str(e)[:80]}")
print()

# ── Environment variable overrides ───────────────────────────────
print("4. Environment variable overrides")
os.environ["MOE_MODEL__HIDDEN_DIM"] = "128"
os.environ["MOE_TRAINING__Z_LOSS_WEIGHT"] = "1e-3"
cfg3 = MoEConfig.from_yaml("configs/smoke.yaml")
print(f"   MOE_MODEL__HIDDEN_DIM=128       → hidden_dim={cfg3.model.hidden_dim}")
print(f"   MOE_TRAINING__Z_LOSS_WEIGHT=1e-3 → z_loss_weight={cfg3.training.z_loss_weight}")
del os.environ["MOE_MODEL__HIDDEN_DIM"]
del os.environ["MOE_TRAINING__Z_LOSS_WEIGHT"]
print()

# ── Round-trip YAML ──────────────────────────────────────────────
print("5. YAML round-trip")
with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
    tmp = f.name
cfg2.to_yaml(tmp)
cfg_rt = MoEConfig.from_yaml(tmp)
assert cfg_rt.model.hidden_dim == cfg2.model.hidden_dim
assert cfg_rt.model.dtype == cfg2.model.dtype
pathlib.Path(tmp).unlink()
print(f"   to_yaml → from_yaml: hidden_dim={cfg_rt.model.hidden_dim} ✓")
print()

# ── Legacy shim ──────────────────────────────────────────────────
print("6. Legacy load_config() shim (backward compat)")
legacy = load_config("configs/smoke.yaml")
assert legacy.raw["model"]["hidden_dim"] == 32
typed = legacy.typed()
assert typed.model.hidden_dim == 32
print(f"   legacy.raw['model']['hidden_dim'] = {legacy.raw['model']['hidden_dim']}  ✓")
print(f"   legacy.typed().model.hidden_dim   = {typed.model.hidden_dim}  ✓")
print()

# ── parallelism.world_size property ─────────────────────────────
print("7. parallelism.world_size property")
from pkg.utils.config import ParallelismConfig  # noqa: E402

p = ParallelismConfig(data_parallel=4, expert_parallel=8, tensor_parallel=2, pipeline_parallel=2)
print(f"   dp=4 × ep=8 × tp=2 × pp=2 = {p.world_size}  (128 GPUs)")
print()
print("Example 03 complete ✓")
