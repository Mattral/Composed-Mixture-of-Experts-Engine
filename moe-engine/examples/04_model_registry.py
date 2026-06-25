#!/usr/bin/env python3
"""
examples/04_model_registry.py
==============================

Register and build custom models using the moe-engine model registry.

Shows:
- Listing built-in registered models
- Registering a custom model with @register_model
- Building models via build_model_from_config
- The full forward pass of ToyMoEModel
- Parameter count breakdown

Run from moe-engine/:
    python examples/04_model_registry.py
"""

import torch
import torch.nn as nn

from pkg.distributed.mesh import build_topology
from pkg.models.registry import (
    build_model_from_config,
    list_registered_models,
    register_model,
)
from pkg.utils.config import MoEConfig

print("=" * 60)
print("moe-engine Example 04: Model Registry")
print("=" * 60)
print()

topo = build_topology(dp_size=1, ep_size=1)
cfg = MoEConfig.from_yaml("configs/smoke.yaml")

# ── List built-ins ────────────────────────────────────────────────
print("1. Built-in registered models")
for name in list_registered_models():
    print(f"   '{name}'")
print()

# ── Build the default model ───────────────────────────────────────
print("2. Build ToyMoEModel via registry")
model = build_model_from_config(cfg, topo)  # dispatches to "toy_moe"
print(f"   Type     : {type(model).__name__}")
print(f"   Device   : {next(model.parameters()).device}")
counts = model.count_parameters()
print(f"   Total params    : {counts['total']:,}")
print(f"   Expert params   : {counts['moe_experts']:,}")
print(f"   Router params   : {counts['moe_router']:,}")
print(f"   Embed params    : {counts['embed']:,}")
print()

# ── Forward pass ─────────────────────────────────────────────────
print("3. Forward pass")
ids = torch.randint(0, cfg.model.vocab_size, (2, cfg.model.sequence_length))
logits = model(ids)
print(f"   Input  : {ids.shape}  (token IDs)")
print(f"   Output : {logits.shape}  (logits)")
print(f"   Max logit: {logits.max():.3f}  Min logit: {logits.min():.3f}")
print()

# ── Register a custom model ───────────────────────────────────────
print("4. Custom model: LinearMoE (purely linear, for testing)")


@register_model("linear_moe")
class LinearMoEModel(nn.Module):
    """Minimal model: embed → linear → lm_head. No MoE blocks."""

    def __init__(self, cfg, topology):
        super().__init__()
        H = cfg.model.hidden_dim
        V = cfg.model.vocab_size
        self.embed = nn.Embedding(V, H)
        self.linear = nn.Linear(H, H)
        self.lm_head = nn.Linear(H, V, bias=False)

    def forward(self, ids):
        return self.lm_head(self.linear(self.embed(ids)))


print(f"   Registered models now: {list_registered_models()}")
custom = build_model_from_config(cfg, topo, arch="linear_moe")
out = custom(ids)
assert out.shape == logits.shape
print(f"   Custom forward: {ids.shape} → {out.shape}  ✓")
print()

# ── Duplicate registration raises ────────────────────────────────
print("5. Duplicate registration is detected")
# toy_moe is already registered at import time via @register_model decorator
try:
    from pkg.models.registry import ModelRegistry

    ModelRegistry.register("toy_moe", type("DupModel", (), {}))  # attempt to overwrite
except ValueError as e:
    print(f"   ValueError raised: {str(e)[:80]}")
print()

print("Example 04 complete ✓")
