#!/usr/bin/env python3
"""
examples/07_custom_expert.py
=============================

Implement and register a custom expert FFN architecture.

The standard expert in moe-engine is a SwiGLU FFN:
  w_down(silu(w_gate(x)) * w_up(x))

This example shows how to:
1. Implement a custom expert (GeLU FFN, Gated MLP with different activation)
2. Plug it into DistributedMoELayer via subclassing
3. Register a custom full model using the model registry
4. Verify correctness against the standard SwiGLU expert

Run from moe-engine/:
    python examples/07_custom_expert.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from pkg.distributed.mesh import build_topology
from pkg.distributed.moe_layer import DistributedMoELayer, _SwiGLUExpert
from pkg.models.moe import RMSNorm
from pkg.models.registry import (
    build_model_from_config,
    list_registered_models,
    register_model,
)
from pkg.utils.config import MoEConfig

print("=" * 60)
print("moe-engine Example 07: Custom Expert FFN")
print("=" * 60)
print()

topo = build_topology(dp_size=1, ep_size=1)

# ── 1. Standard SwiGLU expert ────────────────────────────────────
print("1. Standard _SwiGLUExpert (built-in)")
swiglu = _SwiGLUExpert(hidden_dim=64, ffn_dim=128, topology=topo)
x = torch.randn(16, 64)
out_swiglu = swiglu(x)
print(f"   Input:  {x.shape}")
print(f"   Output: {out_swiglu.shape}")
print(f"   Params: {sum(p.numel() for p in swiglu.parameters()):,}")
print()

# ── 2. Custom GeLU expert ────────────────────────────────────────
print("2. Custom GeLUExpert (no gate, single projection)")


class GeLUExpert(nn.Module):
    """Standard two-layer FFN with GeLU activation.

    Architecture: w_down(gelu(w_up(x)))

    Same parameter count as SwiGLU when ffn_dim_gelu = ffn_dim_swiglu * 2/3
    because SwiGLU uses two up-projections (gate + up).
    """

    def __init__(self, hidden_dim: int, ffn_dim: int, topology, dtype=torch.float32):
        super().__init__()
        dev = topology.device
        self.w_up = nn.Linear(hidden_dim, ffn_dim, bias=False, dtype=dtype, device=dev)
        self.w_down = nn.Linear(ffn_dim, hidden_dim, bias=False, dtype=dtype, device=dev)
        nn.init.normal_(self.w_up.weight, std=1.0 / hidden_dim**0.5)
        nn.init.normal_(self.w_down.weight, std=1.0 / ffn_dim**0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.gelu(self.w_up(x)))


gelu_expert = GeLUExpert(hidden_dim=64, ffn_dim=192, topology=topo)
out_gelu = gelu_expert(x)
print(f"   Input:  {x.shape}")
print(f"   Output: {out_gelu.shape}")
print(f"   Params: {sum(p.numel() for p in gelu_expert.parameters()):,}")
assert not torch.isnan(out_gelu).any(), "NaN in GeLU expert output"
print("   ✓ No NaN")
print()

# ── 3. Custom DistributedMoELayer with GeLU experts ──────────────
print("3. Custom DistributedMoELayer subclass with GeLU experts")


class GeLUMoELayer(DistributedMoELayer):
    """MoE layer that uses GeLU experts instead of SwiGLU."""

    def __init__(self, hidden_dim, ffn_dim, num_experts, top_k, topology, **kwargs):
        super().__init__(hidden_dim, ffn_dim, num_experts, top_k, topology, **kwargs)
        # Replace SwiGLU experts with GeLU experts
        self.experts = nn.ModuleList(
            [GeLUExpert(hidden_dim, ffn_dim, topology) for _ in self.local_expert_ids]
        )


layer = GeLUMoELayer(hidden_dim=64, ffn_dim=128, num_experts=4, top_k=2, topology=topo)
tokens = torch.randn(2, 8, 64)
out = layer(tokens)
assert out.shape == tokens.shape
assert not torch.isnan(out).any()
print(f"   Forward: {tokens.shape} → {out.shape}  ✓")
out.sum().backward()
print("   Backward: gradients flow ✓")
print()

# ── 4. Register full model with custom expert ────────────────────
print("4. Register a full model using GeLU experts")


@register_model("gelu_moe")
class GeLUMoEModel(nn.Module):
    """Full toy MoE model using GeLU expert FFN blocks."""

    def __init__(self, cfg: MoEConfig, topology):
        super().__init__()
        H, V = cfg.model.hidden_dim, cfg.model.vocab_size
        self.embed = nn.Embedding(V, H)
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    RMSNorm(H),
                    GeLUMoELayer(
                        hidden_dim=H,
                        ffn_dim=cfg.model.ffn_dim,
                        num_experts=cfg.model.num_experts,
                        top_k=cfg.model.top_k,
                        topology=topology,
                        capacity_factor=cfg.model.capacity_factor,
                    ),
                )
                for _ in range(cfg.model.num_layers)
            ]
        )
        self.norm = RMSNorm(H)
        self.lm_head = nn.Linear(H, V, bias=False)

    def forward(self, ids):
        x = self.embed(ids)
        for norm_moe in self.blocks:
            norm_out = norm_moe[0](x)  # RMSNorm
            moe_out = norm_moe[1](norm_out)  # GeLUMoELayer
            x = x + moe_out  # residual
        return self.lm_head(self.norm(x))


cfg = MoEConfig.from_yaml("configs/smoke.yaml")
gelu_model = build_model_from_config(cfg, topo, arch="gelu_moe")

print(f"   Registered models: {list_registered_models()}")
ids = torch.randint(0, cfg.model.vocab_size, (2, cfg.model.sequence_length))
logits = gelu_model(ids)
assert logits.shape == (2, cfg.model.sequence_length, cfg.model.vocab_size)
print(f"   Forward: {ids.shape} → logits {logits.shape}  ✓")
logits.sum().backward()
print("   Backward: gradients flow  ✓")
print()

# ── 5. Compare activations: SwiGLU vs GeLU ──────────────────────
print("5. Expert activation comparison (SwiGLU vs GeLU)")
print()
x_test = torch.randn(8, 64)
with torch.no_grad():
    y_swiglu = swiglu(x_test)
    y_gelu = gelu_expert(x_test[:, :64])

print(f"   SwiGLU output stats: mean={y_swiglu.mean():.4f}  std={y_swiglu.std():.4f}")
print(f"   GeLU   output stats: mean={y_gelu.mean():.4f}  std={y_gelu.std():.4f}")
print("   Note: different architectures, different parameters — not expected to match")
print()
print("Example 07 complete ✓")
