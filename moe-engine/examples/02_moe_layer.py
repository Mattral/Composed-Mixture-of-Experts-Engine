#!/usr/bin/env python3
"""
examples/02_moe_layer.py
=========================

Full DistributedMoELayer forward + backward at single-rank (ep_size=1).

Shows:
- Layer construction with ParallelTopology
- Forward pass shapes and invariants
- Backward pass gradient flow
- Telemetry attributes (dispatch_ms, overlap_ratio)

Run from moe-engine/:
    python examples/02_moe_layer.py
"""

import torch

from pkg.distributed.mesh import build_topology
from pkg.distributed.moe_layer import DistributedMoELayer

print("=" * 60)
print("moe-engine Example 02: DistributedMoELayer")
print("=" * 60)
print()

topo = build_topology(dp_size=1, ep_size=1)
print(f"Topology: world_size={topo.world_size}, device={topo.device}")
print()

# ── Construct and forward ────────────────────────────────────────
print("1. Construction and forward pass")
layer = DistributedMoELayer(hidden_dim=128, ffn_dim=256, num_experts=8, top_k=2, topology=topo)
n_params = sum(p.numel() for p in layer.parameters())
print(f"   Layer params: {n_params:,}")
print(f"   Local experts owned: {layer.local_expert_ids}")
print()

tokens = torch.randn(2, 16, 128)  # [B, S, H]
print(f"   Input:  {tokens.shape}")
out = layer(tokens)
print(f"   Output: {out.shape}")
assert out.shape == tokens.shape
assert not torch.isnan(out).any(), "NaN in output!"
print("   ✓ Shape preserved, no NaN")
print()

# ── Telemetry ────────────────────────────────────────────────────
print("2. Telemetry attributes")
print(f"   dispatch_ms       : {layer.last_dispatch_ms:.4f}")
print(f"   combine_ms        : {layer.last_combine_ms:.4f}")
print(f"   expert_compute_ms : {layer.last_expert_compute_ms:.4f}")
print(f"   overlap_ratio     : {layer.last_overlap_ratio:.4f}  (0.0 at ep_size=1)")
print()

# ── Backward ─────────────────────────────────────────────────────
print("3. Backward pass")
loss = out.sum()
loss.backward()
grads = [(n, p.grad) for n, p in layer.named_parameters() if p.grad is not None]
print(f"   Parameters with gradients: {len(grads)}/{sum(1 for _ in layer.parameters())}")
nan_grads = [(n, g) for n, g in grads if torch.isnan(g).any()]
print(f"   NaN gradients: {len(nan_grads)}  (should be 0)")
print()

# ── Expert-to-rank mapping ───────────────────────────────────────
print("4. Expert-to-rank mapping")
ids = torch.arange(8)
ranks = layer._expert_to_rank(ids)
print(f"   Expert IDs: {ids.tolist()}")
print(f"   EP ranks:   {ranks.tolist()}  (all 0 at ep_size=1)")
print()

# ── Scaling test ─────────────────────────────────────────────────
print("5. Scaling: N=512, H=256, E=16, K=2")
layer2 = DistributedMoELayer(hidden_dim=256, ffn_dim=512, num_experts=16, top_k=2, topology=topo)
big = torch.randn(4, 128, 256)  # [B=4, S=128, H=256]
out2 = layer2(big)
assert out2.shape == big.shape
print(f"   {big.shape} → {out2.shape} ✓")
print()
print("Example 02 complete ✓")
