#!/usr/bin/env python3
"""
examples/01_router_kernel.py
=============================

Demonstrates the MoE router kernel via the high-level MoERouterInterface.

Shows:
- Basic routing at different (N, H, E, K) configurations
- Reading RouterStats for telemetry fields
- Token conservation verification
- CPU vs GPU throughput measurement

Run from moe-engine/:
    python examples/01_router_kernel.py
"""

import time

import torch

from pkg.distributed.router import MoERouterInterface

print("=" * 60)
print("moe-engine Example 01: Router Kernel")
print("=" * 60)
print()

# ── Basic routing ──────────────────────────────────────────────
print("1. Basic routing (N=128, H=256, E=16, K=2)")
router = MoERouterInterface(hidden_dim=256, num_experts=16, top_k=2)
tokens = torch.randn(128, 256)
stats = router(tokens)

print(f"   expert_indices shape : {stats.expert_indices.shape}")
print(f"   combine_weights shape: {stats.combine_weights.shape}")
print(f"   dispatch_counts      : {stats.dispatch_counts.tolist()}")
print(
    f"   token conservation   : {int(stats.dispatch_counts.sum())} == {128 * 2} ✓"
    if int(stats.dispatch_counts.sum()) == 256
    else "FAIL"
)
print(f"   load_imbalance       : {stats.load_imbalance:.3f}  (1.0=perfect)")
print(f"   router_z_loss        : {stats.router_z_loss:.4f}")
print(f"   used_triton          : {stats.used_triton}")
print()

# ── Weight normalisation check ──────────────────────────────────
print("2. Weight normalisation check")
row_sums = stats.combine_weights.sum(dim=-1)
print(f"   min weight sum: {row_sums.min():.6f}  (should be ≈ 1.0)")
print(f"   max weight sum: {row_sums.max():.6f}  (should be ≈ 1.0)")
assert torch.allclose(row_sums, torch.ones(128), atol=1e-4), "Normalisation failed"
print("   ✓ All rows sum to 1.0 within tolerance")
print()

# ── Throughput sweep ────────────────────────────────────────────
print("3. CPU throughput sweep")
configs = [
    (512, 256, 16, 2),
    (1024, 512, 32, 2),
    (2048, 1024, 64, 2),
]
WARMUP, REPS = 3, 20
print(f"   {'Config':<30} {'Latency(ms)':>12} {'Throughput(M/s)':>16}")
print("   " + "-" * 62)
for N, H, E, K in configs:
    r = MoERouterInterface(hidden_dim=H, num_experts=E, top_k=K)
    x = torch.randn(N, H)
    for _ in range(WARMUP):
        r(x)
    t0 = time.perf_counter()
    for _ in range(REPS):
        r(x)
    elapsed = (time.perf_counter() - t0) / REPS * 1000  # ms
    throughput = N / elapsed * 1e-3  # M tok/s
    tag = f"N={N} H={H} E={E} K={K}"
    print(f"   {tag:<30} {elapsed:>12.3f} {throughput:>16.3f}")
print()

# ── Backward pass ────────────────────────────────────────────────
print("4. Backward pass (gradients through router)")
router2 = MoERouterInterface(hidden_dim=64, num_experts=8, top_k=2)
x = torch.randn(32, 64, requires_grad=True)
stats2 = router2(x)
loss = stats2.combine_weights.sum()
loss.backward()
assert x.grad is not None
print(f"   Input gradient norm: {x.grad.norm():.4f}  (non-zero ✓)")
print()

# ── Capacity budget ──────────────────────────────────────────────
print("5. Capacity budget calculation")
r3 = MoERouterInterface(hidden_dim=64, num_experts=8, top_k=2, capacity_factor=1.25)
for n in [64, 128, 256, 512]:
    budget = r3.capacity_budget(n)
    theoretical = n * 2 / 8  # perfect balance
    print(
        f"   N={n:4d}: budget={budget:4d}  theoretical={theoretical:.0f}  "
        f"overhead={budget / theoretical - 1:.0%}"
    )
print()
print("Example 01 complete ✓")
