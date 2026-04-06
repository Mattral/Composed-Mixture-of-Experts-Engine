"""
tests/test_routing_quality.py
==============================

v0.2 — Routing quality invariant tests.

These tests validate the new routing quality metrics introduced in v0.2:
  * Expert load imbalance ratio (max_load / mean_load)
  * Router z-loss (auxiliary regularization signal)
  * RouterProfile completeness after every forward pass
  * Imbalance improves with more uniform gate initialization

These are UNIT tests — no distributed collectives, CPU-only, sub-second.
"""

from __future__ import annotations

import math

import pytest
import torch

from pkg.kernels.moe_router import (
    MoERouter,
    _compute_load_imbalance,
    _compute_router_z_loss,
)


# ---------------------------------------------------------------------------
# Load imbalance helpers
# ---------------------------------------------------------------------------

def test_load_imbalance_perfect_balance():
    """A perfectly uniform dispatch_cnt yields imbalance == 1.0."""
    cnt = torch.tensor([10, 10, 10, 10], dtype=torch.long)
    assert abs(_compute_load_imbalance(cnt) - 1.0) < 1e-6


def test_load_imbalance_all_to_one():
    """All tokens routed to one expert yields high imbalance."""
    cnt = torch.tensor([100, 0, 0, 0], dtype=torch.long)
    # max=100, mean=25 → ratio=4.0
    assert abs(_compute_load_imbalance(cnt) - 4.0) < 1e-6


def test_load_imbalance_two_experts():
    cnt = torch.tensor([30, 10, 20, 40], dtype=torch.long)
    expected = 40.0 / 25.0   # max=40, mean=25
    assert abs(_compute_load_imbalance(cnt) - expected) < 1e-6


def test_load_imbalance_zero_counts():
    """All-zero counts should not divide by zero."""
    cnt = torch.zeros(8, dtype=torch.long)
    result = _compute_load_imbalance(cnt)
    assert result == 1.0


# ---------------------------------------------------------------------------
# Z-loss
# ---------------------------------------------------------------------------

def test_router_z_loss_positive():
    """Z-loss must always be non-negative."""
    torch.manual_seed(0)
    logits = torch.randn(64, 32)
    z = _compute_router_z_loss(logits)
    assert z >= 0.0


def test_router_z_loss_zero_logits():
    """With all-zero logits, log(sum(exp(0))) = log(E), z_loss = log(E)^2."""
    E = 16
    logits = torch.zeros(32, E)
    z = _compute_router_z_loss(logits)
    expected = math.log(E) ** 2
    assert abs(z - expected) < 1e-4


def test_router_z_loss_large_logits_bigger():
    """Larger logit magnitudes should produce larger z-loss."""
    torch.manual_seed(1)
    logits_small = torch.randn(64, 16) * 0.01
    logits_large = torch.randn(64, 16) * 10.0
    z_small = _compute_router_z_loss(logits_small)
    z_large = _compute_router_z_loss(logits_large)
    assert z_large > z_small


# ---------------------------------------------------------------------------
# RouterProfile completeness
# ---------------------------------------------------------------------------

def test_router_profile_populated_after_forward():
    """RouterProfile must be non-None after every forward pass."""
    router = MoERouter(hidden_dim=32, num_experts=8, top_k=2)
    tokens = torch.randn(16, 32)
    assert router.last_profile is None
    router(tokens)
    p = router.last_profile
    assert p is not None


def test_router_profile_fields_v02():
    """v0.2 RouterProfile must include load_imbalance and z_loss fields."""
    router = MoERouter(hidden_dim=64, num_experts=16, top_k=2)
    tokens = torch.randn(32, 64)
    router(tokens)
    p = router.last_profile
    # New v0.2 fields
    assert hasattr(p, "expert_load_imbalance"), "Missing expert_load_imbalance"
    assert hasattr(p, "router_z_loss"), "Missing router_z_loss"
    assert p.expert_load_imbalance >= 1.0, "Imbalance ratio must be >= 1.0"
    assert p.router_z_loss >= 0.0, "Z-loss must be non-negative"
    # Existing v0.1 fields still present
    assert p.tokens_per_expert_mean > 0.0
    assert p.tokens_per_expert_std >= 0.0
    assert isinstance(p.used_triton, bool)


def test_router_profile_consistent_with_dispatch():
    """tokens_per_expert_mean == N*K/E exactly."""
    H, E, K, N = 32, 8, 2, 64
    router = MoERouter(hidden_dim=H, num_experts=E, top_k=K)
    tokens = torch.randn(N, H)
    router(tokens)
    p = router.last_profile
    expected_mean = (N * K) / E
    assert abs(p.tokens_per_expert_mean - expected_mean) < 1e-4


# ---------------------------------------------------------------------------
# Imbalance improves with uniform initialization
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_uniform_init_lower_imbalance(seed):
    """Initialization with very small std → near-uniform routing → low imbalance."""
    torch.manual_seed(seed)
    H, E, K, N = 64, 16, 2, 256

    router_uniform = MoERouter(hidden_dim=H, num_experts=E, top_k=K)
    # Override gate to near-zero → near-uniform softmax → near-uniform dispatch
    torch.nn.init.normal_(router_uniform.gate_w, mean=0.0, std=0.001)
    tokens = torch.randn(N, H)
    router_uniform(tokens)
    p_uniform = router_uniform.last_profile

    router_sharp = MoERouter(hidden_dim=H, num_experts=E, top_k=K)
    # Override gate to large values → peaked softmax → imbalanced dispatch
    torch.nn.init.normal_(router_sharp.gate_w, mean=0.0, std=10.0)
    router_sharp(tokens)
    p_sharp = router_sharp.last_profile

    assert p_uniform.expert_load_imbalance <= p_sharp.expert_load_imbalance, (
        f"Expected uniform init to have lower imbalance "
        f"({p_uniform.expert_load_imbalance:.3f}) than "
        f"sharp init ({p_sharp.expert_load_imbalance:.3f})"
    )


# ---------------------------------------------------------------------------
# Integration: imbalance and z_loss in training loop (no torch.distributed)
# ---------------------------------------------------------------------------

def test_routing_quality_emitted_per_step():
    """Simulate multi-step loop; verify profile is refreshed each step."""
    H, E, K, N = 64, 8, 2, 32
    router = MoERouter(hidden_dim=H, num_experts=E, top_k=K)
    imbalances = []
    z_losses = []
    for step in range(5):
        tokens = torch.randn(N, H)
        router(tokens)
        imbalances.append(router.last_profile.expert_load_imbalance)
        z_losses.append(router.last_profile.router_z_loss)

    # All steps must produce valid values
    assert all(v >= 1.0 for v in imbalances)
    assert all(v >= 0.0 for v in z_losses)
    # Profile should differ across steps (different tokens → different routing)
    # (With high probability; technically could all be equal with tiny prob)
    assert len(set(round(v, 6) for v in imbalances)) > 1 or len(imbalances) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
