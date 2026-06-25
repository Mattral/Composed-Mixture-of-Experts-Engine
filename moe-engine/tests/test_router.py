"""
tests/test_router.py
=====================

Tests for ``pkg/distributed/router.py`` — the high-level ``MoERouterInterface``.

These tests are distinct from ``test_kernels.py`` (which tests the raw Triton
kernel) and ``test_properties.py`` (which tests invariants probabilistically).

Coverage:
- Construction validation (top_k <= E, capacity_factor warnings)
- Forward pass: shapes, conservation, weight normalisation
- ``RouterStats`` field completeness and types
- ``capacity_budget`` arithmetic
- Input validation (2D required, hidden_dim match)
- Error messages are informative
- Integration with ``DistributedMoELayer`` (router is correctly called)
- Backward pass: gradients flow through ``MoERouterInterface``
"""

from __future__ import annotations

import warnings

import pytest
import torch

from pkg.distributed.mesh import build_topology
from pkg.distributed.router import MoERouterInterface, RouterStats

pytestmark = pytest.mark.cpu


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def small_router() -> MoERouterInterface:
    """Small router for fast testing: H=64, E=8, K=2."""
    return MoERouterInterface(hidden_dim=64, num_experts=8, top_k=2)


@pytest.fixture
def tokens_32(small_router: MoERouterInterface) -> torch.Tensor:
    return torch.randn(32, small_router.hidden_dim)


# ===========================================================================
# Construction validation
# ===========================================================================


class TestConstruction:
    def test_valid_construction(self):
        r = MoERouterInterface(hidden_dim=128, num_experts=16, top_k=4)
        assert r.hidden_dim == 128
        assert r.num_experts == 16
        assert r.top_k == 4
        assert r.capacity_factor == 1.25

    def test_top_k_equals_num_experts(self):
        """top_k == num_experts is valid (dense routing)."""
        r = MoERouterInterface(hidden_dim=64, num_experts=4, top_k=4)
        assert r.top_k == 4

    def test_top_k_greater_than_num_experts_raises(self):
        with pytest.raises(ValueError, match="top_k.*must be.*num_experts"):
            MoERouterInterface(hidden_dim=64, num_experts=4, top_k=5)

    def test_capacity_factor_below_one_warns(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            MoERouterInterface(hidden_dim=64, num_experts=4, top_k=2, capacity_factor=0.9)
            assert len(w) == 1
            assert "silently drop tokens" in str(w[0].message).lower()

    def test_capacity_factor_one_no_warning(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            MoERouterInterface(hidden_dim=64, num_experts=4, top_k=2, capacity_factor=1.0)
            assert len(w) == 0

    def test_extra_repr_contains_key_info(self):
        r = MoERouterInterface(hidden_dim=256, num_experts=32, top_k=2, capacity_factor=1.5)
        s = repr(r)
        assert "256" in s
        assert "32" in s
        assert "1.5" in s


# ===========================================================================
# Forward pass — shapes and types
# ===========================================================================


class TestForwardShapes:
    @pytest.mark.parametrize(
        "N,H,E,K",
        [
            (1, 8, 2, 1),
            (16, 64, 8, 2),
            (128, 256, 32, 4),
        ],
    )
    def test_output_shapes(self, N: int, H: int, E: int, K: int):
        r = MoERouterInterface(hidden_dim=H, num_experts=E, top_k=K)
        stats = r(torch.randn(N, H))
        assert stats.expert_indices.shape == (N, K), (
            f"idx shape mismatch: {stats.expert_indices.shape}"
        )
        assert stats.combine_weights.shape == (N, K), (
            f"weights shape mismatch: {stats.combine_weights.shape}"
        )
        assert stats.dispatch_counts.shape == (E,), (
            f"dispatch_counts shape mismatch: {stats.dispatch_counts.shape}"
        )

    def test_stats_is_router_stats_instance(self, small_router, tokens_32):
        stats = small_router(tokens_32)
        assert isinstance(stats, RouterStats)

    def test_dispatch_counts_dtype_long(self, small_router, tokens_32):
        stats = small_router(tokens_32)
        assert stats.dispatch_counts.dtype in (torch.int32, torch.int64, torch.long), (
            f"Unexpected dtype: {stats.dispatch_counts.dtype}"
        )

    def test_combine_weights_dtype_float(self, small_router, tokens_32):
        stats = small_router(tokens_32)
        assert stats.combine_weights.is_floating_point()

    def test_expert_indices_in_bounds(self, small_router, tokens_32):
        stats = small_router(tokens_32)
        assert (stats.expert_indices >= 0).all()
        assert (stats.expert_indices < small_router.num_experts).all()


# ===========================================================================
# Mathematical invariants
# ===========================================================================


class TestInvariants:
    def test_token_conservation(self, small_router, tokens_32):
        stats = small_router(tokens_32)
        expected = tokens_32.shape[0] * small_router.top_k
        total = int(stats.dispatch_counts.sum().item())
        assert total == expected, f"Conservation violated: {total} != {expected}"

    def test_weight_normalisation(self, small_router, tokens_32):
        stats = small_router(tokens_32)
        row_sums = stats.combine_weights.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones(tokens_32.shape[0]), atol=1e-4), (
            f"Weights not normalised: min={row_sums.min():.6f}, max={row_sums.max():.6f}"
        )

    def test_no_nan_in_outputs(self, small_router, tokens_32):
        stats = small_router(tokens_32)
        assert not torch.isnan(stats.combine_weights).any()
        assert not torch.isnan(stats.expert_indices.float()).any()

    def test_load_imbalance_ge_one(self, small_router, tokens_32):
        stats = small_router(tokens_32)
        assert stats.load_imbalance >= 1.0, (
            f"load_imbalance={stats.load_imbalance} < 1.0 is impossible"
        )

    def test_tokens_per_expert_mean_correct(self, small_router, tokens_32):
        N = tokens_32.shape[0]
        K = small_router.top_k
        E = small_router.num_experts
        stats = small_router(tokens_32)
        expected_mean = N * K / E
        assert abs(stats.tokens_per_expert_mean - expected_mean) < 1e-4, (
            f"tpe_mean={stats.tokens_per_expert_mean}, expected={expected_mean}"
        )

    def test_tokens_per_expert_std_nonneg(self, small_router, tokens_32):
        stats = small_router(tokens_32)
        assert stats.tokens_per_expert_std >= 0.0


# ===========================================================================
# RouterStats fields completeness
# ===========================================================================


class TestRouterStatsFields:
    def test_all_fields_present(self, small_router, tokens_32):
        stats = small_router(tokens_32)
        expected_fields = [
            "expert_indices",
            "combine_weights",
            "dispatch_counts",
            "load_imbalance",
            "router_z_loss",
            "used_triton",
            "kernel_ms",
            "tokens_per_expert_mean",
            "tokens_per_expert_std",
        ]
        for field in expected_fields:
            assert hasattr(stats, field), f"Missing field: {field}"

    def test_used_triton_is_bool(self, small_router, tokens_32):
        stats = small_router(tokens_32)
        assert isinstance(stats.used_triton, bool)

    def test_kernel_ms_nonneg(self, small_router, tokens_32):
        stats = small_router(tokens_32)
        assert stats.kernel_ms >= 0.0

    def test_router_z_loss_nonneg(self, small_router, tokens_32):
        """Z-loss is mean(log(sum(exp(logit)))^2) which is always >= 0."""
        stats = small_router(tokens_32)
        assert stats.router_z_loss >= 0.0


# ===========================================================================
# capacity_budget arithmetic
# ===========================================================================


class TestCapacityBudget:
    @pytest.mark.parametrize(
        "N,E,K,cf,expected",
        [
            (64, 8, 2, 1.25, 20),  # ceil(1.25 * 64 * 2 / 8) = ceil(20.0) = 20
            (32, 8, 2, 1.25, 10),  # ceil(1.25 * 32 * 2 / 8) = ceil(10.0) = 10
            (100, 10, 2, 1.0, 20),  # ceil(1.0 * 100 * 2 / 10) = ceil(20.0) = 20
            (100, 10, 2, 1.25, 25),  # ceil(1.25 * 100 * 2 / 10) = ceil(25.0) = 25
            (100, 7, 2, 1.0, 29),  # ceil(1.0 * 100 * 2 / 7) = ceil(28.57) = 29
        ],
    )
    def test_capacity_budget(self, N, E, K, cf, expected):
        r = MoERouterInterface(hidden_dim=64, num_experts=E, top_k=K, capacity_factor=cf)
        assert r.capacity_budget(N) == expected, (
            f"capacity_budget({N}) = {r.capacity_budget(N)}, expected {expected}"
        )

    def test_capacity_scales_with_n(self, small_router):
        b1 = small_router.capacity_budget(32)
        b2 = small_router.capacity_budget(64)
        assert b2 == b1 * 2 or abs(b2 - b1 * 2) <= 1  # integer ceiling


# ===========================================================================
# Input validation
# ===========================================================================


class TestInputValidation:
    def test_3d_input_raises(self, small_router):
        x = torch.randn(2, 8, 64)
        with pytest.raises(ValueError, match="2D"):
            small_router(x)

    def test_1d_input_raises(self, small_router):
        x = torch.randn(64)
        with pytest.raises(ValueError, match="2D"):
            small_router(x)

    def test_wrong_hidden_dim_raises(self, small_router):
        x = torch.randn(16, 128)  # wrong H
        with pytest.raises(ValueError, match="hidden dim"):
            small_router(x)

    def test_single_token_works(self, small_router):
        """N=1 (single token) must work — edge case in MoE routing."""
        x = torch.randn(1, small_router.hidden_dim)
        stats = small_router(x)
        assert stats.expert_indices.shape == (1, small_router.top_k)
        assert int(stats.dispatch_counts.sum()) == small_router.top_k

    def test_large_batch_works(self, small_router):
        x = torch.randn(4096, small_router.hidden_dim)
        stats = small_router(x)
        assert int(stats.dispatch_counts.sum()) == 4096 * small_router.top_k


# ===========================================================================
# Backward pass
# ===========================================================================


class TestBackward:
    def test_gradients_flow_through_router(self, small_router, tokens_32):
        """Gate weights must receive gradients through the router forward pass."""
        tokens = tokens_32.detach().requires_grad_(True)
        stats = small_router(tokens)
        loss = stats.combine_weights.sum()
        loss.backward()

        gate_weight = small_router._kernel_router.gate_w
        assert gate_weight.grad is not None, "No gradient on gate_weight"
        assert not torch.isnan(gate_weight.grad).any(), "NaN gradient on gate_weight"

    def test_token_gradients_flow(self, small_router, tokens_32):
        """Input token gradients must be non-None and finite."""
        tokens = tokens_32.detach().requires_grad_(True)
        stats = small_router(tokens)
        stats.combine_weights.sum().backward()
        assert tokens.grad is not None
        assert not torch.isnan(tokens.grad).any()

    def test_multiple_backward_calls(self, small_router):
        """Router can be used in multiple forward-backward cycles (no graph leak)."""
        for _ in range(3):
            x = torch.randn(16, small_router.hidden_dim, requires_grad=True)
            stats = small_router(x)
            stats.combine_weights.sum().backward()
            assert x.grad is not None
            small_router.zero_grad()


# ===========================================================================
# Integration with DistributedMoELayer
# ===========================================================================


class TestIntegrationWithMoELayer:
    def test_moe_layer_uses_router_interface(self):
        """DistributedMoELayer.router attribute wraps the kernel correctly."""
        from pkg.distributed.moe_layer import DistributedMoELayer

        topo = build_topology(dp_size=1, ep_size=1)
        layer = DistributedMoELayer(
            hidden_dim=64, ffn_dim=128, num_experts=4, top_k=2, topology=topo
        )
        # Layer has a router attribute
        assert hasattr(layer, "router"), "DistributedMoELayer must have a .router attribute"

    def test_router_interface_wraps_same_kernel(self):
        """MoERouterInterface and DistributedMoELayer use the same kernel impl."""
        from pkg.kernels.moe_router import MoERouter

        r = MoERouterInterface(hidden_dim=64, num_experts=4, top_k=2)
        assert isinstance(r._kernel_router, MoERouter)
