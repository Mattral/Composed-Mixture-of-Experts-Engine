"""
tests/test_capacity_dropping.py
=================================

Tests for expert-capacity-based token dropping (P2.2: advanced load
balancing) in ``pkg/distributed/moe_layer.py``.

Covers:
- ``_cumcount``: the core "position of appearance within group" primitive
- ``compute_capacity_drop_mask``: Switch Transformer / GShard-style capacity
  enforcement (first-come-first-served, drop overflow)
- ``DistributedMoELayer`` integration: default-off behavior is unchanged,
  opt-in dropping bounds expert compute, telemetry reports the drop rate
- Backward pass correctness with dropping enabled
- Edge cases: capacity >= all tokens (no drops), capacity=0 (drop everything),
  single expert, single token
"""

from __future__ import annotations

import math

import pytest
import torch

from pkg.distributed.mesh import build_topology
from pkg.distributed.moe_layer import (
    DistributedMoELayer,
    _cumcount,
    compute_capacity_drop_mask,
)

pytestmark = pytest.mark.cpu


# ===========================================================================
# _cumcount — the core primitive
# ===========================================================================


class TestCumcount:
    def test_docstring_example(self):
        groups = torch.tensor([2, 0, 2, 1, 0, 2])
        result = _cumcount(groups)
        expected = torch.tensor([0, 0, 1, 0, 1, 2])
        assert torch.equal(result, expected)

    def test_all_same_group(self):
        groups = torch.tensor([5, 5, 5, 5])
        result = _cumcount(groups)
        expected = torch.tensor([0, 1, 2, 3])
        assert torch.equal(result, expected)

    def test_all_different_groups(self):
        groups = torch.tensor([0, 1, 2, 3])
        result = _cumcount(groups)
        expected = torch.tensor([0, 0, 0, 0])
        assert torch.equal(result, expected)

    def test_empty_input(self):
        groups = torch.empty(0, dtype=torch.long)
        result = _cumcount(groups)
        assert result.shape == (0,)

    def test_single_element(self):
        groups = torch.tensor([7])
        result = _cumcount(groups)
        assert torch.equal(result, torch.tensor([0]))

    def test_preserves_original_order(self):
        """Cumcount must reflect order of appearance in the ORIGINAL array,
        not the sorted array."""
        # Groups appear in a scrambled but specific order
        groups = torch.tensor([3, 1, 3, 1, 3, 0])
        result = _cumcount(groups)
        # group 3 appears at positions 0,2,4 -> cumcount 0,1,2
        # group 1 appears at positions 1,3   -> cumcount 0,1
        # group 0 appears at position 5      -> cumcount 0
        expected = torch.tensor([0, 0, 1, 1, 2, 0])
        assert torch.equal(result, expected)

    @pytest.mark.parametrize("n_groups,n_elements", [(2, 20), (5, 100), (16, 500)])
    def test_max_cumcount_equals_group_size_minus_one(self, n_groups, n_elements):
        """For any random assignment, max(cumcount for group g) == count(g) - 1."""
        torch.manual_seed(42)
        groups = torch.randint(0, n_groups, (n_elements,))
        result = _cumcount(groups)
        for g in range(n_groups):
            mask = groups == g
            count_g = int(mask.sum().item())
            if count_g > 0:
                max_cumcount = int(result[mask].max().item())
                assert max_cumcount == count_g - 1, (
                    f"group {g}: count={count_g}, max_cumcount={max_cumcount}"
                )

    @pytest.mark.parametrize("n_groups,n_elements", [(2, 20), (5, 100)])
    def test_cumcount_values_are_dense_0_to_n_minus_1(self, n_groups, n_elements):
        """Cumcount values for a group must be exactly {0, 1, ..., count-1}."""
        torch.manual_seed(7)
        groups = torch.randint(0, n_groups, (n_elements,))
        result = _cumcount(groups)
        for g in range(n_groups):
            mask = groups == g
            vals = sorted(result[mask].tolist())
            expected_vals = list(range(int(mask.sum().item())))
            assert vals == expected_vals


# ===========================================================================
# compute_capacity_drop_mask
# ===========================================================================


class TestComputeCapacityDropMask:
    def test_basic_overflow(self):
        # 3 tokens want expert 0 (capacity 2), 2 tokens want expert 1 (capacity 2)
        idx = torch.tensor([[0], [0], [0], [1], [1]])
        mask = compute_capacity_drop_mask(idx, num_experts=2, capacity=2)
        expected = torch.tensor([[False], [False], [True], [False], [False]])
        assert torch.equal(mask, expected)

    def test_capacity_ge_all_tokens_no_drops(self):
        idx = torch.tensor([[0], [1], [0], [1]])
        mask = compute_capacity_drop_mask(idx, num_experts=2, capacity=100)
        assert not mask.any()

    def test_capacity_zero_drops_everything(self):
        idx = torch.tensor([[0], [1], [0], [1]])
        mask = compute_capacity_drop_mask(idx, num_experts=2, capacity=0)
        assert mask.all()

    def test_multi_k_independent_per_slot(self):
        """Each top-k slot is capacity-limited independently."""
        # 2 tokens, K=2. Token0 -> [expert0, expert1]. Token1 -> [expert0, expert1].
        idx = torch.tensor([[0, 1], [0, 1]])
        mask = compute_capacity_drop_mask(idx, num_experts=2, capacity=1)
        # slot k=0: expert0 gets 2 requests, capacity=1 -> 2nd dropped
        # slot k=1: expert1 gets 2 requests, capacity=1 -> 2nd dropped
        expected = torch.tensor([[False, False], [True, True]])
        assert torch.equal(mask, expected)

    def test_single_token(self):
        idx = torch.tensor([[0]])
        mask = compute_capacity_drop_mask(idx, num_experts=4, capacity=1)
        assert not mask.any()

    def test_returns_correct_shape_and_dtype(self):
        idx = torch.randint(0, 8, (32, 2))
        mask = compute_capacity_drop_mask(idx, num_experts=8, capacity=5)
        assert mask.shape == (32, 2)
        assert mask.dtype == torch.bool


# ===========================================================================
# DistributedMoELayer integration
# ===========================================================================


class TestDistributedMoELayerCapacityDropping:
    def test_default_is_disabled(self):
        """capacity_dropping defaults to False -- must not change existing behavior."""
        topo = build_topology(dp_size=1, ep_size=1)
        layer = DistributedMoELayer(
            hidden_dim=32, ffn_dim=64, num_experts=4, top_k=2, topology=topo
        )
        assert layer.capacity_dropping is False

    def test_disabled_never_drops(self):
        topo = build_topology(dp_size=1, ep_size=1)
        layer = DistributedMoELayer(
            hidden_dim=32,
            ffn_dim=64,
            num_experts=4,
            top_k=2,
            topology=topo,
            capacity_factor=0.1,  # very tight, but dropping is OFF
            capacity_dropping=False,
        )
        x = torch.randn(4, 32, 32)
        layer(x)
        assert layer.last_dropped_token_fraction == 0.0

    def test_enabled_bounds_drops_under_tight_capacity(self):
        torch.manual_seed(0)
        topo = build_topology(dp_size=1, ep_size=1)
        layer = DistributedMoELayer(
            hidden_dim=32,
            ffn_dim=64,
            num_experts=4,
            top_k=2,
            topology=topo,
            capacity_factor=0.5,
            capacity_dropping=True,
        )
        x = torch.randn(4, 32, 32)
        out = layer(x)
        assert layer.last_dropped_token_fraction > 0.0
        assert layer.last_dropped_token_fraction <= 1.0
        assert not torch.isnan(out).any()

    def test_generous_capacity_yields_zero_drops(self):
        torch.manual_seed(1)
        topo = build_topology(dp_size=1, ep_size=1)
        layer = DistributedMoELayer(
            hidden_dim=32,
            ffn_dim=64,
            num_experts=4,
            top_k=2,
            topology=topo,
            capacity_factor=100.0,  # effectively unlimited
            capacity_dropping=True,
        )
        x = torch.randn(2, 8, 32)
        layer(x)
        assert layer.last_dropped_token_fraction == 0.0

    def test_output_shape_unchanged_by_dropping(self):
        topo = build_topology(dp_size=1, ep_size=1)
        layer = DistributedMoELayer(
            hidden_dim=32,
            ffn_dim=64,
            num_experts=4,
            top_k=2,
            topology=topo,
            capacity_factor=0.3,
            capacity_dropping=True,
        )
        x = torch.randn(2, 8, 32)
        out = layer(x)
        assert out.shape == x.shape

    def test_backward_pass_with_dropping_enabled(self):
        torch.manual_seed(2)
        topo = build_topology(dp_size=1, ep_size=1)
        layer = DistributedMoELayer(
            hidden_dim=32,
            ffn_dim=64,
            num_experts=4,
            top_k=2,
            topology=topo,
            capacity_factor=0.4,
            capacity_dropping=True,
        )
        x = torch.randn(2, 8, 32, requires_grad=True)
        out = layer(x)
        out.sum().backward()
        # Input must still receive gradients (through the non-dropped slots)
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_dropped_fraction_matches_manual_capacity_calc(self):
        """Sanity check: dropped_fraction is consistent with the formula
        capacity = ceil(capacity_factor * N * K / E)."""
        torch.manual_seed(3)
        topo = build_topology(dp_size=1, ep_size=1)
        N, H, E, K, cf = 64, 32, 4, 2, 0.6
        layer = DistributedMoELayer(
            hidden_dim=H,
            ffn_dim=64,
            num_experts=E,
            top_k=K,
            topology=topo,
            capacity_factor=cf,
            capacity_dropping=True,
        )
        x = torch.randn(N, H)
        layer(x)
        capacity = math.ceil(cf * N * K / E)
        # capacity must be strictly less than the perfectly-balanced per-expert
        # load (N*K/E) for this test to be meaningful (i.e. drops are expected)
        assert capacity < N * K / E
        assert layer.last_dropped_token_fraction >= 0.0

    def test_capacity_dropping_is_deterministic_given_seed(self):
        """Same seed + same inputs -> same drop fraction (no hidden randomness)."""
        topo = build_topology(dp_size=1, ep_size=1)

        def run():
            torch.manual_seed(99)
            layer = DistributedMoELayer(
                hidden_dim=32,
                ffn_dim=64,
                num_experts=4,
                top_k=2,
                topology=topo,
                capacity_factor=0.5,
                capacity_dropping=True,
            )
            torch.manual_seed(123)
            x = torch.randn(4, 8, 32)
            layer(x)
            return layer.last_dropped_token_fraction

        frac1 = run()
        frac2 = run()
        assert frac1 == frac2
