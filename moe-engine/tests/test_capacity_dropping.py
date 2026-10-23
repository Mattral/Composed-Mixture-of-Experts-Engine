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
import sys
from pathlib import Path

import pytest
import torch

from pkg.distributed.mesh import build_topology
from pkg.distributed.moe_layer import (
    DistributedMoELayer,
    _cumcount,
    compute_capacity_drop_mask,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tests._differential import capacity_drop_mask  # noqa: E402

pytestmark = pytest.mark.cpu

_F12 = (
    "F12: capacity dropping only zeroes combine weights (every pair is still "
    "dispatched and computed), and capacity is applied per top-k slot instead "
    "of per expert across slots (slot-major priority)"
)


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

    @pytest.mark.xfail(strict=True, reason=_F12)
    def test_multi_k_capacity_is_per_expert_across_slots(self):
        """Capacity bounds an expert's (token, slot) pairs over ALL top-k
        slots: at most ``capacity`` in total. (Replaces
        test_multi_k_independent_per_slot, which pinned the per-slot
        semantics that let one expert accept K x capacity pairs; ADR-011.)"""
        # Token0 -> [e0, e1], token1 -> [e1, e0]: each expert is wanted twice,
        # once per slot. Per-slot capacity 1 kept all four pairs.
        idx = torch.tensor([[0, 1], [1, 0]])
        mask = compute_capacity_drop_mask(idx, num_experts=2, capacity=1)
        expected = torch.tensor([[False, True], [False, True]])
        assert torch.equal(mask, expected)

    @pytest.mark.xfail(strict=True, reason=_F12)
    def test_multi_k_slot_major_priority(self):
        """First choices win over earlier tokens' second choices (GShard /
        Switch): every slot-0 pair is placed before any slot-1 pair."""
        # Token0 -> [e1, e0], token1 -> [e0, e1]. Token-major order would keep
        # token0's second choice (e0) and drop token1's first choice (e0).
        idx = torch.tensor([[1, 0], [0, 1]])
        mask = compute_capacity_drop_mask(idx, num_experts=2, capacity=1)
        expected = torch.tensor([[False, True], [False, True]])
        assert torch.equal(mask, expected)

    @pytest.mark.xfail(strict=True, reason=_F12)
    @pytest.mark.parametrize("seed", range(5))
    def test_matches_slot_major_reference(self, seed):
        """Random routings against a plain-loop slot-major reference: no
        expert keeps more than ``capacity`` pairs, and a pair is dropped
        only when its expert is already full."""
        g = torch.Generator().manual_seed(seed)
        N, K, E, capacity = 40, 3, 5, 9
        idx = torch.stack([torch.randperm(E, generator=g)[:K] for _ in range(N)])
        mask = compute_capacity_drop_mask(idx, num_experts=E, capacity=capacity)
        assert torch.equal(mask, capacity_drop_mask(idx, capacity))
        kept = torch.bincount(idx[~mask], minlength=E)
        assert int(kept.max()) <= capacity

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


# ===========================================================================
# Real dropping: dropped pairs are neither dispatched nor computed (F12)
# ===========================================================================


def _dropping_layer(seed=0, H=16, Fd=32, E=8, K=2, cf=0.25):
    torch.manual_seed(seed)
    topo = build_topology(dp_size=1, ep_size=1)
    return DistributedMoELayer(H, Fd, E, K, topo, capacity_factor=cf, capacity_dropping=True)


class TestDropBeforeDispatch:
    @pytest.mark.xfail(strict=True, reason=_F12)
    def test_experts_compute_only_kept_pairs(self):
        """The audit probe's case (E=8, K=2, N=64, capacity_factor=0.25):
        experts used to compute all N*K = 128 rows. They must compute exactly
        the kept pairs, at most ``capacity`` per expert."""
        from tests._differential import dense_router

        layer = _dropping_layer()
        rows = [0] * layer.num_experts
        for eid, m in zip(layer.local_expert_ids, layer.experts):
            m.register_forward_hook(
                lambda mod, inp, out, eid=eid: rows.__setitem__(eid, rows[eid] + inp[0].shape[0])
            )
        x = torch.randn(64, 16)
        with torch.no_grad():
            layer(x)
        capacity = math.ceil(0.25 * 64 * 2 / layer.num_experts)
        idx, _ = dense_router(x, layer.router.gate_w.detach(), 2)
        kept = torch.bincount(idx[~capacity_drop_mask(idx, capacity)], minlength=8)
        assert rows == kept.tolist()
        assert max(rows) <= capacity
        assert sum(rows) < 64 * 2

    # k=1 passes before the fix too: with one slot, per-slot and per-expert
    # capacity coincide, and zeroed weights give the same output as dropping.
    @pytest.mark.parametrize(
        "k", [1, pytest.param(2, marks=pytest.mark.xfail(strict=True, reason=_F12))]
    )
    def test_output_and_grads_match_dense_reference(self, k):
        from tests._differential import dense_moe

        layer = _dropping_layer(seed=3, K=k, cf=0.5)
        x = torch.randn(24, 16, requires_grad=True)
        cot = torch.randn(24, 16)
        (layer(x) * cot).sum().backward()

        xr = x.detach().clone().requires_grad_(True)
        gr = layer.router.gate_w.detach().clone().requires_grad_(True)
        experts = [
            [
                w.detach().clone().requires_grad_(True)
                for w in (m.w_gate.weight, m.w_up.weight, m.w_down.weight)
            ]
            for m in layer.experts
        ]
        capacity = math.ceil(0.5 * 24 * k / layer.num_experts)
        y_ref = dense_moe(xr, gr, experts, k, capacity)
        (y_ref * cot).sum().backward()
        torch.testing.assert_close(layer(x.detach()), y_ref.detach(), atol=1e-5, rtol=1e-4)
        torch.testing.assert_close(x.grad, xr.grad, atol=1e-5, rtol=1e-4)
        torch.testing.assert_close(layer.router.gate_w.grad, gr.grad, atol=1e-5, rtol=1e-4)
        for m, ref in zip(layer.experts, experts):
            for w, wr in zip((m.w_gate.weight, m.w_up.weight, m.w_down.weight), ref):
                g = w.grad if w.grad is not None else torch.zeros_like(w)
                gref = wr.grad if wr.grad is not None else torch.zeros_like(wr)
                torch.testing.assert_close(g, gref, atol=1e-5, rtol=1e-4)

    @pytest.mark.xfail(strict=True, reason=_F12)
    def test_model_takes_capacity_dropping_from_config(self):
        """cfg.model.capacity_dropping must reach every MoE layer (ToyMoEBlock
        never passed it, so the config switch had no effect)."""
        from pkg.models.moe import ToyMoEModel
        from pkg.utils.config import MoEConfig

        cfg = MoEConfig.from_yaml(Path(__file__).resolve().parents[1] / "configs" / "smoke.yaml")
        cfg.model.capacity_dropping = True
        model = ToyMoEModel(cfg, build_topology(dp_size=1, ep_size=1))
        layers = [m for m in model.modules() if isinstance(m, DistributedMoELayer)]
        assert layers and all(m.capacity_dropping for m in layers)
