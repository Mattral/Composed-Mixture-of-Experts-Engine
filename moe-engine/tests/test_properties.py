"""
tests/test_properties.py
=========================

Property-based tests using Hypothesis for routing and parallelism invariants.

Property-based testing finds edge cases that unit tests miss by generating
hundreds of random inputs and verifying invariants hold across all of them.
This is especially valuable for:

- Mathematical invariants (token conservation, weight normalisation)
- Boundary conditions (E=1, K=1, K=E, N=1, very large N)
- Expert ownership (every expert is owned by exactly one EP rank)
- Config validation (cross-field constraints hold for all valid combinations)

These tests run on CPU only and need no GPU.

Reference
---------
- Hypothesis: https://hypothesis.readthedocs.io/
- Strategies used: integers, floats, composite, assume
"""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("hypothesis", reason="hypothesis not installed; pip install hypothesis")

from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from pkg.distributed.mesh import ParallelTopology
from pkg.distributed.router import MoERouterInterface
from pkg.kernels.moe_router import MoERouter
from pkg.models.registry import list_registered_models
from pkg.utils.config import ConfigValidationError, MoEConfig

pytestmark = pytest.mark.cpu

# ---------------------------------------------------------------------------
# Hypothesis settings
# ---------------------------------------------------------------------------
# Use a small deadline to keep CI fast; property tests should each be < 200ms.
# deadline=None: the first Hypothesis example for any test that constructs
# MoERouter may trigger Triton JIT compilation (~1-3s). Triton caches after the
# first call so subsequent examples are fast, but the deadline check fires on
# the initial slow run and produces a FlakyFailure. Since we are testing
# mathematical correctness, not performance, disable the deadline entirely.
_FAST = settings(
    max_examples=50,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
_THOROUGH = settings(
    max_examples=200,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)


# ===========================================================================
# Token conservation — holds for all valid (N, H, E, K) combinations
# ===========================================================================


@_FAST
@given(
    N=st.integers(min_value=1, max_value=128),
    H=st.integers(min_value=8, max_value=256).filter(lambda h: h % 8 == 0),
    E=st.integers(min_value=2, max_value=16),
    K=st.integers(min_value=1, max_value=4),
)
def test_token_conservation_property(N: int, H: int, E: int, K: int) -> None:
    """sum(dispatch_counts) == N * K for all valid (N, H, E, K)."""
    assume(K <= E)
    router = MoERouter(hidden_dim=H, num_experts=E, top_k=K)
    tokens = torch.randn(N, H)
    idx, w, dispatch_counts = router(tokens)
    total = int(dispatch_counts.sum().item())
    assert total == N * K, (
        f"Token conservation failed: sum={total}, expected={N * K} (N={N}, K={K}, E={E}, H={H})"
    )


# ===========================================================================
# Combine weight normalisation — weights sum to 1 per token
# ===========================================================================


@_FAST
@given(
    N=st.integers(min_value=1, max_value=64),
    H=st.integers(min_value=8, max_value=128).filter(lambda h: h % 8 == 0),
    E=st.integers(min_value=2, max_value=8),
    K=st.integers(min_value=1, max_value=4),
)
def test_weight_normalisation_property(N: int, H: int, E: int, K: int) -> None:
    """Combine weights must sum to 1 for every token."""
    assume(K <= E)
    router = MoERouter(hidden_dim=H, num_experts=E, top_k=K)
    tokens = torch.randn(N, H)
    idx, w, _ = router(tokens)
    row_sums = w.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones(N), atol=1e-4), (
        f"Weight normalisation failed: min={row_sums.min():.6f}, max={row_sums.max():.6f}"
    )


# ===========================================================================
# Index validity — all expert indices in [0, E)
# ===========================================================================


@_FAST
@given(
    N=st.integers(min_value=1, max_value=64),
    H=st.integers(min_value=8, max_value=64).filter(lambda h: h % 8 == 0),
    E=st.integers(min_value=2, max_value=16),
    K=st.integers(min_value=1, max_value=4),
)
def test_index_validity_property(N: int, H: int, E: int, K: int) -> None:
    """All expert indices must be in [0, E) for all inputs."""
    assume(K <= E)
    router = MoERouter(hidden_dim=H, num_experts=E, top_k=K)
    tokens = torch.randn(N, H)
    idx, _, _ = router(tokens)
    assert (idx >= 0).all() and (idx < E).all(), (
        f"Index out of bounds: min={idx.min()}, max={idx.max()}, E={E}"
    )


# ===========================================================================
# No NaN — router never produces NaN outputs
# ===========================================================================


@_FAST
@given(
    N=st.integers(min_value=1, max_value=32),
    H=st.integers(min_value=8, max_value=64).filter(lambda h: h % 8 == 0),
    E=st.integers(min_value=2, max_value=8),
    K=st.integers(min_value=1, max_value=2),
    token_scale=st.floats(min_value=0.01, max_value=10.0),
)
def test_no_nan_property(N: int, H: int, E: int, K: int, token_scale: float) -> None:
    """Router outputs must never contain NaN, even with large or small inputs."""
    assume(K <= E)
    router = MoERouter(hidden_dim=H, num_experts=E, top_k=K)
    tokens = torch.randn(N, H) * token_scale
    idx, w, dispatch_counts = router(tokens)
    assert not torch.isnan(idx.float()).any(), "NaN in expert indices"
    assert not torch.isnan(w).any(), "NaN in combine weights"
    assert not torch.isnan(dispatch_counts.float()).any(), "NaN in dispatch counts"


# ===========================================================================
# Expert ownership — every expert owned by exactly one EP rank
# ===========================================================================


@_FAST
@given(
    total_experts=st.integers(min_value=2, max_value=64),
    ep_size=st.integers(min_value=1, max_value=8),
)
def test_expert_ownership_complete(total_experts: int, ep_size: int) -> None:
    """Every expert is owned by exactly one EP rank; no expert is lost or duplicated."""
    assume(ep_size <= total_experts)

    # Collect all experts owned by each rank
    all_owned: list[int] = []
    for rank in range(ep_size):
        topo = ParallelTopology(
            world_size=ep_size,
            rank=rank,
            dp_size=1,
            ep_size=ep_size,
            tp_size=1,
            pp_size=1,
        )
        owned = topo.experts_on_this_rank(total_experts)
        all_owned.extend(owned)

    # Every expert appears exactly once
    all_owned_sorted = sorted(all_owned)
    assert all_owned_sorted == list(range(total_experts)), (
        f"Expert ownership broken: got {all_owned_sorted[:10]}... "
        f"for E={total_experts}, ep_size={ep_size}"
    )


@_FAST
@given(
    total_experts=st.integers(min_value=2, max_value=64),
    ep_size=st.integers(min_value=1, max_value=8),
)
def test_expert_ownership_load_balance(total_experts: int, ep_size: int) -> None:
    """Expert counts per rank differ by at most 1 (round-robin assignment)."""
    assume(ep_size <= total_experts)

    counts = []
    for rank in range(ep_size):
        topo = ParallelTopology(
            world_size=ep_size,
            rank=rank,
            dp_size=1,
            ep_size=ep_size,
            tp_size=1,
            pp_size=1,
        )
        counts.append(len(topo.experts_on_this_rank(total_experts)))

    assert max(counts) - min(counts) <= 1, (
        f"Expert imbalance > 1: counts={counts}, E={total_experts}, ep={ep_size}"
    )


# ===========================================================================
# MoEConfig validation — cross-field constraints hold for random inputs
# ===========================================================================


@_FAST
@given(
    hidden_dim=st.integers(min_value=1, max_value=16).map(lambda x: x * 8),
    num_experts=st.integers(min_value=2, max_value=16),
    top_k=st.integers(min_value=1, max_value=8),
    max_steps=st.integers(min_value=10, max_value=1000),
    warmup_steps=st.integers(min_value=0, max_value=500),
)
def test_config_validation_properties(
    hidden_dim: int,
    num_experts: int,
    top_k: int,
    max_steps: int,
    warmup_steps: int,
) -> None:
    """Valid configs (top_k ≤ E, warmup < max) always load; invalid always raise."""
    # valid_config: top_k <= num_experts AND warmup_steps < max_steps (strict)
    # Note: warmup_steps == max_steps is INVALID (Pydantic requires strictly <)
    valid_config = top_k <= num_experts and warmup_steps < max_steps
    d = {
        "model": {
            "hidden_dim": hidden_dim,
            "num_layers": 1,
            "num_experts": num_experts,
            "top_k": top_k,
            "capacity_factor": 1.25,
            "ffn_dim": hidden_dim * 2,
            "vocab_size": 128,
            "sequence_length": 8,
            "dtype": "float32",
        },
        "training": {
            "global_batch_size": 8,
            "micro_batch_size": 2,
            "learning_rate": 3e-4,
            "weight_decay": 0.1,
            "grad_clip": 1.0,
            "max_steps": max_steps,
            "log_interval": 1,
            "ckpt_interval": 5,
            "warmup_steps": warmup_steps,
            "gradient_accumulation_steps": 1,
        },
        "parallelism": {
            "data_parallel": 1,
            "expert_parallel": 1,
            "tensor_parallel": 1,
            "pipeline_parallel": 1,
        },
        "checkpoint": {
            "local_dir": "/tmp/test",
            "remote_uri": "file:///tmp/r",
            "async_workers": 1,
            "retention": 2,
        },
        "elastic": {
            "min_nodes": 1,
            "max_nodes": 4,
            "rdzv_backend": "c10d",
            "rdzv_endpoint": "localhost:29400",
            "health_check_interval_s": 1.0,
            "drop_grace_period_s": 5.0,
        },
        "telemetry": {
            "log_dir": "/tmp/l",
            "tensorboard_dir": "/tmp/l/tb",
            "json_path": "/tmp/l/s.jsonl",
            "mfu_target": 0.5,
            "hardware_peak_tflops": 989.0,
        },
    }
    if valid_config:
        try:
            cfg = MoEConfig.from_dict(d)
            assert cfg.model.top_k == top_k
            assert cfg.model.num_experts == num_experts
        except ConfigValidationError as e:
            raise AssertionError(
                f"Valid config unexpectedly raised ConfigValidationError: {e}\n"
                f"  top_k={top_k} <= num_experts={num_experts}: {top_k <= num_experts}\n"
                f"  warmup_steps={warmup_steps} < max_steps={max_steps}: {warmup_steps < max_steps}"
            )
    else:
        try:
            MoEConfig.from_dict(d)
            # If no error raised, the config must actually be valid
            # (Hypothesis may generate edge cases we classify as invalid but
            #  Pydantic accepts due to boundary conditions in our logic)
            # Only fail if we are CERTAIN the config is invalid
            if top_k > num_experts:
                raise AssertionError(
                    f"top_k={top_k} > num_experts={num_experts} must raise "
                    f"ConfigValidationError but did not"
                )
            if warmup_steps >= max_steps:
                raise AssertionError(
                    f"warmup_steps={warmup_steps} >= max_steps={max_steps} must raise "
                    f"ConfigValidationError but did not"
                )
        except ConfigValidationError:
            pass  # expected for invalid configs


# ===========================================================================
# RouterStats — MoERouterInterface returns valid stats for all sizes
# ===========================================================================


@_FAST
@given(
    N=st.integers(min_value=1, max_value=32),
    H=st.integers(min_value=1, max_value=8).map(lambda x: x * 8),
    E=st.integers(min_value=2, max_value=8),
    K=st.integers(min_value=1, max_value=3),
)
def test_router_interface_stats_valid(N: int, H: int, E: int, K: int) -> None:
    """MoERouterInterface always returns valid RouterStats for any (N, H, E, K)."""
    assume(K <= E)
    router = MoERouterInterface(hidden_dim=H, num_experts=E, top_k=K)
    tokens = torch.randn(N, H)
    stats = router(tokens)

    assert stats.expert_indices.shape == (N, K)
    assert stats.combine_weights.shape == (N, K)
    assert stats.dispatch_counts.shape == (E,)
    assert int(stats.dispatch_counts.sum()) == N * K
    assert stats.load_imbalance >= 1.0
    assert stats.tokens_per_expert_mean == pytest.approx(N * K / E, rel=1e-4)
    assert not torch.isnan(stats.combine_weights).any()


# ===========================================================================
# Model registry — registered models remain stable
# ===========================================================================


def test_registry_invariant() -> None:
    """toy_moe is always registered after importing pkg.models."""
    assert "toy_moe" in list_registered_models(), (
        "Built-in 'toy_moe' model should always be registered. "
        "Check pkg/models/__init__.py imports pkg.models.registry."
    )
