"""
tests/test_mfu.py
=================

Unit tests for MFU (Model FLOPs Utilization) calculations.
"""

import pytest

from pkg.utils.mfu import compute_mfu, compute_moe_flops

pytestmark = pytest.mark.cpu


def test_compute_mfu_basic():
    """Verify compute_mfu returns valid MFU in range [0.0, 1.0]."""
    batch_tokens = 1024
    param_dense = 1_000_000  # 1M dense params
    param_expert = 500_000  # 500K per expert
    num_experts = 64
    top_k = 2
    world_size = 8
    hardware_peak_tflops = 989.0  # H100 SXM5 BF16
    step_time_sec = 0.1  # 100ms per step

    mfu = compute_mfu(
        batch_tokens=batch_tokens,
        param_dense=param_dense,
        param_expert=param_expert,
        num_experts=num_experts,
        top_k=top_k,
        world_size=world_size,
        hardware_peak_tflops=hardware_peak_tflops,
        step_time_sec=step_time_sec,
    )

    assert 0.0 <= mfu <= 1.0
    # With reasonable defaults, expect MFU in ballpark of [0.3, 0.8]
    assert mfu > 0.0


def test_compute_mfu_sparse_activation():
    """Verify sparse activation (K/E) reduces FLOPs from full model."""
    batch_tokens = 1024
    param_dense = 1_000_000
    param_expert = 500_000
    num_experts = 64
    world_size = 8
    hardware_peak_tflops = 989.0
    step_time_sec = 1.0

    # With K=2, only 2/64 = 3.125% of expert params are active
    mfu_sparse = compute_mfu(
        batch_tokens=batch_tokens,
        param_dense=param_dense,
        param_expert=param_expert,
        num_experts=num_experts,
        top_k=2,
        world_size=world_size,
        hardware_peak_tflops=hardware_peak_tflops,
        step_time_sec=step_time_sec,
    )

    # With K=64 (all experts), 64/64 = 100% of expert params are active
    mfu_dense = compute_mfu(
        batch_tokens=batch_tokens,
        param_dense=param_dense,
        param_expert=param_expert,
        num_experts=num_experts,
        top_k=64,
        world_size=world_size,
        hardware_peak_tflops=hardware_peak_tflops,
        step_time_sec=step_time_sec,
    )

    # Sparse should have lower MFU than dense (fewer FLOPs)
    assert mfu_sparse < mfu_dense


def test_compute_mfu_scaling():
    """Verify MFU scales with batch size and step time."""
    base_batch = 1024
    param_dense = 1_000_000
    param_expert = 500_000
    num_experts = 64
    top_k = 2
    world_size = 8
    hardware_peak_tflops = 989.0
    step_time_sec = 0.1

    # Baseline
    mfu_base = compute_mfu(
        batch_tokens=base_batch,
        param_dense=param_dense,
        param_expert=param_expert,
        num_experts=num_experts,
        top_k=top_k,
        world_size=world_size,
        hardware_peak_tflops=hardware_peak_tflops,
        step_time_sec=step_time_sec,
    )

    # Double batch size -> more tokens -> more FLOPs -> higher MFU
    mfu_double_batch = compute_mfu(
        batch_tokens=base_batch * 2,
        param_dense=param_dense,
        param_expert=param_expert,
        num_experts=num_experts,
        top_k=top_k,
        world_size=world_size,
        hardware_peak_tflops=hardware_peak_tflops,
        step_time_sec=step_time_sec,
    )

    assert mfu_double_batch > mfu_base
    # Should roughly double (2x FLOPs, same time)
    assert abs(mfu_double_batch / mfu_base - 2.0) < 0.01


def test_compute_mfu_world_size_scaling():
    """Verify MFU scales with world size."""
    batch_tokens = 1024
    param_dense = 1_000_000
    param_expert = 500_000
    num_experts = 64
    top_k = 2
    hardware_peak_tflops = 989.0
    step_time_sec = 0.1

    # 8 GPUs
    mfu_8 = compute_mfu(
        batch_tokens=batch_tokens,
        param_dense=param_dense,
        param_expert=param_expert,
        num_experts=num_experts,
        top_k=top_k,
        world_size=8,
        hardware_peak_tflops=hardware_peak_tflops,
        step_time_sec=step_time_sec,
    )

    # 16 GPUs (same FLOPs, but double peak capacity)
    mfu_16 = compute_mfu(
        batch_tokens=batch_tokens,
        param_dense=param_dense,
        param_expert=param_expert,
        num_experts=num_experts,
        top_k=top_k,
        world_size=16,
        hardware_peak_tflops=hardware_peak_tflops,
        step_time_sec=step_time_sec,
    )

    # More GPUs means lower MFU for same throughput (denominator grows)
    assert mfu_16 < mfu_8
    # Should roughly halve
    assert abs(mfu_16 / mfu_8 - 0.5) < 0.01


def test_compute_mfu_edge_cases():
    """Verify compute_mfu handles edge cases gracefully."""
    batch_tokens = 1024
    param_dense = 1_000_000
    param_expert = 500_000
    num_experts = 64
    top_k = 2
    world_size = 1
    hardware_peak_tflops = 989.0

    # Tiny step time (unrealistically fast) should still return valid MFU
    mfu_fast = compute_mfu(
        batch_tokens=batch_tokens,
        param_dense=param_dense,
        param_expert=param_expert,
        num_experts=num_experts,
        top_k=top_k,
        world_size=world_size,
        hardware_peak_tflops=hardware_peak_tflops,
        step_time_sec=1e-6,
    )
    assert 0.0 <= mfu_fast <= 1.0

    # Slow step time (realistic) should return valid MFU
    mfu_slow = compute_mfu(
        batch_tokens=batch_tokens,
        param_dense=param_dense,
        param_expert=param_expert,
        num_experts=num_experts,
        top_k=top_k,
        world_size=world_size,
        hardware_peak_tflops=hardware_peak_tflops,
        step_time_sec=10.0,
    )
    assert 0.0 <= mfu_slow <= 1.0
    # Slow execution should have lower MFU
    assert mfu_slow < mfu_fast


def test_compute_moe_flops_backward_compat():
    """Verify deprecated compute_moe_flops still works."""
    flops = compute_moe_flops(
        hidden_dim=768,
        num_layers=12,
        ffn_dim=3072,
        num_experts=64,
        top_k=2,
        seq_length=2048,
        batch_tokens=1024,
        vocab_size=32000,
    )

    # Should return a positive number
    assert flops > 0
    # Sanity check: should be roughly O(batch * seq * layers * hidden^2)
    expected_order_of_magnitude = 1e9  # billions of FLOPs
    assert flops > expected_order_of_magnitude


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
