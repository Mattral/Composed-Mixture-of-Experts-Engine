"""
tests/test_mfu_v02.py
=====================

v0.2 additions to MFU test coverage.

Tests the new MFUAccountant streaming tracker, compute_mfu_detailed
breakdown, smoothed MFU sliding window, and activation recompute
accounting — all new in v0.2.
"""

from __future__ import annotations

import time

import pytest

from pkg.utils.mfu import (


    MFUAccountant,
    MFUResult,
    compute_mfu,
    compute_mfu_detailed,
    compute_moe_flops,
)


pytestmark = pytest.mark.cpu

# ---------------------------------------------------------------------------
# compute_mfu_detailed — FLOP breakdown
# ---------------------------------------------------------------------------

def test_compute_mfu_detailed_returns_mfuresult():
    res = compute_mfu_detailed(
        batch_tokens=1024,
        param_dense=4_000_000,
        param_expert=2_000_000,
        num_experts=64,
        top_k=2,
        world_size=8,
        hardware_peak_tflops=989.0,
        step_time_sec=0.2,
    )
    assert isinstance(res, MFUResult)
    assert 0.0 <= res.mfu <= 1.0
    assert res.achieved_tflops > 0
    assert res.peak_tflops == pytest.approx(8 * 989.0)
    assert res.step_ms == pytest.approx(200.0, abs=0.1)
    assert res.tokens_per_sec == pytest.approx(1024 / 0.2, rel=1e-4)

def test_compute_mfu_detailed_flop_split():
    """Dense and sparse FLOP fields must be positive and sum < total."""
    res = compute_mfu_detailed(
        batch_tokens=512,
        param_dense=2_000_000,
        param_expert=1_000_000,
        num_experts=32,
        top_k=2,
        world_size=4,
        hardware_peak_tflops=312.0,  # A100 BF16
        step_time_sec=0.1,
    )
    assert res.flops_dense > 0
    assert res.flops_sparse > 0
    # Sparse fraction: K/E = 2/32 = 6.25%; dense >> sparse here
    assert res.flops_dense > res.flops_sparse

def test_compute_mfu_detailed_activation_recompute():
    """Activation recompute (3× multiplier) must produce higher FLOPs than 2×."""
    kwargs = dict(
        batch_tokens=1024,
        param_dense=4_000_000,
        param_expert=2_000_000,
        num_experts=64,
        top_k=2,
        world_size=8,
        hardware_peak_tflops=989.0,
        step_time_sec=0.2,
    )
    res_no_recompute = compute_mfu_detailed(**kwargs, activation_recompute=False)
    res_recompute = compute_mfu_detailed(**kwargs, activation_recompute=True)
    # 3× vs 2× multiplier → recompute version has 50% more FLOPs
    ratio = res_recompute.achieved_tflops / res_no_recompute.achieved_tflops
    assert abs(ratio - 1.5) < 0.01

def test_compute_mfu_no_expert_params():
    """Purely dense model (param_expert=0) should still compute valid MFU."""
    mfu = compute_mfu(
        batch_tokens=2048,
        param_dense=10_000_000,
        param_expert=0,
        num_experts=1,
        top_k=1,
        world_size=1,
        hardware_peak_tflops=312.0,
        step_time_sec=0.05,
    )
    assert 0.0 <= mfu <= 1.0

def test_compute_mfu_sparse_fraction_correctness():
    """K/E sparse fraction must exactly scale expert FLOPs."""
    base = dict(
        batch_tokens=1000,
        param_dense=0,
        param_expert=1_000_000,
        num_experts=64,
        world_size=1,
        hardware_peak_tflops=1.0,   # normalize so result is in absolute TFLOPs
        step_time_sec=1.0,
    )
    res_k1 = compute_mfu_detailed(**base, top_k=1)
    res_k4 = compute_mfu_detailed(**base, top_k=4)
    # k=4 should have 4× the sparse FLOPs of k=1
    ratio = res_k4.flops_sparse / res_k1.flops_sparse
    assert abs(ratio - 4.0) < 0.01

# ---------------------------------------------------------------------------
# MFUAccountant streaming tracker
# ---------------------------------------------------------------------------

def test_mfu_accountant_basic_step():
    acct = MFUAccountant(peak_tflops=989.0)
    acct.configure(flops_per_token=6_000)
    acct.start_step()
    time.sleep(0.005)   # 5ms simulated step
    res = acct.end_step(tokens=1024)
    assert isinstance(res, MFUResult)
    assert 0.0 <= res.mfu <= 1.0
    assert res.tokens_per_sec > 0

def test_mfu_accountant_running_average():
    acct = MFUAccountant(peak_tflops=989.0)
    acct.configure(flops_per_token=6_000)
    for _ in range(5):
        acct.start_step()
        time.sleep(0.002)
        acct.end_step(tokens=512)
    # running average must be between 0 and 1
    assert 0.0 <= acct.running_mfu <= 1.0
    assert len(acct.history) == 5

def test_mfu_accountant_smoothed_window():
    acct = MFUAccountant(peak_tflops=989.0, smoothing_window=3)
    acct.configure(flops_per_token=6_000)
    for _ in range(10):
        acct.start_step()
        time.sleep(0.001)
        acct.end_step(tokens=256)
    # smoothed should only reflect last 3 steps
    assert 0.0 <= acct.smoothed_mfu <= 1.0

def test_mfu_accountant_empty_smoothed():
    acct = MFUAccountant(peak_tflops=989.0)
    assert acct.smoothed_mfu == 0.0

def test_mfu_accountant_summary_str_no_data():
    acct = MFUAccountant(peak_tflops=989.0)
    s = acct.summary_str()
    assert "no data" in s.lower()

def test_mfu_accountant_summary_str_with_data():
    acct = MFUAccountant(peak_tflops=312.0, mfu_target=0.40)
    acct.configure(flops_per_token=4_000)
    acct.start_step()
    time.sleep(0.003)
    acct.end_step(tokens=512)
    s = acct.summary_str()
    assert "tok/s" in s
    assert "step=" in s
    assert "MFU=" in s

def test_mfu_accountant_is_above_target():
    acct = MFUAccountant(peak_tflops=0.000001, mfu_target=0.0)
    acct.configure(flops_per_token=1_000_000_000)
    acct.start_step()
    time.sleep(0.001)
    acct.end_step(tokens=1024)
    # With 1 TFLOP/token and 0.000001 TFLOP peak, MFU >> 1.0, clamped to 1.0
    # which is always above target=0.0
    assert acct.is_above_target()

def test_mfu_accountant_history_accumulates():
    acct = MFUAccountant(peak_tflops=989.0)
    acct.configure(flops_per_token=6_000)
    n = 7
    for _ in range(n):
        acct.start_step()
        time.sleep(0.001)
        acct.end_step(tokens=128)
    assert len(acct.history) == n

# ---------------------------------------------------------------------------
# compute_moe_flops backward compatibility
# ---------------------------------------------------------------------------

def test_compute_moe_flops_positive():
    flops = compute_moe_flops(
        hidden_dim=4096,
        num_layers=32,
        ffn_dim=14336,
        num_experts=64,
        top_k=2,
        seq_length=4096,
        batch_tokens=4096 * 4,
        vocab_size=128256,
    )
    assert flops > 0
    # Should be on the order of 10^15 (petaflops) for this scale
    assert flops > 1e12

@pytest.mark.parametrize("top_k,num_experts", [(1, 8), (2, 64), (4, 128)])
def test_compute_moe_flops_scales_with_k(top_k, num_experts):
    """More active experts per token → more FLOPs."""
    base = dict(
        hidden_dim=1024, num_layers=8, ffn_dim=4096,
        num_experts=num_experts, seq_length=512, batch_tokens=512,
    )
    f1 = compute_moe_flops(**base, top_k=1)
    fk = compute_moe_flops(**base, top_k=top_k)
    if top_k > 1:
        assert fk > f1

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
