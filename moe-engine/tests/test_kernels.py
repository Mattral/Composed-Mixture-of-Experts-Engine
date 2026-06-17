"""
tests/test_kernels.py
=====================

Validates the moe_router kernel against a double-precision PyTorch
autograd reference. Asserts:

  * Forward tolerance:   atol < 1e-5, rtol < 1e-5  on weights & probs
  * Backward tolerance:  atol < 1e-5, rtol < 1e-5  via gradcheck-style
                          comparison of grad_tokens and grad_gate_w.
  * Token conservation:  sum(dispatch_cnt) == N * K
                          unique tokens (rows of idx) is N
                          no -1 / NaN entries
"""

from __future__ import annotations

import math

import pytest
import torch

from pkg.kernels.moe_router import (
    MoERouter,
    moe_topk_route,
    MoERouterFunction,
    _reference_route_fp64,
)


@pytest.mark.parametrize("B,S,H,E,K", [
    (2, 16, 64, 8, 2),
    (1, 32, 128, 16, 1),
    (4, 8, 64, 32, 4),
])
def test_forward_tolerance(B, S, H, E, K):
    torch.manual_seed(0)
    tokens = torch.randn(B * S, H, dtype=torch.float64)
    gate_w = torch.randn(H, E, dtype=torch.float64) * (1.0 / math.sqrt(H))

    # CPU path -- always uses reference fp64.
    idx, w = moe_topk_route(tokens.float(), gate_w.float(), k=K, force_reference=True)
    ref_idx, ref_w, _ = _reference_route_fp64(tokens, gate_w, k=K)

    assert idx.shape == ref_idx.shape == (B * S, K)
    assert w.shape == ref_w.shape == (B * S, K)
    assert torch.equal(idx.cpu(), ref_idx.cpu())

    assert torch.allclose(
        w.double().cpu(), ref_w.double().cpu(), atol=1e-5, rtol=1e-5,
    ), f"forward weight tolerance violated, max_diff={(w.double()-ref_w.double()).abs().max()}"


@pytest.mark.parametrize("B,S,H,E,K", [
    (2, 8, 32, 8, 2),
    (1, 16, 64, 16, 2),
])
def test_backward_tolerance(B, S, H, E, K):
    """Manual analytical backward vs torch.autograd through the reference path."""
    torch.manual_seed(7)
    N = B * S
    tokens = torch.randn(N, H, dtype=torch.float64, requires_grad=True)
    gate_w = (torch.randn(H, E, dtype=torch.float64) / math.sqrt(H)).requires_grad_(True)

    # ------- autograd reference: rerun the math inline & let PyTorch diff it.
    def _ref(tk, gw):
        logits = tk @ gw
        probs = torch.softmax(logits, dim=-1)
        vals, idx = torch.topk(probs, k=K, dim=-1)
        denom = vals.sum(-1, keepdim=True).clamp_min(1e-30)
        return vals / denom, idx

    w_ref, _ = _ref(tokens, gate_w)
    grad_w = torch.randn_like(w_ref)
    (w_ref * grad_w).sum().backward()
    ref_grad_tokens = tokens.grad.detach().clone()
    ref_grad_gate = gate_w.grad.detach().clone()

    # ------- analytical backward through MoERouterAutograd
    tokens2 = tokens.detach().float().requires_grad_(True)
    gate2 = gate_w.detach().float().requires_grad_(True)
    idx, w = MoERouterFunction.apply(tokens2, gate2, K, True)
    (w * grad_w.float()).sum().backward()

    assert torch.allclose(
        tokens2.grad.double(), ref_grad_tokens, atol=1e-5, rtol=1e-5,
    ), f"grad_tokens diff {(tokens2.grad.double()-ref_grad_tokens).abs().max()}"
    assert torch.allclose(
        gate2.grad.double(), ref_grad_gate, atol=1e-5, rtol=1e-5,
    ), f"grad_gate diff {(gate2.grad.double()-ref_grad_gate).abs().max()}"


@pytest.mark.parametrize("B,S,H,E,K", [
    (2, 32, 64, 8, 2),
    (3, 16, 64, 32, 4),
])
def test_token_conservation(B, S, H, E, K):
    """Every token is dispatched exactly K times; no drops, no duplicates."""
    torch.manual_seed(1)
    N = B * S
    tokens = torch.randn(N, H)
    router = MoERouter(hidden_dim=H, num_experts=E, top_k=K)
    idx, w, cnt = router(tokens)

    assert idx.shape == (N, K)
    assert cnt.shape == (E,)
    assert int(cnt.sum().item()) == N * K, "token conservation broken: total dispatch mismatch"
    assert (idx >= 0).all() and (idx < E).all(), "out-of-range expert id detected"
    assert not torch.isnan(w).any(), "NaN combine weight detected"
    # Each token's K slots must be K *distinct* experts (no duplicate slot).
    sorted_idx, _ = torch.sort(idx, dim=-1)
    diffs = sorted_idx[:, 1:] - sorted_idx[:, :-1]
    assert (diffs > 0).all() if K > 1 else True, "duplicate expert assignment within a token"


def test_combine_weights_sum_to_one():
    torch.manual_seed(2)
    tokens = torch.randn(64, 32)
    router = MoERouter(hidden_dim=32, num_experts=16, top_k=2)
    _, w, _ = router(tokens)
    sums = w.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


def test_router_profile_populated():
    torch.manual_seed(3)
    tokens = torch.randn(32, 16)
    router = MoERouter(hidden_dim=16, num_experts=8, top_k=2)
    router(tokens)
    prof = router.last_profile
    assert prof is not None
    assert prof.sram_bytes_per_block > 0
    assert prof.tokens_per_expert_mean > 0


def test_triton_kernels_declare_k_as_constexpr():
    """Regression test (v0.3.2): both Triton kernels must declare ``K`` as
    ``tl.constexpr``, not as a plain runtime argument.

    The kernels use ``tl.static_range(0, K)`` as a loop bound. Triton's
    ``static_range`` requires a compile-time constant; passing a runtime
    ``int32`` value raises
    ``AssertionError('int32[] used as tl.static_range end value is not
    a constexpr')`` at kernel compile time — but only on an actual GPU with
    Triton installed. Every test in this suite runs the CPU fp64 reference
    path (`_triton_forward` is only reachable when
    ``TRITON_AVAILABLE and tokens.is_cuda``), so this bug shipped in v0.2/v0.3
    completely undetected by CI and was only caught on a real T4 GPU run.

    This test inspects the kernel source directly (via `inspect.getsource`)
    so it can catch a regression on CPU-only CI, the same way
    `test_row_parallel_uses_all_reduce_not_reduce_scatter` in
    `test_tensor_parallel.py` guards the RowParallel collective bug.
    """
    import inspect
    from pkg.kernels.moe_router import TRITON_AVAILABLE

    if not TRITON_AVAILABLE:
        pytest.skip("Triton not installed; kernel source unavailable to inspect")

    import pkg.kernels.moe_router as mod

    for kernel_name in ("_router_fwd_kernel", "_router_bwd_kernel"):
        # triton.jit wraps the function; the underlying source is on .fn
        kernel = getattr(mod, kernel_name)
        src = inspect.getsource(kernel.fn if hasattr(kernel, "fn") else kernel)
        assert "K: tl.constexpr" in src, (
            f"{kernel_name} must declare 'K: tl.constexpr' in its signature — "
            f"found a plain 'K' argument, which breaks tl.static_range(0, K) "
            f"on every real GPU invocation (v0.3.2 regression)"
        )


def test_triton_kernel_source_declares_k_as_constexpr():
    """Regression test (v0.3.2), Triton-independent variant.

    Unlike `test_triton_kernels_declare_k_as_constexpr`, this test reads the
    raw source file directly and does not require Triton to be installed —
    it therefore runs unconditionally on CPU-only CI, which is exactly the
    environment that let the original bug ship undetected through v0.2 and
    v0.3. See that test's docstring for the full root-cause explanation.
    """
    import pkg.kernels.moe_router as mod
    src_path = mod.__file__
    with open(src_path, encoding="utf-8") as f:
        src = f.read()

    # Both kernel signatures must declare K as tl.constexpr.
    # Count only the actual parameter declarations (indented, trailing comma),
    # not the explanatory mention in the module docstring above.
    sig_count = src.count("        K: tl.constexpr,\n")
    assert sig_count == 2, (
        f"Expected both _router_fwd_kernel and _router_bwd_kernel to declare "
        f"'K: tl.constexpr,' as a parameter — found {sig_count} occurrence(s). "
        f"If K is reintroduced as a plain runtime argument, "
        f"tl.static_range(0, K) will fail on every real GPU invocation with "
        f"\"AssertionError('int32[] used as tl.static_range end value is not "
        f"a constexpr')\" (v0.3.2 regression)."
    )
    # The bare positional 'K,' pattern (the buggy v0.2/v0.3 form) must be gone
    # from both kernel parameter lists.
    assert "N, H, E, K,\n" not in src, (
        "_router_fwd_kernel signature still declares K as a plain positional "
        "argument (the v0.3.2 bug pattern)"
    )
    assert "N, E, K,\n" not in src, (
        "_router_bwd_kernel signature still declares K as a plain positional "
        "argument (the v0.3.2 bug pattern)"
    )

