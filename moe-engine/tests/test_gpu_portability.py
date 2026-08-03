"""
tests/test_gpu_portability.py
==============================

Permanent pytest coverage for the three A100/Ampere portability fixes
described in ``docs/adr/ADR-005-gpu-architecture-portability.md``.

Context
-------
These three fixes were originally validated only by an ad-hoc, uncommitted
regression suite run in isolation (10/10 passing, per the project handoff),
plus a direct demonstration in Section 8 of
``notebooks/moe_engine_A100_validation.ipynb``. They had no permanent,
CI-enforced pytest coverage. This file closes that gap.

Two of the three fixes involve real-hardware-only behavior (Triton dispatch
eligibility, cross-device tensor placement) that cannot be exercised
end-to-end without a CUDA GPU -- and, for the device-mismatch case, two
distinct CUDA devices in one job. Rather than skip coverage entirely on
CPU-only CI -- which is exactly the blind spot that let the original
``K: tl.constexpr`` bug (ADR-001) and the Pydantic silent-fallback bug
(ADR-003) each ship undetected for a full release cycle -- each fix gets
layered coverage:

  1. A CPU-runnable test of the actual decision/branch logic (via a
     lightweight duck-typed stand-in for a CUDA tensor, or via
     ``inspect.getsource``), which runs on every commit and catches a
     regression in the *logic* even though it cannot prove real kernel or
     device behavior.
  2. Where applicable, a ``@pytest.mark.gpu`` test that exercises the real
     thing on actual hardware, honestly skipped when that hardware is not
     present -- following the precedent already set by
     ``test_backward_gradients_randomized_triton`` in
     ``tests/test_kernels_numerics.py``.

Nothing in this file claims proof it cannot deliver. Where CPU-only CI
cannot exercise the real failure mode, the docstring says so explicitly.

Provenance note (ADR-006)
--------------------------
Writing this file's device-placement tests is what surfaced a real gap:
ADR-005's "Fix 2" describes the device coercion as living inside
``_reference_route_fp64`` itself, but as delivered it only existed in an
unrelated block (``MoERouter.forward``'s z-loss/load-imbalance telemetry
computation). ``_reference_route_fp64`` -- the function actually invoked on
the routing hot path whenever the reference path is taken -- had no device
coercion at all. See ``docs/adr/ADR-006-device-coercion-correction.md`` for
the full writeup; the fix now lives in the right place, and
``test_reference_route_fp64_source_coerces_gate_w_device`` below exists so
this cannot silently regress again.
"""

from __future__ import annotations

import inspect

import pytest
import torch

from pkg.kernels.moe_router import (
    MIN_TRITON_DIM,
    TRITON_AVAILABLE,
    MoERouter,
    MoERouterFunction,
    _reference_route_fp64,
    _should_use_triton,
)

pytestmark = pytest.mark.cpu


# ===========================================================================
# Helpers
# ===========================================================================
class _FakeTensor:
    """Minimal duck-typed stand-in for a tensor's ``.shape``/``.is_cuda``.

    ``_should_use_triton`` only ever reads these two attributes off its
    ``tokens``/``gate_w`` arguments -- it never indexes into or computes on
    the tensor itself (confirm this by re-reading its source before
    changing this helper). That means the full N/H/E x ``MIN_TRITON_DIM``
    boundary logic can be exercised deterministically on CPU-only CI
    without needing real CUDA, the same guarantee the rest of the Tier-0
    suite already relies on elsewhere. This does NOT exercise real Triton
    kernel compilation or dispatch -- that requires actual GPU hardware,
    see the ``gpu``-marked tests below and ADR-005.
    """

    def __init__(self, shape: tuple[int, int], is_cuda: bool = True):
        self.shape = shape
        self.is_cuda = is_cuda


# ===========================================================================
# Fix 1 -- MIN_TRITON_DIM dimension guard (ADR-005 Problem/Fix 1)
# ===========================================================================
class TestMinTritonDimGuard:
    def test_min_triton_dim_is_16(self):
        """Documented contract from ADR-005: threshold is exactly 16."""
        assert MIN_TRITON_DIM == 16

    @pytest.mark.parametrize(
        "N,H,E",
        [
            (16, 16, 16),  # exactly at threshold on all three dims
            (64, 64, 64),  # comfortably above
            (1024, 512, 64),
        ],
    )
    def test_should_use_triton_true_when_all_dims_at_or_above_threshold(self, N, H, E):
        tokens = _FakeTensor((N, H), is_cuda=True)
        gate_w = _FakeTensor((H, E), is_cuda=True)
        # Equals TRITON_AVAILABLE (not unconditionally True) so this test is
        # correct in both a Triton-installed and a Triton-less CI runner.
        assert _should_use_triton(tokens, gate_w, force_reference=False) == TRITON_AVAILABLE

    @pytest.mark.parametrize(
        "N,H,E",
        [
            (4, 64, 64),  # configs/smoke.yaml's actual failure case: undersized N
            (64, 4, 64),  # undersized H
            (64, 64, 4),  # undersized E -- the real configs/smoke.yaml num_experts=4 case
            (15, 16, 16),  # one below threshold on N only -- boundary check
        ],
    )
    def test_should_use_triton_false_when_any_dim_below_threshold(self, N, H, E):
        """Below MIN_TRITON_DIM on ANY of N, H, E -- reference path, always,
        regardless of GPU generation or Triton availability. This is the
        exact ``configs/smoke.yaml`` (``num_experts=4``) shape that failed
        Triton compilation on A100 -- see ADR-005 Problem 1."""
        tokens = _FakeTensor((N, H), is_cuda=True)
        gate_w = _FakeTensor((H, E), is_cuda=True)
        assert not _should_use_triton(tokens, gate_w, force_reference=False)

    def test_should_use_triton_false_when_force_reference(self):
        tokens = _FakeTensor((64, 64), is_cuda=True)
        gate_w = _FakeTensor((64, 64), is_cuda=True)
        assert not _should_use_triton(tokens, gate_w, force_reference=True)

    def test_should_use_triton_false_on_non_cuda_tensors_regardless_of_dims(self):
        tokens = _FakeTensor((1024, 512), is_cuda=False)
        gate_w = _FakeTensor((512, 64), is_cuda=False)
        assert not _should_use_triton(tokens, gate_w, force_reference=False)

    def test_undersized_config_still_routes_correctly_on_cpu(self):
        """End-to-end (not just the guard function): a smoke-scale E=4
        config must still produce correct, conserved routing on CPU -- the
        guard changes *which path* runs, never routing correctness."""
        torch.manual_seed(0)
        router = MoERouter(hidden_dim=16, num_experts=4, top_k=2)
        tokens = torch.randn(8, 16)
        idx, w, dispatch_cnt = router(tokens)
        assert idx.shape == (8, 2)
        assert int(dispatch_cnt.sum().item()) == 8 * 2

    @pytest.mark.gpu
    def test_should_use_triton_real_cuda_dispatch(self):
        """The real thing: on an actual GPU, an undersized config must
        dispatch to the reference path and a comfortably-sized one must
        dispatch to Triton. Skipped without CUDA -- see ADR-005's support
        matrix for which architectures this has actually been run on."""
        if not (TRITON_AVAILABLE and torch.cuda.is_available()):
            pytest.skip("Requires a real CUDA GPU with Triton installed")
        small = torch.randn(4, 4, device="cuda")
        small_gate = torch.randn(4, 4, device="cuda")
        assert not _should_use_triton(small, small_gate, force_reference=False)
        big = torch.randn(64, 64, device="cuda")
        big_gate = torch.randn(64, 64, device="cuda")
        assert _should_use_triton(big, big_gate, force_reference=False)


# ===========================================================================
# Fix 2 -- device placement in the fp64 reference path (ADR-005 Problem/Fix 2)
# ===========================================================================
class TestReferencePathDevicePlacement:
    def test_reference_route_fp64_source_coerces_gate_w_device(self):
        """Regression guard, CPU-only CI (source-inspection technique,
        following the precedent set by
        ``test_triton_kernel_source_declares_k_as_constexpr`` in
        ``test_kernels.py``): ``_reference_route_fp64`` must explicitly
        move ``gate_w`` onto ``tokens``'s device before computing logits.

        See this file's module docstring and ADR-006 for why this test
        exists: the fix was originally delivered in the wrong function
        entirely, and this is the regression guard that would have caught
        it.
        """
        src = inspect.getsource(_reference_route_fp64)
        assert "gate_w.to(device=" in src, (
            "_reference_route_fp64 must explicitly coerce gate_w onto "
            "tokens's device before the matmul -- without this, a "
            "cross-device call raises a device-mismatch RuntimeError "
            "instead of the documented graceful fallback. "
            "See ADR-005 Fix 2 and ADR-006."
        )

    def test_reference_route_fp64_same_device_regression(self):
        """Baseline: normal same-device (CPU) calls are unaffected by the
        device-coercion fix -- shapes and the renormalization invariant
        still hold."""
        torch.manual_seed(0)
        tokens = torch.randn(32, 16, dtype=torch.float64)
        gate_w = torch.randn(16, 8, dtype=torch.float64)
        idx, w, logits = _reference_route_fp64(tokens, gate_w, k=2)
        assert idx.shape == (32, 2)
        assert w.shape == (32, 2)
        assert torch.allclose(w.sum(dim=-1), torch.ones(32, dtype=w.dtype), atol=1e-6)

    def test_reference_route_fp64_preserves_original_dtype(self):
        """Coercing gate_w's *device* must not change the *dtype* contract:
        output weights still come back in tokens' original dtype, not
        float64, regardless of what dtype gate_w started in."""
        torch.manual_seed(0)
        tokens = torch.randn(16, 8, dtype=torch.float32)
        gate_w = torch.randn(8, 4, dtype=torch.float32)
        _, w, logits = _reference_route_fp64(tokens, gate_w, k=2)
        assert w.dtype == torch.float32
        assert logits.dtype == torch.float32

    @pytest.mark.gpu
    def test_reference_route_fp64_cross_device_real_hardware(self):
        """The real thing: tokens and gate_w on two genuinely different
        CUDA devices must not crash. Needs 2+ GPUs in one job -- see
        Handoff Section 7.3, the actual next-phase blocker for this
        project. A single GPU, however large, cannot exercise this test."""
        if torch.cuda.device_count() < 2:
            pytest.skip("Requires 2+ CUDA devices to construct a genuine cross-device call")
        tokens = torch.randn(32, 16, device="cuda:0")
        gate_w = torch.randn(16, 8, device="cuda:1")
        idx, w, logits = _reference_route_fp64(tokens, gate_w, k=2)
        assert idx.device == tokens.device


# ===========================================================================
# Fix 3 -- single source of truth for used_triton (ADR-005 Problem/Fix 3)
# ===========================================================================
class TestUsedTritonSingleSourceOfTruth:
    def test_forward_signature_has_report_parameter(self):
        sig = inspect.signature(MoERouterFunction.forward)
        assert "report" in sig.parameters, (
            "MoERouterFunction.forward must accept a `report` side-channel "
            "parameter -- without it, MoERouter.forward's telemetry has no "
            "way to read back the *actual* used_triton outcome, and risks "
            "recomputing an independent, driftable copy of the same "
            "decision (the original ADR-005 Problem 3 bug)."
        )

    def test_backward_returns_five_values(self):
        """PyTorch's autograd checks gradient-count strictly against
        forward()'s input count. forward() takes 5 non-ctx inputs (tokens,
        gate_w, k, force_reference, report) since the report side-channel
        was added -- backward() must return a matching 5-tuple (None for
        the three non-differentiable ones), or autograd raises at runtime.
        This exact off-by-one was introduced and caught during the A100
        fix's original development -- this test exists so it cannot
        silently regress."""
        torch.manual_seed(0)
        tokens = torch.randn(8, 16, requires_grad=True)
        gate_w = torch.randn(16, 4, requires_grad=True)
        idx, w = MoERouterFunction.apply(tokens, gate_w, 2, False, None)
        w.sum().backward()  # raises "incorrect number of gradients" if backward() drifts
        assert tokens.grad is not None
        assert gate_w.grad is not None

    def test_report_dict_populated_and_matches_should_use_triton(self):
        """The report dict -- not a second, independent computation -- is
        the single source MoERouter.forward reads `used_triton` from. On
        CPU this always resolves to False, but the test still proves the
        plumbing (report mutated in place, read back correctly) is wired
        end-to-end, which is what the original bug (ADR-005 Problem 3)
        broke."""
        torch.manual_seed(0)
        report: dict = {}
        tokens = torch.randn(8, 16)
        gate_w = torch.randn(16, 4)
        MoERouterFunction.apply(tokens, gate_w, 2, False, report)
        assert "used_triton" in report
        assert report["used_triton"] == _should_use_triton(tokens, gate_w, False)

    @pytest.mark.parametrize("force_reference", [True, False])
    def test_router_profile_used_triton_matches_should_use_triton(self, force_reference):
        """End-to-end via the public MoERouter module: `last_profile.used_triton`
        must always agree with an independently-computed `_should_use_triton`
        call on the same inputs -- the exact invariant Problem 3 in ADR-005
        violated (the profiling block used to compute an incomplete,
        independently-derived condition that ignored force_reference and
        the dimension guard)."""
        torch.manual_seed(0)
        router = MoERouter(hidden_dim=16, num_experts=4, top_k=2)
        tokens = torch.randn(8, 16)
        router(tokens, force_reference=force_reference)
        expected = _should_use_triton(tokens, router.gate_w, force_reference)
        assert router.last_profile.used_triton == expected

    def test_no_duplicate_triton_kernel_launch_in_profiling_block(self):
        """Regression guard (source-inspection, CPU-runnable): the
        profiling block in MoERouter.forward must not call
        `_triton_forward` a second time purely to re-derive telemetry --
        that was the redundant-launch-plus-silent-except-pass bug fixed
        alongside Problem 3 (ADR-005)."""
        import pkg.kernels.moe_router as mod

        src = inspect.getsource(mod.MoERouter.forward)
        # Strip full-line comments first. The function's source deliberately
        # *mentions* "_triton_forward(...)" inside a `#`-comment explaining
        # that the duplicate launch was removed -- a naive substring search
        # over the raw source matches that prose and produces a false
        # positive, which would itself be exactly the "looks like it's
        # checking but isn't" failure mode this file exists to guard
        # against (see this project's Section 3.1 design philosophy).
        code_only = "\n".join(line for line in src.splitlines() if not line.strip().startswith("#"))
        assert "_triton_forward(" not in code_only, (
            "MoERouter.forward should read kernel_ms/sram_bytes/achieved_bw "
            "from the `report` dict populated by the single real Triton "
            "launch inside MoERouterFunction.forward, not call "
            "_triton_forward a second time on the profiling path."
        )
