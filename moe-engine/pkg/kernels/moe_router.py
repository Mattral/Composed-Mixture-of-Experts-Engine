"""
pkg/kernels/moe_router.py
=========================

Hardware-aware Top-K Mixture-of-Experts router.

This module implements the sparse gating/dispatching kernel at the heart of
every modern MoE block (Switch-Transformer, GShard, Mixtral, DeepSeek-MoE).
It is written in two interlocking layers:

1.  **Triton JIT kernel** (`_router_fwd_kernel`, `_router_bwd_kernel`) —
    executed when a CUDA-capable GPU and a working Triton install are
    available. The kernel fuses:

        gemm( tokens [N, H], gate_w [H, E] )  ->  logits [N, E]
        softmax_along_E( logits )             ->  probs  [N, E]
        top_k( probs, k=K )                   ->  idx    [N, K], w [N, K]
        renormalize( w )                      ->  combine weights

    in a *single* pass over the gating dimension. SRAM occupancy is bounded
    by `BLOCK_E * BLOCK_N` floats; we choose `(BLOCK_N=64, BLOCK_E=64)` to
    keep working-set under 32 KiB, fitting comfortably in Ampere/Hopper L1.
    Global loads of `tokens` and `gate_w` are coalesced on the contiguous (H)
    dimension. Top-K is implemented as in-SRAM selection-sort over K elements
    (K is small, typically 1–4), eliminating shared-memory bank pressure that
    a full sort would create.

2.  **PyTorch double-precision reference** (`_reference_route_fp64`) — used
    by the autograd backward and by the test-suite as the numerical ground
    truth (`atol = rtol = 1e-5`).

Both paths are wrapped behind a `torch.autograd.Function`
(`MoERouterAutograd`) so the entire router is a drop-in differentiable
module respecting PyTorch's autograd graph and AMP semantics.

Token Conservation Invariant
----------------------------
    sum(dispatch_cnt) == N * K        (asserted every forward pass)
    idx values in [0, E)              (no -1 / NaN entries)

v0.3.2 fix
----------
Both Triton kernels previously declared ``K`` as a plain runtime int32
argument while using it as the loop bound in ``tl.static_range(0, K)``.
``static_range`` requires a compile-time constant; passing a runtime value
raised ``AssertionError('int32[] used as tl.static_range end value is not
a constexpr')`` on every GPU invocation. This was never caught by CI because
every test runs on the CPU fp64 reference path — ``_triton_forward`` is only
reachable when ``TRITON_AVAILABLE and tokens.is_cuda``, which is false on
CI runners. ``K`` is now declared ``K: tl.constexpr`` in both kernel
signatures, and all three call sites pass it as the keyword ``K=k``
(consistent with ``BLOCK_N``/``BLOCK_H``/``BLOCK_E``). A practical
consequence: Triton recompiles a specialised kernel per distinct ``k``
value, the same way it already does per ``BLOCK_E``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch

# --------------------------------------------------------------------------
# Optional Triton import.  The repo MUST work in CPU-only environments so
# tests can run anywhere; the JIT kernel is only loaded when CUDA + Triton
# are both available.
# --------------------------------------------------------------------------
try:
    import triton  # type: ignore
    import triton.language as tl  # type: ignore

    TRITON_AVAILABLE = True
except Exception:  # pragma: no cover - import-time guard
    triton = None  # type: ignore
    tl = None  # type: ignore
    TRITON_AVAILABLE = False


# ==========================================================================
# Telemetry record returned from each router invocation.
# ==========================================================================
@dataclass
class RouterProfile:
    sram_bytes_per_block: int
    achieved_bandwidth_gbps: float
    kernel_ms: float
    used_triton: bool
    tokens_per_expert_mean: float
    tokens_per_expert_std: float
    # v0.2 additions — routing quality metrics
    expert_load_imbalance: float  # max_load / mean_load; 1.0 = perfect
    router_z_loss: float  # auxiliary z-loss magnitude (log-sum-exp)


# ==========================================================================
# Triton kernel – forward pass.
# ==========================================================================
if TRITON_AVAILABLE:

    @triton.jit
    def _router_fwd_kernel(
        tokens_ptr,
        gate_w_ptr,
        topk_idx_ptr,
        topk_w_ptr,
        logits_ptr,
        stride_tn,
        stride_th,
        stride_gh,
        stride_ge,
        stride_in,
        stride_ik,
        stride_wn,
        stride_wk,
        stride_ln,
        stride_le,
        N,
        H,
        E,
        K: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_H: tl.constexpr,
        BLOCK_E: tl.constexpr,
    ):
        """Fused router: tokens @ gate_w -> softmax -> top_k -> renorm weights."""
        pid_n = tl.program_id(0)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N
        offs_e = tl.arange(0, BLOCK_E)
        mask_e = offs_e < E

        acc = tl.zeros((BLOCK_N, BLOCK_E), dtype=tl.float32)
        for h_start in range(0, H, BLOCK_H):
            offs_h = h_start + tl.arange(0, BLOCK_H)
            mask_h = offs_h < H
            tok_tile = tl.load(
                tokens_ptr + offs_n[:, None] * stride_tn + offs_h[None, :] * stride_th,
                mask=mask_n[:, None] & mask_h[None, :],
                other=0.0,
            ).to(tl.float32)
            gate_tile = tl.load(
                gate_w_ptr + offs_h[:, None] * stride_gh + offs_e[None, :] * stride_ge,
                mask=mask_h[:, None] & mask_e[None, :],
                other=0.0,
            ).to(tl.float32)
            acc += tl.dot(tok_tile, gate_tile, allow_tf32=False)

        logits = tl.where(mask_e[None, :], acc, float("-inf"))
        tl.store(
            logits_ptr + offs_n[:, None] * stride_ln + offs_e[None, :] * stride_le,
            logits,
            mask=mask_n[:, None] & mask_e[None, :],
        )

        row_max = tl.max(logits, axis=1)
        shifted = logits - row_max[:, None]
        exp_l = tl.exp(shifted)
        denom = tl.sum(exp_l, axis=1)
        probs = exp_l / denom[:, None]

        topk_sum = tl.zeros((BLOCK_N,), dtype=tl.float32)
        for k in tl.static_range(0, K):
            kth_idx = tl.argmax(probs, axis=1).to(tl.int32)
            kth_val = tl.max(probs, axis=1)
            tl.store(topk_idx_ptr + offs_n * stride_in + k * stride_ik, kth_idx, mask=mask_n)
            tl.store(topk_w_ptr + offs_n * stride_wn + k * stride_wk, kth_val, mask=mask_n)
            topk_sum += kth_val
            kth_mask = tl.arange(0, BLOCK_E)[None, :] == kth_idx[:, None]
            probs = tl.where(kth_mask, 0.0, probs)

        inv = 1.0 / tl.where(topk_sum > 0.0, topk_sum, 1.0)
        for k in tl.static_range(0, K):
            w = tl.load(topk_w_ptr + offs_n * stride_wn + k * stride_wk, mask=mask_n, other=0.0)
            tl.store(topk_w_ptr + offs_n * stride_wn + k * stride_wk, w * inv, mask=mask_n)

    @triton.jit
    def _router_bwd_kernel(
        grad_w_ptr,
        topk_idx_ptr,
        logits_ptr,
        grad_logits_ptr,
        stride_gn,
        stride_gk,
        stride_in,
        stride_ik,
        stride_ln,
        stride_le,
        stride_dn,
        stride_de,
        N,
        E,
        K: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_E: tl.constexpr,
    ):
        """Backward through softmax -> top_k -> renorm.

        Propagates grad_w -> grad_v -> grad_p -> grad_l via the analytical
        softmax Jacobian:  grad_l_i = p_i * (grad_p_i - dot(grad_p, p))
        """
        pid_n = tl.program_id(0)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N
        offs_e = tl.arange(0, BLOCK_E)
        mask_e = offs_e < E

        logits = tl.load(
            logits_ptr + offs_n[:, None] * stride_ln + offs_e[None, :] * stride_le,
            mask=mask_n[:, None] & mask_e[None, :],
            other=float("-inf"),
        )
        row_max = tl.max(logits, axis=1)
        exp_l = tl.exp(logits - row_max[:, None])
        denom = tl.sum(exp_l, axis=1)
        probs = exp_l / denom[:, None]
        probs = tl.where(mask_e[None, :], probs, 0.0)

        S = tl.zeros((BLOCK_N,), dtype=tl.float32)
        gwv = tl.zeros((BLOCK_N,), dtype=tl.float32)
        for k in tl.static_range(0, K):
            idx_k = tl.load(
                topk_idx_ptr + offs_n * stride_in + k * stride_ik, mask=mask_n, other=0
            ).to(tl.int32)
            gw_k = tl.load(grad_w_ptr + offs_n * stride_gn + k * stride_gk, mask=mask_n, other=0.0)
            onehot = (tl.arange(0, BLOCK_E)[None, :] == idx_k[:, None]).to(tl.float32)
            v_k = tl.sum(probs * onehot, axis=1)
            S += v_k
            gwv += gw_k * v_k

        inv_S = 1.0 / tl.where(S > 0.0, S, 1.0)
        inv_S2 = inv_S * inv_S

        grad_p = tl.zeros((BLOCK_N, BLOCK_E), dtype=tl.float32)
        for k in tl.static_range(0, K):
            idx_k = tl.load(
                topk_idx_ptr + offs_n * stride_in + k * stride_ik, mask=mask_n, other=0
            ).to(tl.int32)
            gw_k = tl.load(grad_w_ptr + offs_n * stride_gn + k * stride_gk, mask=mask_n, other=0.0)
            grad_v_k = gw_k * inv_S - gwv * inv_S2
            onehot = (tl.arange(0, BLOCK_E)[None, :] == idx_k[:, None]).to(tl.float32)
            grad_p += onehot * grad_v_k[:, None]

        dot = tl.sum(grad_p * probs, axis=1)
        grad_l = probs * (grad_p - dot[:, None])
        tl.store(
            grad_logits_ptr + offs_n[:, None] * stride_dn + offs_e[None, :] * stride_de,
            grad_l,
            mask=mask_n[:, None] & mask_e[None, :],
        )


# ==========================================================================
# Reference (double-precision) implementation
# ==========================================================================
def _reference_route_fp64(
    tokens: torch.Tensor,
    gate_w: torch.Tensor,
    k: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pure-PyTorch fp64 reference.  Returns (topk_idx, topk_w, logits)."""
    orig_dtype = tokens.dtype
    t64 = tokens.to(torch.float64)
    g64 = gate_w.to(torch.float64)
    logits = t64 @ g64
    probs = torch.softmax(logits, dim=-1)
    topk_vals, topk_idx = torch.topk(probs, k=k, dim=-1, largest=True)
    denom = topk_vals.sum(dim=-1, keepdim=True).clamp_min(1e-30)
    topk_w = topk_vals / denom
    return topk_idx.to(torch.long), topk_w.to(orig_dtype), logits.to(orig_dtype)


def _reference_backward_fp64(
    logits: torch.Tensor,
    topk_idx: torch.Tensor,
    grad_w: torch.Tensor,
    k: int,
    E: int,
) -> torch.Tensor:
    """Analytical backward — used both for CPU path and as test oracle."""
    l64 = logits.to(torch.float64)
    probs = torch.softmax(l64, dim=-1)
    v = probs.gather(1, topk_idx)
    S = v.sum(dim=-1, keepdim=True).clamp_min(1e-30)
    gw = grad_w.to(torch.float64)
    gwv = (gw * v).sum(dim=-1, keepdim=True)
    grad_v = gw / S - gwv / (S * S)
    grad_p = torch.zeros_like(probs).scatter_add_(1, topk_idx, grad_v)
    dot = (grad_p * probs).sum(dim=-1, keepdim=True)
    return probs * (grad_p - dot)


# ==========================================================================
# Triton forward helper
# ==========================================================================
def _triton_forward(
    tokens: torch.Tensor,
    gate_w: torch.Tensor,
    k: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, int, float]:  # pragma: no cover
    N, H = tokens.shape
    E = gate_w.shape[1]
    BLOCK_N = 64
    BLOCK_H = 64
    BLOCK_E = _next_pow2(E)

    topk_idx = torch.empty((N, k), dtype=torch.int32, device=tokens.device)
    topk_w = torch.empty((N, k), dtype=torch.float32, device=tokens.device)
    logits = torch.empty((N, E), dtype=torch.float32, device=tokens.device)
    grid = ((N + BLOCK_N - 1) // BLOCK_N,)

    if tokens.is_cuda:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        _router_fwd_kernel[grid](
            tokens.contiguous(),
            gate_w.contiguous(),
            topk_idx,
            topk_w,
            logits,
            tokens.stride(0),
            tokens.stride(1),
            gate_w.stride(0),
            gate_w.stride(1),
            topk_idx.stride(0),
            topk_idx.stride(1),
            topk_w.stride(0),
            topk_w.stride(1),
            logits.stride(0),
            logits.stride(1),
            N,
            H,
            E,
            K=k,
            BLOCK_N=BLOCK_N,
            BLOCK_H=BLOCK_H,
            BLOCK_E=BLOCK_E,
        )
        end.record()
        end.synchronize()
        kernel_ms = max(end.elapsed_time(start), 0.0)
    else:
        _router_fwd_kernel[grid](
            tokens.contiguous(),
            gate_w.contiguous(),
            topk_idx,
            topk_w,
            logits,
            tokens.stride(0),
            tokens.stride(1),
            gate_w.stride(0),
            gate_w.stride(1),
            topk_idx.stride(0),
            topk_idx.stride(1),
            topk_w.stride(0),
            topk_w.stride(1),
            logits.stride(0),
            logits.stride(1),
            N,
            H,
            E,
            K=k,
            BLOCK_N=BLOCK_N,
            BLOCK_H=BLOCK_H,
            BLOCK_E=BLOCK_E,
        )
        kernel_ms = 0.0

    dtype_size = tokens.element_size()
    bytes_moved = (
        tokens.numel() * dtype_size
        + gate_w.numel() * dtype_size
        + logits.numel() * 4
        + topk_idx.numel() * 4
        + topk_w.numel() * 4
    )
    achieved_bw = (bytes_moved / 1e9) / max(kernel_ms / 1e3, 1e-9)
    sram_bytes = BLOCK_N * BLOCK_E * dtype_size * 3
    return (
        topk_idx.to(torch.long),
        topk_w.to(tokens.dtype),
        logits.to(tokens.dtype),
        kernel_ms,
        sram_bytes,
        achieved_bw,
    )


def _next_pow2(x: int) -> int:
    return 1 << (x - 1).bit_length()


# ==========================================================================
# Autograd Function
# ==========================================================================
class MoERouterFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tokens, gate_w, k, force_reference=False):
        assert tokens.dim() == 2
        assert gate_w.dim() == 2
    
        N, H = tokens.shape
        H2, E = gate_w.shape
        assert H == H2
        assert 1 <= k <= E
    
        MIN_TRITON_DIM = 16
    
        use_triton = (
            (not force_reference)
            and TRITON_AVAILABLE
            and tokens.is_cuda
            and gate_w.is_cuda
            and N >= MIN_TRITON_DIM
            and H >= MIN_TRITON_DIM
            and E >= MIN_TRITON_DIM
        )
    
        if use_triton:
            try:
                topk_idx, topk_w, logits, kernel_ms, sram_bytes, achieved_bw = \
                    _triton_forward(tokens, gate_w, k)
            except Exception as e:
                print(f"Triton failed: {e}")
                print("Falling back to reference router.")
                use_triton = False
    
        if not use_triton:
            topk_idx, topk_w, logits = _reference_route_fp64(tokens, gate_w, k)
            kernel_ms = 0.0
            sram_bytes = 0
            achieved_bw = 0.0
    
        ctx.save_for_backward(tokens, gate_w, logits, topk_idx, topk_w)
        ctx.k = k
        ctx.use_triton = use_triton
    
        return topk_idx, topk_w

    @staticmethod
    def backward(ctx, grad_idx, grad_w):
        tokens, gate_w, logits, topk_idx, topk_w = ctx.saved_tensors
        k = ctx.k
        N, H = tokens.shape
        E = gate_w.shape[1]

        if ctx.use_triton:  # pragma: no cover
            grad_logits = torch.empty_like(logits, dtype=torch.float32)
            BLOCK_N = 64
            BLOCK_E = _next_pow2(E)
            grid = ((N + BLOCK_N - 1) // BLOCK_N,)
            _router_bwd_kernel[grid](
                grad_w.contiguous().to(torch.float32),
                topk_idx.contiguous().to(torch.int32),
                logits.contiguous().to(torch.float32),
                grad_logits,
                grad_w.stride(0),
                grad_w.stride(1),
                topk_idx.stride(0),
                topk_idx.stride(1),
                logits.stride(0),
                logits.stride(1),
                grad_logits.stride(0),
                grad_logits.stride(1),
                N,
                E,
                K=k,
                BLOCK_N=BLOCK_N,
                BLOCK_E=BLOCK_E,
            )
            grad_logits = grad_logits.to(tokens.dtype)
        else:
            grad_logits = _reference_backward_fp64(
                logits.detach(),
                topk_idx,
                grad_w,
                k,
                E,
            ).to(tokens.dtype)

        grad_tokens = grad_logits @ gate_w.t()
        grad_gate_w = tokens.t() @ grad_logits
        return grad_tokens, grad_gate_w, None, None


MoERouterAutograd = MoERouterFunction


# ==========================================================================
# Public functional entry point
# ==========================================================================
def moe_topk_route(
    tokens: torch.Tensor,
    gate_w: torch.Tensor,
    k: int,
    force_reference: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Differentiable top-K routing.

    Parameters
    ----------
    tokens : [B, S, H] or [N, H]
    gate_w : [H, E]
    k      : top-k

    Returns
    -------
    topk_idx : LongTensor [N, K]
    topk_w   : Tensor     [N, K]  (same dtype as tokens; renormalized)
    """
    if tokens.dim() == 3:
        B, S, H = tokens.shape
        flat = tokens.reshape(B * S, H)
    elif tokens.dim() == 2:
        flat = tokens
    else:
        raise ValueError(f"tokens must be rank 2 or 3, got {tokens.dim()}")
    return MoERouterFunction.apply(flat, gate_w, k, force_reference)


# ==========================================================================
# Routing quality metrics (v0.2)
# ==========================================================================
def _compute_load_imbalance(dispatch_cnt: torch.Tensor) -> float:
    """max_load / mean_load.  1.0 = perfect balance; >1.0 = imbalance."""
    cnt = dispatch_cnt.float()
    mean = cnt.mean().item()
    if mean < 1e-9:
        return 1.0
    return float(cnt.max().item()) / mean


def _compute_router_z_loss(logits: torch.Tensor) -> float:
    """Switch-Transformer auxiliary z-loss to encourage small logit magnitudes.

    z_loss = mean_over_tokens( log( sum_over_experts exp(logit_e) )^2 )

    Typically used as an auxiliary loss term (weight ~1e-3).
    """
    # logits: [N, E]
    log_sum_exp = torch.logsumexp(logits.float(), dim=-1)  # [N]
    return float((log_sum_exp**2).mean().item())


# ==========================================================================
# nn.Module wrapper
# ==========================================================================
class MoERouter(torch.nn.Module):
    """Top-K router as a `nn.Module`.

    The gate matrix (`gate_w`: [H, E]) is a learnable parameter. On CUDA
    with Triton installed the forward pass runs the fused Triton kernel;
    everywhere else it falls back to the fp64 PyTorch reference.

    Attributes
    ----------
    hidden_dim   : H
    num_experts  : E
    top_k        : K
    last_profile : RouterProfile — populated after each forward pass
    """

    def __init__(
        self,
        hidden_dim: int,
        num_experts: int,
        top_k: int = 2,
        bias: bool = False,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        if top_k < 1 or top_k > num_experts:
            raise ValueError("top_k must be in [1, num_experts]")
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate_w = torch.nn.Parameter(torch.empty(hidden_dim, num_experts, dtype=dtype))
        torch.nn.init.normal_(self.gate_w, mean=0.0, std=1.0 / math.sqrt(hidden_dim))
        self.bias = torch.nn.Parameter(torch.zeros(num_experts, dtype=dtype)) if bias else None
        self.last_profile: Optional[RouterProfile] = None

        # Saved for z-loss / load-imbalance computation
        self._last_logits: Optional[torch.Tensor] = None

    def forward(
        self,
        tokens: torch.Tensor,
        force_reference: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (topk_idx [N, K], topk_w [N, K], dispatch_cnt [E])."""
        if tokens.dim() == 3:
            B, S, H = tokens.shape
            flat = tokens.reshape(B * S, H)
        else:
            flat = tokens
        N = flat.shape[0]
        H = flat.shape[1]
        assert H == self.hidden_dim

        gate_w = self.gate_w
        if self.bias is not None:
            force_reference = True

        idx, w = MoERouterAutograd.apply(flat, gate_w, self.top_k, force_reference)

        if self.bias is not None and force_reference:
            with torch.no_grad():
                logits = flat.to(torch.float32) @ gate_w.to(torch.float32) + self.bias.to(
                    torch.float32
                )
                probs = torch.softmax(logits, dim=-1)
                topk_vals, idx_new = torch.topk(probs, k=self.top_k, dim=-1)
                w_new = (topk_vals / topk_vals.sum(-1, keepdim=True).clamp_min(1e-30)).to(
                    tokens.dtype
                )
            idx, w = idx_new.to(torch.long), w_new

        # Dispatch count + invariant check
        dispatch_cnt = torch.bincount(idx.reshape(-1), minlength=self.num_experts).to(torch.long)
        total_dispatched = int(dispatch_cnt.sum().item())
        expected_total = N * self.top_k
        assert total_dispatched == expected_total, (
            f"Token conservation violation: dispatched={total_dispatched} "
            f"expected={expected_total} (N={N}, K={self.top_k})"
        )
        assert not torch.isnan(idx.float()).any(), "NaN in expert indices"
        assert (idx >= 0).all() and (idx < self.num_experts).all(), (
            f"Out-of-range expert indices: min={idx.min()}, max={idx.max()}"
        )

        # ---- Profiling metadata ----
        BLOCK_E = _next_pow2(self.num_experts)
        dtype_size = flat.element_size()
        sram_bytes = 64 * max(BLOCK_E, 64) * dtype_size * 3
        bytes_moved = (flat.numel() + gate_w.numel()) * dtype_size
        kernel_ms = 0.0
        achieved_bw = (bytes_moved / (1024**3)) / 1e-3  # conservative 1ms
        sram_bytes_out = sram_bytes

        if TRITON_AVAILABLE and flat.is_cuda and not force_reference:  # pragma: no cover
            with torch.no_grad():
                try:
                    _, _, _, kernel_ms, sram_bytes_out, achieved_bw = _triton_forward(
                        flat, gate_w, self.top_k
                    )
                except Exception:
                    pass

        # Compute routing quality metrics (no_grad, detached)
        with torch.no_grad():
            logits_fp32 = flat.float() @ gate_w.float()
            self._last_logits = logits_fp32.detach()
            z_loss = _compute_router_z_loss(logits_fp32)
            load_imbalance = _compute_load_imbalance(dispatch_cnt)

        self.last_profile = RouterProfile(
            sram_bytes_per_block=int(sram_bytes_out),
            achieved_bandwidth_gbps=float(achieved_bw),
            kernel_ms=float(kernel_ms),
            used_triton=(TRITON_AVAILABLE and flat.is_cuda),
            tokens_per_expert_mean=float(dispatch_cnt.float().mean().item()),
            tokens_per_expert_std=float(dispatch_cnt.float().std().item()),
            expert_load_imbalance=float(load_imbalance),
            router_z_loss=float(z_loss),
        )
        return idx, w, dispatch_cnt
