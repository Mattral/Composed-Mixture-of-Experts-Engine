"""MFU (Model FLOPs Utilization) accountant.

Computes per-step MFU as:

    MFU = achieved_tflops / hardware_peak_tflops

Where `achieved_tflops` is derived from a theoretical FLOP count for the
forward+backward pass of the model architecture, divided by the measured
step time.

For a Transformer MoE layer the per-token FLOP count is:

    flops_dense        = 2 * T_total * P_dense_layers              (dense attention)
    flops_sparse       = 2 * T_total * (K/E) * P_expert_layers     (sparse experts)
    flops_per_step     = flops_dense + flops_sparse
    
    MFU = flops_per_step / (world_size * hardware_peak_flops * step_time_seconds)

Where:
    T_total            = batch_size * seq_len (total tokens)
    P_dense_layers     = parameter count of all non-expert layers
    P_expert_layers    = parameter count of a single expert (replicated across E experts)
    K                  = top-k value
    E                  = total number of experts
    hardware_peak_flops = peak FLOPs of the GPU model (e.g., 989e12 for H100 SXM5 BF16)

The factor 3 covers forward + backward + recomputation (standard convention
used by Chinchilla, PaLM, Llama papers).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional

log = logging.getLogger(__name__)


@dataclass
class MFUResult:
    achieved_tflops: float
    peak_tflops: float
    mfu: float
    step_ms: float
    tokens_per_sec: float


def compute_mfu(
    batch_tokens: int,
    param_dense: int,
    param_expert: int,
    num_experts: int,
    top_k: int,
    world_size: int,
    hardware_peak_tflops: float,
    step_time_sec: float,
) -> float:
    """Compute MoE-aware MFU with sparse activation accounting.

    Parameters
    ----------
    batch_tokens : int
        Total tokens in batch = batch_size * seq_len
    param_dense : int
        Parameter count of dense layers (embeddings, attention, norms)
    param_expert : int
        Parameter count of a single expert module
    num_experts : int
        Total number of experts
    top_k : int
        Number of active experts per token
    world_size : int
        Number of ranks / GPUs
    hardware_peak_tflops : float
        Peak FLOPs of a single GPU (e.g., 989e12 for H100 SXM5 BF16)
    step_time_sec : float
        Elapsed time for one training step in seconds

    Returns
    -------
    float
        MFU as a fraction between 0.0 and 1.0
    """
    # FLOPs for dense layers: 2 * (fwd + bwd) = 2 * T_total * P_dense
    flops_dense = 2 * batch_tokens * param_dense
    
    # FLOPs for expert layers: only K out of E experts are active per token
    # Fraction active = K / E. Per active token: 2 * (fwd + bwd) = 2 * param_expert
    # Total = 2 * T_total * (K/E) * param_expert
    flops_sparse = 2 * batch_tokens * (top_k / max(num_experts, 1)) * param_expert
    
    # Total FLOPs for this step (forward + backward)
    total_flops = flops_dense + flops_sparse
    
    # Achieve capacity across all ranks
    achieved_tflops = (total_flops / step_time_sec) / 1e12
    peak_total_tflops = world_size * hardware_peak_tflops
    
    mfu = achieved_tflops / max(peak_total_tflops, 1e-9)
    
    # Clamp to [0.0, 1.0] for sanity (>1.0 suggests measurement error)
    mfu_clamped = max(0.0, min(1.0, mfu))
    
    # Warn if MFU looks unrealistic
    if mfu_clamped < 0.30:
        log.warning(
            f"MFU = {mfu_clamped:.2%} is very low; check model config/batch size"
        )
    if mfu_clamped > 0.85:
        log.warning(
            f"MFU = {mfu_clamped:.2%} is suspiciously high; may indicate measurement error"
        )
    
    assert 0.0 <= mfu_clamped <= 1.0, (
        f"MFU out of range: {mfu_clamped}. "
        f"Likely inputs: peak_tflops={hardware_peak_tflops}, step_time={step_time_sec}s"
    )
    
    return mfu_clamped


def compute_moe_flops(
    hidden_dim: int,
    num_layers: int,
    ffn_dim: int,
    num_experts: int,
    top_k: int,
    seq_length: int,
    batch_tokens: int,
    vocab_size: int = 0,
) -> int:
    """Deprecated: Use compute_mfu() with batch_tokens and param counts directly.
    
    This function remains for backward compatibility but is not the recommended
    MFU calculation path. Prefer compute_mfu() which uses P_dense + P_expert counts.
    """
    H = hidden_dim
    F = ffn_dim
    E = num_experts
    K = top_k
    S = seq_length

    # Attention: 4*H^2 per token + 4*H*S per token for QK^T and AV.
    flops_attn_per_token = 4 * H * H + 4 * H * S
    # Router: tokens * gate matrix.
    flops_router_per_token = H * E
    # Expert FFN: per active token = 3 GEMMs of size H*F (SwiGLU has gate+up+down).
    # Each GEMM = 2*H*F mac ops. Total = 6*H*F per active token.
    flops_expert_per_token = K * 6 * H * F
    # Final LM head projection (if vocab_size > 0).
    flops_lm_head_per_token = 2 * H * vocab_size if vocab_size else 0

    flops_per_token_fwd = num_layers * (
        flops_attn_per_token + flops_router_per_token + flops_expert_per_token
    ) + flops_lm_head_per_token
    # Forward + backward + activation recompute = 3x forward.
    flops_per_token_total = 3 * flops_per_token_fwd
    return flops_per_token_total * batch_tokens


class MFUAccountant:
    """Streaming MFU tracker. Call `start_step()` / `end_step(tokens)` per iter."""

    def __init__(self, peak_tflops: float, mfu_target: float = 0.55):
        self.peak_tflops = peak_tflops
        self.mfu_target = mfu_target
        self._t0: float = 0.0
        self._flops_per_token: int = 0
        self.history: list[MFUResult] = []
        self._running_mfu: float = 0.0
        self._steps: int = 0

    def configure(self, flops_per_token: int) -> None:
        self._flops_per_token = flops_per_token

    def start_step(self) -> None:
        self._t0 = time.perf_counter()

    def end_step(self, tokens: int) -> MFUResult:
        dt = max(time.perf_counter() - self._t0, 1e-9)
        achieved = (self._flops_per_token * tokens) / dt / 1e12
        mfu = achieved / max(self.peak_tflops, 1e-9)
        res = MFUResult(
            achieved_tflops=achieved,
            peak_tflops=self.peak_tflops,
            mfu=mfu,
            step_ms=dt * 1000.0,
            tokens_per_sec=tokens / dt,
        )
        self.history.append(res)
        self._steps += 1
        self._running_mfu = (self._running_mfu * (self._steps - 1) + mfu) / self._steps
        return res

    @property
    def running_mfu(self) -> float:
        return self._running_mfu

    def is_above_target(self) -> bool:
        return self._running_mfu >= self.mfu_target
