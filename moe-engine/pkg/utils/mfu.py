"""
pkg/utils/mfu.py
================

MFU (Model FLOPs Utilization) accountant.

Computes per-step MFU as:

    MFU = achieved_tflops / (world_size * hardware_peak_tflops)

Where ``achieved_tflops`` is derived from the theoretical FLOP count for
forward + backward of the model architecture divided by the measured step time.

For a Transformer MoE model the per-step FLOPs are:

    flops_dense     = 2 * T_total * P_dense_layers     (non-expert layers)
    flops_sparse    = 2 * T_total * (K / E) * P_expert (sparse expert layers)
    flops_per_step  = flops_dense + flops_sparse

Where:
    T_total        = batch_size * seq_len (total tokens)
    P_dense_layers = parameter count of all non-expert layers
    P_expert       = parameter count of a single expert module
    K              = top-k value
    E              = total number of experts

The *3× rule* (forward + backward + recompute) is baked into the 2× factor
for fwd+bwd; activation recompute is accounted for separately when enabled.

References
----------
- "Training Compute-Optimal Large Language Models" (Hoffmann et al., 2022)
- PaLM: "Scaling Language Modeling with Pathways" (Chowdhery et al., 2022)
- Llama: arXiv:2302.13971
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

log = logging.getLogger(__name__)


@dataclass
class MFUResult:
    achieved_tflops: float
    peak_tflops: float
    mfu: float
    step_ms: float
    tokens_per_sec: float
    # v0.2: breakdown
    flops_dense: float
    flops_sparse: float


def compute_mfu(
    batch_tokens: int,
    param_dense: int,
    param_expert: int,
    num_experts: int,
    top_k: int,
    world_size: int,
    hardware_peak_tflops: float,
    step_time_sec: float,
    activation_recompute: bool = False,
) -> float:
    """Compute MoE-aware MFU with sparse activation accounting.

    Parameters
    ----------
    batch_tokens : int
        Total tokens in batch (batch_size × seq_len)
    param_dense : int
        Parameter count of dense layers (embeddings, attention, norms)
    param_expert : int
        Parameter count of one expert module
    num_experts : int
        Total number of experts
    top_k : int
        Active experts per token
    world_size : int
        Number of GPU ranks
    hardware_peak_tflops : float
        Peak TFLOPS per GPU (e.g., 989.0 for H100 SXM5 BF16)
    step_time_sec : float
        Elapsed wall-clock time for one training step (seconds)
    activation_recompute : bool
        If True, add another forward-pass worth of FLOPs for recomputation

    Returns
    -------
    float
        MFU ∈ [0.0, 1.0]
    """
    # Forward + backward = 2× forward FLOPs (Chinchilla convention)
    fwd_bwd_multiplier = 2.0
    if activation_recompute:
        fwd_bwd_multiplier += 1.0  # extra forward for recompute

    flops_dense = fwd_bwd_multiplier * batch_tokens * param_dense
    # Sparse factor: K / E fraction of expert params active per token
    sparse_fraction = top_k / max(num_experts, 1)
    flops_sparse = fwd_bwd_multiplier * batch_tokens * sparse_fraction * param_expert

    total_flops = flops_dense + flops_sparse
    achieved_tflops = (total_flops / step_time_sec) / 1e12
    peak_total_tflops = world_size * hardware_peak_tflops

    mfu = achieved_tflops / max(peak_total_tflops, 1e-9)
    mfu_clamped = max(0.0, min(1.0, mfu))

    if mfu_clamped < 0.25:
        log.debug(
            "MFU=%.2f%% is low; check batch size / model config "
            "(world_size=%d, step_ms=%.1f, batch_tokens=%d)",
            mfu_clamped * 100,
            world_size,
            step_time_sec * 1000,
            batch_tokens,
        )
    if mfu_clamped > 0.87:
        log.debug(
            "MFU=%.2f%% is suspiciously high; verify hardware_peak_tflops=%.0f",
            mfu_clamped * 100,
            hardware_peak_tflops,
        )

    return mfu_clamped


def compute_mfu_detailed(
    batch_tokens: int,
    param_dense: int,
    param_expert: int,
    num_experts: int,
    top_k: int,
    world_size: int,
    hardware_peak_tflops: float,
    step_time_sec: float,
    activation_recompute: bool = False,
) -> MFUResult:
    """Like ``compute_mfu`` but returns the full breakdown.

    Returns
    -------
    MFUResult
        Contains mfu, achieved_tflops, dense/sparse FLOP breakdown, etc.
    """
    fwd_bwd_mul = 3.0 if activation_recompute else 2.0
    flops_dense = fwd_bwd_mul * batch_tokens * param_dense
    sparse_fraction = top_k / max(num_experts, 1)
    flops_sparse = fwd_bwd_mul * batch_tokens * sparse_fraction * param_expert

    total_flops = flops_dense + flops_sparse
    achieved_tflops = (total_flops / step_time_sec) / 1e12
    peak_total = world_size * hardware_peak_tflops
    mfu = max(0.0, min(1.0, achieved_tflops / max(peak_total, 1e-9)))
    tps = batch_tokens / max(step_time_sec, 1e-9)

    return MFUResult(
        achieved_tflops=achieved_tflops,
        peak_tflops=peak_total,
        mfu=mfu,
        step_ms=step_time_sec * 1000,
        tokens_per_sec=tps,
        flops_dense=flops_dense,
        flops_sparse=flops_sparse,
    )


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
    """Per-token FLOP estimate for a Transformer MoE model.

    Retained for backward compatibility with the existing training loop.
    Prefer ``compute_mfu()`` directly for new code.
    """
    H = hidden_dim
    F = ffn_dim
    E = num_experts
    K = top_k
    S = seq_length

    flops_attn_per_token = 4 * H * H + 4 * H * S
    flops_router_per_token = H * E
    flops_expert_per_token = K * 6 * H * F  # SwiGLU: gate + up + down
    flops_lm_head_per_token = 2 * H * vocab_size if vocab_size else 0

    flops_per_token_fwd = (
        num_layers * (flops_attn_per_token + flops_router_per_token + flops_expert_per_token)
        + flops_lm_head_per_token
    )
    flops_per_token_total = 3 * flops_per_token_fwd
    return flops_per_token_total * batch_tokens


class MFUAccountant:
    """Streaming MFU tracker.  Call ``start_step()`` / ``end_step(tokens)``
    each iteration.

    Also tracks moving average over a configurable window for smoother
    reporting (useful when step times are noisy, especially in the early
    optimizer warmup steps).
    """

    def __init__(
        self,
        peak_tflops: float,
        mfu_target: float = 0.55,
        smoothing_window: int = 50,
    ):
        self.peak_tflops = peak_tflops
        self.mfu_target = mfu_target
        self.smoothing_window = smoothing_window
        self._t0: float = 0.0
        self._flops_per_token: int = 0
        self.history: list[MFUResult] = []
        self._running_mfu: float = 0.0
        self._steps: int = 0
        # Sliding window for smoothed MFU
        self._window: list[float] = []

    def configure(self, flops_per_token: int) -> None:
        self._flops_per_token = flops_per_token

    def start_step(self) -> None:
        self._t0 = time.perf_counter()

    def end_step(self, tokens: int) -> MFUResult:
        dt = max(time.perf_counter() - self._t0, 1e-9)
        achieved = (self._flops_per_token * tokens) / dt / 1e12
        mfu = max(0.0, min(1.0, achieved / max(self.peak_tflops, 1e-9)))
        tps = tokens / dt

        res = MFUResult(
            achieved_tflops=achieved,
            peak_tflops=self.peak_tflops,
            mfu=mfu,
            step_ms=dt * 1000.0,
            tokens_per_sec=tps,
            flops_dense=0.0,  # not decomposed in streaming path
            flops_sparse=0.0,
        )
        self.history.append(res)
        self._steps += 1
        self._running_mfu = (self._running_mfu * (self._steps - 1) + mfu) / self._steps

        # Sliding window
        self._window.append(mfu)
        if len(self._window) > self.smoothing_window:
            self._window.pop(0)

        return res

    @property
    def running_mfu(self) -> float:
        return self._running_mfu

    @property
    def smoothed_mfu(self) -> float:
        """Sliding-window average MFU (last ``smoothing_window`` steps)."""
        if not self._window:
            return 0.0
        return sum(self._window) / len(self._window)

    def is_above_target(self) -> bool:
        return self._running_mfu >= self.mfu_target

    def summary_str(self) -> str:
        if not self.history:
            return "MFU: no data"
        last = self.history[-1]
        return (
            f"MFU={self._running_mfu:.2%} (smooth={self.smoothed_mfu:.2%}) "
            f"| {last.tokens_per_sec:,.0f} tok/s "
            f"| step={last.step_ms:.1f}ms "
            f"| {last.achieved_tflops:.2f}/{self.peak_tflops:.0f} TFLOPs"
        )
