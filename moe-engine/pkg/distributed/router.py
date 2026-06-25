"""
pkg/distributed/router.py
==========================

High-level router interface — wraps the Triton kernel and provides:
  - A clean public API decoupled from kernel implementation details
  - Per-step ``RouterProfile`` with routing quality metrics
  - Configurable dispatch budget (capacity factor)
  - Graceful fallback to fp64 reference on CPU / without Triton

This module is the boundary between the distributed layer (``moe_layer.py``)
and the kernel layer (``pkg/kernels/moe_router.py``). Callers should import
``MoERouterInterface`` from here rather than ``MoERouter`` from the kernels
package directly — that keeps the distributed layer insulated from kernel
implementation changes.

Architecture Decision
---------------------
See ``docs/adr/ADR-001-triton-router-kernel.md`` for the rationale behind the
Triton fused kernel and the fp64 reference fallback strategy.

Public API
----------
    MoERouterInterface   — thin wrapper with routing quality metrics
    RouterStats          — per-step statistics returned by forward()
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

__all__ = [
    "MoERouterInterface",
    "RouterStats",
]


# ---------------------------------------------------------------------------
# Per-step statistics
# ---------------------------------------------------------------------------


@dataclass
class RouterStats:
    """Routing quality statistics for one forward pass.

    Attributes
    ----------
    expert_indices : Tensor ``[N, K]``
        Top-K expert indices for each token.
    combine_weights : Tensor ``[N, K]``
        Renormalised combination weights (sum to 1 per token).
    dispatch_counts : Tensor ``[E]``
        Number of tokens dispatched to each expert.
    load_imbalance : float
        ``max(dispatch_counts) / mean(dispatch_counts)``.
        1.0 = perfect balance. > 1.5 sustained → add z-loss.
    router_z_loss : float
        ``mean(log(sum(exp(logits)))²)`` — Switch Transformer auxiliary loss.
        Emitted as a telemetry signal; not automatically added to training loss.
    used_triton : bool
        True if the Triton GPU kernel executed; False if CPU fp64 fallback used.
    kernel_ms : float
        Wall-clock time of the router kernel in milliseconds.
    tokens_per_expert_mean : float
        ``N * K / E`` — expected tokens per expert under perfect balance.
    tokens_per_expert_std : float
        Standard deviation of dispatch_counts across experts.
    """

    expert_indices: torch.Tensor
    combine_weights: torch.Tensor
    dispatch_counts: torch.Tensor
    load_imbalance: float
    router_z_loss: float
    used_triton: bool
    kernel_ms: float
    tokens_per_expert_mean: float
    tokens_per_expert_std: float


# ---------------------------------------------------------------------------
# MoERouterInterface
# ---------------------------------------------------------------------------


class MoERouterInterface(nn.Module):
    """High-level router interface for distributed MoE layers.

    Wraps ``pkg.kernels.moe_router.MoERouter`` (Triton kernel + fp64 fallback)
    and exposes:

    - A clean ``forward() → RouterStats`` API
    - Capacity factor enforcement (drop tokens beyond capacity budget)
    - Routing quality telemetry (load imbalance, z-loss, kernel timing)

    This class is the **only** entry point from ``DistributedMoELayer`` into
    the kernel layer. It intentionally does not expose ``MoERouter`` internals.

    Parameters
    ----------
    hidden_dim : int
        Token embedding dimension ``H``.
    num_experts : int
        Total number of experts ``E``.
    top_k : int
        Active experts per token ``K``.
    capacity_factor : float
        Each expert's token buffer = ``capacity_factor × (N * K / E)``.
        Values < 1.0 will silently drop tokens (not recommended).
        Default 1.25 gives 25% head-room above mean load.
    dtype : torch.dtype
        Weight dtype (default ``torch.float32``).

    Examples
    --------
    >>> router = MoERouterInterface(hidden_dim=256, num_experts=8, top_k=2)
    >>> stats = router(tokens)          # tokens: [N, H]
    >>> idx    = stats.expert_indices   # [N, K]
    >>> w      = stats.combine_weights  # [N, K]
    >>> imb    = stats.load_imbalance   # float ≥ 1.0
    """

    def __init__(
        self,
        hidden_dim: int,
        num_experts: int,
        top_k: int,
        capacity_factor: float = 1.25,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        if top_k > num_experts:
            raise ValueError(
                f"MoERouterInterface: top_k ({top_k}) must be ≤ num_experts ({num_experts}). "
                "Each token can only be routed to at most num_experts experts."
            )
        if capacity_factor < 1.0:
            import warnings

            warnings.warn(
                f"capacity_factor={capacity_factor} < 1.0 will silently drop tokens "
                "when routing is imbalanced. Consider using ≥ 1.0.",
                UserWarning,
                stacklevel=2,
            )
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor

        # Lazy import to avoid pulling Triton into environments that don't have it
        from pkg.kernels.moe_router import MoERouter

        self._kernel_router = MoERouter(
            hidden_dim=hidden_dim,
            num_experts=num_experts,
            top_k=top_k,
            dtype=dtype,
        )

    def extra_repr(self) -> str:
        return (
            f"hidden_dim={self.hidden_dim}, num_experts={self.num_experts}, "
            f"top_k={self.top_k}, capacity_factor={self.capacity_factor}"
        )

    def forward(self, tokens: torch.Tensor) -> RouterStats:
        """Route tokens to experts and return per-step statistics.

        Parameters
        ----------
        tokens : Tensor ``[N, H]``  (must be 2D — flatten before calling)

        Returns
        -------
        RouterStats
            All routing outputs and quality metrics for this step.

        Raises
        ------
        ValueError
            If ``tokens`` is not 2-dimensional.
        RuntimeError
            If token conservation is violated (``sum(dispatch_counts) != N*K``).
        """
        if tokens.dim() != 2:
            raise ValueError(
                f"MoERouterInterface.forward expects 2D input [N, H], "
                f"got shape {list(tokens.shape)}. "
                "Flatten [B, S, H] → [B*S, H] before calling."
            )

        N, H = tokens.shape
        if H != self.hidden_dim:
            raise ValueError(f"Token hidden dim {H} ≠ router hidden_dim {self.hidden_dim}.")

        # Call the kernel router (Triton or fp64 fallback)
        idx, weights, dispatch_counts = self._kernel_router(tokens)

        # Enforce token conservation invariant
        total_dispatched = int(dispatch_counts.sum().item())
        expected = N * self.top_k
        if total_dispatched != expected:
            raise RuntimeError(
                f"Token conservation violation: dispatched {total_dispatched} "
                f"tokens but expected {expected} (N={N}, K={self.top_k}). "
                "This indicates a bug in the routing kernel."
            )

        # Extract profiling info from last kernel call
        profile = self._kernel_router.last_profile

        # Compute tokens_per_expert statistics
        dc_float = dispatch_counts.float()
        tpe_mean = float(dc_float.mean().item())
        tpe_std = float(dc_float.std().item())

        return RouterStats(
            expert_indices=idx,
            combine_weights=weights,
            dispatch_counts=dispatch_counts,
            load_imbalance=profile.expert_load_imbalance if profile else 1.0,
            router_z_loss=profile.router_z_loss if profile else 0.0,
            used_triton=profile.used_triton if profile else False,
            kernel_ms=profile.kernel_ms if profile else 0.0,
            tokens_per_expert_mean=tpe_mean,
            tokens_per_expert_std=tpe_std,
        )

    def capacity_budget(self, num_tokens: int) -> int:
        """Return the per-expert token capacity for ``num_tokens`` input tokens.

        Budget = ``ceil(capacity_factor × num_tokens × top_k / num_experts)``.
        Tokens beyond budget are dropped (not dispatched to that expert).
        """
        import math

        return math.ceil(self.capacity_factor * num_tokens * self.top_k / self.num_experts)
