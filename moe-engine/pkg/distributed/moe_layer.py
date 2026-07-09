"""
pkg/distributed/moe_layer.py
=============================

DistributedMoELayer: thin orchestrator composing the distributed primitives.

This module contains only two classes:
  - _SwiGLUExpert     — single-expert two-layer SwiGLU FFN (TP-aware)
  - DistributedMoELayer — full MoE block: router + EP + expert FFN + combine

The layer intentionally re-uses the primitives from the surrounding modules
rather than re-implementing low-level logic here.

Token lifecycle (per forward pass)
-----------------------------------
    input tokens [B, S, H]
        |
        v
    router (Triton kernel / fp64 ref)  ->  idx [N, K],  weights [N, K]
        |
        v
    sort by assigned expert  ->  tokens_sorted [N*K, H]
        |
        v
    all_to_all_dispatch  ->  received [total_recv, H]    (EP collective)
        |                              (overlap with expert compute)
        v
    expert FFN  (local experts, default CUDA stream)
        |
        v
    all_to_all_combine  ->  combined_sorted [total_send, H]
        |
        v
    weighted scatter  ->  output [N, H]  ->  reshape [B, S, H]

Telemetry (v0.3)
----------------
After each forward pass the following attributes are set on the layer:
  last_dispatch_ms      - time for the dispatch all-to-all (ms)
  last_combine_ms       - time for the combine all-to-all (ms)
  last_expert_compute_ms- time for local expert FFN compute (ms)
  last_overlap_ratio    - dispatch_ms / expert_compute_ms (clamped to 1.0)

Public API
----------
    DistributedMoELayer
"""

from __future__ import annotations

import math
import time
from typing import List

import torch
import torch.nn as nn

from pkg.distributed.expert_parallel import all_to_all_combine, all_to_all_dispatch
from pkg.distributed.mesh import ParallelTopology
from pkg.distributed.tensor_parallel import ColumnParallelLinear, RowParallelLinear
from pkg.kernels.moe_router import MoERouter


def _cumcount(groups: torch.Tensor) -> torch.Tensor:
    """Return each element's 0-indexed position of appearance within its group.

    This is the "first-come-first-served" ordering used by Switch Transformer
    (Fedus et al. 2021) and GShard (Lepikhin et al. 2020) for expert capacity
    dropping: within each expert's queue, tokens are kept in the order they
    appear in the batch, and any token beyond the capacity limit is dropped.

    Implemented with a stable sort + ``cummax`` trick — no Python loops, no
    ``scatter_reduce`` (which has version-dependent availability), so this
    runs identically on any PyTorch >= 2.0 on CPU or GPU.

    Parameters
    ----------
    groups : LongTensor  ``[N]``
        Group (expert) id for each element.

    Returns
    -------
    LongTensor  ``[N]``
        For each element, its 0-indexed position among prior elements
        sharing the same group value, preserving original order.

    Example
    -------
    >>> _cumcount(torch.tensor([2, 0, 2, 1, 0, 2]))
    tensor([0, 0, 1, 0, 1, 2])
    # idx0 (val=2) is the 1st token routed to expert 2   -> position 0
    # idx1 (val=0) is the 1st token routed to expert 0   -> position 0
    # idx2 (val=2) is the 2nd token routed to expert 2   -> position 1
    # idx3 (val=1) is the 1st token routed to expert 1   -> position 0
    # idx4 (val=0) is the 2nd token routed to expert 0   -> position 1
    # idx5 (val=2) is the 3rd token routed to expert 2   -> position 2
    """
    n = groups.shape[0]
    if n == 0:
        return torch.empty(0, dtype=torch.long, device=groups.device)

    order = torch.argsort(groups, stable=True)
    sorted_groups = groups[order]

    pos = torch.arange(n, device=groups.device)
    is_start = torch.ones_like(sorted_groups, dtype=torch.bool)
    is_start[1:] = sorted_groups[1:] != sorted_groups[:-1]

    start_idx = torch.where(is_start, pos, torch.zeros_like(pos))
    start_idx_filled = torch.cummax(start_idx, dim=0).values
    cumcount_sorted = pos - start_idx_filled

    cumcount = torch.empty_like(cumcount_sorted)
    cumcount[order] = cumcount_sorted
    return cumcount


def compute_capacity_drop_mask(
    idx: torch.Tensor,
    num_experts: int,
    capacity: int,
) -> torch.Tensor:
    """Compute the capacity-drop mask for a routing assignment.

    Follows Switch Transformer / GShard semantics: each expert accepts at
    most ``capacity`` tokens, keeping the first ``capacity`` tokens (by
    order of appearance in the batch) that selected it and dropping the
    remainder. Dropped (token, k) pairs receive zero combine weight — the
    expert FFN output for that slot is simply not added, so the token's
    output for that top-k slot falls back to its other (non-dropped) slots.

    Parameters
    ----------
    idx : LongTensor  ``[N, K]``
        Expert indices assigned to each token (from the router).
    num_experts : int
        Total number of experts ``E`` (unused directly, kept for API clarity
        and future per-expert capacity overrides).
    capacity : int
        Maximum tokens any single expert accepts. Typically
        ``ceil(capacity_factor * N * K / E)``, matching
        :meth:`MoERouterInterface.capacity_budget`.

    Returns
    -------
    BoolTensor  ``[N, K]``
        ``True`` where the (token, k) assignment is dropped (exceeds
        capacity), ``False`` where it is kept.
    """
    N, K = idx.shape
    drop_mask = torch.zeros(N, K, dtype=torch.bool, device=idx.device)
    for k in range(K):
        position_in_queue = _cumcount(idx[:, k])
        drop_mask[:, k] = position_in_queue >= capacity
    return drop_mask


__all__ = ["DistributedMoELayer"]


# ===========================================================================
# SwiGLU Expert FFN
# ===========================================================================


class _SwiGLUExpert(nn.Module):
    """Two-layer SwiGLU FFN: ``w_down(silu(w_gate(x)) * w_up(x))``.

    Both ``w_gate`` and ``w_up`` are :class:`ColumnParallelLinear` so their
    outputs have shape ``[F // tp_size]`` on each rank.  The element-wise
    multiply happens in that sharded space.  ``w_down`` is a
    :class:`RowParallelLinear` that all_reduces once at the output to
    reconstruct the full H dimension.

    At tp_size == 1, all three reduce to plain nn.Linear with no collectives.
    """

    def __init__(
        self,
        hidden_dim: int,
        ffn_dim: int,
        topology: ParallelTopology,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        dev = topology.device
        self.w_gate = ColumnParallelLinear(
            hidden_dim, ffn_dim, bias=False, topology=topology, device=dev, dtype=dtype
        )
        self.w_up = ColumnParallelLinear(
            hidden_dim, ffn_dim, bias=False, topology=topology, device=dev, dtype=dtype
        )
        self.w_down = RowParallelLinear(
            ffn_dim, hidden_dim, bias=False, topology=topology, device=dev, dtype=dtype
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(torch.nn.functional.silu(self.w_gate(x)) * self.w_up(x))


# ===========================================================================
# DistributedMoELayer
# ===========================================================================


class DistributedMoELayer(nn.Module):
    """Full MoE block: router + EP all-to-all + expert FFN + weighted combine.

    v0.3 additions:
      - ``last_overlap_ratio``: dispatch_ms / expert_compute_ms, surfaced in
        training loop telemetry without additional CUDA event overhead.
      - NaN guard on output (assertion after combine step).

    Parameters
    ----------
    hidden_dim : int    Token embedding dimension H.
    ffn_dim : int       Expert FFN intermediate dimension F.
    num_experts : int   Total number of experts E across all EP ranks.
    top_k : int         Active experts per token K.
    topology : ParallelTopology
    capacity_factor : float   EP buffer over-provision factor (default 1.25).
    dtype : torch.dtype       Expert parameter dtype.
    """

    def __init__(
        self,
        hidden_dim: int,
        ffn_dim: int,
        num_experts: int,
        top_k: int,
        topology: ParallelTopology,
        capacity_factor: float = 1.25,
        capacity_dropping: bool = False,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.topology = topology
        self.capacity_factor = capacity_factor
        self.capacity_dropping = capacity_dropping

        self.local_expert_ids: List[int] = topology.experts_on_this_rank(num_experts)
        self.experts = nn.ModuleList(
            [_SwiGLUExpert(hidden_dim, ffn_dim, topology, dtype) for _ in self.local_expert_ids]
        )
        self.router = MoERouter(hidden_dim, num_experts, top_k, dtype=dtype)

        # Telemetry attributes — updated every forward pass.
        self.last_dispatch_ms: float = 0.0
        self.last_combine_ms: float = 0.0
        self.last_expert_compute_ms: float = 0.0
        self.last_overlap_ratio: float = 0.0
        self.last_dropped_token_fraction: float = 0.0

    def extra_repr(self) -> str:
        return (
            f"hidden={self.hidden_dim}, num_experts={self.num_experts}, "
            f"top_k={self.top_k}, local_experts={len(self.local_expert_ids)}, "
            f"ep_size={self.topology.ep_size}"
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Full MoE forward pass.

        Parameters
        ----------
        tokens : Tensor  ``[B, S, H]`` or ``[N, H]``

        Returns
        -------
        Tensor  same shape as input
        """
        # ---- Flatten to [N, H] ----
        original_shape = tokens.shape
        if tokens.dim() == 3:
            _, _, H = tokens.shape  # noqa: F841
        elif tokens.dim() == 2:
            B, S = 1, tokens.shape[0]  # noqa: F841 (kept for readability; H below is what's used)
            H = tokens.shape[1]
        else:
            raise ValueError(f"tokens must be rank 2 or 3, got {tokens.dim()}")
        flat = tokens.reshape(-1, H)
        N = flat.shape[0]

        # ---- Router ----
        idx, weights, dispatch_cnt = self.router(flat)

        # ---- Capacity-based token dropping (optional, off by default) ----
        # When enabled, enforces a hard per-expert token budget following
        # Switch Transformer / GShard semantics: each expert accepts at most
        # `capacity_factor * (N*K/E)` tokens, keeping the first arrivals and
        # dropping the rest. Dropped slots get zero combine weight, so their
        # contribution to the output is silently omitted for that slot (the
        # token still receives contributions from its non-dropped top-k
        # slots, if any). This bounds worst-case expert compute and EP
        # bandwidth at the cost of some tokens being under-processed under
        # imbalanced routing — the standard MoE capacity/quality trade-off.
        self.last_dropped_token_fraction = 0.0
        if self.capacity_dropping:
            capacity = math.ceil(self.capacity_factor * N * self.top_k / self.num_experts)
            drop_mask = compute_capacity_drop_mask(idx, self.num_experts, capacity)
            if drop_mask.any():
                weights = weights.masked_fill(drop_mask, 0.0)
                self.last_dropped_token_fraction = float(drop_mask.float().mean().item())

        # ---- Build per-EP-rank send counts ----
        experts_per_rank = self.num_experts // self.topology.ep_size
        send_counts = torch.zeros(self.topology.ep_size, dtype=torch.long, device=flat.device)
        for ep_r in range(self.topology.ep_size):
            lo = ep_r * experts_per_rank
            hi = lo + experts_per_rank
            mask = ((idx >= lo) & (idx < hi)).any(dim=-1)
            send_counts[ep_r] = int(mask.sum().item()) * self.top_k

        # ---- Sort tokens by assigned expert for contiguous dispatch ----
        sort_order = idx[:, 0].argsort(stable=True)
        tokens_sorted = flat[sort_order].repeat_interleave(self.top_k, dim=0)

        # ---- EP dispatch ----
        received, recv_counts, dispatch_event, self.last_dispatch_ms = all_to_all_dispatch(
            tokens_sorted, send_counts, self.topology
        )

        # ---- Expert FFN compute (default stream, overlaps with dispatch) ----
        t_expert_start = time.perf_counter()
        expert_out = torch.zeros_like(received)
        num_local = len(self.local_expert_ids)
        if num_local > 0 and received.shape[0] > 0:
            chunk = max(1, received.shape[0] // max(num_local, 1))
            for i, _eid in enumerate(self.local_expert_ids):
                lo_e = i * chunk
                hi_e = min(lo_e + chunk, received.shape[0])
                if lo_e < hi_e:
                    expert_out[lo_e:hi_e] = self.experts[i](received[lo_e:hi_e])
        self.last_expert_compute_ms = (time.perf_counter() - t_expert_start) * 1000

        # ---- Comm/compute overlap ratio (v0.3) ----
        denom = max(self.last_expert_compute_ms, 1e-9)
        self.last_overlap_ratio = min(self.last_dispatch_ms / denom, 1.0)

        # ---- EP combine ----
        combined_sorted, self.last_combine_ms = all_to_all_combine(
            expert_out, recv_counts, send_counts, self.topology, dispatch_event
        )

        # ---- Weighted scatter back to original token positions ----
        combined = torch.zeros_like(flat)
        w_flat = weights[sort_order].reshape(-1)
        for k in range(self.top_k):
            slot = combined_sorted[k :: self.top_k]
            combined[sort_order] += w_flat[k :: self.top_k].unsqueeze(-1) * slot[:N]

        assert not torch.isnan(combined).any(), (
            "NaN detected in DistributedMoELayer output after combine step. "
            "Check expert FFN numerics and router weights."
        )

        return combined.reshape(original_shape)

    def _expert_to_rank(self, expert_ids: torch.Tensor) -> torch.Tensor:
        """Map a tensor of expert indices to their owning EP rank.

        Uses the same round-robin remainder assignment as
        :meth:`~pkg.distributed.mesh.ParallelTopology.experts_on_this_rank`
        so the mapping is guaranteed consistent across all callers.

        Parameters
        ----------
        expert_ids : LongTensor  ``[...]``

        Returns
        -------
        LongTensor  ``[...]``  — EP rank index for each expert id.
        """
        ep_size = self.topology.ep_size
        E = self.num_experts
        per_rank = E // ep_size
        rem = E % ep_size

        lookup = torch.zeros(E, dtype=torch.long)
        idx = 0
        for r in range(ep_size):
            count = per_rank + (1 if r < rem else 0)
            lookup[idx : idx + count] = r
            idx += count

        return lookup[expert_ids.long()]
