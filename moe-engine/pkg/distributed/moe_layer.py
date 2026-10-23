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
    collect_aux_losses(model) -> (z_sum, lb_sum)
"""

from __future__ import annotations

import math
import time
from typing import List

import torch
import torch.nn as nn

from pkg.distributed.expert_parallel import all_to_all_combine, all_to_all_dispatch
from pkg.distributed.grad_sync import sum_over_data_ranks
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


def build_expert_rank_table(topology: ParallelTopology, num_experts: int) -> torch.Tensor:
    """Return ``table`` with ``table[e]`` = the EP rank that owns expert ``e``.

    Derived from :meth:`ParallelTopology.experts_on_ep_rank`, the same
    ownership rule that decides which experts each rank builds, so dispatch
    can never send a token to a rank that does not hold its expert (F4).

    Raises if any expert is unowned or owned twice, or if ownership is not
    non-decreasing in expert id. ``forward()`` relies on the last property:
    sorting dispatch items by expert id must also group them by destination
    rank, in rank order, for ``all_to_all_single``'s contiguous splits.
    """
    table = torch.full((num_experts,), -1, dtype=torch.long)
    for r in range(topology.ep_size):
        for e in topology.experts_on_ep_rank(num_experts, r):
            if table[e] != -1:
                raise ValueError(f"expert {e} is owned by EP ranks {int(table[e])} and {r}")
            table[e] = r
    if (table < 0).any():
        missing = (table < 0).nonzero().flatten().tolist()
        raise ValueError(f"experts {missing} are owned by no EP rank")
    if num_experts > 1 and (table[1:] < table[:-1]).any():
        raise ValueError(
            "expert ownership must be non-decreasing in expert id so that sorting "
            f"by expert id groups dispatch items by destination rank; got {table.tolist()}"
        )
    return table


def compute_send_counts(
    expert_ids_flat: torch.Tensor, expert_rank_table: torch.Tensor, ep_size: int
) -> torch.Tensor:
    """Number of dispatch items bound for each EP rank.

    ``send_counts[r]`` counts the items whose expert ``expert_rank_table``
    assigns to rank ``r``; the counts always sum to ``expert_ids_flat.numel()``.
    Replaces the old ``E // ep_size`` contiguous-range arithmetic, which left
    the remainder experts uncounted whenever ``E % ep_size != 0`` (F4).
    """
    owner = expert_rank_table.to(expert_ids_flat.device)[expert_ids_flat.long()]
    return torch.bincount(owner, minlength=ep_size)


__all__ = [
    "DistributedMoELayer",
    "build_expert_rank_table",
    "collect_aux_losses",
    "compute_send_counts",
]


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
        # gather_output=False: w_gate/w_up hand their sharded [.., F//tp]
        # output straight to the element-wise silu*up multiply, which stays
        # in sharded space -- exactly what this class's docstring already
        # claimed. Before ADR-008, ColumnParallelLinear had no way to skip
        # its gather, so this composition would silently break (a shape
        # mismatch feeding w_down) at any tp_size > 1; only ever tested at
        # tp_size=1, where the gather is a no-op and the bug is invisible.
        self.w_gate = ColumnParallelLinear(
            hidden_dim,
            ffn_dim,
            bias=False,
            topology=topology,
            gather_output=False,
            device=dev,
            dtype=dtype,
        )
        self.w_up = ColumnParallelLinear(
            hidden_dim,
            ffn_dim,
            bias=False,
            topology=topology,
            gather_output=False,
            device=dev,
            dtype=dtype,
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
    aux_losses : bool         Compute the router auxiliary losses on every
                              forward (see :meth:`_aux_loss_sums`); read them
                              with :func:`collect_aux_losses`.
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
        aux_losses: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.topology = topology
        self.capacity_factor = capacity_factor
        self.capacity_dropping = capacity_dropping
        self.aux_losses = aux_losses
        # Set by every forward when aux_losses is on (else None): 0-dim
        # tensors in this forward's graph, sums over this rank's tokens.
        self.aux_z_sum: "torch.Tensor | None" = None
        self.aux_lb_sum: "torch.Tensor | None" = None

        self.local_expert_ids: List[int] = topology.experts_on_this_rank(num_experts)
        # expert id -> owning EP rank. Non-persistent: derived from the
        # topology, so it follows .to(device) but is not checkpointed.
        self.register_buffer(
            "_expert_rank_table",
            build_expert_rank_table(topology, num_experts).to(topology.device),
            persistent=False,
        )
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
        self.router.keep_logits = self.aux_losses
        idx, weights, dispatch_cnt, *logits = self.router(flat)
        self.aux_z_sum = self.aux_lb_sum = None
        if self.aux_losses:
            self.aux_z_sum, self.aux_lb_sum = self._aux_loss_sums(logits[0], idx)

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

        # ---- Flatten (token, k) pairs into independently-routable dispatch
        #      items. Each of a token's top_k expert assignments can land on
        #      a DIFFERENT EP rank -- the common case whenever top_k>1 and
        #      ep_size>1 -- so dispatch construction must operate per-slot,
        #      not per-token. See ADR-009. ----
        expert_ids_flat = idx.reshape(-1)  # [N*top_k]
        tokens_expanded = flat.repeat_interleave(self.top_k, dim=0)  # [N*top_k, H]
        # tokens_expanded[i*top_k + k] is token i's embedding, independently
        # paired with expert_ids_flat[i*top_k + k] = idx[i, k].

        # ---- Build per-EP-rank send counts (per dispatch ITEM, not per
        #      token) from the same ownership table that decided which
        #      experts each rank built (F4). ----
        send_counts = compute_send_counts(
            expert_ids_flat, self._expert_rank_table, self.topology.ep_size
        )
        # send_counts.sum() == N * top_k always, by construction: every
        # dispatch item has exactly one expert, hence contributes to exactly
        # one rank's count. (The previous per-token "any slot in range"
        # check double-counted tokens whose top_k experts spanned multiple
        # ranks, and the later E // ep_size range arithmetic counted the
        # remainder experts nowhere when E % ep_size != 0 -- both surfaced
        # as "Split sizes doesn't match total dim 0 size".)

        # ---- Sort dispatch items by target expert id for contiguous
        #      per-rank blocks (build_expert_rank_table guarantees ownership
        #      is non-decreasing in expert id, so sorting by global expert id
        #      also groups by destination rank) ----
        sort_order = expert_ids_flat.argsort(stable=True)  # [N*top_k]
        tokens_sorted = tokens_expanded[sort_order]
        expert_ids_sorted = expert_ids_flat[sort_order]

        # ---- EP dispatch: tokens AND their expert ids. The tag is required
        #      for correctness -- each sender's block is internally sorted
        #      by expert id, but all_to_all_single concatenates senders'
        #      blocks in SENDER RANK order, not globally by expert id, so
        #      `received` as a whole is not guaranteed expert-sorted. ----
        received, recv_counts, dispatch_event, self.last_dispatch_ms = all_to_all_dispatch(
            tokens_sorted, send_counts, self.topology
        )
        received_expert_ids, _, _, _ = all_to_all_dispatch(
            expert_ids_sorted.unsqueeze(-1), send_counts, self.topology
        )
        received_expert_ids = received_expert_ids.squeeze(-1)

        # ---- Expert FFN compute (default stream, overlaps with dispatch)
        #      Selects each row by its actual received expert id (boolean
        #      mask), not an equal-division chunk assumption -- routing is
        #      essentially never perfectly balanced in practice, and the
        #      previous equal-chunk slicing applied the wrong expert to
        #      many tokens whenever it wasn't (reproducible even at
        #      ep_size=1; see ADR-009). ----
        t_expert_start = time.perf_counter()
        expert_out = torch.zeros_like(received)
        num_local = len(self.local_expert_ids)
        if num_local > 0 and received.shape[0] > 0:
            for i, eid in enumerate(self.local_expert_ids):
                sel = received_expert_ids == eid
                if sel.any():
                    expert_out[sel] = self.experts[i](received[sel])
        self.last_expert_compute_ms = (time.perf_counter() - t_expert_start) * 1000

        # ---- Comm/compute overlap ratio (v0.3) ----
        denom = max(self.last_expert_compute_ms, 1e-9)
        self.last_overlap_ratio = min(self.last_dispatch_ms / denom, 1.0)

        # ---- EP combine ----
        combined_sorted, self.last_combine_ms = all_to_all_combine(
            expert_out, recv_counts, send_counts, self.topology, dispatch_event
        )

        # ---- Weighted scatter back to original token positions ----
        # combined_sorted is in `sort_order`'s order; invert the permutation
        # to place each dispatch item's result back at its (token, k) flat
        # position, then apply combine weights and sum over k per token.
        combined_flat = torch.empty_like(tokens_expanded)  # [N*top_k, H]
        combined_flat[sort_order] = combined_sorted
        combined = (combined_flat.reshape(N, self.top_k, H) * weights.unsqueeze(-1)).sum(dim=1)

        assert not torch.isnan(combined).any(), (
            "NaN detected in DistributedMoELayer output after combine step. "
            "Check expert FFN numerics and router weights."
        )

        return combined.reshape(original_shape)

    def _aux_loss_sums(self, logits: torch.Tensor, idx: torch.Tensor):
        """Router auxiliary losses of this forward, as **sums** over this
        rank's tokens (F5). ``logits`` are the router's differentiable fp32
        logits, returned by its forward when ``keep_logits`` is set (built and
        returned there so that FSDP2 unshards its parameters for them in
        forward and backward). With ``l_t`` the logits of
        token ``t``, ``p_t = softmax(l_t)`` and ``E`` experts::

            z_sum  = sum_t logsumexp(l_t)^2                (ST-MoE z-loss)
            lb_sum = E * sum_e f_e * sum_t p_te            (Switch load balance)

        ``f_e`` is the fraction of the micro-batch's top-k assignments routed
        to expert ``e`` (before any capacity dropping), counted over the
        **global** micro-batch: one all-reduce of ``E + 1`` numbers over the
        DP and EP groups, never TP (TP ranks route the same tokens). It is a
        constant for autograd, as in Switch Transformer. Every rank of the
        DP and EP groups must therefore run this forward.

        The trainer divides these sums, like the main loss, by the global
        loss-token count (ADR-011), which makes them layout-independent.
        """
        z_sum = (torch.logsumexp(logits, dim=-1) ** 2).sum()

        stats = torch.zeros(self.num_experts + 1, dtype=torch.float64, device=logits.device)
        stats[: self.num_experts] = torch.bincount(idx.reshape(-1), minlength=self.num_experts)
        stats[self.num_experts] = logits.shape[0]
        stats = sum_over_data_ranks(stats, self.topology)
        f = stats[: self.num_experts] / (stats[self.num_experts] * self.top_k).clamp_min(1.0)
        probs_sum = torch.softmax(logits, dim=-1).sum(dim=0)
        lb_sum = self.num_experts * (f.to(probs_sum.dtype) * probs_sum).sum()
        return z_sum, lb_sum

    def _expert_to_rank(self, expert_ids: torch.Tensor) -> torch.Tensor:
        """Map a tensor of expert indices to their owning EP rank.

        Reads the table built by :func:`build_expert_rank_table` from
        :meth:`~pkg.distributed.mesh.ParallelTopology.experts_on_ep_rank`,
        so it always agrees with which experts each rank actually holds and
        with the ``send_counts`` used for dispatch.

        Parameters
        ----------
        expert_ids : LongTensor  ``[...]``

        Returns
        -------
        LongTensor  ``[...]``  — EP rank index for each expert id, on
        ``expert_ids``'s device.
        """
        return self._expert_rank_table.to(expert_ids.device)[expert_ids.long()]


def collect_aux_losses(model: nn.Module):
    """Sum the router auxiliary losses of the last forward over **every**
    :class:`DistributedMoELayer` in ``model`` that has ``aux_losses`` on (F5).

    Returns ``(z_sum, lb_sum)``: 0-dim tensors in the forward's graph, sums
    over this rank's tokens (see :meth:`DistributedMoELayer._aux_loss_sums`).
    Add ``z_weight * z_sum + lb_weight * lb_sum`` to the micro-batch's summed
    main loss before ``backward()``; ``grad_sync.finalize_grads`` then divides
    both by the global loss-token count (ADR-011). Call once per forward.
    """
    layers = [m for m in model.modules() if isinstance(m, DistributedMoELayer) and m.aux_losses]
    if not layers:
        raise ValueError("collect_aux_losses: no DistributedMoELayer has aux_losses enabled")
    if any(m.aux_z_sum is None for m in layers):
        raise RuntimeError("collect_aux_losses: call it after the model's forward pass")
    z_sum = sum(m.aux_z_sum for m in layers)
    lb_sum = sum(m.aux_lb_sum for m in layers)
    return z_sum, lb_sum
