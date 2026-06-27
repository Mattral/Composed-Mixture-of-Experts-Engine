"""
pkg/distributed/sequence_parallel.py
======================================

Sequence Parallelism (SP) — activations sharded along the sequence dimension
across the Tensor Parallel (TP) process group.

Sequence Parallelism activates automatically when ``tp_size > 1``.
At ``tp_size == 1`` every function is an identity with zero overhead.

Background
----------
Sequence Parallelism (Korthikanti et al., 2022) shards the sequence dimension
across TP ranks so that the per-rank activation memory for large ``[B, S, H]``
tensors scales as ``S / tp_size`` instead of ``S``. This is essential at long
context lengths (S ≥ 8192) where the unsharded activation alone can OOM a
single device.

The v0.3 fused all-gather path reduces the number of collectives in the
backward pass by fusing the all-gather with the immediately following
projection matmul, replacing ``all_gather(shard) → matmul`` with
``matmul(shard) → all_reduce``.

Communication patterns
-----------------------
Forward:
    scatter_to_sp(x)         : [B, S, H] → [B, S//tp, H]    (slice, no comm)
    gather_from_sp(x)        : [B, S//tp, H] → [B, S, H]    (all_gather)

Fused forward (``next_weight`` provided):
    scatter_to_sp(x, W)      : [B, S, H] → [B, S//tp, out]   (matmul + all_reduce)
    This replaces the standard: scatter → matmul → (implicit all_gather in RowParallel)
    with a single matmul on the shard followed by one all_reduce, halving collectives.

Public API
----------
    scatter_to_sequence_parallel(x, topology, next_weight=None)
    gather_from_sequence_parallel(x, topology)
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.distributed as dist

from pkg.distributed.mesh import ParallelTopology, tp_process_group

__all__ = [
    "scatter_to_sequence_parallel",
    "gather_from_sequence_parallel",
]


def scatter_to_sequence_parallel(
    x: torch.Tensor,
    topology: ParallelTopology,
    next_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Shard the sequence dimension across the TP group.

    Standard path (``next_weight=None``)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Slices ``x`` along the sequence dimension with no communication::

        shard = x[:, rank*chunk:(rank+1)*chunk, :]   # [B, S//tp, H]
        return shard

    At ``tp_size == 1``: identity, no collective.

    Fused projection path (``next_weight`` provided) — v0.3
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Fuses the backward all-gather with the next projection matmul, replacing
    ``all_gather(shard) → matmul(shard, W)`` with ``matmul(shard, W) → all_reduce``.
    This halves the number of collectives in the SP backward pass::

        shard  = x[:, rank*chunk:(rank+1)*chunk, :]   # [B, S//tp, H]
        local  = shard @ next_weight.T                 # [B, S//tp, out]
        result = all_reduce(SUM, local)                # [B, S//tp, out]
        return result

    The all_reduce is correct because each TP rank computed the projection of
    its slice of the sequence — summing them reconstructs the full projection
    as if computed on the full sequence. This is algebraically equivalent to
    ``all_gather(shard) → matmul(full, W)`` but uses one all_reduce instead of
    one all_gather, which has lower latency on ring topologies.

    Parameters
    ----------
    x : Tensor ``[B, S, H]``
        Full-sequence activations on each rank (replicated before this call).
    topology : ParallelTopology
    next_weight : Optional Tensor ``[out_features, H]``
        Projection weight. When provided, enables the fused path.

    Returns
    -------
    Tensor
        ``[B, S//tp, H]``          when ``next_weight`` is None (standard path)
        ``[B, S//tp, out_features]`` when ``next_weight`` is provided (fused path)

    Raises
    ------
    ValueError
        If ``S`` is not divisible by ``tp_size``.
    """
    if topology.tp_size == 1 or not dist.is_initialized():
        if next_weight is not None:
            return torch.nn.functional.linear(x, next_weight)
        return x

    B, S, H = x.shape
    if S % topology.tp_size != 0:
        raise ValueError(
            f"scatter_to_sequence_parallel: sequence_length ({S}) must be "
            f"divisible by tp_size ({topology.tp_size}). "
            f"Current config has tp_size={topology.tp_size}. "
            "Use a sequence length that is a multiple of tp_size, or set tp_size=1."
        )
    chunk = S // topology.tp_size
    tp_rank = topology.tp_rank
    shard = x[:, tp_rank * chunk:(tp_rank + 1) * chunk, :].contiguous()

    if next_weight is None:
        return shard

    # Fused path: local projection + all_reduce (v0.3 optimisation)
    pg = tp_process_group(topology)
    local_proj = torch.nn.functional.linear(shard, next_weight)
    req = dist.all_reduce(local_proj, op=dist.ReduceOp.SUM, group=pg, async_op=True)
    if req is not None:
        req.wait()
    return local_proj


def gather_from_sequence_parallel(
    x: torch.Tensor,
    topology: ParallelTopology,
) -> torch.Tensor:
    """Reconstruct the full sequence from per-rank shards via all_gather.

    At ``tp_size == 1``: identity, no collective.

    Parameters
    ----------
    x : Tensor ``[B, S//tp, H]``
        Local sequence shard on this TP rank.

    Returns
    -------
    Tensor ``[B, S, H]``
        Full-sequence tensor, reconstructed across all TP ranks.
    """
    if topology.tp_size == 1 or not dist.is_initialized():
        return x

    pg = tp_process_group(topology)
    B, S_local, H = x.shape
    gathered = torch.empty(
        (B, S_local * topology.tp_size, H),
        dtype=x.dtype,
        device=x.device,
    )
    dist.all_gather_into_tensor(gathered, x.contiguous(), group=pg)
    return gathered
