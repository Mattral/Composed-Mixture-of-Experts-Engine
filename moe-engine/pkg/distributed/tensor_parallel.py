"""
pkg/distributed/tensor_parallel.py
===================================

Tensor-parallel linear layers and sequence-parallel utilities.

Implements:
  - ColumnParallelLinear  — shards output features across the TP group
  - RowParallelLinear     — shards input features across the TP group
  - scatter_to_sequence_parallel  — shard [B, S, H] along S across TP ranks
  - gather_from_sequence_parallel — reconstruct full [B, S, H] from shards

All primitives degrade gracefully to plain nn.Linear / identity when
tp_size == 1 or dist is not initialized, so the entire test suite runs
on CPU without any collective operations.

At tp_size > 1, Tensor Parallelism follows the Megatron-LM column/row
sharding convention:
  - ColumnParallel: weight [out//tp, in]  → all_gather output → [N, out]
  - RowParallel:    weight [out, in//tp]  → all_reduce partial sums → [N, out]

The v0.3 SP fusion:
  scatter_to_sequence_parallel(x, topology, next_weight=W) fuses the
  all-gather back into a single all_reduce(matmul(shard, W)), halving
  the number of collectives per Sequence-Parallel layer.

Public API
----------
    ColumnParallelLinear
    RowParallelLinear
    scatter_to_sequence_parallel
    gather_from_sequence_parallel
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn

from pkg.distributed.mesh import ParallelTopology, tp_process_group

__all__ = [
    "ColumnParallelLinear",
    "RowParallelLinear",
    "scatter_to_sequence_parallel",
    "gather_from_sequence_parallel",
]


# ===========================================================================
# ColumnParallelLinear
# ===========================================================================

class ColumnParallelLinear(nn.Module):
    """Linear that shards the output features across the TP group.

    Weight shape: ``[out_features // tp_size, in_features]`` per rank.

    Forward pass::

        local_out = x @ weight.T                    # [N, out//tp]
        gathered  = all_gather(local_out)           # [N, out]
        return gathered + bias

    At ``tp_size == 1``: identity — no collective, no process group.

    Parameters
    ----------
    in_features : int
    out_features : int
    bias : bool
        If True, a full-rank (un-sharded) bias is added after the gather.
    topology : Optional[ParallelTopology]
        Provides tp_size and the TP process group.  If None, tp_size = 1.
    device, dtype : passed to the Parameter constructor.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        topology: Optional[ParallelTopology] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        tp_size = topology.tp_size if topology is not None else 1
        self.tp_size = tp_size
        self.tp_group = tp_process_group(topology) if topology is not None else None

        if out_features % tp_size != 0:
            raise ValueError(
                f"ColumnParallelLinear: out_features ({out_features}) must be "
                f"divisible by tp_size ({tp_size})"
            )
        local_out = out_features // tp_size
        self.weight = nn.Parameter(
            torch.empty(local_out, in_features, device=device, dtype=dtype)
        )
        self.bias: Optional[nn.Parameter] = (
            nn.Parameter(torch.zeros(out_features, device=device, dtype=dtype))
            if bias else None
        )
        nn.init.normal_(self.weight, mean=0.0, std=1.0 / math.sqrt(in_features))

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"tp_size={self.tp_size}, bias={self.bias is not None}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        local_out = torch.nn.functional.linear(x, self.weight, bias=None)

        if self.tp_size == 1 or self.tp_group is None:
            return local_out if self.bias is None else local_out + self.bias

        gathered = torch.empty(
            (*x.shape[:-1], self.out_features),
            dtype=x.dtype, device=x.device,
        )
        req = dist.all_gather_into_tensor(
            gathered, local_out, group=self.tp_group, async_op=True
        )
        if req is not None:
            req.wait()

        if self.bias is not None:
            gathered = gathered + self.bias
        return gathered


# ===========================================================================
# RowParallelLinear
# ===========================================================================

class RowParallelLinear(nn.Module):
    """Linear that shards the input features across the TP group.

    Weight shape: ``[out_features, in_features // tp_size]`` per rank.

    Forward pass::

        local_out = slice(x) @ weight.T             # [N, out]  partial sums
        result    = all_reduce(SUM, local_out)      # [N, out]  full result
        return result + bias

    The all_reduce is correct because each rank computed a partial dot
    product over its slice of in_features; the full result is their sum.
    reduce_scatter + all_gather would produce two collectives with wrong
    semantics (scatter distributes chunks, not sums them).

    At ``tp_size == 1``: identity — no collective.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        topology: Optional[ParallelTopology] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        tp_size = topology.tp_size if topology is not None else 1
        self.tp_size = tp_size
        self.tp_group = tp_process_group(topology) if topology is not None else None

        if in_features % tp_size != 0:
            raise ValueError(
                f"RowParallelLinear: in_features ({in_features}) must be "
                f"divisible by tp_size ({tp_size})"
            )
        local_in = in_features // tp_size
        self.weight = nn.Parameter(
            torch.empty(out_features, local_in, device=device, dtype=dtype)
        )
        self.bias: Optional[nn.Parameter] = (
            nn.Parameter(torch.zeros(out_features, device=device, dtype=dtype))
            if bias else None
        )
        nn.init.normal_(self.weight, mean=0.0, std=1.0 / math.sqrt(in_features))

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"tp_size={self.tp_size}, bias={self.bias is not None}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        local_out = torch.nn.functional.linear(x, self.weight, bias=None)

        if self.tp_size == 1 or self.tp_group is None:
            return local_out if self.bias is None else local_out + self.bias

        req = dist.all_reduce(
            local_out, op=dist.ReduceOp.SUM,
            group=self.tp_group, async_op=True,
        )
        if req is not None:
            req.wait()

        if self.bias is not None:
            local_out = local_out + self.bias
        return local_out


# ---------------------------------------------------------------------------
# Sequence Parallelism — re-exported from pkg.distributed.sequence_parallel
# (v0.3.2: SP extracted into its own module; kept here for backward compat)
# ---------------------------------------------------------------------------
from pkg.distributed.sequence_parallel import (  # noqa: E402, F401
    scatter_to_sequence_parallel,
    gather_from_sequence_parallel,
)

# ===========================================================================

def scatter_to_sequence_parallel(
    x: torch.Tensor,
    topology: ParallelTopology,
    next_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Shard the sequence dimension across the TP group.

    Standard path (``next_weight=None``)::

        shard = x[:, rank*chunk:(rank+1)*chunk, :]   # [B, S//tp, H]
        return shard

    Fused projection path (``next_weight`` provided) — v0.3::

        shard    = x[:, rank*chunk:(rank+1)*chunk, :]   # [B, S//tp, H]
        local    = shard @ next_weight.T                 # [B, S//tp, out]
        result   = all_reduce(SUM, local)               # [B, S//tp, out]
        return result

    The fused path replaces ``all_gather(shard) → matmul`` with
    ``matmul → all_reduce``, saving one collective per SP layer
    (all_gather has higher latency than all_reduce on typical interconnects).

    Parameters
    ----------
    x : Tensor  ``[B, S, H]``
    topology : ParallelTopology
    next_weight : Optional[Tensor, shape ``[out_features, H]``]
        When provided, fuses the gather with the subsequent projection.

    Returns
    -------
    Tensor
        ``[B, S//tp, H]``          when next_weight is None (standard path)
        ``[B, S//tp, out_features]`` when next_weight is provided (fused path)
    """
    if topology.tp_size == 1 or not dist.is_initialized():
        if next_weight is not None:
            return torch.nn.functional.linear(x, next_weight)
        return x

    B, S, H = x.shape
    if S % topology.tp_size != 0:
        raise ValueError(
            f"scatter_to_sequence_parallel: sequence_length ({S}) must be "
            f"divisible by tp_size ({topology.tp_size})"
        )
    chunk = S // topology.tp_size
    shard = x[:, topology.tp_rank * chunk:(topology.tp_rank + 1) * chunk, :]

    if next_weight is None:
        return shard

    # Fused path: local projection + all_reduce.
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
    """Reconstruct the full sequence from shards across the TP group.

    Parameters
    ----------
    x : Tensor  ``[B, S//tp, H]``  (local shard)

    Returns
    -------
    Tensor  ``[B, S, H]``
    """
    if topology.tp_size == 1 or not dist.is_initialized():
        return x

    pg = tp_process_group(topology)
    B, S_local, H = x.shape
    gathered = torch.empty(
        (B, S_local * topology.tp_size, H),
        dtype=x.dtype, device=x.device,
    )
    dist.all_gather_into_tensor(gathered, x, group=pg)
    return gathered
