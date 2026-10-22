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
  - ColumnParallel: weight [out//tp, in] → local_out [N, out//tp], then
    all_gather to [N, out] UNLESS gather_output=False (skip the gather —
    use this when feeding directly into a RowParallelLinear; see ADR-008)
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
import torch.nn as nn

from pkg.distributed.comm_ops import copy_to_group, gather_along_dim, reduce_from_group
from pkg.distributed.grad_sync import mark_tensor_parallel_shard
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
        Not currently supported together with ``gather_output=False`` — see
        that parameter's docstring.
    topology : Optional[ParallelTopology]
        Provides tp_size and the TP process group.  If None, tp_size = 1.
    gather_output : bool
        If True (default, and the only behavior before ADR-008): all_gather
        the sharded output back to the full ``out_features`` width before
        returning. If False: return the local ``out_features // tp_size``
        shard directly, with no collective. Set this to False when feeding
        the output straight into a :class:`RowParallelLinear` — that's the
        standard Megatron-LM Column→Row MLP composition (see
        ``_SwiGLUExpert`` in ``pkg/distributed/moe_layer.py``), and gathering
        first would be both wasteful (an unnecessary collective) and wrong
        (``RowParallelLinear`` expects a sharded input matching its own
        ``in_features // tp_size`` weight shape, not the full width).
    device, dtype : passed to the Parameter constructor.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        topology: Optional[ParallelTopology] = None,
        gather_output: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        tp_size = topology.tp_size if topology is not None else 1
        self.tp_size = tp_size
        self.tp_rank = topology.tp_rank if topology is not None else 0
        self.tp_group = tp_process_group(topology) if topology is not None else None
        self.gather_output = gather_output

        if out_features % tp_size != 0:
            raise ValueError(
                f"ColumnParallelLinear: out_features ({out_features}) must be "
                f"divisible by tp_size ({tp_size})"
            )
        local_out = out_features // tp_size
        self.weight = nn.Parameter(torch.empty(local_out, in_features, device=device, dtype=dtype))
        mark_tensor_parallel_shard(self.weight)  # distinct per TP rank: never broadcast
        self.bias: Optional[nn.Parameter] = (
            nn.Parameter(torch.zeros(out_features, device=device, dtype=dtype)) if bias else None
        )
        nn.init.normal_(self.weight, mean=0.0, std=1.0 / math.sqrt(in_features))

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"tp_size={self.tp_size}, bias={self.bias is not None}, "
            f"gather_output={self.gather_output}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.tp_size == 1 or self.tp_group is None:
            local_out = torch.nn.functional.linear(x, self.weight, bias=None)
            return local_out if self.bias is None else local_out + self.bias

        # x is replicated across the TP group but each rank only multiplies
        # it by its own weight shard, so each rank's dL/dx is a partial sum:
        # copy_to_group all-reduces it in backward (Megatron's f; F2). Without
        # it, every rank's x.grad held only its shard's contribution.
        local_out = torch.nn.functional.linear(
            copy_to_group(x, self.tp_group), self.weight, bias=None
        )

        if not self.gather_output:
            if self.bias is not None:
                raise NotImplementedError(
                    "ColumnParallelLinear(gather_output=False, bias=True) is "
                    "not implemented. The correct behavior would add each "
                    "rank's shard of bias (out_features // tp_size values), "
                    "not the full-width bias this class currently allocates "
                    "-- doing that without a test to verify it against has "
                    "no place in this codebase's standards. No current "
                    "caller needs this combination: _SwiGLUExpert (the only "
                    "user of gather_output=False) uses bias=False throughout. "
                    "See ADR-008."
                )
            return local_out

        # Gather the out_features shards along the LAST dim. ADR-008: a raw
        # all_gather_into_tensor concatenates along dim 0 only, which needed
        # a movedim workaround; it also wrote in place into a fresh buffer,
        # which raises in grad mode on Gloo and has no autograd edge (F2).
        # gather_along_dim concatenates along an explicit dim and its
        # backward hands each rank its slice of the gradient.
        gathered = gather_along_dim(local_out, -1, self.tp_group)  # [..., out_features]

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
        self.weight = nn.Parameter(torch.empty(out_features, local_in, device=device, dtype=dtype))
        mark_tensor_parallel_shard(self.weight)  # distinct per TP rank: never broadcast
        self.bias: Optional[nn.Parameter] = (
            nn.Parameter(torch.zeros(out_features, device=device, dtype=dtype)) if bias else None
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

        # Sum the per-rank partial products; backward is the identity, since
        # the output is replicated (Megatron's g; F2). The previous in-place
        # all_reduce only gave the right backward by accident.
        local_out = reduce_from_group(local_out, self.tp_group)

        if self.bias is not None:
            local_out = local_out + self.bias
        return local_out


# ---------------------------------------------------------------------------
# Sequence Parallelism — re-exported from pkg.distributed.sequence_parallel
# (v0.3.2: SP extracted into its own module; kept here for backward compat)
# ---------------------------------------------------------------------------
from pkg.distributed.sequence_parallel import (  # noqa: E402, F401, F811
    gather_from_sequence_parallel,
    scatter_to_sequence_parallel,
)
