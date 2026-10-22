"""
pkg/distributed/comm_ops.py
===========================

Autograd-aware collectives for expert, tensor and sequence parallelism.

Raw ``torch.distributed`` calls write into freshly allocated buffers, so
autograd has no edge from a collective's output back to its input: a loss
computed after the collective silently stops back-propagating at it (F1-F3
in the October 2026 audit). Each primitive here is a
``torch.autograd.Function`` whose backward is the collective's adjoint.

Public API
----------
    all_to_all(x, output_split_sizes, input_split_sizes, group)
        EP dispatch/combine. Backward: the reverse all-to-all with the split
        sizes swapped.
    copy_to_group(x, group)      fwd identity          bwd all-reduce(sum)
    reduce_from_group(x, group)  fwd all-reduce(sum)   bwd identity
    gather_along_dim(x, dim, group)  fwd all-gather + cat on ``dim``
                                     bwd keep this rank's chunk
    split_along_dim(x, dim, group)   fwd keep this rank's chunk of ``dim``
                                     bwd all-gather + cat on ``dim``

The last four are Megatron-LM's conjugate pairs. Gradient conventions: a
tensor replicated across the group carries the full gradient on every rank;
a sharded tensor carries its shard of it. So ``copy_to_group`` (replicated
input feeding per-rank computation) sums the per-rank partial gradients, and
``gather_along_dim`` (sharded input, replicated output) hands each rank its
slice. Gathering by concatenating along an explicit ``dim`` also removes the
``all_gather_into_tensor`` dim-0-only layout pitfall (ADR-007/ADR-008)
without any transpose.

Split sizes are plain Python lists, computed once by the caller and saved on
``ctx``, so backward needs no extra synchronisation to recover them.

Every function is the identity when ``torch.distributed`` is not
initialised, so single-process code paths are unaffected.
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.distributed as dist

__all__ = [
    "all_to_all",
    "copy_to_group",
    "reduce_from_group",
    "gather_along_dim",
    "split_along_dim",
]


def _all_to_all(
    x: torch.Tensor,
    output_split_sizes: List[int],
    input_split_sizes: List[int],
    group: "Optional[dist.ProcessGroup]",
) -> torch.Tensor:
    out = torch.empty(
        (sum(output_split_sizes), *x.shape[1:]),
        dtype=x.dtype,
        device=x.device,
    )
    dist.all_to_all_single(
        out,
        x.contiguous(),
        output_split_sizes=output_split_sizes,
        input_split_sizes=input_split_sizes,
        group=group,
    )
    return out


class _AllToAll(torch.autograd.Function):
    """Uneven all-to-all along dim 0. Backward sends each gradient row back
    to the rank its forward row came from: the same exchange with input and
    output split sizes swapped."""

    @staticmethod
    def forward(ctx, x, output_split_sizes, input_split_sizes, group):
        ctx.output_split_sizes = output_split_sizes
        ctx.input_split_sizes = input_split_sizes
        ctx.group = group
        return _all_to_all(x, output_split_sizes, input_split_sizes, group)

    @staticmethod
    def backward(ctx, grad_out):
        grad_in = _all_to_all(grad_out, ctx.input_split_sizes, ctx.output_split_sizes, ctx.group)
        return grad_in, None, None, None


def all_to_all(
    x: torch.Tensor,
    output_split_sizes: List[int],
    input_split_sizes: List[int],
    group: "Optional[dist.ProcessGroup]" = None,
) -> torch.Tensor:
    """Differentiable ``all_to_all_single`` along dim 0 with uneven splits.

    Rows ``x[sum(input_split_sizes[:r]) : ...][:input_split_sizes[r]]`` go to
    rank ``r``; the result holds ``output_split_sizes[r]`` rows from each rank
    ``r``, in rank order. Integer tensors (e.g. expert ids) pass through with
    no autograd graph, since they never require grad.
    """
    if not dist.is_initialized():
        return x
    return _AllToAll.apply(x, list(output_split_sizes), list(input_split_sizes), group)


# ===========================================================================
# Tensor / sequence parallel conjugate pairs
# ===========================================================================


def _all_reduce(x: torch.Tensor, group: "Optional[dist.ProcessGroup]") -> torch.Tensor:
    out = x.contiguous().clone()
    dist.all_reduce(out, op=dist.ReduceOp.SUM, group=group)
    return out


def _gather(x: torch.Tensor, dim: int, group: "Optional[dist.ProcessGroup]") -> torch.Tensor:
    x = x.contiguous()
    parts = [torch.empty_like(x) for _ in range(dist.get_world_size(group))]
    dist.all_gather(parts, x, group=group)
    return torch.cat(parts, dim=dim)


def _split(x: torch.Tensor, dim: int, group: "Optional[dist.ProcessGroup]") -> torch.Tensor:
    world = dist.get_world_size(group)
    if x.shape[dim] % world != 0:
        raise ValueError(
            f"split_along_dim: size {x.shape[dim]} of dim {dim} is not divisible "
            f"by the group size {world}"
        )
    return x.chunk(world, dim=dim)[dist.get_rank(group)].contiguous()


class _CopyToGroup(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, group):
        ctx.group = group
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_out):
        return _all_reduce(grad_out, ctx.group), None


class _ReduceFromGroup(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, group):
        return _all_reduce(x, group)

    @staticmethod
    def backward(ctx, grad_out):
        return grad_out, None


class _GatherAlongDim(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, dim, group):
        ctx.dim, ctx.group = dim, group
        return _gather(x, dim, group)

    @staticmethod
    def backward(ctx, grad_out):
        return _split(grad_out, ctx.dim, ctx.group), None, None


class _SplitAlongDim(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, dim, group):
        ctx.dim, ctx.group = dim, group
        return _split(x, dim, group)

    @staticmethod
    def backward(ctx, grad_out):
        return _gather(grad_out, ctx.dim, ctx.group), None, None


def copy_to_group(x: torch.Tensor, group: "Optional[dist.ProcessGroup]" = None) -> torch.Tensor:
    """Identity forward; all-reduce of the gradient in backward.

    Use where a replicated tensor enters per-rank computation (Megatron's
    ``f`` operator), e.g. the input of a column-parallel linear: each rank's
    gradient w.r.t. it is only its shard's partial contribution.
    """
    if not dist.is_initialized():
        return x
    return _CopyToGroup.apply(x, group)


def reduce_from_group(x: torch.Tensor, group: "Optional[dist.ProcessGroup]" = None) -> torch.Tensor:
    """All-reduce (sum) forward; identity backward (Megatron's ``g``).

    Use to sum per-rank partial results into a replicated tensor, e.g. the
    output of a row-parallel linear.
    """
    if not dist.is_initialized():
        return x
    return _ReduceFromGroup.apply(x, group)


def gather_along_dim(
    x: torch.Tensor, dim: int, group: "Optional[dist.ProcessGroup]" = None
) -> torch.Tensor:
    """All-gather equal-sized shards and concatenate them along ``dim``.

    Backward keeps this rank's slice of the (replicated) gradient.
    """
    if not dist.is_initialized():
        return x
    return _GatherAlongDim.apply(x, dim, group)


def split_along_dim(
    x: torch.Tensor, dim: int, group: "Optional[dist.ProcessGroup]" = None
) -> torch.Tensor:
    """Keep this rank's equal-sized chunk of a replicated tensor along ``dim``.

    Backward all-gathers the per-rank gradients into the full gradient. The
    size of ``dim`` must be divisible by the group size.
    """
    if not dist.is_initialized():
        return x
    return _SplitAlongDim.apply(x, dim, group)
