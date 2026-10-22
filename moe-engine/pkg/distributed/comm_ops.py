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

Split sizes are plain Python lists, computed once by the caller and saved on
``ctx``, so backward needs no extra synchronisation to recover them.

Every function is the identity when ``torch.distributed`` is not
initialised, so single-process code paths are unaffected.
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.distributed as dist

__all__ = ["all_to_all"]


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
