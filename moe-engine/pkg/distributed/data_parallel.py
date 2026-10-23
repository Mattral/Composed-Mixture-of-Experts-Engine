"""
pkg/distributed/data_parallel.py
==================================

FSDP2 (Fully Sharded Data Parallel v2) wrapper utilities.

Wraps model parameters with FSDP2 along the DP mesh axis -- all of them,
expert weights included. The DP mesh groups the ranks that share an EP (and
TP) coordinate, so the expert weights each such group holds are one EP
rank's experts: FSDP2 shards them over DP and sums their gradients over DP
(expert-data-parallelism), while they stay EP-sharded across the EP axis
(F14-fsdp; tested by tests/test_grad_sync.py::
test_fsdp2_root_wrap_makes_experts_expert_data_parallel and the dp2 x ep2
layouts in tests/test_differential.py).

Public API
----------
    apply_fsdp2(model, topology, mixed_precision_dtype) -> nn.Module
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from pkg.distributed.mesh import ParallelTopology

__all__ = ["apply_fsdp2"]

# ---------------------------------------------------------------------------
# Optional FSDP2 import — gracefully degrade to identity on older PyTorch.
# ---------------------------------------------------------------------------
try:
    from torch.distributed._composable.fsdp import FSDPModule, MixedPrecisionPolicy, fully_shard

    _HAS_FSDP2 = True
except Exception:  # pragma: no cover
    fully_shard = None  # type: ignore
    MixedPrecisionPolicy = None  # type: ignore
    FSDPModule = None  # type: ignore
    _HAS_FSDP2 = False

# FSDP2 methods apply_fsdp2 relies on: the DP SUM of the global-mean objective
# (F22) and the reduction of unused expert parameters (F25). They depend on
# the torch version; verified with torch 2.14.1 (ADR-011).
_REQUIRED_FSDP2_METHODS = (
    "set_force_sum_reduction_for_comms",
    "set_gradient_divide_factor",
    "set_reduce_scatter_unused_params",
)


def apply_fsdp2(
    model: nn.Module,
    topology: ParallelTopology,
    mixed_precision_dtype: Optional[torch.dtype] = None,
) -> nn.Module:
    """Wrap model parameters with FSDP2 along the DP mesh axis.

    Each router and each ``nn.Linear`` / ``nn.LayerNorm`` is its own FSDP2
    unit; the root ``fully_shard(model)`` manages every remaining parameter,
    **including the expert weights** of every ``DistributedMoELayer``. That
    is expert-data-parallelism, and it is correct: the DP group of a rank
    holds the same EP rank's experts, so FSDP2 shards them across DP ranks
    that own the same experts and reduces their gradients over exactly those
    ranks. (Until M2 this docstring said experts were excluded; they never
    were. F14-fsdp.)

    Gradient reduction over DP is a SUM (ADR-011), unused parameters reduce
    zeros (F25), and grad_sync's parameter marks survive wrapping (F26).

    Parameters
    ----------
    model : nn.Module
        The model to wrap.
    topology : ParallelTopology
        Provides the DP mesh and dp_size.
    mixed_precision_dtype : Optional[torch.dtype]
        If provided (e.g. torch.bfloat16), applies mixed-precision policy
        with full-precision gradient reduction.

    Returns
    -------
    nn.Module
        The same model with FSDP2 applied (or unchanged if conditions are
        not met).
    """
    # Conditions for FSDP2 wrapping:
    #   1. PyTorch FSDP2 composable API is available.
    #   2. DP degree > 1 (otherwise sharding is a no-op).
    #   3. A DeviceMesh is available (needed for the "dp" sub-mesh).
    if not _HAS_FSDP2 or topology.dp_size == 1 or topology.mesh is None:
        return model
    missing = [m for m in _REQUIRED_FSDP2_METHODS if not hasattr(FSDPModule, m)]
    if missing:
        raise RuntimeError(
            f"apply_fsdp2: FSDP2 in torch {torch.__version__} lacks {missing}, which the "
            "DP gradient reduction relies on (ADR-011). Use the pinned torch version."
        )

    dp_mesh = topology.mesh["dp"] if "dp" in topology.mesh.mesh_dim_names else topology.mesh
    # fully_shard needs a policy object even without mixed precision: an
    # explicit mp_policy=None makes FSDP2 raise at the first forward (F24).
    mp_policy = (
        MixedPrecisionPolicy(param_dtype=mixed_precision_dtype, reduce_dtype=torch.float32)
        if mixed_precision_dtype is not None
        else MixedPrecisionPolicy()
    )

    # Import here to avoid circular import at module level.
    from pkg.distributed.grad_sync import SEQUENCE_PARALLEL_ATTR, TENSOR_PARALLEL_SHARD_ATTR
    from pkg.distributed.moe_layer import DistributedMoELayer

    # fully_shard replaces every parameter with a new DTensor parameter, which
    # drops grad_sync's marks; without them sync_replicated_grads never sums an
    # SP-region weight over TP (F26). Record them per (module, name) and put
    # them back on the new parameters after wrapping.
    marks = [
        (module, pname, attr)
        for module in model.modules()
        for pname, p in module.named_parameters(recurse=False)
        for attr in (SEQUENCE_PARALLEL_ATTR, TENSOR_PARALLEL_SHARD_ATTR)
        if getattr(p, attr, False)
    ]

    for name, module in model.named_modules():
        if isinstance(module, DistributedMoELayer):
            # The router is its own FSDP unit; the expert weights are left to
            # the root wrap below (expert-data-parallel over DP).
            if fully_shard is not None:
                fully_shard(module.router, mesh=dp_mesh, mp_policy=mp_policy)
        elif isinstance(module, (nn.Linear, nn.LayerNorm)) and name:
            if fully_shard is not None:
                fully_shard(module, mesh=dp_mesh, mp_policy=mp_policy)

    if fully_shard is not None:
        fully_shard(model, mesh=dp_mesh, mp_policy=mp_policy)
        # DP ranks route their tokens to different experts, so in one backward
        # an expert can get a gradient on one DP rank and none on another.
        # FSDP2 would then reduce-scatter different parameter sets on different
        # ranks: a hang, or (experts being equal-sized) one expert's gradient
        # summed into another's. Reduce zeros for unused parameters (F25).
        model.set_reduce_scatter_unused_params(True)
        # SUM over DP, not FSDP2's default mean: the objective is the mean over
        # the global batch, divided once by the global token count in
        # grad_sync.finalize_grads (ADR-011, F22). Per FSDP module, since the
        # setters do not recurse. Without force-sum, a divide factor of 1 uses
        # ReduceOp.PREMUL_SUM, which Gloo rejects.
        for module in model.modules():
            if isinstance(module, FSDPModule):
                module.set_force_sum_reduction_for_comms(True)
                module.set_gradient_divide_factor(1.0)
    for module, pname, attr in marks:
        setattr(getattr(module, pname), attr, True)
    return model
