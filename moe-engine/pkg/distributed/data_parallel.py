"""
pkg/distributed/data_parallel.py
==================================

FSDP2 (Fully Sharded Data Parallel v2) wrapper utilities.

Wraps model parameters with FSDP2 along the DP mesh axis.
Expert weights (inside DistributedMoELayer) are explicitly excluded
because they are already EP-sharded and must not be DP-wrapped.

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
    from torch.distributed._composable.fsdp import MixedPrecisionPolicy, fully_shard

    _HAS_FSDP2 = True
except Exception:  # pragma: no cover
    fully_shard = None  # type: ignore
    MixedPrecisionPolicy = None  # type: ignore
    _HAS_FSDP2 = False


def apply_fsdp2(
    model: nn.Module,
    topology: ParallelTopology,
    mixed_precision_dtype: Optional[torch.dtype] = None,
) -> nn.Module:
    """Wrap model parameters with FSDP2 along the DP mesh axis.

    Expert parameters are **excluded** from FSDP2 wrapping.  They are already
    sharded across the EP axis and FSDP2 wrapping them would shard them a
    second time, corrupting the expert assignment logic.

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

    dp_mesh = topology.mesh["dp"] if "dp" in topology.mesh.mesh_dim_names else topology.mesh
    mp_policy = (
        MixedPrecisionPolicy(
            param_dtype=mixed_precision_dtype,
            reduce_dtype=torch.float32,
        )
        if mixed_precision_dtype is not None and MixedPrecisionPolicy is not None
        else None
    )

    # Import here to avoid circular import at module level.
    from pkg.distributed.moe_layer import DistributedMoELayer

    for name, module in model.named_modules():
        if isinstance(module, DistributedMoELayer):
            # Only wrap the router inside the MoE layer — expert weights stay EP-sharded.
            if fully_shard is not None:
                fully_shard(module.router, mesh=dp_mesh, mp_policy=mp_policy)
        elif isinstance(module, (nn.Linear, nn.LayerNorm)) and name:
            if fully_shard is not None:
                fully_shard(module, mesh=dp_mesh, mp_policy=mp_policy)

    if fully_shard is not None:
        fully_shard(model, mesh=dp_mesh, mp_policy=mp_policy)
    return model
