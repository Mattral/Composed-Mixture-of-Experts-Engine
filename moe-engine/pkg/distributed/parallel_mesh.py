"""
pkg/distributed/parallel_mesh.py
=================================

Backward-compatibility shim.

All symbols that were previously implemented in this monolith are now
maintained in focused submodules:

    pkg/distributed/mesh.py              — ParallelTopology, build_topology
    pkg/distributed/tensor_parallel.py   — ColumnParallelLinear, RowParallelLinear, SP
    pkg/distributed/expert_parallel.py   — all_to_all_dispatch, all_to_all_combine
    pkg/distributed/pipeline_parallel.py — PipelineStage, 1F1B schedule
    pkg/distributed/data_parallel.py     — apply_fsdp2
    pkg/distributed/moe_layer.py         — DistributedMoELayer

This file re-exports all public names so that existing code importing from
``pkg.distributed.parallel_mesh`` continues to work without modification.

New code should import directly from the submodule that owns the symbol,
or from ``pkg.distributed`` (the package __init__) for the public API.

.. deprecated::
   Importing from ``pkg.distributed.parallel_mesh`` is deprecated.
   Use ``pkg.distributed`` or the specific submodule instead.
"""

from __future__ import annotations

# Re-export everything from the focused submodules.
from pkg.distributed.mesh import (         # noqa: F401
    ParallelTopology,
    build_topology,
    tp_process_group  as _tp_process_group,
    pp_process_group  as _pp_process_group,
)
from pkg.distributed.tensor_parallel import (  # noqa: F401
    ColumnParallelLinear,
    RowParallelLinear,
    scatter_to_sequence_parallel,
    gather_from_sequence_parallel,
)
from pkg.distributed.expert_parallel import (  # noqa: F401
    all_to_all_dispatch,
    all_to_all_combine,
)
from pkg.distributed.pipeline_parallel import PipelineStage   # noqa: F401
from pkg.distributed.data_parallel import apply_fsdp2         # noqa: F401
from pkg.distributed.moe_layer import DistributedMoELayer     # noqa: F401
from pkg.distributed.moe_layer import _SwiGLUExpert           # noqa: F401  (private; tests inspect it)

# Legacy aliases kept for backward compat (private names some tests access)
_tp_process_group = _tp_process_group
_pp_process_group = _pp_process_group

__all__ = [
    "ParallelTopology",
    "build_topology",
    "ColumnParallelLinear",
    "RowParallelLinear",
    "scatter_to_sequence_parallel",
    "gather_from_sequence_parallel",
    "all_to_all_dispatch",
    "all_to_all_combine",
    "PipelineStage",
    "apply_fsdp2",
    "DistributedMoELayer",
]
