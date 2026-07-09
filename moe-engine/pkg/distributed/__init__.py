"""
pkg/distributed
===============

Distributed primitives for 4D parallelism: DP + EP + TP + PP.

Submodule responsibilities
--------------------------
mesh.py              — ParallelTopology, build_topology, process group management
tensor_parallel.py   — ColumnParallelLinear, RowParallelLinear, scatter/gather SP
expert_parallel.py   — all_to_all_dispatch, all_to_all_combine (EP collectives)
pipeline_parallel.py — PipelineStage, 1F1B schedule, inter-stage send/recv
data_parallel.py     — apply_fsdp2 (FSDP2 wrapping along DP axis)
moe_layer.py         — DistributedMoELayer (thin orchestrator)

Public surface
--------------
Import from this package for the most stable interface.
Internal implementation helpers should be imported from their submodules
directly only when necessary.
"""

from __future__ import annotations

# -- Data Parallelism --
from pkg.distributed.data_parallel import apply_fsdp2

# -- Expert Parallelism --
from pkg.distributed.expert_parallel import (
    all_to_all_combine,
    all_to_all_dispatch,
)

# -- Topology --
from pkg.distributed.mesh import (
    ParallelTopology,
    build_topology,
    pp_process_group,
    tp_process_group,
)

# -- MoE Layer --
from pkg.distributed.moe_layer import DistributedMoELayer, compute_capacity_drop_mask

# -- Pipeline Parallelism --
from pkg.distributed.pipeline_parallel import PipelineStage

# -- High-level Router Interface --
from pkg.distributed.router import MoERouterInterface, RouterStats

# -- Sequence Parallelism (own module since v0.3.2) --
from pkg.distributed.sequence_parallel import (
    gather_from_sequence_parallel,
    scatter_to_sequence_parallel,
)

# -- Tensor Parallelism --
from pkg.distributed.tensor_parallel import (
    ColumnParallelLinear,
    RowParallelLinear,
)

__all__ = [
    # Topology
    "ParallelTopology",
    "build_topology",
    "tp_process_group",
    "pp_process_group",
    # Tensor Parallelism
    "ColumnParallelLinear",
    "RowParallelLinear",
    "scatter_to_sequence_parallel",
    "gather_from_sequence_parallel",
    # Expert Parallelism
    "all_to_all_dispatch",
    "all_to_all_combine",
    # Pipeline Parallelism
    "PipelineStage",
    # Data Parallelism
    "apply_fsdp2",
    # MoE Layer
    "DistributedMoELayer",
    "compute_capacity_drop_mask",
    # High-level Router Interface
    "MoERouterInterface",
    "RouterStats",
]
