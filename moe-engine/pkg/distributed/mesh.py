"""
pkg/distributed/mesh.py
=======================

Device mesh construction and parallel topology descriptor.

Responsibilities (single concern):
  - Build the 4D DeviceMesh (DP x TP x PP x EP).
  - Compute per-rank coordinates (dp_rank, tp_rank, pp_rank, ep_rank).
  - Compute expert ownership for a given EP rank.
  - Create/cache per-axis process groups (TP, PP).

Nothing in this file knows about Triton, FSDP, or expert FFNs.

Public API
----------
    ParallelTopology      - frozen dataclass: 4D rank coordinates + mesh
    build_topology(...)   - initialise PG and return a ParallelTopology
    tp_process_group(...) - return (or create) the TP process group
    pp_process_group(...) - return (or create) the PP process group
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist

try:
    from torch.distributed.device_mesh import init_device_mesh, DeviceMesh
    _HAS_DEVICE_MESH = True
except Exception:                                                        # pragma: no cover
    init_device_mesh = None                                              # type: ignore
    DeviceMesh = object                                                  # type: ignore
    _HAS_DEVICE_MESH = False

__all__ = [
    "ParallelTopology",
    "build_topology",
    "tp_process_group",
    "pp_process_group",
]

# ---------------------------------------------------------------------------
# Module-level process group caches.  Keyed by rank-coordinate tuples so
# each (dp_rank, ep_rank, tp_size) combination gets exactly one PG instance.
# ---------------------------------------------------------------------------
_TP_GROUPS: Dict[Tuple[int, int, int], dist.ProcessGroup] = {}
_PP_GROUPS: Dict[Tuple[int, int], dist.ProcessGroup] = {}


# ===========================================================================
# Topology descriptor
# ===========================================================================

@dataclass(frozen=True)
class ParallelTopology:
    """Immutable descriptor for the 4D parallelism mesh on this rank.

    All ranks must construct topologies with the same (dp_size, ep_size,
    tp_size, pp_size) values.  The product must equal world_size.

    Attributes
    ----------
    world_size : int
        Total number of distributed ranks.
    rank : int
        Global rank of this process.
    dp_size, ep_size, tp_size, pp_size : int
        Extents of each parallelism axis.
    mesh : DeviceMesh or None
        PyTorch DeviceMesh object (None in single-process / CPU-only mode).
    device : torch.device
        Device for tensors on this rank (cuda:N or cpu).
    """

    world_size: int
    rank: int
    dp_size: int
    ep_size: int
    tp_size: int = 1
    pp_size: int = 1
    mesh: Optional[object] = field(default=None, compare=False, repr=False)  # DeviceMesh
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))

    # ------------------------------------------------------------------
    # Rank coordinates (computed from global rank + axis sizes)
    # ------------------------------------------------------------------

    @property
    def dp_rank(self) -> int:
        """This rank's position along the data-parallel axis."""
        denominator = self.tp_size * self.pp_size * self.ep_size
        return (self.rank // denominator) % self.dp_size

    @property
    def tp_rank(self) -> int:
        """This rank's position along the tensor-parallel axis."""
        denominator = self.pp_size * self.ep_size
        return (self.rank // denominator) % self.tp_size

    @property
    def pp_rank(self) -> int:
        """This rank's position along the pipeline-parallel axis."""
        return (self.rank // self.ep_size) % self.pp_size

    @property
    def ep_rank(self) -> int:
        """This rank's position along the expert-parallel axis."""
        return self.rank % self.ep_size

    # ------------------------------------------------------------------
    # Expert ownership
    # ------------------------------------------------------------------

    def experts_on_this_rank(self, total_experts: int) -> List[int]:
        """Return global expert indices owned by this EP rank.

        Remainder experts are round-robin assigned to the lowest EP ranks
        so resharding after a node drop never leaves experts orphaned.

        Parameters
        ----------
        total_experts : int
            Total number of experts across all EP ranks.

        Returns
        -------
        List[int]
            Sorted list of expert indices owned by this rank.
        """
        per_rank = total_experts // self.ep_size
        rem = total_experts - per_rank * self.ep_size
        start = self.ep_rank * per_rank + min(self.ep_rank, rem)
        extra = 1 if self.ep_rank < rem else 0
        return list(range(start, start + per_rank + extra))

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def is_first_pp_stage(self) -> bool:
        return self.pp_rank == 0

    def is_last_pp_stage(self) -> bool:
        return self.pp_rank == self.pp_size - 1

    def validate_world_size(self) -> None:
        """Assert the 4D product equals world_size.  Call after distributed init."""
        expected = self.dp_size * self.ep_size * self.tp_size * self.pp_size
        if expected != self.world_size:
            raise ValueError(
                f"Parallelism product dp({self.dp_size}) × ep({self.ep_size}) × "
                f"tp({self.tp_size}) × pp({self.pp_size}) = {expected} "
                f"!= world_size({self.world_size}).\n"
                "Adjust the parallelism config so the product matches the "
                "number of GPUs."
            )


# ===========================================================================
# Process group helpers
# ===========================================================================

def tp_process_group(topology: ParallelTopology) -> "Optional[dist.ProcessGroup]":
    """Return (or lazily create) the TP process group for this rank.

    All ranks sharing the same (dp_rank, pp_rank, ep_rank) triplet but
    differing in tp_rank form a single tensor-parallel group.

    Returns None when tp_size == 1 or dist is not initialized.
    """
    if topology.tp_size == 1 or not dist.is_initialized():
        return None

    # Prefer DeviceMesh's pre-built group when available.
    if topology.mesh is not None:
        try:
            return topology.mesh["tp"].get_group()
        except Exception:
            pass

    assert (
        topology.world_size
        == topology.dp_size * topology.tp_size * topology.ep_size
    )
    key = (topology.dp_rank, topology.ep_rank, topology.tp_size)
    if key in _TP_GROUPS:
        return _TP_GROUPS[key]

    ranks = [
        topology.dp_rank * topology.tp_size * topology.ep_size
        + tp * topology.ep_size
        + topology.ep_rank
        for tp in range(topology.tp_size)
    ]
    group = dist.new_group(ranks=ranks)
    _TP_GROUPS[key] = group
    return group


def pp_process_group(topology: ParallelTopology) -> "Optional[dist.ProcessGroup]":
    """Return (or lazily create) the PP process group for this rank.

    All ranks sharing the same (dp_rank, tp_rank, ep_rank) triplet but
    differing in pp_rank form a single pipeline group.

    Returns None when pp_size == 1 or dist is not initialized.
    """
    if topology.pp_size == 1 or not dist.is_initialized():
        return None

    if topology.mesh is not None:
        try:
            return topology.mesh["pp"].get_group()
        except Exception:
            pass

    key = (topology.dp_rank * topology.ep_size + topology.ep_rank, topology.pp_size)
    if key in _PP_GROUPS:
        return _PP_GROUPS[key]

    base = (
        topology.dp_rank * topology.tp_size * topology.pp_size * topology.ep_size
        + topology.tp_rank * topology.pp_size * topology.ep_size
        + topology.ep_rank
    )
    ranks = [base + pp * topology.ep_size for pp in range(topology.pp_size)]
    group = dist.new_group(ranks=ranks)
    _PP_GROUPS[key] = group
    return group


# ===========================================================================
# Topology constructor
# ===========================================================================

def build_topology(
    dp_size: int,
    ep_size: int,
    tp_size: int = 1,
    pp_size: int = 1,
    device_type: str = "cuda",
) -> ParallelTopology:
    """Initialise the process group and return a :class:`ParallelTopology`.

    Falls back to a degenerate 1-rank topology on CPU-only or single-process
    environments so the entire test suite runs without a GPU.

    Parameters
    ----------
    dp_size : int   Data-parallel degree.
    ep_size : int   Expert-parallel degree.
    tp_size : int   Tensor-parallel degree (default 1).
    pp_size : int   Pipeline-parallel degree (default 1).
    device_type : str  "cuda" or "cpu".

    Returns
    -------
    ParallelTopology

    Raises
    ------
    ValueError
        If the product dp × ep × tp × pp != world_size (only in multi-rank runs).
    """
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_initialized() else 0

    # Single-process or no DeviceMesh: return a degenerate 1-rank topology.
    if world_size == 1 or not _HAS_DEVICE_MESH:
        device_type_actual = (
            device_type if torch.cuda.is_available() and device_type == "cuda" else "cpu"
        )
        dev = torch.device(device_type_actual)
        return ParallelTopology(
            world_size=1, rank=0,
            dp_size=1, ep_size=1, tp_size=1, pp_size=1,
            mesh=None, device=dev,
        )

    # Multi-rank: validate and build the mesh.
    expected = dp_size * tp_size * pp_size * ep_size
    if expected != world_size:
        raise ValueError(
            f"dp({dp_size}) × tp({tp_size}) × pp({pp_size}) × ep({ep_size}) = {expected} "
            f"must equal world_size({world_size}).\n"
            "Adjust parallelism config to match the number of available GPUs."
        )

    mesh_shape: List[int] = [dp_size]
    mesh_dim_names: List[str] = ["dp"]
    if tp_size > 1:
        mesh_shape.append(tp_size)
        mesh_dim_names.append("tp")
    if pp_size > 1:
        mesh_shape.append(pp_size)
        mesh_dim_names.append("pp")
    mesh_shape.append(ep_size)
    mesh_dim_names.append("ep")

    mesh = init_device_mesh(
        device_type,
        tuple(mesh_shape),
        mesh_dim_names=tuple(mesh_dim_names),
    )
    local_device_idx = rank % max(torch.cuda.device_count(), 1)
    dev = torch.device(f"{device_type}:{local_device_idx}")
    return ParallelTopology(
        world_size=world_size, rank=rank,
        dp_size=dp_size, ep_size=ep_size, tp_size=tp_size, pp_size=pp_size,
        mesh=mesh, device=dev,
    )
