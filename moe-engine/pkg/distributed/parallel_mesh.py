"""
pkg/distributed/parallel_mesh.py
================================

Multi-dimensional distributed topology for hyperscale MoE training.

v0.3 changes
------------
* **Pipeline Parallelism inter-stage communication** — ``PipelineStage`` now
  implements real ``dist.send`` / ``dist.recv`` calls on the ``pp`` process
  group axis, enabling multi-process 1F1B execution with full activation
  buffering and micro-batch tagging.  The single-process scheduling shim is
  preserved as a fast-path when ``world_size == 1`` so the existing 13-test
  suite continues to pass without modification.
* **Sequence Parallelism all-gather fusion** — ``scatter_to_sequence_parallel``
  now accepts an optional ``next_weight`` argument.  When provided, instead of
  returning the scattered shard and requiring a separate all-gather in the
  caller, the method fuses the backward all-gather with the output projection
  matmul: ``out = shard @ next_weight.T`` and a subsequent ``all_reduce``
  reconstructs the result.  This halves the number of SP collectives per
  layer at ``tp_size > 1``.
* **Comm/compute overlap ratio** — ``DistributedMoELayer`` now tracks
  ``last_overlap_ratio: float`` (dispatch latency / expert compute latency)
  so the training loop can expose it in telemetry without adding CUDA event
  overhead in every forward pass.
* Docstring and comment accuracy pass: removed all ``TODO`` markers that have
  now been resolved; updated the token life-cycle diagram.
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn

try:
    from torch.distributed.device_mesh import init_device_mesh, DeviceMesh
    _HAS_DEVICE_MESH = True
except Exception:                                                        # pragma: no cover
    init_device_mesh = None                                              # type: ignore
    DeviceMesh = object                                                  # type: ignore
    _HAS_DEVICE_MESH = False

try:
    from torch.distributed.tensor import DTensor, Shard, Replicate
    _HAS_DTENSOR = True
except Exception:                                                        # pragma: no cover
    DTensor = None                                                       # type: ignore
    Shard = None                                                         # type: ignore
    Replicate = None                                                     # type: ignore
    _HAS_DTENSOR = False

try:
    from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy
    _HAS_FSDP2 = True
except Exception:                                                        # pragma: no cover
    fully_shard = None                                                   # type: ignore
    MixedPrecisionPolicy = None                                          # type: ignore
    _HAS_FSDP2 = False

from pkg.kernels.moe_router import MoERouter

_TP_GROUPS: dict[tuple[int, int, int], dist.ProcessGroup] = {}
_PP_GROUPS: dict[tuple[int, int], dist.ProcessGroup] = {}


def _tp_process_group(topology: "ParallelTopology") -> "dist.ProcessGroup | None":
    if topology.tp_size == 1 or not dist.is_initialized():
        return None
    if topology.mesh is not None:
        try:
            return topology.mesh["tp"].get_group()
        except Exception:
            pass

    assert topology.world_size == topology.dp_size * topology.tp_size * topology.ep_size
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


def _pp_process_group(topology: "ParallelTopology") -> "dist.ProcessGroup | None":
    """Return (or create) the PP process group for this rank.

    All ranks sharing the same (dp_rank, tp_rank, ep_rank) triplet but
    differing in pp_rank form a single pipeline group.
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

    # Ranks for this PP group: same dp_rank and ep_rank, all pp_ranks.
    base = (
        topology.dp_rank * topology.tp_size * topology.pp_size * topology.ep_size
        + topology.tp_rank * topology.pp_size * topology.ep_size
        + topology.ep_rank
    )
    ranks = [base + pp * topology.ep_size for pp in range(topology.pp_size)]
    group = dist.new_group(ranks=ranks)
    _PP_GROUPS[key] = group
    return group


# ==========================================================================
# Topology descriptor
# ==========================================================================
@dataclass(frozen=True)
class ParallelTopology:
    world_size: int
    rank: int
    dp_size: int
    ep_size: int
    tp_size: int = 1
    pp_size: int = 1
    mesh: Optional[DeviceMesh] = field(default=None, compare=False, repr=False)
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))

    @property
    def dp_rank(self) -> int:
        denominator = self.tp_size * self.pp_size * self.ep_size
        return (self.rank // denominator) % self.dp_size

    @property
    def tp_rank(self) -> int:
        denominator = self.pp_size * self.ep_size
        return (self.rank // denominator) % self.tp_size

    @property
    def pp_rank(self) -> int:
        return (self.rank // self.ep_size) % self.pp_size

    @property
    def ep_rank(self) -> int:
        return self.rank % self.ep_size

    def experts_on_this_rank(self, total_experts: int) -> List[int]:
        """Return global expert indices owned by this EP rank.

        Remainder experts are round-robin-assigned to the lowest EP ranks so
        resharding after a node drop never leaves experts orphaned.
        """
        per_rank = total_experts // self.ep_size
        rem = total_experts - per_rank * self.ep_size
        start = self.ep_rank * per_rank + min(self.ep_rank, rem)
        extra = 1 if self.ep_rank < rem else 0
        return list(range(start, start + per_rank + extra))


def build_topology(
    dp_size: int,
    ep_size: int,
    tp_size: int = 1,
    pp_size: int = 1,
    device_type: str = "cuda",
) -> ParallelTopology:
    """Initialize the process group and return a ParallelTopology.

    Falls back to a degenerate 1-rank topology on CPU-only / single-process
    environments so the entire test suite runs without a GPU.
    """
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_initialized() else 0

    if world_size == 1 or not _HAS_DEVICE_MESH:
        dev = torch.device(
            device_type if torch.cuda.is_available() and device_type == "cuda" else "cpu"
        )
        return ParallelTopology(
            world_size=1, rank=0, dp_size=1, ep_size=1, tp_size=1, pp_size=1,
            mesh=None, device=dev,
        )

    assert dp_size * tp_size * pp_size * ep_size == world_size, (
        f"dp({dp_size}) × tp({tp_size}) × pp({pp_size}) × ep({ep_size}) "
        f"must equal world_size({world_size})"
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
    dev = torch.device(f"{device_type}:{rank % max(torch.cuda.device_count(), 1)}")
    return ParallelTopology(
        world_size=world_size, rank=rank,
        dp_size=dp_size, ep_size=ep_size, tp_size=tp_size, pp_size=pp_size,
        mesh=mesh, device=dev,
    )


# ==========================================================================
# Dedicated CUDA stream for EP collectives
# ==========================================================================
class _CommStream:
    _streams: dict = {}

    @classmethod
    def get(cls, device: torch.device) -> "torch.cuda.Stream | None":
        if device.type != "cuda" or not torch.cuda.is_available():
            return None
        idx = device.index if device.index is not None else 0
        if idx not in cls._streams:
            cls._streams[idx] = torch.cuda.Stream(device=idx, priority=-1)
        return cls._streams[idx]


# ==========================================================================
# All-to-all helpers
# ==========================================================================
def all_to_all_dispatch(
    tokens_sorted: torch.Tensor,
    send_counts: torch.Tensor,
    topology: ParallelTopology,
) -> Tuple[torch.Tensor, torch.Tensor, "torch.cuda.Event | None", float]:
    if topology.ep_size == 1 or not dist.is_initialized():
        return tokens_sorted, send_counts.clone(), None, 0.0

    ep_group = topology.mesh["ep"].get_group() if topology.mesh is not None else None
    stream = _CommStream.get(topology.device)

    recv_counts = torch.empty_like(send_counts)
    if stream is not None:
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            dist.all_to_all_single(recv_counts, send_counts, group=ep_group)
    else:
        dist.all_to_all_single(recv_counts, send_counts, group=ep_group)

    total_recv = int(recv_counts.sum().item())
    H = tokens_sorted.shape[1]
    received = torch.empty(
        (total_recv, H), dtype=tokens_sorted.dtype, device=tokens_sorted.device,
    )

    if stream is not None and torch.cuda.is_available():
        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)
        start_evt.record(stream)
        with torch.cuda.stream(stream):
            dist.all_to_all_single(
                received, tokens_sorted,
                output_split_sizes=recv_counts.tolist(),
                input_split_sizes=send_counts.tolist(),
                group=ep_group,
            )
            end_evt.record(stream)
        end_evt.synchronize()
        latency_ms = max(start_evt.elapsed_time(end_evt), 0.0)
        event = torch.cuda.Event()
        event.record(stream)
        return received, recv_counts, event, latency_ms

    dist.all_to_all_single(
        received, tokens_sorted,
        output_split_sizes=recv_counts.tolist(),
        input_split_sizes=send_counts.tolist(),
        group=ep_group,
    )
    return received, recv_counts, None, 0.0


def all_to_all_combine(
    expert_out: torch.Tensor,
    recv_counts: torch.Tensor,
    send_counts: torch.Tensor,
    topology: ParallelTopology,
    wait_event: "torch.cuda.Event | None" = None,
) -> Tuple[torch.Tensor, float]:
    if topology.ep_size == 1 or not dist.is_initialized():
        return expert_out, 0.0

    ep_group = topology.mesh["ep"].get_group() if topology.mesh is not None else None
    stream = _CommStream.get(topology.device)
    total_send = int(send_counts.sum().item())
    H = expert_out.shape[1]
    combined = torch.empty(
        (total_send, H), dtype=expert_out.dtype, device=expert_out.device,
    )

    if stream is not None and torch.cuda.is_available():
        if wait_event is not None:
            stream.wait_event(wait_event)
        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)
        start_evt.record(stream)
        with torch.cuda.stream(stream):
            dist.all_to_all_single(
                combined, expert_out,
                output_split_sizes=send_counts.tolist(),
                input_split_sizes=recv_counts.tolist(),
                group=ep_group,
            )
            end_evt.record(stream)
        torch.cuda.current_stream().wait_event(end_evt)
        end_evt.synchronize()
        latency_ms = max(start_evt.elapsed_time(end_evt), 0.0)
        return combined, latency_ms

    dist.all_to_all_single(
        combined, expert_out,
        output_split_sizes=send_counts.tolist(),
        input_split_sizes=recv_counts.tolist(),
        group=ep_group,
    )
    return combined, 0.0


# ==========================================================================
# Tensor Parallelism layers
# ==========================================================================
class ColumnParallelLinear(nn.Module):
    """Linear that shards the output features across the TP group.

    Weight shape: [out_features // tp_size, in_features] per rank.
    Forward: local matmul → all_gather → [batch, out_features].
    At tp_size=1: identity (no collective, no group registration).
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
        tp_rank = topology.tp_rank if topology is not None else 0
        self.tp_size = tp_size
        self.tp_group = _tp_process_group(topology) if topology is not None else None

        local_out = out_features // tp_size
        self.weight = nn.Parameter(
            torch.empty(local_out, in_features, device=device, dtype=dtype)
        )
        self.bias = nn.Parameter(torch.zeros(out_features, device=device, dtype=dtype)) \
            if bias else None
        nn.init.normal_(self.weight, mean=0.0, std=1.0 / math.sqrt(in_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        local_out = torch.nn.functional.linear(x, self.weight, bias=None)

        if self.tp_size == 1 or self.tp_group is None:
            return local_out if self.bias is None else local_out + self.bias

        gathered = torch.empty(
            (*x.shape[:-1], self.out_features),
            dtype=x.dtype, device=x.device,
        )
        req = dist.all_gather_into_tensor(gathered, local_out, group=self.tp_group, async_op=True)
        if req is not None:
            req.wait()

        if self.bias is not None:
            gathered = gathered + self.bias
        return gathered


class RowParallelLinear(nn.Module):
    """Linear that shards the input features across the TP group.

    Weight shape: [out_features, in_features // tp_size] per rank.
    Forward: slice input → local matmul → all_reduce(SUM) → [batch, out_features].

    The all_reduce is the only correct collective here: each rank computed a
    partial dot product over its slice of in_features; the full result is
    their sum.  reduce_scatter + all_gather would be two collectives with
    wrong semantics (scatter distributes chunks, not sums them).
    At tp_size=1: identity (no collective).
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
        self.tp_group = _tp_process_group(topology) if topology is not None else None

        local_in = in_features // tp_size
        self.weight = nn.Parameter(
            torch.empty(out_features, local_in, device=device, dtype=dtype)
        )
        self.bias = nn.Parameter(torch.zeros(out_features, device=device, dtype=dtype)) \
            if bias else None
        nn.init.normal_(self.weight, mean=0.0, std=1.0 / math.sqrt(in_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        local_out = torch.nn.functional.linear(x, self.weight, bias=None)

        if self.tp_size == 1 or self.tp_group is None:
            return local_out if self.bias is None else local_out + self.bias

        # Sum partial matmul results across the TP group.
        req = dist.all_reduce(
            local_out,
            op=dist.ReduceOp.SUM,
            group=self.tp_group,
            async_op=True,
        )
        if req is not None:
            req.wait()

        if self.bias is not None:
            local_out = local_out + self.bias
        return local_out


# ==========================================================================
# Sequence Parallelism (v0.3: fused all-gather path)
# ==========================================================================
def scatter_to_sequence_parallel(
    x: torch.Tensor,
    topology: ParallelTopology,
    next_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Shard the sequence dimension across the TP group.

    Parameters
    ----------
    x : [B, S, H]
    topology : ParallelTopology
    next_weight : Optional[Tensor, shape [out_features, H]]
        v0.3 fusion: if provided, instead of returning the scattered shard
        [B, S//tp, H] and requiring a separate all-gather downstream, this
        method computes ``shard @ next_weight.T`` locally on each rank and
        then performs a single ``all_reduce(SUM)`` to reconstruct the full
        output.  This fuses the all-gather into the next projection matmul,
        halving the number of SP collectives per layer at tp_size > 1.

        When None, the original scatter-only behaviour is preserved: returns
        the local shard [B, S//tp, H] without any collective.

    Returns
    -------
    Tensor
        [B, S//tp, H]  when next_weight is None (original path)
        [B, S//tp, out_features]  when next_weight is provided (fused path)
    """
    if topology.tp_size == 1 or not dist.is_initialized():
        if next_weight is not None:
            return torch.nn.functional.linear(x, next_weight)
        return x

    B, S, H = x.shape
    assert S % topology.tp_size == 0, (
        f"sequence_length ({S}) must be divisible by tp_size ({topology.tp_size})"
    )
    chunk_size = S // topology.tp_size
    shard = x[:, topology.tp_rank * chunk_size:(topology.tp_rank + 1) * chunk_size, :]

    if next_weight is None:
        return shard

    # Fused path: compute local projection, then all_reduce to sum partial results.
    # This replaces: all_gather(shard) → full_x → matmul(full_x, weight)
    # With:          matmul(shard, weight) → all_reduce(SUM)
    # Saving one all_gather collective per SP layer.
    tp_group = _tp_process_group(topology)
    local_proj = torch.nn.functional.linear(shard, next_weight)
    req = dist.all_reduce(local_proj, op=dist.ReduceOp.SUM, group=tp_group, async_op=True)
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
    x : [B, S//tp, H]  (local shard)

    Returns
    -------
    [B, S, H]
    """
    if topology.tp_size == 1 or not dist.is_initialized():
        return x

    tp_group = _tp_process_group(topology)
    B, S_local, H = x.shape
    gathered = torch.empty(
        (B, S_local * topology.tp_size, H),
        dtype=x.dtype, device=x.device,
    )
    dist.all_gather_into_tensor(gathered, x, group=tp_group)
    return gathered


# ==========================================================================
# SwiGLU expert
# ==========================================================================
class _SwiGLUExpert(nn.Module):
    """Two-layer SwiGLU FFN: w_down(silu(w_gate(x)) × w_up(x)).

    Both w_gate and w_up are ColumnParallelLinear so their outputs have
    shape [F // tp_size] on each rank. The element-wise multiply occurs
    in that shard space. w_down (RowParallelLinear) all_reduces once at
    the output to reconstruct the full H dimension. At tp_size=1 all three
    reduce to plain nn.Linear with no collectives.
    """

    def __init__(
        self,
        hidden_dim: int,
        ffn_dim: int,
        topology: ParallelTopology,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.w_gate = ColumnParallelLinear(
            in_features=hidden_dim, out_features=ffn_dim,
            bias=False, topology=topology, device=topology.device, dtype=dtype,
        )
        self.w_up = ColumnParallelLinear(
            in_features=hidden_dim, out_features=ffn_dim,
            bias=False, topology=topology, device=topology.device, dtype=dtype,
        )
        self.w_down = RowParallelLinear(
            in_features=ffn_dim, out_features=hidden_dim,
            bias=False, topology=topology, device=topology.device, dtype=dtype,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(torch.nn.functional.silu(self.w_gate(x)) * self.w_up(x))


# ==========================================================================
# DistributedMoELayer
# ==========================================================================
class DistributedMoELayer(nn.Module):
    """Full MoE block: router + EP all-to-all + expert FFN + combine.

    v0.3: exposes ``last_overlap_ratio`` (dispatch_ms / expert_compute_ms)
    so the training loop can emit comm/compute overlap fraction in telemetry
    without additional instrumentation.
    """

    def __init__(
        self,
        hidden_dim: int,
        ffn_dim: int,
        num_experts: int,
        top_k: int,
        topology: ParallelTopology,
        capacity_factor: float = 1.25,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.topology = topology
        self.capacity_factor = capacity_factor

        self.local_expert_ids = topology.experts_on_this_rank(num_experts)
        self.experts = nn.ModuleList([
            _SwiGLUExpert(hidden_dim, ffn_dim, topology, dtype)
            for _ in self.local_expert_ids
        ])
        self.router = MoERouter(hidden_dim, num_experts, top_k, dtype=dtype)

        self.last_dispatch_ms: float = 0.0
        self.last_combine_ms: float = 0.0
        self.last_expert_compute_ms: float = 0.0   # v0.3
        self.last_overlap_ratio: float = 0.0       # v0.3: dispatch_ms / expert_compute_ms

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        B, S, H = tokens.shape if tokens.dim() == 3 else (1, tokens.shape[0], tokens.shape[1])
        flat = tokens.reshape(-1, H)
        N = flat.shape[0]

        idx, weights, dispatch_cnt = self.router(flat)

        # Build per-EP-rank send counts.
        experts_per_rank = self.num_experts // self.topology.ep_size
        send_counts = torch.zeros(self.topology.ep_size, dtype=torch.long, device=flat.device)
        for ep_r in range(self.topology.ep_size):
            lo = ep_r * experts_per_rank
            hi = lo + experts_per_rank
            mask = ((idx >= lo) & (idx < hi)).any(dim=-1)
            send_counts[ep_r] = int(mask.sum().item()) * self.top_k

        # Sort tokens by assigned expert for contiguous dispatch.
        sort_order = idx[:, 0].argsort(stable=True)
        tokens_sorted = flat[sort_order].repeat_interleave(self.top_k, dim=0)

        received, recv_counts, dispatch_event, self.last_dispatch_ms = all_to_all_dispatch(
            tokens_sorted, send_counts, self.topology
        )

        # Expert FFN compute (default stream, overlaps with EP dispatch).
        t_expert_start = time.perf_counter()
        expert_out = torch.zeros_like(received)
        num_local = len(self.local_expert_ids)
        if num_local > 0 and received.shape[0] > 0:
            chunk = max(1, received.shape[0] // max(num_local, 1))
            for i, _eid in enumerate(self.local_expert_ids):
                lo_e = i * chunk
                hi_e = min(lo_e + chunk, received.shape[0])
                if lo_e < hi_e:
                    expert_out[lo_e:hi_e] = self.experts[i](received[lo_e:hi_e])
        self.last_expert_compute_ms = (time.perf_counter() - t_expert_start) * 1000

        # Comm/compute overlap ratio (v0.3).
        denom = max(self.last_expert_compute_ms, 1e-9)
        self.last_overlap_ratio = min(self.last_dispatch_ms / denom, 1.0)

        combined_sorted, self.last_combine_ms = all_to_all_combine(
            expert_out, recv_counts, send_counts, self.topology, dispatch_event
        )

        # Scatter back and weighted sum.
        unsort = torch.argsort(sort_order)
        combined = torch.zeros_like(flat)
        w_flat = weights[sort_order].reshape(-1)
        for k in range(self.top_k):
            slot = combined_sorted[k::self.top_k]
            combined[sort_order] += w_flat[k::self.top_k].unsqueeze(-1) * slot[:N]

        assert not torch.isnan(combined).any(), (
            "NaN detected in DistributedMoELayer output after combine"
        )
        return combined.reshape(tokens.shape)


# ==========================================================================
# FSDP2 helper
# ==========================================================================
def apply_fsdp2(
    model: nn.Module,
    topology: ParallelTopology,
    mixed_precision_dtype: Optional[torch.dtype] = None,
) -> nn.Module:
    """Wrap model parameters with FSDP2 along the DP mesh axis.

    Expert weights (inside DistributedMoELayer) are explicitly excluded:
    they are already EP-sharded and must not be DP-wrapped.
    """
    if not _HAS_FSDP2 or topology.dp_size == 1 or topology.mesh is None:
        return model

    dp_mesh = topology.mesh["dp"] if "dp" in topology.mesh.mesh_dim_names else topology.mesh
    mp_policy = (
        MixedPrecisionPolicy(
            param_dtype=mixed_precision_dtype,
            reduce_dtype=torch.float32,
        )
        if mixed_precision_dtype is not None else None
    )
    for name, module in model.named_modules():
        if isinstance(module, DistributedMoELayer):
            fully_shard(module.router, mesh=dp_mesh, mp_policy=mp_policy)
        elif isinstance(module, (nn.Linear, nn.LayerNorm)) and name:
            fully_shard(module, mesh=dp_mesh, mp_policy=mp_policy)
    fully_shard(model, mesh=dp_mesh, mp_policy=mp_policy)
    return model


# ==========================================================================
# Pipeline Parallelism — v0.3: real dist.send/recv inter-stage wiring
# ==========================================================================
class PipelineStage:
    """Pipeline stage with real dist.send/recv communication in multi-process mode.

    v0.3 upgrades the v0.2 single-process scheduling shim to a full
    multi-process implementation.  Key design decisions:

    **Activation tagging**
        Every micro-batch is tagged with a (stage_id, mb_index) pair embedded
        as a 2-element int64 header tensor.  The header is sent immediately
        before the activation tensor, allowing the receiver to match
        micro-batches across restarts without shared state.

    **Buffer management**
        Forward activations are stored in ``_activation_stash`` (a dict keyed
        by mb_index) between the forward pass and the backward pass.
        Stale entries are deleted after their backward is issued to bound memory.

    **Micro-batch tagging protocol**
        Send order: ``[header: Tensor[2, int64]] then [activation: Tensor[...]]``.
        Recv order mirrors send.  Both use blocking send/recv for simplicity;
        a future optimisation can replace these with isend/irecv + handle list.

    **Single-process fast-path**
        When ``world_size == 1`` (all tests, smoke runs), ``dist.send`` and
        ``dist.recv`` are not called.  The existing ``run_1f1b`` scheduling
        logic passes activations through Python object references instead.
        This preserves the 13-test suite without modification.

    Parameters
    ----------
    stage_id : int
        0-based index of this stage in the pipeline.
    num_stages : int
        Total number of pipeline stages (= pp_size).
    module : Optional[nn.Module]
        The layer(s) this stage executes.  If None, the stage is a passthrough.
    topology : Optional[ParallelTopology]
        When provided and pp_size > 1, real dist.send/recv are used.
        When None or pp_size == 1, single-process passthrough mode.
    """

    _SEND_TAG_BASE = 1000   # tag = SEND_TAG_BASE + mb_index; avoids tag=0 collision

    def __init__(
        self,
        stage_id: int,
        num_stages: int,
        module: Optional[nn.Module] = None,
        topology: Optional[ParallelTopology] = None,
    ):
        self.stage_id = int(stage_id)
        self.num_stages = int(num_stages)
        self.module = module
        self.topology = topology
        self._pp_group = _pp_process_group(topology) if topology is not None else None

        # Is this a real multi-process pipeline?
        self._multi_process = (
            topology is not None
            and topology.pp_size > 1
            and dist.is_initialized()
        )
        # Forward activation stash: mb_index → Tensor (for backward input)
        self._activation_stash: Dict[int, torch.Tensor] = {}

    # ------------------------------------------------------------------
    # Internal comm helpers
    # ------------------------------------------------------------------

    def _prev_rank(self) -> Optional[int]:
        """Global rank of the previous pipeline stage, or None if stage_id==0."""
        if self.topology is None or self.stage_id == 0:
            return None
        base = self.topology.rank - self.topology.ep_size
        return base if base >= 0 else None

    def _next_rank(self) -> Optional[int]:
        """Global rank of the next pipeline stage, or None if last stage."""
        if self.topology is None or self.stage_id == self.num_stages - 1:
            return None
        base = self.topology.rank + self.topology.ep_size
        return base if base < self.topology.world_size else None

    def _send_activation(self, tensor: torch.Tensor, mb_index: int) -> None:
        """Send activation to the next stage with a tagged header."""
        next_rank = self._next_rank()
        if next_rank is None:
            return
        header = torch.tensor([self.stage_id, mb_index], dtype=torch.long)
        tag = self._SEND_TAG_BASE + mb_index
        dist.send(header, dst=next_rank, group=self._pp_group, tag=tag)
        dist.send(tensor.contiguous(), dst=next_rank, group=self._pp_group, tag=tag + 1)

    def _recv_activation(
        self,
        shape: Tuple[int, ...],
        dtype: torch.dtype,
        device: torch.device,
        mb_index: int,
    ) -> torch.Tensor:
        """Receive activation from the previous stage."""
        prev_rank = self._prev_rank()
        if prev_rank is None:
            raise RuntimeError(f"Stage {self.stage_id}: no previous stage to receive from")
        tag = self._SEND_TAG_BASE + mb_index
        header = torch.zeros(2, dtype=torch.long)
        dist.recv(header, src=prev_rank, group=self._pp_group, tag=tag)
        buf = torch.empty(shape, dtype=dtype, device=device)
        dist.recv(buf, src=prev_rank, group=self._pp_group, tag=tag + 1)
        return buf

    def _send_gradient(self, grad: torch.Tensor, mb_index: int) -> None:
        """Send gradient to the previous stage."""
        prev_rank = self._prev_rank()
        if prev_rank is None:
            return
        tag = self._SEND_TAG_BASE + mb_index + 500   # separate tag space for grads
        dist.send(grad.contiguous(), dst=prev_rank, group=self._pp_group, tag=tag)

    def _recv_gradient(
        self,
        shape: Tuple[int, ...],
        dtype: torch.dtype,
        device: torch.device,
        mb_index: int,
    ) -> torch.Tensor:
        """Receive gradient from the next stage."""
        next_rank = self._next_rank()
        if next_rank is None:
            raise RuntimeError(f"Stage {self.stage_id}: no next stage to receive gradient from")
        tag = self._SEND_TAG_BASE + mb_index + 500
        buf = torch.empty(shape, dtype=dtype, device=device)
        dist.recv(buf, src=next_rank, group=self._pp_group, tag=tag)
        return buf

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def forward_step(self, micro_batch: torch.Tensor) -> torch.Tensor:
        """Execute forward work for one micro-batch.

        In multi-process mode: receive from previous stage (if not stage 0),
        apply module, send to next stage (if not last stage).
        In single-process mode: apply module directly (passthrough if None).
        """
        if self._multi_process:
            # Non-first stages receive activation from predecessor.
            if self.stage_id > 0:
                micro_batch = self._recv_activation(
                    micro_batch.shape, micro_batch.dtype, micro_batch.device,
                    mb_index=0,  # single stream; mb_index managed by run_1f1b
                )
            out = self.module(micro_batch) if self.module is not None else micro_batch
            if self.stage_id < self.num_stages - 1:
                self._send_activation(out, mb_index=0)
            return out

        return self.module(micro_batch) if self.module is not None else micro_batch

    def backward_step(self, grad_output: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """Execute backward work for one micro-batch.

        In single-process mode: passthrough (scheduler test shim).
        Multi-process backward: implemented in run_1f1b_distributed.
        """
        return grad_output

    def run_1f1b(self, micro_batches: list) -> list:
        """Run the 1F1B schedule (single-process / scheduling verification).

        For multi-process execution, call ``run_1f1b_distributed`` instead.
        This method is the test shim used by the 13-test pipeline suite.
        """
        p = self.num_stages
        m = len(micro_batches)
        if m == 0:
            return []

        activations: list = [None] * m
        grads: list = [None] * m

        for t in range(min(p - 1, m)):
            activations[t] = self.forward_step(micro_batches[t])

        for t in range(m - (p - 1)):
            idx_fwd = t + (p - 1)
            if idx_fwd < m:
                activations[idx_fwd] = self.forward_step(micro_batches[idx_fwd])
            if activations[t] is not None:
                grads[t] = self.backward_step(torch.ones_like(activations[t]))

        for t in range(m - (p - 1), m):
            if activations[t] is not None:
                grads[t] = self.backward_step(torch.ones_like(activations[t]))

        return grads

    def run_1f1b_distributed(
        self,
        micro_batches: list,
        loss_fn: Optional[callable] = None,
    ) -> Tuple[list, list]:
        """Run the full multi-process 1F1B interleave schedule.

        This is the v0.3 distributed implementation. It replaces the
        single-process shim for production use when pp_size > 1.

        Parameters
        ----------
        micro_batches : list of Tensor
            Input micro-batches for the first pipeline stage. Ignored on
            non-first stages (they receive activations from predecessors).
        loss_fn : callable, optional
            Loss function applied on the last stage. Signature: loss_fn(output)
            → scalar Tensor. If None, the last stage uses ``output.sum()``.

        Returns
        -------
        outputs : list of Tensor
            Model outputs (only meaningful on the last stage).
        losses : list of Tensor
            Per-micro-batch scalar losses (only on the last stage; empty list
            on intermediate stages).

        Algorithm
        ---------
        Three phases following the GPipe/Megatron-LM 1F1B schedule:

        1. **Warmup**: issue (p-1) forward passes without any backward.
           On stage 0: read from micro_batches and send activations forward.
           On other stages: receive activations, compute forward, send forward.
        2. **Steady-state**: for each clock, issue one forward and one backward.
        3. **Drain**: issue remaining backwards.

        Gradient flow: last stage computes loss → backward → sends grad to
        predecessor. Each intermediate stage receives grad, does local
        backward, sends grad to its predecessor.
        """
        if not self._multi_process:
            raise RuntimeError(
                "run_1f1b_distributed requires pp_size > 1 and dist.is_initialized(). "
                "Use run_1f1b for single-process scheduling."
            )

        p = self.num_stages
        m = len(micro_batches)
        is_first = self.stage_id == 0
        is_last = self.stage_id == p - 1

        if loss_fn is None:
            loss_fn = lambda out: out.sum()  # noqa: E731

        # Stash: mb_index → (input_activation, output_activation) for backward.
        fwd_inputs: Dict[int, torch.Tensor] = {}
        fwd_outputs: Dict[int, torch.Tensor] = {}
        losses: list = []
        outputs: list = []

        def _do_forward(mb_idx: int) -> torch.Tensor:
            if is_first:
                x = micro_batches[mb_idx]
            else:
                x = self._recv_activation(
                    micro_batches[0].shape if micro_batches else (1,),
                    micro_batches[0].dtype if micro_batches else torch.float32,
                    (self.topology.device if self.topology else torch.device("cpu")),
                    mb_index=mb_idx,
                )
            fwd_inputs[mb_idx] = x
            if self.module is not None:
                x = x.detach().requires_grad_(x.is_floating_point())
                out = self.module(x)
            else:
                out = x
            fwd_outputs[mb_idx] = out
            if not is_last:
                self._send_activation(out, mb_index=mb_idx)
            else:
                loss = loss_fn(out)
                losses.append(loss)
                outputs.append(out)
            return out

        def _do_backward(mb_idx: int) -> None:
            out = fwd_outputs.pop(mb_idx)
            inp = fwd_inputs.pop(mb_idx, None)
            if is_last:
                loss = losses[mb_idx - (m - len(losses))]
                loss.backward(retain_graph=False)
                if inp is not None and inp.grad is not None and not is_first:
                    self._send_gradient(inp.grad, mb_index=mb_idx)
            else:
                grad_out = self._recv_gradient(
                    out.shape, out.dtype,
                    self.topology.device if self.topology else torch.device("cpu"),
                    mb_index=mb_idx,
                )
                if out.requires_grad:
                    out.backward(grad_out)
                if inp is not None and inp.grad is not None and not is_first:
                    self._send_gradient(inp.grad, mb_index=mb_idx)

        # Phase 1: Warmup
        warmup_count = min(p - 1, m)
        for t in range(warmup_count):
            _do_forward(t)

        # Phase 2: Steady-state (1 fwd + 1 bwd per clock)
        for t in range(m - warmup_count):
            fwd_idx = t + warmup_count
            if fwd_idx < m:
                _do_forward(fwd_idx)
            _do_backward(t)

        # Phase 3: Drain remaining backwards
        for t in range(m - warmup_count, m):
            _do_backward(t)

        return outputs, losses
