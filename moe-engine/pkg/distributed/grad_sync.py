"""
pkg/distributed/grad_sync.py
============================

Gradient synchronisation for parameters that are replicated across ranks
which see different data (F21).

Which parameters, over which groups
-----------------------------------
* **Non-expert parameters over the EP group.** EP ranks process different
  tokens, so for every parameter outside a ``DistributedMoELayer``'s
  ``experts`` -- the router, dense layers, and tensor-parallel weight shards
  of dense layers alike -- the EP group acts as data parallelism and each
  rank's gradient is a partial sum. Expert parameters are EP-sharded; the
  EP all-to-all backward already delivers every rank's contribution to them.
* **Sequence-parallel parameters over the TP group.** Inside a
  sequence-parallel region each TP rank sees a different shard of the
  sequence, so a replicated parameter used there (e.g. the ``next_weight``
  of ``scatter_to_sequence_parallel``) holds only its shard's partial
  gradient. Such parameters must be marked with
  :func:`mark_sequence_parallel`. Everywhere else TP ranks see identical
  data and hold identical gradients, so nothing else is reduced over TP
  (doing so would multiply those gradients by ``tp_size``).
* **Never the DP group.** Data parallelism is owned by FSDP2
  (:func:`~pkg.distributed.data_parallel.apply_fsdp2`), which reduces over
  the DP mesh. The EP and TP groups share only the rank itself with the DP
  group, so this module cannot double-reduce anything FSDP2 reduces. FSDP2
  gradients are DTensors; their local shard is reduced.

Loss-normalisation convention
-----------------------------
The global objective these reductions implement is::

    L = (1 / dp_size) * sum over all ranks r of L_r

where ``L_r`` is whatever loss rank ``r`` computes on its own tokens (for
example a mean over them). That is: a **sum** over EP ranks and over
sequence-parallel shards, a **mean** over DP replicas (FSDP2 averages).
Expert gradients already have exactly this form (the dispatch/combine
backward sums over EP ranks; FSDP2 averages over DP), so all reductions here
use ``SUM``. To train on the mean over EP ranks as well, divide each rank's
loss by ``ep_size``.

Usage
-----
Call :func:`sync_replicated_grads` once per optimizer step, after all
micro-batch backward passes and before gradient clipping and
``optimizer.step()`` (Megatron's ``finalize_model_grads`` position). Every
rank must call it on the same model structure.

Initialisation (F14-init)
-------------------------
:func:`broadcast_replicated_params` makes every replicated parameter
identical across the ranks that replicate it, once, right after model
construction (before FSDP2 wrapping and optimizer creation). Without it,
ranks seeded differently start from different replicas, and no later step
reconciles them. Broadcasts go from each group's lowest rank:

* over DP: every parameter (DP ranks are full replicas);
* over EP: non-expert parameters (expert weights are EP shards);
* over TP: every parameter not marked as a TP shard via
  :func:`mark_tensor_parallel_shard` (``ColumnParallelLinear`` /
  ``RowParallelLinear`` mark their weights).

Pipeline stages hold different layers and are never involved.

Public API
----------
    mark_sequence_parallel(param)
    mark_tensor_parallel_shard(param)
    sync_replicated_grads(model, topology)
    broadcast_replicated_params(model, topology)
"""

from __future__ import annotations

from typing import Optional, Set

import torch
import torch.distributed as dist
import torch.nn as nn

from pkg.distributed.mesh import ParallelTopology, tp_process_group

try:
    from torch.distributed.tensor import DTensor
except Exception:  # pragma: no cover
    DTensor = None  # type: ignore

__all__ = [
    "mark_sequence_parallel",
    "mark_tensor_parallel_shard",
    "sync_replicated_grads",
    "broadcast_replicated_params",
    "SEQUENCE_PARALLEL_ATTR",
    "TENSOR_PARALLEL_SHARD_ATTR",
]

SEQUENCE_PARALLEL_ATTR = "sequence_parallel"
TENSOR_PARALLEL_SHARD_ATTR = "tensor_parallel_shard"


def mark_sequence_parallel(param: nn.Parameter) -> nn.Parameter:
    """Mark a replicated parameter as used inside a sequence-parallel region,
    so :func:`sync_replicated_grads` sums its gradient over the TP group."""
    setattr(param, SEQUENCE_PARALLEL_ATTR, True)
    return param


def mark_tensor_parallel_shard(param: nn.Parameter) -> nn.Parameter:
    """Mark a parameter as a tensor-parallel shard (different on every TP
    rank by design), so :func:`broadcast_replicated_params` never overwrites
    it with another rank's shard."""
    setattr(param, TENSOR_PARALLEL_SHARD_ATTR, True)
    return param


def _expert_param_ids(model: nn.Module) -> Set[int]:
    from pkg.distributed.moe_layer import DistributedMoELayer

    ids: Set[int] = set()
    for module in model.modules():
        if isinstance(module, DistributedMoELayer):
            ids.update(id(p) for p in module.experts.parameters())
    return ids


def _ep_group(topology: ParallelTopology) -> "Optional[dist.ProcessGroup]":
    if topology.ep_size == 1:
        return None
    if topology.mesh is None:
        raise RuntimeError("sync_replicated_grads: ep_size > 1 requires a DeviceMesh topology")
    return topology.mesh["ep"].get_group()


def _local(t: torch.Tensor) -> torch.Tensor:
    if DTensor is not None and isinstance(t, DTensor):
        return t.to_local()
    return t


def sync_replicated_grads(model: nn.Module, topology: ParallelTopology) -> None:
    """Sum replicated parameters' gradients over the ranks that saw
    different data for them; see the module docstring for which parameters,
    which groups, and the loss convention. No-op when ``ep_size`` and
    ``tp_size`` are both 1 (whatever ``dp_size`` is) or when
    ``torch.distributed`` is not initialised.

    A parameter whose ``.grad`` is ``None`` on this rank still takes part,
    with a zero gradient, so every rank issues the same collectives.
    """
    if not dist.is_initialized():
        return
    ep_group = _ep_group(topology)
    tp_group = tp_process_group(topology) if topology.tp_size > 1 else None
    if ep_group is None and tp_group is None:
        return

    experts = _expert_param_ids(model)
    for p in model.parameters():
        over_ep = ep_group is not None and id(p) not in experts
        over_tp = tp_group is not None and getattr(p, SEQUENCE_PARALLEL_ATTR, False)
        if not (over_ep or over_tp):
            continue
        if p.grad is None:
            p.grad = torch.zeros_like(p)
        grad = _local(p.grad)
        if over_ep:
            dist.all_reduce(grad, op=dist.ReduceOp.SUM, group=ep_group)
        if over_tp:
            dist.all_reduce(grad, op=dist.ReduceOp.SUM, group=tp_group)


def broadcast_replicated_params(model: nn.Module, topology: ParallelTopology) -> None:
    """Make replicated parameters identical across the ranks that replicate
    them, broadcasting from each group's lowest rank (F14-init); see the
    module docstring for which parameters go over which group. Call once
    after model construction, before FSDP2 wrapping and optimizer creation.
    No-op when ``torch.distributed`` is not initialised.
    """
    if not dist.is_initialized():
        return
    experts = _expert_param_ids(model)
    plan = []
    if topology.dp_size > 1:
        if topology.mesh is None:
            raise RuntimeError("broadcast_replicated_params: dp_size > 1 requires a DeviceMesh")
        plan.append((topology.mesh["dp"].get_group(), lambda p: True))
    ep_group = _ep_group(topology)
    if ep_group is not None:
        plan.append((ep_group, lambda p: id(p) not in experts))
    if topology.tp_size > 1:
        plan.append(
            (
                tp_process_group(topology),
                lambda p: not getattr(p, TENSOR_PARALLEL_SHARD_ATTR, False),
            )
        )
    with torch.no_grad():
        for group, wanted in plan:
            src = dist.get_global_rank(group, 0)
            for p in model.parameters():
                if wanted(p):
                    dist.broadcast(_local(p.data), src=src, group=group)
