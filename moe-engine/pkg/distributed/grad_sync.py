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
  (:func:`~pkg.distributed.data_parallel.apply_fsdp2`), which SUM-reduces
  over the DP mesh. The EP and TP groups share only the rank itself with the
  DP group, so this module cannot double-reduce anything FSDP2 reduces.
  FSDP2 gradients are DTensors; their local shard is reduced.

Training objective (ADR-011, F22)
---------------------------------
The objective is the **mean over the global batch's loss-bearing tokens**::

    L = (1 / N_global) * sum over loss-bearing tokens t of loss_t

where the global batch is every micro-batch of one optimizer step on every
data-distinct rank (DP x EP). TP ranks of one (dp, ep) position hold the
same tokens and compute the loss on TP-replicated outputs, so each token
counts once. It does not depend on the (dp, ep, tp/sp) layout, on how many
loss-bearing tokens each rank holds, or on the accumulation count.

How it is implemented:

* each micro-batch backpropagates its rank's **sum** of per-token losses
  (no division);
* every cross-rank gradient reduction is a **SUM**: over EP and the TP group
  here, over DP in FSDP2 (``apply_fsdp2`` sets a divide factor of 1);
* once per step, :func:`finalize_grads` runs :func:`sync_replicated_grads`
  and divides every gradient by ``N_global`` from :func:`global_token_count`.

Expert gradients have the same form: the dispatch/combine backward sums over
EP ranks and FSDP2 sums over DP.

Usage
-----
Call :func:`finalize_grads` once per optimizer step, after all micro-batch
backward passes and before gradient clipping and ``optimizer.step()``
(Megatron's ``finalize_model_grads`` position). Every rank must call it on
the same model structure. :func:`sync_replicated_grads` alone (no
normalisation) is what it runs first.

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
    sum_over_data_ranks(tensor, topology) -> tensor
    global_token_count(local_count, topology) -> int
    finalize_grads(model, topology, num_tokens)
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
    "sum_over_data_ranks",
    "global_token_count",
    "finalize_grads",
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


def sum_over_data_ranks(t: torch.Tensor, topology: ParallelTopology) -> torch.Tensor:
    """SUM-all-reduce ``t`` in place over the DP and EP groups, the ranks
    that hold different tokens, and return it. Never over TP: TP ranks hold
    the same tokens. One all-reduce per non-trivial group; a no-op when
    ``torch.distributed`` is not initialised."""
    if not dist.is_initialized():
        return t
    if topology.dp_size > 1:
        if topology.mesh is None:
            raise RuntimeError("sum_over_data_ranks: dp_size > 1 requires a DeviceMesh topology")
        dist.all_reduce(t, op=dist.ReduceOp.SUM, group=topology.mesh["dp"].get_group())
    ep_group = _ep_group(topology)
    if ep_group is not None:
        dist.all_reduce(t, op=dist.ReduceOp.SUM, group=ep_group)
    return t


def global_token_count(local_count: int, topology: ParallelTopology) -> int:
    """Number of loss-bearing tokens in the global batch: ``local_count``
    (this rank's count over all micro-batches of the step) summed over the
    data-distinct ranks (:func:`sum_over_data_ranks`)."""
    count = torch.tensor([int(local_count)], dtype=torch.int64, device=topology.device)
    return int(sum_over_data_ranks(count, topology).item())


def finalize_grads(model: nn.Module, topology: ParallelTopology, num_tokens: int) -> None:
    """Turn accumulated per-token-loss **sums** into the gradient of the
    global-batch mean (ADR-011): :func:`sync_replicated_grads`, then divide
    every gradient by ``num_tokens`` (from :func:`global_token_count`).

    Call once per optimizer step, after the last micro-batch's backward and
    before clipping. A step with no loss-bearing tokens anywhere has zero
    gradients and is left as is.
    """
    sync_replicated_grads(model, topology)
    if num_tokens <= 0:
        return
    with torch.no_grad():
        for p in model.parameters():
            if p.grad is not None:
                _local(p.grad).div_(num_tokens)


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
