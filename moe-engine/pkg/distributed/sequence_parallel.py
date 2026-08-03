"""
pkg/distributed/sequence_parallel.py
======================================

Sequence Parallelism (SP) — activations sharded along the sequence dimension
across the Tensor Parallel (TP) process group.

Sequence Parallelism activates automatically when ``tp_size > 1``.
At ``tp_size == 1`` every function is an identity with zero overhead.

Background
----------
Sequence Parallelism (Korthikanti et al., 2022) shards the sequence dimension
across TP ranks so that the per-rank activation memory for large ``[B, S, H]``
tensors scales as ``S / tp_size`` instead of ``S``. This is essential at long
context lengths (S ≥ 8192) where the unsharded activation alone can OOM a
single device.

Communication patterns
-----------------------
Forward:
    scatter_to_sp(x)         : [B, S, H] → [B, ceil(S/tp), H]    (slice, no comm)
    gather_from_sp(x)        : [B, ceil(S/tp), H] → [B, S, H]    (all_gather)

Fused forward (``next_weight`` provided) — v0.3, corrected in ADR-007
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    scatter_to_sp(x, W)      : [B, S, H] → [B, ceil(S/tp), out]   (matmul, NO comm)

``next_weight`` here is a *replicated* weight — identical on every rank, not
sharded. A per-token ``Linear`` applied to an already sequence-sharded slice
is complete and correct on its own: token ``i``'s output only ever depends on
token ``i``'s input and the (shared) weight, so slicing by sequence commutes
freely with the projection. No collective is needed or correct here — see
ADR-007 for the full derivation and the bug this replaces (a spurious
``all_reduce`` that summed *different tokens'* projections together, which
doesn't reconstruct anything and generally isn't even shape-compatible once
the affected shards differ in size). This fusion's actual saving is
avoiding a would-be ``gather → project`` round trip when the projection
itself doesn't need tensor-parallel sharding; it does **not** implement (and
should not be confused with) Megatron's ``RowParallelLinear + SP``
reduce-scatter fusion, which needs a row-sharded weight differing per rank
and is not implemented anywhere in this codebase — see ADR-007's "Deferred"
section if that pattern is ever needed.

Non-divisible sequence lengths
-------------------------------
``S`` need not be divisible by ``tp_size``. Internally, ``x`` is zero-padded
along the sequence dimension up to ``ceil(S / tp_size) * tp_size`` before
sharding, so every rank holds an equal-sized shard (only the last rank's
shard may contain padding rows). Zero-padding is safe here specifically
*because* every operation in this module is per-token (slicing, and a
per-token ``Linear``) — nothing mixes information across the sequence
dimension, so padding rows never influence real tokens' outputs. This would
NOT be safe to reuse as-is for an operation with cross-token dependencies
(e.g. attention) without a padding mask.

To recover the original (unpadded) sequence length, pass ``orig_seq_len`` to
``gather_from_sequence_parallel`` — it strips the padding after the
all_gather. This is opt-in and explicit by design: forgetting it produces a
visibly-wrong *shape* (padded length) rather than silently-wrong *values*,
which is a much safer failure mode than trying to infer padding implicitly.

Public API
----------
    scatter_to_sequence_parallel(x, topology, next_weight=None)
    gather_from_sequence_parallel(x, topology, orig_seq_len=None)
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.distributed as dist

from pkg.distributed.mesh import ParallelTopology, tp_process_group

__all__ = [
    "scatter_to_sequence_parallel",
    "gather_from_sequence_parallel",
]


def _sp_shard_layout(seq_len: int, tp_size: int) -> Tuple[int, int]:
    """Compute the per-rank shard size and total padding for sharding
    ``seq_len`` tokens evenly across ``tp_size`` ranks.

    Pure arithmetic, no tensors or communication involved — deliberately
    factored out so the padding/shard-boundary logic is fully unit-testable
    on CPU without ``dist.init_process_group``, the same rationale
    ``tests/test_gpu_portability.py`` uses for isolating ``_should_use_triton``
    from real CUDA. This is the one function in this module worth getting
    exactly right in isolation: every other function's correctness reduces
    to "did it slice/pad/strip using these two numbers."

    Returns
    -------
    (chunk, pad_amount) : Tuple[int, int]
        ``chunk`` is the per-rank shard length (``ceil(seq_len / tp_size)``).
        ``pad_amount`` is how many zero rows must be appended to ``seq_len``
        before it divides evenly into ``tp_size`` shards of size ``chunk``
        (zero when ``seq_len`` already divides evenly).
    """
    if tp_size < 1:
        raise ValueError(f"tp_size must be >= 1, got {tp_size}")
    if seq_len < 1:
        raise ValueError(f"seq_len must be >= 1, got {seq_len}")
    chunk = -(-seq_len // tp_size)  # ceil division, no float rounding
    pad_amount = chunk * tp_size - seq_len
    return chunk, pad_amount


def scatter_to_sequence_parallel(
    x: torch.Tensor,
    topology: ParallelTopology,
    next_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Shard the sequence dimension across the TP group.

    Standard path (``next_weight=None``)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Zero-pads ``x`` up to a multiple of ``tp_size`` if needed, then slices
    along the sequence dimension with no communication::

        shard = x_padded[:, rank*chunk:(rank+1)*chunk, :]   # [B, chunk, H]
        return shard

    At ``tp_size == 1``: identity, no collective, no padding.

    Fused projection path (``next_weight`` provided)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Applies a replicated (not tensor-parallel-sharded) weight to the local
    shard. No collective — see this module's docstring and ADR-007 for why
    this is correct (and why an ``all_reduce`` here would not be)::

        shard  = x_padded[:, rank*chunk:(rank+1)*chunk, :]   # [B, chunk, H]
        return shard @ next_weight.T                          # [B, chunk, out]

    Parameters
    ----------
    x : Tensor ``[B, S, H]``
        Full-sequence activations on each rank (replicated before this call).
        ``S`` need not be divisible by ``tp_size``.
    topology : ParallelTopology
    next_weight : Optional Tensor ``[out_features, H]``
        A *replicated* projection weight (identical on every rank). When
        provided, enables the fused path. Do not pass a tensor-parallel
        row-sharded weight here — that is a different, unimplemented fusion
        pattern; see this module's docstring.

    Returns
    -------
    Tensor
        ``[B, ceil(S/tp), H]``            when ``next_weight`` is None
        ``[B, ceil(S/tp), out_features]`` when ``next_weight`` is provided

        When ``S`` is not divisible by ``tp_size``, the last rank's shard
        includes zero-padding rows. Pass ``orig_seq_len=S`` to
        ``gather_from_sequence_parallel`` to strip them back out.
    """
    if topology.tp_size == 1 or not dist.is_initialized():
        if next_weight is not None:
            return torch.nn.functional.linear(x, next_weight)
        return x

    B, S, H = x.shape
    chunk, pad_amount = _sp_shard_layout(S, topology.tp_size)
    if pad_amount > 0:
        pad = x.new_zeros(B, pad_amount, H)
        x = torch.cat([x, pad], dim=1)

    tp_rank = topology.tp_rank
    shard = x[:, tp_rank * chunk : (tp_rank + 1) * chunk, :].contiguous()

    if next_weight is None:
        return shard

    # Fused path: local projection with a replicated weight. No collective —
    # see module docstring and ADR-007. (Prior to ADR-007 this called
    # dist.all_reduce here, which summed different tokens' projections
    # across ranks -- not a reconstruction of anything, and not even
    # shape-compatible once ranks' shards could differ in size. Removed,
    # not replaced: nothing needs to be reduced here.)
    return torch.nn.functional.linear(shard, next_weight)


def gather_from_sequence_parallel(
    x: torch.Tensor,
    topology: ParallelTopology,
    orig_seq_len: Optional[int] = None,
) -> torch.Tensor:
    """Reconstruct the full sequence from per-rank shards via all_gather.

    At ``tp_size == 1``: identity, no collective.

    Parameters
    ----------
    x : Tensor ``[B, S_local, H]``
        Local sequence shard on this TP rank (as produced by
        ``scatter_to_sequence_parallel``).
    orig_seq_len : Optional[int]
        If the original (pre-scatter) sequence length was not divisible by
        ``tp_size``, pass it here to strip the trailing zero-padding rows
        after the all_gather. Required for an exact round-trip in that case;
        a no-op (and safe to omit) whenever the original length already
        divided evenly. Must be <= ``S_local * tp_size``, or this raises —
        that combination can only mean a caller error (wrong ``topology``,
        or a shard that wasn't produced by ``scatter_to_sequence_parallel``
        with this ``topology``).

    Returns
    -------
    Tensor ``[B, S, H]``
        Full-sequence tensor (``S = orig_seq_len`` if given, else
        ``S_local * tp_size``), reconstructed across all TP ranks.
    """
    if topology.tp_size == 1 or not dist.is_initialized():
        if orig_seq_len is not None and x.shape[1] != orig_seq_len:
            raise ValueError(
                f"gather_from_sequence_parallel: orig_seq_len={orig_seq_len} "
                f"does not match x's sequence length ({x.shape[1]}) at "
                "tp_size=1, where no padding is ever applied by "
                "scatter_to_sequence_parallel. This points at a caller bug "
                "(wrong orig_seq_len, or x wasn't produced by "
                "scatter_to_sequence_parallel with this topology)."
            )
        return x

    pg = tp_process_group(topology)
    B, S_local, H = x.shape
    padded_total = S_local * topology.tp_size

    # dist.all_gather_into_tensor concatenates along dim 0 ONLY -- it treats
    # its input as a flat per-rank buffer and places rank r's entire buffer
    # at flat offset r * input.numel() in the output. That means an output
    # tensor shaped (B, S_local*tp_size, H) is only correct when B == 1; for
    # B > 1 it corrupts data across the batch dimension (each rank's whole
    # (B, S_local, H) block lands contiguously rather than being interleaved
    # per-batch the way (B, S_local*tp_size, H) requires). Move the sequence
    # dimension to the front -- where dim-0 concatenation is actually the
    # semantics we want -- gather there, then move it back. See ADR-007.
    x_t = x.transpose(0, 1).contiguous()  # [S_local, B, H]
    gathered_t = torch.empty(
        (padded_total, B, H),
        dtype=x.dtype,
        device=x.device,
    )
    dist.all_gather_into_tensor(gathered_t, x_t, group=pg)
    gathered = gathered_t.transpose(0, 1).contiguous()  # [B, padded_total, H]

    if orig_seq_len is None:
        return gathered
    if orig_seq_len > padded_total:
        raise ValueError(
            f"gather_from_sequence_parallel: orig_seq_len={orig_seq_len} "
            f"exceeds the gathered sequence length ({padded_total} = "
            f"S_local={S_local} * tp_size={topology.tp_size}). This can "
            "only happen if x wasn't produced by scatter_to_sequence_parallel "
            "with this same topology, or the wrong orig_seq_len was passed."
        )
    return gathered[:, :orig_seq_len, :]
