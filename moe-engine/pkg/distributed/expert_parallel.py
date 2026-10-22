"""
pkg/distributed/expert_parallel.py
====================================

Expert-parallel all-to-all dispatch and combine primitives.

Responsibilities:
  - all_to_all_dispatch  — send each token to the EP rank owning its expert
  - all_to_all_combine   — return processed tokens to their originating rank
  - _CommStream          — singleton CUDA stream per device for EP collectives

The communication is deliberately kept on a dedicated CUDA stream so the
expert FFN compute (default stream) can overlap with the next all-to-all.
The overlap ratio is measured and surfaced as a telemetry metric.

All functions degrade gracefully to no-ops when ep_size == 1 or dist is
not initialized.

Public API
----------
    all_to_all_dispatch(tokens_sorted, send_counts, topology) -> (received, recv_counts, event, latency_ms)
    all_to_all_combine(expert_out, recv_counts, send_counts, topology, wait_event) -> (combined, latency_ms)
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.distributed as dist

from pkg.distributed.comm_ops import all_to_all
from pkg.distributed.mesh import ParallelTopology

__all__ = [
    "all_to_all_dispatch",
    "all_to_all_combine",
]


# ===========================================================================
# Dedicated CUDA stream for EP collectives
# ===========================================================================


class _CommStream:
    """Module-level singleton: one high-priority CUDA stream per device index.

    Using a dedicated stream allows EP all-to-all to overlap with expert
    FFN compute on the default stream.  The overlap ratio is measured by
    comparing the stream's event elapsed time against the expert compute time.
    """

    _streams: dict = {}

    @classmethod
    def get(cls, device: torch.device) -> "Optional[torch.cuda.Stream]":
        if device.type != "cuda" or not torch.cuda.is_available():
            return None
        idx = device.index if device.index is not None else 0
        if idx not in cls._streams:
            cls._streams[idx] = torch.cuda.Stream(device=idx, priority=-1)
        return cls._streams[idx]


# ===========================================================================
# All-to-all helpers
# ===========================================================================


def all_to_all_dispatch(
    tokens_sorted: torch.Tensor,
    send_counts: torch.Tensor,
    topology: ParallelTopology,
) -> Tuple[torch.Tensor, torch.Tensor, "Optional[torch.cuda.Event]", float]:
    """Send sorted tokens to the EP ranks that own their assigned experts.

    Parameters
    ----------
    tokens_sorted : Tensor  ``[N * top_k, H]``
        Tokens sorted by assigned expert, repeated ``top_k`` times per token.
    send_counts : Tensor  ``[ep_size]``  long
        Number of tokens to send to each EP rank.
    topology : ParallelTopology

    Returns
    -------
    received : Tensor  ``[total_recv, H]``
        Tokens received from all EP ranks.
    recv_counts : Tensor  ``[ep_size]``  long
        Number of tokens received from each EP rank (inverse of send_counts).
    event : Optional[cuda.Event]
        CUDA event recorded after the collective; used by combine to ensure
        ordering.  None on CPU.
    latency_ms : float
        Elapsed time of the dispatch collective in milliseconds.

    Raises
    ------
    RuntimeError
        On **every** EP rank at once, if any rank's ``send_counts`` is
        invalid (wrong length, negative, or not summing to that rank's
        ``tokens_sorted.shape[0]``). See the count-exchange comment below.
    """
    if topology.ep_size == 1 or not dist.is_initialized():
        return tokens_sorted, send_counts.clone(), None, 0.0

    ep_group = topology.mesh["ep"].get_group() if topology.mesh is not None else None
    stream = _CommStream.get(topology.device)
    ep_size = topology.ep_size

    # Exchange send counts so every rank knows how many tokens it will
    # receive -- together with a per-rank validity flag, in ONE collective
    # that every rank always enters (F4). If a rank checked its counts
    # locally and raised, its peers would already be blocked in a collective
    # it never joins and hang until the collective timeout (an invalid split
    # used to raise Gloo's "Split sizes doesn't match" on that rank only).
    # Instead each rank sends (count_for_peer, ok) to every peer, so all
    # ranks see every rank's flag and abort together before the data
    # all-to-all, naming the bad rank(s).
    valid = (
        send_counts.dim() == 1
        and send_counts.numel() == ep_size
        and bool((send_counts >= 0).all())
        and int(send_counts.sum()) == tokens_sorted.shape[0]
    )
    counts = (
        send_counts.to(torch.long)
        if valid
        else torch.zeros(ep_size, dtype=torch.long, device=send_counts.device)
    )
    meta_out = torch.stack([counts, torch.full_like(counts, int(valid))], dim=1)  # [ep, 2]
    meta_in = torch.empty_like(meta_out)
    _all_to_all_single_on_stream(meta_in, meta_out, ep_group, stream)
    peer_valid = meta_in[:, 1]
    if not bool(peer_valid.all()):
        bad = (peer_valid == 0).nonzero().flatten().tolist()
        raise RuntimeError(
            f"EP dispatch aborted on all {ep_size} EP ranks: invalid send_counts on "
            f"EP rank(s) {bad}. Each rank's send_counts must have ep_size entries, be "
            "non-negative, and sum to the number of rows it dispatches."
        )
    recv_counts = meta_in[:, 0].contiguous()

    # Split sizes as Python lists, computed once: the autograd Function saves
    # them for its backward (the reverse exchange), so no second sync (F1).
    out_splits = recv_counts.tolist()
    in_splits = send_counts.tolist()

    if stream is not None and torch.cuda.is_available():
        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)
        start_evt.record(stream)
        with torch.cuda.stream(stream):
            received = all_to_all(tokens_sorted, out_splits, in_splits, ep_group)
            end_evt.record(stream)
        end_evt.synchronize()
        latency_ms = max(start_evt.elapsed_time(end_evt), 0.0)
        completion_event = torch.cuda.Event()
        completion_event.record(stream)
        return received, recv_counts, completion_event, latency_ms

    # CPU path. Differentiable (F1): backward returns each gradient row to
    # the rank its token came from.
    received = all_to_all(tokens_sorted, out_splits, in_splits, ep_group)
    return received, recv_counts, None, 0.0


def all_to_all_combine(
    expert_out: torch.Tensor,
    recv_counts: torch.Tensor,
    send_counts: torch.Tensor,
    topology: ParallelTopology,
    wait_event: "Optional[torch.cuda.Event]" = None,
) -> Tuple[torch.Tensor, float]:
    """Return processed tokens to their originating EP ranks.

    The inverse of :func:`all_to_all_dispatch`.  The ``send_counts`` here
    are the dispatch's original send counts; the ``recv_counts`` are what
    was received during dispatch (now used as our send sizes for combine).

    Parameters
    ----------
    expert_out : Tensor  ``[total_recv, H]``
        Expert FFN outputs for all received tokens.
    recv_counts : Tensor  ``[ep_size]``  long
        Tokens received per EP rank during dispatch (= our send sizes here).
    send_counts : Tensor  ``[ep_size]``  long
        Tokens sent per EP rank during dispatch (= our recv sizes here).
    topology : ParallelTopology
    wait_event : Optional[cuda.Event]
        If provided, the combine stream waits for this event before issuing
        the collective (ensures ordering with the preceding dispatch).

    Returns
    -------
    combined : Tensor  ``[total_send, H]``
        Combined tokens ready to be re-scattered to original positions.
    latency_ms : float
        Elapsed time of the combine collective in milliseconds.
    """
    if topology.ep_size == 1 or not dist.is_initialized():
        return expert_out, 0.0

    ep_group = topology.mesh["ep"].get_group() if topology.mesh is not None else None
    stream = _CommStream.get(topology.device)
    out_splits = send_counts.tolist()
    in_splits = recv_counts.tolist()

    if stream is not None and torch.cuda.is_available():
        if wait_event is not None:
            stream.wait_event(wait_event)
        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)
        start_evt.record(stream)
        with torch.cuda.stream(stream):
            combined = all_to_all(expert_out, out_splits, in_splits, ep_group)
            end_evt.record(stream)
        torch.cuda.current_stream().wait_event(end_evt)
        end_evt.synchronize()
        latency_ms = max(start_evt.elapsed_time(end_evt), 0.0)
        return combined, latency_ms

    # CPU path. Differentiable (F1): backward sends each expert-output
    # gradient row back to the rank that computed it.
    combined = all_to_all(expert_out, out_splits, in_splits, ep_group)
    return combined, 0.0


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------


def _all_to_all_single_on_stream(
    output: torch.Tensor,
    input_: torch.Tensor,
    group: "Optional[dist.ProcessGroup]",
    stream: "Optional[torch.cuda.Stream]",
) -> None:
    """Run all_to_all_single on the given CUDA stream (or default stream)."""
    if stream is not None and torch.cuda.is_available():
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            dist.all_to_all_single(output, input_, group=group)
    else:
        dist.all_to_all_single(output, input_, group=group)
