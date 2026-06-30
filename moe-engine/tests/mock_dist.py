"""
tests/mock_dist.py
==================

Mocked distributed backend for CPU-based testing of collective operations.

Allows distributed logic (all_to_all, all_reduce, all_gather) to be tested
on a single CPU process without initialising ``torch.distributed``. This
enables testing of:

- all_to_all_dispatch / all_to_all_combine with controllable send counts
- Collective latency measurement paths (returns synthetic values)
- Comm/compute overlap ratio calculation
- EP rank ownership logic and expert assignment

Usage in tests
--------------
::

    from tests.mock_dist import mock_ep_size_1, MockTopology

    def test_dispatch_single_rank():
        topo = MockTopology(ep_size=1)
        tokens = torch.randn(32, 64)
        send_counts = torch.tensor([32])
        received, recv_counts, event, ms = all_to_all_dispatch(
            tokens, send_counts, topo
        )
        assert received.shape == tokens.shape   # identity at ep_size=1

Design
------
At ``ep_size == 1`` all collective operations are already no-ops in the
production code (returning the input unchanged). MockTopology provides a
``ParallelTopology``-compatible object that reports ``ep_size=1`` and
``dist.is_initialized() == False`` so the no-op paths execute without any
mock patching required.

For testing at ``ep_size > 1`` on CPU, ``MockDistEnv`` provides a
``threading``-based simulation of all_to_all_single using shared memory
queues, compatible with single-process test execution.
"""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import torch

__all__ = [
    "MockTopology",
    "MockDistEnv",
    "make_mock_topology",
]


# ===========================================================================
# MockTopology — drop-in ParallelTopology for ep_size=1 tests
# ===========================================================================


@dataclass
class MockTopology:
    """Minimal ParallelTopology-compatible object for unit tests.

    At ``ep_size=1`` all collective operations in ``expert_parallel.py`` and
    ``tensor_parallel.py`` are already identity / no-ops (guarded by
    ``if topology.ep_size == 1 or not dist.is_initialized(): return ...``).
    MockTopology sets ep_size=1 and provides all required attributes.

    Parameters
    ----------
    ep_size : int    Expert parallel size (default 1 → all collectives are no-ops).
    tp_size : int    Tensor parallel size (default 1 → all TP ops are no-ops).
    pp_size : int    Pipeline parallel size (default 1).
    dp_size : int    Data parallel size (default 1).
    """

    ep_size: int = 1
    tp_size: int = 1
    pp_size: int = 1
    dp_size: int = 1
    rank: int = 0
    world_size: int = 1
    mesh: None = field(default=None, repr=False)
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))

    @property
    def dp_rank(self) -> int:
        return 0

    @property
    def tp_rank(self) -> int:
        return 0

    @property
    def pp_rank(self) -> int:
        return 0

    @property
    def ep_rank(self) -> int:
        return 0

    def is_first_pp_stage(self) -> bool:
        return True

    def is_last_pp_stage(self) -> bool:
        return True

    def experts_on_this_rank(self, total_experts: int) -> List[int]:
        return list(range(total_experts))

    def validate_world_size(self) -> None:
        pass


def make_mock_topology(
    ep_size: int = 1,
    tp_size: int = 1,
    pp_size: int = 1,
    dp_size: int = 1,
) -> MockTopology:
    """Factory for MockTopology — convenience wrapper for test parametrisation."""
    return MockTopology(
        ep_size=ep_size,
        tp_size=tp_size,
        pp_size=pp_size,
        dp_size=dp_size,
        world_size=ep_size * tp_size * pp_size * dp_size,
    )


# ===========================================================================
# MockDistEnv — threading-based all_to_all simulation for ep_size > 1 tests
# ===========================================================================


class MockDistEnv:
    """Threading-based simulation of all_to_all_single for single-process tests.

    Allows testing ``ep_size > 1`` dispatch/combine logic without spawning
    multiple processes. Uses in-process shared queues per (src, dst) pair to
    simulate the all-to-all communication pattern.

    Usage
    -----
    ::

        env = MockDistEnv(world_size=4)
        # In each "rank" thread:
        received = env.all_to_all(rank, output_buf, input_tensor,
                                   recv_sizes, send_sizes)
        env.barrier()

    Thread-safety: all methods are protected by a condition variable.
    The barrier() call blocks until all world_size ranks have called it.

    Example test
    ------------
    ::

        from tests.mock_dist import MockDistEnv
        import torch

        def worker(rank, env, results):
            N_per_rank = 8
            H = 16
            # Each rank sends N_per_rank tokens to every other rank
            tokens = torch.full((N_per_rank * env.world_size, H), float(rank))
            send_sizes = [N_per_rank] * env.world_size
            recv_sizes = [N_per_rank] * env.world_size
            received = env.all_to_all(rank, tokens, send_sizes, recv_sizes)
            results[rank] = received
            env.barrier(rank)

        env = MockDistEnv(world_size=4)
        results = [None] * 4
        threads = [threading.Thread(target=worker, args=(r, env, results))
                   for r in range(4)]
        for t in threads: t.start()
        for t in threads: t.join()

        # Rank 0 should receive tokens from ranks 0,1,2,3 in that order
        for r in range(4):
            chunk = results[0][r * N_per_rank:(r+1) * N_per_rank]
            assert (chunk == r).all()
    """

    def __init__(self, world_size: int):
        self.world_size = world_size
        self._lock = threading.Condition()
        self._barrier_count = 0
        # queues[(src, dst)] → Queue holding tensors sent from src to dst
        self._queues: Dict[Tuple[int, int], queue.Queue] = {
            (s, d): queue.Queue() for s in range(world_size) for d in range(world_size)
        }

    def all_to_all(
        self,
        rank: int,
        input_tensor: torch.Tensor,
        send_sizes: List[int],
        recv_sizes: List[int],
        timeout: float = 10.0,
    ) -> torch.Tensor:
        """Simulate all_to_all_single for ``rank``.

        Splits ``input_tensor`` into ``len(send_sizes)`` chunks and enqueues
        each chunk to its destination queue. Then dequeues from all source
        queues to produce the received tensor.

        Parameters
        ----------
        rank : int              This rank's index.
        input_tensor : Tensor   Full send buffer; split by ``send_sizes``.
        send_sizes : list[int]  Tokens to send to each rank (sum = input rows).
        recv_sizes : list[int]  Tokens to receive from each rank.
        timeout : float         Seconds to wait for each peer tensor.

        Returns
        -------
        Tensor  Concatenation of received chunks, ordered by sender rank.
        """
        assert len(send_sizes) == self.world_size
        assert len(recv_sizes) == self.world_size

        # Send phase: split and enqueue
        offset = 0
        for dst in range(self.world_size):
            n = send_sizes[dst]
            chunk = input_tensor[offset : offset + n].clone()
            self._queues[(rank, dst)].put(chunk)
            offset += n

        # Receive phase: dequeue from each source
        received_chunks = []
        for src in range(self.world_size):
            chunk = self._queues[(src, rank)].get(timeout=timeout)
            expected = recv_sizes[src]
            assert chunk.shape[0] == expected, (
                f"Rank {rank}: expected {expected} tokens from rank {src}, got {chunk.shape[0]}"
            )
            received_chunks.append(chunk)

        return torch.cat(received_chunks, dim=0)

    def barrier(self, rank: int) -> None:
        """Block until all ranks have called barrier()."""
        with self._lock:
            self._barrier_count += 1
            if self._barrier_count == self.world_size:
                self._barrier_count = 0
                self._lock.notify_all()
            else:
                self._lock.wait_for(lambda: self._barrier_count == 0, timeout=30.0)
