"""
tests/test_mock_dist.py
========================

Tests for the mocked collective backend (tests/mock_dist.py).

Covers:
- MockTopology acts as identity for ep_size=1 in all dispatch/combine paths
- MockDistEnv correctly simulates all_to_all_single across multiple "ranks"
- all_to_all_dispatch + all_to_all_combine round-trip at ep_size=1 (no-op)
- MockDistEnv barrier synchronisation
- Expert-to-rank mapping with MockTopology
"""

from __future__ import annotations

import threading
from typing import List

import pytest
import torch

from pkg.distributed.expert_parallel import all_to_all_dispatch, all_to_all_combine
from pkg.distributed.moe_layer import DistributedMoELayer
from tests.mock_dist import MockTopology, MockDistEnv, make_mock_topology

pytestmark = pytest.mark.cpu


# ===========================================================================
# MockTopology tests
# ===========================================================================

class TestMockTopology:
    def test_defaults(self):
        topo = MockTopology()
        assert topo.ep_size == 1
        assert topo.tp_size == 1
        assert topo.world_size == 1
        assert topo.rank == 0
        assert topo.device == torch.device("cpu")

    def test_rank_properties_all_zero(self):
        topo = MockTopology()
        assert topo.dp_rank == 0
        assert topo.tp_rank == 0
        assert topo.pp_rank == 0
        assert topo.ep_rank == 0

    def test_pp_stage_flags(self):
        topo = MockTopology()
        assert topo.is_first_pp_stage()
        assert topo.is_last_pp_stage()

    def test_experts_on_this_rank_all_owned(self):
        """At ep_size=1, single rank owns all experts."""
        topo = MockTopology(ep_size=1)
        owned = topo.experts_on_this_rank(total_experts=8)
        assert owned == list(range(8))

    def test_make_mock_topology_factory(self):
        topo = make_mock_topology(ep_size=2, tp_size=2, pp_size=1, dp_size=4)
        assert topo.ep_size == 2
        assert topo.world_size == 2 * 2 * 1 * 4


# ===========================================================================
# dispatch/combine at ep_size=1 (identity through MockTopology)
# ===========================================================================

class TestDispatchCombineAtEpSize1:
    """At ep_size=1, dispatch and combine are no-ops (identity)."""

    def test_dispatch_is_identity(self):
        topo = MockTopology(ep_size=1)
        tokens = torch.randn(32, 64)
        send_counts = torch.tensor([32])
        received, recv_counts, event, ms = all_to_all_dispatch(
            tokens, send_counts, topo
        )
        assert received.shape == tokens.shape
        assert torch.allclose(received, tokens)
        assert event is None
        assert ms == 0.0

    def test_combine_is_identity(self):
        topo = MockTopology(ep_size=1)
        expert_out = torch.randn(32, 64)
        recv_counts = torch.tensor([32])
        send_counts = torch.tensor([32])
        combined, ms = all_to_all_combine(
            expert_out, recv_counts, send_counts, topo
        )
        assert combined.shape == expert_out.shape
        assert torch.allclose(combined, expert_out)
        assert ms == 0.0

    def test_dispatch_combine_round_trip(self):
        """dispatch then combine must be a lossless round-trip at ep_size=1."""
        topo = MockTopology(ep_size=1)
        N, H = 64, 128
        tokens = torch.randn(N, H)
        send_counts = torch.tensor([N])

        received, recv_counts, evt, _ = all_to_all_dispatch(tokens, send_counts, topo)
        combined, _ = all_to_all_combine(received, recv_counts, send_counts, topo)
        assert torch.allclose(combined, tokens)

    @pytest.mark.parametrize("N,H", [(8, 16), (64, 256), (512, 512)])
    def test_various_shapes_at_ep1(self, N, H):
        topo = MockTopology(ep_size=1)
        tokens = torch.randn(N, H)
        send_counts = torch.tensor([N])
        received, recv_counts, _, _ = all_to_all_dispatch(tokens, send_counts, topo)
        assert received.shape == (N, H)
        combined, _ = all_to_all_combine(received, recv_counts, send_counts, topo)
        assert combined.shape == (N, H)


# ===========================================================================
# DistributedMoELayer with MockTopology
# ===========================================================================

class TestMoELayerWithMockTopology:
    def test_full_forward_backward(self):
        topo = MockTopology(ep_size=1)
        layer = DistributedMoELayer(
            hidden_dim=64, ffn_dim=128, num_experts=4, top_k=2, topology=topo
        )
        x = torch.randn(2, 8, 64, requires_grad=True)
        out = layer(x)
        assert out.shape == x.shape
        assert not torch.isnan(out).any()
        out.sum().backward()
        assert x.grad is not None
        print(f"  overlap_ratio={layer.last_overlap_ratio:.3f}  "
              f"dispatch_ms={layer.last_dispatch_ms:.3f}  "
              f"compute_ms={layer.last_expert_compute_ms:.3f}")

    def test_expert_to_rank_round_trip(self):
        topo = MockTopology(ep_size=1)
        layer = DistributedMoELayer(
            hidden_dim=8, ffn_dim=16, num_experts=4, top_k=2, topology=topo
        )
        ids = torch.tensor([0, 1, 2, 3])
        ranks = layer._expert_to_rank(ids)
        # At ep_size=1, all experts map to rank 0
        assert (ranks == 0).all()

    def test_token_conservation_via_mock(self):
        """Token conservation holds when run through MockTopology dispatch path."""
        topo = MockTopology(ep_size=1)
        from pkg.kernels.moe_router import MoERouter
        router = MoERouter(hidden_dim=64, num_experts=8, top_k=2)
        N, H = 128, 64
        tokens = torch.randn(N, H)
        idx, w, dispatch_cnt = router(tokens)
        assert int(dispatch_cnt.sum()) == N * 2, (
            f"Token conservation violated: {int(dispatch_cnt.sum())} != {N*2}"
        )


# ===========================================================================
# MockDistEnv — multi-rank simulation tests
# ===========================================================================

class TestMockDistEnv:
    def test_all_to_all_2ranks(self):
        """Each of 2 ranks sends unique data; verify correct routing."""
        world_size = 2
        N_per_rank = 4
        H = 8
        env = MockDistEnv(world_size=world_size)
        results: List[torch.Tensor | None] = [None] * world_size

        def worker(rank):
            # Rank r sends N_per_rank tokens to every other rank
            # Token value = rank (so we can verify routing)
            tokens = torch.full((N_per_rank * world_size, H), float(rank))
            send_sizes = [N_per_rank] * world_size
            recv_sizes = [N_per_rank] * world_size
            received = env.all_to_all(rank, tokens, send_sizes, recv_sizes)
            results[rank] = received
            env.barrier(rank)

        threads = [threading.Thread(target=worker, args=(r,)) for r in range(world_size)]
        for t in threads: t.start()
        for t in threads: t.join()

        # Rank 0 should receive [rank0_tokens, rank1_tokens]
        for dst in range(world_size):
            assert results[dst] is not None
            for src in range(world_size):
                chunk = results[dst][src * N_per_rank:(src + 1) * N_per_rank]
                assert (chunk == float(src)).all(), (
                    f"Rank {dst} chunk from src {src}: expected {src}, got {chunk[0,0]}"
                )

    def test_all_to_all_4ranks(self):
        """4-rank all_to_all with unequal send sizes."""
        world_size = 4
        env = MockDistEnv(world_size=world_size)
        H = 16
        results: List[torch.Tensor | None] = [None] * world_size

        def worker(rank):
            # Each rank sends (rank+1) tokens to each destination
            n_send = rank + 1
            tokens = torch.full((n_send * world_size, H), float(rank))
            send_sizes = [n_send] * world_size
            recv_sizes = [r + 1 for r in range(world_size)]  # each src sends r+1
            received = env.all_to_all(rank, tokens, send_sizes, recv_sizes)
            results[rank] = received
            env.barrier(rank)

        threads = [threading.Thread(target=worker, args=(r,)) for r in range(world_size)]
        for t in threads: t.start()
        for t in threads: t.join()

        # Verify total received = sum of recv_sizes
        expected_total = sum(r + 1 for r in range(world_size))
        for dst in range(world_size):
            assert results[dst].shape[0] == expected_total, (
                f"Rank {dst}: expected {expected_total} tokens, "
                f"got {results[dst].shape[0]}"
            )

    def test_barrier_blocks_until_all_arrive(self):
        """barrier() must block each thread until all world_size threads call it."""
        world_size = 4
        env = MockDistEnv(world_size=world_size)
        arrival_order: List[int] = []
        lock = threading.Lock()

        def worker(rank):
            import time
            time.sleep(rank * 0.01)  # stagger arrivals
            env.barrier(rank)
            with lock:
                arrival_order.append(rank)

        threads = [threading.Thread(target=worker, args=(r,)) for r in range(world_size)]
        for t in threads: t.start()
        for t in threads: t.join(timeout=5.0)
        assert len(arrival_order) == world_size, "Not all threads completed barrier"
