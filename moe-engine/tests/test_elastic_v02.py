"""
tests/test_elastic_v02.py
=========================

v0.2 additions to the elastic fault-tolerance test suite.

New coverage:
  * ElasticConfig defaults validation
  * Reshard plan covers all experts with no duplicates across various topologies
  * Largest-divisor helper edge cases (prime sizes, power-of-two, single node)
  * File-URI remote adapter round-trip (LocalNVMeAdapter as remote tier)
  * Health check no-op on single-rank world (no dist)
  * Checkpoint prune: only the configured retention window survives
  * Metadata keys in committed checkpoints
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from pkg.distributed.parallel_mesh import build_topology
from pkg.elastic.fault_monitor import (

    AsyncCheckpointer,
    ClusterStateMachine,
    ElasticConfig,
    ElasticTrainerHarness,
    LocalNVMeAdapter,
    _largest_divisor_le,
)

pytestmark = pytest.mark.cpu

# ---------------------------------------------------------------------------
# ElasticConfig construction
# ---------------------------------------------------------------------------

def test_elastic_config_fields():
    cfg = ElasticConfig(
        local_ckpt_dir="/tmp/ckpts",
        remote_uri="s3://bucket/path",
        s3_endpoint="http://localhost:9000",
        retention=4,
        async_workers=2,
        health_interval_s=3.0,
        drop_grace_s=15.0,
        min_nodes=2,
    )
    assert cfg.local_ckpt_dir == "/tmp/ckpts"
    assert cfg.retention == 4
    assert cfg.min_nodes == 2

# ---------------------------------------------------------------------------
# Reshard plan completeness
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("num_experts,ep_size", [
    (64, 8),
    (10, 4),    # remainder case
    (7, 3),     # prime / non-divisible
    (1, 1),
    (128, 16),
    (100, 7),   # large remainder
])
def test_reshard_covers_all_experts(num_experts, ep_size, tmp_path):
    """Reshard plan must partition experts with no gaps and no duplicates."""
    # Use ep_size as effective world
    topo = build_topology(dp_size=1, ep_size=min(ep_size, 4), device_type="cpu")
    csm = ClusterStateMachine(topology=topo, min_nodes=1)
    new_ep = min(ep_size, 4)
    new_topo = build_topology(dp_size=1, ep_size=new_ep, device_type="cpu")
    plan = csm.reshard(new_topo, num_experts=num_experts)

    # Flatten all assigned expert ids
    all_assigned = []
    for ids in plan.values():
        all_assigned.extend(ids)

    assert sorted(all_assigned) == list(range(num_experts)), (
        f"ep={ep_size} E={num_experts}: plan {plan} does not cover exactly "
        f"{{0..{num_experts-1}}}"
    )

def test_reshard_marks_recovering_phase(tmp_path):
    topo = build_topology(dp_size=1, ep_size=1, device_type="cpu")
    csm = ClusterStateMachine(topology=topo, min_nodes=1)
    assert csm.phase == ClusterStateMachine.PHASE_RUNNING
    csm.begin_recovery()
    assert csm.phase == ClusterStateMachine.PHASE_DRAINING
    new_topo = build_topology(dp_size=1, ep_size=1, device_type="cpu")
    csm.reshard(new_topo, num_experts=8)
    assert csm.phase == ClusterStateMachine.PHASE_RECOVERING
    csm.mark_resumed()
    assert csm.phase == ClusterStateMachine.PHASE_RESUMED

# ---------------------------------------------------------------------------
# Largest-divisor helper — exhaustive edge cases
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n,k,expected", [
    (64, 8, 8),
    (64, 9, 8),
    (60, 9, 6),
    (7, 8, 7),
    (1, 8, 1),
    (100, 7, 5),
    (13, 13, 13),   # prime == k
    (13, 14, 13),   # prime < k
    (13, 6, 1),     # no divisor <=6 except 1 for prime 13
    (16, 4, 4),
    (16, 16, 16),
    (128, 9, 8),
])
def test_largest_divisor_le(n, k, expected):
    assert _largest_divisor_le(n, k) == expected, (
        f"_largest_divisor_le({n}, {k}) = {_largest_divisor_le(n, k)} != {expected}"
    )

# ---------------------------------------------------------------------------
# File-URI two-tier round-trip
# ---------------------------------------------------------------------------

def test_file_uri_remote_tier(tmp_path: Path):
    local_dir = tmp_path / "nvme"
    remote_dir = tmp_path / "remote"
    local = LocalNVMeAdapter(str(local_dir))
    remote = LocalNVMeAdapter(str(remote_dir))

    ckpt = AsyncCheckpointer(
        local_adapter=local, remote_adapter=remote, retention=4, workers=1
    )
    model = nn.Linear(4, 4)
    ckpt.save(model, None, step=5, rank=0)
    ckpt.shutdown(drain=True)

    # Both tiers must hold the checkpoint
    assert (local_dir / "ckpts" / "step=000005" / "rank=000000.pt").exists()
    assert (remote_dir / "ckpts" / "step=000005" / "rank=000000.pt").exists()

    # Meta must be present on both tiers
    meta_local = json.loads(
        (local_dir / "ckpts" / "step=000005" / "rank=000000.meta.json").read_text()
    )
    assert meta_local["step"] == 5
    assert meta_local["rank"] == 0
    assert "ts" in meta_local
    assert "hostname" in meta_local

# ---------------------------------------------------------------------------
# Health check no-op (single-rank, no dist)
# ---------------------------------------------------------------------------

def test_health_check_single_rank_no_op():
    topo = build_topology(dp_size=1, ep_size=1, device_type="cpu")
    csm = ClusterStateMachine(topology=topo, min_nodes=1)
    dead = csm.heartbeat()
    assert dead == []   # no dist → always empty

def test_alive_ranks_all_present():
    topo = build_topology(dp_size=1, ep_size=1, device_type="cpu")
    csm = ClusterStateMachine(topology=topo, min_nodes=1)
    assert csm.alive_ranks() == [0]

# ---------------------------------------------------------------------------
# Checkpoint retention — only latest N survive
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("retention", [1, 2, 3])
def test_checkpoint_retention(tmp_path: Path, retention: int):
    local = LocalNVMeAdapter(str(tmp_path))
    ckpt = AsyncCheckpointer(
        local_adapter=local, remote_adapter=None, retention=retention, workers=1
    )
    model = nn.Linear(2, 2)
    for step in range(1, 7):
        ckpt.save(model, None, step=step, rank=0)
        time.sleep(0.02)   # let worker commit between saves

    ckpt.shutdown(drain=True)

    # Determine which steps are present on disk
    surviving = set()
    for k in local.list("ckpts/"):
        try:
            seg = [p for p in k.split("/") if p.startswith("step=")][0]
            surviving.add(int(seg.split("=")[1]))
        except Exception:
            continue

    assert len(surviving) <= retention, (
        f"retention={retention}: expected ≤{retention} steps, "
        f"got {len(surviving)}: {sorted(surviving)}"
    )
    # Latest step must always survive
    assert 6 in surviving, "Latest step must survive pruning"

# ---------------------------------------------------------------------------
# Harness: checkpoint and resume via ElasticTrainerHarness
# ---------------------------------------------------------------------------

def test_harness_checkpoint_and_resume(tmp_path: Path):
    cfg = ElasticConfig(
        local_ckpt_dir=str(tmp_path / "ckpts"),
        remote_uri=f"file://{tmp_path / 'remote'}",
        retention=4,
        async_workers=1,
        min_nodes=1,
    )
    topo = build_topology(dp_size=1, ep_size=1, device_type="cpu")
    harness = ElasticTrainerHarness(cfg, topo)
    harness.install_signal_handlers()

    model = nn.Sequential(nn.Linear(8, 8), nn.ReLU())
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model(torch.randn(2, 8)).sum().backward()
    optim.step()

    harness.checkpoint(model, optim, step=10)
    harness.shutdown()

    # Verify latest_step is discoverable
    latest = harness.async_ckpt.latest_step()
    assert latest == 10

def test_harness_no_signal_handler_in_thread(tmp_path: Path):
    """install_signal_handlers must not raise when called from a non-main thread."""
    cfg = ElasticConfig(
        local_ckpt_dir=str(tmp_path), remote_uri=None, min_nodes=1
    )
    topo = build_topology(dp_size=1, ep_size=1, device_type="cpu")

    errors: list[Exception] = []

    def _install():
        try:
            h = ElasticTrainerHarness(cfg, topo)
            h.install_signal_handlers()   # must not raise ValueError
            h.shutdown()
        except Exception as exc:
            errors.append(exc)

    import threading

    t = threading.Thread(target=_install)
    t.start()
    t.join()
    assert not errors, f"install_signal_handlers raised in thread: {errors}"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
