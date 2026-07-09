# Testing Guide

**Version:** v0.3.3  
**Last updated:** July 2026

---

## Philosophy

Tests are written for reality: most contributors do not have constant access
to a multi-GPU cluster. 90% of the test suite runs on a single CPU core in
under 60 seconds. GPU and multi-process tests exist and pass, but they are
opt-in and clearly separated from the fast Tier-0 suite.

The four-tier model matches the four-tier benchmark strategy in `docs/benchmarks.md`.

---

## Four-Tier Test Model

| Tier | Hardware | Command | Time | When to run |
|------|----------|---------|------|-------------|
| **0 — CPU** | Any machine | `make test-cpu` | ~60s | Every commit |
| **1 — GPU** | T4 / RTX 4090 | `make test-gpu` | ~5 min | Daily on GPU machines |
| **2 — Multi-process** | 2–8 processes (1 node) | `torchrun` or Docker Compose | ~15 min | Weekly |
| **3 — Cluster chaos** | 4+ nodes | `make chaos-a` / `make chaos-b` | ~5 min | When cluster is available |

### Tier 0: CPU-only (must pass on every commit)

```bash
# All 348 CPU tests
make test-cpu

# Equivalent manual command
pytest tests/ \
  --ignore=tests/test_chaos.py \
  --ignore=tests/test_smoke_e2e.py \
  -m cpu \
  -k "not (2rank or multiprocess or distributed_invariants)" \
  -q

# Expected: 348 passed, 1 skipped (Triton GPU), 1 xfailed (seed=2 routing, documented statistical edge case)
```

### Tier 1: GPU (Triton kernel, real CUDA)

```bash
make test-gpu

# Equivalent
pytest tests/test_kernels.py tests/test_kernels_numerics.py -m gpu -v

# Key tests:
# test_kernels.py::test_triton_kernels_declare_k_as_constexpr
# test_kernels_numerics.py::test_router_fwd_bwd_numerics (30 configs)
```

### Tier 2: Multi-process (2-rank mp.spawn)

These run automatically in the full suite but are slow and require loopback
networking (`GLOO_SOCKET_IFNAME=lo`):

```bash
GLOO_SOCKET_IFNAME=lo pytest tests/ \
  -k "2rank or multiprocess or distributed_invariants" -v
```

Tests covered:
- `test_tensor_parallel.py::test_column_row_parallel_2rank_numerically_correct`
- `test_pipeline_parallel.py::test_pp_multiprocess_2stage_activation_flow`
- `test_sequence_parallel_v03.py::test_sp_fused_2rank_numerically_correct`
- `test_distributed_invariants.py::test_token_conservation_distributed`

### Tier 3: Chaos (fault injection)

```bash
# Scenario B: storage stall (10s I/O delay) — expect 10/10 = 100%
make chaos-b

# Scenario A: node kill + recovery — expect ~85% (Gloo race, known)
make chaos-a
```

---

## Test File Reference

| File | Tier | Tests | What it covers |
|------|------|------:|----------------|
| `test_config.py` | 0 | 38 | Full `MoEConfig` Pydantic system, incl. `large_scale.yaml` |
| `test_kernels.py` | 0/1 | 7 | Router invariants (conservation, NaN, bounds, constexpr) |
| `test_kernels_numerics.py` | 0/1 | 13 | `atol=rtol=1e-5` vs fp64 ref; 30 `(H,E,K)` configs |
| `test_routing_quality.py` | 0 | 12 | Load imbalance, z-loss; stochastic seed sweep |
| `test_tensor_parallel.py` | 0/2 | 19 | Column/Row parallel at tp=1; 2-rank numerics |
| `test_pipeline_parallel.py` | 0/2 | 16 | 1F1B scheduling; 2-rank `mp.spawn` activation flow |
| `test_sequence_parallel_v03.py` | 0/2 | 8 | SP scatter/gather; fused next_weight 2-rank check |
| `test_distributed.py` | 0 | 4 | `DistributedMoELayer` expert-to-rank mapping |
| `test_distributed_invariants.py` | 2 | 2 | Token conservation + backward NaN (4-process Gloo) |
| `test_elastic.py` | 0 | 7 | Async checkpointing core (NVMe + S3 mock) |
| `test_elastic_v02.py` | 0 | 10 | Expert resharding after node drop |
| `test_mfu.py` | 0 | 6 | MFU formula correctness |
| `test_mfu_v02.py` | 0 | 15 | `MFUAccountant` streaming tracker |
| `test_telemetry.py` | 0 | 22 | `StructuredLogger` JSONL output; overlap ratio field |
| `test_smoke_e2e.py` | 0 | 3 | End-to-end smoke: config → topology → model → step |
| `test_chaos.py` | 3 | 3 | Scenario A (node kill) + Scenario B (storage stall) |

**Total Tier-0 CPU: 348 passing** (v0.3.3, July 2026)

---

## pytest Markers

All markers are registered in `pyproject.toml` under `[tool.pytest.ini_options]`.
`--strict-markers` is set, so unregistered markers are errors.

```toml
markers = [
    "cpu: tests that run on CPU only — the Tier-0 fast suite (pytest -m cpu)",
    "gpu: tests that require a CUDA-capable GPU (pytest -m gpu)",
    "chaos: fault-injection tests; opt-in (pytest -m chaos)",
]
```

Usage:

```bash
pytest tests/ -m cpu          # 348 tests, ~20s
pytest tests/ -m gpu          # requires CUDA + Triton
pytest tests/ -m chaos        # requires torchrun + Gloo
pytest tests/ -m "cpu or gpu" # both tiers
pytest tests/ -m "not chaos"  # everything except chaos
```

Every test file that runs on CPU has:
```python
pytestmark = pytest.mark.cpu
```
at module level, placed after all imports using `ast.end_lineno` detection to
avoid splitting multi-line `from X import (...)` blocks.

---

## Running Specific Tests

```bash
# Config system only (38 tests, ~0.3s)
pytest tests/test_config.py -v

# Router correctness (7 tests, ~2s on CPU)
pytest tests/test_kernels.py -v

# Full numerics sweep (13 tests, ~15s on CPU)
pytest tests/test_kernels_numerics.py -v

# Pipeline scheduling (16 tests including 2-rank)
pytest tests/test_pipeline_parallel.py -v

# Elastic/checkpointing (17 tests)
pytest tests/test_elastic.py tests/test_elastic_v02.py -v

# Telemetry (22 tests)
pytest tests/test_telemetry.py -v

# Single test by name
pytest tests/test_config.py::TestEnvOverrides::test_hidden_dim_override -v
```

---

## Known Flaky Tests

### `test_routing_quality.py::test_uniform_init_lower_imbalance[2]`

Stochastic test that fails at seed=2. The seed generates an adversarial token
distribution where uniform initialisation produces a locally higher imbalance
than the random baseline. This is a pre-existing issue in the original
codebase and is present in the original repo's CI. It does not reflect a bug
in the refactored code.

Status: **non-blocking**. Marked `xfail(strict=False)` with the mechanism
documented inline; excluded from the 348-passing count when counting
stable tests. Tracked as a known issue.

### Multi-process tests (`2rank`, `multiprocess`, `distributed_invariants`)

These require `localhost` IPv6 or IPv4 loopback networking. In some container
environments (Docker with restricted networking), they fail with:
`errno: 97 - Address family not supported`.

Fix: `export GLOO_SOCKET_IFNAME=lo` before running.

### Chaos Scenario A

Approximately 85% pass rate due to a Gloo `connectFullMesh` socket race when
restarting a rank after SIGKILL. This is environment-specific and not a code
bug. Fix planned for v0.4 (switch chaos harness to NCCL).

---

## conftest.py Fixtures

Defined in `tests/conftest.py`:

| Fixture | Scope | Purpose |
|---------|-------|---------|
| `free_port` | function | Ephemeral TCP port for distributed init |
| `free_port_pair` | function | Two distinct ports (for dual-PG tests) |
| `_destroy_dist_after_test` | function, autouse | Clean up `dist.init_process_group` after every test |
| `work_dir` | function | Isolated `tmp_path/work` directory |

The `_destroy_dist_after_test` autouse fixture prevents distributed state
leaking between tests when a test fails before calling `dist.destroy_process_group`.

---

## Adding New Tests

### Required pattern for CPU tests

```python
import pytest
pytestmark = pytest.mark.cpu          # module-level; must be after all imports

def test_my_feature():
    from pkg.distributed.mesh import build_topology
    topo = build_topology(dp_size=1, ep_size=1)
    # ... assert something
```

### Required pattern for GPU tests

```python
import pytest
import torch

pytestmark = pytest.mark.cpu          # still mark as cpu for the easy path

@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="requires CUDA")
def test_triton_kernel_on_real_gpu():
    from pkg.kernels.moe_router import MoERouter
    router = MoERouter(hidden_dim=256, num_experts=8, top_k=2).cuda()
    # ...
```

### Required pattern for 2-rank tests

```python
import torch.multiprocessing as mp
import pytest

pytestmark = pytest.mark.cpu

def _worker(rank, world_size, port, result_queue):
    import torch.distributed as dist
    dist.init_process_group(
        backend="gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank, world_size=world_size,
    )
    # ... run test logic ...
    result_queue.put(("ok", rank))
    dist.destroy_process_group()

def test_my_2rank_thing(free_port):
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [ctx.Process(target=_worker, args=(r, 2, free_port, q)) for r in range(2)]
    for p in procs: p.start()
    for p in procs: p.join(timeout=30)
    results = [q.get_nowait() for _ in range(2)]
    assert all(r[0] == "ok" for r in results)
```

---

## CI Configuration

The `.github/workflows/ci.yml` runs:

```yaml
- name: Tier-0 CPU tests
  run: |
    cd moe-engine
    pytest tests/ -m cpu \
      -k "not (2rank or multiprocess or distributed_invariants)" \
      --tb=short -q
```

Tier-1 GPU tests run on a separate self-hosted runner with CUDA. Tier-3 chaos
tests are not in CI — they run manually when cluster access is available.

See `.github/workflows/ci.yml` for the full pipeline.
