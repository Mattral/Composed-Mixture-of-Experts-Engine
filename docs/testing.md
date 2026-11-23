# Testing Guide

**Version:** v0.3.3  
**Last updated:** July 2026

---

## Philosophy

Tests are written for reality: most contributors do not have constant access
to a multi-GPU cluster. The Tier-0 job finishes in about 10 seconds on CPU.
The differential job also runs on every push and takes several minutes.
GPU and chaos jobs are manual.

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
# CI Tier-0 job. `make test-cpu` is wider: it does not ignore the four
# differential files below.
pytest tests/ \
  --ignore=tests/test_chaos.py \
  --ignore=tests/test_smoke_e2e.py \
  --ignore=tests/test_differential.py \
  --ignore=tests/test_comm_ops.py \
  --ignore=tests/test_grad_sync.py \
  --ignore=tests/test_train_step.py \
  -m cpu \
  -k "not (2rank or multiprocess or distributed_invariants)" \
  -q

# 417 passed, 3 skipped, 17 deselected, 1 xfailed
# The 3 skips are the Triton GPU path and two CUDA-only tests in
# test_gpu_portability.py. The xfail is seed=2 in test_routing_quality.py.
# The 38 config tests are included in the 417 passed.

# Differential job (separate; 243 passed). Also runs on every push.
GLOO_SOCKET_IFNAME=lo pytest tests/test_differential.py tests/test_comm_ops.py \
  tests/test_grad_sync.py tests/test_train_step.py -q
```

### Tier 1: GPU (Triton kernel, real CUDA)

```bash
make test-gpu

# Equivalent
pytest tests/test_kernels.py tests/test_kernels_numerics.py -m gpu -v

# Key tests:
# test_kernels.py::test_triton_kernels_declare_k_as_constexpr
# test_kernels_numerics.py (31 cases: forward, backward, determinism, edge cases)
```

### Tier 2: Multi-process (2-rank mp.spawn)

The Tier-0 job filters `2rank or multiprocess or distributed_invariants`
out of its invocation (17 deselected). The differential job does not use
that filter. It runs `test_differential.py`, `test_comm_ops.py`,
`test_grad_sync.py`, and `test_train_step.py` (243 passed) and does not
include the named tests in this section. Run those named tests manually
with loopback networking (`GLOO_SOCKET_IFNAME=lo`):

```bash
GLOO_SOCKET_IFNAME=lo pytest tests/ \
  -k "2rank or multiprocess or distributed_invariants" -v
```

Status on 2026-11-23 (WSL Ubuntu, CPU torch 2.14.1): 17 passed.
These cover TP (`test_tensor_parallel.py`), SP (`test_sequence_parallel_v03.py`),
the EP layer (`test_moe_layer.py`), `test_distributed_invariants.py`, and four
pipeline-parallel cases (`test_pipeline_parallel.py::test_pp_multiprocess_*`),
two of which compare 2- and 3-stage outputs and gradients with a dense
reference. The pipeline cases failed until the F16 fix on 2026-11-23; see
`roadmap.md`, "Multi-process tests outside push CI". This status is from Linux.
ADR-010 reported that the old pipeline tests hung on Windows; they have not
been run on Windows since the fix.

### Tier 3: Chaos (fault injection)

```bash
# Scenario B: storage stall (10s I/O delay). Recorded 10/10 in RESULTS.md.
# The Chaos CI job is workflow_dispatch only and did not run on the latest push.
make chaos-b

# Scenario A: node kill + recovery — recorded ~17/20 (failures attributed to a Gloo race in containers)
make chaos-a
```

---

## Test File Reference

| File | Tier | Tests | What it covers |
|------|------|------:|----------------|
| `test_config.py` | 0 | 38 | Full `MoEConfig` Pydantic system, incl. `large_scale.yaml`. Included in the Tier-0 passed count. |
| `test_router.py` | 0 | 39 | `MoERouterInterface` |
| `test_router_losses.py` | 0 | 7 | Aux router losses |
| `test_registry.py` | 0 | 20 | Model registry and factory |
| `test_capacity_dropping.py` | 0 | 35 | Expert capacity dropping |
| `test_mock_dist.py` | 0 | 17 | Mocked collective backend |
| `test_properties.py` | 0 | 9 | Hypothesis property tests |
| `test_kernels.py` | 0/1 | 11 | Router invariants (conservation, NaN, bounds, constexpr) |
| `test_kernels_numerics.py` | 0/1 | 31 | `atol=rtol=1e-5` vs fp64 ref; 30 `(H,E,K)` configs |
| `test_routing_quality.py` | 0 | 16 | Load imbalance, z-loss; stochastic seed sweep |
| `test_tensor_parallel.py` | 0/2 | 25 | Column/Row parallel at tp=1 incl. gather_output modes (ADR-008); 2-rank checks |
| `test_pipeline_parallel.py` | 0/2 | 29 | 1F1B scheduling (25, Tier-0); 2- and 3-stage `mp.spawn` activation flow and gradients vs dense (4, deselected; all pass) |
| `test_sequence_parallel_v03.py` | 0/2 | 26 | SP scatter/gather incl. non-divisible-length padding; fused `next_weight` path (ADR-007) |
| `test_distributed.py` | 0 | 5 | `DistributedMoELayer` expert-to-rank mapping |
| `test_distributed_invariants.py` | 2 | 2 | Token conservation + backward NaN (4-process Gloo) — ✅ ADR-009 |
| `test_moe_layer.py` | 0/2 | 18 | DistributedMoELayer per-token correctness (ADR-009) |
| `test_elastic.py` | 0 | 7 | Async checkpointing core (NVMe + S3 mock) |
| `test_elastic_v02.py` | 0 | 30 | Expert resharding after node drop |
| `test_mfu.py` | 0 | 6 | MFU formula correctness |
| `test_mfu_v02.py` | 0 | 17 | `MFUAccountant` streaming tracker |
| `test_telemetry.py` | 0 | 28 | `StructuredLogger` JSONL output; overlap ratio field |
| `test_smoke_e2e.py` | 0 | 3 | End-to-end smoke: config → topology → model → step. Ignored by the Tier-0 job. |
| `test_gpu_portability.py` | 0/1 | 22 | ADR-005's three A100 fixes + ADR-006's device-coercion correction; CPU logic/source tests, 2 hardware-gated `gpu`-tier tests |
| `test_differential.py` | differential | 215 | Multi-process EP/TP/SP vs a dense reference |
| `test_comm_ops.py` | differential | 9 | Collective forward and VJP checks |
| `test_grad_sync.py` | differential | 15 | Gradient sync checks |
| `test_train_step.py` | differential | 4 | Train-step checks vs a dense reference |
| `test_chaos.py` | 3 | 3 | Scenario A (node kill) + Scenario B (storage stall). Manual CI job. |

Counts are `pytest --collect-only` items, including parametrized cases. Summing this table is not a passing total: Tier-0 deselects 16 items, skips 3, and xfails 1, and it does not run the differential, smoke, or chaos files.

**Tier-0 job: 417 passed, 3 skipped, 17 deselected, 1 xfailed.** The 38 `test_config.py` items are inside that passed count. **Differential job: 243 passed.** Full collection is 687 items in 28 files.

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
pytest tests/ -m cpu          # marker only; not the Tier-0 job command
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

# Router correctness (11 collected items)
pytest tests/test_kernels.py -v

# Full numerics sweep (31 collected items)
pytest tests/test_kernels_numerics.py -v

# Pipeline scheduling (27 collected items, including 2-rank)
pytest tests/test_pipeline_parallel.py -v

# Elastic/checkpointing (37 collected items)
pytest tests/test_elastic.py tests/test_elastic_v02.py -v

# Telemetry (28 collected items)
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
documented inline; it is the 1 xfailed in the Tier-0 summary, not one of the 417 passed, when counting
stable tests. Tracked as a known issue.

### Multi-process tests (`2rank`, `multiprocess`, `distributed_invariants`)

These require `localhost` IPv6 or IPv4 loopback networking. In some container
environments (Docker with restricted networking), they fail with:
`errno: 97 - Address family not supported`.

Fix: `export GLOO_SOCKET_IFNAME=lo` before running.

### Chaos Scenario A

The recorded result is ~17 of 20 runs resuming. Failures are attributed to a
Gloo `connectFullMesh` socket race when restarting a rank after SIGKILL in
containers. This job is `workflow_dispatch` only and does not run on push CI.
A switch of the chaos harness to NCCL is planned for v0.4.

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
