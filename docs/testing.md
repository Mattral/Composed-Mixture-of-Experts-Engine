# Testing Guide

**Version:** v0.3  
**Last updated:** June 2026

---

## Test Suite at a Glance

```
pytest tests/ -v --ignore=tests/test_chaos.py
# → 148 passed, 1 skipped (GPU-only Triton path)  ~60s on CPU
```

| File | Tests | What it covers |
|---|--:|---|
| `test_kernels.py` | 5 | Router fwd/bwd shapes, token conservation, NaN guard |
| `test_kernels_numerics.py` | 13 | 30-config Triton vs fp64 reference, `atol=rtol=1e-5` |
| `test_routing_quality.py` | 12 | Load imbalance math, z-loss invariants, RouterProfile completeness |
| `test_tensor_parallel.py` | 19 | Column/Row shape + grad + dtype + tp=1 equivalence; **2-rank mp.spawn** |
| `test_pipeline_parallel.py` | 16 | 1F1B schedule; **2-rank mp.spawn PP** (v0.3) |
| `test_distributed.py` | 4 | MoE layer fwd/bwd shapes, topology construction |
| `test_distributed_invariants.py` | 2 | 4-process Gloo: token conservation + NaN check |
| `test_elastic.py` | 7 | NVMe round-trip, chunked writes, async save/load, retention |
| `test_elastic_v02.py` | 10 | Reshard edge cases (primes, remainders), file-URI tier, harness |
| `test_mfu.py` | 6 | MFU formula, sparse fraction, backward compat |
| `test_mfu_v02.py` | 15 | MFUAccountant, MFUResult breakdown, smoothed window |
| `test_telemetry.py` | 17 | JSON completeness, 100-thread safety, WandB mock (v0.3) |
| `test_sequence_parallel_v03.py` | 8 | SP fused path; **2-rank mp.spawn SP** (v0.3) |
| `test_smoke_e2e.py` | 2 | Full train.py loop, all v0.3 envelope keys, S3 mock |
| `test_chaos.py` | 3 | Baseline ✅, Scenario B ✅, Scenario A ⚠️ |

---

## Running Tests

### Full non-chaos suite (recommended daily driver)

```bash
cd moe-engine
pytest tests/ -v --ignore=tests/test_chaos.py
```

### Specific test files

```bash
# Router kernel numerics
pytest tests/test_kernels.py tests/test_kernels_numerics.py -v

# Tensor parallelism (includes 2-rank mp.spawn end-to-end)
pytest tests/test_tensor_parallel.py -v

# Pipeline parallelism 1F1B schedule
pytest tests/test_pipeline_parallel.py -v

# Routing quality metrics (v0.2)
pytest tests/test_routing_quality.py -v

# Elastic fault tolerance
pytest tests/test_elastic.py tests/test_elastic_v02.py -v

# MFU accounting (v0.2)
pytest tests/test_mfu.py tests/test_mfu_v02.py -v

# Telemetry thread safety and field completeness (v0.2)
pytest tests/test_telemetry.py -v

# Smoke end-to-end
pytest tests/test_smoke_e2e.py -v
```

### Triton numerics driver (standalone)

Validates Triton kernel against fp64 reference across 30 parametrised
configurations without the pytest harness:

```bash
python tests/run_numerics_tests.py
```

Covers forward/backward correctness, token conservation, weight normalisation,
and deterministic behaviour across `H ∈ {64, 128, 256, 512}`,
`E ∈ {8, 16, 32, 64, 128, 256}`, `K ∈ {1, 2, 4}`.

### Benchmark suite

```bash
# CPU sweep (no GPU required)
python benchmarks/run_benchmark.py --json /tmp/bench.json

# GPU sweep (requires CUDA + Triton)
python benchmarks/run_benchmark.py --cuda --json /tmp/bench_gpu.json

# Token conservation sweep only
python benchmarks/run_benchmark.py --json /dev/null
```

---

## Chaos Tests

Chaos tests require `torchrun` on PATH and Gloo installed.
Use the loopback network interface on single machines.

```bash
# Baseline (no fault, warm path — always passes)
GLOO_SOCKET_IFNAME=lo pytest tests/test_chaos.py -v -k "baseline" -m chaos

# Scenario B — storage stall (10s injected I/O latency — always passes)
GLOO_SOCKET_IFNAME=lo pytest tests/test_chaos.py -v -k "scenario_b" -m chaos

# Scenario A — node kill + recovery (~85% pass rate; known Gloo race)
CHAOS_FAULT_TOLERANT=1 GLOO_SOCKET_IFNAME=lo \
  pytest tests/test_chaos.py -v -k "scenario_a" -m chaos
```

**Scenario A known issue:** `connectFullMesh` in the Gloo backend races with
socket cleanup after SIGKILL. The root cause and mitigation are documented in
`roadmap.md §Known Deficiencies`. Do not mark Scenario A as blocking in CI —
the `.github/workflows/ci.yml` runs it with `continue-on-error: true`.

### Measuring pass rate over many runs (v0.3.1, dependency fixed in v0.3.2)

To measure the actual Scenario A pass rate on your machine, run the test
multiple times. Two options, depending on what's installed:

**Option A — `pytest-repeat`** (now in `requirements.txt` directly as of
v0.3.2 — previously it was only in `pyproject.toml`'s `dev` extras, which a
plain `pip install -r requirements.txt && pip install -e .` would silently
skip):

```bash
pip install -r requirements.txt   # now includes pytest-repeat>=0.9.3
# or: pip install -e ".[dev]"

CHAOS_FAULT_TOLERANT=1 GLOO_SOCKET_IFNAME=lo \
  pytest tests/test_chaos.py -v -m chaos -k "scenario_a" --count=20
```

**Option B — bash loop** (zero extra dependencies, works anywhere):

```bash
PASS=0; FAIL=0
for i in $(seq 1 20); do
  if CHAOS_FAULT_TOLERANT=1 GLOO_SOCKET_IFNAME=lo \
     pytest tests/test_chaos.py -q -m chaos -k "scenario_a" >/dev/null 2>&1; then
    PASS=$((PASS+1))
  else
    FAIL=$((FAIL+1))
  fi
done
echo "Scenario A: ${PASS}/20 passed (${FAIL} failed)"
```

Both approaches are equivalent. Option B is recommended for one-off checks
(e.g., Colab notebooks) where installing an additional pytest plugin adds
unnecessary setup steps.

---

## Test Categories Explained

### Kernel correctness (`test_kernels.py`, `test_kernels_numerics.py`)

Every router invariant is tested independently so failures pinpoint the exact
broken property:

- **Token conservation:** `sum(dispatch_cnt) == N × K` across 100 random seeds
- **Index validity:** `idx ∈ [0, E)`, no NaN, no -1
- **Weight normalisation:** `w.sum(dim=-1) ≈ 1.0` (atol=1e-5)
- **Backward tolerance:** Triton grad matches fp64 reference at atol=rtol=1e-5
- **RouterProfile completeness:** all fields populated after every forward

### Routing quality metrics (`test_routing_quality.py`) — v0.2

Tests `expert_load_imbalance` and `router_z_loss` as independently testable
functions, not just as attributes on `RouterProfile`. Key invariants:

- `_compute_load_imbalance` — perfect balance → 1.0; all-to-one → E; zero → 1.0
- `_compute_router_z_loss` — always ≥ 0; zero logits → `log(E)²`; large logits > small
- `RouterProfile` — always populated after forward; v0.2 fields present
- Uniform gate init → lower imbalance than sharp gate init (5-seed parametrised)

### Tensor parallelism (`test_tensor_parallel.py`) — v0.2 upgraded

The most important test in this file is the **2-rank mp.spawn correctness test**
(`test_column_row_parallel_2rank_numerically_correct`). It:

1. Spawns 2 real Gloo CPU workers via `mp.spawn`.
2. Builds `ColumnParallelLinear(H→F)` + `RowParallelLinear(F→H)`.
3. Reconstructs full weights via `all_gather` on every rank.
4. Runs both the sharded path (Column+Row with real collectives) and the
   reference single-rank matmul.
5. Asserts max absolute difference < 1e-5.

This is the definitive proof the collectives are correct end-to-end.
Single-process tests at tp_size=1 alone are insufficient — they exercise
the identity path only.

Additional structural tests verify:
- `RowParallelLinear.forward` uses `all_reduce`, not `reduce_scatter_tensor`
  (source-inspection test to prevent regression of the v0.1 collective bug)
- `_SwiGLUExpert.w_gate` is `ColumnParallelLinear`, not `nn.Linear`
  (sharding consistency across the SwiGLU multiply)

### Pipeline parallelism (`test_pipeline_parallel.py`) — v0.3

Single-process tests (all 13 from v0.2, unchanged) verify the `run_1f1b`
scheduling invariants. Two new v0.3 tests exercise the real multi-process path:

- `test_pp_multiprocess_2stage_activation_flow` — 2-rank `mp.spawn` + Gloo:
  stage 0 (identity) → stage 1 (scale-by-2). Verifies activations flow
  through real `dist.send`/`dist.recv` and last stage outputs are 2× input.
- `test_pp_multiprocess_correct_micro_batch_count` — verifies last stage
  produces exactly `m` outputs across `m=6` micro-batches.
- `test_run_1f1b_distributed_raises_single_process` — verifies a clear
  `RuntimeError` when `run_1f1b_distributed` is called without `dist` + `pp_size>1`.

### Elastic fault tolerance (`test_elastic.py`, `test_elastic_v02.py`)

Core invariants exercised:

- `LocalNVMeAdapter` round-trip: write 256 MB chunks, O_DIRECT attempt, atomic rename
- `AsyncCheckpointer` save → load cycle with both tiers
- Retention pruning: only the latest N checkpoints survive; the most recent always survives
- Metadata `.meta.json`: step, rank, ts, hostname all present on both tiers
- Reshard plan completeness: all experts covered, no duplicates, across 7 topology
  configs including primes and non-divisible remainders
- `_largest_divisor_le`: 12 edge cases (primes, power-of-two, k > n, k = 1)
- `ClusterStateMachine` phase transitions: RUNNING → DRAINING → RECOVERING → RESUMED
- `install_signal_handlers` is a no-op (not a raise) when called from a non-main thread

### MFU accounting (`test_mfu.py`, `test_mfu_v02.py`)

- `compute_mfu_detailed` returns `MFUResult` with `flops_dense` and `flops_sparse`
- Activation recompute multiplier: recompute path has exactly 1.5× the FLOPs of
  the non-recompute path (3× vs 2× multiplier)
- `K/E` sparse fraction: k=4 has exactly 4× sparse FLOPs vs k=1 (parametrised)
- `MFUAccountant.smoothed_mfu`: only reflects last `smoothing_window` steps
- `MFUAccountant.summary_str()`: contains `tok/s`, `step=`, `MFU=`

### Telemetry (`test_telemetry.py`) — v0.3

- **Thread safety:** 100 concurrent `emit()` calls produce 100 uncorrupted JSONL
  lines with no duplicate step numbers
- **Field completeness:** all 12 keys in `REQUIRED_KEYS` present in every record
- **v0.2 routing fields round-trip:** `expert_load_imbalance`, `router_z_loss`
- **v0.3 overlap ratio round-trip:** `comm_compute_overlap_ratio` at 5 values
- **v0.3 WandB mock tests:** `WandBSink` disabled without `WANDB_API_KEY`;
  inactive at rank>0; `log()` calls `wandb.log` with correct section-prefixed
  keys including v0.3 fields; `log_config` calls `wandb.config.update`
- **Non-rank-0 suppresses TensorBoard:** `rank=1` must not create TB event files
- **`close()` idempotence:** second call must not raise

### Smoke end-to-end (`test_smoke_e2e.py`)

Runs a complete `train.py` loop (with `--smoke`) and asserts:
- All REQUIRED_KEYS present in JSONL (including v0.2 `routing` section)
- `routing.expert_load_imbalance` and `routing.router_z_loss` present
- v0.3: `collective.expert_compute_ms` and `collective.comm_compute_overlap_ratio` present
- Checkpoint shard written to local tier
- S3 upload attempted (mocked via `moto`)

---

## Writing New Tests

### Invariant-first pattern

```python
def test_my_new_invariant():
    """State the mathematical invariant in the docstring."""
    # Arrange: minimal setup
    layer = MyLayer(H=64, E=16)
    x = torch.randn(32, 64)

    # Act
    out, metadata = layer(x)

    # Assert: one specific invariant per test function
    assert out.shape == (32, 64), f"Expected [32,64], got {out.shape}"
```

### Multi-process pattern (for distributed tests)

```python
def _worker(rank, world_size, result_queue):
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    # ... exercise the distributed primitive ...
    result_queue.put((rank, result_value))
    dist.destroy_process_group()

def test_my_distributed_primitive():
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [ctx.Process(target=_worker, args=(r, 2, q)) for r in range(2)]
    for p in procs: p.start()
    for p in procs: p.join(timeout=60); assert p.exitcode == 0
    results = {q.get_nowait() for _ in range(2)}
    # assert on results
```

### Pytest markers

```python
@pytest.mark.chaos          # torchrun-based; only runs with -m chaos
@pytest.mark.skipif(        # GPU-only
    not torch.cuda.is_available(), reason="CUDA required"
)
```

### Sequence Parallelism fused path (`test_sequence_parallel_v03.py`) — v0.3

Tests the `next_weight` fused path introduced in v0.3:

- At `tp_size=1`: fused output matches `nn.functional.linear(x, w)` exactly (7 configs)
- `test_sp_fused_2rank_numerically_correct` — 2-rank `mp.spawn` + Gloo: each rank
  computes `shard @ w.T` locally then `all_reduce(SUM)`; result matches full-sequence
  `linear(x, w)` to atol=1e-5. This is the definitive multi-process SP correctness proof.
- Scatter-only path (next_weight=None) is unchanged and identity at tp_size=1

### What every new test must have

1. A docstring explaining the invariant being tested, not just the mechanism.
2. An assertion message that explains what went wrong and what was expected.
3. A `pytest.mark` if it is slow, GPU-only, or chaos-dependent.
4. A corresponding entry in this table and in `CONTRIBUTING.md` if it adds a new coverage area.

---

## CI Integration

The GitHub Actions workflow (`.github/workflows/ci.yml`) runs:

| Job | Trigger | What runs |
|---|---|---|
| `lint` | every push + PR | ruff + mypy |
| `unit` | every push + PR | full suite (148 tests) on Python 3.10 and 3.11 |
| `benchmark` | every push + PR | CPU benchmark smoke (verifies no regression) |
| `docker` | push to main/dev | Docker build smoke |
| `chaos` | push to main only | Scenario B (blocking) + Scenario A (non-blocking) |

Scenario A failures in CI are expected and non-blocking. Any other failure is
a blocker that must be resolved before merge.
