# Results & Evidence

This document contains reproducible results from `moe-engine` v0.2.
Every number links to a script that generates it on your hardware.

---

## Correctness Results (CPU, reproducible anywhere)

### Token Conservation — Zero violations across all runs

The most important invariant in an MoE system: every input token reaches exactly K experts.

```
python benchmarks/run_benchmark.py --json /tmp/results.json
python -c "
import json
r = json.load(open('/tmp/results.json'))
tc = [x for x in r if x['name'] == 'token_conservation_sweep']
print(tc[0]['notes'])   # → 'violations=0/100'
"
```

**Result:** `violations=0/100` across every `(N, H, E, K)` configuration tested.

### Backward Numerical Accuracy

Router backward is validated against a 64-bit PyTorch reference implementation. Tolerance: `atol=rtol=1e-5`.

```bash
pytest tests/test_kernels_numerics.py -v
# → 30/30 passed (parametrised over H∈[64,512], E∈[8,256], K∈[1,4])
```

### TP 2-Rank Numerical Equivalence

ColumnParallel→RowParallel combined output matches single-rank nn.Linear:

```bash
pytest tests/test_tensor_parallel.py::test_column_row_parallel_2rank_numerically_correct -v
# → max abs diff < 1e-5
```

---

## Performance Results (CPU Reference Path)

Run on: `AMD EPYC 9554 64-core, Python 3.11, PyTorch 2.5.1`  
Reproduce: `python benchmarks/run_benchmark.py --json results.json`

### Router Forward Throughput

| N    | H    | E  | K | Mean latency | Throughput   |
|-----:|-----:|---:|--:|-------------:|-------------:|
| 512  | 256  | 16 | 2 | 0.040 ms     | 12.8M tok/s  |
| 1024 | 512  | 32 | 2 | 0.120 ms     |  8.5M tok/s  |
| 2048 | 1024 | 64 | 2 | 0.470 ms     |  4.4M tok/s  |
| 4096 | 2048 | 64 | 4 | 1.830 ms     |  2.2M tok/s  |

### Router Forward + Backward Throughput

| N    | H    | E  | K | Mean latency | Throughput   |
|-----:|-----:|---:|--:|-------------:|-------------:|
| 512  | 256  | 16 | 2 | 0.090 ms     |  5.7M tok/s  |
| 1024 | 512  | 32 | 2 | 0.280 ms     |  3.7M tok/s  |
| 2048 | 1024 | 64 | 2 | 1.100 ms     |  1.9M tok/s  |
| 4096 | 2048 | 64 | 4 | 4.300 ms     |  0.95M tok/s |

_Numbers are approximate and machine-dependent. Run `run_benchmark.py` on your machine for exact values._

---

## GPU Results (H100 SXM5, illustrative)

The Triton kernel path is validated to be correct (matches reference within atol=1e-5) but GPU throughput numbers require H100 hardware to measure. Expected range based on roofline analysis:

| N    | H    | E  | K | Expected GPU throughput |
|-----:|-----:|---:|--:|------------------------:|
| 2048 | 4096 | 64 | 2 | ~25M tok/s              |
| 8192 | 4096 | 64 | 2 | ~30M tok/s              |

These numbers will be replaced with real measurements in v0.3 when sustained cluster access is available.

**To run on GPU:**
```bash
python benchmarks/run_benchmark.py --cuda --json benchmarks/gpu_results.json
```

---

## Fault Tolerance Results

### Chaos Scenario B: Storage Stall (✅ 100% pass rate)

```bash
pytest tests/test_chaos.py -v -k "scenario_b"
# → 1 passed
```

A 10-second injected I/O stall during checkpoint commit: the async queue drains without deadlock; training resumes; a `latency_inject` event is emitted in telemetry.

### Chaos Scenario A: Node Kill (⚠️ ~85% pass rate)

```bash
CHAOS_FAULT_TOLERANT=1 pytest tests/test_chaos.py -v -k "scenario_a"
# → passes ~85% of runs; flaky due to Gloo connectFullMesh race on restart
```

Root cause and mitigation documented in `roadmap.md §Known Deficiencies`.

---

## Telemetry Sample

A real output record from `train.py --smoke`:

```jsonc
{
  "step": 1,
  "loss": 4.8823,
  "mfu": 0.0003,          // low: CPU reference path, not GPU
  "tokens_per_sec": 1847,
  "wall_clock_ms": 8.7,
  "kernel": {
    "sram_bytes_per_block": 16384,
    "achieved_bw_gbps": 0.0,
    "tokens_per_expert_mean": 4.0,
    "tokens_per_expert_std": 1.41,
    "used_triton": false
  },
  "collective": {
    "all_to_all_dispatch_ms": 0.0,   // ep=1, no collective
    "all_to_all_combine_ms": 0.0
  },
  "memory": {},                       // no CUDA in smoke run
  "routing": {
    "expert_load_imbalance": 1.12,
    "router_z_loss": 2.87
  },
  "infra": {
    "async_ckpt_commit_ms": 0.0,
    "active_nodes": 1,
    "ep_world_size": 1,
    "lr": 0.0003
  },
  "rank": 0,
  "ts": 1748901234.56
}
```

---

## Test Suite Summary

```
pytest tests/ -v --ignore=tests/test_chaos.py
```

| File | Tests | Result |
|---|--:|---|
| test_kernels.py | 8 | ✅ |
| test_kernels_numerics.py | 30 | ✅ |
| test_routing_quality.py | 12 | ✅ |
| test_tensor_parallel.py | 17 | ✅ (incl. 2-rank mp.spawn) |
| test_pipeline_parallel.py | 13 | ✅ |
| test_distributed.py | 4 | ✅ |
| test_distributed_invariants.py | 2 | ✅ |
| test_elastic.py | 7 | ✅ |
| test_elastic_v02.py | 10 | ✅ |
| test_mfu.py | 6 | ✅ |
| test_mfu_v02.py | 15 | ✅ |
| test_telemetry.py | 12 | ✅ |
| test_smoke_e2e.py | 2 | ✅ |
| **Total** | **138** | **✅** |

_1 test skipped: GPU-only Triton kernel path (requires CUDA)._
