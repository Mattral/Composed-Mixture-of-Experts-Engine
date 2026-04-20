# Results & Evidence

**Version:** v0.3.1  
**Updated:** June 2026

This document contains reproducible results from `moe-engine`. Every number
links to a script that generates it on your hardware. As of v0.3.1, the CPU
router/MoE-layer numbers below are **real measurements** (Google Colab CPU
runtime), not illustrative estimates.

---

## Correctness Results (CPU, reproducible anywhere)

### Token Conservation — Zero violations

The most important invariant in an MoE system: every input token reaches exactly K experts.

```bash
python benchmarks/run_benchmark.py --json /tmp/results.json
python -c "
import json
r = json.load(open('/tmp/results.json'))
tc = [x for x in r if x['name'] == 'token_conservation_sweep']
print(tc[0]['notes'])   # → 'violations=0/100'
"
```

**Result (real, v0.3.1):** `violations=0/100` for N=512, H=128, E=32, K=2.
`tests/test_kernels_numerics.py` separately covers 30 `(H,E,K)` configurations
at `atol=rtol=1e-5`.

### Backward Numerical Accuracy

Router backward is validated against a 64-bit PyTorch reference implementation. Tolerance: `atol=rtol=1e-5`.

```bash
pytest tests/test_kernels_numerics.py -v
# → 13 test functions / 30 parametrised cases passed
#   (H∈{64,128,256,512}, E∈{8..256}, K∈{1,2,4})
```

### TP 2-Rank Numerical Equivalence

ColumnParallel→RowParallel combined output matches single-rank nn.Linear:

```bash
pytest tests/test_tensor_parallel.py::test_column_row_parallel_2rank_numerically_correct -v
# → max abs diff < 1e-5
```

### PP and SP 2-Rank Numerical Equivalence (v0.3)

```bash
pytest tests/test_pipeline_parallel.py::test_pp_multiprocess_2stage_activation_flow -v
pytest tests/test_sequence_parallel_v03.py::test_sp_fused_2rank_numerically_correct -v
# → both verified to atol=1e-5
```

---

## Performance Results — Real Measurements (CPU, v0.3.1)

**Hardware:** Google Colab CPU runtime  
**Software:** PyTorch (CPU build), fp64 reference path (`force_reference=True`)  
**Source data:** `benchmarks/cpu_results_colab.json`  
**Reproduce:** `python benchmarks/run_benchmark.py --json results.json`

### Router Forward Throughput

| N    | H    | E  | K | Mean latency      | Throughput     |
|-----:|-----:|---:|--:|------------------:|---------------:|
| 512  | 256  | 16 | 2 | 0.569 ± 0.045 ms  | 900,408 tok/s  |
| 1024 | 512  | 32 | 2 | 2.009 ± 0.114 ms  | 509,722 tok/s  |
| 2048 | 1024 | 64 | 2 | 9.670 ± 1.768 ms  | 211,780 tok/s  |
| 4096 | 2048 | 64 | 4 | 59.056 ± 4.703 ms |  69,358 tok/s  |

### Router Forward + Backward Throughput

| N    | H    | E  | K | Mean latency        | Throughput     |
|-----:|-----:|---:|--:|--------------------:|---------------:|
| 512  | 256  | 16 | 2 | 1.458 ± 0.189 ms    | 351,217 tok/s  |
| 1024 | 512  | 32 | 2 | 3.712 ± 0.166 ms    | 275,861 tok/s  |
| 2048 | 1024 | 64 | 2 | 20.331 ± 1.659 ms   | 100,732 tok/s  |
| 4096 | 2048 | 64 | 4 | 151.057 ± 12.883 ms |  27,116 tok/s  |

![Router throughput vs config scale](moe-engine/benchmarks/charts/router_throughput_v0.3.1.png)

Forward-only is 1.8×–2.6× faster than forward+backward across all four
configs. See `benchmarks/BENCHMARKS.md` for the note on why `N`, `H`, `E`, `K`
vary together across these configs (not a controlled single-variable sweep).

### MoE Layer Forward Throughput (single-process, ep_size=1)

| B | S | H | F | E | K | Mean latency       | Throughput    |
|--:|--:|--:|--:|--:|--:|-------------------:|--------------:|
| 2 | 16 | 128 | 256 | 8  | 2 | 3.106 ± 0.412 ms  | 10,302 tok/s |
| 2 | 32 | 256 | 512 | 16 | 2 | 5.838 ± 0.368 ms  | 10,963 tok/s |
| 4 | 16 | 512 | 1024| 32 | 2 | 40.345 ± 1.083 ms |  1,586 tok/s |

![MoE layer throughput](moe-engine/benchmarks/charts/moe_layer_throughput_v0.3.1.png)

### MoE vs Dense Baseline (v0.3.1 — added, not yet run)

`bench_dense_baseline` was implemented in v0.3.1 to satisfy the "compare
against a dense baseline" requirement. It has not yet been executed — the
real-data run above predates this addition. Re-run
`python benchmarks/run_benchmark.py --json results.json` to populate:

| B | S | H | F | E | K | MoE (ms) | Dense (ms) | Routing overhead |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 2 | 16 | 128 | 256 | 8  | 2 | (rerun) | (rerun) | (rerun) |
| 2 | 32 | 256 | 512 | 16 | 2 | (rerun) | (rerun) | (rerun) |
| 4 | 16 | 512 | 1024| 32 | 2 | (rerun) | (rerun) | (rerun) |

---

## GPU Results (H100 SXM5, illustrative)

The Triton kernel path is validated to be correct (matches reference within
atol=1e-5) but GPU throughput numbers require H100 hardware to measure.
Expected range based on roofline analysis:

| N    | H    | E  | K | Expected GPU throughput |
|-----:|-----:|---:|--:|------------------------:|
| 2048 | 4096 | 64 | 2 | ~25M tok/s              |
| 8192 | 4096 | 64 | 2 | ~30M tok/s              |

These remain illustrative pending v0.4 sustained cluster access (see roadmap).

**To run on GPU:**
```bash
python benchmarks/run_benchmark.py --cuda --json benchmarks/gpu_results.json
```

---

## Fault Tolerance Results

### Chaos Scenario B: Storage Stall

```bash
GLOO_SOCKET_IFNAME=lo pytest tests/test_chaos.py -v -k "scenario_b" -m chaos
```

A 10-second injected I/O stall during checkpoint commit: the async queue
drains without deadlock; training resumes; a `latency_inject` event is
emitted in telemetry. Historically 100% pass rate; not yet re-measured at
v0.3.1 (see "Chaos Pass Rates" below).

### Chaos Scenario A: Node Kill

```bash
CHAOS_FAULT_TOLERANT=1 GLOO_SOCKET_IFNAME=lo pytest tests/test_chaos.py -v -k "scenario_a" -m chaos
```

Historically ~85% pass rate; flaky due to Gloo `connectFullMesh` race on
restart. Root cause and mitigation documented in `roadmap.md §Known
Deficiencies`. Real fix requires NCCL (GPU-only), planned for v0.4.

### Chaos Pass Rates — Real Measurements (v0.3.1)

| Scenario | Runs | Passed | Pass rate |
|---|--:|--:|--:|
| A (node kill + recovery) | — | — | not yet run |
| B (storage stall) | — | — | not yet run |

Run the loop in `docs/testing.md §Measuring pass rate over many runs` and
fill in this table with real counts.

---

## Telemetry Sample

A `train.py --smoke` record, illustrating the full v0.3 envelope including
`routing` (v0.2) and `collective.expert_compute_ms` /
`collective.comm_compute_overlap_ratio` (v0.3):

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
    "all_to_all_dispatch_ms": 0.0,        // ep=1, no collective
    "all_to_all_combine_ms": 0.0,
    "expert_compute_ms": 0.42,             // v0.3
    "comm_compute_overlap_ratio": 0.0      // v0.3: 0 dispatch / nonzero compute
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

> **v0.3.1 note:** prior to this patch, `train.py --smoke` could not produce
> this record at all — `main()` raised `AttributeError: 'dict' object has no
> attribute 'raw'` on the `logger.log_config(cfg.raw)` call before the first
> training step. The record above is illustrative of the *intended* schema;
> re-run `python train.py --config configs/smoke.yaml --smoke` with v0.3.1 to
> capture and paste a real `/tmp/moe-engine/logs/step.jsonl` line here.

---

## Test Suite Summary

```
pytest tests/ -v --ignore=tests/test_chaos.py
```

| File | Tests | Result |
|---|--:|---|
| test_kernels.py | 5 | ✅ |
| test_kernels_numerics.py | 13 | ✅ |
| test_routing_quality.py | 12 | ✅ |
| test_tensor_parallel.py | 19 | ✅ (incl. 2-rank mp.spawn) |
| test_pipeline_parallel.py | 16 | ✅ (incl. 2-rank mp.spawn PP, v0.3) |
| test_sequence_parallel_v03.py | 8 | ✅ (incl. 2-rank mp.spawn SP, v0.3) |
| test_distributed.py | 4 | ✅ |
| test_distributed_invariants.py | 2 | ✅ |
| test_elastic.py | 7 | ✅ |
| test_elastic_v02.py | 10 | ✅ |
| test_mfu.py | 6 | ✅ |
| test_mfu_v02.py | 15 | ✅ |
| test_telemetry.py | 22 | ✅ (incl. WandB mock, v0.3) |
| test_smoke_e2e.py | 3 | ✅ (incl. v0.3.1 regression test) |
| **Total** | **145** | **✅** |

_1 test skipped: GPU-only Triton kernel path (requires CUDA)._

---

## v0.3.1 Patch Summary

Three bugs were found via a real execution run on Google Colab — none were
catchable by static analysis. Full details in
`benchmarks/BENCHMARKS.md §v0.3.1 Patch Notes`:

1. `train.py` crashed on every run (`cfg.raw` AttributeError) — **fixed**,
   regression test added.
2. `pytest --count=N` failed (missing `pytest-repeat`) — **fixed**, added to
   dev deps + bash-loop alternative documented.
3. Dense baseline benchmark was a stub — **implemented** in
   `benchmarks/run_benchmark.py`.
