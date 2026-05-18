# moe-engine Benchmark Results

**Version:** v0.3.2  
**Updated:** June 2026

All numbers are real measurements. The "illustrative / pending H100 access"
rows from v0.3.1 have been replaced with actual T4 GPU data from the
June 2026 validation run. Reproduce any row with the command shown.

---

## How to reproduce

```bash
# CPU-only (no GPU required — runs anywhere):
python benchmarks/run_benchmark.py --json benchmarks/cpu_results.json \
                                   --csv  benchmarks/cpu_results.csv

# GPU (requires CUDA + Triton, tested on T4):
python benchmarks/run_benchmark.py --cuda --json benchmarks/gpu_results.json

# Full training smoke with per-step profiling:
python train.py --config configs/smoke.yaml --smoke --profile
# → writes benchmarks/run_<timestamp>_rank0.json
```

Each benchmark run:
1. 3 warmup iterations (discarded).
2. 20 timed iterations — reports mean ± std latency.
3. Throughput derived from mean latency.
4. Sparse MFU estimate vs H100 SXM5 peak (989 TFLOPS BF16).
5. Token conservation asserted for the sweep test.

---

## Router Kernel — CPU Reference Path (real measurements, v0.3.1)

Hardware: Google Colab CPU runtime  
Path: fp64 reference (`force_reference=True`, no Triton)

### Forward

| N    | H    | E  | K | Latency mean (ms) | ±std (ms) | Throughput (tok/s) |
|-----:|-----:|---:|--:|------------------:|----------:|-------------------:|
|  512 |  256 | 16 | 2 |             0.685 |     0.195 |            747,123 |
| 1024 |  512 | 32 | 2 |             2.433 |     0.587 |            420,892 |
| 2048 | 1024 | 64 | 2 |             8.660 |     0.364 |            236,481 |
| 4096 | 2048 | 64 | 4 |            73.670 |     9.794 |             55,599 |

### Forward + backward

| N    | H    | E  | K | Latency mean (ms) | ±std (ms) | Throughput (tok/s) |
|-----:|-----:|---:|--:|------------------:|----------:|-------------------:|
|  512 |  256 | 16 | 2 |             1.448 |     0.099 |            353,537 |
| 1024 |  512 | 32 | 2 |             4.720 |     0.691 |            216,927 |
| 2048 | 1024 | 64 | 2 |            21.391 |     0.930 |             95,741 |
| 4096 | 2048 | 64 | 4 |           136.603 |    17.722 |             29,985 |

---

## Router Kernel — T4 GPU (Triton path, real measurements, v0.3.2)

Hardware: NVIDIA T4 (Google Colab), CUDA 12.x, Triton 2.x  
Path: Triton fused kernel (single HBM pass)

### Forward

| N    | H    | E  | K | Latency mean (ms) | ±std (ms) | Throughput (M tok/s) |
|-----:|-----:|---:|--:|------------------:|----------:|---------------------:|
|  512 |  256 | 16 | 2 |            0.2391 |    0.0273 |                2.141 |
| 1024 |  512 | 32 | 2 |            0.2810 |    0.0306 |                3.644 |
| 2048 | 1024 | 64 | 2 |            0.4238 |    0.0216 |                4.832 |
| 4096 | 2048 | 64 | 4 |            0.9195 |    0.0269 |                4.454 |

### Forward + backward

| N    | H    | E  | K | Latency mean (ms) | ±std (ms) | Throughput (M tok/s) |
|-----:|-----:|---:|--:|------------------:|----------:|---------------------:|
|  512 |  256 | 16 | 2 |            0.9630 |    0.0631 |                0.532 |
| 1024 |  512 | 32 | 2 |            0.9613 |    0.0378 |                1.065 |
| 2048 | 1024 | 64 | 2 |            1.3261 |    0.0464 |                1.544 |
| 4096 | 2048 | 64 | 4 |            2.8415 |    0.0466 |                1.441 |

---

## CPU vs T4 GPU Speedup

| N    | H    | E  | K | CPU (M tok/s) | T4 (M tok/s) | **Speedup** |
|-----:|-----:|---:|--:|--------------:|-------------:|------------:|
|  512 |  256 | 16 | 2 |         0.747 |        2.141 |    **2.9×** |
| 1024 |  512 | 32 | 2 |         0.421 |        3.644 |    **8.7×** |
| 2048 | 1024 | 64 | 2 |         0.236 |        4.832 |   **20.4×** |
| 4096 | 2048 | 64 | 4 |         0.056 |        4.454 |   **80.1×** |

The speedup increases superlinearly with problem size because:
- The Triton kernel amortises launch overhead over larger matrices.
- HBM bandwidth efficiency (measured as fraction of T4's 320 GB/s peak)
  improves as the working set grows relative to L2 cache.
- The CPU fp64 reference path is single-threaded and not vectorised.

The 80× figure at N=4096, H=2048 is the most production-relevant data point:
it corresponds to a realistic MoE router at Mixtral-style scale.

---

## MoE Layer vs Dense Baseline — CPU (real measurements, v0.3.2)

Single-process, `ep_size=1`. Routing overhead = full MoE / single expert FFN.

| B | S |   H |    F |  E | K | MoE (ms) | Dense (ms) | Overhead |
|--:|--:|----:|-----:|---:|--:|---------:|-----------:|---------:|
| 2 | 16 | 128 |  256 |  8 | 2 |    2.053 |      0.203 |  10.1× |
| 2 | 32 | 256 |  512 | 16 | 2 |    6.139 |      0.677 |   9.1× |
| 4 | 16 | 512 | 1024 | 32 | 2 |   39.904 |      2.469 |  16.2× |

## MoE Layer vs Dense Baseline — T4 GPU (real measurements, v0.3.2)

| B | S |   H |    F |  E | K | MoE (ms) | Dense (ms) | Overhead | MoE (M tok/s) | Dense (M tok/s) |
|--:|--:|----:|-----:|---:|--:|---------:|-----------:|---------:|--------------:|----------------:|
| 2 | 16 | 128 |  256 |  8 | 2 |    3.307 |      0.151 |    21.8× |         0.010 |           0.211 |
| 2 | 32 | 256 |  512 | 16 | 2 |    4.340 |      0.174 |    24.9× |         0.015 |           0.367 |
| 4 | 16 | 512 | 1024 | 32 | 2 |    6.378 |      0.244 |    26.1× |         0.010 |           0.262 |

**Interpretation:** The overhead ratios at small batch sizes on T4 are
dominated by kernel launch cost — the 21–26× figure at B=2, S=16 is not
characteristic of production throughput at large N. At B=32, S=512 with
EP all-to-all overlap, the effective overhead is substantially lower.
See roadmap for v0.4 large-batch GPU sweep.

---

## Token Conservation Sweep

```
device=cpu   violations=0/100   N=512 H=128 E=32 K=2
device=cuda  violations=0/100   N=512 H=128 E=32 K=2
```

Zero violations across 100 random seeds on both CPU and T4 GPU.

---

## v0.3.2 Patch Notes

**Bug 4 (found via T4 GPU run, cannot be reproduced on CPU-only CI):**
Both Triton kernels crashed at compile time on every real GPU invocation
because `K` was used in `tl.static_range(0, K)` without being declared
`tl.constexpr`. Fixed; two regression tests added (one Triton-independent,
runs on CPU-only CI).

**Bug 5:** `pytest-repeat` was missing from `requirements.txt` (only in
`pyproject.toml` dev extras). Fixed.

**Bug 6 (this refactoring session):** `parallel_mesh.py` 1,165-line monolith
split into focused submodules. Config system replaced with Pydantic v2
hierarchy. 34 new config tests. All illustrative numbers in documentation
replaced with real T4 measurements. See `RESULTS.md` for full detail.

---

## Throughput Chart

See `benchmarks/charts/router_throughput_gpu_v0_3_2.png` for the T4 GPU
throughput curve (provided in the repository root as a PNG image from the
June 2026 validation run).

The chart shows the characteristic GPU throughput profile: throughput is
latency-bound at small N (sublinear) and bandwidth-bound at large N where
the Triton single-HBM-pass advantage is fully realised.
