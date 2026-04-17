# Benchmarks Guide

**Version:** v0.3  
**Last updated:** June 2026

This document explains how to measure performance using moe-engine's built-in
tooling, what each metric means, and how to interpret results. For the actual
benchmark numbers, see `benchmarks/BENCHMARKS.md` and `RESULTS.md`.

---

## Benchmark Tooling

### `benchmarks/run_benchmark.py` — Reproducible micro-benchmark suite

Measures router forward/backward throughput, MoE layer latency, and token
conservation across a range of configurations. Runs on CPU (no GPU required)
or GPU with `--cuda`.

```bash
# CPU sweep (reproducible anywhere)
python benchmarks/run_benchmark.py --json results.json --csv results.csv

# GPU sweep (requires CUDA + Triton)
python benchmarks/run_benchmark.py --cuda --json gpu_results.json

# Parse results
python -c "
import json
for r in json.load(open('results.json')):
    status = 'PASS' if r['passed'] else 'FAIL'
    print(f\"{status} {r['name']:30s} {r['batch_ms_mean']:7.2f}ms  {r['tokens_per_sec']/1e6:.2f}M tok/s\")
"
```

Each benchmark run:
1. Runs 3 warmup iterations (discarded).
2. Times 20 measured iterations.
3. Reports mean and standard deviation of latency.
4. Derives throughput (tokens/sec) from mean latency.
5. Computes a rough MFU estimate against H100 peak (989 TFLOPS BF16).
6. Asserts correctness invariants (token conservation for the sweep test).

### Training loop profiling with `--profile`

```bash
python train.py --config configs/smoke.yaml --smoke --profile
# Writes: benchmarks/run_<timestamp>_rank0.json
```

With `torchrun` (GPU, 4 ranks):

```bash
torchrun --standalone --nproc_per_node=4 \
  train.py --config configs/default.yaml --max-steps 100 --profile
# Rank 0 writes: benchmarks/run_<timestamp>_rank0.json
```

The profile JSON contains per-step records with:
`step`, `step_ms`, `mfu`, `tokens_per_sec`, `loss`, `dispatch_ms`,
`combine_ms`, `load_imbalance`.

---

## Metrics Reference

### MFU (Model FLOPs Utilization)

```
MFU = achieved_TFLOPs / (world_size × hardware_peak_TFLOPs)
```

For a MoE model with sparse expert activation:

```
achieved_TFLOPs = (flops_dense + flops_sparse) / step_time_s / 1e12

flops_dense  = 2 × batch_tokens × P_dense
flops_sparse = 2 × batch_tokens × (K / E) × P_expert
```

The `K/E` factor accounts for the fact that only `K` of `E` experts fire per
token. Ignoring this produces artificially high MFU numbers. Activation
recompute (when enabled) multiplies FLOPs by 1.5× (3× fwd instead of 2×).

`compute_mfu_detailed()` returns a `MFUResult` with `flops_dense` and
`flops_sparse` separately so you can see how much compute is in attention vs.
experts.

Set `hardware_peak_tflops` in your config to the correct value for your GPU:

| GPU | BF16 TFLOPS |
|---|--:|
| H100 SXM5 | 989 |
| H100 PCIe | 756 |
| A100 SXM4 | 312 |
| A100 PCIe | 312 |
| RTX 4090 | 165 |

### Routing quality metrics (v0.2)

**`expert_load_imbalance = max(dispatch_cnt) / mean(dispatch_cnt)`**

- 1.0 = perfect balance; every expert receives the same number of tokens.
- 1.1–1.3 = acceptable range for random input.
- > 1.5 sustained = pathological routing; consider adding z-loss auxiliary.
- > 2.0 = routing collapse; all tokens routed to same few experts.

**`router_z_loss = mean(log(Σ exp(logit_e))²)`**

- The Switch-Transformer auxiliary loss. Encourages small logit magnitudes,
  which correlates with uniform routing.
- Emitted as a telemetry signal, not automatically added to the training loss.
- Add `z_loss_weight × router_z_loss` to your loss function to reduce imbalance.
  Typical weight: 1e-3.

### Collective latency

**`all_to_all_dispatch_ms`** and **`all_to_all_combine_ms`** are measured with
real CUDA events on the dedicated EP stream.

**`expert_compute_ms`** (v0.3) — wall-clock time of the expert FFN compute
(all local experts) measured with `time.perf_counter`. At ep=1 this is the
dominant compute cost; higher overlap_ratio means better utilisation.

**`comm_compute_overlap_ratio`** (v0.3) — `dispatch_ms / expert_compute_ms`.
A value near 1.0 means dispatch duration ≈ expert compute duration, so
overlap is near-complete. A value > 1.0 means dispatch is the bottleneck
(communication-bound). Target: 0.3–0.6 at EP=8 with NVLink. At `ep_size=1` both are always 0.0
(no collective issued). At `ep_size>1` with NVLink, typical values at EP=8:

| Metric | Expected range (H100 NVLink) |
|---|---|
| dispatch_ms | 0.3–0.8 ms |
| combine_ms | 0.3–0.8 ms |
| dispatch/(dispatch + expert_compute) | 20–40% |

If dispatch_ms > 2ms at EP=8, check: (a) NVLink vs PCIe topology,
(b) NCCL algorithm selection (`NCCL_ALGO`), (c) packet size alignment.

### Throughput

**`tokens_per_sec`** = `batch_tokens / step_time_s`

For comparison: GPT-3 (175B dense) training was reported at ~150K tokens/sec
across 1024 A100s. A 70B MoE model at 128 H100s should target ≥ 300K tokens/sec
at 45%+ MFU. The exact number depends on EP size, sequence length, and batch size.

---

## How to Benchmark Correctly

### 1. Warm up before measuring

The first 5–10 training steps incur JIT compilation (Triton), CUDA graph
capture, and optimizer state allocation. Exclude these from MFU averages.
`MFUAccountant.smoothed_mfu` uses a sliding window (default 50 steps) which
naturally discards early noisy steps.

### 2. Use stable batch sizes

MFU is inversely proportional to step time. Larger batches hide communication
overhead and increase MFU — this is expected behaviour, not a bug. Compare
MFU numbers at the same tokens-per-step.

### 3. Record configuration alongside numbers

Every benchmark result must be paired with:
- `world_size`, `dp`, `ep`, `tp`, `pp`
- `hidden_dim`, `num_experts`, `top_k`, `ffn_dim`, `dtype`
- `sequence_length`, `micro_batch_size`, `gradient_accumulation_steps`
- GPU model and driver version
- PyTorch + CUDA + Triton versions

Without these, numbers are not reproducible.

### 4. Run for at least 100 steps

Step time variance from GC, OS scheduler, and NCCL algorithm selection can be
10–20% between individual steps. Averages over 100+ steps are stable.

### 5. Compare relative, not absolute

CPU reference path MFU will always be near 0% (no CUDA hardware). GPU MFU
should be compared between configurations on the same hardware, or against
published numbers on the same GPU model.

---

## Interpreting Telemetry Output

### What "good" looks like on a single H100 node (8 GPUs)

At `hidden_dim=4096, E=64, K=2, B=4, S=4096, EP=8, TP=1`:

| Metric | Target |
|---|---|
| MFU | 40–55% |
| tokens_per_sec | 60K–100K |
| dispatch_ms | < 0.8 ms |
| combine_ms | < 0.8 ms |
| expert_load_imbalance | < 1.3 |
| peak_allocated_gb | < 70 GB |

### Common low-MFU causes

| Symptom | Likely cause |
|---|---|
| MFU < 20% | Batch too small; GPU starved |
| dispatch_ms > 2ms | PCIe instead of NVLink; NCCL algorithm |
| expert_load_imbalance > 1.5 | Routing collapse; add z-loss |
| step_ms highly variable | GC pressure; increase `gradient_accumulation_steps` |
| peak_allocated_gb >> expected | Gradient accumulation buffers not freed; call `optimizer.zero_grad(set_to_none=True)` |

---

## Adding Benchmark Results to the Repo

If you capture reproducible numbers on real hardware, add them to
`benchmarks/BENCHMARKS.md` with:

1. Hardware description (GPU model, count, interconnect)
2. Software versions (PyTorch, CUDA, Triton)
3. Exact config file used (or diff from default)
4. Command line
5. Summary statistics: MFU mean/p50, tokens_per_sec mean, dispatch_ms
6. Any environment caveats (numa binding, CPU affinity, etc.)

GPU numbers in `benchmarks/BENCHMARKS.md` are currently marked as illustrative
pending sustained cluster access. Real measurements will replace them in v0.3.
