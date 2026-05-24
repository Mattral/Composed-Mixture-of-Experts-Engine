# Benchmarks Guide

**Version:** v0.3.2  
**Last updated:** June 2026

This document explains how to measure performance using moe-engine's built-in
tooling, what each metric means, and how to interpret results correctly.
For the actual measured numbers, see `moe-engine/benchmarks/BENCHMARKS.md`
(per-config tables) and `RESULTS.md` (summary with context).

---

## Benchmark Tooling

### `benchmarks/run_benchmark.py` — Reproducible micro-benchmark suite

Measures router forward/backward throughput, full MoE layer latency versus a
dense baseline, and the token conservation sweep across a range of
configurations. Runs on CPU with no GPU required, or on a real GPU with `--cuda`.

```bash
# CPU sweep (reproducible on any machine, no GPU needed)
cd moe-engine/
python benchmarks/run_benchmark.py \
    --json benchmarks/cpu_results.json \
    --csv  benchmarks/cpu_results.csv

# GPU sweep (requires CUDA + Triton, e.g. T4 or H100)
python benchmarks/run_benchmark.py \
    --cuda \
    --json benchmarks/gpu_results.json \
    --csv  benchmarks/gpu_results.csv

# Parse results programmatically
python -c "
import json
for r in json.load(open('benchmarks/gpu_results.json')):
    status = 'PASS' if r['passed'] else 'FAIL'
    print(f\"{status} {r['name']:28s} device={r['device']}  \"\
          f\"{r['batch_ms_mean']:7.3f}±{r['batch_ms_std']:.3f}ms  \"\
          f\"{r['tokens_per_sec']/1e6:.3f}M tok/s\")
"
```

**What each run does:**

1. **Warmup** — 3 iterations discarded to eliminate JIT compilation
   (`triton.jit`) and CUDA graph capture from the measured time.
2. **Measurement** — 20 timed iterations using `time.perf_counter()` on CPU
   or CUDA events on GPU (higher accuracy, includes kernel synchronisation).
3. **Statistics** — mean and standard deviation of per-iteration latency.
4. **Throughput** — `N / batch_ms_mean × 1000` tokens/sec derived from mean.
5. **MFU estimate** — rough fraction of H100 SXM5 peak (989 TFLOPS BF16).
   Low values are expected on CPU and on T4 (which is not the target hardware).
6. **Correctness** — token conservation asserted for the sweep test
   (`violations=0/100`).

### Training loop profiling with `--profile`

The `--profile` flag in `train.py` records per-step telemetry and writes a
structured JSON to `benchmarks/` on exit.

```bash
# Single-process CPU smoke run with profiling
python train.py --config configs/smoke.yaml --smoke --profile
# → writes benchmarks/run_<unix_ts>_rank0.json

# Multi-GPU run (4 ranks, local node) with profiling
torchrun --standalone --nproc_per_node=4 \
  train.py --config configs/default.yaml --max-steps 100 --profile
# → rank 0 writes benchmarks/run_<unix_ts>_rank0.json
```

The profile JSON contains:

```json
{
  "config": { "hidden_dim": 4096, "num_experts": 64, "top_k": 2, "dp": 4, "ep": 2 },
  "steps": 100,
  "mfu_mean": 0.43,
  "mfu_p50": 0.44,
  "step_ms_mean": 412.3,
  "tokens_per_sec_mean": 79847,
  "per_step": [
    {
      "step": 5,
      "step_ms": 414.2,
      "mfu": 0.432,
      "tokens_per_sec": 79203,
      "loss": 4.821,
      "dispatch_ms": 0.38,
      "combine_ms": 0.41,
      "load_imbalance": 1.12
    }
  ]
}
```

### CLI shortcut

```bash
# Via the typer CLI (installed with pip install -e ".[dev]")
moe benchmark --cuda --json benchmarks/gpu_results.json

# Validate config before benchmarking
moe validate configs/default.yaml
```

---

## Metrics Reference

### MFU (Model FLOPs Utilization)

MFU is the fraction of the hardware's theoretical peak FLOP rate that the
training loop actually achieves, end-to-end including all communication and
memory overhead.

```
MFU = achieved_TFLOPS / (world_size × hardware_peak_TFLOPS)

achieved_TFLOPS = total_flops_per_step / step_time_s / 1e12
```

For a MoE model with sparse activation, FLOPs per step are split into:

```
flops_dense  = 2 × batch_tokens × P_dense
             = 2 × (B × S) × (attention + embedding + lm_head params)

flops_sparse = 2 × batch_tokens × (K / E) × P_expert
             = 2 × (B × S) × (top_k / num_experts) × expert_ffn_params
```

The `K/E` factor is essential. Because only `K` of `E` experts activate per
token, the effective FLOPs are a fraction `K/E` of what a dense model would
compute. Omitting this factor produces artificially inflated MFU numbers that
do not reflect actual hardware utilisation.

When activation recompute (`torch.utils.checkpoint`) is enabled, multiply the
total FLOPs by 1.5× (3× forward instead of 2× — two forwards and one backward
with each at 1× FLOPs).

```python
# Implemented in pkg/utils/mfu.py
from pkg.utils.mfu import compute_moe_flops, MFUAccountant

flops_per_token = compute_moe_flops(
    hidden_dim=4096, num_layers=32, ffn_dim=14336,
    num_experts=64,  top_k=2,      seq_length=4096,
    batch_tokens=1,  vocab_size=128256,
)

accountant = MFUAccountant(peak_tflops=989.0, mfu_target=0.55)
accountant.start_step()
# ... training step ...
result = accountant.end_step(tokens=batch_size * seq_length)
print(f"MFU: {result.mfu:.1%}  tokens/sec: {result.tokens_per_sec:,.0f}")
```

**Hardware peak TFLOPS reference** — set `telemetry.hardware_peak_tflops` in
your config to the correct value:

| GPU | BF16 TFLOPS | FP8 TFLOPS | Notes |
|-----|------------:|------------|-------|
| H100 SXM5 | 989 | 1,979 | Dense tensor-core peak |
| H100 PCIe | 756 | 1,513 | |
| H100 NVL  | 835 | 1,671 | |
| A100 SXM4 80GB | 312 | — | No FP8 |
| A100 PCIe 80GB | 312 | — | |
| RTX 4090  | 165 | 330 | |
| **T4**    | **65** | — | Used for v0.3.2 validation |

The T4 MFU numbers in the validation run are very low because T4 is a
mid-tier inference card, not a training accelerator. They are not comparable
to H100 or A100 numbers. T4 was used only to confirm the Triton kernel
compiles and runs correctly.

---

### Routing quality metrics

**`expert_load_imbalance = max(dispatch_cnt) / mean(dispatch_cnt)`**

Measures how unevenly tokens are distributed across experts. A value of 1.0
means perfect balance; every expert receives exactly `N×K / E` tokens. Values
above 1.0 indicate that some experts are processing more tokens than average.

Interpretation at `N=2048, E=64, K=2` (64 tokens per expert when balanced):

| Value | Meaning | Action |
|-------|---------|--------|
| 1.0–1.2 | Normal random variation | None needed |
| 1.2–1.5 | Mild imbalance | Monitor; add z-loss if sustained |
| 1.5–2.0 | Significant imbalance | Add `z_loss_weight ≥ 1e-3` to loss |
| > 2.0 | Routing collapse | Strong z-loss; check gate weight init |

In the v0.3.2 T4 validation run at H=4096, E=64, K=2, N=2048:
`expert_load_imbalance ≈ 1.08` — within the normal random variation range.

**`router_z_loss = mean(log(Σ_e exp(logit_e))²)`**

The auxiliary regularisation loss from Switch Transformer. It penalises large
gate logit magnitudes, which correlates with concentrated (imbalanced) routing.
Small logits → nearly-uniform softmax → more balanced expert assignment.

This is emitted as a telemetry signal every step, not automatically added to
your training loss. To use it:

```python
# In your training loop:
z_loss_weight = 1e-3  # typical starting value; tune on your model
loss = ce_loss + z_loss_weight * step_record.routing["router_z_loss"]
```

Typical z-loss values with random initialisation: `2.5–4.0`. After training
stabilises with z-loss penalty: `0.5–1.5`.

---

### Collective latency and overlap

**`all_to_all_dispatch_ms`** and **`all_to_all_combine_ms`**

Measured with real CUDA events on the dedicated EP CUDA stream
(`_CommStream`, a high-priority singleton per device). The start event is
recorded immediately before `dist.all_to_all_single` and the stop event
immediately after. Values reflect only the collective duration, not any
compute that overlaps with it.

On a real multi-GPU node at `ep_size=8` with NVLink:
- Typical dispatch: 0.3–0.8 ms (highly configuration-dependent)
- Typical combine: 0.3–0.8 ms
- At `ep_size=1`: always 0.0 ms (no collective issued)

**`expert_compute_ms`** (v0.3)

Wall-clock time of all local expert FFN compute for one step, measured with
`time.perf_counter()` after the dispatch all-to-all completes and before the
combine begins. At `ep_size=1`, this is the dominant cost in the MoE layer.

**`comm_compute_overlap_ratio = dispatch_ms / expert_compute_ms`** (v0.3)

The fraction of expert compute time that is covered by the dispatch all-to-all
running concurrently on the dedicated CUDA stream. A value of 0.5 means the
dispatch collective is hidden inside 50% of the expert compute time.

```
overlap_ratio interpretation:

  0.0       → no overlap (ep_size=1, or very fast compute + slow comm)
  0.3–0.6   → good overlap (target at EP=8 with NVLink)
  > 1.0     → dispatch longer than compute (communication-bound)
```

When `overlap_ratio > 1.0`, the training run is communication-bound. Possible
causes: NVLink not available (using PCIe), `ep_size` too large for the batch
size, or NCCL algorithm selection (`NCCL_ALGO=RING` vs `TREE`).

**How the overlap is implemented:**

```python
# expert_parallel.py — _CommStream sends dispatch on a dedicated stream
stream = _CommStream.get(device)          # high-priority CUDA stream
with torch.cuda.stream(stream):
    dist.all_to_all_single(received, tokens_sorted, ...)
    dispatch_event.record(stream)

# moe_layer.py — expert FFN runs on default stream concurrently
t0 = time.perf_counter()
expert_out = expert_ffn(received)         # default CUDA stream
expert_compute_ms = (time.perf_counter() - t0) * 1000

# Combine waits for dispatch_event before sending
stream.wait_event(dispatch_event)
dist.all_to_all_single(combined, expert_out, ...)
```

The two streams run in parallel. The GPU scheduler executes both until
either the dispatch collective needs the `received` buffer (already filled)
or the expert FFN needs to read `received` (also already filled — dispatch
completed before expert FFN started in the steady state).

**Target values on H100 SXM5, EP=8, NVLink:**

| Metric | Target |
|--------|--------|
| dispatch_ms | < 0.6 ms |
| combine_ms | < 0.6 ms |
| overlap_ratio | 0.3–0.6 |
| expert_compute_ms | 1.0–3.0 ms (depends on E, F, batch) |

If `dispatch_ms > 2 ms` at EP=8, investigate:
- NVLink vs PCIe topology (`nvidia-smi topo -m`)
- NCCL algorithm (`export NCCL_ALGO=RING` may help on ring topologies)
- Packet size alignment (tensor size should be a multiple of 128 bytes)
- NCCL debug logs (`export NCCL_DEBUG=INFO`)

---

### Throughput

**`tokens_per_sec = batch_tokens / step_time_s`**

where `batch_tokens = micro_batch_size × sequence_length × gradient_accumulation_steps`.

For reference (published or publicly estimated numbers):

| System | GPUs | tokens/sec | Notes |
|--------|-----:|------------|-------|
| GPT-3 (175B dense) | 1024 × A100 | ~150K | OpenAI 2020 estimate |
| PaLM (540B dense)  | 6144 × TPUv4 | ~280K | Google 2022 |
| Mixtral 8×7B (MoE) | 32 × H100 | ~200K | Estimated |
| moe-engine target  | 64 × H100 | ≥ 300K | v0.4 goal |

---

## How to Benchmark Correctly

### 1. Always warm up before measuring

The first 5–15 training steps include:
- Triton kernel JIT compilation (`triton.jit` caches after first call)
- CUDA graph capture (if enabled)
- Optimizer state allocation (first `optimizer.step()`)
- NCCL algorithm selection (first collective call)

All of these inflate step time significantly. `MFUAccountant` uses a sliding
window (default 50 steps) so early noisy steps are naturally de-weighted. For
one-shot benchmarks, discard at least 5 iterations.

### 2. Use stable, large batch sizes

MFU scales with batch size up to the point where the GPU is compute-bound.
At small batch sizes (B < 4 for a large model), the GPU is memory-bandwidth-bound
and communication overhead dominates. Compare MFU numbers only at the same
`tokens_per_step` across configurations.

Practical rule: at `H=4096, F=14336`, a batch of at least 512 tokens per GPU is
needed to hide memory-bandwidth overhead and achieve meaningful MFU.

### 3. Record configuration alongside every number

Every benchmark result must be paired with its full configuration. Without
this, numbers are not reproducible. Required fields:

```
world_size, dp, ep, tp, pp
hidden_dim, ffn_dim, num_experts, top_k, dtype
sequence_length, micro_batch_size, gradient_accumulation_steps
GPU model, GPU count, interconnect (NVLink / InfiniBand / PCIe)
torch version, CUDA version, Triton version
```

The `--profile` flag captures most of this automatically in the JSON output.
`moe info` prints the software versions.

### 4. Run for at least 100 steps after warmup

Step-to-step variance from OS scheduler jitter, NCCL algorithm variation, and
garbage collection is typically 5–15%. Averages over ≥ 100 steps are stable
to within 2%.

### 5. Compare relative, not absolute

CPU reference path MFU is always near 0% (no CUDA hardware). GPU MFU should
be compared between configurations on the same hardware, or against published
numbers on the same GPU model. T4 MFU numbers are not comparable to H100 MFU
numbers — the T4's 65 TFLOPS BF16 peak is 15× lower than the H100.

### 6. Token conservation is a correctness check, not a performance metric

`violations=0/100` confirms that the routing dispatch logic is correct: every
input token is dispatched exactly `K` times and no tokens are lost or
duplicated. Run this check any time you modify `pkg/kernels/moe_router.py` or
`pkg/distributed/expert_parallel.py`.

---

## Interpreting Telemetry Output

### Step record structure (v0.3 full envelope)

```jsonc
{
  "step": 42,
  "loss": 4.217,
  "mfu": 0.431,
  "tokens_per_sec": 79203,
  "wall_clock_ms": 412.3,

  "kernel": {
    "sram_bytes_per_block": 49152,    // 48 KiB Triton tile
    "achieved_bw_gbps": 312.4,        // HBM bandwidth used by router kernel
    "tokens_per_expert_mean": 64.0,   // N*K/E
    "tokens_per_expert_std": 8.3,     // std dev across experts this step
    "used_triton": true               // false = fp64 CPU fallback
  },

  "collective": {
    "all_to_all_dispatch_ms": 0.38,
    "all_to_all_combine_ms":  0.41,
    "expert_compute_ms":      1.24,
    "comm_compute_overlap_ratio": 0.31  // dispatch_ms / expert_compute_ms
  },

  "routing": {
    "expert_load_imbalance": 1.08,
    "router_z_loss": 2.87
  },

  "memory": {
    "peak_allocated_gb": 42.1,
    "peak_reserved_gb": 44.0
  },

  "infra": {
    "async_ckpt_commit_ms": 0.0,
    "active_nodes": 8,
    "ep_world_size": 8,
    "lr": 0.0003
  },

  "rank": 0,
  "ts": 1748901234.56
}
```

### What "good" looks like on a single H100 node (8 GPUs, NVLink)

Target configuration: `H=4096, E=64, K=2, F=14336, B=8, S=4096, EP=8, TP=1, DP=1`.

| Metric | Target | Below target → |
|--------|--------|----------------|
| MFU | 40–55% | Batch too small / comm-bound |
| tokens_per_sec | 60K–100K | Batch or sequence length too short |
| dispatch_ms | < 0.8 ms | PCIe instead of NVLink |
| combine_ms | < 0.8 ms | PCIe instead of NVLink |
| overlap_ratio | 0.3–0.6 | ep_size too small, or batch too large |
| expert_load_imbalance | < 1.3 | Routing collapse; add z-loss |
| peak_allocated_gb | < 70 GB | OOM → reduce B or S |
| async_ckpt_commit_ms | < 100 ms | NVMe saturation; reduce async_workers |

### Common low-MFU diagnoses

| Symptom | Likely root cause | Fix |
|---------|------------------|-----|
| MFU < 20% with GPU | Batch too small (< 256 tokens/GPU) | Increase B or S |
| MFU < 20% with GPU | Communication-bound (dispatch_ms >> compute) | Reduce EP, check NVLink |
| dispatch_ms > 2 ms at EP=8 | PCIe instead of NVLink | Use NVLink nodes |
| overlap_ratio > 1.0 | Dispatch longer than compute | Reduce EP or increase batch |
| expert_load_imbalance > 1.5 | Routing collapse | `z_loss_weight = 1e-3` |
| step_ms highly variable | GC pressure | `optimizer.zero_grad(set_to_none=True)` |
| peak_allocated_gb ≫ expected | Grad buffers not freed | `zero_grad(set_to_none=True)` |
| `used_triton: false` | Triton not installed or CUDA unavailable | `pip install triton` |
| NaN in loss after step 1 | Router logit overflow | Check gate weight init |

---

## Four-Tier Benchmark Strategy

Matching the four-tier test model from `docs/testing.md`:

| Tier | Hardware | Benchmark command | What it measures |
|------|----------|-------------------|-----------------|
| **0 CPU** | Any machine | `python benchmarks/run_benchmark.py` | Router correctness + CPU throughput baseline |
| **1 GPU** | T4 / RTX 4090 | `python benchmarks/run_benchmark.py --cuda` | Triton kernel correctness + GPU throughput |
| **2 Multi-GPU** | 4–8 × H100 | `torchrun ... train.py --profile` | EP overlap, TP bandwidth, PP scheduling |
| **3 Cluster** | 64+ × H100 | Full training run with telemetry | MFU at scale, chaos resilience, ckpt throughput |

**Most development happens at Tier 0 and Tier 1.** The token conservation
sweep, numerical correctness tests, and CPU throughput baselines are all
deterministic and fast. Real GPU throughput numbers require at least Tier 1.

---

## T4 GPU Validation Results (June 2026)

The June 2026 T4 validation confirmed the Triton kernel is functionally correct
at production-like shapes. These numbers are in `moe-engine/benchmarks/BENCHMARKS.md`.

**Headline: 80.1× GPU speedup over CPU reference at N=4096, H=2048, E=64, K=4.**

| Config | CPU (M tok/s) | T4 (M tok/s) | Speedup |
|--------|:---:|:---:|:---:|
| N=512, H=256, E=16, K=2 | 0.747 | 2.141 | 2.9× |
| N=1024, H=512, E=32, K=2 | 0.421 | 3.644 | 8.7× |
| N=2048, H=1024, E=64, K=2 | 0.236 | 4.832 | 20.4× |
| N=4096, H=2048, E=64, K=4 | 0.056 | 4.454 | **80.1×** |

These numbers are from a single T4 card (65 TFLOPS BF16 peak). On H100 SXM5
(989 TFLOPS BF16 peak), with the same kernel, the raw throughput should scale
approximately linearly with TFLOPS, giving an estimated T4→H100 multiplier
of ~15×.

Reproduce with the T4 validation notebook:
```
moe-engine/notebooks/moe_engine_v032_T4_validation.ipynb
```

---

## Adding Real Benchmark Results

If you run a benchmark on real hardware, contribute the results to
`moe-engine/benchmarks/BENCHMARKS.md` with:

1. **Hardware:** GPU model, count, interconnect (NVLink / InfiniBand / PCIe)
2. **Software:** PyTorch version, CUDA version, Triton version, Python version
3. **Config:** exact YAML file or diff from `configs/default.yaml`
4. **Command:** exact command line used to launch
5. **Results:** MFU mean/p50, tokens_per_sec, dispatch_ms, load_imbalance
6. **Caveats:** numa binding, CPU affinity, any non-default environment settings

All GPU numbers in `BENCHMARKS.md` currently come from the T4 validation run.
Multi-GPU H100 / A100 numbers will be added in v0.4 when sustained cluster
access is available.
