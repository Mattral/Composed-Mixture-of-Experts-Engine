<div align="center">

# `moe-engine` · Composed Mixture-of-Experts Engine

**A research-grade runtime for Mixture-of-Experts training**

[![Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](https://github.com/Mattral/Composed-Mixture-of-Experts-Engine/blob/main/LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5%2B-ee4c2c.svg)](https://pytorch.org/)
[![Triton](https://img.shields.io/badge/Triton-3.x-9333ea.svg)](https://triton-lang.org/)
[![Tests](https://img.shields.io/badge/tests-416%20passed%20%2B%20243%20differential-brightgreen)](https://github.com/Mattral/Composed-Mixture-of-Experts-Engine/blob/main/moe-engine/tests/)
[![GPU Validated](https://img.shields.io/badge/GPU-single--GPU%20router%20%28T4%2C%20A100%29-blue)](docs/adr/ADR-005-gpu-architecture-portability.md)
[![Chaos B](https://img.shields.io/badge/Chaos%20B-recorded%2010%2F10%20%28manual%20job%29-yellow)](https://github.com/Mattral/Composed-Mixture-of-Experts-Engine/blob/main/RESULTS.md#fault-tolerance--chaos-test-results)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/Mattral/Composed-Mixture-of-Experts-Engine/pulls)
[![Version](https://img.shields.io/badge/version-v0.3.3-orange)](https://github.com/Mattral/Composed-Mixture-of-Experts-Engine/releases)

</div>

> **Associated Publication**  
> This repository is accompanied by the preprint:  
> **moe-engine: A Fault-Tolerant Runtime for Hyperscale Mixture-of-Experts Training**  
> Min Htet Myet, June 2026  
> [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20688837.svg)](https://doi.org/10.5281/zenodo.20688837)  
> [Read on Zenodo](https://zenodo.org/records/20688837) · [PDF](https://zenodo.org/records/20688837/files/moe-engine-preprint-v2.pdf)
>
> Superseded by this README and the Documentation entry in `CHANGELOG.md`: the preprint's fault-tolerance wording, its superlinear speedup wording, its Scenario A pass-rate wording, and its test-count wording.

> **v0.3.2 (June 2026)**: Fixed Triton kernel compile-time crash that prevented all real-GPU runs since v0.2 (undetected by CPU-only CI). Added real dense-baseline measurements. Fixed `train.py` config crash. See `benchmarks/BENCHMARKS.md` for full patch notes.

---

## What this is

`moe-engine` is a **research-grade infrastructure runtime** for training large Mixture-of-Experts (MoE) language models. The design target is large clusters. Validation is currently single-GPU (T4, A100) and 2-rank CPU. Multi-node and 8+ GPU runs remain open.

**This is not a model definition.** It is the systems layer (Triton kernels + 4D parallelism + checkpoint and recovery code + observability) that a real MoE model runs on.

Documented router speedup at N=4096, H=2048, E=64, K=4 is **80.1× (T4) / 58.7× (A100) versus the CPU reference path; no GPU baseline comparison yet**. The CPU baseline differs by host (0.056 M tok/s in the T4 table, 0.175 M tok/s in the A100 table).

Core capabilities:
- Fused **Triton router kernel** (single HBM pass, analytic backward pass). Push CI runs the CPU path. The Triton GPU job is `workflow_dispatch` only and was skipped on the latest push.
- **4D parallelism** — Data (FSDP2) × Expert (all-to-all) × Tensor (Column/RowParallel + SP fusion) × Pipeline (tagged 1F1B)
- Strict mathematical invariants (token conservation, NaN guards, index validity)
- Scenario A (node kill) recovers in ~17/20 runs (failures attributed to a Gloo race in containers). Scenario B (storage stall) is recorded at 10/10 and runs via `workflow_dispatch` only, not on push CI.
- **Async two-tier checkpointing** (pinned-host → NVMe O_DIRECT → S3/MinIO with atomic rename), covered by the CPU test suite
- MoE-aware telemetry (JSONL, plus optional TensorBoard, Prometheus, and WandB sinks)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Training Loop                            │
│  train.py  ←  load_config  ←  configs/{default,smoke}.yaml      │
└───────────────────┬─────────────────────────────────────────────┘
                    │
     ┌──────────────▼───────────────┐
     │   DistributedMoELayer        │  pkg/distributed/parallel_mesh.py (shim)
     │                              │  → moe_layer.py + expert_parallel.py
     │  ┌──────────┐ ┌──────────┐   │
     │  │MoERouter │ │ Experts  │   │  pkg/kernels/moe_router.py
     │  │(Triton)  │ │(SwiGLU)  │   │
     │  └────┬─────┘ └────▲─────┘   │
     │       │  EP a2a     │        │
     │  ┌────▼──────────────────┐   │
     │  │  all_to_all dispatch  │   │  dedicated CUDA stream
     │  │  all_to_all combine   │   │  compute-comm overlap
     │  └───────────────────────┘   │
     └──────────────┬───────────────┘
                    │
     ┌──────────────▼───────────────┐
     │   ElasticTrainerHarness      │  pkg/elastic/fault_monitor.py
     │                              │
     │  AsyncCheckpointer           │  background I/O threads
     │    NVMe tier  (fast)         │  pinned host → O_DIRECT write
     │    S3/MinIO   (durable)      │  atomic rename + remote mirror
     │                              │
     │  ClusterStateMachine         │  heartbeat → evict → reshard
     │    evict dead ranks          │  → reload → resume (no restart)
     │    reshard expert owners     │
     └──────────────┬───────────────┘
                    │
     ┌──────────────▼──────────────┐
     │   Telemetry                 │  pkg/telemetry/logger.py
     │   JSONL + TensorBoard       │  real CUDA event timing
     │   Prometheus /metrics       │  routing + overlap metrics
     │   WandB (optional)          │  WANDB_API_KEY-gated
     └─────────────────────────────┘
```

**4D Parallelism mesh:** `(dp × tp × pp × ep)`
- **DP** — FSDP2 per-parameter sharding via DTensor along the data axis
- **EP** — Expert Parallelism: each rank owns `E / ep_size` experts; non-blocking `all_to_all_single` dispatch + combine on a dedicated high-priority CUDA stream with compute overlap
- **TP** — Tensor Parallelism on expert FFNs (`ColumnParallelLinear` / `RowParallelLinear`) + Sequence Parallelism with `next_weight` fusion (halves collectives)
- **PP** — Pipeline Parallelism with `PipelineStage` + 1F1B schedule and activation tagging for restart/stall robustness

See `docs/ARCHITECTURE.md` for the detailed internal data-flow diagram (Triton router internals, dedicated-stream mechanics, token lifecycle, and per-layer forward/backward pass).

---

## T4 GPU Validation (June 2026) — Key Results

All numbers below are **real measurements** from `benchmarks/gpu_results_t4.json`.  
Reproduce everything: see [`notebooks/moe_engine_v032_T4_validation.ipynb`](notebooks/moe_engine_v032_T4_validation.ipynb) (13 sections, open in Colab with T4).

### Router throughput — CPU vs T4 GPU (Triton kernel)

| Config                  | CPU (M tok/s) | T4 GPU (M tok/s) | Speedup   |
|-------------------------|---------------|------------------|-----------|
| N=512, H=256, E=16, K=2 | 0.747         | 2.141            | **2.9×**  |
| N=1024, H=512, E=32, K=2| 0.421         | 3.644            | **8.7×**  |
| N=2048, H=1024, E=64, K=2 | 0.236       | 4.832            | **20.4×** |
| N=4096, H=2048, E=64, K=4 | 0.056       | 4.454            | **80.1×** |

Speedup in this table is versus the CPU reference path on the T4 host. There is no GPU baseline comparison yet.

### GPU router throughput chart (v0.3.2, T4, real measurements)

![Router throughput GPU v0.3.2](benchmarks/charts/router_throughput_gpu_v0_3_2.png)

*Forward-only (blue) and forward+backward (orange) throughput on T4 GPU (Triton kernel) vs CPU reference path (green dashed). Log scale.  
Source: `benchmarks/gpu_results_t4.json`, June 2026 T4 validation run.*

> **Note:** If the chart image is not yet present in `moe-engine/benchmarks/charts/`, run Section 9 of the validation notebook on a T4 and copy `router_throughput_gpu_v0_3_2.png` into that folder.

### MoE layer throughput (v0.3.1, CPU, real measurements)

![MoE layer throughput v0.3.1](benchmarks/charts/moe_layer_throughput_v0.3.1.png)

*Full `DistributedMoELayer` forward (orange) vs single dense SwiGLU FFN baseline (blue) on CPU.  
Source: `benchmarks/cpu_results_colab.json`.*



---


### A100 GPU Validation — July 2026

Second real-GPU validation, on an NVIDIA A100-SXM4-80GB via Lightning AI
Studio. This run surfaced and fixed a Turing-vs-Ampere `tl.dot` tensor-core
minimum tile size defect invisible on T4, plus a `used_triton` telemetry
correctness bug — both documented in
[`docs/adr/ADR-005-gpu-architecture-portability.md`](../docs/adr/ADR-005-gpu-architecture-portability.md).

**Hardware:** NVIDIA A100-SXM4-80GB, 81920 MiB, driver 580.159.03
**Software:** torch 2.8.0+cu128, CUDA 12.8

![Router throughput A100](benchmarks/router_throughput_A100.png)

*Forward-only (blue) and forward+backward (orange) throughput on A100
(Triton kernel) vs CPU reference path (green dashed). Log scale.
Source: `benchmarks/gpu_results_a100.json`, July 2026 validation run.*

#### Router throughput — CPU vs A100

Speedup here is versus the CPU reference path on the A100 host (0.175 M tok/s at N=4096 in this table, versus 0.056 M tok/s in the T4 table). There is no GPU baseline comparison yet.

| Config | CPU (M tok/s) | A100 (M tok/s) | Speedup |
|--------|:-------------:|:--------------:|:-------:|
| N=512, H=256, E=16, K=2 | 1.168 | 2.031 | 1.7× |
| N=1024, H=512, E=32, K=2 | 0.826 | 4.006 | 4.8× |
| N=2048, H=1024, E=64, K=2 | 0.776 | 5.934 | 7.6× |
| N=4096, H=2048, E=64, K=4 | 0.175 | 10.271 | **58.7×** |

#### Router forward+backward — CPU vs A100

| Config | CPU (M tok/s) | A100 (M tok/s) | Speedup |
|--------|:-------------:|:--------------:|:-------:|
| N=512, H=256, E=16, K=2 | 0.435 | 0.572 | 1.3× |
| N=1024, H=512, E=32, K=2 | 0.456 | 1.097 | 2.4× |
| N=2048, H=1024, E=64, K=2 | 0.312 | 2.034 | 6.5× |
| N=4096, H=2048, E=64, K=4 | 0.101 | 3.721 | 37.0× |

These A100 cells, including 58.7× and 37.0×, are the documented values. They were not reproduced from `benchmarks/gpu_results_a100.json` and are left unchanged.

Reproduce with
[`moe-engine/notebooks/moe_engine_A100_validation.ipynb`](notebooks/moe_engine_A100_validation.ipynb).

**GPU architecture support matrix** (full detail in ADR-005):

| Architecture | Status | Validated |
|---|---|---|
| NVIDIA T4 (Turing, sm_75) | ✅ | June 2026 |
| NVIDIA A100-SXM4-80GB (Ampere, sm_80) | ✅ | July 2026 |
| NVIDIA H100 (Hopper, sm_90) | ❌ Not yet tested | — |


---

### Chaos resilience

| Scenario              | Description                        | Runs | Pass Rate    |
|-----------------------|------------------------------------|------|--------------|
| **Scenario B**        | Storage stall (10s I/O delay)      | 10   | **10/10**    |
| **Scenario A**        | Node kill + recovery (SIGKILL)     | 20   | **~17/20**   |

Scenario B's 10/10 result is from a manual run. The Chaos job is `workflow_dispatch` only and does not run on push CI.
Scenario A recovers in ~17/20 runs (failures attributed to a Gloo race in containers). A change from Gloo to NCCL in the chaos harness is planned for v0.4.

See **[`RESULTS.md`](RESULTS.md)** for every real number with exact reproduction commands and full telemetry samples.

---

## What is Actually Built

| Component                              | Status          | Detail |
|----------------------------------------|-----------------|--------|
| **Triton router — forward**            | ✅ CPU path in push CI | Fused matmul+softmax+topK+renorm; single HBM pass; SRAM 64×64. 80.1× (T4) versus the CPU reference path at N=4096; no GPU baseline comparison yet. The Triton GPU job is manual and was skipped on the latest push. |
| **Triton router — backward**           | ✅ CPU path in push CI | Analytic Jacobian; `atol=rtol=1e-5` vs fp64 ref; 30+ configs tested. The GPU job did not run on the latest push. |
| **Token conservation invariant**       | ✅ CPU benchmark in push CI | `sum(dispatch_cnt) == N×K` every forward. The latest push benchmark reported `violations=0/100`. The GPU sweep is not part of push CI. |
| **Expert load imbalance metric**       | ✅ Verified     | `max/mean` load per step; emitted in telemetry |
| **Router z-loss**                      | ✅ Verified     | Auxiliary regulariser emitted per step |
| **EP all-to-all (dispatch + combine)** | ✅ CPU differential job | `all_to_all_single` on Gloo in the differential job (243 passed). Dedicated CUDA-stream overlap is not part of push CI. |
| **Compute-comm overlap**               | ✅ Verified     | Expert FFN (default stream) overlaps a2a (dedicated stream) |
| **Overlap ratio telemetry**            | ✅ v0.3         | `dispatch_ms / expert_compute_ms` recorded every step |
| **DP via FSDP2**                       | ✅ Verified     | `fully_shard` along DP axis; expert weights excluded from sharding |
| **Tensor Parallelism**                 | ✅ v0.2         | `ColumnParallelLinear` + `RowParallelLinear`; both `w_gate`/`w_up` ColumnParallel; 2-rank verified |
| **Sequence Parallelism + fusion**      | ✅ v0.3         | `scatter/gather`; `next_weight` param halves SP collectives; 2-rank verified |
| **Pipeline Parallelism (multi-proc)**  | ✅ v0.3         | `run_1f1b_distributed` with activation tagging; real `dist.send/recv`; 2-rank verified |
| **MFU accounting**                     | ✅ v0.2         | MoE-sparse formula `(K/E)×P_expert`; streaming `MFUAccountant` |
| **Pydantic `MoEConfig`**               | ✅ v0.3.2       | Validated hierarchy; env-var overrides; field-level errors; 38 tests, included in the Tier-0 count |
| **Async two-tier checkpointing**       | ✅ CI-verified (CPU suite) | NVMe (`O_DIRECT`, atomic rename) → S3/MinIO mirror; background threads |
| **TorchElastic recovery**              | ✅ CPU unit tests in push CI | Evict → reshard (round-robin) → reload. The SIGKILL chaos scenario is not part of push CI. |
| **Structured JSONL + multi-sink telemetry** | ✅ CPU tests | Thread-safe JSONL. Optional TensorBoard, Prometheus, and WandB sinks. A zero-cost claim for disabled sinks is not yet validated. |
| **WandB integration**                  | ✅ v0.3         | `WANDB_API_KEY` env; `--wandb-project`; `log_config()` records full YAML |
| **Docker + docker-compose**            | ✅ v0.2         | Multi-stage image; 1/4/8-GPU targets + monitoring stack. The Docker CI job is manual and was skipped on the latest push. |
| **Kubernetes manifests**               | ✅ v0.2         | Single-node Job + multi-node Indexed Job with etcd rendezvous; PVC for checkpoints |
| **Benchmark suite**                    | ✅ v0.2         | CPU + GPU sweeps; JSON/CSV output; chart generation |
| **Typer CLI**                          | ✅ v0.3.2       | `moe train / benchmark / validate / info` |
| **Chaos Scenario B (storage stall)**   | Manual          | Recorded 10/10 in RESULTS.md. The Chaos job is `workflow_dispatch` only and was skipped on the latest push. |
| **Chaos Scenario A (node kill)**       | ⚠️ ~17/20       | Recovers in ~17/20 runs (failures attributed to a Gloo race in containers). |
| **Nsight/CUPTI profiling**             | ❌ Planned v0.4 | Requires GPU hardware + sustained runs |
| **Real sustained multi-node data**     | ❌ Planned v0.4 | Requires large cluster access |

---

## Getting Started

### Prerequisites
- Python ≥ 3.10
- PyTorch ≥ 2.5
- Triton ≥ 3.0 (optional — CPU fallback always works)
- CUDA ≥ 12.4 (optional)

### 1. Clone & Install
```bash
git clone https://github.com/Mattral/Composed-Mixture-of-Experts-Engine.git
cd Composed-Mixture-of-Experts-Engine/moe-engine
pip install -e ".[dev]"
```

### 2. Validate Configs
```bash
make validate-config
# or
python scripts/validate_config.py configs/
```

### 3. CPU Smoke Test (no GPU, ~5–10 s)
```bash
make smoke
# or
python train.py --config configs/smoke.yaml --smoke
```
Expected: 2 steps, JSONL telemetry at `/tmp/moe-engine/logs/step.jsonl`.

### 4. CPU tests
```bash
make test-cpu
```
`make test-cpu` is wider than the push CI Tier-0 job: it does not ignore `test_differential.py`, `test_comm_ops.py`, `test_grad_sync.py`, or `test_train_step.py`.

The Tier-0 job reports `416 passed, 3 skipped, 16 deselected, 1 xfailed`. That count includes the 38 config tests. The differential job reports `243 passed`.

### 5. Using the Typer CLI
```bash
moe train --config configs/smoke.yaml --smoke
moe benchmark --cuda --json benchmarks/gpu_results.json
moe validate configs/
moe info
```

### 6. Docker Examples
```bash
docker compose -f deploy/docker/docker-compose.yml run --rm smoke
docker compose -f deploy/docker/docker-compose.yml run --rm train-4gpu
```

### 7. Kubernetes (single-node)
```bash
kubectl apply -f deploy/k8s/namespace.yaml
kubectl apply -f deploy/k8s/pvc.yaml
kubectl apply -f deploy/k8s/training-job.yaml
kubectl logs -n moe-engine -l job-name=moe-training -f
```

Multi-node IndexedJob + etcd manifests are also in `deploy/k8s/`.

---

## Configuration (Pydantic v2)

```python
from pkg.utils.config import MoEConfig, ConfigValidationError

cfg = MoEConfig.from_yaml("configs/smoke.yaml")

print(cfg.model.hidden_dim)      # 32
print(cfg.model.num_experts)     # 4
print(cfg.parallelism.data_parallel)

# Environment variable overrides supported
# MOE_TRAINING__LEARNING_RATE=1e-4 python train.py --config ...

try:
    bad = MoEConfig.from_dict({"model": {"top_k": 99, "num_experts": 4}})
except ConfigValidationError as e:
    print(e)   # Clear field-level error message
```

See `configs/default.yaml` for a production-scale shape (H=4096, E=64, dp=8, ep=8). That configuration has not been run. The schema is in `pkg/utils/config.py`.

---

## Telemetry Envelope (every training step)

```json
{
  "step": 100,
  "loss": 3.42,
  "mfu": 0.48,
  "tokens_per_sec": 42800,
  "wall_clock_ms": 78.4,
  "kernel": { "used_triton": true, "sram_bytes_per_block": 49152, ... },
  "collective": {
    "all_to_all_dispatch_ms": 0.72,
    "all_to_all_combine_ms": 0.68,
    "expert_compute_ms": 1.84,
    "comm_compute_overlap_ratio": 0.39
  },
  "routing": {
    "expert_load_imbalance": 1.08,
    "router_z_loss": 2.34
  },
  "memory": { "peak_allocated_gb": 62.4, ... },
  "infra": { "async_ckpt_commit_ms": 12.3, "active_nodes": 64, ... },
  "rank": 0,
  "ts": 1748901234.56
}
```

Sinks:
- Always: structured JSONL
- Optional: TensorBoard, Prometheus `/metrics` (10 gauges), WandB (only when `WANDB_API_KEY` is set). A zero-cost claim for disabled sinks is not yet validated.

---

## Mathematical Invariants (enforced in every forward pass)

1. **Token conservation**: `sum(dispatch_cnt) == N × K`
2. **No NaN / invalid indices**: router indices ∈ `[0, E)`
3. **Combine shape correctness**: output of `all_to_all_combine` exactly `[N, H]`
4. **No NaN activations** after combine
5. **Router weight normalisation**: `w.sum(dim=-1) == 1.0` (`atol=1e-5`)

All five are validated in `test_kernels*.py` and `test_distributed_invariants.py`.

---

## Engineering Lessons (selected)

- **Activation tagging in multi-process PP** (`[stage_id, mb_index]` header before every activation tensor) is how 1F1B carries a stage id and a micro-batch index on each activation.
- **Pinned-host → NVMe (O_DIRECT, 256 MiB chunks, atomic rename) → S3** is the checkpoint path exercised by the CPU tests.

### Not yet validated

- ~2.7× lower memory traffic for the single-HBM-pass router versus a three-pass pipeline.
- ~40% reduction in collective cost at EP=8.
- Zero-cost telemetry when WandB and Prometheus are disabled.

More rationale lives in `docs/DESIGN.md` and the original engineering lessons section.

---

## Test Suite

```bash
# CI Tier-0 CPU job — 416 passed, 3 skipped, 16 deselected, 1 xfailed
# (includes the 38 config tests)
pytest tests/ \
  --ignore=tests/test_chaos.py \
  --ignore=tests/test_smoke_e2e.py \
  --ignore=tests/test_differential.py \
  --ignore=tests/test_comm_ops.py \
  --ignore=tests/test_grad_sync.py \
  --ignore=tests/test_train_step.py \
  -m cpu \
  -k "not (2rank or multiprocess or distributed_invariants)"

# Differential job — 243 passed (not included above)
GLOO_SOCKET_IFNAME=lo pytest tests/test_differential.py tests/test_comm_ops.py \
  tests/test_grad_sync.py tests/test_train_step.py

# GPU kernel tests (manual CI job; requires CUDA + Triton)
make test-gpu

# Chaos tests (manual CI job; requires torchrun)
make chaos-b          # Scenario B — recorded 10/10; not run on push CI
make chaos-a          # Scenario A — recorded ~17/20
```

`make test-cpu` runs the Tier-0 selection and the four differential files together.

Key test files:
- `test_kernels*.py` — router numerics + invariants (30+ configs vs fp64)
- `test_*parallel*.py` — 2-rank `mp.spawn` correctness for TP, SP fusion, PP with tagging
- `test_elastic*.py` + `test_chaos.py` — async checkpointing + fault injection
- `test_config.py` — 38 tests for the full `MoEConfig` system, included in Tier-0
- `test_differential.py`, `test_comm_ops.py`, `test_grad_sync.py`, `test_train_step.py` — differential job, 243 passed
- `test_smoke_e2e.py` — full `train.py` loop regression test; ignored by the Tier-0 job

---

## Repository Layout

```
moe-engine/                          # installable package root
├── train.py                         # TorchElastic entrypoint (config → topology → elastic harness → training loop)
├── Makefile                         # test-cpu, test-gpu, smoke, benchmark, validate-config, lint, clean
├── pyproject.toml                   # build metadata, pytest markers (cpu/gpu/chaos), dependencies
│
├── configs/
│   ├── smoke.yaml                   # Tiny CPU-friendly config (H=32, E=4)
│   └── default.yaml                 # Production-scale shape (H=4096, E=64, dp=8, ep=8); not run
│
├── pkg/
│   ├── kernels/moe_router.py        # Triton MoERouter (fwd + analytic bwd) + CPU fallback
│   ├── distributed/
│   │   ├── parallel_mesh.py         # Backward-compat shim only (re-exports from the modules below)
│   │   ├── mesh.py                  # ParallelTopology, build_topology, device mesh + process groups
│   │   ├── moe_layer.py             # DistributedMoELayer + _SwiGLUExpert
│   │   ├── expert_parallel.py       # all_to_all_dispatch / combine on dedicated CUDA stream
│   │   ├── tensor_parallel.py       # Column/RowParallelLinear + SP scatter/gather + next_weight fusion
│   │   ├── pipeline_parallel.py     # PipelineStage + run_1f1b_distributed (with activation tagging)
│   │   └── data_parallel.py         # apply_fsdp2 (FSDP2 + DTensor, expert-excluded)
│   ├── elastic/fault_monitor.py     # ElasticTrainerHarness, AsyncCheckpointer, ClusterStateMachine
│   ├── telemetry/logger.py          # StructuredLogger, StepRecord, real CUDA events, Prometheus, WandB
│   ├── models/moe.py                # ToyMoEModel + build_model factory (for validation & smoke)
│   └── utils/
│       ├── config.py                # MoEConfig (Pydantic v2) + load_config + validation
│       └── mfu.py                   # MFUAccountant + compute_moe_flops (MoE-aware)
│
├── scripts/
│   ├── cli.py                       # Typer CLI: moe train / benchmark / validate / info
│   ├── validate_config.py           # Standalone coloured YAML validator
│   └── launch.sh                    # Multi-node torchrun launcher helper
│
├── benchmarks/
│   ├── run_benchmark.py             # CPU + GPU sweep runner (JSON/CSV + charts)
│   ├── BENCHMARKS.md                # Full methodology + all real numbers + patch notes
│   ├── charts/                      # Generated PNG/SVG throughput charts
│   └── *.json                       # Raw results (cpu_results_colab.json, gpu_results.json, ...)
│
├── notebooks/
│   └── moe_engine_v032_T4_validation.ipynb   # 13-section T4 reproduction notebook
│
├── deploy/
│   ├── docker/                      # Multi-stage Dockerfile + docker-compose (smoke/4gpu/8gpu + monitoring)
│   └── k8s/                         # namespace, pvc, single-node Job, multi-node IndexedJob + etcd
│
├── tests/                           # 28 test modules, 685 collected items
│   ├── test_kernels*.py
│   ├── test_*parallel*.py
│   ├── test_elastic*.py + test_chaos.py
│   ├── test_config.py               # 38 tests; included in the Tier-0 job
│   └── ...
│
│
docs/                                # Top-level documentation (outside moe-engine/)
├── ARCHITECTURE.md              # Component map, token lifecycle, detailed data-flow diagrams
├── DESIGN.md                    # System design rationale & engineering trade-offs
├── testing.md                   # Four-tier test strategy
├── benchmarks.md                # Metrics reference and benchmark guide
├── LIMITED_HARDWARE_GUIDE.md    # Developing without a GPU cluster
├── CONTRIBUTING.md              # Contribution workflow, PR checklist, code standards
└── adr/                         # Architecture Decision Records (ADR-001 through ADR-004)
│
└── README.md (package-level README)
```

Top-level repo files include this `README.md`, `RESULTS.md`, `roadmap.md`, `LICENSE`, `.github/workflows/`, etc.

---

## T4 Validation Notebook

[`moe-engine/notebooks/moe_engine_v032_T4_validation.ipynb`](moe-engine/notebooks/moe_engine_v032_T4_validation.ipynb)

Open in Colab with a T4 to reproduce:
- All router throughput numbers and the two charts shown above
- Full router-result JSON generation (committed as `benchmarks/gpu_results_t4.json`)
- Chaos Scenario A/B pass rates
- Production-scale Triton kernel sanity check (H=4096, E=64, K=2)

---

## v0.3.2 Changelog Highlights

**Architectural cleanup**
- `parallel_mesh.py` (1,165 lines) split into 6 focused modules (< 380 lines each)
- Backward-compat shim preserves all existing imports — zero breaking changes
- `pkg/models/moe.py` extracted; `build_model()` factory added

**Testing & validation**
- `test_config.py`: 38 tests for the full `MoEConfig` system, included in the Tier-0 count
- `@pytest.mark.cpu` / `@pytest.mark.gpu` markers on all relevant tests
- Push CI: **416 passed, 3 skipped, 16 deselected, 1 xfailed** in Tier-0, plus **243 passed** in the differential job
- Every illustrative number replaced with real T4 measurements

**Developer experience**
- `Makefile` targets: `test-cpu`, `test-gpu`, `smoke`, `benchmark`, `validate-config`, `lint`, `clean`
- `scripts/cli.py`: full Typer CLI (`moe train`, `moe benchmark`, `moe validate`, `moe info`)
- `notebooks/moe_engine_v032_T4_validation.ipynb`: complete 13-section validation notebook

See `benchmarks/BENCHMARKS.md` for the complete changelog.

---

## Roadmap (v0.4 priorities)

See [`roadmap.md`](roadmap.md) and [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the full plan. High-priority items:
- Replace Gloo with NCCL in chaos harness → fix Scenario A flakiness
- Real 8-GPU+ benchmark data and end-to-end MFU validation
- Nsight/CUPTI roofline integration
- Expert capacity overflow re-routing
- Non-divisible sequence length support in Sequence Parallelism

---

## Navigation & Further Reading

| Resource                        | Purpose                                              |
|---------------------------------|------------------------------------------------------|
| [`RESULTS.md`](RESULTS.md)      | Every numerical result + full telemetry samples      |
| [`roadmap.md`](roadmap.md)      | Honest status + detailed v0.4 plan                   |
| [`benchmarks/BENCHMARKS.md`](moe-engine/benchmarks/BENCHMARKS.md) | Methodology + patch notes                     |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | Deep design + detailed data-flow diagrams     |
| T4 validation notebook          | Full reproduction of GPU numbers & charts            |
| `deploy/`                       | Docker & Kubernetes manifests (single + multi-node)  |
| Issues / Discussions            | Questions, bug reports, feature requests             |

---

## Citation

If you use `moe-engine` in research, please cite the preprint:

```bibtex
@misc{myet2026moeengine,
  author = {Min Htet Myet},
  title  = {moe-engine: A Fault-Tolerant Runtime for Hyperscale Mixture-of-Experts Training},
  year   = {2026},
  url    = {https://github.com/Mattral/Composed-Mixture-of-Experts-Engine},
  note   = {v0.3.2 preprint. Zenodo: https://doi.org/10.5281/zenodo.20688837}
}
```

---

## License

Apache 2.0. See [`LICENSE`](LICENSE).

Contributions are welcome. Please open an issue first for larger changes so we can align on scope and design.

---

*Validation is currently single-GPU (T4, A100) and 2-rank CPU. Multi-node and 8+ GPU runs remain open. Scenario A recovers in ~17/20 runs (failures attributed to a Gloo race in containers). Scenario B is recorded at 10/10 via `workflow_dispatch` only.*
