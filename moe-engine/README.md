# moe-engine

**A fault-tolerant runtime for hyperscale Mixture-of-Experts training.**

[![Tests](https://img.shields.io/badge/tests-260%20passing-brightgreen)](RESULTS.md)
[![Chaos B](https://img.shields.io/badge/Chaos%20B-10%2F10%20✅-brightgreen)](RESULTS.md#fault-tolerance--chaos-test-results)
[![T4 validated](https://img.shields.io/badge/T4%20GPU-validated%20June%202026-blue)](notebooks/moe_engine_v032_T4_validation.ipynb)
[![Version](https://img.shields.io/badge/version-v0.3.2-orange)](roadmap.md)
[![License](https://img.shields.io/badge/license-Apache%202.0-lightgrey)](../LICENSE)

---

## What it does

moe-engine is a research-grade infrastructure runtime for training large
Mixture-of-Experts (MoE) language models under the realistic constraint of
**continuous node failures at hyperscale (10K+ GPUs)**. It is not a model
definition — it is the systems layer that distributes, checkpoints, and
recovers training when nodes inevitably disappear.

It combines:
- A fused **Triton router kernel** (single HBM pass, analytic backward)
- **4D parallelism** — Data × Expert × Tensor × Pipeline
- **Strict mathematical invariants** for correctness (token conservation, NaN guards, index validity)
- **Elastic fault tolerance** with automatic expert resharding
- **Async two-tier checkpointing** (NVMe staging → S3 durable)
- **Rich MoE-aware telemetry** including comm/compute overlap ratio

---

## T4 GPU Validation (June 2026) — Key Results

All numbers are real measurements from `gpu_results.json` (attached to this release).  
Reproduce: see [`notebooks/moe_engine_v032_T4_validation.ipynb`](notebooks/moe_engine_v032_T4_validation.ipynb).

### Router throughput — CPU vs T4 GPU (Triton kernel)

| Config | CPU (M tok/s) | T4 GPU (M tok/s) | Speedup |
|--------|:-------------:|:----------------:|:-------:|
| N=512, H=256, E=16, K=2 | 0.747 | 2.141 | **2.9×** |
| N=1024, H=512, E=32, K=2 | 0.421 | 3.644 | **8.7×** |
| N=2048, H=1024, E=64, K=2 | 0.236 | 4.832 | **20.4×** |
| N=4096, H=2048, E=64, K=4 | 0.056 | 4.454 | **80.1×** |

GPU speedup scales superlinearly — the Triton single-HBM-pass advantage
becomes fully realised as the matrix size grows relative to the T4's L2 cache.

### GPU router throughput chart (v0.3.2, T4, real measurements)

![Router throughput GPU v0.3.2](benchmarks/charts/router_throughput_gpu_v0_3_2.png)

*Forward-only (blue) and forward+backward (orange) throughput on T4 GPU
(Triton kernel) vs CPU reference path (green dashed). Log scale.
Source: `gpu_results.json`, June 2026 T4 validation run.*

> **Note:** If the GPU chart image is not yet in `benchmarks/charts/`, run Section 9
> of the validation notebook on a T4 and copy `router_throughput_gpu_v0_3_2.png`
> into `benchmarks/charts/`.

### MoE layer throughput (v0.3.1, CPU, real measurements)

![MoE layer throughput v0.3.1](benchmarks/charts/moe_layer_throughput_v0.3.1.png)

*Full `DistributedMoELayer` forward (orange) vs single dense SwiGLU FFN baseline
(blue) on CPU. Source: `benchmarks/cpu_results_colab.json`.*

### Chaos resilience

| Scenario | Description | Runs | Pass Rate |
|----------|-------------|:----:|:---------:|
| **Scenario B** | Storage stall (10s I/O delay) | 10 | **100% ✅** |
| **Scenario A** | Node kill + recovery (SIGKILL) | 20 | **~85% ⚠️** |

Scenario A is flaky due to a Gloo `connectFullMesh` race in containerised
environments. Fix (replace Gloo with NCCL in chaos harness) is planned for v0.4.

See **[`RESULTS.md`](RESULTS.md)** for every real number with reproduction commands.

---

## What is actually built

| Component | Status | Detail |
|---|---|---|
| **Triton router — forward** | ✅ CI-verified | Fused matmul+softmax+topK+renorm; single HBM pass; 80.1× over CPU at N=4096 |
| **Triton router — backward** | ✅ CI-verified | Analytic Jacobian; `atol=rtol=1e-5` vs fp64 ref; 30 configs tested |
| **Token conservation** | ✅ CI-verified | `sum(dispatch_cnt) == N×K` every forward; 100-seed sweep; CPU + GPU |
| **Expert load imbalance** | ✅ v0.2 | `max/mean` load per step; in telemetry |
| **Router z-loss** | ✅ v0.2 | Auxiliary regulariser emitted per step |
| **EP all-to-all** | ✅ CI-verified | `all_to_all_single`; dedicated CUDA stream; CUDA event sync |
| **Compute-comm overlap** | ✅ | Expert FFN default stream ∥ a2a dedicated stream |
| **Overlap ratio telemetry** | ✅ v0.3 | `dispatch_ms / expert_compute_ms` in every step record |
| **DP via FSDP2** | ✅ | `fully_shard` along DP axis; expert weights excluded |
| **Tensor Parallelism** | ✅ v0.2 | `ColumnParallel + RowParallel`; both `w_gate`/`w_up` ColumnParallel; 2-rank verified |
| **Sequence Parallelism** | ✅ v0.2 | `scatter/gather`; active at `tp_size > 1` |
| **SP fused all-gather** | ✅ v0.3 | `next_weight` param halves SP collectives; 2-rank verified |
| **Pipeline Parallelism (1-proc)** | ✅ v0.2 | `PipelineStage` + 1F1B; 13 unit tests |
| **Pipeline Parallelism (multi-proc)** | ✅ v0.3 | `run_1f1b_distributed`; real `dist.send/recv`; activation tagging; 2-rank verified |
| **MFU accounting** | ✅ v0.2 | MoE-sparse `(K/E)×P_expert`; streaming tracker |
| **Pydantic MoEConfig** | ✅ v0.3.2 | Validated hierarchy; env-var overrides; field-level errors; 34 tests |
| **Async two-tier checkpoint** | ✅ CI-verified | NVMe (O_DIRECT, atomic rename) → S3; background thread |
| **TorchElastic recovery** | ✅ CI-verified | SIGKILL → reshard → reload → resume |
| **Structured JSONL telemetry** | ✅ | Thread-safe; TensorBoard + Prometheus + WandB |
| **WandB integration** | ✅ v0.3 | `WANDB_API_KEY` env; `--wandb-project`; `log_config()` |
| **Docker + docker-compose** | ✅ v0.2 | Multi-stage; 1/4/8-GPU targets; monitoring stack |
| **Kubernetes manifests** | ✅ v0.2 | Single-node + multi-node Indexed Job; PVC; etcd |
| **Benchmark suite** | ✅ v0.2 | CPU+GPU sweeps; JSON/CSV; chart generation |
| **CLI** | ✅ v0.3.2 | `moe train / benchmark / validate / info` (typer) |
| **Chaos Scenario B** | ✅ CI-verified | 100% pass rate (10/10) |
| **Chaos Scenario A** | ⚠️ Flaky | ~85%; Gloo race; fix planned v0.4 |
| **Nsight/CUPTI profiling** | ❌ v0.4 | Needs GPU hardware |
| **Real multi-node data** | ❌ v0.4 | Needs sustained cluster access |

---

## Quick start

```bash
# 1. Clone and install
git clone https://github.com/Mattral/Composed-Mixture-of-Experts-Engine.git
cd Composed-Mixture-of-Experts-Engine/moe-engine
pip install -e ".[dev]"

# 2. Validate configs
make validate-config
# or: python scripts/cli.py validate configs/

# 3. Smoke test (CPU, ~5s, no GPU needed)
make smoke
# or: python train.py --config configs/smoke.yaml --smoke

# 4. Full CPU test suite (235 tests, ~60s)
make test-cpu
# or: pytest tests/ -m cpu -k "not (2rank or multiprocess)"

# 5. GPU smoke (requires T4/RTX)
python train.py --config configs/smoke.yaml --smoke
# expects: Triton kernel compiles, step.jsonl written, MoE forward/backward clean

# 6. GPU benchmark sweep
make benchmark CUDA=1
# or: python benchmarks/run_benchmark.py --cuda --json benchmarks/gpu_results.json
```

---

## Repository layout

```
moe-engine/
├── train.py                    Training entrypoint (TorchElastic + 4D parallel)
├── Makefile                    test-cpu / test-gpu / smoke / benchmark / lint / clean
├── pyproject.toml              pytest markers (cpu, gpu, chaos); dependencies
├── requirements.txt            Pinned runtime + dev dependencies
│
├── configs/
│   ├── smoke.yaml              Toy config (H=32, E=4) — fast, CPU-only development
│   └── default.yaml            Production config (H=4096, E=64) — 64 GPUs
│
├── pkg/
│   ├── distributed/            4D parallelism (split into 6 focused modules)
│   │   ├── mesh.py             ParallelTopology, build_topology
│   │   ├── tensor_parallel.py  Column/RowParallelLinear, scatter/gather SP
│   │   ├── expert_parallel.py  all_to_all dispatch/combine, _CommStream
│   │   ├── pipeline_parallel.py PipelineStage, 1F1B schedule
│   │   ├── data_parallel.py    apply_fsdp2 (expert-excluded FSDP2)
│   │   ├── moe_layer.py        DistributedMoELayer, _SwiGLUExpert
│   │   └── parallel_mesh.py    ← backward-compat shim only
│   ├── kernels/moe_router.py   Triton fwd+bwd kernel; fp64 reference; RouterProfile
│   ├── elastic/fault_monitor.py AsyncCheckpointer; ClusterStateMachine
│   ├── telemetry/logger.py     StructuredLogger; StepRecord; Prometheus; WandB
│   ├── models/moe.py           RMSNorm; ToyMoEBlock; ToyMoEModel; build_model
│   └── utils/
│       ├── config.py           MoEConfig (Pydantic v2); load_config (legacy shim)
│       └── mfu.py              MFUAccountant; compute_moe_flops
│
├── tests/                      235 CPU tests + GPU-specific tests
│   ├── test_config.py          34 tests for MoEConfig system (new in v0.3.2)
│   ├── test_kernels.py         Router invariants (conservation, no-NaN, bounds)
│   ├── test_kernels_numerics.py 30 configs vs fp64 ref; atol=rtol=1e-5
│   ├── test_pipeline_parallel.py 13 single-process 1F1B tests + 2-rank mp.spawn
│   ├── test_chaos.py           Scenario A + B fault-injection tests
│   └── ...                     (14 test files total)
│
├── benchmarks/
│   ├── run_benchmark.py        CPU + GPU sweep; JSON/CSV + chart output
│   ├── BENCHMARKS.md           All real numbers; no illustrative entries
│   ├── charts/                 PNG/SVG throughput charts
│   └── *.json                  Colab run data (cpu_results_colab.json, etc.)
│
├── notebooks/
│   └── moe_engine_v032_T4_validation.ipynb   Full T4 validation (13 sections)
│
├── scripts/
│   ├── cli.py                  typer CLI: moe train / benchmark / validate / info
│   ├── validate_config.py      Standalone YAML validator
│   └── launch.sh               Multi-node torchrun launcher
│
├── deploy/
│   ├── docker/                 Dockerfile, docker-compose (1/4/8-GPU, monitoring)
│   └── k8s/                    Kubernetes Job + Indexed Job manifests
│
└── docs/
    ├── ARCHITECTURE.md         This document: component map, token lifecycle, design
    ├── DESIGN.md               System design rationale
    ├── testing.md              Four-tier test strategy (cpu/gpu/multi-node/chaos)
    └── ...
```

---

## Config system (Pydantic v2)

```python
from pkg.utils.config import MoEConfig, ConfigValidationError

# Load and validate (errors caught at load time with field-level messages)
cfg = MoEConfig.from_yaml("configs/smoke.yaml")

# All fields strongly typed
hidden_dim: int   = cfg.model.hidden_dim       # 32
num_experts: int  = cfg.model.num_experts      # 4
lr: float         = cfg.training.learning_rate # 3e-4
world_size: int   = cfg.parallelism.world_size # 1

# Environment variable overrides
# MOE_TRAINING__LEARNING_RATE=1e-4 python train.py --config ...

# Validation catches bad configs immediately
try:
    bad = MoEConfig.from_dict({"model": {"top_k": 99, "num_experts": 4, ...}})
except ConfigValidationError as e:
    print(e)  # "Config validation failed: [model] top_k (99) must be <= num_experts (4)"
```

---

## CLI

```bash
# Validate configs before launching anything
moe validate configs/
moe validate configs/default.yaml configs/smoke.yaml

# Single-GPU smoke run
moe train --config configs/smoke.yaml --smoke

# Multi-GPU (4 processes, local)
moe train --config configs/default.yaml --nproc 4

# GPU benchmark
moe benchmark --cuda --json benchmarks/gpu_results.json

# Environment info
moe info
```

---

## Test suite

```bash
# Fast Tier-0 CPU suite (235 tests, ~60s, no GPU required)
pytest tests/ -m cpu -k "not (2rank or multiprocess or distributed_invariants)"

# GPU kernel tests (requires CUDA + Triton)
pytest tests/test_kernels.py -m gpu -v

# Chaos tests (requires torchrun + Gloo; ~3 min for Scenario B)
GLOO_SOCKET_IFNAME=lo pytest tests/test_chaos.py -m chaos -k "scenario_b"

# All in one:
make test-cpu         # Tier-0 CPU
make test-gpu         # Tier-1 GPU
make chaos-b          # Scenario B (10/10 expected)
make chaos-a          # Scenario A (~85% expected)
```

---

## T4 Validation Notebook

[`notebooks/moe_engine_v032_T4_validation.ipynb`](notebooks/moe_engine_v032_T4_validation.ipynb)

Open in Colab with a T4 GPU to reproduce:
- All numbers in `RESULTS.md` and `BENCHMARKS.md`
- The throughput chart in `benchmarks/charts/router_throughput_gpu_v0_3_2.png`
- Chaos Scenario A and B pass rates
- Production-scale Triton kernel sanity check (H=4096, E=64, K=2)

---

## v0.3.2 changelog

**P0.1 — Architectural cleanup**
- `parallel_mesh.py` (1,165 lines) split into 6 focused modules, each < 380 lines
- Backward-compat shim preserves all existing imports — zero breaking changes
- `pkg/models/moe.py` extracted from `train.py`; `build_model()` factory
- `__all__` on every `pkg/**/__init__.py`

**P0.2 — Testing & validation**
- `test_config.py`: 34 new tests covering the full `MoEConfig` system
- `@pytest.mark.cpu` on all 14 CPU test files; `@pytest.mark.gpu` registered
- Total: **260 tests passing** (up from 201)
- Every "illustrative" number in docs replaced with real T4 measurements

**P0.3 — Developer experience**
- `Makefile`: `test-cpu`, `test-gpu`, `smoke`, `benchmark`, `validate-config`, `lint`, `clean`
- `scripts/cli.py`: `typer` CLI with `train`, `benchmark`, `validate`, `info`
- `scripts/validate_config.py`: standalone YAML validator with coloured output
- `notebooks/moe_engine_v032_T4_validation.ipynb`: full 13-section validation notebook

---

## Roadmap

See [`roadmap.md`](roadmap.md) and [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for
the full v0.4 plan. High-priority items:

- **v0.4**: Replace Gloo with NCCL in chaos harness → fix Scenario A flakiness
- **v0.4**: Real 8-GPU+ benchmark data and end-to-end MFU validation
- **v0.4**: Nsight/CUPTI roofline integration
- **v0.4**: Expert capacity overflow re-routing
- **v0.4**: Non-divisible sequence length in Sequence Parallelism

---

## Citation

If you use moe-engine in your research, please cite the preprint:

```bibtex
@misc{myet2026moeengine,
  author = {Min Htet Myet},
  title  = {moe-engine: A Fault-Tolerant Runtime for Hyperscale
             Mixture-of-Experts Training},
  year   = {2026},
  url    = {https://github.com/Mattral/Composed-Mixture-of-Experts-Engine},
  note   = {v0.3.2 preprint. Zenodo: https://doi.org/10.5281/zenodo.20647577}
}
```

---

## License

Apache 2.0. See [`LICENSE`](../LICENSE).
