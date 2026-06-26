# Architecture Overview

**Version:** v0.3.2  
**Last updated:** June 2026

---

## Purpose

`moe-engine` is a research-grade training runtime for large Mixture-of-Experts
language models at hyperscale (10K+ GPUs). It is not a model definition — it
is the infrastructure layer that distributes, checkpoints, and recovers training
across thousands of GPUs under continuous node failure.

The central constraint: **at hyperscale, nodes die continuously**. Every design
decision flows from that premise.

---

## Component Map (v0.3.2)

The v0.3.2 refactoring split the original 1,165-line `parallel_mesh.py` monolith
into focused single-responsibility modules. Each module now has one concern.

```
train.py
│
├── pkg/utils/
│   ├── config.py               MoEConfig (Pydantic v2 validated hierarchy)
│   │                           ModelConfig / TrainingConfig / ParallelismConfig
│   │                           CheckpointConfig / ElasticConfig / TelemetryConfig
│   │                           load_config() (legacy shim)
│   └── mfu.py                  MFUAccountant, MFUResult
│                               compute_mfu / compute_mfu_detailed / compute_moe_flops
│
├── pkg/distributed/
│   ├── __init__.py             Curated public API surface (__all__)
│   ├── mesh.py                 ParallelTopology (frozen dataclass)
│   │                           build_topology() — DeviceMesh + process groups
│   │                           tp_process_group() / pp_process_group()
│   │                           experts_on_this_rank() — round-robin EP ownership
│   ├── tensor_parallel.py      ColumnParallelLinear / RowParallelLinear
│   │                           scatter_to_sequence_parallel (fused next_weight path)
│   │                           gather_from_sequence_parallel
│   ├── expert_parallel.py      all_to_all_dispatch / all_to_all_combine
│   │                           _CommStream — dedicated high-priority CUDA stream
│   │                           comm/compute overlap telemetry
│   ├── pipeline_parallel.py    PipelineStage — real dist.send/recv (v0.3)
│   │                           run_1f1b() — single-process scheduling shim
│   │                           run_1f1b_distributed() — multi-process 1F1B
│   │                           Activation tagging (mb_index headers)
│   ├── data_parallel.py        apply_fsdp2() — expert-weights-excluded FSDP2
│   ├── moe_layer.py            DistributedMoELayer (thin orchestrator)
│   │                           _SwiGLUExpert (TP-aware two-layer FFN)
│   │                           _expert_to_rank() — EP rank lookup
│   │                           Telemetry: last_dispatch_ms, last_overlap_ratio
│   └── parallel_mesh.py        ← backward-compat shim only (re-exports all of above)
│                               New code should not import from here.
│
├── pkg/models/
│   └── moe.py                  RMSNorm, ToyMoEBlock, ToyMoEModel
│                               build_model() — factory, moves to topology.device
│                               count_parameters() — breakdown by component
│
├── pkg/kernels/
│   └── moe_router.py           MoERouter (public interface)
│                               MoERouterFunction (autograd.Function)
│                               Triton fwd kernel (moe_topk_route_fwd)
│                               Triton bwd kernel (moe_topk_route_bwd)
│                               _reference_route_fp64 (CPU fallback)
│                               _compute_load_imbalance / _compute_router_z_loss
│                               RouterProfile dataclass
│
├── pkg/elastic/
│   └── fault_monitor.py        AsyncCheckpointer (NVMe + S3 two-tier)
│                               LocalNVMeAdapter / S3Adapter
│                               ClusterStateMachine (RUNNING→DRAINING→RECOVERING)
│                               ElasticTrainerHarness
│                               _largest_divisor_le (resharding math)
│
├── pkg/telemetry/
│   └── logger.py               StructuredLogger (thread-safe JSONL)
│                               StepRecord (full v0.3 envelope)
│                               PrometheusExporter / WandBSink
│
└── scripts/
    ├── validate_config.py      Validate configs/*.yaml at load time
    └── cli.py                  typer CLI: train / benchmark / validate
```

---

## 4D Parallelism Mesh

```
World = dp_size × tp_size × pp_size × ep_size

  ┌──────────────────────────────────────────────────────────────┐
  │  DP axis  (FSDP2 per-parameter DTensor sharding)            │
  │  TP axis  (ColumnParallel / RowParallel / SP fused gather)  │
  │  PP axis  (PipelineStage 1F1B; real dist.send/recv in v0.3) │
  │  EP axis  (all_to_all_single on dedicated CUDA stream)      │
  └──────────────────────────────────────────────────────────────┘
```

Each axis is implemented in its own module and is independently testable
at `tp_size=1` / `pp_size=1` / `ep_size=1` without requiring a distributed
environment.

---

## Token Lifecycle (per forward pass)

```
input tokens [B, S, H]
    │
    ▼
MoERouter (Triton kernel / fp64 ref)
    → expert indices idx [N, K]
    → routing weights  w  [N, K]
    → dispatch_cnt [ep_size]
    │
    ▼ sort by expert (contiguous dispatch)
tokens_sorted [N*K, H]
    │
    ▼ all_to_all_dispatch (EP collective, dedicated stream)
received [total_recv, H]  ← overlaps with ▼
    │
    ▼ local expert FFN (_SwiGLUExpert, default stream)
expert_out [total_recv, H]
    │
    ▼ all_to_all_combine (EP collective, stream sync)
combined_sorted [total_send, H]
    │
    ▼ weighted scatter to original positions
output [B, S, H]  ← NaN guard asserted
```

---

## Fault Recovery Sequence

```
Node death detected (health_check or SIGTERM)
    │
    ▼
ClusterStateMachine: RUNNING → DRAINING
    │ (drain current micro-batches from 1F1B pipeline)
    ▼
AsyncCheckpointer: flush in-flight NVMe write, wait for S3 upload
    │
    ▼
TorchElastic: restart dead rank, re-init process group
    │
    ▼
ClusterStateMachine: DRAINING → RECOVERING
    │ (_largest_divisor_le selects new ep_size)
    │ (expert resharding: round-robin to surviving ranks)
    ▼
AsyncCheckpointer: load latest checkpoint from NVMe tier (fast)
    │
    ▼
ClusterStateMachine: RECOVERING → RESUMED
    │ training continues with reduced world size
    ▼
Background: S3 upload of post-recovery checkpoint
```

---

## Async Two-Tier Checkpointing

```
Training loop (foreground)
    │ optimizer.step()
    │ harness.checkpoint(model, optimizer, step)
    │
    ▼
AsyncCheckpointer.save() — non-blocking
    │ ├─ Tier 1: LocalNVMeAdapter
    │ │    serialize → RAM buffer
    │ │    write to NVMe (O_DIRECT attempt, fallback buffered)
    │ │    atomic rename: .tmp → final path
    │ │    prune oldest (retain last N)
    │ │
    │ └─ Tier 2: S3Adapter (background thread)
    │       read from NVMe → chunked upload
    │       write .meta.json (step, rank, ts, hostname)
    │
    ▼ (background workers; training continues immediately)
```

---

## Key Design Invariants

Every invariant is asserted at runtime AND tested independently:

| Invariant | Where enforced | Test |
|-----------|---------------|------|
| Token conservation: `sum(dispatch_cnt) == N×K` | `MoERouter.forward()` | `test_kernels.py::test_token_conservation` |
| No NaN in router output | `MoERouterFunction.forward()` | `test_kernels.py::test_no_nan_in_indices` |
| Index validity: `idx ∈ [0, E)` | `MoERouterFunction.forward()` | `test_kernels.py::test_index_validity` |
| Weight normalisation: `w.sum(-1) ≈ 1` | `MoERouterFunction.forward()` | `test_kernels_numerics.py` |
| No NaN in layer output | `DistributedMoELayer.forward()` | `test_distributed_invariants.py` |
| Expert coverage: all experts assigned | `ClusterStateMachine._reshard()` | `test_elastic_v02.py` |
| Retention: latest checkpoint always survives | `AsyncCheckpointer._prune()` | `test_elastic.py` |
| Config validation at load time | `MoEConfig.from_yaml()` | `test_config.py` |

---

## Module Size Targets (post v0.3.2)

Per MOE_instructions v2.1: no file in `pkg/distributed/` should exceed ~450 lines.

| Module | Lines | Status |
|--------|------:|--------|
| `mesh.py` | 316 | ✅ |
| `tensor_parallel.py` | 308 | ✅ |
| `expert_parallel.py` | 231 | ✅ |
| `pipeline_parallel.py` | 378 | ✅ |
| `data_parallel.py` | 101 | ✅ |
| `moe_layer.py` | 272 | ✅ |
| `parallel_mesh.py` (shim) | 68 | ✅ |
| **Total** | **1,674** | **vs 1,165 monolith** |

Each module now has a single, reviewable responsibility.
