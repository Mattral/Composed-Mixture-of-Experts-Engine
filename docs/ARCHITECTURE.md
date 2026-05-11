# Architecture Overview

**Version:** v0.3  
**Last updated:** June 2026

---

## Purpose

`moe-engine` is a production-grade training runtime for large Mixture-of-Experts
language models. It is not a model definition — it is the infrastructure layer
that distributes, checkpoints, and recovers training across thousands of GPUs.

The central constraint: **at hyperscale, nodes die continuously**. Every design
decision flows from that premise.

---

## Component Map

```
train.py
│
├── pkg/utils/config.py          YAML config loader
│
├── pkg/distributed/
│   └── parallel_mesh.py         4D topology (DP × TP × PP × EP)
│                                DistributedMoELayer
│                                ColumnParallelLinear / RowParallelLinear
│                                scatter/gather_sequence_parallel
│                                PipelineStage (1F1B schedule)
│                                apply_fsdp2
│
├── pkg/kernels/
│   └── moe_router.py            MoERouter
│                                MoERouterFunction (autograd.Function)
│                                Triton fwd + bwd kernels
│                                _reference_route_fp64 (CPU fallback)
│                                _compute_load_imbalance
│                                _compute_router_z_loss
│                                RouterProfile dataclass
│
├── pkg/elastic/
│   └── fault_monitor.py         AsyncCheckpointer
│                                LocalNVMeAdapter / S3Adapter
│                                ClusterStateMachine
│                                ElasticTrainerHarness
│                                _largest_divisor_le
│
├── pkg/telemetry/
│   └── logger.py                StructuredLogger
│                                StepRecord
│                                PrometheusExporter
│
└── pkg/utils/
    └── mfu.py                   MFUAccountant
                                 MFUResult
                                 compute_mfu / compute_mfu_detailed
                                 compute_moe_flops
```

---

## 4D Parallelism Mesh

```
World = dp_size × tp_size × pp_size × ep_size

  ┌──────────────────────────────────────────────────┐
  │  DP axis  (FSDP2 per-parameter DTensor sharding) │
  │  TP axis  (ColumnParallel / RowParallel / SP)    │
  │  PP axis  (PipelineStage 1F1B schedule)          │
  │  EP axis  (all_to_all_single on dedicated stream)│
  └──────────────────────────────────────────────────┘
```

**Data Parallelism (DP):**  
`apply_fsdp2()` wraps every non-MoE module with `fully_shard` along the `dp`
mesh axis using PyTorch 2.5+ DTensor. Expert weights are explicitly excluded —
they are already partitioned across the `ep` axis and must not be FSDP-wrapped.

**Tensor Parallelism (TP):**  
Expert FFN layers use `ColumnParallelLinear` (splits output features, all-gathers
to full width) and `RowParallelLinear` (splits input features, all-reduces partial
outputs). Both `w_gate` and `w_up` in the SwiGLU expert are `ColumnParallelLinear`
so that their element-wise product occurs in shard space `[F // tp_size]` before
`w_down` (RowParallel) all-reduces to the full hidden dimension. Sequence
Parallelism (SP) activates automatically at `tp_size > 1`.

**Pipeline Parallelism (PP):**  
`PipelineStage` encapsulates a single pipeline stage. `run_1f1b(micro_batches)`
implements the three-phase 1F1B schedule in single-process mode (test fast-path).
`run_1f1b_distributed(micro_batches)` — **v0.3** — implements the full multi-process
1F1B schedule with real `dist.send` / `dist.recv` on the PP group axis, activation
tagging (stage_id + mb_index header), and three-phase execution (warmup → steady → drain).

**Expert Parallelism (EP):**  
Each EP rank owns `E // ep_size` experts (remainder experts are assigned
round-robin). Token dispatch uses `all_to_all_single` on a dedicated high-priority
CUDA stream (`_CommStream`), overlapping with expert FFN compute on the default
stream. A CUDA event records the dispatch boundary so combine waits only as long
as necessary.

---

## Token Life-Cycle in One MoE Layer

```
tokens [N_local, H]
        │
        ▼
 MoERouter (Triton kernel or CPU reference)
        │
        ├── topk_idx  [N_local, K]
        ├── topk_w    [N_local, K]   (renormalised combine weights)
        └── dispatch_cnt [E]          (load imbalance + z-loss computed here)
        │
        ▼
 sort tokens by expert id → tokens_sorted [N_local × K, H]
        │
        ▼  all_to_all_dispatch (EP dedicated stream)
        │
 expert FFN on this rank [n_recv, H] → SwiGLU: silu(gate(x)) × up(x) → down(·)
        │
        ▼  all_to_all_combine (EP dedicated stream)
        │
 scatter → weighted sum over K slots
        │
        ▼
 combined output [N_local, H]   (NaN check enforced)
```

---

## Kernel Architecture: Triton Router

The router kernel (`_router_fwd_kernel`) is a single Triton JIT kernel that fuses:

1. `tokens [N, H] @ gate_w [H, E]` → `logits [N, E]`
2. `softmax(logits, dim=-1)` → `probs [N, E]`
3. `top_k(probs, K)` → `idx [N, K]`, `vals [N, K]`
4. `renormalize(vals)` → `combine_weights [N, K]`

All four operations execute in a single pass over the gating dimension, with
`(BLOCK_N=64, BLOCK_E=64)` tiles held in L1/SRAM (~16 KiB). Global loads are
coalesced on the contiguous H dimension. Top-K uses in-SRAM selection sort (K
iterations, O(K×E)), which outperforms bitonic sort for K ∈ {1,2,4}.

The backward kernel (`_router_bwd_kernel`) propagates `grad_w → grad_p → grad_l`
using the analytic softmax Jacobian: `grad_l_i = p_i × (grad_p_i − Σ(grad_p · p))`.
Validated against the fp64 PyTorch reference at `atol=rtol=1e-5`.

On CPU or without Triton, `_reference_route_fp64` provides the identical
computation in fp64 PyTorch — used as the ground truth for all numerics tests.

---

## Checkpoint Architecture: Two-Tier Async

```
Training thread
     │
     │  D2H copy (NVLink, ~tens of ms for sharded param)
     ▼
 Background I/O thread (AsyncCheckpointer)
     │
     ├── LocalNVMeAdapter
     │    └── 256 MB chunk writes, O_DIRECT (fallback to buffered)
     │         atomic: tmp → final (rename)
     │
     └── S3Adapter / LocalNVMeAdapter (remote tier)
          └── boto3 multipart upload or file copy
```

The training thread pays only the D2H copy cost (~tens of ms). All disk and
network I/O runs in a background thread pool. Atomic rename ensures every
checkpoint is either fully present or absent — no partial reads possible.

Checkpoint metadata (`.meta.json`) records step, rank, timestamp, and hostname
alongside every `.pt` shard. Retention pruning deletes old checkpoints after
committing, bounding total disk usage to `retention × shard_size`.

---

## Telemetry Architecture

Every training step emits one `StepRecord` to three sinks simultaneously:

| Sink | Format | Notes |
|---|---|---|
| JSONL file | newline-delimited JSON | all fields; thread-safe via `RLock` |
| TensorBoard | scalar summaries | rank 0 only |
| Prometheus | gauges on `/metrics` | optional; port configurable |

v0.2 routing quality fields: `routing.expert_load_imbalance`, `routing.router_z_loss`.

v0.3 collective fields (new):
- `collective.expert_compute_ms` — wall-clock time of expert FFN compute per step
- `collective.comm_compute_overlap_ratio` — `dispatch_ms / expert_compute_ms` (overlap fraction)

WandB sink (v0.3): active when `WANDB_API_KEY` is set; all numeric fields logged
under section prefixes (`collective/dispatch_ms`, `routing/z_loss`, etc.).

---

## Design Principles

**Measure, don't estimate.**  
All telemetry values come from real measurements: CUDA events for collective
latency, `torch.cuda.memory_stats()` for peak memory, actual elapsed time for
step duration. There are no placeholder or fabricated numbers in the runtime.

**Fail-fast on invariants.**  
Token conservation (`sum(dispatch_cnt) == N × K`), index bounds (`idx ∈ [0, E)`),
combine weight normalisation, and post-combine NaN checks are all asserted in the
forward path. The invariant fires immediately at the offending layer, not
silently further downstream.

**Single correct collective per pattern.**  
RowParallel uses `all_reduce(SUM)` — the correct primitive for summing partial
matrix products. ColumnParallel uses `all_gather_into_tensor`. Sequence parallel
uses `scatter` (forward) and `all_gather` (backward). Each pattern uses exactly
one collective, not a composed sequence of two.

**Link, don't duplicate.**  
Documentation references source files and test functions rather than
re-describing behaviour inline. When behaviour changes, update both.

---

## Implementation Status

| Component | Status | Notes |
|---|---|---|
| Triton forward kernel | ✅ | Fused matmul+softmax+topK+renorm; SRAM-tiled |
| Triton backward kernel | ✅ | Analytic Jacobian; `atol=rtol=1e-5` vs fp64 ref |
| Token conservation invariant | ✅ | 100-seed sweep clean |
| DP via FSDP2 + DTensor | ✅ | Per-param sharding; expert layers excluded |
| EP all-to-all (dispatch + combine) | ✅ | Dedicated CUDA stream; event sync; ep=1 no-op |
| Compute-comm overlap | ✅ | Expert FFN default stream, a2a dedicated stream |
| ColumnParallelLinear | ✅ | all-gathers to full width; tp=1 identity |
| RowParallelLinear | ✅ | all_reduce(SUM); tp=1 identity |
| SwiGLU w_gate + w_up both ColumnParallel | ✅ | Consistent shard space through SwiGLU |
| Sequence Parallelism (scatter/gather) | ✅ | no-op at tp_size=1 |
| SP all-gather fusion | ✅ v0.3 | `next_weight` param fuses backward all-gather with next projection; halves SP collectives |
| PipelineStage 1F1B (single-process) | ✅ | 3-phase schedule; 13 unit tests; fast-path |
| PipelineStage multi-process PP | ✅ v0.3 | `run_1f1b_distributed`; dist.send/recv; 2-rank mp.spawn verified |
| Async two-tier checkpointing | ✅ | NVMe (O_DIRECT) + S3; atomic rename |
| TorchElastic state machine | ✅ | evict → reshard → reload → resume |
| Etcd rendezvous (>100 nodes) | ✅ | Backend selector; c10d fallback |
| MFU accounting (sparse-aware) | ✅ | `(K/E) × P_expert`; streaming tracker |
| Routing quality metrics | ✅ | load imbalance + z-loss per step |
| Structured JSONL + TensorBoard | ✅ | Thread-safe; RLock |
| Prometheus `/metrics` endpoint | ✅ | Optional; 10 gauges (v0.3: +expert_compute_ms, +overlap_ratio) |
| WandB integration | ✅ v0.3 | `WandBSink`; `WANDB_API_KEY` env; `--wandb-project` flag; `log_config` |
| Docker + docker-compose | ✅ | Multi-stage; smoke/4-GPU/8-GPU targets |
| Kubernetes manifests | ✅ | Single-node Job + 16-node Indexed Job |
| Benchmark suite | ✅ | `benchmarks/run_benchmark.py`; CPU+GPU; JSON/CSV |
| Chaos Scenario A fix (NCCL) | ❌ | v0.4 — Gloo race mitigated ~85%; needs GPU |
| Nsight/CUPTI roofline | ❌ | v0.4 |
| Real multi-node benchmark data | ❌ | v0.4 — needs cluster |
