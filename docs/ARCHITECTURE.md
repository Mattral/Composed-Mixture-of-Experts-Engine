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

## What Is Actually Built

| Component | Status | Detail |
|---|---|---|
| **Triton router kernel — forward** | ✅ CI-verified | Fused matmul+softmax+topK+renorm; single HBM pass; SRAM tile 64×64 |
| **Triton router kernel — backward** | ✅ CI-verified | Analytic ∂/∂logits through softmax→topK→renorm; `atol=rtol=1e-5` |
| **Token conservation invariant** | ✅ CI-verified | `sum(dispatch_cnt) == N×K` asserted every forward; 100-seed sweep |
| **Expert load imbalance metric** | ✅ v0.2 | `max_load / mean_load` tracked per step; logged to telemetry |
| **Router z-loss** | ✅ v0.2 | Auxiliary regularisation signal; emitted per step |
| **DP+EP device mesh** | ✅ CI-verified | `init_device_mesh` (PyTorch 2.5+); degenerate 1-rank fallback |
| **EP all-to-all (dispatch + combine)** | ✅ CI-verified | Non-blocking `all_to_all_single`; dedicated CUDA stream; event sync |
| **Compute-comm overlap** | ✅ | Expert FFN runs on default stream while a2a is in flight |
| **Comm/compute overlap ratio** | ✅ v0.3 | `dispatch_ms / expert_compute_ms`; emitted in `collective` telemetry block |
| **FSDP2 sharding** | ✅ | `fully_shard` along DP axis; per-param DTensor; MixedPrecision |
| **Tensor Parallelism** | ✅ v0.2 | `ColumnParallelLinear` + `RowParallelLinear`; both `w_gate` and `w_up` ColumnParallel; `all_reduce` in RowParallel; 2-rank mp.spawn verified |
| **Sequence Parallelism** | ✅ v0.2 | `scatter/gather_sequence_parallel`; active when `tp_size > 1` |
| **SP all-gather fusion** | ✅ v0.3 | `next_weight` param fuses backward all-gather with next projection matmul; halves SP collectives; 2-rank mp.spawn verified |
| **Pipeline Parallelism (single-process)** | ✅ v0.2 | `PipelineStage` + 1F1B schedule; warmup/steady/drain phases; 13 unit tests |
| **Pipeline Parallelism (multi-process)** | ✅ v0.3 | `run_1f1b_distributed`; real `dist.send`/`dist.recv` on PP group; activation tagging; 2-rank mp.spawn verified |
| **MFU accounting** | ✅ v0.2 | MoE-sparse formula: `(K/E)×P_expert`; `MFUAccountant` streaming tracker |
| **Real CUDA telemetry** | ✅ v0.2 | CUDA events on dispatch + combine; `memory_stats()` peak GB |
| **WandB integration** | ✅ v0.3 | `WandBSink`; active when `WANDB_API_KEY` set; `--wandb-project` flag; `log_config()` records hyperparameters |
| **Pydantic MoEConfig** | ✅ v0.3.2 | Validated hierarchical config; env-var overrides; `ConfigValidationError` with field paths |
| **Async two-tier checkpointing** | ✅ CI-verified | Pinned host → NVMe (O_DIRECT, 256 MB chunks, atomic rename) → S3 |
| **TorchElastic state machine** | ✅ CI-verified | Evict → reshard (round-robin) → reload → resume |
| **Etcd rendezvous** | ✅ v0.2 | `ElasticTrainerHarness` backend selector; c10d (<100 nodes) / etcd (>100) |
| **Prometheus metrics** | ✅ v0.3 | Optional in-process `/metrics` endpoint; 10 gauges (incl. `expert_compute_ms`, `comm_compute_overlap_ratio`) |
| **Docker + docker-compose** | ✅ v0.2 | Multi-stage image; 1/4/8-GPU compose targets; monitoring stack |
| **Kubernetes manifests** | ✅ v0.2 | Single-node Job + multi-node Indexed Job; PVC; etcd rendezvous |
| **Benchmark suite** | ✅ v0.2 | `benchmarks/run_benchmark.py`; CPU+GPU sweeps; JSON/CSV output |
| **Chaos: storage stall (Scenario B)** | ✅ CI-verified | 10s injected stall; queue drains; no deadlock; **100% pass rate** |
| **Chaos: node kill + recovery (Scenario A)** | ⚠️ Flaky | ~85% pass rate; Gloo `connectFullMesh` timeout on 4-rank restart |
| **Nsight/CUPTI integration** | ❌ Planned v0.4 | Requires GPU hardware |
| **Real multi-node benchmark data** | ❌ Planned v0.4 | Requires sustained cluster access |

---

## Component Map (v0.3.2)

The v0.3.2 refactoring split the original 1,165-line `parallel_mesh.py` monolith
into focused single-responsibility modules. Each module now has one concern and
remains reviewable by a single engineer in one sitting.

```
train.py
│
├── pkg/utils/
│   ├── config.py               MoEConfig — Pydantic v2 validated hierarchy
│   │                           ├── ModelConfig       (H, E, K, F, dtype, …)
│   │                           ├── TrainingConfig    (lr, warmup, grad_clip, …)
│   │                           ├── ParallelismConfig (dp, ep, tp, pp sizes)
│   │                           ├── CheckpointConfig  (NVMe dir, S3 URI, …)
│   │                           ├── ElasticConfig     (min/max nodes, rdzv, …)
│   │                           └── TelemetryConfig   (log_dir, MFU target, …)
│   │                           ConfigValidationError (field-level messages)
│   │                           load_config() — legacy shim; new code uses from_yaml()
│   └── mfu.py                  MFUAccountant, MFUResult
│                               compute_mfu / compute_mfu_detailed / compute_moe_flops
│
├── pkg/distributed/
│   ├── __init__.py             Curated public API (__all__ covers all symbols)
│   ├── mesh.py                 ParallelTopology (frozen dataclass)
│   │                           ├── dp_rank / tp_rank / pp_rank / ep_rank (computed)
│   │                           ├── experts_on_this_rank() — round-robin ownership
│   │                           ├── is_first/last_pp_stage()
│   │                           └── validate_world_size()
│   │                           build_topology(dp, ep, tp, pp) — DeviceMesh + PGs
│   │                           tp_process_group() / pp_process_group() (cached)
│   │
│   ├── tensor_parallel.py      ColumnParallelLinear — shards output [out//tp, in]
│   │                           RowParallelLinear    — shards input  [out, in//tp]
│   │                           scatter_to_sequence_parallel(x, topo, next_weight=None)
│   │                           gather_from_sequence_parallel(x, topo)
│   │                           (fused next_weight path halves SP collectives — v0.3)
│   │
│   ├── expert_parallel.py      all_to_all_dispatch(tokens_sorted, send_counts, topo)
│   │                           all_to_all_combine(expert_out, recv_cnts, send_cnts, topo)
│   │                           _CommStream — singleton high-priority CUDA stream per device
│   │                           (returns dispatch_event for overlap with expert FFN compute)
│   │
│   ├── pipeline_parallel.py    PipelineStage(stage_id, num_stages, module, topology)
│   │                           ├── forward_step(mb) — recv → compute → send
│   │                           ├── run_1f1b(micro_batches) — single-process fast path
│   │                           └── run_1f1b_distributed(micro_batches, loss_fn)
│   │                               — real dist.send/recv; 3-phase 1F1B (v0.3)
│   │                               — activation tagging: [stage_id, mb_index] header
│   │
│   ├── data_parallel.py        apply_fsdp2(model, topology, mixed_precision_dtype)
│   │                           (expert weights excluded — already EP-sharded)
│   │
│   ├── moe_layer.py            _SwiGLUExpert(H, F, topology, dtype)
│   │                           DistributedMoELayer(H, F, E, K, topology, …)
│   │                           ├── _expert_to_rank(ids) — EP rank lookup table
│   │                           ├── last_dispatch_ms / last_combine_ms
│   │                           ├── last_expert_compute_ms
│   │                           └── last_overlap_ratio = dispatch_ms / compute_ms
│   │
│   └── parallel_mesh.py        ← backward-compat shim only (68 lines)
│                               re-exports everything above; new code should not import here
│
├── pkg/models/
│   └── moe.py                  RMSNorm, ToyMoEBlock, ToyMoEModel
│                               build_model(cfg, topology) — factory, moves to device
│                               count_parameters() — breakdown by component
│
├── pkg/kernels/
│   └── moe_router.py           MoERouter (public interface)
│                               MoERouterFunction (autograd.Function)
│                               _router_fwd_kernel (Triton JIT)
│                               _router_bwd_kernel (Triton JIT)
│                               _reference_route_fp64 (CPU fallback, ground truth)
│                               RouterProfile (sram_bytes, bw_gbps, imbalance, z_loss)
│
├── pkg/elastic/
│   └── fault_monitor.py        AsyncCheckpointer (NVMe + S3 two-tier)
│                               LocalNVMeAdapter / S3Adapter
│                               ClusterStateMachine (RUNNING→DRAINING→RECOVERING→RESUMED)
│                               ElasticTrainerHarness
│                               _largest_divisor_le (resharding math)
│
├── pkg/telemetry/
│   └── logger.py               StructuredLogger (thread-safe JSONL)
│                               StepRecord (full v0.3 envelope with all fields)
│                               PrometheusExporter / WandBSink
│
├── scripts/
│   ├── cli.py                  typer CLI: moe train / benchmark / validate / info
│   ├── validate_config.py      Standalone YAML validator (used by make validate-config)
│   ├── launch.sh               Multi-node torchrun launcher
│   └── simulate_node_failure.sh  Manual chaos injection helper
│
└── notebooks/
    └── moe_engine_v032_T4_validation.ipynb   Full T4 GPU validation (13 sections)
```

---

## 4D Parallelism Mesh

```
World = dp_size × tp_size × pp_size × ep_size

  ┌────────────────────────────────────────────────────────────────────┐
  │  DP axis  (FSDP2 per-parameter DTensor sharding along dp mesh)    │
  │  TP axis  (ColumnParallel / RowParallel / SP fused all-gather)    │
  │  PP axis  (PipelineStage 1F1B; real dist.send/recv in v0.3)       │
  │  EP axis  (all_to_all_single on dedicated high-priority CUDA stream│
  └────────────────────────────────────────────────────────────────────┘
```

**Data Parallelism (DP):**
`apply_fsdp2()` wraps every non-MoE module with `fully_shard` along the `dp`
mesh axis using PyTorch 2.5+ DTensor. Expert weights are explicitly excluded —
they are already partitioned across the `ep` axis and must not be FSDP-wrapped
a second time. MixedPrecisionPolicy is applied when `dtype=bfloat16` to
accumulate gradients in fp32 while computing in bf16.

**Tensor Parallelism (TP):**
Expert FFN layers use `ColumnParallelLinear` (splits output features across
`tp_size` ranks, then `all_gather_into_tensor` to full width) and
`RowParallelLinear` (splits input features, then `all_reduce(SUM)` of partial
matrix products). Both `w_gate` and `w_up` in the SwiGLU expert are
`ColumnParallelLinear` so that their element-wise product `silu(gate(x)) * up(x)`
occurs in shard space `[..., F // tp_size]` before `w_down` (RowParallel)
`all_reduce`s to the full hidden dimension `H`. This is the correct factorisation
— ColumnParallel for both gate and up, RowParallel for down — and matches the
Megatron-LM convention. Sequence Parallelism (SP) activates automatically at
`tp_size > 1` via `scatter_to_sequence_parallel` (scatter forward, gather
backward). In v0.3 the backward all-gather is fused with the subsequent
projection matmul via the `next_weight` parameter, halving the number of
collectives per SP layer.

**Pipeline Parallelism (PP):**
`PipelineStage` encapsulates a single pipeline stage's compute and communication.
`run_1f1b(micro_batches)` implements the three-phase 1F1B schedule in
single-process mode — used by the 13-test unit test suite and as a fast path
when `pp_size == 1`. `run_1f1b_distributed(micro_batches, loss_fn)` — **v0.3** —
implements the full multi-process 1F1B schedule with real `dist.send` /
`dist.recv` on the PP process group axis. Each inter-stage transfer uses
activation tagging: a 2-element `int64` header `[stage_id, mb_index]` is sent
immediately before the activation tensor so receivers can match micro-batches
correctly across restarts or reordering. The three phases are Warmup (issue
`p-1` forwards without backward), Steady-state (one forward + one backward per
clock tick), and Drain (issue remaining backwards). Multi-process correctness
is verified by 2-rank `mp.spawn` tests.

**Expert Parallelism (EP):**
Each EP rank owns `E // ep_size` experts; remainder experts are assigned
round-robin to the lowest-indexed EP ranks so resharding after a node drop
never leaves any expert unowned. Token dispatch uses `all_to_all_single` on
a dedicated high-priority CUDA stream (`_CommStream`, singleton per device
index), overlapping with expert FFN compute on the default stream. A CUDA event
(`dispatch_event`) records the moment the dispatch collective completes; the
combine collective calls `stream.wait_event(dispatch_event)` before sending,
ensuring correct ordering without over-synchronising. The overlap ratio
`dispatch_ms / expert_compute_ms` is measured every forward pass and emitted
as `collective.comm_compute_overlap_ratio` in the telemetry record.

---

## Kernel Architecture: Triton Router

The router kernel (`_router_fwd_kernel`) is a single Triton JIT kernel that fuses:

1. `tokens [N, H] @ gate_w [H, E]` → `logits [N, E]`
2. `softmax(logits, dim=-1)` → `probs [N, E]`
3. `top_k(probs, K)` → `idx [N, K]`, `vals [N, K]`
4. `renormalize(vals)` → `combine_weights [N, K]`

All four operations execute in a single pass over the gating dimension, with
`(BLOCK_N=64, BLOCK_E=64)` tiles held in L1/SRAM (~49 KiB for H=1024).
Global loads are coalesced on the contiguous `H` dimension. Top-K uses
in-SRAM selection sort (K iterations, O(K×E)), which outperforms bitonic
sort for K ∈ {1,2,4} at the block sizes used here.

`K` is declared `tl.constexpr` in both the forward and backward kernel
signatures — this was a v0.3.2 bug fix. Without `constexpr`, `tl.static_range`
fails to unroll the top-K loop at Triton compile time on real GPU hardware,
causing a `CompilationError` that is not reproducible on CPU.

The backward kernel (`_router_bwd_kernel`) propagates `grad_w → grad_p → grad_l`
using the analytic softmax Jacobian:
`grad_l_i = p_i × (grad_p_i − Σⱼ(grad_p_j × p_j))`.
Validated against the fp64 PyTorch reference at `atol=rtol=1e-5` across 30
`(H, E, K)` configurations in `tests/test_kernels_numerics.py`.

On CPU or without Triton, `_reference_route_fp64` provides the identical
computation in fp64 PyTorch — used as the ground truth for all numeric tests.

---

## Token Life-Cycle in One MoE Layer

```
tokens [B, S, H]  →  reshape  →  flat [N, H]   (N = B×S)
        │
        ▼
 MoERouter (Triton kernel or CPU fp64 reference)
        │
        ├── idx  [N, K]              expert indices
        ├── w    [N, K]              renormalised combine weights
        └── dispatch_cnt [E]         tokens per expert (conservation checked)
        │
        ▼
 sort by expert_id → tokens_sorted [N×K, H]   (contiguous send regions)
        │
        ▼  all_to_all_dispatch ─── dedicated CUDA stream ──►  ┐
                                                               │
 expert FFN on this rank ◄── received [total_recv, H]         │  (overlap)
   _SwiGLUExpert: silu(gate(x)) × up(x) → down(·)            │
        │  ◄── expert_compute_ms measured here                 │
        │                                                      │
        └── expert_out [total_recv, H] ─► all_to_all_combine ─┘
                                          (waits for dispatch_event)
        │
        ▼
 weighted scatter → combined [N, H]
   combined[sort_order] += w[k] × slot[k]   for k in range(K)
        │
        ▼  NaN guard (assert)
        │
        ▼
 reshape → output [B, S, H]
```

---

## Checkpoint Architecture: Two-Tier Async

```
Training thread (foreground)
     │
     │  optimizer.step()
     │  harness.checkpoint(model, optimizer, step)
     ▼
 AsyncCheckpointer.save()  ─── non-blocking, returns immediately
     │
     ▼ (background thread pool, async_workers=4 default)
     │
     ├── Tier 1: LocalNVMeAdapter
     │    ├── D2H copy of model state dict to pinned host RAM
     │    ├── serialize to temp file (.tmp suffix)
     │    ├── fsync (O_DIRECT on Linux; fallback to buffered on other OS)
     │    ├── atomic rename: .tmp → final path
     │    └── prune oldest checkpoints (retain last `retention` steps)
     │
     └── Tier 2: S3Adapter (background, after Tier 1 completes)
          ├── boto3 multipart upload (or local file copy for file:// URIs)
          ├── write .meta.json (step, rank, ts, hostname, shard_hash)
          └── prune remote (keep last `retention` remote commits)
```

The training thread pays only the D2H copy cost (~tens of ms per shard).
All disk I/O and network I/O runs in the background. Atomic rename ensures
every checkpoint on NVMe is either fully present or absent — no partial reads.

After a node drop, `ElasticTrainerHarness.recover()` calls
`AsyncCheckpointer.load()` which reads from the NVMe tier (fast, local) rather
than S3 (slow, remote), keeping recovery time under 30 seconds even for large
shards.

---

## Fault Recovery Sequence

```
Node death detected (SIGTERM or health_check timeout)
        │
        ▼
ClusterStateMachine → RUNNING → DRAINING
        │  (drain in-flight micro-batches from 1F1B pipeline)
        ▼
AsyncCheckpointer → flush in-flight NVMe write
        │  (wait for background thread; atomic rename guarantees integrity)
        ▼
TorchElastic → restart dead rank, re-init process group
        │  (c10d rendezvous or etcd; new world_size = old - dead_ranks)
        ▼
ClusterStateMachine → DRAINING → RECOVERING
        │  new_ep_size = _largest_divisor_le(E, new_world_size)
        │  expert resharding: round-robin to surviving EP ranks
        ▼
AsyncCheckpointer → load(model, optimizer, latest_step, rank)
        │  (reads from NVMe tier; ~seconds even at 7B param scale)
        ▼
ClusterStateMachine → RECOVERING → RESUMED
        │  training continues with reduced world_size
        ▼
Background → S3 upload of post-recovery checkpoint
```

---

## Telemetry Architecture

Every training step emits one `StepRecord` to three sinks simultaneously:

| Sink | Format | Notes |
|---|---|---|
| JSONL file | newline-delimited JSON | all fields; thread-safe via `RLock`; rank 0 |
| TensorBoard | scalar summaries | `SummaryWriter`; rank 0 only |
| Prometheus | gauges on `/metrics` | optional; port configurable; 10 gauges |

WandB sink (v0.3): active when `WANDB_API_KEY` is set; all numeric fields logged
under section prefixes (`collective/dispatch_ms`, `routing/z_loss`, etc.).
`log_config()` records all hyperparameters as WandB config on run start.

v0.3 `collective` telemetry fields (new):
- `collective.expert_compute_ms` — wall-clock time of expert FFN compute per step
- `collective.comm_compute_overlap_ratio` — `dispatch_ms / expert_compute_ms`

v0.2 `routing` telemetry fields:
- `routing.expert_load_imbalance` — `max_tokens_per_expert / mean_tokens_per_expert`
- `routing.router_z_loss` — `mean(log_sum_exp(logits)²)`, auxiliary regulariser

---

## Key Design Invariants

Every invariant is asserted at runtime AND has a dedicated test:

| Invariant | Where enforced | Test |
|-----------|---------------|------|
| Token conservation: `sum(dispatch_cnt) == N×K` | `MoERouter.forward()` | `test_kernels.py::test_token_conservation` |
| No NaN in router output | `MoERouterFunction.forward()` | `test_kernels.py::test_no_nan_in_indices` |
| Index validity: `idx ∈ [0, E)` | `MoERouterFunction.forward()` | `test_kernels.py::test_index_validity` |
| Weight normalisation: `w.sum(-1) ≈ 1` | `MoERouterFunction.forward()` | `test_kernels_numerics.py` |
| No NaN in MoE layer output | `DistributedMoELayer.forward()` | `test_distributed_invariants.py` |
| Config valid at load time | `MoEConfig.from_yaml()` | `test_config.py` (34 tests) |
| Expert coverage after reshard | `ClusterStateMachine._reshard()` | `test_elastic_v02.py` |
| Checkpoint retention | `AsyncCheckpointer._prune()` | `test_elastic.py` |

---

## Module Size Budget (post v0.3.2 refactoring)

Per MOE_instructions v2.1: no file in `pkg/distributed/` should exceed ~450 lines.

| Module | Lines | Status |
|--------|------:|--------|
| `mesh.py` | 316 | ✅ |
| `tensor_parallel.py` | 308 | ✅ |
| `expert_parallel.py` | 231 | ✅ |
| `pipeline_parallel.py` | 378 | ✅ |
| `data_parallel.py` | 101 | ✅ |
| `moe_layer.py` | ~290 | ✅ |
| `parallel_mesh.py` (shim) | 68 | ✅ |

Each module is independently importable and testable without a GPU.

---

## Design Principles

**Measure, don't estimate.**
All telemetry values come from real measurements: CUDA events for collective
latency, `torch.cuda.memory_stats()` for peak memory, `time.perf_counter()` for
CPU-path timing, actual elapsed wall-clock time for step duration.
There are no placeholder or fabricated numbers in the runtime or documentation
(all GPU numbers in RESULTS.md come from `gpu_results.json`, a real T4 run).

**Fail-fast on invariants.**
Token conservation, index bounds, weight normalisation, config constraints, and
post-combine NaN checks are all asserted in the forward path and at config load
time. Errors fire at the offending layer with a descriptive message, not silently
further downstream as mysterious shape mismatches.

**Single correct collective per pattern.**
`RowParallelLinear` uses `all_reduce(SUM)` — the correct primitive for summing
partial matrix products across TP ranks. `ColumnParallelLinear` uses
`all_gather_into_tensor`. EP dispatch uses `all_to_all_single`. Sequence
Parallelism uses `scatter` (forward) and `all_gather` (backward). Each pattern
uses exactly one collective, correctly chosen.

**Single-process must always work.**
Every distributed primitive degrades gracefully to an identity operation at
`tp_size=1`, `pp_size=1`, `ep_size=1`, or when `dist` is not initialised.
This means 90% of the codebase can be developed and tested on a laptop with
no GPU and no multi-process setup.

**Link, don't duplicate.**
Documentation references source files and test functions rather than re-describing
behaviour inline. When behaviour changes, update one place.
