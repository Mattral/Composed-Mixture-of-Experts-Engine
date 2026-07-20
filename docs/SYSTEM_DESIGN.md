# System Design

**Version:** v0.3.3
**Last updated:** July 2026

This document describes the runtime system as it exists in the codebase.
Every claim here is grounded in `moe-engine/pkg/` and `moe-engine/train.py`
and can be verified by running the commands shown.

---

## System Scope

`moe-engine` is a distributed training runtime for MoE language models at
hyperscale. Its responsibilities:

- Correct sparse token routing (Triton kernel + invariant enforcement)
- 4D parallel tensor distribution (DP × TP × PP × EP)
- Expert-capacity enforcement with bounded token dropping (v0.3.3)
- Fault-tolerant checkpointing (async two-tier NVMe → S3, schema-versioned)
- Elastic recovery without operator intervention (evict → reshard → resume)
- Observable, accurate telemetry (JSONL + TensorBoard + Prometheus + WandB)
- Validated configuration (Pydantic v2, fails loudly and immediately on bad input)

It is explicitly **not** a model definition. A real model (e.g., Mixtral)
plugs into the runtime by implementing the same `nn.Module` interface as
`ToyMoEModel` in `pkg/models/moe.py`, or by registering itself with
`pkg.models.registry.register_model` and letting `train.py` build it from
`cfg.model.arch`.

---

## Package Layout (post v0.3.2 split)

The original `pkg/distributed/parallel_mesh.py` (1,165 lines, every
distributed concern in one file) was split into eight single-responsibility
modules. `parallel_mesh.py` itself is now a 68-line backward-compatibility
shim that re-exports everything below — existing imports of
`pkg.distributed.parallel_mesh.X` continue to work unchanged.

```
pkg/distributed/
├── mesh.py               ParallelTopology, build_topology, _CommStream
├── tensor_parallel.py    ColumnParallelLinear, RowParallelLinear
├── sequence_parallel.py  scatter_to_sequence_parallel, gather_from_sequence_parallel
├── expert_parallel.py    all_to_all_dispatch, all_to_all_combine
├── pipeline_parallel.py  PipelineStage, run_1f1b, run_1f1b_distributed
├── data_parallel.py      apply_fsdp2
├── router.py             MoERouterInterface (high-level wrapper over the kernel)
├── moe_layer.py          DistributedMoELayer, _SwiGLUExpert, capacity dropping
└── parallel_mesh.py      backward-compat shim only — new code should not import here
```

```
pkg/
├── kernels/moe_router.py   Triton fwd+bwd kernels, fp64 reference, MoERouter
├── models/
│   ├── moe.py              RMSNorm, ToyMoEBlock, ToyMoEModel
│   └── registry.py         register_model, build_model_from_config, ModelRegistry
├── elastic/fault_monitor.py  AsyncCheckpointer, ClusterStateMachine, schema versioning
├── telemetry/logger.py       StructuredLogger, StepRecord, PrometheusExporter, WandBSink
└── utils/
    ├── config.py           MoEConfig (Pydantic v2), load_config (legacy shim)
    └── mfu.py               compute_mfu, MFUAccountant
```

---

## Core Components

### `train.py` — Entrypoint

Responsibilities:

1. Parse `--config` via `pkg.utils.config.MoEConfig.from_yaml` (Pydantic v2 —
   validates every field at load time; raises `ConfigValidationError` with a
   field path and human-readable message on any invalid value, rather than
   surfacing as an opaque crash mid-training).
2. Bootstrap distributed process group (NCCL on GPU, Gloo on CPU).
3. Build `ParallelTopology` and clamp parallelism axes to actual world size.
4. Construct the model via `pkg.models.registry.build_model_from_config`
   (dispatches on `cfg.model.arch`, default `"toy_moe"` →
   `pkg.models.moe.ToyMoEModel`) and apply FSDP2 via `apply_fsdp2`.
5. Configure `StructuredLogger`, `MFUAccountant`, and `ElasticTrainerHarness`.
6. Resume from latest checkpoint if available (schema-version checked).
7. Run training loop: LR schedule, aux z-loss (if `z_loss_weight > 0`),
   gradient accumulation, clip, step.
8. Emit `StepRecord` per log interval with kernel, collective, routing,
   memory, and infra fields fully populated from real runtime measurements,
   including the v0.3.2 fields (`sparse_mfu`, `dead_expert_count`,
   `routing_efficiency`, `active_experts`) and the v0.3.3 field
   (`dropped_token_fraction`).
9. If `--profile`: write per-step benchmark JSON to `benchmarks/`.

---

### `pkg/utils/config.py` — Configuration System (Pydantic v2)

**`MoEConfig`** — root config, composed of six typed sub-configs:
`ModelConfig`, `TrainingConfig`, `ParallelismConfig`, `CheckpointConfig`,
`ElasticConfig`, `TelemetryConfig`. `pydantic>=2.0.0` is a **hard runtime
dependency** (not optional) — the config system's entire purpose is
catching invalid values before they reach training, so a silently-degraded
validator would be worse than an import error. If pydantic is missing, the
module raises `ImportError` immediately on import with installation
instructions, rather than falling back to an unvalidated shim.

Cross-field validators enforce:
- `top_k <= num_experts`
- `warmup_steps < max_steps`
- `min_nodes <= max_nodes`
- `hidden_dim % 8 == 0`
- `dtype ∈ {float32, bfloat16, float16}`
- `remote_uri` starts with `s3://` or `file://`

**`MoEConfig.from_yaml(path)`** — load + validate; raises
`ConfigValidationError` with the field path and constraint description on
failure (e.g. `[model] top_k (8) must be <= num_experts (4)`).

**`_coerce_env_value(val)`** — environment-variable override coercion.
Tries native `int()`/`float()` before falling back to `yaml.safe_load()`.
This exists because `yaml.safe_load("1e-5")` returns the **string**
`"1e-5"`, not the float, under YAML 1.1's stricter exponential-notation
grammar — a well-known PyYAML footgun that would otherwise silently break
`MOE_TRAINING__LEARNING_RATE=1e-5`-style overrides.

**`load_config(path)`** — legacy shim returning an object with `.raw`,
`.model`, `.training`, etc. dict-style access, plus `.typed()` to upgrade
to the modern `MoEConfig`. Preserved for backward compatibility; new code
should call `MoEConfig.from_yaml` directly.

---

### `pkg/distributed/mesh.py` — Topology and Device Mesh

**`ParallelTopology`** — immutable frozen dataclass holding:
- `world_size`, `rank`, `dp_size`, `tp_size`, `pp_size`, `ep_size`
- `dp_rank`, `tp_rank`, `pp_rank`, `ep_rank` (computed from global rank)
- `device` and `mesh` (PyTorch `DeviceMesh`, or `None` on single-rank CPU)

**`build_topology(...)`** — constructs topology. At `world_size=1` returns a
degenerate single-rank topology with no `DeviceMesh` (the entire CPU test
suite runs through this path). At `world_size>1` creates a `DeviceMesh` via
`init_device_mesh` (PyTorch 2.5+).

**`_CommStream`** — singleton high-priority CUDA stream per device, used by
`expert_parallel.py` to overlap EP dispatch/combine with expert FFN compute
on the default stream.

---

### `pkg/distributed/expert_parallel.py` — EP Collectives

**`all_to_all_dispatch` / `all_to_all_combine`** — wrappers around
`dist.all_to_all_single` that record CUDA event timing on `_CommStream`.
Return `(output, event_or_None, latency_ms)`. At `ep_size=1` or without
`dist` initialised, return the input tensor directly — no collective, zero
overhead, which is why the entire test suite runs on a single CPU process.

---

### `pkg/distributed/tensor_parallel.py` + `sequence_parallel.py`

**`ColumnParallelLinear`** — weight shape `[F // tp_size, H]` per rank.
Forward: local matmul → `all_gather_into_tensor` across TP group → `[F]`.
At `tp_size=1`: identity (no collective).

**`RowParallelLinear`** — weight shape `[H, F // tp_size]` per rank.
Forward: slice input to `[..., F // tp_size]` → local matmul →
`all_reduce(SUM)`. The `all_reduce` is the correct and only collective
needed: each rank computed a partial dot product, which must be summed.

**`scatter_to_sequence_parallel` / `gather_from_sequence_parallel`** —
extracted into their own module in v0.3.2 (previously inline in
`tensor_parallel.py`). SP helpers that shard/reconstruct the sequence
dimension across the TP group. No-op at `tp_size=1`. The `next_weight`
parameter (v0.3) fuses the backward all-gather with the subsequent
projection matmul, replacing `all_gather → matmul` with `matmul →
all_reduce`, halving the number of collectives per SP layer.

---

### `pkg/distributed/pipeline_parallel.py` — `PipelineStage`

- `forward_step(mb)` — applies `self.module` if set, else passthrough
- `run_1f1b(micro_batches)` — single-process 1F1B scheduling; fast-path
  used by the unit test suite
- `run_1f1b_distributed(micro_batches, loss_fn)` — full multi-process 1F1B
  with `dist.send`/`dist.recv` on the PP process group; activation and
  gradient tagging via an explicit `mb_index` header; three-phase
  (warmup → steady-state → drain) schedule. Verified by 2-rank `mp.spawn`
  tests.

---

### `pkg/distributed/router.py` — High-Level Router Interface (v0.3.2)

**`MoERouterInterface`** — thin wrapper around the kernel-level `MoERouter`
that:
- Validates input shape (`ValueError` if not 2D, or hidden-dim mismatch)
- Enforces the token-conservation invariant explicitly with a descriptive
  `RuntimeError` if violated
- Exposes `capacity_budget(num_tokens)` — `ceil(capacity_factor * N * K / E)`
- Returns a `RouterStats` dataclass: `expert_indices`, `combine_weights`,
  `dispatch_counts`, `load_imbalance`, `router_z_loss`, `used_triton`,
  `kernel_ms`, `tokens_per_expert_mean`, `tokens_per_expert_std`

This is the boundary between the distributed layer and the kernel layer —
`DistributedMoELayer` calls the kernel-level `MoERouter` directly (for
maximum performance), while `MoERouterInterface` is available for callers
that want the validated, documented, telemetry-rich interface without
touching kernel internals.

---

### `pkg/distributed/moe_layer.py` — `DistributedMoELayer`

The primary MoE building block:
- Owns `len(local_expert_ids)` `_SwiGLUExpert` modules (each rank owns
  `E // ep_size` experts, remainder assigned round-robin).
- Routes tokens via `MoERouter`, dispatches via `all_to_all`, computes
  expert FFN, combines, records `last_dispatch_ms`, `last_combine_ms`,
  `last_expert_compute_ms`, `last_overlap_ratio`
  (`dispatch_ms / expert_compute_ms`).
- Enforces post-combine NaN check.

**`_SwiGLUExpert`** — two-layer SwiGLU FFN:
- `w_gate`: `ColumnParallelLinear(H → F)` — splits output features
- `w_up`:   `ColumnParallelLinear(H → F)` — splits output features
- `w_down`: `RowParallelLinear(F → H)` — splits input features, all_reduces
- Forward: `w_down(silu(w_gate(x)) × w_up(x))`
- Both gate and up are `ColumnParallel` so the elementwise multiply occurs
  in shard space `[F // tp_size]`. `w_down` all_reduces once at the output.

**Expert capacity dropping (v0.3.3)** — `capacity_dropping: bool = False`
constructor parameter (default off, zero behavior change unless enabled).
When `True`, enforces a hard per-expert token budget following Switch
Transformer / GShard semantics:

- `_cumcount(groups)` — vectorised "position of appearance within group"
  primitive (stable sort + `cummax`, no Python loops, no
  `scatter_reduce` version dependencies).
- `compute_capacity_drop_mask(idx, num_experts, capacity)` — for each
  top-k slot independently, keeps the first `capacity` tokens (by order
  of appearance in the batch) that selected each expert and drops the
  remainder (zero combine weight for that slot).
- `capacity = ceil(capacity_factor * N * K / E)`.
- Reports `last_dropped_token_fraction` on the layer, surfaced in
  telemetry as `StepRecord.dropped_token_fraction`.

25 dedicated tests in `tests/test_capacity_dropping.py` cover the
`_cumcount` primitive, the drop mask, and full-layer integration
(default-off unchanged behaviour, tight-capacity bounded drops, backward
pass correctness, determinism).

---

### `pkg/models/` — Model Definitions and Registry

**`pkg/models/moe.py`**: `RMSNorm`, `ToyMoEBlock` (norm + MoE layer +
residual), `ToyMoEModel` (embed → N × block → norm → lm_head).

**`pkg/models/registry.py`** (v0.3.2): decorator-based model registry
decoupling `train.py` from any specific architecture.

- `@register_model("name")` — decorator; registers a class under a unique
  name, raises `ValueError` (naming the existing holder) on duplicate
  registration.
- `build_model_from_config(cfg, topology, arch=None)` — factory; dispatches
  to the class registered under `arch` (or `cfg.model.arch`, default
  `"toy_moe"`); raises `KeyError` listing all available names if the arch
  is unregistered.
- `ModelRegistry` — class-level API (`.register`, `.get`, `.list`) for
  programmatic access.

`ToyMoEModel` is registered as `"toy_moe"` on import of `pkg.models`.

---

### `pkg/kernels/moe_router.py` — Router Kernel

**`MoERouter(hidden_dim, num_experts, top_k)`** — `nn.Module` wrapper:
- `gate_w: Parameter[H, E]` — learnable gating matrix
- `forward(tokens) → (topk_idx, topk_w, dispatch_cnt)` — runs kernel,
  asserts token conservation, computes `RouterProfile`
- `last_profile: RouterProfile` — populated every forward; includes
  `sram_bytes_per_block`, `achieved_bandwidth_gbps`, `kernel_ms`,
  `used_triton`, `tokens_per_expert_mean/std`, `expert_load_imbalance`
  (`max_load / mean_load`), `router_z_loss` (Switch-Transformer auxiliary
  signal)

**`MoERouterFunction`** — `torch.autograd.Function`:
- `forward`: Triton kernel (GPU + Triton) or fp64 reference (CPU / fallback)
- `backward`: Triton backward kernel (GPU) or analytic fp64 (CPU / fallback)
- Both paths validated at `atol=rtol=1e-5` against fp64 reference across
  30 `(H, E, K)` configurations
- `K` is declared `tl.constexpr` in both kernel signatures — this was a
  v0.3.2 bug fix; without it, `tl.static_range` fails to compile on real
  GPU hardware (not reproducible on the CPU-only reference path, which
  masked it until T4 validation).

---

### `pkg/elastic/fault_monitor.py` — Fault Tolerance

**`CHECKPOINT_SCHEMA_VERSION = 2`** (v0.3.2) — every checkpoint's
`.meta.json` embeds a schema version. `_check_schema_compatibility(meta)`
logs at `info` level for older-but-compatible checkpoints (missing fields
default gracefully) and at `warning` level for newer-than-runtime
checkpoints (potential field loss). Schema v2 added `moe_engine_version`
(read from `pkg.__version__`, the single source of truth — not a hardcoded
literal) and `torch_version` for compatibility diagnostics.

**`LocalNVMeAdapter`** — file-backed key-value store: chunked writes
(256 MB), `O_DIRECT` attempt with buffered fallback, atomic rename
(`.tmp` → final path).

**`S3Adapter`** — boto3-backed remote tier with multipart upload.

**`AsyncCheckpointer`** — background-thread checkpoint manager:
- `save(model, optim, step, rank, extra_meta=None)`: enqueues a save;
  worker thread commits to both local and remote adapters; records
  `last_commit_ms`
- `load(model, optim, step, rank)`: synchronous load from local tier;
  runs schema compatibility check
- `latest_step()`: discovers latest available step
- Retention pruning: after each commit, deletes steps older than
  `retention`

**`ClusterStateMachine`** — rank health tracking: `RUNNING → DRAINING →
RECOVERING → RESUMED`. `reshard(new_topo, num_experts)` computes
expert→rank assignment using `_largest_divisor_le(E, new_ep_size)`.

**`ElasticTrainerHarness`** — top-level driver: `install_signal_handlers`,
`checkpoint`, `health_check`, `recover`, `shutdown`.

---

### `pkg/telemetry/logger.py` — Structured Telemetry

**`StepRecord`** — dataclass for one training step. v0.3.3 fields (all
also injected into the `routing` dict for backward-compatible JSON):

```python
StepRecord(
    step, loss, mfu, tokens_per_sec, wall_clock_ms,
    kernel={...}, collective={...}, memory={...}, infra={...},
    routing={
        "expert_load_imbalance": ...,      # v0.2
        "router_z_loss": ...,              # v0.2
        "sparse_mfu": ...,                 # v0.3.2: mfu * (K/E)
        "dead_expert_count": ...,          # v0.3.2
        "routing_efficiency": ...,         # v0.3.2
        "active_experts": ...,             # v0.3.2
        "dropped_token_fraction": ...,     # v0.3.3: capacity-drop rate
    },
    sparse_mfu=..., dead_expert_count=..., routing_efficiency=...,
    active_experts=..., dropped_token_fraction=...,  # typed field access
)
```

**`StructuredLogger`** — thread-safe four-sink emitter: JSONL (RLock-
protected), TensorBoard (rank 0 only), Prometheus (`PrometheusExporter`,
15 gauges), WandB (`WandBSink`, active when `WANDB_API_KEY` is set).

**`PrometheusExporter`** gauges include the v0.3.3 additions
`moe_dropped_token_fraction`, `moe_sparse_mfu`, `moe_dead_expert_count`,
`moe_routing_efficiency`, `moe_active_experts` alongside the v0.2/v0.3
gauges (`moe_step_loss`, `moe_mfu`, `moe_tokens_per_sec`,
`moe_all_to_all_dispatch_ms`, `moe_all_to_all_combine_ms`,
`moe_peak_memory_gb`, `moe_expert_load_imbalance`, `moe_router_z_loss`,
`moe_expert_compute_ms`, `moe_comm_compute_overlap_ratio`). Graceful
no-op when `prometheus_client` is not installed.

---

### `pkg/utils/mfu.py` — MFU Accounting

**`compute_mfu(...)`**:
```
MFU = (2 × T × P_dense + 2 × T × (K/E) × P_expert) / (step_s × world × peak_TFLOPS)
```
Activation recompute adds a third forward pass worth of FLOPs (3× multiplier).

**`MFUAccountant`** — streaming tracker: `start_step()`/`end_step(tokens)`,
`running_mfu`, `smoothed_mfu` (sliding window, default 50 steps),
`summary_str()`.

---

## Dataflow: One Training Step

```
Input IDs [B, S]
    │ embed
    ▼
x [B, S, H]
    │ ToyMoEBlock (repeated num_layers)
    │
    ├── RMSNorm(x)
    │
    └── DistributedMoELayer
         │
         ├── MoERouter → (idx, w, dispatch_cnt)
         │    ├── assert token_conservation
         │    └── populate RouterProfile (load_imbalance, z_loss)
         │
         ├── [v0.3.3, optional] capacity drop mask → zero weight on overflow
         │
         ├── sort tokens by expert_id
         │
         ├── all_to_all_dispatch (EP, dedicated stream)
         │
         ├── _SwiGLUExpert per local expert:
         │    w_down(silu(w_gate(x)) × w_up(x))
         │
         ├── all_to_all_combine (EP, dedicated stream, waits on dispatch event)
         │
         └── assert no NaN → weighted sum → x_out [B, S, H]
    │
    ▼
RMSNorm → lm_head → logits [B, S, V]
    │
    ▼
cross_entropy(logits, targets) + z_loss_weight × router_z_loss (if enabled)
    │ backward → gradient clip → optimizer step
    │
    ▼
StepRecord → StructuredLogger (JSONL + TensorBoard + Prometheus + WandB)
    │
    └── AsyncCheckpointer.save (background thread, if ckpt_interval)
```

---

## Validation and Testing

Verify with: `pytest tests/ -m cpu -k "not (2rank or multiprocess or distributed_invariants)" --ignore=tests/test_chaos.py --ignore=tests/test_smoke_e2e.py`

| Test file | What it verifies |
|---|---|
| `test_config.py` | Pydantic `MoEConfig` validation, env overrides, legacy shim (38 tests) |
| `test_kernels.py` | Router fwd/bwd shapes, token conservation, NaN checks, `K: tl.constexpr` |
| `test_kernels_numerics.py` | 30 configs, Triton vs fp64 `atol=rtol=1e-5` |
| `test_routing_quality.py` | load_imbalance math, z_loss invariants, `RouterProfile` |
| `test_router.py` | `MoERouterInterface`: construction, invariants, `capacity_budget`, backward pass (33 tests) |
| `test_registry.py` | Model registry: registration, duplicate detection, factory dispatch (20 tests) |
| `test_capacity_dropping.py` | `_cumcount`, `compute_capacity_drop_mask`, layer integration (22 tests) |
| `test_tensor_parallel.py` | Column/Row shape+grad+dtype+numerical; 2-rank `mp.spawn` |
| `test_pipeline_parallel.py` | 1F1B schedule; single-process invariants; 2-rank `mp.spawn` PP |
| `test_sequence_parallel_v03.py` | SP fused `next_weight` path; 2-rank `mp.spawn` SP |
| `test_distributed.py` | MoE layer fwd/bwd shapes, topology construction |
| `test_distributed_invariants.py` | 4-process Gloo: token conservation, NaN guard |
| `test_elastic.py` / `test_elastic_v02.py` | NVMe round-trip, reshard edge cases, schema versioning |
| `test_mfu.py` / `test_mfu_v02.py` | MFU formula, `MFUAccountant`, sparse fraction scaling |
| `test_telemetry.py` | JSON completeness (`REQUIRED_KEYS`), thread safety, v0.3.3 fields |
| `test_mock_dist.py` | `MockTopology`/`MockDistEnv` — collective simulation without multi-process |
| `test_properties.py` | Hypothesis property-based tests: conservation, ownership, config invariants |
| `test_smoke_e2e.py` | Full `train.py` loop, all telemetry envelope keys, S3 mock |
| `test_chaos.py` | Scenario B ✅ 100%; Scenario A ⚠️ ~85% |

**Total: 348 passing tests** (21 test files), 1 skip (Triton GPU path, no
CUDA in this environment), 1 `xfail` (documented statistical edge case in
`test_routing_quality.py`, seed=2). 0 lint errors, 0 format violations
(`ruff check` / `ruff format --check`).

---

## Known Limitations (v0.3.3)

| Limitation | Root Cause | Planned Fix |
|---|---|---|
| Chaos Scenario A ~85% pass rate | Gloo `connectFullMesh` socket race after SIGKILL | v0.4 (NCCL, needs GPU) |
| No multi-node banchmark data yet | Two single-GPU architectures now validated (T4, A100); no multi-node/multi-GPU cluster data yet | v0.4 |
| SP `sequence_length % tp_size == 0` required | No padding for non-divisible lengths | v0.4 |
| PP `run_1f1b_distributed` not in chaos tests | `_chaos_worker.py` uses a dense model | v0.4 |
| No Nsight/CUPTI roofline | Requires GPU hardware | v0.4 |
| Capacity **dropping** implemented; capacity **re-routing** (to next-best expert) not yet | Re-routing needs real EP bandwidth data to tune the fallback policy | v0.4 |
| Package ships as one `pyproject.toml`, not split into standalone installable sub-libraries (router kernel, async checkpointer, telemetry) | Deliberate scope decision, lowest priority (P2.3) | When resources allow |
