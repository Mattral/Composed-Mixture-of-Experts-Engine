# System Design

**Version:** v0.3  
**Last updated:** June 2026

This document describes the runtime system as it exists in the codebase.
Every claim here is grounded in `moe-engine/pkg/` and `moe-engine/train.py`.

---

## System Scope

`moe-engine` is a distributed training runtime for MoE language models at
hyperscale. Its responsibilities:

- Correct sparse token routing (Triton kernel + invariant enforcement)
- 4D parallel tensor distribution (DP × TP × PP × EP)
- Fault-tolerant checkpointing (async two-tier NVMe → S3)
- Elastic recovery without operator intervention (evict → reshard → resume)
- Observable, accurate telemetry (JSONL + TensorBoard + Prometheus)

It is explicitly **not** a model definition. A real model (e.g., Mixtral)
plugs into the runtime by implementing the same `nn.Module` interface as
`_ToyMoEBlock` in `train.py`.

---

## Core Components

### `train.py` — Entrypoint

Responsibilities:

1. Parse `--config` via `pkg.utils.config.load_config`.
2. Bootstrap distributed process group (NCCL on GPU, Gloo on CPU).
3. Build `ParallelTopology` and clamp parallelism axes to actual world size.
4. Construct `_ToyMoEModel` and apply FSDP2 via `apply_fsdp2`.
5. Configure `StructuredLogger`, `MFUAccountant`, and `ElasticTrainerHarness`.
6. Resume from latest checkpoint if available.
7. Run training loop: LR schedule, gradient accumulation, clip, step.
8. Emit `StepRecord` per log interval with kernel, collective, routing, memory,
   and infra fields fully populated from real runtime measurements.
9. If `--profile`: write per-step benchmark JSON to `benchmarks/`.

v0.2: LR schedule, gradient accumulation, routing quality fields, `--profile`.
v0.3: `--wandb-project` flag, `--no-wandb` flag, `logger.log_config(cfg.raw)` after init.

---

### `pkg/distributed/parallel_mesh.py` — Topology and Communication

**`ParallelTopology`** — immutable dataclass holding:
- `world_size`, `rank`, `dp_size`, `tp_size`, `pp_size`, `ep_size`
- `dp_rank`, `tp_rank`, `pp_rank`, `ep_rank` (computed from global rank)
- `device` and `mesh` (PyTorch DeviceMesh, or None on single-rank CPU)

**`build_topology(...)`** — constructs topology. At `world_size=1` returns a
degenerate single-rank topology with no DeviceMesh (all tests run CPU-only).
At `world_size>1` creates a DeviceMesh via `init_device_mesh` (PyTorch 2.5+).

**`_CommStream`** — singleton CUDA stream per process for EP collectives.
High-priority stream; created lazily on first GPU use.

**`all_to_all_dispatch` / `all_to_all_combine`** — wrappers around
`dist.all_to_all_single` that optionally record CUDA event timing. Returns
`(output, event_or_None, latency_ms)`. At `ep_size=1` or without dist,
returns the input tensor directly (no collective, zero overhead).

**`DistributedMoELayer`** — the primary MoE building block:
- Owns `len(local_expert_ids)` `_SwiGLUExpert` modules.
- Routes tokens via `MoERouter`, dispatches via `all_to_all`, computes
  expert FFN, combines, records `last_dispatch_ms`, `last_combine_ms`.
- v0.3: records `last_expert_compute_ms` and `last_overlap_ratio`
  (`dispatch_ms / expert_compute_ms`) for comm/compute overlap telemetry.
- Enforces post-combine NaN check.

**`_SwiGLUExpert`** — two-layer SwiGLU FFN:
- `w_gate`: `ColumnParallelLinear(H → F)` — splits output features
- `w_up`:   `ColumnParallelLinear(H → F)` — splits output features
- `w_down`: `RowParallelLinear(F → H)` — splits input features, all_reduces
- Forward: `w_down(silu(w_gate(x)) × w_up(x))`
- Both gate and up are ColumnParallel so the elementwise multiply occurs in
  shard space `[F // tp_size]`. w_down all_reduces once at the output.

**`ColumnParallelLinear`** — weight shape `[F // tp_size, H]` per rank.
Forward: local matmul → `all_gather_into_tensor` across TP group → `[F]`.
At `tp_size=1`: identity (no collective).

**`RowParallelLinear`** — weight shape `[H, F // tp_size]` per rank.
Forward: slice input to `[..., F // tp_size]` → local matmul → `all_reduce(SUM)`.
The all_reduce is the correct and only collective needed: each rank computed a
partial dot product, which must be summed across the group.

**`scatter_to_sequence_parallel` / `gather_from_sequence_parallel`** —
SP helpers that shard/reconstruct the sequence dimension across the TP group.
No-op at `tp_size=1`.

**`PipelineStage`** — lightweight 1F1B schedule implementation:
- `forward_step(mb)` — applies `self.module` if set, else passthrough
- `backward_step(grad)` — passthrough (used by `run_1f1b` test shim)
- `run_1f1b(micro_batches)` — single-process 1F1B scheduling; fast-path for tests
- `run_1f1b_distributed(micro_batches, loss_fn)` — **v0.3**: full multi-process
  1F1B with `dist.send`/`dist.recv` on PP group; activation tagging; 3-phase

**`apply_fsdp2`** — wraps every non-`DistributedMoELayer` module with
`fully_shard` along the `dp` mesh axis. Expert weights are intentionally excluded
(they are EP-sharded, not DP-sharded). Supports `MixedPrecisionPolicy` for bf16.

---

### `pkg/kernels/moe_router.py` — Router Kernel

**`MoERouter(hidden_dim, num_experts, top_k)`** — `nn.Module` wrapper:
- `gate_w: Parameter[H, E]` — learnable gating matrix
- `forward(tokens) → (topk_idx, topk_w, dispatch_cnt)` — runs kernel,
  asserts token conservation, computes `RouterProfile`
- `last_profile: RouterProfile` — populated every forward; includes:
  - `sram_bytes_per_block`, `achieved_bandwidth_gbps`, `kernel_ms`
  - `used_triton` (bool)
  - `tokens_per_expert_mean`, `tokens_per_expert_std`
  - **`expert_load_imbalance`** (v0.2): `max_load / mean_load`
  - **`router_z_loss`** (v0.2): Switch-Transformer auxiliary signal

**`MoERouterFunction`** — `torch.autograd.Function`:
- `forward`: Triton kernel (GPU + Triton) or fp64 reference (CPU / fallback)
- `backward`: Triton backward kernel (GPU) or analytic fp64 (CPU / fallback)
- Both paths validated at `atol=rtol=1e-5` against fp64 reference

**`_compute_load_imbalance(dispatch_cnt)`** — `max / mean`; standalone
testable; returns 1.0 on zero counts (no division by zero).

**`_compute_router_z_loss(logits)`** — `mean(logsumexp(logits)²)`;
standalone testable; non-negative by construction.

---

### `pkg/elastic/fault_monitor.py` — Fault Tolerance

**`LocalNVMeAdapter`** — file-backed key-value store:
- `put(key, data)`: write in 256 MB chunks; attempt `O_DIRECT`; atomic rename
- `get(key)`: read and return bytes
- `list(prefix)`: enumerate keys under prefix
- `delete(key)`: remove file

**`S3Adapter`** — boto3-backed remote tier with multipart upload.

**`AsyncCheckpointer`** — background-thread checkpoint manager:
- `save(model, optim, step, rank)`: enqueues a `SHARDED_STATE_DICT` save;
  worker thread commits to both local and remote adapters; records
  `last_commit_ms`
- `load(model, optim, step, rank)`: synchronous load from local tier
- `latest_step()`: discovers latest available step from local tier key listing
- `shutdown(drain)`: waits for queue to empty if `drain=True`
- Retention pruning: after each commit, deletes steps older than `retention`

**`ClusterStateMachine`** — rank health tracking:
- States: `RUNNING → DRAINING → RECOVERING → RESUMED`
- `heartbeat()`: returns list of dead ranks (from dist barrier or timeout)
- `alive_ranks()`: current surviving rank list
- `begin_recovery()`: transitions to DRAINING
- `reshard(new_topo, num_experts)`: computes expert→rank assignment using
  `_largest_divisor_le(E, new_ep_size)` for valid EP size; returns plan dict;
  transitions to RECOVERING
- `mark_resumed()`: transitions to RESUMED

**`_largest_divisor_le(n, k)`** — finds largest divisor of `n` that is ≤ `k`.
Handles prime `n`, `k > n`, `k = 1`, and all remainder cases. 12 parametrised
edge cases in `test_elastic_v02.py`.

**`ElasticTrainerHarness`** — top-level driver:
- `install_signal_handlers()`: registers SIGTERM/SIGUSR1 to drain checkpoint
  queue before exit; safe no-op when called from non-main thread
- `checkpoint(model, optim, step)`: delegates to `AsyncCheckpointer.save`
- `health_check()`: calls `ClusterStateMachine.heartbeat`
- `recover(model, optim, num_experts)`: runs full reshard → reload cycle
- `shutdown()`: drains checkpoint queue, destroys process groups

---

### `pkg/telemetry/logger.py` — Structured Telemetry

**`StepRecord`** — dataclass for one training step:

```python
StepRecord(
    step, loss, mfu, tokens_per_sec, wall_clock_ms,
    kernel={...},      # sram_bytes, bw_gbps, used_triton, expert load stats
    collective={...},  # all_to_all_dispatch_ms, all_to_all_combine_ms
    memory={...},      # peak_allocated_gb, reserved_gb, leak_delta_gb
    infra={...},       # async_ckpt_commit_ms, active_nodes, ep_world_size, lr
    routing={...},     # expert_load_imbalance, router_z_loss  [v0.2]
    # collective also includes in v0.3:
    # expert_compute_ms, comm_compute_overlap_ratio
)
```

**`StructuredLogger`** — thread-safe four-sink emitter:
- JSONL: RLock-protected file handle; `also_stdout` option for rank 0
- TensorBoard: `SummaryWriter` (rank 0 only)
- Prometheus: optional `PrometheusExporter` (10 gauges: +expert_compute_ms, +overlap_ratio)
- **WandB** (v0.3): `WandBSink`; active when `WANDB_API_KEY` set; logs all
  numeric fields under section prefixes (`collective/dispatch_ms`, etc.)
- `log_config(cfg)`: forwards hyperparameters to `wandb.config.update`
- `close()`: idempotent; calls `WandBSink.finish()`

**`PrometheusExporter`** — 8 gauges on `/metrics`:
`moe_step_loss`, `moe_mfu`, `moe_tokens_per_sec`,
`moe_all_to_all_dispatch_ms`, `moe_all_to_all_combine_ms`,
`moe_peak_memory_gb`, `moe_expert_load_imbalance`, `moe_router_z_loss`.
Graceful no-op when `prometheus_client` is not installed.

---

### `pkg/utils/mfu.py` — MFU Accounting

**`compute_mfu(...)`** — scalar MFU in [0, 1]:
```
MFU = (2 × T × P_dense + 2 × T × (K/E) × P_expert) / (step_s × world × peak_TFLOPS)
```
Activation recompute adds a third forward pass worth of FLOPs (3× multiplier).

**`compute_mfu_detailed(...)`** — returns `MFUResult` dataclass with:
`achieved_tflops`, `peak_tflops`, `mfu`, `step_ms`, `tokens_per_sec`,
`flops_dense`, `flops_sparse`.

**`MFUAccountant`** — streaming tracker:
- `start_step()` / `end_step(tokens)` — wall-clock timing per step
- `running_mfu` — cumulative average
- `smoothed_mfu` — sliding-window average (configurable window, default 50)
- `summary_str()` — human-readable one-liner for console

---

## Dataflow: One Training Step

```
Input IDs [B, S]
    │ embed
    ▼
x [B, S, H]
    │ _ToyMoEBlock (repeated num_layers)
    │
    ├── _RMSNorm(x)
    │
    └── DistributedMoELayer
         │
         ├── MoERouter → (idx, w, dispatch_cnt)
         │    ├── assert token_conservation
         │    └── populate RouterProfile (load_imbalance, z_loss)  [v0.2]
         │
         ├── sort tokens by expert_id
         │
         ├── all_to_all_dispatch (EP, dedicated stream)
         │
         ├── _SwiGLUExpert per local expert:
         │    w_down(silu(w_gate(x)) × w_up(x))
         │    both gate + up: ColumnParallel (shard F // tp_size)
         │    w_down: RowParallel (all_reduce → full H)
         │
         ├── all_to_all_combine (EP, dedicated stream, waits on dispatch event)
         │
         └── assert no NaN → weighted sum → x_out [B, S, H]
    │
    ▼
_RMSNorm → lm_head → logits [B, S, V]
    │
    ▼
cross_entropy(logits, targets) / grad_accum
    │ backward
    │ gradient clip
    │ optimizer step
    │
    ▼
StepRecord → StructuredLogger (JSONL + TensorBoard + Prometheus)
    │
    └── AsyncCheckpointer.save (background thread, if ckpt_interval)
```

---

## Validation and Testing

| Test file | What it verifies |
|---|---|
| `test_kernels.py` | Router fwd/bwd shapes, token conservation, NaN checks |
| `test_kernels_numerics.py` | 30 configs, Triton vs fp64 `atol=rtol=1e-5` |
| `test_routing_quality.py` | load_imbalance math, z_loss invariants, RouterProfile |
| `test_tensor_parallel.py` | Column/Row shape+grad+dtype+numerical; **2-rank mp.spawn** |
| `test_pipeline_parallel.py` | 1F1B schedule; single-process invariants; 2-rank mp.spawn PP (v0.3) |
| `test_sequence_parallel_v03.py` | SP fused next_weight path; 2-rank mp.spawn SP (v0.3) |
| `test_distributed.py` | MoE layer fwd/bwd shapes, topology construction |
| `test_distributed_invariants.py` | 4-process Gloo: token conservation, NaN guard |
| `test_elastic.py` | NVMe round-trip, chunked write, async save/load, retention |
| `test_elastic_v02.py` | Reshard edge cases, file-URI tier, harness round-trip |
| `test_mfu.py` | MFU formula, sparse fraction, backward compat |
| `test_mfu_v02.py` | MFUAccountant, MFUResult breakdown, smoothed window |
| `test_telemetry.py` | JSON completeness, thread safety (100 concurrent), routing fields |
| `test_telemetry.py` | JSON completeness, thread safety, WandB mock (v0.3) |
| `test_smoke_e2e.py` | Full train.py loop, all v0.3 envelope keys incl. overlap, S3 mock |
| `test_chaos.py` | Scenario B ✅; Scenario A ⚠️ ~85% |

**Total: 148 test functions. 33 Python files. 0 syntax errors.**

---

## Known Limitations (v0.3)

| Limitation | Root Cause | Planned Fix |
|---|---|---|
| Chaos Scenario A ~85% pass rate | Gloo `connectFullMesh` socket race after SIGKILL | v0.4 (NCCL, needs GPU) |
| No real multi-node benchmark data | Requires sustained cluster access | v0.4 |
| SP `sequence_length % tp_size == 0` required | No padding for non-divisible lengths | v0.4 |
| PP `run_1f1b_distributed` not in chaos tests | `_chaos_worker.py` uses dense model | v0.4 |
| No Nsight/CUPTI roofline | Out-of-scope for current cycle | v0.4 |
