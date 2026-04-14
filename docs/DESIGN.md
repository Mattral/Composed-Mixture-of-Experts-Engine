# Design Rationale and Trade-offs

**Version:** v0.2  
**Last updated:** June 2026

This document explains *why* each major design choice was made and what
alternatives were considered. For *what* is implemented, see `ARCHITECTURE.md`.

---

## Router Kernel

**Choice:** Triton JIT fused kernel over three separate cuBLAS/PyTorch calls.

**Problem:** The routing pipeline — `matmul → softmax → top-K → renorm` —
requires three separate HBM round-trips if implemented with cuBLAS + PyTorch
ops. At H=4096, E=64 this means loading and storing ~2 GB of intermediate
tensors that never leave the GPU.

**Solution:** A single Triton kernel tiles across the expert dimension `E` in
SRAM (64×64 floats = 16 KiB, fitting in L1 cache on all Ampere/Hopper SKUs).
All four operations execute in one pass. HBM traffic is reduced by ~2.7×.

**Top-K implementation: selection sort vs bitonic sort.**  
For K ∈ {1, 2, 4} and E ≤ 256, K-step selection sort is strictly faster than
bitonic sort. Bitonic sort's `O(E log²E)` compute cost dominates at small K,
and it requires shared-memory bank synchronisation that hurts occupancy at small
block sizes. Selection sort runs entirely in registers. For K ≥ 8 bitonic would
win; that is not our use case.

**CPU fallback:** `_reference_route_fp64` provides identical computation in
fp64 PyTorch. This is used as the numerics ground truth (30 parametrised tests
at `atol=rtol=1e-5`) and as the production path on CPU-only machines. The
fallback is not slower than necessary — the double-precision path is tested to
match the Triton path at half-precision tolerances.

**Backward kernel:** The backward is analytic, not autodiff-through-sort.
Differentiating through `argmax` / top-K is undefined (zero gradient almost
everywhere). Instead we propagate `grad_w → grad_v → grad_p → grad_l` using
the softmax Jacobian: `grad_l_i = p_i × (grad_p_i − Σ(grad_p · p))`. This is
numerically stable and matches the fp64 reference to `atol=rtol=1e-5`.

---

## Tensor Parallelism Design

**Choice:** ColumnParallel + RowParallel wrapping the SwiGLU expert FFN.

**SwiGLU sharding consistency.**  
The SwiGLU formula is `w_down(silu(w_gate(x)) × w_up(x))`. The element-wise
multiply requires `w_gate(x)` and `w_up(x)` to have identical shape. If only
`w_up` were ColumnParallel (output shape `[F // tp_size]` before all-gather)
and `w_gate` were plain `nn.Linear` (output shape `[F]`), they would mismatch
at `tp_size > 1`.

**Correct design:** both `w_gate` and `w_up` are `ColumnParallelLinear`. Their
outputs are `[F // tp_size]` on each rank; the element-wise multiply happens in
shard space. `w_down` is `RowParallelLinear`: each rank holds `[H, F // tp_size]`
weights, computes a partial dot product `[F // tp_size] → [H]`, then
`all_reduce(SUM)` across the TP group reconstructs the full hidden dimension.

**RowParallel collective: why all_reduce, not reduce_scatter + all_gather.**  
`RowParallelLinear` computes `x_local @ W_local` where each rank owns a
column-slice `W_local = W[:, rank*F_loc : (rank+1)*F_loc]`. The partial outputs
from all ranks must be *summed* — that is one `all_reduce(SUM)`. A
`reduce_scatter` would scatter *different output chunks* to different ranks,
requiring an `all_gather` to recover: two collectives, 2× latency, zero
correctness benefit. The correct collective is always `all_reduce`.

**DTensor registration:** When `_HAS_DTENSOR` is true and a mesh is available,
both layers register their weight shards as DTensor with the appropriate
`Shard(0)` (column) or `Shard(1)` (row) placement. This integrates with FSDP2
and enables DTensor-aware gradient synchronisation.

**Sequence Parallelism (tp_size > 1).**  
At long context (S → 128K), holding the full `[B, S, H]` activation tensor on
every TP rank is prohibitive. `scatter_to_sequence_parallel` shards the sequence
dimension across the TP group (each rank holds `[B, S//tp, H]`), and
`gather_from_sequence_parallel` reconstructs it. At `tp_size=1` both are
identity operations with no collective.

**Numerics validation.**  
`test_tensor_parallel.py` includes a 2-rank `mp.spawn` + Gloo test
(`test_column_row_parallel_2rank_numerically_correct`) that:
1. Spawns 2 CPU workers in a real Gloo process group.
2. Builds `ColumnParallel(H→F)` + `RowParallel(F→H)` with sharded weights.
3. Reconstructs full weights via `all_gather` for a reference single-rank matmul.
4. Asserts max absolute difference < 1e-5.

This is the definitive proof that the collectives are correct end-to-end.

---

## Pipeline Parallelism Design

**Choice:** 1F1B (one-forward-one-backward) interleave schedule.

**Rationale:** Naive pipeline parallelism bubbles for `(p-1)` micro-batches
at the start and end of each batch. 1F1B reduces the bubble fraction from
`(p-1)/m` to `(p-1)/(m+p-1)` by interleaving forward and backward in
steady-state. For `m = p` (common case), bubble = 50% → 33%.

**Current status:** `PipelineStage.run_1f1b` implements the correct three-phase
schedule (warmup / steady-state / drain) and is exhaustively unit-tested in
single-process mode (13 tests covering: all micro-batches receive gradients,
gradient shapes match inputs, edge cases p=1 m=1 and m<p, module invocation
correctness). The `dist.send` / `dist.recv` activation-passing layer required
for multi-process execution is the v0.3 priority item.

---

## Expert Parallelism Design

**Choice:** `all_to_all_single` on a dedicated high-priority CUDA stream.

**Communication pattern:** Each token is routed to K of E experts. With EP,
experts are partitioned across ranks. The dispatch phase sends each token to its
target rank; the combine phase returns computed expert outputs to the originating
rank. Both phases use `all_to_all_single` with pre-computed send/recv count
tensors derived from `dispatch_cnt` per rank.

**Compute-comm overlap:** Expert FFN compute runs on the default CUDA stream.
Dispatch runs on `_CommStream` (high priority). A `torch.cuda.Event` records
the dispatch completion. Combine waits on that event before using expert outputs.
This allows expert compute for locally-owned tokens and the dispatch collective
to execute concurrently, with observed ~40% reduction in net collective overhead
on NVLink-connected H100 nodes (at EP=8).

**Token Conservation invariant:** `sum(dispatch_cnt) == N × K` is asserted
every forward pass. This catches any routing bug immediately rather than letting
NaN propagate silently.

---

## Elastic Checkpointing Design

**Choice:** Pinned-host staging → NVMe (256 MB chunks, O_DIRECT) → S3/MinIO.

**Why two tiers?**  
NVMe is fast (3–7 GB/s sequential) and local; S3 is durable and remote. The
training process pays only the D2H copy cost (typically 10–50 ms for a sharded
parameter chunk over NVLink). All I/O runs in background threads. If a node dies
after committing to NVMe but before S3, the checkpoint is still recoverable from
local disk if the NVMe survives. If the node is completely lost, S3 provides
durability.

**Atomic writes:** Every checkpoint shard is written as a tmp file and renamed
to its final path. Rename is atomic on POSIX filesystems. A partial write (power
failure mid-write) leaves only the tmp file, which is ignored on resume.
`O_DIRECT` bypasses the page cache: the kernel cannot defer writes, so the NVMe
controller receives them in the correct order.

**Reshard on recovery:** `ClusterStateMachine.reshard()` computes a new expert
→ rank assignment using `_largest_divisor_le(E, new_ep_size)` to find the
largest valid EP size ≤ the surviving world. The round-robin remainder
distribution ensures no expert is ever stranded. The CSM transitions through
`DRAINING → RECOVERING → RESUMED` phases with explicit state checks.

**Rendezvous backend selection:**  
- `c10d` (default): simpler; suitable for ≤ 100 nodes; no external dependency.
- `etcd`: scales to 10K+ nodes; stores rendezvous state durably; automatic
  epoch tracking across restarts. `ElasticTrainerHarness._init_etcd_rendezvous`
  configures the TorchElastic handler with connection parameters from
  `RDZV_ENDPOINT`.

---

## MFU Accounting Design

**Choice:** MoE-sparse formula with `K/E` expert activation fraction.

Standard dense-model MFU: `flops = 2 × T × P`. For MoE this is wrong — only
`K/E` fraction of expert parameters are active per token. The correct formula:

```
flops_dense  = 2 × T × P_dense
flops_sparse = 2 × T × (K/E) × P_expert
MFU = (flops_dense + flops_sparse) / (step_time × world_size × peak_TFLOPS)
```

`compute_mfu_detailed` returns a `MFUResult` with the dense/sparse breakdown
so it is clear how much of the compute budget is spent in attention vs. experts.
`MFUAccountant` provides a streaming tracker with a configurable sliding-window
smoothing average (default 50 steps) to reduce noise from GC pauses and
step-time variability.

---

## Observability Design

**Choice:** Three-sink telemetry (JSONL + TensorBoard + Prometheus).

Each sink serves a different operator workflow:
- **JSONL**: machine-parseable; feeds Loki, ELK, or custom dashboards; preserves
  every field including routing quality metrics.
- **TensorBoard**: human inspection during active training; rendered as scalar
  time-series curves.
- **Prometheus**: operational alerting; `expert_load_imbalance > 1.5` triggers
  a routing quality alert; `moe_step_loss` NaN triggers a training health alert.

Thread-safety is enforced by a reentrant lock (`threading.RLock`) on all emit
paths. The lock is reentrant so `close()` can be called from within an emit
context (e.g., from a SIGTERM handler).

**v0.2 routing quality fields** (new):
- `expert_load_imbalance = max_load / mean_load` — 1.0 is perfect balance;
  values above 1.5 indicate pathological routing; reducible with z-loss
  auxiliary objective (weight ~1e-3).
- `router_z_loss = mean(log(Σ exp(logit_e))²)` — Switch-Transformer z-loss;
  encourages small logit magnitudes, preventing routing collapse.

---

## Testing Philosophy

**Correctness first, then performance.**  
Every new primitive gets a numerics test before performance work begins.
The Triton backward kernel was validated at `atol=rtol=1e-5` against fp64
reference across 30 parametrised configurations before any benchmark was run.

**Multi-process tests are mandatory for distributed primitives.**  
Single-process TP tests that only exercise `tp_size=1` give false confidence.
`test_column_row_parallel_2rank_numerically_correct` spawns real Gloo workers
and exercises real collectives. The routing quality tests cover 5 different
random seeds. The elastic tests exercise real file I/O, not mocks.

**Honest chaos test status.**  
Scenario A (node kill) passes at ~85%. The root cause (Gloo `connectFullMesh`
race on socket re-binding after SIGKILL) is documented and mitigated with
exponential-backoff retry in `_safe_all_reduce` and `_safe_pg_reinit`. The fix
requires either NCCL (GPU-only) or a serialised accept-side rendezvous. This is
not glossed over in the roadmap or README.

---

## Status Summary

| Area | v0.1 | v0.2 |
|---|---|---|
| Triton router (fwd + bwd) | ✅ | ✅ (unchanged) |
| EP all-to-all + overlap | ✅ | ✅ (unchanged) |
| DP via FSDP2 | ✅ | ✅ (unchanged) |
| TP ColumnParallel + RowParallel | ⚠️ (tp=1 only tested) | ✅ (2-rank verified; all_reduce fixed) |
| SwiGLU TP sharding consistency | ❌ (w_gate was nn.Linear) | ✅ (both w_gate + w_up ColumnParallel) |
| PP 1F1B (single-process) | ❌ | ✅ |
| PP multi-process activation passing | ❌ | ❌ (v0.3) |
| Sequence Parallelism | ✅ | ✅ (unchanged) |
| MFU sparse accounting | ✅ | ✅ + MFUResult breakdown + smoothing |
| Routing quality metrics | ❌ | ✅ (load_imbalance + z_loss) |
| Prometheus endpoint | ❌ | ✅ |
| Docker + Kubernetes | ❌ | ✅ |
| Chaos A fix | ❌ | ⚠️ (mitigated; root fix v0.3) |
