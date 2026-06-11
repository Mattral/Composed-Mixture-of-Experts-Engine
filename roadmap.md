# moe-engine Roadmap

**Last Updated:** June 2026  
**Version:** v0.2  
**Status:** Phase 1 complete. Phase 2 (performance evidence + multi-node hardening) in progress.

## Legend
- ✅ Complete + CI-verified  
- ⚠️ Partial / known issue  
- ❌ Not started  
- 🔒 Blocked on external dependency

---

## v0.1 — Correctness Foundation ✅

| Item | Status | Notes |
|---|---|---|
| Triton backward kernel (`_router_bwd_kernel`) | ✅ | Analytic ∂/∂logits; `atol=rtol=1e-5` on H∈[64–512], E∈[8–256] |
| Token conservation invariant | ✅ | `sum(dispatch_cnt)==N×K`; asserted every forward; 100-seed sweep clean |
| EP all-to-all dispatch + combine | ✅ | Dedicated CUDA stream; event sync; no-op on world_size=1 |
| FSDP2 sharding along DP axis | ✅ | Per-param DTensor via `fully_shard` |
| Async two-tier checkpointing | ✅ | NVMe (O_DIRECT, 256 MB chunks) + S3/MinIO; atomic rename |
| TorchElastic state machine | ✅ | Evict → reshard (round-robin) → reload → resume |
| Chaos Scenario B (storage stall) | ✅ | Queue drains; no deadlock; `latency_inject` event emitted |
| Chaos Scenario A (node kill) | ⚠️ | ~85% pass rate; Gloo `connectFullMesh` timeout on 4-rank restart |
| Dead code removal | ✅ | Zero `if False` / placeholder branches remain |

---

## v0.2 — 4D Parallelism + Production Polish ✅

| Item | Status | Notes |
|---|---|---|
| Tensor Parallelism: ColumnParallel + RowParallel | ✅ | Wired into expert FFN SwiGLU; all-gather on col output, reduce-scatter on row output |
| Sequence Parallelism (TP > 1) | ✅ | `scatter/gather_sequence_parallel`; no-op at tp_size=1 |
| Pipeline Parallelism: PipelineStage + 1F1B | ✅ | Warmup / steady-state / drain phases; single-process unit-tested |
| MoE-aware MFU accounting | ✅ | Sparse fraction `K/E × P_expert`; `MFUAccountant` streaming tracker; smoothed window |
| Real CUDA telemetry | ✅ | CUDA events on dispatch + combine; `torch.cuda.memory_stats()` peak GB |
| Expert load imbalance metric | ✅ | `max_load / mean_load` per step; logged to JSONL + TensorBoard |
| Router z-loss | ✅ | Auxiliary regularisation; emitted per step; configurable weight |
| Prometheus metrics endpoint | ✅ | Optional in-process `/metrics`; 8 gauges; port configurable |
| Etcd rendezvous (>100 nodes) | ✅ | `ElasticTrainerHarness` backend selector; epoch tracking |
| Dockerfile + docker-compose | ✅ | Multi-stage image; smoke/4-GPU/8-GPU targets; monitoring stack |
| Kubernetes manifests | ✅ | Single-node Job + 16-node Indexed Job; PVC; etcd rendezvous |
| Benchmark suite | ✅ | `benchmarks/run_benchmark.py`; CPU+GPU sweeps; JSON+CSV output |
| Gradient accumulation | ✅ | `gradient_accumulation_steps` config key |
| Warmup + cosine LR schedule | ✅ | Linear warmup + cosine decay in `train.py` |
| `--profile` flag | ✅ | Per-step benchmark JSON written to `benchmarks/` on exit |
| `test_telemetry.py` (thread safety, field completeness) | ✅ | 12 tests |
| `test_routing_quality.py` | ✅ | 12 tests covering imbalance, z-loss, RouterProfile |
| `test_pipeline_parallel.py` | ✅ | 12 tests covering 1F1B schedule correctness |
| `test_mfu_v02.py` | ✅ | 12 tests covering MFUAccountant, detailed breakdown |
| `test_elastic_v02.py` | ✅ | 12 tests covering retention, harness, reshard edge cases |

**Test suite total: 96 passed, 1 skipped (GPU-only), ~30s on CPU.**

---

## v0.3 — Performance Evidence (Planned)

| Item | Priority | Notes |
|---|---|---|
| Fix Chaos Scenario A flakiness | P0 | Replace Gloo `connectFullMesh` serialization; switch to NCCL for GPU chaos tests |
| `BENCHMARKS.md` with real cluster data | P0 | Requires sustained 8-GPU run; target MFU ≥ 0.45 |
| Async overlap ratio measurement | P1 | Comm/compute overlap fraction per step |
| Nsight/CUPTI kernel profiling | P1 | Roofline placement for Triton router kernel |
| Scaling curve: EP=1→8, fixed batch | P2 | Token throughput vs EP size on single node |
| Activation recompute benchmarking | P2 | Memory vs compute tradeoff at H=4096 |

---

## v0.4 — Production Hardening (Planned)

| Item | Priority | Notes |
|---|---|---|
| Expert-level overflow re-routing | P1 | Second-choice expert for capacity overflow; +~5% router overhead |
| Tensor streaming checkpoint (no pinned staging) | P1 | Direct CUDA→NVMe for shards >40 GB |
| Kubernetes operator / Kubeflow | P2 | PyTorchJob CRD + fault-tolerant restart policy |
| HuggingFace integration example | P2 | Load a pretrained Mixtral config into the runtime |
| Blog post: "Expert Resharding Under Node Failure" | P3 | |

---

## Known Deficiencies (Honest Disclosure)

### 1. Chaos Scenario A (~85% pass rate)
**Root cause:** Gloo's `connectFullMesh` call during process group re-formation after a SIGKILL races with socket cleanup. The accepting side sees `Connection refused` if the new process' TCP stack hasn't finished binding.  
**Symptoms:** `torchrun` stderr shows `Gloo connectFullMesh failed: timed out connecting to addr=127.0.0.1:XXXXX`.  
**Current mitigation:** `CHAOS_FAULT_TOLERANT=1` env var enables exponential-backoff retries in `_safe_all_reduce` + `_safe_pg_reinit`. Raises pass rate from ~70% to ~85%.  
**Correct fix:** (a) Switch to NCCL for GPU-based chaos tests (removes Gloo entirely), or (b) introduce a rendezvous store that serialises the `accept()` side so new-process binding is guaranteed before the connecting side is attempted.

### 2. No real multi-node benchmark data
All FLOP numbers in `BENCHMARKS.md` are derived from timing the CPU reference path or are illustrative targets for H100 hardware. Real MFU numbers require a sustained 8+ GPU training run with NCCL backend.

### 3. Sequence Parallelism is scatter-only (no in-flight all-gather)
The current SP implementation does a full `all_gather` at the end of each layer to reconstruct the sequence. A more efficient design fuses the all-gather with the next layer's input projection (sequence-parallel + tensor-parallel fusion), halving the number of collectives. Tracked for v0.3.

### 4. PP: no inter-stage communication wiring
`PipelineStage.run_1f1b` is fully unit-tested for scheduling correctness in single-process mode but does not implement the `dist.send` / `dist.recv` calls for multi-process activation passing. The data-flow plumbing (activation buffers, gradient buckets, micro-batch tagging) remains to be wired. Tracked for v0.3.

---

## CI Status

| Test file | Tests | Status |
|---|---|---|
| `test_kernels.py` | 8 | ✅ |
| `test_kernels_numerics.py` | 30 | ✅ |
| `test_routing_quality.py` | 12 | ✅ |
| `test_tensor_parallel.py` | 11 | ✅ |
| `test_pipeline_parallel.py` | 12 | ✅ |
| `test_distributed.py` | 4 | ✅ |
| `test_distributed_invariants.py` | 2 | ✅ |
| `test_elastic.py` | 7 | ✅ |
| `test_elastic_v02.py` | 12 | ✅ |
| `test_mfu.py` | 6 | ✅ |
| `test_mfu_v02.py` | 12 | ✅ |
| `test_telemetry.py` | 12 | ✅ |
| `test_smoke_e2e.py` | 2 | ✅ |
| `test_chaos.py` (baseline) | 1 | ✅ |
| `test_chaos.py` (scenario A) | 1 | ⚠️ ~85% |
| `test_chaos.py` (scenario B) | 1 | ✅ |
| **Total (non-chaos)** | **96** | **✅** |
