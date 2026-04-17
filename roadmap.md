# moe-engine Roadmap

**Last Updated:** June 2026  
**Version:** v0.3  
**Status:** Phase 2 complete. Phase 3 (performance evidence on real hardware) is the next milestone.

## Legend
- ✅ Complete + CI-verified  
- ⚠️ Partial / known issue  
- ❌ Not started  

---

## v0.1 — Correctness Foundation ✅

All items complete. See v0.2 roadmap for details.

---

## v0.2 — 4D Parallelism + Production Polish ✅

All items complete. See prior roadmap entries for details.

Key fixes delivered in v0.2.x patch releases:
- RowParallelLinear: wrong collective (reduce_scatter+all_gather → all_reduce)
- SwiGLUExpert w_gate: plain nn.Linear → ColumnParallelLinear (TP consistency)
- test_tensor_parallel: added 2-rank mp.spawn numerical correctness test
- configs: gradient_accumulation_steps added to both smoke.yaml and default.yaml
- test_distributed_invariants: dynamic ports, real build_topology, mp.Queue results
- conftest.py: free_port(), autouse dist cleanup, work_dir fixture
- launch.sh: rdzv_backend read from config (not hardcoded to c10d)

---

## v0.3 — PP Inter-stage Comms + SP Fusion + Overlap Measurement ✅

| Item | Status | Notes |
|---|---|---|
| **PP dist.send/recv inter-stage wiring** | ✅ | `PipelineStage.run_1f1b_distributed`: real Gloo send/recv; activation tagging; 3-phase 1F1B; backward gradient flow |
| **PP multi-process tests (2-rank mp.spawn)** | ✅ | `test_pp_multiprocess_2stage_activation_flow`; `test_pp_multiprocess_correct_micro_batch_count` |
| **SP all-gather fusion** | ✅ | `scatter_to_sequence_parallel(next_weight=...)` fuses backward all-gather with projection matmul; single all_reduce instead of all_gather + matmul |
| **SP fused path tests** | ✅ | 8 tests including 2-rank mp.spawn numerical verification |
| **Comm/compute overlap ratio** | ✅ | `DistributedMoELayer.last_overlap_ratio` = dispatch_ms / expert_compute_ms; emitted in telemetry `collective.comm_compute_overlap_ratio` |
| **Expert compute latency** | ✅ | `DistributedMoELayer.last_expert_compute_ms`; emitted in telemetry |
| **`PipelineStage` topology-aware** | ✅ | Accepts `topology` parameter; `_prev_rank()` / `_next_rank()` from PP group; fast-path preserved for tp=1 |
| **`run_1f1b_distributed` error on single-process** | ✅ | Clear RuntimeError when called without pp_size>1 + dist |
| **Test suite: 134 tests** | ✅ | 0 syntax errors across 33 Python files |

**Test suite total: 148 passed, 1 skipped (GPU-only Triton), ~60s on CPU.**

---

## v0.3 Known Deficiencies

### 1. Chaos Scenario A (~85% pass rate) — unchanged from v0.2
**Root cause:** Gloo `connectFullMesh` socket race after SIGKILL.  
**Current mitigation:** `CHAOS_FAULT_TOLERANT=1` + exponential backoff (~85% pass rate).  
**Correct fix:** Replace Gloo with NCCL for GPU chaos tests, or use a rendezvous store that serialises the accept() side. Deferred — requires GPU hardware.

### 2. PP `run_1f1b_distributed` not exercised in chaos tests
The distributed PP path is verified by 2-rank mp.spawn unit tests but is not yet wired into the full chaos suite (which uses the Gloo path only). The `_chaos_worker.py` uses a dense model without PP. Adding PP to the chaos worker is a v0.4 item.

### 3. SP fusion: sequence_length % tp_size == 0 required
`scatter_to_sequence_parallel` asserts divisibility. Non-divisible sequence lengths (e.g., with packing) require padding logic. Tracked for v0.4.

### 4. No real multi-node benchmark data
CPU numbers are real and reproducible. GPU MFU numbers remain illustrative pending sustained cluster access. This is the top v0.4 priority.

---

## v0.4 — Performance Evidence + Production Hardening (Next)

| Item | Priority | Notes |
|---|---|---|
| Fix Chaos Scenario A (NCCL backend) | P0 | Requires GPU; replaces Gloo in chaos worker |
| Real 8-GPU benchmark numbers | P0 | Target MFU ≥ 0.45; fills `benchmarks/BENCHMARKS.md` |
| Nsight/CUPTI roofline profiling | P1 | Router kernel placement on compute roofline |
| Expert capacity overflow re-routing | P1 | Second-choice expert; +~5% router overhead |
| PP in chaos worker | P1 | End-to-end chaos testing of distributed PP path |
| Direct CUDA→NVMe checkpoint streaming | P2 | No pinned host staging for shards >40 GB |
| Kubernetes operator / Kubeflow | P2 | PyTorchJob CRD |
| HuggingFace Mixtral config integration | P2 | Load pretrained config into moe-engine runtime |
| SP: non-divisible sequence support | P2 | Padding logic for packing / variable-length contexts |

---

## CI Status (v0.3)

| Test file | Tests | Status |
|---|---|---|
| `test_kernels.py` | 5 | ✅ |
| `test_kernels_numerics.py` | 13 | ✅ |
| `test_routing_quality.py` | 12 | ✅ |
| `test_tensor_parallel.py` | 19 | ✅ |
| `test_pipeline_parallel.py` | 16 | ✅ (incl. 2-rank mp.spawn PP) |
| `test_sequence_parallel_v03.py` | 8 | ✅ (incl. 2-rank mp.spawn SP fusion) |
| `test_distributed.py` | 4 | ✅ |
| `test_distributed_invariants.py` | 2 | ✅ |
| `test_elastic.py` | 7 | ✅ |
| `test_elastic_v02.py` | 10 | ✅ |
| `test_mfu.py` | 6 | ✅ |
| `test_mfu_v02.py` | 15 | ✅ |
| `test_telemetry.py` | 12 | ✅ |
| `test_smoke_e2e.py` | 2 | ✅ |
| `test_chaos.py` (baseline + B) | 2 | ✅ |
| `test_chaos.py` (scenario A) | 1 | ⚠️ ~85% |
| **Total (non-chaos)** | **133** | **✅** |
