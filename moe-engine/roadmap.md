# moe-engine Inner Roadmap

## Legend
✅ Complete + CI-verified  ⚠️ Partial / known issue  ❌ Not started

---

## v0.1 — Correctness Foundation ✅ COMPLETE

- [✅] **P0-1: Triton Backward Kernel** — analytic ∂/∂logits through softmax→topK→renorm; `atol=rtol=1e-5`
- [✅] **P0-2: Dead Code Removal** — zero `if False` / placeholder branches; grep-verified
- [✅] **P0-3a: Chaos Scenario B** (storage stall) — queue drains; no deadlock; event emitted
- [⚠️] **P0-3b: Chaos Scenario A** (node kill) — ~85% pass rate; Gloo `connectFullMesh` race; mitigated but not fixed

---

## v0.2 — 4D Parallelism + Production Polish ✅ COMPLETE (96 tests passing)

- [✅] **P1-1: Tensor Parallelism** — `ColumnParallelLinear` + `RowParallelLinear`; wired into expert FFN
- [✅] **P1-2: Pipeline Parallelism** — `PipelineStage` + 1F1B schedule; single-process unit-tested
- [✅] **P1-3: Real MFU** — `K/E × P_expert` sparse formula; `MFUAccountant` streaming tracker; smoothing
- [✅] **P1-4: Real Telemetry** — CUDA events on a2a; `memory_stats()`; load imbalance; z-loss
- [✅] **P2-1: NVMe Streaming Writes** — 256 MB chunks; O_DIRECT with fallback; atomic rename
- [✅] **P2-2: Etcd Rendezvous** — backend selector in `ElasticTrainerHarness`; c10d / etcd
- [✅] **P2-3: Sequence Parallelism** — `scatter/gather_sequence_parallel`; active at tp_size>1
- [✅] **Deployment** — Docker multi-stage; docker-compose (smoke/4GPU/8GPU/monitoring); K8s manifests
- [✅] **Benchmarks** — `benchmarks/run_benchmark.py`; CPU+GPU sweep; JSON+CSV; `BENCHMARKS.md`
- [✅] **Routing Quality** — `expert_load_imbalance`, `router_z_loss` per step
- [✅] **Prometheus** — optional in-process `/metrics` endpoint; 8 gauges

---

## v0.3 — Performance Evidence (Next)

- [❌] Fix Chaos Scenario A (NCCL backend or serialised accept side)
- [❌] `BENCHMARKS.md` with real 8-GPU cluster MFU numbers
- [❌] Async overlap ratio measurement
- [❌] Nsight/CUPTI roofline profiling for Triton kernel

---

## v0.4 — Production Hardening

- [❌] Expert-level capacity overflow re-routing (second-choice expert)
- [❌] Direct CUDA→NVMe streaming checkpoint (no pinned host staging)
- [❌] Kubernetes operator / Kubeflow PyTorchJob CRD
- [❌] PP inter-stage `dist.send`/`dist.recv` wiring for multi-process activation passing

---

## Honest Disclosure

See root `roadmap.md` §"Known Deficiencies" for detailed write-ups on:
1. Chaos Scenario A flakiness (Gloo race condition)
2. Missing real benchmark data
3. SP all-gather not fused with next-layer projection
4. PP no inter-stage communication wiring
