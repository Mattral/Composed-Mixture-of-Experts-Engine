# moe-engine Roadmap

**Last Updated:** June 2026  
**Version:** v0.3.2  
**Status:** P0 and P1 complete. P2 (scale hardening) begins with cluster access.

## Legend
- ✅ Complete + CI-verified  
- ⚠️ Partial / known issue  
- ❌ Not started  
- 🔜 In progress

---

## v0.1 — Correctness Foundation ✅

All items complete. Core Triton router kernel, token conservation invariant, basic 4D mesh construction.

---

## v0.2 — 4D Parallelism + Production Polish ✅

All items complete. TP/EP/PP primitives, FSDP2, async two-tier checkpointing, WandB, MFU accounting, structured telemetry.

Key fixes in v0.2.x:
- `RowParallelLinear`: `reduce_scatter+all_gather` → correct `all_reduce`
- `SwiGLUExpert` `w_gate`: plain `nn.Linear` → `ColumnParallelLinear` (TP consistency)
- 2-rank `mp.spawn` numerical correctness tests for TP
- `conftest.py`: `free_port()`, autouse dist cleanup, `work_dir` fixture
- `launch.sh`: `rdzv_backend` read from config, not hardcoded

---

## v0.3 — PP Comms + SP Fusion + Overlap + T4 Validation ✅

| Item | Status | Notes |
|---|---|---|
| PP `dist.send`/`recv` inter-stage wiring | ✅ | `run_1f1b_distributed`: real Gloo send/recv; activation tagging; 3-phase 1F1B |
| PP 2-rank `mp.spawn` tests | ✅ | activation flow + micro-batch count verified |
| SP `next_weight` all-gather fusion | ✅ | halves SP collectives; 2-rank verified |
| Comm/compute overlap ratio telemetry | ✅ | `dispatch_ms / expert_compute_ms` in every step record |
| WandB `WandBSink` + `log_config()` | ✅ | zero-cost when `WANDB_API_KEY` absent |
| Prometheus `/metrics` (10 gauges) | ✅ | optional, port-configurable |
| T4 GPU validation | ✅ | Triton kernel verified; 80.1× speedup at N=4096; Chaos B 10/10 |
| Triton `K: tl.constexpr` fix (v0.3.2) | ✅ | Bug that prevented all real-GPU runs since v0.1 |
| Real GPU numbers in all docs | ✅ | All "illustrative" entries replaced with real T4 measurements |

---

## v0.3.2 — Architectural Refactoring + P0/P1 Completion ✅

This release addressed all P0 and P1 items from MOE instructions v2.1.

### P0.1 — Architectural Cleanup ✅

| Item | Status | Detail |
|---|---|---|
| Split `parallel_mesh.py` monolith (1,165 lines) | ✅ | 7 focused modules (≤380 lines each) + backward-compat shim |
| `mesh.py` | ✅ | `ParallelTopology`, `build_topology`, process group cache |
| `tensor_parallel.py` | ✅ | `Column/RowParallelLinear`, SP scatter/gather |
| `sequence_parallel.py` | ✅ | SP extracted to own module; re-exported for backward compat |
| `expert_parallel.py` | ✅ | `all_to_all_dispatch/combine`, `_CommStream` |
| `pipeline_parallel.py` | ✅ | `PipelineStage`, `run_1f1b`, `run_1f1b_distributed` |
| `data_parallel.py` | ✅ | `apply_fsdp2` with expert-weight exclusion |
| `moe_layer.py` | ✅ | `DistributedMoELayer`, `_SwiGLUExpert`, `_expert_to_rank` |
| `router.py` | ✅ | High-level `MoERouterInterface`, `RouterStats` dataclass |
| Pydantic `MoEConfig` hierarchy | ✅ | 6 sub-configs, env-var overrides, `ConfigValidationError` |
| `pkg/models/moe.py` extracted | ✅ | `RMSNorm`, `ToyMoEBlock`, `ToyMoEModel`, `build_model` |
| `pkg/models/registry.py` | ✅ | `@register_model`, `build_model_from_config`, `ModelRegistry` |
| `__all__` on all packages | ✅ | 17 `__all__` declarations across `pkg/` |

### P0.2 — Testing & Validation ✅

| Item | Status | Detail |
|---|---|---|
| `@pytest.mark.cpu` on all CPU test files | ✅ | 16 test files decorated |
| `@pytest.mark.gpu`, `@pytest.mark.chaos` registered | ✅ | `pyproject.toml` with `--strict-markers` |
| `test_config.py` | ✅ | 34 new tests for full `MoEConfig` system |
| `test_mock_dist.py` / `mock_dist.py` | ✅ | 17 tests; `MockTopology`, `MockDistEnv` |
| `test_properties.py` | ✅ | 9 property-based tests (Hypothesis); token conservation, expert ownership, config invariants |
| **Total: 260 tests passing** | ✅ | Up from 201 (v0.3.1) |
| Real GPU data in all docs | ✅ | `gpu_results.json` → `RESULTS.md`, `BENCHMARKS.md` |

### P0.3 — Basic DX ✅

| Item | Status | Detail |
|---|---|---|
| `Makefile` | ✅ | `test-cpu`, `test-gpu`, `smoke`, `benchmark`, `benchmark-gpu`, `validate-config`, `lint`, `chaos-a`, `chaos-b`, `clean` |
| `scripts/cli.py` (Typer) | ✅ | `moe train / benchmark / validate / info` |
| `scripts/validate_config.py` | ✅ | Coloured output, exit code 1 on failure |
| Config error messages | ✅ | Field-level path + actionable description |

### P1.1 — Deeper Modularity ✅

| Item | Status | Detail |
|---|---|---|
| `sequence_parallel.py` extracted | ✅ | Own module; backward-compat re-exports |
| `router.py` high-level interface | ✅ | Separates distributed layer from kernel details |
| Model registry/factory pattern | ✅ | `@register_model("toy_moe")`, `build_model_from_config` |
| Module-level docstrings | ✅ | Every module has `__all__`, purpose, and public API docs |

### P1.2 — Testing Maturity ✅

| Item | Status | Detail |
|---|---|---|
| Mocked collective backends | ✅ | `MockTopology` + `MockDistEnv` threading simulation |
| Property-based tests (Hypothesis) | ✅ | 9 tests × 50 examples each |
| CI updated to `-m cpu` | ✅ | `.github/workflows/ci.yml` with 6 jobs |
| Limited Hardware Guide | ✅ | `docs/LIMITED_HARDWARE_GUIDE.md` (220 lines) |

### P1.3 — Documentation ✅

| Item | Status | Detail |
|---|---|---|
| ADRs | ✅ | ADR-001 (Triton), ADR-002 (checkpointing), ADR-003 (Pydantic), ADR-004 (4D parallelism) |
| Sequence diagrams | ✅ | 4 Mermaid diagrams in `docs/ARCHITECTURE.md` |
| `docs/benchmarks.md` | ✅ | 517 lines; full MFU formula, routing metrics, collective latency |
| `docs/testing.md` | ✅ | Four-tier model, markers, fixture reference |
| `docs/quickstart.md` | ✅ | v0.3.2; CLI, registry, troubleshooting |
| `docs/CONTRIBUTING.md` | ✅ | P0/P1/P2 status, PR checklist, code standards |

### P1.4 — DX Polish ✅

| Item | Status | Detail |
|---|---|---|
| `.pre-commit-config.yaml` | ✅ | ruff, mypy, nbqa, detect-secrets |
| `pyproject.toml` dev extras | ✅ | hypothesis, typer, ruff, mypy, pre-commit |
| One-command setup | ✅ | `pip install -e ".[dev]"` + `pre-commit install` |

---

## v0.4 — Scale Hardening (Planned, requires cluster)

| Item | Priority | Status |
|---|---|---|
| Fix Chaos Scenario A (Gloo → NCCL in chaos harness) | P0 | ❌ Needs GPU |
| Real 8-GPU+ benchmark data + MFU validation | P0 | ❌ Needs cluster |
| Nsight/CUPTI roofline integration | P1 | ❌ Needs GPU |
| Expert capacity overflow re-routing | P1 | ❌ |
| Non-divisible sequence length in SP | P1 | ❌ |
| Distributed checkpoint versioning | P1 | ❌ |
| Pipeline PP → end-to-end chaos scenarios | P1 | ❌ |
| Direct CUDA-to-NVMe checkpoint streaming | P2 | ❌ |
| Kubernetes operator (auto-scaling) | P2 | ❌ |
| HuggingFace integration examples | P2 | ❌ |
| Extract reusable components (Triton kernel, async ckpt) | P2 | ❌ |

---

## Honest Status Summary (June 2026)

**Fully proven at single-rank / CPU:**
- All routing invariants (conservation, NaN, bounds, normalisation)
- All parallelism scheduling logic (1F1B, TP/EP/PP/SP)
- Config validation, async checkpointing, telemetry, fault state machine

**Proven at 2-rank `mp.spawn` on CPU:**
- TP numerical correctness (Column/RowParallel)
- SP fused all-gather
- PP inter-stage `dist.send`/`recv` with activation tagging

**Proven on single T4 GPU:**
- Triton kernel compiles and runs at H ∈ {256, 512, 1024, 2048}, E ∈ {16, 32, 64}
- 80.1× GPU speedup over CPU reference at N=4096
- Token conservation: `violations=0/100` on CUDA
- Chaos Scenario B: 100% pass rate (10/10)

**NOT yet proven:**
- End-to-end MFU at 8+ GPUs
- Chaos Scenario A at NCCL (still Gloo, ~85%)
- Production throughput at H=4096 with EP=8, DP=8
