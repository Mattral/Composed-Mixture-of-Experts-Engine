# moe-engine Roadmap

**Last Updated:** 2026-11-23  
**Version:** v0.3.3 (pre-release, 2026-11-20)  
**Status:** P0, P1 and the CPU-doable part of P2.2 are complete. The October 2026 correctness work (M0–M2, ADR-010, ADR-011) is in v0.3.3. Multi-process pipeline parallelism is open. Remaining P2 items require GPU cluster access.

## Legend
- ✅ Complete; where it is tested is in the row or in the README status table
- ⚠️ Partial / known issue
- ❌ Not started
- 🔜 In progress

Push CI runs lint, the Tier-0 CPU suite, the multi-process differential suite (CPU/Gloo), the config tests and the CPU benchmark. The GPU, Docker and Chaos B jobs are `workflow_dispatch` only and have no recorded run in this repository's Actions history. mypy is `continue-on-error` and reports type errors.

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

## v0.3 — PP Comms + SP Fusion + Overlap + T4 Validation ⚠️ (multi-process PP open)

| Item | Status | Notes |
|---|---|---|
| PP `dist.send`/`recv` inter-stage wiring | ⚠️ | `run_1f1b_distributed`: Gloo send/recv; activation tagging; 3-phase 1F1B. Written, but does not run correctly across processes (next row) |
| PP 2-rank `mp.spawn` tests | ❌ | Both fail on Linux (2026-11-23): the last stage runs 0 of the expected micro-batches. Not in any CI job. Open, see "Multi-process tests outside push CI" |
| SP `next_weight` fusion | ✅ | Zero collectives for a replicated weight (ADR-007, August 2026). The v0.3 description, "halves collectives", was replaced by ADR-007, which removed an incorrect `all_reduce` and fixed `gather_from_sequence_parallel` for batch size > 1. The `*_2rank` tests pass when run manually; no CI job runs them. SP itself is in the push differential job |
| Comm/compute overlap ratio telemetry | ✅ | `dispatch_ms / expert_compute_ms` in every step record |
| WandB `WandBSink` + `log_config()` | ✅ | inactive when `WANDB_API_KEY` is absent. A zero-cost claim is not yet validated. |
| Prometheus `/metrics` (10 gauges) | ✅ | optional, port-configurable |
| T4 GPU validation | ✅ single-GPU router | 80.1× versus the CPU reference path at N=4096; no GPU baseline comparison yet. Chaos B 10/10 is a recorded manual run. |
| Triton `K: tl.constexpr` fix (v0.3.2) | ✅ | Bug that prevented all real-GPU runs since v0.1 |
| Real GPU numbers in all docs | ✅ | All "illustrative" entries replaced with real T4 measurements |

---

## v0.3.2 — Architectural Refactoring + P0/P1 Completion ✅

This release addressed all P0 and P1 items from MOE instructions v2.1.

### P0.1 — Architectural Cleanup ✅

| Item | Status | Detail |
|---|---|---|
| Split `parallel_mesh.py` monolith (1,165 lines) | ✅ | 7 focused modules (≤393 lines each) + backward-compat shim |
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
| `@pytest.mark.cpu` on all CPU test files | ✅ | 27 test modules marked `cpu` (`test_chaos.py` is marked `chaos`) |
| `@pytest.mark.gpu`, `@pytest.mark.chaos` registered | ✅ | `pyproject.toml` with `--strict-markers` |
| `test_config.py` | ✅ | 38 tests in the current file; all 38 run inside the Tier-0 job |
| `test_mock_dist.py` / `mock_dist.py` | ✅ | 17 tests; `MockTopology`, `MockDistEnv` |
| `test_properties.py` | ✅ | 9 property-based tests (Hypothesis); token conservation, expert ownership, config invariants |
| **Push CI pytest** | ✅ | Tier-0 and config jobs. Current counts (2026-11-23): Tier-0 417 passed, 3 skipped, 15 deselected, 1 xfailed; differential job 243 passed. The 38 config tests are inside the Tier-0 count. |
| Real GPU data in all docs | ✅ | Committed T4 file is `moe-engine/benchmarks/gpu_results_t4.json` → `RESULTS.md`, `BENCHMARKS.md` |

### P0.3 — Basic DX ✅

| Item | Status | Detail |
|---|---|---|
| `Makefile` | ✅ | `test-cpu`, `test-gpu`, `smoke`, `benchmark`, `benchmark-gpu`, `validate-config`, `lint`, `chaos-a`, `chaos-b`, `clean` |
| `scripts/cli.py` (Typer) | ✅ | `train / benchmark / validate / info`, run as `python scripts/cli.py ...`; no `moe` console script is declared |
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
| CI updated to `-m cpu` | ✅ | `.github/workflows/ci.yml` with 6 jobs at v0.3.2; 8 today, including `test-differential` (October 2026) |
| Limited Hardware Guide | ✅ | `docs/LIMITED_HARDWARE_GUIDE.md` (220 lines) |

### P1.3 — Documentation ✅

| Item | Status | Detail |
|---|---|---|
| ADRs | ✅ | ADR-001 (Triton), ADR-002 (checkpointing), ADR-003 (Pydantic), ADR-004 (4D parallelism). ADR-005 to ADR-011 followed; see `docs/adr/README.md` |
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

## v0.3.3 — CI Hardening, Load Balancing, Correctness Fixes ✅

Released 2026-11-20 as a pre-release. It covers everything since v0.3.2:
the CI fixes and P2.2 load balancing below, the August 2026 TP/SP/dispatch
fixes (ADR-007 to ADR-009), and the October 2026 correctness work
(M0–M2, ADR-010 and ADR-011).

### CI fixes (root-caused, not patched)

| Bug | Root cause | Fix |
|---|---|---|
| `test_config.py`: 8 tests silently not raising `ConfigValidationError` | `pydantic` was never declared in `pyproject.toml`/`requirements.txt` — CI installed exactly what was declared, so the module ran through the no-validation fallback shim | Added `pydantic>=2.0.0` as a **hard runtime dependency** (not optional); removed the silent-degradation shim entirely — the module now fails loudly and immediately if pydantic is missing, since a silently-broken validator is more dangerous than an import error |
| `test_learning_rate_override`: `TypeError: str - float` | `yaml.safe_load("1e-5")` returns the **string** `"1e-5"`, not the float `1e-05` — a well-known PyYAML 1.1 grammar quirk (exponential notation without a decimal point or explicit sign parses as a string) | Added `_coerce_env_value()`: tries native `int()`/`float()` before falling back to `yaml.safe_load()` for booleans/null/other scalars |
| `test_properties.py`: Hypothesis `FlakyFailure` on `deadline` | First example for any test constructing `MoERouter` can trigger Triton JIT compilation (~1-3s), exceeding the 500ms deadline; Triton caches after first call so this is a one-time cost being tested as if it were steady-state | Set `deadline=None` on both Hypothesis `settings()` profiles — these tests check correctness, not performance |
| `test_uniform_init_lower_imbalance[2]` | Seed 2 generates a genuinely pathological token distribution where near-zero gate weights still produce marginally higher imbalance than sharp init, at E=16 | Marked `xfail(strict=False)` with a precise, mechanism-level explanation — this is a real statistical edge case, not a code bug, and the surrounding seeds all pass |
| Docker build: `pytorch/pytorch:2.5.1-cuda12.4.1-cudnn9-devel not found` | That exact tag combination does not exist on Docker Hub | Rewrote `Dockerfile` to use verified tags (`2.6.0-cuda12.6-cudnn9-{devel,runtime}`), added a `runtime-cpu` stage (`python:3.11-slim`, no CUDA) for CI smoke-testing without a multi-GB GPU image pull |
| GPU test job: stuck "Waiting for a runner" indefinitely | `test-gpu` ran on every `push`, but no self-hosted GPU runner was registered, so the job blocked forever and could stall the pipeline | Gated `docker` and `test-gpu` behind `workflow_dispatch` with explicit boolean inputs (`run_docker`, `run_gpu`) — they never run automatically and never block the push/PR pipeline |
| `docker-compose.yml` / k8s manifests: stale `v0.2` image tags | Never updated after the v0.3.2 rename | Bumped to `v0.3.2`/`v0.3.3` consistently across `docker-compose.yml`, `training-job.yaml`, `training-job-multinode.yaml` |

### P2.2 — Advanced Load Balancing ✅ (CPU-doable portion complete)

| Item | Status | Detail |
|---|---|---|
| Expert capacity dropping | ✅ | `moe_layer.py`. Since M2 (F12): per expert across top-k slots, slot-major priority, dropped before dispatch (ADR-011). 35 tests in `test_capacity_dropping.py` (Tier-0) plus EP cases in the differential job |
| `DistributedMoELayer.capacity_dropping` | ✅ | Opt-in flag, default `False`. `cfg.model.capacity_dropping` reaches every layer since M2 |
| `dropped_token_fraction` telemetry | ✅ | Wired through `StepRecord`, Prometheus gauge, `train.py` |
| `configs/large_scale.yaml` | ✅ | E=256, top_k=8 fine-grained MoE config exercising capacity dropping + z-loss at scale; 5 dedicated config tests |
| Router auxiliary losses | ✅ (M2) | Before M2 the z-loss was a detached float read from block 0 and did not train the router (F5). It is now a tensor-valued z-loss plus a Switch load-balancing loss from every MoE layer (ADR-011); `test_router_losses.py` and differential `aux-*` layouts |

---

### Correctness audit fixes — M0, M1, M2 (October 2026) ✅

An October 2026 audit compared the distributed paths against a dense
single-device reference with distinct per-rank weights. Before these fixes,
gradients were wrong whenever any parallel dimension was > 1. Details are in
[ADR-010](docs/adr/ADR-010-backward-correctness-and-replicated-gradients.md),
[ADR-011](docs/adr/ADR-011-layout-independent-global-mean-objective.md) and
[`CHANGELOG.md`](CHANGELOG.md). All verification is CPU/Gloo.

| Item | Status | Evidence |
|---|---|---|
| M0 — differential test harness and `test-differential` CI job | ✅ | `test_differential.py`, `test_comm_ops.py`, `test_grad_sync.py`, `test_train_step.py`: 243 passed (push CI and local, 2026-11-23) |
| F1–F3 — autograd-aware EP, TP and SP collectives (`comm_ops.py`) | ✅ | Differential job; finite-difference VJP checks in `test_comm_ops.py` |
| F4 — `E % ep_size != 0` without crash or hang | ✅ | Differential job (E=5, E=10 at ep=2 and ep=4; invalid-count cases) |
| F21, F14-init — replicated-gradient sync, init broadcast | ✅ | `test_grad_sync.py` |
| F20 — no import-time `GLOO_SOCKET_IFNAME` | ✅ | `test_elastic_v02.py` (Tier-0) |
| F22 — global-mean objective over the global batch | ✅ | `test_layout_*` across single, dp2, ep2, sp2, dp2×ep2, dp2×sp2, ep2×sp2 |
| F24, F25, F26 — FSDP2 fixes at dp = 2 | ✅ | `test_grad_sync.py`, dp2 layouts in the differential job |
| F5 — router auxiliary losses | ✅ | `test_router_losses.py` (Tier-0), `aux-*` layouts |
| F12 — capacity dropping before dispatch | ✅ | `test_capacity_dropping.py` (Tier-0), EP capacity cases |
| F14 — global gradient-norm clipping; experts sharded over DP | ✅ | `clip-*` layouts; dp2×ep2 three-step replica test |
| F18 — seeded, TP-consistent training batches | ✅ | `test_train_step.py` (tp = 2 step vs dense reference) |
| F27 — remove the unparsable `signal-correctness.yml` | ✅ | Workflow removed |
| F16 — multi-process pipeline parallelism | ❌ Open | See "Multi-process tests outside push CI" |

Not verified by any of the above: GPU, NCCL, the CUDA-stream overlap path,
dp > 2, multi-node.

---

## v0.4 — Scale Hardening (Planned)

| Item | Priority | Status |
|---|---|---|
| Multi-process pipeline parallelism (`run_1f1b_distributed`, F16) | **P0** | ❌ Both 2-process tests fail; see below |
| CI job for the multi-process tests the Tier-0 name filter deselects | **P0** | ❌ 15 tests, run manually only (13 pass, 2 fail); see below |
| Fix Chaos Scenario A (Gloo → NCCL in chaos harness) | P0 | ❌ Needs GPU |
| Real 8-GPU+ benchmark data + MFU validation | P0 | ⚠️ Two single-GPU architectures validated (T4 June 2026, A100 July 2026 — see ADR-005); multi-GPU and multi-node data ❌, needs 2+ GPUs in one job |
| GPU/NCCL run of the EP/TP/SP/FSDP2 paths and the CUDA-stream overlap | P0 | ❌ Needs GPU |
| Nsight/CUPTI roofline integration | P1 | ❌ Needs GPU |
| Expert capacity overflow **re-routing** (vs. dropping) | P1 | ❌ Dropping done (v0.3.3); re-routing to next-best expert needs real EP bandwidth data to tune |
| Non-divisible sequence length in SP | P1 | ✅ ADR-007 |
| Pipeline PP → end-to-end chaos scenarios | P1 | ❌ Depends on multi-process PP |
| Console-script entry point for the Typer CLI (`moe`) | P2 | ❌ `pyproject.toml` declares none; use `python scripts/cli.py` |
| Direct CUDA-to-NVMe checkpoint streaming | P2 | ❌ |
| Kubernetes operator (auto-scaling) | P2 | ❌ |
| HuggingFace integration examples | P2 | ❌ |
| Extract reusable components (Triton kernel, async ckpt) as standalone packages | P2 | ❌ |

---

### Multi-process tests outside push CI (updated 2026-11-23)

The Tier-0 job deselects tests whose names match
`2rank or multiprocess or distributed_invariants` (15 on the current tree).
The differential job runs other multi-process tests
(`test_differential.py`, `test_comm_ops.py`, `test_grad_sync.py`,
`test_train_step.py`) and does not run these 15. No CI job runs them.
Run them with:

```bash
cd moe-engine
GLOO_SOCKET_IFNAME=lo pytest tests/ \
  --ignore=tests/test_chaos.py --ignore=tests/test_smoke_e2e.py \
  --ignore=tests/test_differential.py --ignore=tests/test_comm_ops.py \
  --ignore=tests/test_grad_sync.py --ignore=tests/test_train_step.py \
  -k "2rank or multiprocess or distributed_invariants" -v
```

Result on 2026-11-23 (WSL Ubuntu, Python 3.12, CPU torch 2.14.1): 13 passed,
2 failed.

| Test | File | Status | History |
|---|---|---|---|
| `test_column_row_parallel_2rank_numerically_correct` | `test_tensor_parallel.py` | ✅ Passes | Fixed in ADR-008 |
| `test_column_parallel_gather_output_true_2rank`, `..._false_2rank`, `..._false_bias_true_raises_2rank` | `test_tensor_parallel.py` | ✅ Pass | Added in ADR-008 |
| `test_swiglu_expert_2rank_numerically_correct` | `test_tensor_parallel.py` | ✅ Passes | Added in ADR-008 |
| `test_sp_fused_2rank_numerically_correct`, `test_sp_fused_2rank_no_collective_used` | `test_sequence_parallel_v03.py` | ✅ Pass | Fixed / added in ADR-007 |
| `test_sp_nondivisible_2rank_round_trip`, `test_sp_fewer_tokens_than_ranks_2rank`, `test_sp_gather_2rank_batch_dimension_not_corrupted` | `test_sequence_parallel_v03.py` | ✅ Pass | Added in ADR-007 |
| `test_moe_layer_2rank_matches_per_token_reference` | `test_moe_layer.py` | ✅ Passes | Added in ADR-009 |
| `test_token_conservation_distributed`, `test_distributed_backward_no_nan` | `test_distributed_invariants.py` | ✅ Pass | Fixed in ADR-009 |
| `test_pp_multiprocess_2stage_activation_flow` | `test_pipeline_parallel.py` | ❌ Fails | Rank 0: Gloo `Connection closed by peer` after rank 1 exits early |
| `test_pp_multiprocess_correct_micro_batch_count` | `test_pipeline_parallel.py` | ❌ Fails | `assert 0 == 6`: the last stage produces no outputs |

The two pipeline failures share one cause. `run_1f1b_distributed` takes the
micro-batch count from `len(micro_batches)`, and non-first stages receive an
empty list, so the last stage schedules zero micro-batches and exits while
stage 0 is still sending. ADR-010 reports that these tests hang on Windows;
that has not been re-checked.

Until 2026-11-23 a sixteenth test, `test_mock_dist.py::test_all_to_all_2ranks`,
was also deselected by the name filter although it runs in a single process
(threads over `MockDistEnv`). It is now `test_all_to_all_two_threads` and runs
in Tier-0.

ADR-004's table lists "2-rank PP mp.spawn verified". ADRs are not edited;
this section is the current status.

---

## Status Summary (2026-11-23)

**Push CI (CPU), every push and PR:**
- Routing invariants (conservation, NaN, bounds, normalisation), router
  numerics vs an fp64 reference, single-process 1F1B scheduling, config
  validation, checkpointing (local and `file://` tiers), telemetry, fault
  state machine, capacity dropping, router auxiliary losses
- Tier-0: `417 passed, 3 skipped, 15 deselected, 1 xfailed` (includes the
  38 config tests)
- Differential job (multi-process, Gloo): `243 passed`. EP, TP, SP, FSDP2 at
  dp = 2 and their combinations at 2 and 4 processes match a dense
  single-device reference for outputs, every gradient and post-step
  parameters
- ruff check and `ruff format --check` pass. mypy is `continue-on-error` and
  reports type errors

**Manual runs on CPU (not in any CI job):**
- 13 of the 15 deselected multi-process tests pass; the two
  pipeline-parallel tests fail (see above)
- `test_smoke_e2e.py` (3 passed), including the S3 tier under moto
- Chaos Scenario B: recorded 10/10 (`RESULTS.md`), and 10/10 again on
  2026-11-23 on a laptop CPU; the `chaos-b` job has no recorded run
- Chaos Scenario A: recorded ~17/20, failures attributed to a Gloo race in
  containers

**Single T4 GPU** (June 2026, Google Colab):
- Triton kernel compiles and runs at H ∈ {256, 512, 1024, 2048}, E ∈ {16, 32, 64}
- 80.1× versus the CPU reference path at N=4096; no GPU baseline comparison
  yet. The T4 host CPU baseline in that table is 0.056 M tok/s.
- Token conservation: `violations=0/100` on CUDA

**Single A100-SXM4-80GB GPU** (July 2026, Lightning AI Studio,
`torch==2.8.0+cu128`; ADR-005):
- The same Triton kernel source runs on Ampere (`sm_80`) after three fixes:
  a `MIN_TRITON_DIM=16` dimension guard, explicit device placement in the
  fp64 reference path, and a single source of truth for `used_triton`
  telemetry
- 56.0× versus the CPU reference path at N=4096 (router forward-only); no GPU
  baseline comparison yet. The A100 table's CPU baseline is 0.183 M tok/s.
  38.4× at the same config for forward+backward. The smaller forward rows are
  1.80×, 5.65× and 11.80×. Cells are the rounded `tokens_per_sec` values in
  `moe-engine/benchmarks/gpu_results_a100.json`.

**Not yet verified:**
- Multi-process pipeline parallelism
- Anything on NCCL or with more than one GPU, including the CUDA-stream
  compute/communication overlap
- End-to-end MFU at 8+ GPUs. The single-GPU notebooks (T4 and A100) exercise
  one GPU each, whatever topology a config requests
- Scenario A on NCCL
- Production throughput at H=4096 with EP=8, DP=8
- E=256 fine-grained MoE at hardware scale (config validated, wiring tested at
  toy dimensions)
- Real EP all-to-all bandwidth (needed to tune capacity re-routing)
- H100 (Hopper, `sm_90`): no run yet; empty row in ADR-005's support matrix
