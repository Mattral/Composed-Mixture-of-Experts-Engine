# Limited Hardware Development Guide

**Version:** v0.3.2  
**Last updated:** June 2026

> **This is the realistic default.** Most contributors will not have
> constant access to a multi-GPU cluster. This guide explains how to make
> full, productive progress on moe-engine with only a laptop or a single T4.

---

## The 80/20 rule for GPU-limited development

**80–90% of moe-engine can be developed and validated on CPU only.**

The core insight from MOE instructions v2.1: the current bottlenecks are
structural and DX-related, not fundamental GPU-compute problems. Configuration
validation, routing invariants, pipeline scheduling, checkpoint logic,
telemetry, and most distributed logic can all be tested on a single CPU core.

The remaining 10–20% that genuinely requires a GPU:
- Triton kernel compilation and real kernel performance numbers
- CUDA event timing (dispatch_ms, combine_ms)
- End-to-end MFU validation at production batch sizes
- GPU memory profiling

---

## Tier 0: CPU-only development (laptop / CI)

### Setup (one-time, 2 minutes)

```bash
# From the repo root
cd moe-engine/
pip install -e ".[dev]"          # installs PyTorch CPU, Pydantic, pytest, etc.
python scripts/validate_config.py configs/   # verify both configs load cleanly
```

### Daily development loop

```bash
# After any change to pkg/:
make test-cpu                     # 235 tests, ~60s

# After changing configs/:
make validate-config              # Pydantic validation, ~0.3s

# After changing train.py:
make smoke                        # end-to-end smoke run, ~5s

# Before committing:
make lint                         # ruff + mypy
make format                       # ruff format auto-fix
```

### What you can fully develop and test on CPU

| Feature | CPU test coverage | Test file |
|---------|:-----------------:|-----------|
| Pydantic MoEConfig system | ✅ 34 tests | `test_config.py` |
| Router invariants (conservation, NaN, bounds) | ✅ 7 tests | `test_kernels.py` |
| Numerical correctness vs fp64 ref | ✅ 30 configs | `test_kernels_numerics.py` |
| TP/PP/SP scheduling logic | ✅ 43 tests | `test_tensor_parallel.py`, `test_pipeline_parallel.py`, `test_sequence_parallel_v03.py` |
| Async checkpointing (NVMe + S3 mock) | ✅ 17 tests | `test_elastic.py`, `test_elastic_v02.py` |
| MFU accounting math | ✅ 21 tests | `test_mfu.py`, `test_mfu_v02.py` |
| Telemetry / JSONL output | ✅ 22 tests | `test_telemetry.py` |
| EP dispatch/combine (ep_size=1 no-op) | ✅ via mock | `test_mock_dist.py` |
| End-to-end training loop | ✅ smoke | `test_smoke_e2e.py` |
| Routing quality metrics | ✅ 12 tests | `test_routing_quality.py` |

### What CPU testing does NOT cover

| Feature | Why GPU is needed |
|---------|------------------|
| Triton kernel compilation | `triton.jit` requires CUDA hardware |
| Real CUDA event timing | `torch.cuda.Event` requires CUDA |
| MFU at production scale | Needs batch size >> 512 to be meaningful |
| EP all-to-all real latency | Real collective latency only on multi-GPU |
| GPU memory profiling | `torch.cuda.memory_stats()` |

For these, use the T4 notebook (Section "Tier 1" below).

---

## Tier 1: Single T4 GPU (Google Colab free tier)

**Cost:** Free.  
**Setup time:** ~5 minutes (upload zip, run cells 0–2 of the notebook).

### When to use

- After any change to `pkg/kernels/moe_router.py`
- After any change to `pkg/distributed/expert_parallel.py`
- After any change to `train.py` that affects the training loop
- Before tagging a release

### Setup

1. Open `moe-engine/notebooks/moe_engine_v032_T4_validation.ipynb` in Google Colab.
2. `Runtime → Change runtime type → T4 GPU → Save`.
3. Upload `moe_engine_v032_final.zip` using the Files panel.
4. Run cells 0–3: environment check, install, codebase verification, smoke test.

This takes ~5 minutes and confirms that the Triton kernel compiles and runs on
real GPU hardware.

### What T4 validation covers

- Triton kernel compilation at H ∈ {256, 512, 1024, 2048}, E ∈ {16, 32, 64}, K ∈ {2, 4}
- Token conservation sweep: `violations=0/100` on CUDA
- `router_fwd` and `router_fwd_bwd` throughput at all benchmark configs
- Full `DistributedMoELayer` forward at production-scale shapes (H=4096, E=64)
- Chaos Scenario A and B pass rates

### Interpreting T4 numbers

T4 peak is 65 TFLOPS BF16. MFU will be very low (~0.1–0.5%). This is expected — T4 is an inference card, not a training accelerator. The T4 validates **correctness and relative performance** (CPU vs GPU speedup), not absolute production MFU.

For absolute MFU targets, use H100 SXM5 (989 TFLOPS BF16, ~40–55% MFU at `H=4096, B=8, S=4096, EP=8`). See `docs/benchmarks.md`.

---

## Tier 2: 4-GPU local node (if available)

With 4 × RTX 3090 / 4090 or similar:

```bash
torchrun --standalone --nproc_per_node=4 \
  train.py --config configs/default.yaml \
  --max-steps 20 --smoke --profile
```

This tests EP all-to-all overlap, TP bandwidth, PP scheduling, and FSDP2 at
`world_size=4`. Not required for most development.

---

## Feature development checklist (limited hardware)

Use this checklist when developing a new feature without cluster access:

**Before coding:**
- [ ] Can this be tested on CPU? (Answer is yes for ~90% of features.)
- [ ] Write the CPU tests first (`tests/test_<feature>.py`, mark `@pytest.mark.cpu`).
- [ ] Add `make test-cpu` to the development loop.

**During development:**
- [ ] Run `make test-cpu` after every meaningful change.
- [ ] Run `make smoke` to confirm the training loop end-to-end.
- [ ] Run `make validate-config` after any config schema changes.

**Before PR / merge:**
- [ ] `make lint && make format`
- [ ] `make test-cpu` passes (235+ tests)
- [ ] `make validate-config` passes
- [ ] `make smoke` passes
- [ ] If Triton/GPU paths were modified: run T4 Colab notebook Sections 0–5.

**After merge (when T4 access is available):**
- [ ] Full T4 notebook (all 13 sections)
- [ ] Archive `gpu_results.json` to `benchmarks/` if numbers changed
- [ ] Update `RESULTS.md` if GPU numbers changed

---

## Which files are safe to modify without GPU access

**Safe (CPU-testable):**
- `pkg/utils/config.py` — config system (34 CPU tests)
- `pkg/distributed/mesh.py` — topology math (CPU tests)
- `pkg/distributed/pipeline_parallel.py` — 1F1B scheduling (13 CPU tests)
- `pkg/distributed/data_parallel.py` — FSDP2 wrappers (unit testable at dp=1)
- `pkg/elastic/fault_monitor.py` — checkpointing (17 CPU tests)
- `pkg/telemetry/logger.py` — telemetry (22 CPU tests)
- `pkg/utils/mfu.py` — MFU math (21 CPU tests)
- `pkg/models/moe.py` — model architecture (smoke test)
- `docs/`, `tests/`, `configs/` — all CPU-testable

**Requires T4 validation:**
- `pkg/kernels/moe_router.py` — Triton JIT kernels
- `pkg/distributed/expert_parallel.py` — CUDA stream management
- `train.py` — training loop with GPU path

**Requires multi-GPU cluster:**
- Real EP all-to-all performance data
- MFU at production scale (EP=8, H=4096)
- Chaos Scenario A with >2 ranks

---

## Common mistakes (and how to avoid them)

**Mistake: Making Triton changes that only fail on real GPU**

The `K: tl.constexpr` bug (v0.3.1 → v0.3.2) only triggered on real GPU
hardware; the CPU reference path has no such requirement. This type of bug is
caught by Section 2 of the T4 notebook (`verify_codebase_is_v032`).

Prevention: whenever modifying `pkg/kernels/moe_router.py`, run at least the
T4 Colab notebook Section 4 (smoke test) before merging.

**Mistake: Assuming CPU timing = GPU timing**

CPU reference path timing (ms per step) is completely unrelated to GPU timing.
The CPU path uses fp64 and is single-threaded. GPU timing requires real CUDA
events. Never extrapolate GPU performance from CPU benchmarks.

**Mistake: Forgetting `pytestmark = pytest.mark.cpu`**

Tests without `pytestmark` will still run under `pytest -m cpu` only if they
are not marked with `gpu` or `chaos`. But they will be excluded from
`pytest -m cpu` if they have no marker at all when `--strict-markers` is set.
Always add `pytestmark = pytest.mark.cpu` after imports.

**Mistake: Testing with `ep_size > 1` on CPU**

Real `ep_size > 1` requires `dist.init_process_group`. Use `MockDistEnv` from
`tests/mock_dist.py` for single-process multi-rank simulation, or use
`ep_size=1` (which is the no-op path). See `tests/test_mock_dist.py` for examples.
