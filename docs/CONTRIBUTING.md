# Contributing to moe-engine

**Version:** v0.3.3  
**Last updated:** July 2026

moe-engine is a research-grade runtime for fault-tolerant MoE training at
hyperscale. Contributions should improve correctness, observability,
reproducibility, or operational resilience — or reduce the structural and DX
debt identified in MOE instructions v2.1.

---

## Getting started in 5 minutes (no GPU required)

```bash
# 1. Clone the repository (or unzip moe_engine_v032_final.zip)
cd moe-engine/

# 2. Install in editable mode with dev dependencies
pip install -e ".[dev]"
pip install pre-commit && pre-commit install

# 3. Verify setup
python scripts/validate_config.py configs/   # both configs should pass
make smoke                                    # ~5s smoke run
make test-cpu                                 # ~20s, 348 tests expected

# 4. Verify your environment
python scripts/cli.py info
```

No GPU. No Docker. No cluster. You're ready to develop.

---

## Essential reading (in order)

1. [`README.md`](../moe-engine/README.md) — what is actually built, key results
2. [`docs/ARCHITECTURE.md`](ARCHITECTURE.md) — component map, token lifecycle, design principles
3. [`docs/LIMITED_HARDWARE_GUIDE.md`](LIMITED_HARDWARE_GUIDE.md) — how to develop without a GPU cluster
4. [`docs/testing.md`](testing.md) — four-tier test model, markers, fixtures
5. [`docs/benchmarks.md`](benchmarks.md) — MFU, routing quality, collective latency
6. [`docs/adr/README.md`](adr/README.md) — why major design decisions were made
7. [`RESULTS.md`](../RESULTS.md) — all real numbers with reproduction commands

---

## Development workflow

### Standard loop

```bash
# Make changes to pkg/
make test-cpu                # must pass before any PR
make smoke                   # must pass
make validate-config         # if you changed configs/ or pkg/utils/config.py
make lint                    # ruff + mypy
```

### Before opening a PR

```bash
make format                  # auto-fix formatting
make test-cpu                # 348 tests
make lint                    # zero ruff errors
make validate-config         # configs valid
python scripts/cli.py info   # environment summary for PR description
```

### If you modified Triton kernels

Run the T4 Colab notebook (minimum Sections 0–4):
`moe-engine/notebooks/moe_engine_v032_T4_validation.ipynb`

Attach the output of Section 2 (codebase verification) and Section 4 (smoke test) to the PR.

---

## What to work on

### P0 — Complete ✅

- Monolith split (6 focused distributed modules)
- Pydantic MoEConfig with 38 tests
- Model extracted from train.py
- `__all__` on all packages
- Markers (`@pytest.mark.cpu/gpu/chaos`)
- Makefile, CLI, validate_config.py
- Real GPU numbers in all docs

### P1 — In progress

- `sequence_parallel.py` extracted ✅
- ADRs (001–004) ✅
- Mocked collective backend ✅
- Pre-commit hooks ✅
- Limited Hardware Guide ✅
- CI updated to use `-m cpu` ✅
- Property-based tests (Hypothesis) — **open**
- Model registry/factory pattern — **open**

### P2 — Planned

- Fix Chaos Scenario A (replace Gloo with NCCL in chaos harness)
- Real 8-GPU benchmark data
- Nsight/CUPTI roofline integration
- Expert capacity overflow re-routing
- Non-divisible sequence lengths in SP

See [`roadmap.md`](../roadmap.md) for the full plan.

---

## Code standards

### Python style

- `ruff` for linting and formatting (line length 100)
- Type hints on all public functions
- Docstrings on all public classes and functions (NumPy style)
- No `print()` in library code — use `logging.getLogger(__name__)`

### Test requirements

Every PR that adds functionality must include tests:

```python
import pytest
pytestmark = pytest.mark.cpu   # module-level, after all imports

def test_my_feature():
    from pkg.distributed.mesh import build_topology
    topo = build_topology(dp_size=1, ep_size=1)
    # ... assert something meaningful
```

Minimum coverage expectations:
- New config fields: add to `tests/test_config.py`
- New distributed primitives: add to relevant test file, test at `ep_size=1`
- New CLI commands: test via `scripts/cli.py` import + function call
- New Triton kernel changes: T4 notebook Section 2 check + smoke test

### Invariant checking

All critical mathematical properties must be asserted in the forward path,
not just in tests:

```python
# In production code:
assert not torch.isnan(output).any(), "NaN in output — check gate weight init"
assert int(dispatch_cnt.sum()) == N * K, f"Token conservation violated: {dispatch_cnt.sum()} != {N*K}"
```

This makes errors fire immediately at the layer where they occur, not
mysteriously downstream.

### Documentation

- Update `docs/ARCHITECTURE.md` if you add a new module or change a design decision
- Add an ADR (`docs/adr/ADR-NNN-title.md`) for any significant architectural choice
- Update `RESULTS.md` if your change affects benchmark numbers
- Update `docs/testing.md` if you add new test categories or fixtures

---

## Commit message format

```
<type>(<scope>): <short description>

[Optional body — explain WHY, not WHAT]

[Optional footer — references, breaking changes]
```

Types: `feat`, `fix`, `perf`, `test`, `docs`, `refactor`, `chore`

Examples:
```
feat(distributed): extract sequence_parallel.py as own module

P1.1 per MOE instructions v2.1. SP functions remain importable from
tensor_parallel.py (backward-compat re-exports) and pkg.distributed.

fix(kernels): declare K as tl.constexpr in both Triton kernels

Without constexpr, tl.static_range fails at compile time on real GPU
hardware. Not reproducible on CPU, which masked it until T4 validation.

test(mock_dist): add MockDistEnv for multi-rank simulation

Allows testing ep_size>1 dispatch/combine logic without spawning
multiple processes. See tests/test_mock_dist.py for examples.
```

---

## Pull request checklist

```
- [ ] make test-cpu passes (348+ tests)
- [ ] make lint passes (zero ruff errors)
- [ ] make validate-config passes
- [ ] make smoke passes
- [ ] New tests added (with pytestmark = pytest.mark.cpu)
- [ ] Docstrings on new public functions/classes
- [ ] ARCHITECTURE.md updated if new module added
- [ ] ADR added if significant design decision made
- [ ] RESULTS.md updated if numbers changed
- [ ] T4 notebook run if Triton/GPU paths modified
- [ ] Commit message follows format above
```

---

## Architecture Decision Records

Before making a significant design change, check `docs/adr/` to understand
why existing decisions were made. If your change reverses or supersedes an
ADR, create a new ADR explaining why.

See [`docs/adr/README.md`](adr/README.md) for the ADR index and template.

---

## Getting help

- Open an issue on GitHub describing the problem, your environment (`python scripts/cli.py info`), and the exact error.
- For Triton/GPU issues, attach the output of `python scripts/cli.py info` and the T4 notebook Section 2 (codebase verification).
- For test failures, attach the full `pytest --tb=long` output.
