# ===========================================================================
# moe-engine Makefile  —  v0.3.3
# ===========================================================================
# Run from: moe-engine/   (the directory containing pyproject.toml)
#
# Targets:
#   test-cpu        CPU suite, wider than the CI Tier-0 job (also runs the differential files)
#   test-gpu        Tier-1 GPU suite (requires CUDA + Triton)
#   smoke           Single-rank smoke run (CPU, ~5s)
#   benchmark       CPU benchmark sweep → /tmp/moe_bench_cpu.json
#   benchmark-gpu   GPU benchmark sweep → /tmp/moe_bench_gpu.json
#   validate-config Validate all configs/*.yaml with Pydantic
#   lint            ruff check + mypy
#   format          ruff format (auto-fix)
#   chaos-b         Scenario B storage stall (10/10 expected, ~3 min)
#   chaos-a         Scenario A node kill (~85%, count=20, ~5 min)
#   clean           Remove __pycache__ and temp artifacts
#   help            Show this message
#
# Environment variables:
#   PYTHON      Python interpreter      (default: python)
#   PYTEST_OPTS Extra pytest flags      (default: empty)
#   CONFIG      Config for smoke target (default: configs/smoke.yaml)
# ===========================================================================

PYTHON     ?= python
PYTEST     ?= $(PYTHON) -m pytest
PYTEST_OPTS ?=
CONFIG     ?= configs/smoke.yaml

.PHONY: help test-cpu test-gpu smoke benchmark benchmark-gpu \
        validate-config lint format chaos-b chaos-a clean

help:
	@echo ""
	@echo "moe-engine v0.3.3 — Make targets"
	@echo "=================================="
	@echo "  test-cpu        CPU suite (wider than the CI Tier-0 job)"
	@echo "  test-gpu        Tier-1 GPU suite (requires CUDA + Triton)"
	@echo "  smoke           Single-rank smoke run (~5s, CPU)"
	@echo "  benchmark       CPU benchmark sweep"
	@echo "  benchmark-gpu   GPU benchmark sweep (requires CUDA)"
	@echo "  validate-config Validate configs/*.yaml"
	@echo "  lint            ruff + mypy"
	@echo "  format          ruff format (auto-fix)"
	@echo "  chaos-b         Chaos Scenario B (storage stall, 100% expected)"
	@echo "  chaos-a         Chaos Scenario A (node kill, ~85% expected)"
	@echo "  clean           Remove __pycache__ and temp files"
	@echo ""
	@echo "  PYTHON=$(PYTHON)  CONFIG=$(CONFIG)"
	@echo ""

# ---------------------------------------------------------------------------
# CPU suite. Wider than CI's Tier-0 job, which also ignores
# test_differential.py, test_comm_ops.py, test_grad_sync.py, and
# test_train_step.py. That narrower job last reported:
#   417 passed, 3 skipped, 17 deselected, 1 xfailed
# The four differential files, as their own job, last reported 243 passed.
# The 38 config tests are inside the Tier-0 count.
# ---------------------------------------------------------------------------
test-cpu:
	@echo "==> CPU suite (includes differential files; CI Tier-0 does not)"
	$(PYTEST) tests/ \
	  --ignore=tests/test_chaos.py \
	  --ignore=tests/test_smoke_e2e.py \
	  -m cpu \
	  -k "not (2rank or multiprocess or distributed_invariants)" \
	  -q $(PYTEST_OPTS)
	@echo ""
	@echo "==> Done. CI Tier-0: 417 passed, 3 skipped, 17 deselected, 1 xfailed."
	@echo "    Differential files alone: 243 passed. This target runs both selections."

# ---------------------------------------------------------------------------
# Tier 1: GPU (requires CUDA + Triton)
# ---------------------------------------------------------------------------
test-gpu:
	@echo "==> Tier-1 GPU suite (requires CUDA + Triton)"
	$(PYTEST) tests/test_kernels.py tests/test_kernels_numerics.py \
	  -m gpu -v $(PYTEST_OPTS)

# ---------------------------------------------------------------------------
# Smoke: fast end-to-end single-rank train run (CPU, ~5s)
# ---------------------------------------------------------------------------
smoke:
	@echo "==> Smoke run (config=$(CONFIG))"
	$(PYTHON) train.py --config $(CONFIG) --smoke

# ---------------------------------------------------------------------------
# Benchmark sweeps
# ---------------------------------------------------------------------------
benchmark:
	@echo "==> CPU benchmark sweep"
	$(PYTHON) benchmarks/run_benchmark.py \
	  --json /tmp/moe_bench_cpu.json \
	  --csv  /tmp/moe_bench_cpu.csv
	@echo "==> Done. Results: /tmp/moe_bench_cpu.json"

benchmark-gpu:
	@echo "==> GPU benchmark sweep (requires CUDA + Triton)"
	$(PYTHON) benchmarks/run_benchmark.py \
	  --cuda \
	  --json /tmp/moe_bench_gpu.json \
	  --csv  /tmp/moe_bench_gpu.csv
	@echo "==> Done. Results: /tmp/moe_bench_gpu.json"

# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------
validate-config:
	@echo "==> Validating configs/*.yaml"
	$(PYTHON) scripts/validate_config.py configs/

# ---------------------------------------------------------------------------
# Lint and type checking
# ---------------------------------------------------------------------------
lint:
	@echo "==> ruff check"
	-ruff check pkg/ train.py benchmarks/ scripts/ tests/ examples/
	@echo ""
	@echo "==> ruff format --check"
	-ruff format --check pkg/ train.py benchmarks/ scripts/ tests/ examples/
	@echo ""
	@echo "==> mypy"
	-mypy pkg/ --ignore-missing-imports --no-error-summary

format:
	@echo "==> ruff format (auto-fix)"
	ruff format pkg/ train.py benchmarks/ scripts/ tests/

# ---------------------------------------------------------------------------
# Chaos tests (require torchrun + Gloo; run manually or in dedicated CI)
# ---------------------------------------------------------------------------
chaos-b:
	@echo "==> Chaos Scenario B (storage stall, 10/10 expected, ~3 min)"
	GLOO_SOCKET_IFNAME=lo $(PYTEST) tests/test_chaos.py \
	  -v -m chaos -k "scenario_b" $(PYTEST_OPTS)

chaos-a:
	@echo "==> Chaos Scenario A (node kill, ~85% expected, 20 runs)"
	CHAOS_FAULT_TOLERANT=1 GLOO_SOCKET_IFNAME=lo \
	$(PYTEST) tests/test_chaos.py \
	  -v -m chaos -k "scenario_a" --count=20 $(PYTEST_OPTS)

# ---------------------------------------------------------------------------
# Clean
# ---------------------------------------------------------------------------
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null; true
	find . -name "*.pyc" -delete 2>/dev/null; true
	find . -name ".mypy_cache" -type d -exec rm -rf {} + 2>/dev/null; true
	find . -name ".ruff_cache" -type d -exec rm -rf {} + 2>/dev/null; true
	find . -name "*.egg-info" -type d -exec rm -rf {} + 2>/dev/null; true
	rm -f /tmp/moe_bench_*.json /tmp/moe_bench_*.csv
	@echo "==> Clean complete."
