# ===========================================================================
# moe-engine Makefile
# ===========================================================================
# Standard targets:
#   make test-cpu          Full CPU test suite (no GPU, no chaos, ~60s)
#   make test-gpu          GPU-only tests (requires CUDA + Triton)
#   make smoke             Single-rank smoke train run (CPU, ~3s)
#   make benchmark         CPU benchmark sweep + JSON output
#   make lint              ruff check + mypy
#   make format            ruff format (auto-fixes style)
#   make validate-config   Validate all YAML configs
#   make chaos-b           Chaos Scenario B (storage stall, ~3 min)
#   make chaos-a           Chaos Scenario A (node kill, ~85% pass rate)
#   make clean             Remove __pycache__ and temp files
#   make help              Show this message
#
# Environment variables:
#   PYTHON       Python interpreter   (default: python)
#   PYTEST_OPTS  Extra pytest flags   (default: empty)
#   CONFIG       Config file for smoke (default: configs/smoke.yaml)
# ===========================================================================

PYTHON     ?= python
PYTEST     ?= $(PYTHON) -m pytest
PYTEST_OPTS ?=
CONFIG     ?= configs/smoke.yaml

.PHONY: help test-cpu test-gpu smoke benchmark lint format validate-config \
        chaos-b chaos-a clean

help:
	@echo ""
	@echo "moe-engine Makefile"
	@echo "==================="
	@echo "  test-cpu        Full CPU test suite (~60s, no GPU needed)"
	@echo "  test-gpu        GPU-only tests (requires CUDA + Triton)"
	@echo "  smoke           Single-rank smoke train run (CPU, ~3s)"
	@echo "  benchmark       CPU benchmark sweep, writes /tmp/moe_bench_*.json"
	@echo "  lint            ruff + mypy"
	@echo "  format          ruff format (auto-fix)"
	@echo "  validate-config Validate configs/*.yaml"
	@echo "  chaos-b         Chaos Scenario B (storage stall, ~3 min, 10/10)"
	@echo "  chaos-a         Chaos Scenario A (node kill, ~20 runs, known ~85%)"
	@echo "  clean           Remove __pycache__ and temp artifacts"
	@echo ""
	@echo "  PYTHON=$(PYTHON)  CONFIG=$(CONFIG)"
	@echo ""

# ---------------------------------------------------------------------------
# Tier 0: CPU-only (excludes multi-process, GPU, chaos) — run on every commit
# ---------------------------------------------------------------------------
test-cpu:
	@echo "==> Tier-0 CPU suite (excludes GPU, chaos, multiprocess)"
	$(PYTEST) tests/ \
	  --ignore=tests/test_chaos.py \
	  --ignore=tests/test_smoke_e2e.py \
	  -k "not (2rank or multiprocess or distributed_invariants)" \
	  -q $(PYTEST_OPTS)
	@echo ""
	@echo "==> Tier-0 complete."

# ---------------------------------------------------------------------------
# Tier 1: GPU tests (requires CUDA + Triton)
# ---------------------------------------------------------------------------
test-gpu:
	@echo "==> Tier-1 GPU suite (requires CUDA + Triton)"
	$(PYTEST) tests/ \
	  --ignore=tests/test_chaos.py \
	  -m gpu \
	  -v $(PYTEST_OPTS)

# ---------------------------------------------------------------------------
# Smoke: fast end-to-end single-rank train (2 steps, toy model, CPU)
# ---------------------------------------------------------------------------
smoke:
	@echo "==> Smoke run (config=$(CONFIG))"
	$(PYTHON) train.py --config $(CONFIG) --smoke

# ---------------------------------------------------------------------------
# Benchmark: CPU reference path sweep
# ---------------------------------------------------------------------------
benchmark:
	@echo "==> CPU benchmark sweep"
	$(PYTHON) benchmarks/run_benchmark.py \
	  --json /tmp/moe_bench_cpu.json \
	  --csv  /tmp/moe_bench_cpu.csv
	@echo "==> Done. Results: /tmp/moe_bench_cpu.json"

# ---------------------------------------------------------------------------
# Validate all configs
# ---------------------------------------------------------------------------
validate-config:
	@echo "==> Validating configs/*.yaml"
	$(PYTHON) scripts/validate_config.py configs/

# ---------------------------------------------------------------------------
# Lint and type checking
# ---------------------------------------------------------------------------
lint:
	@echo "==> ruff check"
	-ruff check pkg/ train.py benchmarks/
	@echo ""
	@echo "==> mypy"
	-mypy pkg/ --ignore-missing-imports --no-error-summary

format:
	@echo "==> ruff format"
	ruff format pkg/ train.py benchmarks/ tests/

# ---------------------------------------------------------------------------
# Chaos tests (require torchrun + Gloo; run manually or in dedicated CI)
# ---------------------------------------------------------------------------
chaos-b:
	@echo "==> Chaos Scenario B (storage stall, 10/10 expected)"
	GLOO_SOCKET_IFNAME=lo $(PYTEST) tests/test_chaos.py \
	  -v -m chaos -k "scenario_b"

chaos-a:
	@echo "==> Chaos Scenario A (node kill, ~85% expected, count=20)"
	CHAOS_FAULT_TOLERANT=1 GLOO_SOCKET_IFNAME=lo \
	  $(PYTEST) tests/test_chaos.py \
	  -v -m chaos -k "scenario_a" --count=20

# ---------------------------------------------------------------------------
# Clean
# ---------------------------------------------------------------------------
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null; true
	find . -name "*.pyc" -delete 2>/dev/null; true
	find . -name ".mypy_cache" -type d -exec rm -rf {} + 2>/dev/null; true
	find . -name ".ruff_cache" -type d -exec rm -rf {} + 2>/dev/null; true
	rm -f /tmp/moe_bench_*.json /tmp/moe_bench_*.csv
	@echo "==> Clean complete."
