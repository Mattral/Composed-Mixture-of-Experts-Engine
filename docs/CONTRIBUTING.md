# Contributing to moe-engine

**Version:** v0.2  
**Last updated:** June 2026

moe-engine is a reference implementation for production-grade distributed
Mixture-of-Experts training infrastructure. Contributions should improve
correctness, observability, reproducibility, or operational resilience.

---

## Getting Started

### 1. Read the essential docs

In order:

1. `README.md` — architecture overview and results
2. `docs/ARCHITECTURE.md` — component map and implementation status
3. `docs/DESIGN.md` — why each design choice was made
4. `docs/SYSTEM_DESIGN.md` — precise API reference for all modules
5. `docs/testing.md` — test suite guide and how to write new tests
6. `RESULTS.md` — reproducible results and telemetry samples
7. `roadmap.md` — honest status and known deficiencies

### 2. Install in editable mode with dev dependencies

```bash
cd moe-engine/moe-engine
pip install -e ".[dev]"
```

### 3. Run the full test suite

```bash
pytest tests/ -v --ignore=tests/test_chaos.py
# Expected: 123 passed, 1 skipped (~45s on CPU)
```

### 4. Validate Triton numerics

```bash
python tests/run_numerics_tests.py
# Expected: 30/30 passed
```

### 5. Run the benchmark suite to establish your baseline

```bash
python benchmarks/run_benchmark.py --json /tmp/baseline.json
```

---

## Contribution Workflow

1. **Open an issue** if the bug or feature is not already tracked. Include:
   - Failing test or assertion if it's a bug.
   - Motivation and design sketch if it's a feature.

2. **Branch from `main`** with a descriptive name:
   ```bash
   git checkout -b fix/row-parallel-collective-correctness
   git checkout -b feat/pp-inter-stage-send-recv
   git checkout -b docs/update-v03-system-design
   ```

3. **Implement with minimal scope.** One logical change per PR. Don't bundle
   a bug fix with a refactor.

4. **Add or update tests.** Every change to runtime behaviour must include
   a test that fails before the fix and passes after. For distributed changes,
   use `mp.spawn` or Gloo multi-process tests — single-process tests at
   `tp_size=1` do not exercise actual collectives.

5. **Run the full suite before submitting:**
   ```bash
   pytest tests/ -v --ignore=tests/test_chaos.py
   python benchmarks/run_benchmark.py --json /tmp/bench.json
   ```

6. **Submit a pull request** with:
   - Concise description of the change.
   - Reasoning for the design (especially if it differs from DESIGN.md).
   - The exact commands used to verify correctness.
   - Updated documentation if you changed runtime behaviour.

---

## Test Coverage Requirements

| Change type | Required tests |
|---|---|
| New kernel op or collective | Numerics test vs fp64 reference (`atol=rtol=1e-5`) |
| New distributed primitive | `mp.spawn` multi-process test firing real collectives |
| New telemetry field | `test_telemetry.py` REQUIRED_KEYS addition + round-trip test |
| New config key | `test_smoke_e2e.py` consumption test |
| New elastic behaviour | File-I/O round-trip in `test_elastic_v02.py` |
| Bug fix | Test that reproduces the bug before the fix |
| Documentation only | No code test required; verify links are accurate |

### Test file ownership

| Area | Primary test file |
|---|---|
| Router kernel numerics | `test_kernels_numerics.py` |
| Router kernel correctness | `test_kernels.py` |
| Routing quality metrics | `test_routing_quality.py` |
| Tensor parallelism | `test_tensor_parallel.py` |
| Pipeline parallelism | `test_pipeline_parallel.py` |
| EP / MoE layer | `test_distributed.py`, `test_distributed_invariants.py` |
| Elastic checkpointing | `test_elastic.py`, `test_elastic_v02.py` |
| MFU accounting | `test_mfu.py`, `test_mfu_v02.py` |
| Telemetry | `test_telemetry.py` |
| End-to-end | `test_smoke_e2e.py` |
| Chaos / TorchElastic | `test_chaos.py` |

---

## Code Standards

### Correctness over cleverness

Prefer explicit, readable code over terse cleverness. The test suite is the
safety net, not type annotations or assertions alone.

### Every new public function needs a docstring

```python
def _compute_load_imbalance(dispatch_cnt: torch.Tensor) -> float:
    """Compute expert load imbalance ratio.

    Parameters
    ----------
    dispatch_cnt : Tensor [E]
        Number of tokens dispatched to each expert.

    Returns
    -------
    float
        max_load / mean_load.  1.0 = perfect balance.
        Returns 1.0 when all counts are zero (no division by zero).
    """
```

### Assert with informative messages

```python
assert total_dispatched == expected_total, (
    f"Token conservation violation: dispatched={total_dispatched} "
    f"expected={expected_total} (N={N}, K={self.top_k})"
)
```

### No magic numbers without explanation

```python
BLOCK_N = 64   # 64 tokens per Triton block — fits in L1 cache with BLOCK_E=64
BLOCK_E = 64   # 64 experts per block — 64×64 float32 = 16 KiB < L1 32 KiB
```

### Collective correctness rules (critical)

When writing distributed code, use the correct collective for each pattern:

| Pattern | Correct collective | Wrong |
|---|---|---|
| Sum partial matmul outputs (RowParallel) | `all_reduce(SUM)` | `reduce_scatter + all_gather` |
| Replicate sharded output (ColumnParallel) | `all_gather_into_tensor` | `broadcast` |
| Shard sequence across TP group | `scatter` + `all_gather` | n/a |
| EP token dispatch | `all_to_all_single` | `all_gather` |

Include a comment explaining *why* the specific collective was chosen.
The `test_row_parallel_uses_all_reduce_not_reduce_scatter` test enforces
the RowParallel rule structurally — add similar structural tests for new collectives.

### Documentation alignment

If you change runtime behaviour, update the corresponding doc in `docs/`.
If you add a new config key, add it to `docs/SETUP_AND_OPERATIONS.md`.
If you add a new telemetry field, add it to the telemetry table in
`docs/SYSTEM_DESIGN.md` and the `StepRecord` docstring.

---

## Known Open Areas (Good First Contributions)

| Area | Difficulty | File | Roadmap |
|---|---|---|---|
| PP `dist.send`/`dist.recv` inter-stage wiring | Hard | `pkg/distributed/parallel_mesh.py` | v0.3 |
| Chaos Scenario A NCCL backend | Medium | `tests/test_chaos.py`, `tests/_chaos_worker.py` | v0.3 |
| SP all-gather fused with next projection | Hard | `pkg/distributed/parallel_mesh.py` | v0.4 |
| Real 8-GPU benchmark numbers | Medium (needs hardware) | `benchmarks/BENCHMARKS.md` | v0.3 |
| WandB / Weights & Biases integration | Easy | `pkg/telemetry/logger.py` | v0.3 |
| Expert capacity overflow re-routing | Medium | `pkg/distributed/parallel_mesh.py` | v0.4 |
| Nsight/CUPTI kernel profiling | Hard | new: `pkg/profiling/` | v0.3 |

---

## Security and Secrets

- Never commit credentials, tokens, or keys to source.
- Use environment variables for `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`,
  `S3_ENDPOINT_URL`.
- If you find a security issue, open a GitHub issue without including secret
  material. Describe the behaviour and reproduction steps only.

---

## CI Requirements

All PRs must pass:

1. `lint` — ruff + mypy (non-fatal for mypy until type coverage improves)
2. `unit` — full non-chaos suite on Python 3.10 and 3.11
3. `benchmark` — CPU benchmark smoke (no regressions)

Push to `main` additionally runs:

4. `docker` — Docker build smoke
5. `chaos` — Scenario B (blocking), Scenario A (non-blocking)

---

## License

Apache 2.0. See `LICENSE`. By submitting a PR you agree to license your
contribution under the same terms.
