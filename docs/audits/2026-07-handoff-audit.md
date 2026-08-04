# Handoff Audit — July 2026

**Scope:** Verification pass against `MOE_ENGINE_HANDOFF.md`, performed at
the start of a new session per that document's own instruction ("the first
real task in the new chat, before anything else, should be pulling a
fresh, current copy of the actual repo and diffing it against what this
document describes"). This doc records what was checked, how, and what
was found — so a future reader doesn't have to re-derive it from git
history or re-run the same checks from scratch.

**Why this doc exists:** the handoff document was explicit that it was
written after a sandbox reset, from memory and partial reconstruction, and
graded its own claims by confidence level rather than asserting uniform
certainty. That's a good practice; this doc is the follow-through on it —
closing the loop by actually checking, not just noting that checking was
recommended.

---

## Method

A fresh copy of the repo (the uploaded
`Composed-Mixture-of-Experts-Engine-main.zip`) was extracted into an
isolated environment, a venv was built from `requirements.txt` /
`pyproject.toml` (torch 2.13.0+cu130 CPU build, triton 3.7.1 — no GPU
available in this environment; see Limitations below), and the package
was installed editable. From there:

- Every file the handoff's own "Section 6: files worth re-reading fresh"
  list named was read in full, not skimmed.
- The documented Tier-0 test command was run verbatim:
  `pytest tests/ --ignore=tests/test_chaos.py --ignore=tests/test_smoke_e2e.py -m cpu -k "not (2rank or multiprocess or distributed_invariants)"`
- `ruff check` / `ruff format --check` were run — first without CI's flags
  (409 false-positive errors, purely from missing `--select`/`--line-length`
  — see Finding 6), then with them (clean, confirming the handoff's claim).
- Every doc file the handoff flagged as having a "pending edit of uncertain
  status" was grepped for the specific claimed content.
- `grep -rn` sweeps were run for stale version strings, since one gap
  (deploy tags) surfaced by inspection made it worth checking systematically
  rather than assuming it was the only instance.

## Findings

| # | Item | Handoff claimed | Actual state found | Resolution |
|---|---|---|---|---|
| 1 | Version consistency (`pkg.__version__`, `pyproject.toml`) | Consistent at 0.3.3 | ✅ Confirmed accurate | None needed |
| 2 | Tier-0 CPU test count | 348 passed, 1 skipped, 7 deselected, 1 xfailed | ✅ Exact match, verified by real run | None needed |
| 3 | `ruff check` / `ruff format --check` clean | Clean across `pkg/`, `train.py`, `scripts/`, `benchmarks/`, `tests/`, `examples/` | ✅ Confirmed accurate, but only reproducible with CI's exact flags (`--select E,F,W,I --ignore E501`, `--line-length 100`) — no `[tool.ruff]` config existed to make that the default | Added `[tool.ruff]` to `pyproject.toml` so bare `ruff check .` matches CI (Finding 6) |
| 4 | `moe_router.py` Fix 1 (`MIN_TRITON_DIM` guard, `_should_use_triton`) | Present, correct | ✅ Confirmed accurate | None needed; added permanent test coverage (was previously untested — see Finding 7) |
| 5 | `moe_router.py` Fix 3 (single-source-of-truth `used_triton`, `report` param, 5-value `backward()`) | Present, correct | ✅ Confirmed accurate | None needed; added permanent test coverage |
| 6 | ADR-005 content | Thorough, accurate | ✅ Confirmed accurate as a *decision record* | None needed |
| 7 | `docs/adr/README.md` index has an ADR-005 row | "Pending, unconfirmed" per handoff | ✅ Already present | None needed |
| 8 | Root `README.md` A100 badge + results section | "Pending, unconfirmed" per handoff | ✅ Already present | None needed |
| 9 | `docs/SYSTEM_DESIGN.md` Known Limitations table reflects two GPU architectures | "Pending, unconfirmed" per handoff | ✅ Already present | None needed |
| 10 | **`roadmap.md` reflects A100 validation** | "Pending, unconfirmed, likely still missing" per handoff | ❌ **Confirmed missing** — zero mentions of "A100" anywhere in either `roadmap.md` file | **Fixed** — see below |
| 11 | **`moe_router.py` Fix 2 (device coercion) lives in `_reference_route_fp64`** | High confidence, present and correct | ❌ **Not present in that function.** An equivalent-looking fix existed in an unrelated block (`MoERouter.forward`'s telemetry computation); `_reference_route_fp64` itself had no device coercion | **Fixed** — see ADR-006 for full writeup |
| 12 | **Deploy image tags bumped to v0.3.3** | Bug-fix table + roadmap.md both record this as done | ❌ **Still read v0.3.2** across `Makefile`, both k8s manifests, `docker-compose.yml`, `Dockerfile` comments | **Fixed** — mechanical bump, verified with a repo-wide grep afterward |
| 13 | (not in handoff — found independently) `train.py` auto-reduce-parallelism logic | N/A | Existing logic correctly reduces `dp`/`ep` to fit `WORLD_SIZE`, but did so with **no log line** — a config override happening silently | **Fixed** — added a `logging.warning` reporting requested vs. actual topology |
| 14 | (not in handoff — found independently) Two `roadmap.md` files diverge | N/A | `moe-engine/roadmap.md` ("Inner Roadmap") is stale and contradicts the root `roadmap.md` on several completed items (e.g. PP inter-stage wiring, comm/compute overlap telemetry) | **Fixed non-destructively** — added a superseded banner pointing to the root file; content left intact for history |
| 15 | (not in handoff — found independently) `make lint` scope narrower than CI | N/A | Makefile's `lint` target only covered `pkg/ train.py benchmarks/ scripts/`, omitting `tests/` and `examples/` that CI covers, and ran `ruff format --check` not at all | **Fixed** — scope now matches CI exactly |

### Finding 11 in detail (the one substantive bug)

This is the only finding above that's a correctness issue rather than
documentation debt, so it gets its own ADR rather than just a table row:
**`docs/adr/ADR-006-device-coercion-correction.md`**. Short version: ADR-005
described a device-mismatch fix as living inside `_reference_route_fp64()`;
the code that actually did that `.to(device=...)` coercion lived in a
different, unrelated function instead, and the routing hot path was
unprotected. Fixed in the correct location; a source-inspection regression
test (`tests/test_gpu_portability.py::test_reference_route_fp64_source_coerces_gate_w_device`)
now guards against it recurring silently. Full context, including *why*
this is recorded as a new ADR rather than a quiet patch, is in ADR-006
itself — this audit doc intentionally doesn't duplicate that reasoning.

Worth being precise about what this finding does and doesn't establish:
the fix restores the code to match what ADR-005 always claimed was true.
It has not been validated against a real cross-device call on actual
hardware — that needs 2+ GPUs in one job, which is the same blocker
Handoff Section 7.3 already identifies as the project's next unlock, not
something this audit resolves.

## Changes made this pass

1. `pkg/kernels/moe_router.py` — moved the device-coercion fix into
   `_reference_route_fp64()` (ADR-006).
2. `tests/test_gpu_portability.py` — new file, permanent pytest coverage
   for all three ADR-005 fixes (Handoff Section 7.1, Task 1). 20 CPU tests,
   2 honestly-skipped `@pytest.mark.gpu` tests. Building this file is what
   surfaced Finding 11 — a naive first draft of one of its own checks
   (string-matching for a removed function call) had a false-positive
   against explanatory comment text, and was corrected before being
   trusted; see that test's docstring.
3. `roadmap.md` — added the missing A100 section to "Honest Status
   Summary," updated the v0.4 table's "Real 8-GPU+ benchmark data" row to
   reflect that two single-GPU architectures are now validated.
4. `moe-engine/roadmap.md` — added a superseded banner (Finding 14).
5. `Makefile`, `deploy/k8s/training-job.yaml`,
   `deploy/k8s/training-job-multinode.yaml`, `deploy/docker/docker-compose.yml`,
   `deploy/docker/Dockerfile` — version tags bumped v0.3.2 → v0.3.3
   (Finding 12).
6. `pyproject.toml` — added `[tool.ruff]` / `[tool.ruff.lint]` matching
   CI's exact flags (Finding 3/15).
7. `Makefile` — `lint` target scope widened to match CI, added
   `ruff format --check` (Finding 15).
8. `train.py` — added a `logging.warning` when the auto-reduce-parallelism
   logic overrides the config's requested topology (Finding 13).
9. `docs/adr/ADR-006-device-coercion-correction.md` — new ADR (Finding 11).
10. `docs/adr/README.md` — added the ADR-006 index row.

## Explicitly deferred (not done this pass)

- **`paper/` left untouched, deliberately.** `paper/README.md`'s
  consistency table, `paper/preprint_v3/preprint_v3.tex`, and
  `paper/ieee_paper/ieee_paper.tex` all report "348 passing tests" (or
  "148 → 348" as the release-arc growth figure) as an empirical claim tied
  to the already-published/archived state of these papers (per prior
  session context, a Zenodo v2 archive with this preprint is live). These
  are frozen, point-in-time artifacts, not living documents — the same
  principle as ADR immutability applies: retroactively editing a
  published paper's reported numbers to match a later test count would
  misrepresent what was actually true when the paper was written, not fix
  a documentation gap. `paper/README.md`'s consistency table is correctly
  cross-referencing the papers' own claims against v0.3.3, so leaving its
  "348" row alone is what keeps it accurate — changing it to 368 would
  break that cross-reference, not fix it. (`RESULTS.md` and
  `docs/SYSTEM_DESIGN.md`, by contrast, are living documents describing
  current repo state, which is why those were updated.)
- **Non-divisible sequence length in SP** (Handoff Task 7.1.4 /
  `roadmap.md` v0.4 P1 item). This is real algorithm work, not a
  correctness gap or doc fix, and this pass was scoped to verification +
  low-risk fixes. Queued for the next iteration.
- **PR #18 (GitHub) disposition** (Handoff Task 7.1.3). Requires the
  repo owner's decision and write credentials this environment doesn't
  have — flagged back to Min rather than acted on unilaterally.
- **Anything in Handoff Section 7.2 or 7.3** (multi-GPU MFU validation,
  NCCL migration for Chaos Scenario A, real H100 run). All require 2+ real
  GPUs in one job, which this environment cannot provide. The genuine
  multi-GPU notebook and NCCL-backend chaos-harness work are unwritten
  engineering, not just "run it and record numbers" — worth flagging
  explicitly so this isn't mistaken for a task that GPU access alone
  unlocks.
- Pre-existing `mypy` errors surfaced by `make lint` (in `config.py`,
  `mesh.py`, `expert_parallel.py`, `moe.py`, `data_parallel.py`,
  `fault_monitor.py`) were left untouched. CI runs `mypy` with
  `continue-on-error: true` by explicit, documented design — this is not
  a gap, just noted here so a future reader doesn't rediscover the same
  question.

## Verification (end state of this pass)

```
$ pytest tests/ --ignore=tests/test_chaos.py --ignore=tests/test_smoke_e2e.py \
    -m cpu -k "not (2rank or multiprocess or distributed_invariants)"
368 passed, 3 skipped, 7 deselected, 1 xfailed

$ ruff check pkg/ train.py benchmarks/ scripts/ tests/ examples/
All checks passed!

$ ruff format --check pkg/ train.py benchmarks/ scripts/ tests/ examples/
65 files already formatted
```

(Test count rose from 348 → 368: the 20 new CPU-runnable tests in
`test_gpu_portability.py`. Skipped rose from 1 → 3: the pre-existing
Triton-GPU skip in `test_kernels_numerics.py`, plus this file's two new
`@pytest.mark.gpu` tests. Those two are *also* tagged `cpu` — inherited
from this file's module-level `pytestmark`, the same convention
`test_kernels_numerics.py` already uses for its one GPU test — so `-m cpu`
still collects them and they self-skip via an internal `pytest.skip()`
call rather than being marker-deselected. The `gpu` tag additionally makes
them selectable on their own via `pytest -m gpu`, for whenever the
dedicated GPU CI job runs.)


