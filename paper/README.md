# moe-engine Papers (v3, July 2026)

Superseded by the repository README and the Documentation entry in `CHANGELOG.md`: the papers' fault-tolerance wording, superlinear speedup wording, Scenario A pass-rate wording (~85% stated as the rate, with the Gloo race stated as the cause), and test-count wording (148→348). The `.tex` sources and compiled PDFs are unchanged apart from a superseded note at the top of each `.tex` file. Current status is single-GPU (T4, A100) and 2-rank CPU. Multi-node and 8+ GPU runs remain open.

Two documents. Numbers below are the claims as written in the papers. Where a claim changed, the README and `CHANGELOG.md` are the current text.

## `preprint_v3/` — Version 3 Preprint

Full-length academic preprint, single-column, continuing the numbering and
voice of `moe-engine-preprint-v2.pdf`. Reports:

- First real-GPU (T4) validation of the router kernel, including the
  `constexpr` compilation defect it surfaced
- The v0.3.2/v0.3.3 architectural refactor (monolith → 8 modules)
- The Pydantic silent-degradation CI bug, reported as a case study
- Expert capacity enforcement (Switch Transformer / GShard style)
- Full evaluation section with T4 numbers (80.1× versus the CPU reference path at N=4096; no GPU baseline comparison yet; the T4 host CPU baseline in that table is 0.056 M tok/s. Chaos B 10/10 is a manual result, not push CI. Token conservation 0/100)
- Honest limitations section (multi-node data, Chaos A, capacity at scale)

**Files:**
- `preprint_v3.tex` — LaTeX source (`article` class, no external `.cls` needed)
- `preprint_v3.pdf` — compiled, 9 pages

**To recompile:**
```bash
pdflatex preprint_v3.tex
pdflatex preprint_v3.tex   # second pass for cross-references
```

## `ieee_paper/` — IEEE Conference Paper (two-column)

Condensed systems paper in IEEE conference format, covering the same
technical contributions with IEEE structure (Abstract, Index Terms,
Roman-numeral sections, numbered IEEE-style references, an algorithm
block for the `cumcount` capacity-dropping primitive).

**Files:**
- `ieee_paper.tex` — LaTeX source (`IEEEtran` conference class)
- `ieee_paper.pdf` — compiled, 4 pages
- `IEEEtran.cls` — included for convenience; **not needed on Overleaf**,
  which has IEEEtran built in. Included here only so this paper compiles
  standalone outside Overleaf too.

**To use in Overleaf:**
1. Create a new blank project.
2. Upload `ieee_paper.tex` (do **not** upload `IEEEtran.cls` — Overleaf
   already has it; uploading a second copy can cause a version conflict).
3. Compile with pdfLaTeX (Overleaf's default).

**To recompile locally:**
```bash
pdflatex ieee_paper.tex
pdflatex ieee_paper.tex   # second pass for cross-references
```

## Consistency with the codebase

Every number in both documents traces to a file in this repository:

| Claim | Source |
|---|---|
| 80.1× versus the CPU reference path at N=4096 (T4). No GPU baseline comparison yet. | `RESULTS.md`, `moe-engine/benchmarks/gpu_results_t4.json`. The paper text also says the speedup increases super-linearly. That wording is superseded by the README and `CHANGELOG.md`. |
| Chaos B 10/10 (100%) | `RESULTS.md`. Manual run. The Chaos job is `workflow_dispatch` only and does not run on push CI. |
| Chaos A ~85% in the paper text | Current docs: ~17 of 20 runs. Failures are attributed to a Gloo race in containers. The paper's ~85% and root-cause wording is superseded by the README and `CHANGELOG.md`. |
| Tier-0: 416 passed, 3 skipped, 16 deselected, 1 xfailed. Differential: 243 passed. 28 files, 685 collected items. The 38 config tests are inside the Tier-0 count. | `docs/SYSTEM_DESIGN.md`, `docs/testing.md`. GPU, Docker, and Chaos jobs are manual and were skipped on the latest push. |
| `constexpr` defect | `docs/adr/ADR-001-triton-router-kernel.md` |
| Pydantic silent-degradation bug | `pkg/utils/config.py` history, this session's CI-fix work |
| `_cumcount` / capacity dropping | `pkg/distributed/moe_layer.py`, `tests/test_capacity_dropping.py` |
| Module decomposition (8 modules) | `pkg/distributed/`, `docs/adr/ADR-004-4d-parallelism-composition.md` |

If any number in the codebase changes in a future release, these two
documents should be updated to match — they are not meant to drift from
the artifact they describe.
