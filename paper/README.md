# moe-engine Papers (v3, July 2026)

Two documents, both grounded in the verified facts in `RESULTS.md` and
`docs/SYSTEM_DESIGN.md` as of v0.3.3 — no invented numbers.

## `preprint_v3/` — Version 3 Preprint

Full-length academic preprint, single-column, continuing the numbering and
voice of `moe-engine-preprint-v2.pdf`. Reports:

- First real-GPU (T4) validation of the router kernel, including the
  `constexpr` compilation defect it surfaced
- The v0.3.2/v0.3.3 architectural refactor (monolith → 8 modules)
- The Pydantic silent-degradation CI bug, reported as a case study
- Expert capacity enforcement (Switch Transformer / GShard style)
- Full evaluation section with real T4 numbers (80.1× speedup, Chaos B
  100%, token conservation 0/100)
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
| 80.1× GPU speedup at N=4096 | `RESULTS.md`, `benchmarks/BENCHMARKS.md`, `benchmarks/gpu_results.json` |
| Chaos B 10/10 (100%) | `RESULTS.md`, `tests/test_chaos.py` |
| Chaos A ~85% | `roadmap.md`, `docs/SYSTEM_DESIGN.md` Known Limitations |
| 348 passing tests, 21 files | `docs/SYSTEM_DESIGN.md`, verified via `pytest --collect-only` |
| `constexpr` defect | `docs/adr/ADR-001-triton-router-kernel.md` |
| Pydantic silent-degradation bug | `pkg/utils/config.py` history, this session's CI-fix work |
| `_cumcount` / capacity dropping | `pkg/distributed/moe_layer.py`, `tests/test_capacity_dropping.py` |
| Module decomposition (8 modules) | `pkg/distributed/`, `docs/adr/ADR-004-4d-parallelism-composition.md` |

If any number in the codebase changes in a future release, these two
documents should be updated to match — they are not meant to drift from
the artifact they describe.
