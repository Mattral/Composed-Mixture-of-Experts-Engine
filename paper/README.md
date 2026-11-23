# moe-engine Papers

Working drafts. Not peer reviewed. The figures in these drafts predate, and are superseded by, `README.md`, `RESULTS.md`, and `CHANGELOG.md`.

The `.tex` sources and compiled PDFs are kept in place. They were not edited in the documentation update that recorded the current figures.

## Where the drafts differ from the code (as of v0.3.4, 2026-11-24)

Both drafts describe the codebase at v0.3.3 or earlier. On these points they no longer match the code and tests, or leave out a change that matters:

| Topic | Drafts say | Current status |
|---|---|---|
| Multi-process pipeline parallelism | Validated by 2-rank `mp.spawn` tests | Those tests failed until 2026-11-23. Since the F16 fix, 2- and 3-stage runs match a dense reference on CPU/Gloo, in tests that no CI job runs (`CHANGELOG.md`, `roadmap.md`) |
| Sequence-parallel fusion | Backward all-gather replaced by a single all-reduce, validated at 2 ranks | The all-reduce was incorrect. The fused path for a replicated weight needs no collective at all (ADR-007) |
| EP, TP and SP gradients | Not discussed | Gradients were wrong whenever any parallel dimension was > 1, until the October 2026 fixes (ADR-010). They now match a dense reference on CPU/Gloo in push CI. The October work also changed the training objective and made the router losses train the router (ADR-011) |
| Capacity dropping | Applied per top-k slot; over-capacity slots get zero combine weight | Per expert across slots, slot-major, dropped before dispatch (ADR-011, F12) |
| Test counts | Counts at the time of writing | See `README.md`, "Test Suite" |

The router-kernel results (T4 and A100 speedups versus the CPU reference path) are unchanged.

## `preprint_v3/` — Version 3 draft

Full-length draft, single-column.

**Files:**
- `preprint_v3.tex` — LaTeX source (`article` class, no external `.cls` needed)
- `preprint_v3.pdf` — compiled, 9 pages

**To recompile:**
```bash
pdflatex preprint_v3.tex
pdflatex preprint_v3.tex   # second pass for cross-references
```

## `ieee_paper/` — IEEE-format draft (two-column)

Condensed draft in IEEE conference format (Abstract, Index Terms, Roman-numeral sections, numbered references, an algorithm block for the `cumcount` capacity-dropping primitive).

**Files:**
- `ieee_paper.tex` — LaTeX source (`IEEEtran` conference class)
- `ieee_paper.pdf` — compiled, 4 pages
- `IEEEtran.cls` — included for convenience; **not needed on Overleaf**, which has IEEEtran built in. Included here only so this paper compiles standalone outside Overleaf too.

**To use in Overleaf:**
1. Create a new blank project.
2. Upload `ieee_paper.tex` (do **not** upload `IEEEtran.cls` — Overleaf already has it; uploading a second copy can cause a version conflict).
3. Compile with pdfLaTeX (Overleaf's default).

**To recompile locally:**
```bash
pdflatex ieee_paper.tex
pdflatex ieee_paper.tex   # second pass for cross-references
```
