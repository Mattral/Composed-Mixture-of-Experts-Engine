# moe-engine Papers

Working drafts. Not peer reviewed. The figures in these drafts predate, and are superseded by, `README.md`, `RESULTS.md`, and `CHANGELOG.md`.

The `.tex` sources and compiled PDFs are kept in place. They were not edited in the documentation update that recorded the current figures.

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
