# History rewrite

**Date:** 2026-10-30

**Reason:** Unify author identities and trim history to the MoE-engine project, dropping files that were never meant to be versioned.

## What changed

- History now starts at the first MoE-engine commit ("phase 1, initial features, not completed yet", 2026-02-24). The one earlier commit, a LICENSE-only "Initial commit" from 2024-12-09, is gone.
- `.claude/` and `docs/audits/2026-07-handoff-audit.md` are removed from every commit. Two commits that only touched those paths are gone too.
- `Co-authored-by: Claude` trailers are removed from commit messages.
- Author and committer identities are unified as `Min Htet Myet (Mattral)`.

File contents at every branch tip are otherwise unchanged. Every other commit has a new SHA.

## Old → new SHAs

Commits cited in tracked files:

| Old | New | Cited in |
|---|---|---|
| `d9cf60c72f4575fc46a332986c0af19fe60bdd44` | `37d3215760f61aeade1dd78c3d1527624a446a5e` | `CHANGELOG.md` |
| `edafea5ab679f3a0cbca9135f2d983244f52d20a` | `61e974a649cbbb1af7ad9d77447dcfe18b3c8ef6` | `docs/audits/2026-07-handoff-audit.md` (removed) |
| `a9b7627f5f6df49e002cce4aa922bdb274d1fade` | `b07278b1e108b633fb0a717d2f91c90b963ffe6e` | `docs/audits/2026-07-handoff-audit.md` (removed) |

Tags:

| Tag | Old | New |
|---|---|---|
| `v0.3.2` | `af971df6e491b1cebeeaf484469845c9127704d7` | `eab089c61834368f4b75a017407f901e1f2546d7` |

## Notes

- The Zenodo DOI archive ([10.5281/zenodo.20688837](https://doi.org/10.5281/zenodo.20688837)) still reflects the pre-rewrite snapshot. Its commit SHAs are the old ones above.
- Forks and existing clones keep the old history. Re-clone instead of pulling.
