# MoE Dispatch Correctness Audit — August 2026

**Scope:** Direct follow-up to `docs/audits/2026-08-tensor-parallel-audit.md`'s
recommendation — the `test_distributed_invariants.py` items in
`roadmap.md`'s "2-Rank CI Blind Spot" table. Full technical writeup:
`docs/adr/ADR-009-moe-dispatch-correctness.md`. This doc is the shorter
process/verification companion.

## What was found, in the order it was found

1. Both `test_distributed_invariants.py` tests crashed with the identical
   error (`Split sizes doesn't match total dim 0 size`) inside
   `DistributedMoELayer.forward()`'s EP dispatch — strongly suggesting a
   shared root cause, confirmed by tracing both to the same code.
2. Root cause: `send_counts[ep_r] = mask.sum() * top_k` overcounts whenever
   a token's `top_k` experts span multiple EP ranks. Confirmed concretely
   with a small reproduction (`N=8, top_k=2, ep_size=2`): computed
   `send_counts.sum()=24` against an actual dispatch-item count of 16.
3. Fixing the count alone wasn't sufficient — the underlying dispatch
   *grouping* (sort by `idx[:, 0]` only, repeat-interleave) doesn't
   correctly route `top_k>=2` slots to their individual destination ranks
   either. Both needed fixing together.
4. Designing the fix's local expert-selection step surfaced a **third,
   independent, more severe bug**: the code divided received tokens into
   *equal-sized* chunks, one per local expert — correct only under
   perfectly balanced routing. Confirmed concretely with an unbalanced
   4-expert routing (actual counts `[7, 6, 2, 5]`): the wrong expert
   processed many tokens. This bug is independent of `ep_size` entirely —
   reproduces at `ep_size=1`, no distributed setup at all. Direct
   verification: `DistributedMoELayer`'s output diverged from an
   independent per-token reference by ~1.95 absolute, on a plain
   single-process forward pass with no parallelism whatsoever.
5. No test anywhere in the codebase checked per-token correctness before
   this pass — existing tests (`test_capacity_dropping.py`,
   `test_mock_dist.py`, `test_smoke_e2e.py`) only checked shapes,
   NaN-absence, and drop-fraction bounds.

## Changes made

1. `pkg/distributed/moe_layer.py` — rewrote `DistributedMoELayer.forward()`'s
   dispatch/expert-compute/combine section: dispatch items are now
   `(token, k)` pairs routed independently; `send_counts` counts per-item,
   not per-token; received rows are tagged with their expert id via a
   second small all_to_all; local expert selection uses boolean masking on
   the actual received expert ids instead of equal-division slicing.
2. `tests/test_moe_layer.py` — new file (none existed before). 14 tests:
   per-token reference comparison across 6 parametrized shapes chosen to
   make balanced routing implausible, a 20-seed sweep, a gradient-routing
   test, `send_counts` arithmetic in isolation, and a genuine 2-rank
   `mp.spawn` test against a correctly seed-matched reference.
3. `docs/adr/ADR-009-...md` — new ADR.
4. `docs/adr/README.md`, `roadmap.md`, `docs/testing.md`, `RESULTS.md`,
   `docs/SYSTEM_DESIGN.md` — index/status updates. Tier-0 count: 380 → 393.

## A construction subtlety worth recording

The 2-rank test's reference had to mirror exactly how
`DistributedMoELayer.__init__` consumes its RNG: each rank independently
re-seeds and builds only its own local experts, which means (with the same
seed everywhere) rank 1's local experts are bit-identical to rank 0's at
the same local index — not the distinct weights a naive single sequential
build across all experts would produce. The reference
(`_build_seed_matched_reference`) replicates this per-rank pattern
explicitly. Separately, `build_topology` validates against the real
`dist.get_world_size()` once `init_process_group` has been called in a
process, so the reference has to be built *before* that call in each
worker, not after.

## Verification discipline

```
$ # direct diagnostic before any fix
$ python3 -c "... send_counts.sum()=24 vs tokens_sorted.shape[0]=16 ..."
MISMATCH confirmed

$ # per-token reference check, ep_size=1, before the fix
max_diff (buggy code vs manual per-token reference): 1.9511678218841553
Confirms bug

$ # after the fix
max_diff: 4.76837158203125e-07

$ # all new tests reverted against the pre-fix code
$ pytest tests/test_moe_layer.py -q
9 failed, 5 passed  # the 5 passes are the send_counts arithmetic tests,
                     # which check the correct formula in isolation and
                     # don't call into the model -- expected to pass
                     # regardless of which version of moe_layer.py is present
$ # fix restored
$ pytest tests/test_moe_layer.py -v
14 passed

$ pytest tests/test_distributed_invariants.py -v
2 passed  # both previously-crashing tests, at real 4-rank execution

$ pytest tests/ --ignore=tests/test_chaos.py --ignore=tests/test_smoke_e2e.py \
    -m cpu -k "not (2rank or multiprocess or distributed_invariants)"
393 passed, 3 skipped, 16 deselected, 1 xfailed

$ ruff check / ruff format --check
All checks passed! / 2 files already formatted
```

## Explicitly deferred

Two "2-Rank CI Blind Spot" items remain open: both
`test_pipeline_parallel.py` 2-rank tests. Not investigated in this pass.
One shows a different-looking symptom (an IPv6/IPv4 `localhost` addressing
error) than anything found in ADR-007/008/009 — flagged as plausibly a
different root cause, not assumed to be related without evidence.

## Limitations

Same as every prior audit in this series: CPU/Gloo `mp.spawn` only for the
2-rank test; the more severe finding (equal-chunk expert selection) needed
no distributed setup at all to verify. No real GPU or NCCL claim made.
