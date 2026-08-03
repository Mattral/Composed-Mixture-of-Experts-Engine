# SP Correctness Audit — August 2026

**Scope:** Started as the roadmap's v0.4 P1 item "non-divisible sequence
length in SP." Expanded significantly after discovering the fused SP path's
one real-multi-rank test had never actually run in CI, and was broken. Full
technical writeup is in
`docs/adr/ADR-007-sequence-parallel-correctness-and-nondivisible-length.md`;
this doc is the shorter process/verification companion, in the same spirit
as `docs/audits/2026-07-handoff-audit.md`.

## What triggered the expansion

Implementing non-divisible sequence length required touching
`scatter_to_sequence_parallel`'s fused (`next_weight`) path. Before adding
to it, its existing 2-rank test was run directly (rather than assumed
correct because `roadmap.md` said "2-rank verified") and crashed with a
shape-mismatch `RuntimeError`. Root-causing that surfaced a second, more
fundamental bug in `gather_from_sequence_parallel` (broken for any batch
size > 1), and re-running the broader 2-rank/multiprocess test tier —
previously invisible, since every CI job filters those tests out by name —
surfaced four more pre-existing failures in unrelated files, not fixed in
this pass.

## Findings

| # | Item | Claimed | Actual | Resolution |
|---|---|---|---|---|
| 1 | SP fused path (`next_weight`) | "✅ halves SP collectives; 2-rank verified" (roadmap.md) | ❌ Never ran in CI; algorithm was wrong (all_reduce summed different tokens' projections) | **Fixed** — zero-collective local projection is the correct answer for a replicated weight; see ADR-007 |
| 2 | `gather_from_sequence_parallel` | Implicitly assumed correct (no test ever exercised it at real multi-rank, any batch size) | ❌ Broken for `B>1`: `all_gather_into_tensor` concatenates along dim 0 only, buffer was allocated assuming concatenation along the sequence dim | **Fixed** — transpose-gather-transpose |
| 3 | `ColumnParallelLinear` (`tensor_parallel.py`) | ADR-004: "✅ 2-rank TP mp.spawn verified" | ❌ Same root-cause bug as #2, different dimension | **Characterized, not fixed** — different file, out of this pass's scope |
| 4 | PP 2-rank tests (`test_pipeline_parallel.py`, 2 tests) | ADR-004: "✅ 2-rank PP mp.spawn verified" | ❌ Both fail (worker crash; micro-batch count mismatch) | **Not investigated** — error signatures captured only |
| 5 | `test_distributed_invariants.py` (2 tests) | Implied passing (no contrary claim found, but never run) | ❌ Both fail, `Split sizes doesn't match total dim 0 size` | **Not investigated** — error signatures captured only |
| 6 | The 2-rank/multiprocess test tier generally | Implied to "run automatically in the full suite" (`docs/testing.md`, prior wording) | ❌ No CI job anywhere runs it | **Corrected in docs/testing.md**; recommended a scheduled CI job in ADR-007 (not implemented) |

Findings 3–5 were deliberately not root-caused in this pass — each is
plausibly its own investigation of comparable scope to what fixing SP took
here, and bundling them in risked doing all of them shallowly rather than
any of them properly.

## Changes made this pass

1. `pkg/distributed/sequence_parallel.py` — fixed the fused-path all_reduce
   bug (Finding 1), fixed the gather batch-corruption bug (Finding 2), added
   non-divisible sequence length support (`_sp_shard_layout` helper,
   zero-padding in scatter, `orig_seq_len` stripping in gather).
2. `tests/test_sequence_parallel_v03.py` — rewritten. 26 tests (up from 10):
   pure-function padding-arithmetic tests, corrected 2-rank fused-path
   comparison (was comparing a shard against the full reference — a second,
   independent bug in the test itself), a zero-collectives regression guard
   (mock-spied), 2-rank non-divisible round-trip tests, and a batch-corruption
   regression test isolated from the non-divisible cases. All three new
   regression tests were verified to actually fail against the pre-fix code
   (temporarily reverted, confirmed failing, restored) before being trusted
   — the same discipline applied to the false-positive catch in
   `test_gpu_portability.py` during the July pass.
3. `docs/adr/ADR-007-...md` — new ADR, full technical writeup.
4. `docs/adr/README.md` — ADR-007 index row.
5. `roadmap.md` — corrected the false "halves collectives; 2-rank verified"
   claim; marked non-divisible sequence length done; added a new "2-Rank CI
   Blind Spot" section documenting Findings 3–5 prominently, as P0.
6. `docs/testing.md` — corrected the false "run automatically in the full
   suite" claim for the 2-rank/multiprocess tier; added per-test status.
7. Test-count propagation (368 → 380) across `docs/SYSTEM_DESIGN.md`,
   `docs/testing.md`, `docs/CONTRIBUTING.md`, `docs/LIMITED_HARDWARE_GUIDE.md`,
   `docs/deployment.md`, `docs/quickstart.md`, `RESULTS.md`, `roadmap.md` —
   same discipline as the July pass: living docs updated, historical
   changelog lines in `RESULTS.md` preserved with a new entry added rather
   than overwritten, `paper/` left untouched (same reasoning as before —
   frozen, already-published artifacts).

## Explicitly deferred

- Findings 3, 4, 5 above (`ColumnParallelLinear`, both PP 2-rank tests, both
  `distributed_invariants` tests) — captured as error signatures in
  `roadmap.md`'s "2-Rank CI Blind Spot" section and ADR-007. Recommended
  next priority.
- Adding an actual CI job that runs the 2-rank/multiprocess tier (ADR-007's
  recommendation) — a CI/infra change, not made unilaterally in this pass.
- The Megatron-style `RowParallelLinear + SP` reduce-scatter fusion pattern
  — a genuinely different, more advanced feature; not needed to fix what
  was broken, and not implemented.

## Verification

```
$ pytest tests/test_sequence_parallel_v03.py -v
26 passed

$ pytest tests/ --ignore=tests/test_chaos.py --ignore=tests/test_smoke_e2e.py \
    -m cpu -k "not (2rank or multiprocess or distributed_invariants)"
380 passed, 3 skipped, 11 deselected, 1 xfailed

$ ruff check pkg/distributed/sequence_parallel.py tests/test_sequence_parallel_v03.py
All checks passed!
$ ruff format --check pkg/distributed/sequence_parallel.py tests/test_sequence_parallel_v03.py
2 files already formatted
```

Additionally, and specifically because this pass's own finding was "a test
existed but never actually ran / wasn't actually checking the right thing"
— the same failure mode as ADR-006 — each of the three new regression
tests (`test_sp_fused_2rank_no_collective_used`,
`test_sp_nondivisible_2rank_round_trip`,
`test_sp_gather_2rank_batch_dimension_not_corrupted`) was verified against
the pre-fix code before being trusted:

```
$ # pkg/distributed/sequence_parallel.py temporarily reverted to the
$ # pre-fix gather implementation
$ pytest tests/test_sequence_parallel_v03.py -k "batch_dimension or nondivisible or fewer_tokens"
3 failed  # confirmed genuine failures, not false passes
$ # fix restored
$ pytest tests/test_sequence_parallel_v03.py -v
26 passed
```

## Limitations of this audit

Same CPU-only, no-GPU, no-multi-node constraints as the July pass. All
fixes here were verified via Gloo `mp.spawn` on CPU, not real NCCL or real
multi-GPU hardware. Findings 3–5 are reported as symptoms (error messages),
not diagnosed to root cause — treating them as "probably the same bug
pattern as Finding 1/2" without actually verifying that for each one would
be exactly the kind of unverified claim this project's philosophy exists to
avoid, so this doc does not make that claim.
