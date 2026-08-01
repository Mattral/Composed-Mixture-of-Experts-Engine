# ADR-007: Sequence Parallelism Correctness Fixes, Non-Divisible Sequence Length Support, and the 2-Rank CI Blind Spot

**Status:** Accepted
**Date:** August 2026
**Deciders:** Min Htet Myet
**Supersedes:** —
**Related:** ADR-004 (4D Parallelism Composition), ADR-006 (device-coercion correction — the same general pattern: a claimed-verified fix that a permanent-test-coverage pass found wasn't actually true)

## Context

This ADR started as the roadmap's v0.4 P1 item "non-divisible sequence
length in SP." Implementing it required touching
`scatter_to_sequence_parallel`'s fused (`next_weight`) path, which is where
things expanded considerably.

### Finding 1: the fused SP path's all_reduce was never correct

`docs/adr/README.md`'s v0.3 entry and `roadmap.md` both describe the fused
path as *"halves SP collectives; 2-rank verified."* It was not verified —
`test_sp_fused_2rank_numerically_correct`, the one test that exercises it
at real `tp_size>1`, crashed with a shape-mismatch `RuntimeError`. It had
never been run in CI: every job in `.github/workflows/ci.yml` filters out
tests with `2rank` in the name, with no other job picking them up anywhere
in the pipeline.

Working through the actual math: the fused path shards `x` by *sequence*
(each rank holds different tokens), projects through a *replicated* weight
(identical on every rank — confirmed by how the test constructs it, same
seed on every rank, no per-rank sharding), then `all_reduce(SUM)`s the
result. That sums *different tokens'* projections together at each local
index — not a reconstruction of anything, and generally not even
shape-compatible once ranks' shards can differ in size (which non-divisible
support makes routine, not exceptional).

For a per-token operation (a plain `Linear`, no cross-token mixing) applied
with a *replicated* weight to an *already sequence-sharded* input, the
correct answer requires **zero communication**: each rank's local
projection of its own tokens is already complete and correct on its own.
There's nothing to reduce.

I checked whether the intended design might instead be Megatron's real
`RowParallelLinear + SP` reduce-scatter fusion (row-sharded weight, partial
per-rank contributions to the *same* output, reduce+shard in one collective)
— genuinely useful, and it would explain wanting a collective here. But that
needs a differently-shaped `next_weight` (row-sharded, different per rank),
which contradicts how the existing tests construct `w`, and nothing in the
codebase (`RowParallelLinear`, `moe_layer.py`, any ADR) suggests it was ever
wired up this way. No ADR documents this feature's design at all — it never
got the scrutiny the rest of ADR-004's parallelism decisions did, which is
plausibly how the bug went unquestioned.

### Finding 2: `gather_from_sequence_parallel` was broken for batch size > 1

Fixing Finding 1 and adding non-divisible-length tests required exercising
`gather_from_sequence_parallel` at real `tp_size>1` — which, it turned out,
no passing test had ever done at any batch size. `dist.all_gather_into_tensor`
concatenates along dimension 0 only (it treats its input as a flat per-rank
buffer and places each rank's whole buffer at a flat offset in the output).
The function allocated its output buffer as `(B, S_local*tp_size, H)` —
correct only when `B == 1`, since only then does dim-0 concatenation
coincide with "concatenate along the sequence dimension" under row-major
layout. For any real batch size, this either crashes on a shape mismatch
(what every case in this session's testing happened to do) or, for
specific `(B, S_local, tp_size)` combinations where the shapes coincidentally
match, would silently interleave data across the batch dimension instead of
crashing — a more dangerous failure mode that this session did not happen to
hit, but the arithmetic doesn't rule out.

This is a **more fundamental problem than Finding 1**: `gather_from_sequence_parallel`
is the primary, non-fused reconstruction primitive. Every real use of
Sequence Parallelism depends on it, not just the fused-path optimization.

### Finding 3 (identified, not fixed here): the same root cause elsewhere

Characterizing Finding 2 while diagnosing it surfaced a structurally
identical bug in `ColumnParallelLinear.forward()`
(`pkg/distributed/tensor_parallel.py`): it also calls
`dist.all_gather_into_tensor` expecting concatenation along the *last*
(feature) dimension, which is a different mismatch from Finding 2's but the
same root misunderstanding of that collective's actual semantics.

Re-running the full (previously CI-invisible) 2-rank/multiprocess test
suite surfaced four more pre-existing failures, not touched in this pass:

| Test | File | Symptom |
|---|---|---|
| `test_column_row_parallel_2rank_numerically_correct` | `test_tensor_parallel.py` | `ColumnParallelLinear`'s all_gather shape mismatch (Finding 3) |
| `test_pp_multiprocess_2stage_activation_flow` | `test_pipeline_parallel.py` | Worker crash, `Connection closed by peer` (secondary symptom of a rank-0 failure) |
| `test_pp_multiprocess_correct_micro_batch_count` | `test_pipeline_parallel.py` | Expected 6 micro-batch outputs, got 0 |
| `test_token_conservation_distributed` | `test_distributed_invariants.py` | `RuntimeError: Split sizes doesn't match total dim 0 size` |
| `test_distributed_backward_no_nan` | `test_distributed_invariants.py` | Same `Split sizes` error as above |

None of these were investigated beyond capturing the error signature above
— each is plausibly its own root-cause investigation, and ADR-004's table
claims of *"✅ 2-rank TP mp.spawn verified"* and *"✅ 2-rank PP mp.spawn
verified"* are, per this evidence, not currently true of the code as it
stands, whatever was true when ADR-004 was written. This is the clear next
priority after this ADR, not something to fold into it.

### The systemic issue underneath all of this

Every one of `.github/workflows/ci.yml`'s pytest invocations filters out
`2rank or multiprocess or distributed_invariants` by name, with no other
job — scheduled, workflow_dispatch, or otherwise — running them anywhere.
This is a **blind spot affecting the entire multi-rank test tier**, not a
one-off gap. Six distinct tests across four files fail when actually run,
none of them from this session's changes. See "Consequences" below for a
recommendation.

## Decision

1. **`scatter_to_sequence_parallel`'s fused path**: removed the incorrect
   `all_reduce`. A replicated weight applied per-token to an already
   sequence-sharded input is correct with zero collectives — return the
   local projection directly.

2. **`gather_from_sequence_parallel`**: fixed via transpose-gather-transpose
   — move the sequence dimension to position 0 (where
   `all_gather_into_tensor`'s concatenation semantics actually apply),
   gather, then move it back. Correct for any batch size.

3. **Non-divisible sequence length**: `scatter_to_sequence_parallel` no
   longer raises `ValueError` when `S % tp_size != 0`. It zero-pads `x` up
   to `ceil(S/tp_size) * tp_size` before sharding (every rank gets an equal
   shard; only the last rank's may contain padding rows). Zero-padding is
   safe here specifically because every operation in this module is
   per-token — nothing mixes across the sequence dimension. The
   shard-boundary arithmetic is factored into a small pure function,
   `_sp_shard_layout(seq_len, tp_size) -> (chunk, pad_amount)`, deliberately
   unit-testable without `dist.init_process_group` (same rationale as
   isolating `_should_use_triton` in ADR-005/`test_gpu_portability.py`).

4. **`gather_from_sequence_parallel(..., orig_seq_len=None)`**: new optional
   parameter. When the original sequence length wasn't divisible by
   `tp_size`, passing it strips the padding after the all_gather. Omitting
   it when padding was added returns the padded-length result — a visibly
   wrong *shape*, not a silently wrong *value*, which was a deliberate
   design choice over trying to track padding implicitly.

5. **Test coverage** (`tests/test_sequence_parallel_v03.py`, rewritten):
   pure-function tests for `_sp_shard_layout` across evenly-divisible,
   non-divisible, and edge cases (fewer tokens than ranks); the corrected
   2-rank fused-path test (now comparing each rank's output against the
   matching *slice* of the reference, not the whole thing — comparing a
   shard against the full tensor was a second, independent bug in the
   original test, on top of the algorithm bug); a 2-rank test asserting the
   fused path issues **zero** collectives (a `unittest.mock.patch.object`
   spy on `all_reduce`/`all_gather`/`all_gather_into_tensor` — this is the
   test that would have caught the original bug fastest, as a direct
   assertion on the wrong collective being called, rather than downstream
   via a shape crash); 2-rank non-divisible round-trip tests, plain and
   fused; and a batch-dimension regression test isolated from the
   non-divisible cases (`B=4`, evenly-divisible `S`), so Finding 2 isn't
   only ever caught incidentally by the non-divisible tests. All three new
   regression tests were verified against the pre-fix code (temporarily
   reverted, confirmed to fail, then restored) before being trusted.

6. **`ColumnParallelLinear`'s analogous bug (Finding 3) is not fixed here.**
   Different file, different class, no test in this session's actual scope
   depends on it. Characterized and flagged, not silently left, and not
   silently expanded into.

7. **The other four 2-rank failures are not investigated here.** Flagged
   with their error signatures for a dedicated follow-up.

## Consequences

**Positive:**
- Sequence Parallelism's two core primitives are now provably correct at
  real multi-rank execution, for the first time — both fixes were verified
  to actually catch the bug they target (reverted, confirmed failing,
  restored), not just verified to pass afterward.
- Non-divisible sequence length is supported, cleanly, with an explicit,
  hard-to-misuse API (`orig_seq_len` is opt-in and fails loud, not silent,
  when forgotten).
- The "halves collectives" claim is replaced with a stronger, honestly
  provable one: zero collectives for the fused path under a replicated
  weight. This is a better result than what was originally (incorrectly)
  claimed, not a downgrade.
- A previously invisible, systemic CI gap is now on the record with
  concrete evidence (six failing tests, four files), rather than continuing
  to be invisible.

**Negative / neutral:**
- `roadmap.md`'s and ADR-004's "2-rank verified" claims for TP and PP are
  now known to be false as written, pending the follow-up work in Finding
  3's table. This ADR does not fix those; it establishes that they need
  fixing.
- The Megatron-style `RowParallelLinear + SP` reduce-scatter fusion — a
  genuinely different, more advanced pattern with a real "halve collectives"
  payoff — remains unimplemented. If a future need for tensor-parallel-sharded
  (not replicated) projections fused with SP arises, that's new design work,
  not something this ADR's fix provides for free.
- None of these fixes have been run on real multi-GPU hardware or under
  NCCL — verified via CPU/Gloo `mp.spawn` only, consistent with every other
  distributed-correctness claim in this codebase's CPU-only tier.

## Recommendation (not implemented here)

Add a CI job — scheduled (e.g. nightly/weekly) if per-commit cost is a
concern, but *some* job — that actually runs the `2rank`/`multiprocess`/
`distributed_invariants` tests currently filtered out of every existing
job. Six failing tests across four files were sitting invisible before this
session; the same class of gap will keep recurring silently otherwise, the
same way ADR-006 found a claimed-fixed bug that had quietly never been
fixed. A red nightly job that gets ignored is still strictly better than a
green per-commit job that was never checking the thing it implies it checks.
