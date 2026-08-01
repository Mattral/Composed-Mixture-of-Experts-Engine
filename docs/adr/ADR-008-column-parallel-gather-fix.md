# ADR-008: `ColumnParallelLinear` Gather-Dimension Bug and the Missing `gather_output` Mode

**Status:** Accepted
**Date:** August 2026
**Deciders:** Min Htet Myet
**Supersedes:** —
**Related:** ADR-007 (Sequence Parallelism correctness — Finding 3 in that ADR named this exact bug as characterized-but-not-fixed; this ADR is that fix), ADR-004 (4D Parallelism Composition — this ADR's evidence contradicts its "✅ 2-rank TP mp.spawn verified" claim, which ADR-004 itself is not edited to reflect, per this project's ADR-immutability rule)

## Context

ADR-007 characterized (but explicitly did not fix) a bug in
`ColumnParallelLinear.forward()`: it calls `dist.all_gather_into_tensor`
expecting concatenation along the *last* dimension (`out_features`), but
that collective concatenates along dimension 0 only. This is the same root
misunderstanding ADR-007 fixed in `gather_from_sequence_parallel`, in a
different dimension.

Fixing it surfaced a second, more consequential bug underneath: once the
gather itself was correct, the `2-rank` composition test
(`test_column_row_parallel_2rank_numerically_correct`) still failed —
`RowParallelLinear` received a full-width `(B, out_features)` tensor from
`ColumnParallelLinear`, but its weight expects a sharded
`(B, out_features // tp_size)` input. This is not a bug in
`RowParallelLinear` (its `all_reduce` is genuinely correct — partial sums
over a feature-sharded input, verified by `test_row_parallel_uses_all_reduce_not_reduce_scatter`
and unaffected by anything in this ADR). The bug is that
`ColumnParallelLinear` had no way to *not* gather.

This matters far beyond the test: `_SwiGLUExpert`
(`pkg/distributed/moe_layer.py`) — the actual expert FFN used throughout
`DistributedMoELayer`, i.e. the real model code every MoE expert in this
codebase runs — composes `w_gate`/`w_up` (`ColumnParallelLinear`) directly
into `w_down` (`RowParallelLinear`):

```python
return self.w_down(torch.nn.functional.silu(self.w_gate(x)) * self.w_up(x))
```

`_SwiGLUExpert`'s own docstring already claims the correct design: *"Both
`w_gate` and `w_up` are `ColumnParallelLinear` so their outputs have shape
`[F // tp_size]` on each rank. The element-wise multiply happens in that
sharded space."* That was true of the intent, never true of the code —
`ColumnParallelLinear` always gathered, so `w_down` would receive a
full-width tensor and crash on a shape mismatch. This was invisible because
`test_swiglu_expert_numerically_correct_tp1`, the only numerical-correctness
test for this class, runs at `tp_size=1` only — where the gather is a no-op
and the mismatch cannot occur.

This is a materially larger finding than a broken unit test: it means the
actual expert compute path in this codebase's Mixture-of-Experts model has
never worked at real tensor-parallel scale (`tp_size > 1`), which is the
entire point of tensor parallelism.

## Decision

1. **Fix the gather dimension** — same technique as ADR-007's
   `gather_from_sequence_parallel` fix: move the sharded dimension
   (`out_features`, the *last* dim here rather than the *middle* one SP's
   sequence dimension was) to position 0 via `movedim(-1, 0)`, gather there,
   move it back.

2. **Add `gather_output: bool = True` to `ColumnParallelLinear`** — the
   standard Megatron-LM convention for exactly this situation. Defaults to
   `True` (the only behavior that existed before this ADR) for backward
   compatibility with anything expecting the full gathered output. Set to
   `False` to return the local `out_features // tp_size` shard directly,
   with zero collectives — the correct choice when feeding straight into a
   `RowParallelLinear`.

3. **`bias=True` combined with `gather_output=False` raises
   `NotImplementedError`**, not a silent guess. The mathematically correct
   behavior would add each rank's *shard* of bias, but `self.bias` is
   currently allocated full-width (`out_features`) on every rank — sharding
   it correctly is a real (small) design question with no current caller
   and no test to verify it against. No current usage needs this
   combination (`_SwiGLUExpert`, the only `gather_output=False` caller,
   uses `bias=False` throughout). Implementing an unverified bias-sharding
   scheme here would repeat exactly the pattern ADR-006 and ADR-007 both
   found and corrected — a plausible-looking fix nobody actually checked.

4. **`_SwiGLUExpert.w_gate`/`w_up` now pass `gather_output=False`** — this
   is the fix that makes the class's own docstring true. `w_down` is
   unchanged (`RowParallelLinear` was never the buggy half).

5. **Test coverage**: the existing 2-rank composition test now correctly
   uses `gather_output=False` (previously it queried a code path that
   didn't exist, discovered while fixing this — see "How this was found"
   below). Four new tests close gaps the investigation surfaced along the
   way: `gather_output=True` standalone at 2-rank (nothing else exercises
   this path at real multi-rank — confirmed by temporarily reverting just
   the dimension fix and observing the rest of the file's tests still all
   passed), `gather_output=False` standalone at 2-rank, the
   `bias=True`-raises guard at real `tp_size>1` specifically, and —
   the most important addition — `_SwiGLUExpert` itself at genuine 2-rank
   TP, closing the exact gap (tp_size=1-only testing) that let the
   composition bug ship. All four were verified against the pre-fix code
   before being trusted (temporarily reverted, confirmed failing, restored).

## How this was found

Fixing the gather-dimension bug (item 1) alone was not sufficient to make
the existing 2-rank test pass — it progressed to a *different* error
(`RowParallelLinear` shape mismatch), which is what led to discovering item
2. Then adding a `gather_output=True` standalone test (to specifically
cover the path the composition test no longer exercised) required
confirming, empirically, that no other test in the file reached real
multi-rank collectives for that path — done by temporarily reverting the
dimension fix and re-running the file's other tests, which all still
passed, confirming the gap rather than assuming it. This layered-discovery
pattern — fix one bug, hit a different failure, find a second bug, then
verify the fix doesn't just move the blind spot elsewhere — is the same
shape ADR-007 went through with `scatter_to_sequence_parallel` and
`gather_from_sequence_parallel`.

## Consequences

**Positive:**
- `ColumnParallelLinear` is now correct at real multi-rank in both modes.
- `_SwiGLUExpert` — the actual expert compute used throughout
  `DistributedMoELayer` — now genuinely works at `tp_size > 1`, verified by
  a dedicated 2-rank test rather than inferred from lower-level primitive
  tests passing.
- The `gather_output` API matches an established, well-known convention
  (Megatron-LM), rather than inventing a bespoke one.

**Negative / neutral:**
- ADR-004's "✅ 2-rank TP mp.spawn verified" claim is now known to have been
  false as written (the code, not necessarily ADR-004's original testing
  claims — this ADR does not investigate what actually happened at the time
  ADR-004 was accepted). Per this project's immutability rule, ADR-004
  itself is not edited; this ADR and `roadmap.md`'s "2-Rank CI Blind Spot"
  section are the correction of record.
- `bias=True` + `gather_output=False` remains unimplemented. If a future
  `ColumnParallelLinear` user needs both, that's real (if small) design
  work, not something this ADR provides.
- Not run on real multi-GPU hardware or NCCL — verified via CPU/Gloo
  `mp.spawn` only, same constraint as every other distributed-correctness
  claim in this codebase's CPU-only tier.
- Three of the four remaining "2-Rank CI Blind Spot" failures from ADR-007
  (both `test_pipeline_parallel.py` tests, both
  `test_distributed_invariants.py` tests) are untouched by this ADR.
