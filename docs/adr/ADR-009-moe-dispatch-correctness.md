# ADR-009: `DistributedMoELayer` Dispatch Correctness — Three Compounding Bugs

**Status:** Accepted
**Date:** August 2026
**Deciders:** Min Htet Myet
**Supersedes:** —
**Related:** ADR-007, ADR-008 (same investigation thread — the "2-Rank CI Blind Spot" in `roadmap.md`). This ADR closes the `test_distributed_invariants.py` items in that table.

## Context

`test_distributed_invariants.py::test_token_conservation_distributed` and
`test_distributed_backward_no_nan` both failed with the same error,
`RuntimeError: Split sizes doesn't match total dim 0 size`, raised inside
`DistributedMoELayer.forward()`'s expert-parallel (EP) dispatch. Neither
test had ever run in CI (the systemic "2-Rank CI Blind Spot" ADR-007
identified). Root-causing the crash surfaced three compounding bugs, only
the first of which was needed to explain the crash — the other two are
silent, and more severe.

### Bug 1: `send_counts` overcounted whenever a token's `top_k` experts spanned multiple EP ranks

```python
mask = ((idx >= lo) & (idx < hi)).any(dim=-1)
send_counts[ep_r] = int(mask.sum().item()) * self.top_k
```

`mask` is true for a token if *any* of its `top_k` assigned experts falls
in rank `ep_r`'s range, but the count is then multiplied by the *full*
`top_k`, regardless of how many of that token's slots actually fall in
this range. A token whose 2 experts land on 2 different ranks gets counted
as contributing `top_k` items to *each* rank — `2 × top_k` total instead
of `top_k`. Confirmed concretely (`N=8, top_k=2, ep_size=2`):
`send_counts.sum()` came to 24 against an actual dispatch-item count of 16.
This is the direct cause of the crash — `dist.all_to_all_single`'s
`input_split_sizes` must sum to the input tensor's actual size.

### Bug 2: dispatch items weren't grouped by their own destination rank

```python
sort_order = idx[:, 0].argsort(stable=True)
tokens_sorted = flat[sort_order].repeat_interleave(self.top_k, dim=0)
```

Sorting by only `idx[:, 0]` (a token's *first* assigned expert) and then
repeating the token embedding `top_k` times assumes all `top_k` copies of
a token belong together at the same destination — true only when
`top_k=1`. For `top_k>1`, a token's `k>=1` slots can be destined for a
completely different EP rank than its `k=0` slot, and this construction
never accounts for that. Fixing Bug 1's counting alone would not have made
dispatch correct — the underlying grouping was wrong regardless of how
the counts were computed.

### Bug 3: local expert selection assumed perfectly balanced routing

```python
chunk = max(1, received.shape[0] // max(num_local, 1))
for i, _eid in enumerate(self.local_expert_ids):
    lo_e, hi_e = i * chunk, min((i + 1) * chunk, received.shape[0])
    expert_out[lo_e:hi_e] = self.experts[i](received[lo_e:hi_e])
```

This divides `received` into **equal-sized** chunks, one per local expert
— correct only if every local expert receives exactly the same number of
tokens, which essentially never happens with real (softmax-gated) routing.
Confirmed concretely with an unbalanced 4-expert routing (actual per-expert
counts `[7, 6, 2, 5]`): the equal-chunk slicing applied expert 1's FFN to
2 of expert 0's tokens and 3 of expert 1's actual tokens, and expert 2's
FFN never touched most of its own assigned tokens at all.

**This bug is independent of `ep_size` and of Bugs 1–2** — it reproduces
at `ep_size=1`, no distributed setup whatsoever, with only routing
imbalance (the normal case). Direct verification: `DistributedMoELayer`'s
output diverged from an independently-computed per-token reference by
**~1.95 in absolute terms** (not a numerical-precision-scale difference) on
a plain single-process forward pass. This means `DistributedMoELayer` has
never correctly computed its forward pass under realistic (imbalanced)
routing, at any parallelism configuration, in this codebase's history.

### Why none of this was caught

No test anywhere in the codebase checked `DistributedMoELayer`'s output
against a per-token reference. `test_capacity_dropping.py`,
`test_mock_dist.py`, and `test_smoke_e2e.py` check shapes, NaN-absence, and
drop-fraction bounds — never that the correct expert processed the correct
token. Bug 3 in particular required no distributed execution at all to
reproduce, so this gap was not about the 2-rank CI blind spot; it was
simply never checked.

## Decision

Rewrote the dispatch/expert-compute/combine section of
`DistributedMoELayer.forward()`:

1. **Treat each `(token, k)` pair as an independently-routable dispatch
   item.** `expert_ids_flat = idx.reshape(-1)` and
   `tokens_expanded = flat.repeat_interleave(top_k, dim=0)` pair each
   token's embedding with its own slot-specific expert assignment, not the
   token's first assignment only.
2. **`send_counts` counts dispatch items whose own expert falls in each
   rank's range** — not tokens with *any* matching slot. Sums to
   `N * top_k` by construction, always.
3. **Sort dispatch items by their own expert id** (not `idx[:, 0]`) for
   contiguous per-rank blocks — correct now that grouping happens per-slot.
4. **Received rows are tagged with their expert id via a second, small
   all_to_all** (`received_expert_ids`), since each sender's block is
   internally expert-sorted but the concatenation of all senders' blocks is
   in *sender rank* order, not globally expert-sorted — the tag is
   necessary information, not an optimization.
5. **Local expert selection uses boolean masking on the received expert
   ids** (`sel = received_expert_ids == eid`) instead of equal-division
   slicing — correct regardless of how tokens are actually distributed
   across local experts.
6. **Combine inverts the same permutation** (`combined_flat[sort_order] =
   combined_sorted`) and reshapes to `[N, top_k, H]` before the weighted
   sum over `k`, replacing the previous stride-`top_k` un-interleaving that
   only made sense for the old (incorrect) sort order.

### Test coverage (`tests/test_moe_layer.py`, new — no such file existed before)

- Direct per-token reference comparison (`_manual_reference`: an
  independent, deliberately naive O(N·top_k) Python loop that shares no
  code with the implementation under test) across 6 parametrized
  `(N, H, F, E, K)` shapes chosen specifically to make balanced routing
  implausible (`N` not a multiple of `E`, `top_k` up to `E` itself), plus a
  20-seed sweep at one fixed imbalance-prone shape — a single lucky seed
  landing on accidentally-balanced routing would not have caught Bug 3.
- A gradient-flow test: confirms a token's gradient only reaches the
  expert(s) it was actually routed to, catching a wrong-expert-selected
  regression even in a hypothetical case where forward-pass *values*
  happened to coincide.
- `send_counts` arithmetic checked in isolation (pure tensor logic, 5
  parametrized cases, no model needed) — a fast, direct guard on the exact
  invariant Bug 1 violated.
- A genuine 2-rank `mp.spawn` test reproducing the original
  `test_distributed_invariants.py` scenario, checked against a **correctly
  seed-matched** reference (see below) rather than token-conservation
  counts and NaN-absence alone.
- All new tests were verified against the pre-fix code before being
  trusted: reverted, confirmed 9 of 14 fail (the other 5 are the
  `send_counts` arithmetic tests, which check the *correct* formula in
  isolation and don't call into the model, so correctly pass regardless of
  which version of `moe_layer.py` is present), restored.

### A subtlety in the 2-rank test's reference construction

Building a trustworthy reference for the 2-rank test required understanding
exactly how `DistributedMoELayer.__init__` consumes its RNG:
`nn.ModuleList([_SwiGLUExpert(...) for _ in self.local_expert_ids])` runs
independently per rank, each rank re-seeding fresh. With the same seed on
every rank, this means rank 1's local experts are **bit-identical** to rank
0's local experts at the same local index — not the different weights a
naive single sequential build across all `num_experts` experts would
produce. The reference deliberately mirrors this per-rank re-seeding
pattern (`_build_seed_matched_reference`) rather than assuming a "simpler"
construction that would silently check against the wrong ground truth.
Separately, `build_topology` validates its requested mesh shape against the
*real* `dist.get_world_size()` once `dist.init_process_group` has been
called in a process — so the reference must be built **before** that call,
not inside an already-2-rank worker requesting a degenerate 1-rank
topology.

## Consequences

**Positive:**
- `DistributedMoELayer`'s forward pass is now verified correct against an
  independent per-token reference, at any routing balance, for the first
  time in this codebase's history — not just shape/NaN-checked.
- Bug 3, the most severe finding, is fixed for every configuration
  (including pure single-GPU, `ep_size=1`), not only the distributed case
  that originally surfaced this investigation.
- Both `test_distributed_invariants.py` tests pass at real 4-rank
  execution; the "2-Rank CI Blind Spot" table in `roadmap.md` now has 3 of
  5 items fixed.

**Negative / neutral:**
- Dispatch now issues two all_to_all collectives (tokens + expert-id tags)
  instead of one. Correctness required this; it was not optimized further
  (e.g. concatenating tokens and tags into a single collective) — a
  reasonable follow-up if this becomes a measured bottleneck on real
  hardware, not addressed here.
- Not run on real multi-GPU hardware or NCCL — CPU/Gloo only, same
  constraint as ADR-007 and ADR-008.
- The two remaining "2-Rank CI Blind Spot" items
  (`test_pp_multiprocess_2stage_activation_flow`,
  `test_pp_multiprocess_correct_micro_batch_count`) are untouched by this
  ADR. Their failure mode (an IPv6/IPv4 `localhost` addressing error
  observed in this session, not investigated further) looks different in
  kind from anything found here — worth treating as a separate
  investigation, not assumed to share a root cause.
