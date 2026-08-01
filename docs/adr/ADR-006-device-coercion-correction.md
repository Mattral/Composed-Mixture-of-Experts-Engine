# ADR-006: Correction — Device Coercion Belongs in `_reference_route_fp64`, Not Only the Telemetry Block

**Status:** Accepted
**Date:** July 2026
**Deciders:** Min Htet Myet
**Supersedes:** —
**Related:** ADR-005 (GPU Architecture Portability), which this ADR corrects an implementation gap in without altering ADR-005's substance, per this directory's immutability rule.

## Context

ADR-005's "Fix 2" describes a device-mismatch fix as follows:

> The fp64 reference computation now explicitly moves the gate weight and
> token tensors to a common device (`gate_w.to(device=flat.device, ...)`)
> before computing logits, rather than assuming device consistency that
> the dimension-guard change in Fix 1 could otherwise violate.

This reads as, and was intended to be, a description of
`_reference_route_fp64()` in `pkg/kernels/moe_router.py` — the function
`MoERouterFunction.forward()` calls whenever the reference (non-Triton)
path is taken, which after ADR-005 Fix 1's `MIN_TRITON_DIM` guard now
includes every undersized-config forward pass, not only explicit
`force_reference=True` calls.

A permanent-test-coverage pass (Handoff Section 7.1, Task 1 —
`tests/test_gpu_portability.py`) found that this was not actually true.
The only place in `moe_router.py` performing `gate_w.to(device=..., ...)`
was a **different, unrelated** block: the z-loss / load-imbalance
telemetry computation inside `MoERouter.forward()`, which runs
unconditionally after routing (regardless of which path was used) purely
to compute quality metrics for logging. That block happens to use a local
variable also named similarly in prose (`flat`) and happens to compute
logits via a similar-looking matmul, which is almost certainly how the
two got conflated when ADR-005 was originally written up.

`_reference_route_fp64()` itself had **no device coercion at all** — it
called `.to(torch.float64)` independently on `tokens` and `gate_w`, which
changes dtype but not device. A caller whose `tokens` and `gate_w`
happened to live on different devices would hit a device-mismatch
`RuntimeError` inside the `t64 @ g64` matmul, on exactly the code path
ADR-005 documents as fixed.

This is worth recording as its own ADR rather than a silent code fix
because it is precisely the failure mode this project's own design
philosophy exists to catch: a fix that *looks* like it's checking (a
correctly-worded ADR, a plausible-looking `.to(device=...)` call
somewhere in the file) without actually *being* the fix in the place that
matters. Section 3.1 of the project handoff — the `K: tl.constexpr`
bug (ADR-001) and the Pydantic silent-validation-fallback bug (ADR-003)
— are both the same shape of problem. This one was self-inflicted by the
project's own documentation, not an external dependency, which makes it
worth being equally honest about.

## Decision

1. `_reference_route_fp64()` now explicitly coerces `gate_w` onto
   `tokens.device` before the matmul:

   ```python
   g64 = gate_w.to(device=tokens.device, dtype=torch.float64)
   ```

   This is the actual fix ADR-005 Fix 2 describes, now living in the
   function it describes.

2. The original fix in `MoERouter.forward()`'s telemetry block is **left
   in place**, not removed. It is redundant now that the reference path
   itself is safe, but it is harmless, and removing working defensive
   code purely for tidiness is not worth the risk for a one-line no-op in
   the common case where devices already match.

3. `docs/adr/ADR-005-gpu-architecture-portability.md` itself is **not
   edited**. Its Decision section's wording is subtly inaccurate about
   *where* Fix 2 lives, but the ADR-immutability rule in this directory's
   `README.md` exists precisely so a reader can trust that an "Accepted"
   ADR reflects what was decided *at the time*, without needing to
   suspect later silent edits. This ADR is the correction of record; a
   reader who wants the accurate current location should follow ADR-005 →
   ADR-006 the same way they would follow any superseding decision.

4. Permanent regression coverage was added in
   `tests/test_gpu_portability.py::TestReferencePathDevicePlacement`:
   - A CPU-runnable source-inspection test
     (`test_reference_route_fp64_source_coerces_gate_w_device`) that fails
     if the `.to(device=...)` call is ever removed from
     `_reference_route_fp64` again — this is the test that would have
     caught this gap on day one, had it existed before ADR-005 was
     written.
   - A `@pytest.mark.gpu` test
     (`test_reference_route_fp64_cross_device_real_hardware`), honestly
     skipped without 2+ CUDA devices, that constructs a genuine
     cross-device call and confirms it no longer raises. This has not yet
     been run on real hardware — 2+ GPU access is Handoff Section 7.3's
     open blocker, not something this ADR resolves on its own.

## Consequences

**Positive:**
- The actual routing hot path is now defended against a device-mismatch
  crash, not just a telemetry side-computation that happens to look
  similar.
- A source-inspection regression test exists so this specific gap cannot
  reappear silently; a future refactor that deletes the coercion will
  fail CI immediately with a message pointing at this ADR.
- The gap and its correction are both on the record, rather than the
  fix being applied quietly with no trace that ADR-005 was ever
  inaccurate. Anyone auditing this project later (human or another Claude
  instance) can reconstruct exactly what happened and why, which is the
  whole point of keeping ADRs in the first place.

**Negative / neutral:**
- ADR-005 now contains one sentence that is subtly wrong about
  implementation location, permanently, by design (per the immutability
  rule). This is a real cost of the immutability rule in exchange for a
  larger benefit (never having to wonder whether an "Accepted" ADR was
  quietly rewritten after the fact); this ADR is the mitigation.
- The device-mismatch scenario this fix protects against has still never
  been exercised on real hardware with two distinct CUDA devices — this
  ADR restores the *documented* fix to its *documented* location, but
  "restores documented behavior" and "empirically validated on the
  failure mode it targets" are not the same claim, and this ADR does not
  conflate them.

## Alternatives rejected

- **Editing ADR-005's Decision section directly.** Rejected: violates
  this directory's own immutability rule, and would remove the historical
  record of what was actually claimed at the time — which is the more
  useful thing for future debugging of "why did we think this was
  fixed?" questions.
- **Removing the now-redundant coercion in the telemetry block.**
  Rejected: no benefit beyond tidiness, non-zero (if small) risk of
  reintroducing a crash if some future code path calls that block with
  genuinely mismatched devices before reaching the routing computation.
- **Silently fixing the code with no ADR, since it's "just" restoring
  documented behavior.** Rejected: this is exactly the kind of gap this
  project has repeatedly found value in writing down explicitly (see
  Context above) — a silent fix here would be inconsistent with the
  standard the rest of the project holds itself to.
