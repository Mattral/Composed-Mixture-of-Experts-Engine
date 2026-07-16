# ADR-005: GPU Architecture Portability (Turing vs. Ampere) and Honest Fallback Telemetry

**Status:** Accepted
**Date:** July 2026
**Deciders:** Min Htet Myet
**Supersedes:** —
**Related:** ADR-001 (Fused Triton Router Kernel)

---

## Context

The router kernel was validated on a single NVIDIA T4 (Turing, `sm_75`) in
June 2026 (see `RESULTS.md`, `docs/adr/ADR-001-triton-router-kernel.md`).
When the same notebook workflow was later run against an NVIDIA
A100-SXM4-80GB (Ampere, `sm_80`) on Lightning AI Studio, it failed on
configurations that had worked, or had simply never been exercised, on T4
— specifically, small-expert-count configurations such as
`configs/smoke.yaml` (`num_experts=4`).

Two distinct problems were found and fixed together, and it is worth
separating them precisely because they have different root causes and
different classes of fix.

### Problem 1: `tl.dot` tensor-core minimum tile size

Triton's `tl.dot` dispatches to the GPU's tensor-core `mma` (matrix-multiply-
accumulate) instructions when operand shapes are large enough, and falls
back to a slower path otherwise. The minimum tile dimension for reliable
`mma` dispatch differs across GPU generations and Triton versions. A
configuration with `num_experts=4` (used deliberately in
`configs/smoke.yaml` to keep the CPU smoke test fast) is well under any
reasonable tensor-core tile minimum. On T4, in the environment and Triton
version used for the June 2026 validation, this either did not trigger a
hard failure or was simply never exercised directly through the Triton
path (the smoke test's default execution goes through the CPU reference
path in CI regardless). On A100, in a materially newer software stack
(`torch==2.8.0+cu128` at time of testing, versus an older stack used for
T4 validation), the same undersized configuration caused Triton kernel
compilation to fail outright the first time it was attempted on real GPU
hardware.

This is architecturally the same *class* of issue as the `K: tl.constexpr`
defect in ADR-001: a real-hardware-only failure mode that a CPU-only
reference path cannot surface, compounded here by a *second* variable —
GPU architecture generation — that a single-GPU validation pass cannot
surface either. One real GPU is necessary but not sufficient; the
generation of that GPU also matters.

### Problem 2: Device-mismatch in the fp64 reference path

Independently of the above, a device-mismatch defect existed in the fp64
reference computation used when the Triton path is not taken (either by
explicit `force_reference=True`, by the presence of a router bias forcing
reference mode, or — after the fix below — by the new dimension guard).
On CPU-only runs this defect is unreachable, because every tensor involved
is already on the same device by construction. It only manifests when the
reference path executes with CUDA tensors, which is precisely the
situation the dimension guard in Problem 1 newly creates on every GPU
generation once undersized configurations start falling back to the
reference path instead of crashing.

### Problem 3 (discovered during the fix, not part of the original report):
`used_triton` telemetry could silently misreport ground truth

While fixing Problems 1 and 2, a third, pre-existing issue was found: the
`RouterProfile.used_triton` field — surfaced in every `StepRecord` as
`kernel.used_triton` — was computed independently in two places:

1. Correctly, inside `MoERouterFunction.forward`, which has full visibility
   into `force_reference`, the dimension guard, and any runtime exception
   during Triton compilation.
2. Incompletely, inside `MoERouter.forward`'s separate profiling block,
   which only checked `TRITON_AVAILABLE and flat.is_cuda` — ignoring
   `force_reference`, the dimension guard, and any exception fallback
   entirely.

This meant `used_triton=True` could be reported in telemetry even when the
fp64 reference path had actually executed the forward pass — for any
`force_reference=True` call, any bias-enabled router, and, after the
dimension-guard fix below, any configuration with `N`, `H`, or `E` under
the tensor-core minimum. This is the same failure archetype documented in
`docs/adr/ADR-003-pydantic-config.md` (a status signal that can silently
stop reflecting reality) applied to a different subsystem. We treat this
as a correctness bug, not a cosmetic one, for the same reason given there:
a telemetry field whose entire purpose is reporting which code path ran
must not be able to lie about it.

A closely related inefficiency was found in the same code: the profiling
block in `MoERouter.forward` re-invoked `_triton_forward(...)` a *second
time* on every forward pass purely to re-derive `kernel_ms`,
`sram_bytes_per_block`, and `achieved_bandwidth_gbps` for telemetry —
doubling the number of Triton kernel launches on the hot path — and wrapped
that second launch in its own `except Exception: pass`, which meant a
failure of the *profiling-only* relaunch would also pass silently.

## Decision

### Fix 1 — `MIN_TRITON_DIM` as a documented, permanent kernel contract

`pkg/kernels/moe_router.py` now declares a module-level constant,
`MIN_TRITON_DIM = 16`. Below this threshold on any of `N`, `H`, or `E`,
the Triton path is not attempted at all; the fp64 reference path is used
instead, deterministically, on every GPU generation. This is documented as
an intentional part of the kernel's public contract, not a one-off
crash-avoidance patch: any caller with dimensions below this threshold
should expect the reference path, and should expect it to be reported
truthfully (see Fix 3).

### Fix 2 — Explicit device placement in the reference path

The fp64 reference computation now explicitly moves the gate weight and
token tensors to a common device (`gate_w.to(device=flat.device, ...)`)
before computing logits, rather than assuming device consistency that the
dimension-guard change in Fix 1 could otherwise violate.

### Fix 3 — Single source of truth for the Triton-eligibility decision

A single function, `_should_use_triton(tokens, gate_w, force_reference)`,
is now the only place this decision is computed. Both
`MoERouterFunction.forward` (which decides which path to execute) and
`MoERouter.forward` (which reports `used_triton` in telemetry) call it, or
— for the exception-fallback case, which `_should_use_triton` cannot see
in advance — read the *actual* outcome back through a small non-tensor
side channel (`report: Optional[dict]`, passed as the 5th argument to the
custom `autograd.Function`, mutated in place, and never touched by
autograd since it is not a tensor). This closes the class of bug entirely:
it is no longer possible for the eligibility check and the telemetry
report to independently drift out of sync, because there is only one
computation of "did Triton actually run," not two.

The same side channel now also carries the real `kernel_ms`,
`sram_bytes`, and `achieved_bw` from the one Triton launch that already
happens, eliminating the redundant second kernel launch and its own
silent `except: pass` fallback described above.

### Fix 4 — Commit real per-architecture benchmark data as regression baselines

`benchmarks/gpu_results_t4.json` (from the original June 2026 T4
validation) and `benchmarks/gpu_results_a100.json` (from the July 2026
A100 validation on Lightning AI Studio) are both committed to the
repository as permanent, dated reference artifacts. Any future
contributor validating on either architecture — or a new one — has a
concrete "known good" baseline to diff against, rather than a single
architecture's numbers standing in implicitly for "GPU performance" in
general.

## GPU Architecture Support Matrix

| Architecture | Compute Capability | Status | Validated | Notes |
|---|---|---|---|---|
| NVIDIA T4 (Turing) | `sm_75` | ✅ Validated | June 2026 | `benchmarks/gpu_results_t4.json`; Colab |
| NVIDIA A100-SXM4-80GB (Ampere) | `sm_80` | ✅ Validated | July 2026 | `benchmarks/gpu_results_a100.json`; Lightning AI Studio; required Fixes 1–3 above |
| NVIDIA H100 (Hopper) | `sm_90` | ❌ Not yet tested | — | No access at time of writing |
| CPU (fp64 reference) | — | ✅ Validated | Ongoing | Every commit, via Tier-0 CI |

## Answering "how do we lock our code on?"

We do not lock the code to one GPU generation. We do the opposite:
we make the same code path behave correctly, and *report honestly what it
did*, on any generation, via three layers:

1. **Defensive dimension guards** (`MIN_TRITON_DIM`) mean an undersized
   config degrades to a slower-but-correct path instead of crashing,
   regardless of how strict a given architecture's tensor-core minimum
   turns out to be.
2. **Telemetry that cannot lie** (`_should_use_triton` as single source of
   truth) means that when a fallback does happen — for whatever reason,
   on whatever hardware — the system says so, rather than silently
   reporting the fast path was taken.
3. **A committed, per-architecture benchmark archive** means "this works on
   GPUs" is never asserted in the abstract. Each architecture actually
   validated gets its own dated, diffable JSON file in `benchmarks/`, and
   the support matrix above states plainly which architectures have and
   have not been checked.

A fourth practice, not yet formalized as of this ADR, is recommended for
future work: pin `torch`/`triton` to an exact tested version pair per
validated architecture (rather than the current `>=` ranges in
`pyproject.toml`), and have the validation notebook for each architecture
print and soft-warn if the installed versions fall outside the range that
produced the committed baseline for that architecture. This was not
implemented in this pass because only one new architecture's software
stack (A100 / Lightning AI Studio, `torch==2.8.0+cu128`) has been observed
so far — a version pin derived from a single data point is not yet
well-founded, and asserting a hard range risks blocking a future
architecture's validation for the wrong reason. The A100 validation
notebook does print the installed `torch`/`triton`/CUDA versions on every
run specifically so this data accumulates for a future, better-founded
version-pinning decision.

## Consequences

**Positive:**
- The router kernel now runs correctly on both validated architectures
  using the same unmodified source, with no architecture-specific branches
  in the kernel code itself.
- `used_triton` and the four `kernel.*` telemetry fields derived from it
  are now provably consistent with what actually executed, closing a
  silent-misreporting gap that existed on every `force_reference=True` or
  bias-enabled call, on every architecture, since the kernel was written.
- One fewer Triton kernel launch per forward pass on the hot path (the
  redundant profiling relaunch is gone).
- Two dated, real, committed benchmark archives now exist instead of one,
  giving future contributors an actual cross-architecture comparison
  rather than a single anecdotal data point.

**Negative / trade-offs:**
- `MIN_TRITON_DIM = 16` means very small toy configurations (below this
  threshold on any dimension) never exercise the Triton path at all, even
  on hardware where they might have happened to work. We consider this
  the correct trade-off: a deterministic, documented, honestly-reported
  fallback is preferable to an undefined "might work depending on
  hardware and Triton version" boundary.
- The exact tensor-core minimum for future architectures (H100 and beyond)
  is not yet known; `MIN_TRITON_DIM = 16` is derived from the Ampere
  failure observed here, not from a specification. It should be revisited
  if H100 validation surfaces a different effective minimum.

## Alternatives rejected

**Architecture-specific code paths** (e.g., `if capability == (8, 0): ...`):
rejected because it does not scale — every new GPU generation would need
its own branch, discovered reactively by another crash, rather than a
single dimension guard that is conservative across generations by
construction.

**Silently catching and ignoring the Ampere compilation failure** (i.e.,
widening the existing `except Exception` fallback in
`MoERouterFunction.forward` to cover this case without the dimension
guard): rejected because it would have masked the underlying tensor-core
constraint indefinitely rather than making it an explicit, documented part
of the kernel's contract, and would have done nothing about the separate
`used_triton` telemetry bug — a silent fallback with honest telemetry is
categorically different from a silent fallback with dishonest telemetry,
and we consider the second one unacceptable regardless of how the fallback
is triggered.
