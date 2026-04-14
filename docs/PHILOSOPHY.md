# Philosophy

**Version:** v0.2  
**Last updated:** June 2026

moe-engine is built around a single constraint: **at 10K+ GPUs, nodes die
continuously**. Every design decision is derived from that premise.

---

## Core Principles

### 1. Measure, don't estimate

Every number in the runtime is a real measurement, not an approximation or
placeholder. Specifically:

- **Collective latency** (`all_to_all_dispatch_ms`, `all_to_all_combine_ms`)
  is measured with `torch.cuda.Event` timing on the dedicated EP CUDA stream.
- **Peak memory** (`peak_allocated_gb`, `reserved_gb`) comes from
  `torch.cuda.memory_stats()`, not from model parameter counts.
- **MFU** uses the MoE-aware sparse formula `(K/E) × P_expert` — not the
  dense approximation that overestimates FLOPs by a factor of `E/K`.
- **Routing quality** (`expert_load_imbalance`, `router_z_loss`) is computed
  from actual `dispatch_cnt` per step, not sampled or estimated.

The test suite enforces this: `test_telemetry.py` asserts that all REQUIRED_KEYS
are present and `test_mfu_v02.py` verifies that the sparse fraction scales
correctly with K.

### 2. Fail fast on invariants

Distributed bugs propagate silently. A token routing error does not manifest
as a crash — it manifests as slightly wrong gradients, six hours later, after
the checkpoint is overwritten. We prevent this by asserting invariants in the
forward pass:

- **Token conservation:** `sum(dispatch_cnt) == N × K` after every router call.
- **Index validity:** `idx ∈ [0, E)`, no NaN, no -1.
- **Combine NaN guard:** post-combine output checked for NaN before returning.
- **Topology product:** `dp × tp × pp × ep == world_size` at launch.

Each invariant fires immediately at the layer that broke it, with a message
that includes all the relevant values. This is non-negotiable: no invariant
check is removed for performance.

### 3. Every collective must be the correct collective

This is the most commonly violated principle in distributed systems code.
We have a strict rule: each communication pattern has exactly one correct
collective, and using a different collective (even one with similar semantics)
is a correctness bug.

Examples from the codebase:

| Pattern | Correct | Wrong (and why) |
|---|---|---|
| Sum partial matmul outputs (RowParallel) | `all_reduce(SUM)` | `reduce_scatter + all_gather` — two collectives, semantically wrong: reduce_scatter sends chunks to different ranks, not sums them |
| Replicate sharded column output (ColumnParallel) | `all_gather_into_tensor` | `broadcast` — broadcast assumes all ranks start with same data |
| EP token dispatch | `all_to_all_single` | `all_gather` — all_gather replicates, not redistributes |

The `test_row_parallel_uses_all_reduce_not_reduce_scatter` test in
`test_tensor_parallel.py` enforces this structurally via source inspection.
New collective patterns must include similar guards.

### 4. TP sharding must be consistent through every operation

A mixed TP implementation — where some layers are sharded and others are not —
produces incorrect results silently. The SwiGLU expert demonstrates this: both
`w_gate` and `w_up` must be `ColumnParallelLinear` because their element-wise
product must occur in the same shard space. Making one `nn.Linear` and the other
`ColumnParallel` is a bug that only manifests at `tp_size > 1`.

The rule: **either a tensor is fully sharded or fully replicated at every point
in the compute graph**. Mixed-sharding intermediate states are a correctness hazard.

### 5. Tests must exercise real collectives

Single-process tests at `tp_size=1` do not validate distributed code. At
`tp_size=1`, `ColumnParallelLinear` and `RowParallelLinear` reduce to plain
`nn.Linear` — no collectives fire. A test that only covers this path gives
false confidence.

The rule: every distributed primitive must have a multi-process test that
fires real collectives and validates numerical equivalence to a reference.
`test_column_row_parallel_2rank_numerically_correct` is the template.

### 6. Honest documentation

We document what is actually built, not what we plan to build. The roadmap
tracks planned features separately from completed ones. The "Known Deficiencies"
section in `roadmap.md` is not a weakness — it is evidence that the system
is understood and its boundaries are known.

Specifically:
- Chaos Scenario A passes at ~85%, not 100%. This is documented with its root
  cause, current mitigation, and the correct fix.
- Pipeline parallelism is single-process only in v0.2. The multi-process
  activation-passing layer is explicitly deferred to v0.3, not silently absent.
- GPU benchmark numbers are illustrative until we have sustained cluster access.
  The CPU numbers are real and reproducible.

### 7. Observability is a first-class feature

Every new runtime path must expose telemetry. The v0.2 routing quality metrics
(`expert_load_imbalance`, `router_z_loss`) were added alongside the code that
computes them, not as an afterthought. The Prometheus endpoint was added in the
same release as the metrics it exposes.

The principle: if behaviour cannot be observed, it cannot be debugged, and it
cannot be trusted at scale.

### 8. Link, don't duplicate

Documentation points to source files and test functions rather than
re-describing behaviour inline. When code changes, there is one place to update
(the code), not two (code + documentation). This principle is enforced by
writing documentation in terms of function names and test names that will break
if the code changes in ways that make the docs wrong.

---

## What This Means in Practice

**For new features:** add telemetry before the feature ships. Add a numerics
test before optimising the kernel. Add the invariant assertion before adding
the optimised path that could violate it.

**For bug fixes:** write the test that reproduces the bug, then write the fix,
then verify the test passes. Update the doc if the fix changes documented
behaviour.

**For documentation:** every claim should be verifiable by running a command.
"Token conservation holds" is verified by `python tests/run_numerics_tests.py`.
"RowParallel uses all_reduce" is verified by
`pytest tests/test_tensor_parallel.py::test_row_parallel_uses_all_reduce_not_reduce_scatter`.

---

## Evidence

These principles are instantiated in the codebase today:

| Principle | Evidence |
|---|---|
| Measure, don't estimate | CUDA events in `_CommStream`; `memory_stats()` in `StructuredLogger` |
| Fail fast on invariants | `assert total_dispatched == expected_total` in `MoERouter.forward` |
| Correct collective per pattern | `all_reduce` in `RowParallelLinear`; structural test enforces it |
| Consistent TP sharding | Both `w_gate` and `w_up` are `ColumnParallelLinear` in `_SwiGLUExpert` |
| Real collective tests | `test_column_row_parallel_2rank_numerically_correct` (mp.spawn, Gloo) |
| Honest documentation | `roadmap.md §Known Deficiencies`; Scenario A documented as ~85% |
| Observability first | `routing.expert_load_imbalance` and `router_z_loss` in every StepRecord |
| Link, don't duplicate | Test names cited in `docs/testing.md`; function names in `SYSTEM_DESIGN.md` |
