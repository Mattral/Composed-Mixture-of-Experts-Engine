# Architecture Decision Records (ADRs)

This directory contains Architecture Decision Records for moe-engine.

An ADR documents a significant architectural decision: its context, the
alternatives considered, the decision made, and the consequences.
ADRs are immutable once accepted — subsequent decisions that supersede an
ADR create a new ADR referencing the old one.

## Index

| ID | Title | Status |
|----|-------|--------|
| [ADR-001](ADR-001-triton-router-kernel.md) | Fused Triton Router Kernel | Accepted |
| [ADR-002](ADR-002-async-two-tier-checkpointing.md) | Async Two-Tier Checkpointing | Accepted |
| [ADR-003](ADR-003-pydantic-config.md) | Pydantic v2 Configuration System | Accepted |
| [ADR-004](ADR-004-4d-parallelism-composition.md) | 4D Parallelism Composition | Accepted |
| [ADR-005](ADR-005-gpu-architecture-portability.md) | GPU Architecture Portability (Turing vs. Ampere) and Honest Fallback Telemetry | Accepted |
| [ADR-006](ADR-006-device-coercion-correction.md) | Correction — Device Coercion Belongs in `_reference_route_fp64`, Not Only the Telemetry Block | Accepted |
| [ADR-007](ADR-007-sequence-parallel-correctness-and-nondivisible-length.md) | Sequence Parallelism Correctness Fixes, Non-Divisible Sequence Length Support, and the 2-Rank CI Blind Spot | Accepted |
| [ADR-008](ADR-008-column-parallel-gather-fix.md) | `ColumnParallelLinear` Gather-Dimension Bug and the Missing `gather_output` Mode | Accepted |
| [ADR-009](ADR-009-moe-dispatch-correctness.md) | `DistributedMoELayer` Dispatch Correctness — Three Compounding Bugs | Accepted |
| [ADR-010](ADR-010-backward-correctness-and-replicated-gradients.md) | Backward Correctness Under EP/TP/SP, Replicated-Gradient Sync, and the Differential Test Harness | Accepted |

## Template

```markdown
# ADR-NNN: Title

**Status:** Proposed | Accepted | Deprecated | Superseded  
**Date:** YYYY-MM  
**Deciders:** Name(s)  
**Supersedes:** ADR-NNN (if applicable)

## Context
[Why does this decision need to be made? What is the problem?]

## Decision
[What was decided? Be specific.]

## Consequences
[What are the positive and negative outcomes?]

## Alternatives rejected
[What else was considered and why was it not chosen?]
```
