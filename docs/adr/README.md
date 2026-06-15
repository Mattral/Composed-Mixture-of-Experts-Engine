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
