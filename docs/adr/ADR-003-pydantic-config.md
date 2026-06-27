# ADR-003: Pydantic v2 Configuration System

**Status:** Accepted  
**Date:** June 2026 (v0.3.2)  
**Deciders:** Min Htet Myet  
**Supersedes:** Flat dict config system (pre-v0.3.2)

---

## Context

The original configuration system loaded YAML into a flat Python dict
(`cfg["model"]["hidden_dim"]`) with no validation, no type safety, and no
constraint checking. Errors such as `top_k=99` with `num_experts=4`, a typo in
`dtype`, or a negative `learning_rate` would surface as mysterious shape
mismatches or runtime crashes 20+ minutes into training — long after the config
was loaded.

This violates the **Fail-fast on invariants** principle (see ARCHITECTURE.md)
and was identified as one of the top maintainability blockers in the MOE
instructions v2.1 assessment.

Alternatives considered:
1. **Flat dict + manual validation**: ad hoc `assert` checks at config load time. Low leverage; easy to miss cases.
2. **`dataclasses` + custom validation**: verbose, no YAML schema generation, no nested defaults.
3. **`attrs`**: powerful but less widely adopted in the PyTorch ecosystem than Pydantic.
4. **Pydantic v2**: industry standard, excellent error messages with field paths, automatic JSON schema, environment variable override support, and round-trip YAML serialisation.

## Decision

Replace the flat dict with a **Pydantic v2 `BaseModel` hierarchy** (`MoEConfig`):

```
MoEConfig
├── ModelConfig       (H, E, K, F, dtype, vocab_size, seq_len)
├── TrainingConfig    (lr, warmup, grad_clip, max_steps, ...)
├── ParallelismConfig (dp, ep, tp, pp → world_size property)
├── CheckpointConfig  (local_dir, remote_uri, retention, ...)
├── ElasticConfig     (min/max nodes, rdzv_backend, ...)
└── TelemetryConfig   (log_dir, mfu_target, peak_tflops, ...)
```

Cross-field validators (`@model_validator`) enforce:
- `top_k <= num_experts`
- `warmup_steps < max_steps`
- `min_nodes <= max_nodes`
- `hidden_dim % 8 == 0`
- `dtype ∈ {float32, bfloat16, float16}`
- `remote_uri` starts with `s3://` or `file://`

Environment variable overrides via `MOE_<SECTION>__<KEY>=value`.

Backward-compatible `load_config()` shim preserves all existing call sites.
34 dedicated tests in `tests/test_config.py`.

## Consequences

**Positive:**
- Config errors caught at load time with field-level messages (`[model] top_k (99) must be <= num_experts (4)`).
- Round-trip `to_yaml() / from_yaml()` idempotent.
- `parallelism.world_size` property computed automatically.
- `MOE_TRAINING__LEARNING_RATE=1e-4` overrides work via env vars.
- All 34 config tests pass on CPU in < 0.5s.

**Negative / trade-offs:**
- Pydantic v2 is a required dependency (added to `requirements.txt`).
- Minor overhead at startup (~5ms for validation) — irrelevant at training scale.
- The Pydantic shim fallback (for environments without Pydantic) skips runtime validation but preserves import compatibility.

## Migration

Old code:
```python
cfg = load_config("configs/smoke.yaml")
h = cfg.raw["model"]["hidden_dim"]
```

New code (preferred):
```python
cfg = MoEConfig.from_yaml("configs/smoke.yaml")
h = cfg.model.hidden_dim
```

Both patterns work simultaneously during the transition period.
