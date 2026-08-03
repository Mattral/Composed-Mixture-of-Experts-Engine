# Tensor Parallel Correctness Audit — August 2026

**Scope:** Direct follow-up to `docs/audits/2026-08-sp-correctness-audit.md`'s
recommendation — the first item in `roadmap.md`'s "2-Rank CI Blind Spot"
table. Full technical writeup:
`docs/adr/ADR-008-column-parallel-gather-fix.md`. This doc is the shorter
process/verification companion.

## What was found, in the order it was found

1. `ColumnParallelLinear.forward()`'s `all_gather_into_tensor` call
   concatenated along the wrong dimension (same root cause ADR-007 fixed in
   `gather_from_sequence_parallel`, different dimension: `out_features`,
   the last dim, not a middle one).
2. Fixing #1 alone did not make the 2-rank test pass — it progressed to a
   *different* failure: `RowParallelLinear` received a full-width tensor
   from `ColumnParallelLinear` instead of the sharded input its weight
   shape expects.
3. Root-causing #2 revealed `ColumnParallelLinear` had no way to skip its
   gather — and that `_SwiGLUExpert` (the actual expert FFN used throughout
   `DistributedMoELayer`) composes Column→Row exactly this way, meaning the
   real model's expert compute has never worked at `tp_size > 1`. Only ever
   tested at `tp_size=1`, where the gather is a no-op and the bug is
   invisible.

## Changes made

1. `pkg/distributed/tensor_parallel.py` — fixed the gather dimension
   (movedim technique, same as ADR-007); added `gather_output: bool = True`
   parameter to `ColumnParallelLinear`; `bias=True` + `gather_output=False`
   raises `NotImplementedError` rather than guessing at unverified
   sharded-bias semantics.
2. `pkg/distributed/moe_layer.py` — `_SwiGLUExpert.w_gate`/`w_up` now pass
   `gather_output=False`, making the class's own docstring claim actually
   true.
3. `tests/test_tensor_parallel.py` — fixed the existing 2-rank composition
   test (needed `gather_output=False` on `col`, which didn't exist before
   this pass); added 4 new tests: `gather_output=True` standalone at
   2-rank, `gather_output=False` standalone at 2-rank, the bias-raises
   guard at real `tp_size>1`, and `_SwiGLUExpert` itself at genuine 2-rank
   TP (25 tests total, up from 21).
4. `docs/adr/ADR-008-...md` — new ADR.
5. `docs/adr/README.md`, `roadmap.md`, `docs/testing.md`, `RESULTS.md`,
   `docs/SYSTEM_DESIGN.md` — index/status updates. Tier-0 count unchanged
   at 380 (all 4 new tests are 2-rank, correctly excluded from Tier-0).

## Verification discipline

Every fix in this pass was verified against the pre-fix code before being
trusted — not just verified to pass afterward:

```
$ # gather-dimension fix reverted
$ pytest tests/test_tensor_parallel.py -k gather_output_true
1 failed  # RuntimeError: invalid tensor size — confirmed genuine
$ # fix restored

$ # _SwiGLUExpert's gather_output=False reverted (ColumnParallelLinear's
$ # fix left in place, to isolate exactly which change this test guards)
$ pytest tests/test_tensor_parallel.py -k swiglu_expert_2rank
1 failed  # RuntimeError: mat1 and mat2 shapes cannot be multiplied (4x64 and 32x32)
$ # the exact original crash, reproduced on demand — fix restored

$ pytest tests/test_tensor_parallel.py -v
25 passed

$ pytest tests/ --ignore=tests/test_chaos.py --ignore=tests/test_smoke_e2e.py \
    -m cpu -k "not (2rank or multiprocess or distributed_invariants)"
380 passed, 3 skipped, 15 deselected, 1 xfailed

$ ruff check / ruff format --check
All checks passed! / 3 files already formatted
```

## Explicitly deferred

Three of the four "2-Rank CI Blind Spot" items remain open: both
`test_pipeline_parallel.py` 2-rank tests, and both
`test_distributed_invariants.py` tests. Not investigated in this pass —
see `roadmap.md`'s "2-Rank CI Blind Spot" table for their captured error
signatures.

## Limitations

Same as every prior audit in this series: CPU/Gloo `mp.spawn` only, no real
GPU or NCCL, no claim made beyond what was actually verified here.
