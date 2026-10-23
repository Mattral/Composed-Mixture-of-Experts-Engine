# Changelog

All notable changes to this project are recorded here. Verification claims
state the environment they were run in; nothing below was run on a GPU, with
NCCL, or across multiple hosts unless it says so.

## Unreleased — M2: MoE semantics (October 2026, in progress)

### Changed
- **F22 — training objective.** The objective is now the mean over the
  global batch's loss-bearing tokens, independent of the (dp, ep, tp/sp)
  layout, of per-rank token counts and of the gradient-accumulation count.
  Before, each rank's own mean was summed over EP/SP and averaged over DP and
  micro-batches. Micro-batches backpropagate local loss sums;
  `grad_sync.finalize_grads` divides by `grad_sync.global_token_count` once
  per step, and FSDP2 now sums over DP. Supersedes ADR-010's loss convention:
  see [ADR-011](docs/adr/ADR-011-layout-independent-global-mean-objective.md).
  The CI differential job pins `torch==2.14.1`, because the FSDP2 methods
  this relies on depend on the torch version.

- **F5 — router auxiliary losses.** The router z-loss was a detached float
  read from block 0 only, and there was no load-balancing loss, so neither
  trained the router. Every MoE layer now produces tensor-valued z-loss and
  Switch load-balancing sums. The expert fractions are global over the
  micro-batch (DP × EP). `collect_aux_losses` gathers them from every layer,
  and `train.py` adds them with `z_loss_weight` (default 0.0) and the new
  `load_balance_weight` (default 1e-2). See ADR-011, "Router auxiliary
  losses".

### Removed
- **F27 — `.github/workflows/signal-correctness.yml`.** It was invalid YAML
  from the day it was added (`d9cf60c`, 2026-02-24): an unquoted
  `name: Alternative: ...` on line 25. GitHub could not parse it, so it never
  ran a job and reported a failed run on every push to any branch. Its checks
  either already run in `ci.yml` or were obsolete:
  - numerics tests: `test_kernels_numerics.py` runs in the Tier-0 CPU job;
  - token conservation: the `benchmark` job's conservation assertion, plus
    `test_router.py` and `test_kernels.py` in Tier-0;
  - Python syntax: `ruff check` in the lint job;
  - "assertion guards" (greps for three source lines): obsolete, since none
    of the three lines exists any more and grepping source text is not a
    behavioural check.

### Fixed
- **F23 — biased router.** `MoERouter(bias=True)` recomputed its routing
  under `no_grad`, so its combine weights carried no gradient.
- **F24 — FSDP2 without mixed precision.** `apply_fsdp2` passed
  `mp_policy=None`, so FSDP2 raised at the first forward of every fp32 run at
  dp > 1.
- **F25 — FSDP2 with sparse expert use.** DP ranks whose tokens skipped
  different experts reduce-scattered different parameter sets: a hang, or one
  expert's gradient silently summed into another's. Unused parameters now
  reduce zeros.
- **F26 — parameter marks under FSDP2.** FSDP2 replaces parameters with new
  DTensor parameters, which dropped `mark_sequence_parallel`, so at dp > 1
  with tp > 1 SP-region weights were never summed over TP. `apply_fsdp2` now
  restores the marks.

### Added
- `tests/test_differential.py::test_layout_*`: one global batch with unequal
  loss masks and 2 micro-batches, run under single, dp2, ep2, sp2, dp2×ep2,
  dp2×sp2 and ep2×sp2 and compared against a dense global-mean reference
  (CPU/Gloo; FSDP2 at dp = 2 on CPU/Gloo only). The `aux-*` variants add the
  router auxiliary losses.
- `tests/test_router_losses.py` (Tier-0): biased-router gradients (F23);
  auxiliary-loss values and gradients against a dense reference,
  aggregation over every layer, and config defaults (F5).

## Unreleased — M1: correct gradients under EP/TP/SP (October 2026)

Fixes from the October 2026 correctness audit. Before this, **training was
wrong whenever any parallelism dimension was > 1**: see
[ADR-010](docs/adr/ADR-010-backward-correctness-and-replicated-gradients.md).

### Fixed
- **F1 — EP backward.** Expert parameters received no gradient and the MoE
  layer's input gradient was 84–100% wrong under expert parallelism. EP
  dispatch/combine now use an autograd-aware all-to-all
  (`pkg/distributed/comm_ops.py`).
- **F2 — TP backward.** `ColumnParallelLinear(gather_output=True)` raised in
  grad mode; the Column→Row MLP's input gradient was wrong. Column/Row now use
  Megatron's conjugate collectives (`copy_to_group`, `reduce_from_group`,
  `gather_along_dim`).
- **F3 — SP.** `gather_from_sequence_parallel` raised during the forward pass
  whenever its input required grad; scatter returned only each rank's chunk
  of the gradient. Both now use `gather_along_dim` / `split_along_dim`.
- **F4 — non-divisible expert counts.** `E % ep_size != 0` crashed or
  deadlocked. `send_counts` now derives from the single expert-ownership rule
  (`ParallelTopology.experts_on_ep_rank`), and invalid counts on any rank make
  every rank abort together instead of hanging.
- **F21 — replicated-parameter gradients.** Router/dense-layer gradients under
  EP, and sequence-parallel replicated weights, were never summed across the
  ranks that saw different data. New `sync_replicated_grads`
  (`pkg/distributed/grad_sync.py`), called by `train.py` before clipping;
  the loss convention is documented in that module.
- **F14-init — replica initialisation.** Replicated parameters are now
  broadcast once after model construction (`broadcast_replicated_params`).
- **F20 — Gloo interface.** Importing `pkg.elastic.fault_monitor` no longer
  forces `GLOO_SOCKET_IFNAME=lo` process-wide.

### Added
- `tests/test_differential.py`: distributed-vs-dense-reference tests
  (outputs, every parameter gradient, input gradients, post-step params) with
  distinct per-rank seeds; strict-xfail markers tied to finding IDs.
- `tests/test_comm_ops.py`: finite-difference VJP checks of each collective at
  2 ranks in float64. `tests/test_grad_sync.py`: group-membership tests for
  gradient sync and init broadcast.
- CI job `test-differential` running the multi-process Gloo tests on standard
  GitHub runners.

### Verification
CPU + Gloo, torch 2.14.1+cpu: all 134 differential cases pass on Windows 11
(Python 3.11.9) and WSL Ubuntu 24.04 (Python 3.12.3), and in CI on
ubuntu-22.04 (Python 3.11).

### Known limitations
- Pipeline parallelism (F16): both multi-process PP tests still fail (Linux:
  stage 1 receives 0 of 6 outputs; Windows: hang). Scheduled for M4.
- A rank that raises for an unrelated reason before EP dispatch can still
  leave its peers blocked; only invalid `send_counts` fail on all ranks.
- Not verified: GPU/NCCL (CUDA-stream paths), FSDP2 in a real dp > 1 run,
  multi-node.
