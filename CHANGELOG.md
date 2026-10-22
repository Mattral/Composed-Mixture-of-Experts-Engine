# Changelog

All notable changes to this project are recorded here. Verification claims
state the environment they were run in; nothing below was run on a GPU, with
NCCL, or across multiple hosts unless it says so.

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
