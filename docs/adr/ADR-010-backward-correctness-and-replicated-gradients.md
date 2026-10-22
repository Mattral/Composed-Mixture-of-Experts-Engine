# ADR-010: Backward Correctness Under EP/TP/SP, Replicated-Gradient Sync, and the Differential Test Harness

**Status:** Accepted
**Date:** October 2026
**Deciders:** Min Htet Myet
**Supersedes:** —
**Related:** ADR-007, ADR-008, ADR-009 (forward-pass fixes in the same layers); October 2026 audit (findings F1–F4, F14, F15, F20, F21).

## Context

Before this change, **the backward pass was never correct whenever any
parallelism dimension was > 1**. Every EP, TP and SP collective was a raw
`torch.distributed` call writing into a freshly allocated buffer, so
autograd had no edge across it:

| Finding | Area | Symptom (CPU, Gloo, 2 ranks) |
|---|---|---|
| F1 | EP dispatch/combine | expert weights got **no gradient**; layer input grad 84–100% wrong |
| F2 | TP Column/Row | `ColumnParallelLinear(gather_output=True)` **raised** in grad mode; Column→Row MLP input grad rel. err. 0.75 |
| F3 | SP scatter/gather | gather **raised during forward** whenever its input required grad; scatter gave each rank only its own chunk's grad |
| F4 | EP send_counts | `E % ep_size != 0` crashed (split-size error) or **deadlocked** (ranks whose counts happened to be valid blocked in the all-to-all) |
| F21 | gradient sync | replicated params (router, dense layers) under EP, and SP-region replicated weights, stepped on **per-rank partial gradients**; nothing reduced them |
| F14-init | initialisation | replicated params were never broadcast; differently seeded ranks started from different replicas |
| F20 | dev env | importing `pkg.elastic.fault_monitor` set `GLOO_SOCKET_IFNAME=lo` process-wide, breaking Gloo on Windows/macOS and pinning multi-host Linux to loopback |

**How this went unnoticed.** The distributed tests checked forward only, often
under `torch.no_grad()`, with the **same seed on every rank** (so expert *j*
was identical on every rank and a token sent to the wrong rank still
matched), every EP case had `E % ep_size == 0`, and CI excluded the
multi-process tier. 409 tests passed while EP/TP/SP training was broken (F15).
It was found by an external audit that ran probes comparing against a
dense single-device reference with distinct per-rank weights.

## Decision

1. **Differential test harness (M0), as the regression guard.**
   `tests/test_differential.py` compares EP, TP, SP and a composite EP model
   against a **dense single-device reference that imports nothing from
   `pkg`**, with distinct per-rank seeds, checking the output, **every
   parameter gradient** (each rank against its shard), input gradients and
   post-optimizer-step parameters, at fp32 `atol=1e-5, rtol=1e-4`. Cases
   that fail are `xfail(strict=True)` naming their finding, so a fix that
   makes one pass fails CI until its marker is removed. A dedicated CI job
   (`test-differential`) runs it, `tests/test_comm_ops.py` and
   `tests/test_grad_sync.py` on standard GitHub runners (Gloo on CPU).

2. **Autograd-correct collectives in one module**, `pkg/distributed/comm_ops.py`:
   `all_to_all` (backward = reverse exchange with split sizes swapped; sizes
   saved on `ctx`, no extra sync), and Megatron's conjugate pairs
   `copy_to_group` / `reduce_from_group`, `gather_along_dim` /
   `split_along_dim`. EP dispatch/combine, `ColumnParallelLinear` /
   `RowParallelLinear` and SP scatter/gather route through them.
   Concatenating along an explicit dim removes the ADR-007/ADR-008 transpose
   and movedim workarounds. Each Function is unit-tested at 2 ranks in
   float64 with a **lockstep global finite-difference VJP check**
   (`distributed_fd_check`; per-rank `torch.autograd.gradcheck` cannot see
   that a rank's output depends on its peers' inputs).

3. **One source of truth for expert ownership, and no-hang dispatch (F4).**
   `ParallelTopology.experts_on_ep_rank` decides ownership; `send_counts` is a
   `bincount` over the derived expert→rank table. The count exchange carries
   a per-rank validity flag in the single collective every rank enters, so
   an invalid `send_counts` on any rank makes **all** ranks raise the same
   error before the data all-to-all.

4. **Replicated-gradient sync with an explicit loss convention (F21).**
   `pkg/distributed/grad_sync.sync_replicated_grads`, called by the trainer
   once per optimizer step before clipping: SUM-all-reduce of non-expert
   params' grads over the EP group, and of params marked
   `mark_sequence_parallel` over the TP group; never over DP (FSDP2 owns
   it, and the groups intersect only in the rank itself, so nothing is
   double-reduced). The objective is
   `(1/dp) · Σ over all ranks of each rank's local loss` — a sum over EP
   ranks and SP shards, a mean over DP — which is exactly the form expert
   gradients already have.

5. **Init broadcast (F14-init).** `broadcast_replicated_params` runs once after
   model construction: every param over DP, non-expert params over EP, and
   params not marked `tensor_parallel_shard` over TP.

6. **No import-time environment mutation for Gloo (F20).**

## Consequences

* EP, TP and SP now train with gradients that match a dense reference: all
  134 differential cases pass with 0 xfail on Windows 11 and WSL Ubuntu
  24.04 (and on GitHub's ubuntu-22.04 runners in CI); CPU/Gloo only.
* The suite can no longer go green with a broken backward: a regression in
  any covered collective fails a dense comparison, and a fix that lands
  without removing its strict-xfail marker fails CI.
* Under the documented convention, users who want a mean over EP ranks must
  divide their local loss by `ep_size`.
* **Not verified:** anything on GPU or NCCL (the CUDA-stream paths in
  `expert_parallel.py` are unexercised), FSDP2 in a real dp > 1 run (the DP
  non-interaction is tested structurally), multi-node.
* **Known limitations:** pipeline parallelism still fails its multi-process
  tests (F16, M4) — on Windows it hangs and leaves workers that keep the
  pytest process alive ~30 min at exit. A rank that raises for an unrelated
  reason *before* EP dispatch can still leave its peers blocked in the
  count exchange; only invalid `send_counts` are made fail-together.

## Alternatives rejected

* **`torch.distributed.nn.functional` collectives.** Uneven-split all-to-all
  support varies across versions and the semantics (which side of a
  replicated tensor sums) are implicit; local Functions are explicit and
  unit-tested.
* **Per-parameter gradient hooks for F21.** They fire per micro-batch under
  gradient accumulation and interleave with FSDP2's own reduce-scatter; one
  explicit call per optimizer step is simpler to reason about.
* **MEAN reduction over EP.** It would be inconsistent with expert gradients,
  which the all-to-all backward already sums over EP ranks.
* **Comparing distributed runs against an ep=1 `DistributedMoELayer`.** It
  shares dispatch code with the system under test (ADR-009's bugs reproduced
  at ep=1); the reference must share none.
