# ADR-011: Layout-Independent Training Objective (Mean Over the Global Batch)

**Status:** Accepted
**Date:** October 2026
**Deciders:** Min Htet Myet
**Supersedes:** ADR-010, Decision 4 ("Replicated-gradient sync with an explicit loss convention"), as far as the loss convention goes. The F21 reductions themselves stay. ADR-010 is not edited.
**Related:** ADR-010; October 2026 audit, M2 (findings F22, F24, F25, F26).

## Context

M1 shipped this objective (ADR-010, `pkg/distributed/grad_sync.py`):

    L = (1 / dp) · Σ over all ranks r of L_r

where `L_r` is whatever rank `r` computes on its own tokens, in practice a
mean over them, divided by the number of accumulation steps. The gradient
a parameter receives therefore depends on how the global batch is laid out:

* **EP and SP sum** per-rank losses: at ep = 2, two per-rank means give 2×
  the global mean.
* **DP averages** per-rank means. That equals the global mean only when
  every DP rank holds the same number of loss-bearing tokens.
* **Gradient accumulation** averages per-micro-batch means. That weights
  tokens unequally whenever micro-batches hold different token counts.

With loss masks, padding or packed sequences, token counts are rarely
equal. Changing `dp`, `ep`, `tp` or `gradient_accumulation_steps` then
changes the effective learning rate and how much weight each token gets.

**Evidence (F22).** `tests/test_differential.py::test_layout_*` uses one
fixed global batch: 4 sequences × 2 micro-batches, with loss masks keeping
7, 2, 5, 1 and 3, 6, 0, 4 tokens. It runs the engine path under seven
layouts and compares against a dense reference whose loss is the mean over
the global batch's loss-bearing tokens. Under the M1 convention every
layout's gradients were wrong. Relative error of `inp.weight.grad`
(CPU/Gloo, Windows and WSL identical):

| layout | single | ep2 | sp2 | ep2-sp2 | dp2-ep2 | dp2-sp2 | dp2 |
|---|---|---|---|---|---|---|---|
| rel. err. | 0.065 | 1.15 | 0.052 | 1.28 | 1.11 | 0.29 | 0.33 |

The dp > 1 numbers could only be measured after two FSDP2 bugs that this
test exposed were fixed:

* **F24.** `apply_fsdp2` passed `mp_policy=None` whenever mixed precision
  was off, and FSDP2 raised at the first forward. No fp32 run at dp > 1
  had ever taken a step.
* **F25.** DP ranks whose tokens skip different experts reduce-scattered
  different parameter sets. That causes a hang, or, because experts are
  equal-sized, one expert's gradient silently summed into another's.

## Decision

1. **The objective is the mean over the global batch's loss-bearing tokens:**

       L = (1 / N_global) · Σ over loss-bearing tokens t of ℓ_t

   * The **global batch** is every micro-batch of one optimizer step, on
     every data-distinct rank (DP × EP).
   * TP ranks of one (dp, ep) position hold the same tokens, and SP only
     shards the regions between them. The loss is computed on TP-replicated
     outputs, after the SP gather, so each token counts once.
   * `N_global` counts loss-bearing tokens, not all tokens: masked and
     padded tokens do not dilute the mean.

2. **Gradient accumulation: accumulate unnormalised sums, divide once.**
   * Each micro-batch backpropagates its local **sum** `Σ ℓ_t`.
   * After the last micro-batch, `grad_sync.finalize_grads(model, topology,
     N_global)` sums replicated gradients over EP/SP (`sync_replicated_grads`,
     unchanged) and divides every gradient by `N_global`.
   * `N_global = grad_sync.global_token_count(local_count, topology)`: one
     int64 all-reduce over the DP and EP groups per optimizer step.
   * Clipping and `optimizer.step()` come after `finalize_grads`.

   The alternative, counting all micro-batches' tokens before the first
   backward, was rejected. It needs every micro-batch's targets (or masks)
   before any compute, so a streaming loader must prefetch the whole step.
   Dividing at the end needs only each rank's running count. Megatron-LM's
   per-token loss mode makes the same choice. The cost is that gradients
   hold unnormalised sums until `finalize_grads`. That is harmless with
   fp32 gradients and fp32 reduction (`apply_fsdp2` reduces in fp32 under
   mixed precision) but is an overflow risk for fp16 gradient buffers.

3. **Every cross-rank gradient reduction is a SUM.** That means EP and the
   TP group for SP-region parameters (`sync_replicated_grads`), and now
   also DP. `apply_fsdp2` sets, on every FSDP2 module:
   * `set_force_sum_reduction_for_comms(True)` and
     `set_gradient_divide_factor(1.0)`: FSDP2 sums over DP instead of
     averaging.
   * `set_reduce_scatter_unused_params(True)`: experts unused on one DP rank
     contribute zeros (F25).

4. **These FSDP2 methods depend on the torch version.**
   * They exist in the pinned torch, **2.14.1** (CPU wheel), on Windows,
     WSL Ubuntu 24.04 and the CI differential job. I have not checked which
     release introduced each one.
   * `tests/test_grad_sync.py::test_fsdp2_force_sum_api_present` fails if
     any of them is missing.
   * `apply_fsdp2` raises at wrap time, naming the installed torch version,
     rather than training with the wrong reduction.
   * The CI differential job installs `torch==2.14.1`, so CI tests the
     version that was verified.
   * Gloo-specific behaviour seen with 2.14.1: a divide factor of 1
     *without* force-sum makes FSDP2 use `ReduceOp.PREMUL_SUM`, which Gloo
     rejects ("Cannot use ReduceOp.PREMUL_SUM with Gloo"). With force-sum
     it uses plain `SUM` plus a post-scale. The default (no divide factor)
     uses AVG, which works on Gloo but is the mean this ADR replaces.
   * Re-check all of this on every torch upgrade, and on NCCL before any
     GPU claim.

5. **Scope.**
   * The router auxiliary losses (F5) and capacity dropping (F12) are M2
     items with their own sections below.
   * Capacity dropping depends on the layout by definition, because capacity
     is a per-group budget. It is therefore not part of the cross-layout
     invariance test.

## Consequences

* The same global batch gives the same gradients and the same post-step
  parameters under single, dp2, ep2, sp2 (tp2), dp2×ep2 and ep2×sp2, each
  matching a dense global-mean reference at fp32 `atol=1e-5, rtol=1e-4`.
  CPU + Gloo only (Windows 11, WSL Ubuntu 24.04, GitHub ubuntu runners).
* dp2×sp2 still fails, on `sp_w` only (strict xfail **F26**). FSDP2
  replaces every parameter with a new DTensor parameter, so the
  `mark_sequence_parallel` attribute is lost. At dp > 1 with tp > 1,
  `sync_replicated_grads` then never sums an SP-region weight over TP.
  F22 had been hiding this.
* FSDP2 at dp = 2 is now exercised on CPU/Gloo (layouts dp2, dp2×ep2 and
  dp2×sp2, plus unit tests). **Not verified:** FSDP2 on NCCL or GPU,
  dp > 2, multi-node.
* The learning rate no longer has to be retuned when the layout or the
  accumulation count changes, provided the global batch stays the same.
* `train.py` logs each rank's mean CE over its own tokens; the objective
  itself is the global mean.
* `train.py` still trains on per-rank `torch.randint` data with no seeding
  (F18). That includes TP ranks, which should see identical data. This ADR
  fixes the objective, not the data path.

## Alternatives rejected

* **Keep FSDP2's DP mean and scale each rank's loss by `dp`.** It is
  numerically equivalent, but the convention would then live in two
  places, the loss scale and the FSDP2 default. Every reduction being a
  SUM, with one division by `N_global`, puts it in one place.
* **MEAN reductions with per-group divisors.** These are correct only when
  every rank holds the same token count, which is exactly the assumption
  that fails.
* **A non-FSDP DP path for testing.** It is not needed: FSDP2 runs at dp = 2
  on CPU/Gloo once F24 and F25 are fixed, so the tests exercise the real
  path.
