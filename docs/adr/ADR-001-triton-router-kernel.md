# ADR-001: Fused Triton Router Kernel

**Status:** Accepted  
**Date:** June 2026  
**Deciders:** Min Htet Myet  
**Supersedes:** —  
**Superseded by:** —

---

## Context

The MoE router computes four sequential operations on every forward pass:
`tokens @ gate_w → softmax → top-K → renormalize`. In a naive PyTorch
implementation each of these launches a separate CUDA kernel, with each
launch requiring a round-trip through HBM for the intermediate `[N, E]`
logit tensor. At `N=4096, E=64`, this tensor is `4096 × 64 × 4 bytes = 1 MB`
per intermediate — read and written four times, for `4 MB` of unnecessary
HBM traffic per router call.

The alternatives considered were:
1. **Pure PyTorch** — four separate kernels, maximum HBM traffic, no JIT overhead.
2. **torch.compile** — fuses to varying degrees depending on the operation mix; top-K fusion is unreliable; the softmax + top-K + renorm pattern is not consistently fused.
3. **Custom CUDA kernel** — full control, but requires complex CUDA C++ and separate compilation infrastructure.
4. **Custom Triton kernel** — Python-level JIT, fuses all four operations, compatible with PyTorch autograd.

## Decision

We implement a custom **Triton JIT kernel** (`_router_fwd_kernel`) that fuses
all four operations into a single pass over the gating dimension:

```
tile [N, E] → (1) logits = tokens @ gate_w
            → (2) probs  = softmax(logits)
            → (3) topk   = selection_sort_k(probs, K)  [in SRAM]
            → (4) w      = topk_vals / sum(topk_vals)
```

Tile size: `BLOCK_N=64, BLOCK_E=64`, held in L1/SRAM (~49 KiB for float32).
Top-K uses in-SRAM selection sort (K iterations, O(K×E)) which outperforms
bitonic sort for K ∈ {1,2,4}.

`K` is declared `tl.constexpr` in both kernels so `tl.static_range(K)` can
be unrolled at compile time. This was discovered as a v0.3.2 bug fix:
omitting `constexpr` causes `CompilationError` on real GPU hardware (but not
on CPU paths used in CI, which masked it).

A full-precision fp64 PyTorch reference implementation (`_reference_route_fp64`)
serves as ground truth for numerical correctness testing at `atol=rtol=1e-5`.

## Consequences

**Positive:**
- Single HBM pass for the entire routing computation.
- 80.1× throughput improvement over CPU reference at N=4096, H=2048 (T4 validation).
- Analytically correct backward via `MoERouterFunction` (autograd.Function).
- Validated against fp64 reference across 30 `(H, E, K)` configurations.

**Negative / trade-offs:**
- Triton must be installed; falls back to fp64 reference on CPU or without Triton.
- Triton compilation is JIT — first call incurs compile time (~1–3s per unique `(H, E, K, BLOCK_*)` combination; cached after.
- `K: tl.constexpr` means a separate compiled kernel per `top_k` value. This is correct behaviour but means three kernel variants for K ∈ {1,2,4}.

## Alternatives rejected

**torch.compile**: Profile showed inconsistent fusion of softmax + top-K across PyTorch 2.x versions. The renormalization step was never fused with top-K in any tested configuration.

**Custom CUDA C++**: Would require a separate CUDA compilation step, breaking the pure-Python install path. Triton achieves equivalent performance with dramatically lower maintenance burden.
