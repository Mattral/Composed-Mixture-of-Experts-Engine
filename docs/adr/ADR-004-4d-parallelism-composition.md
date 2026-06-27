# ADR-004: 4D Parallelism Composition (DP × EP × TP × PP)

**Status:** Accepted  
**Date:** June 2026  
**Deciders:** Min Htet Myet

---

## Context

Large MoE training at scale requires multiple forms of parallelism simultaneously.
The key question is how to compose them and what invariants each combination must maintain.

The four axes in moe-engine:
- **DP** (Data Parallel): replicates the model, shards the batch.
- **EP** (Expert Parallel): shards experts across ranks; uses all_to_all.
- **TP** (Tensor Parallel): shards individual weight matrices within a layer.
- **PP** (Pipeline Parallel): stages the model depth-wise across ranks.

Alternative compositions considered:
1. **DP + EP only** (simplest): limits scale to EP×DP GPUs.
2. **DP + EP + TP** (Megatron-style): adds TP to fit large expert weights; no PP.
3. **Full 4D** (DP × EP × TP × PP): maximum scale; required for very large models.

## Decision

We implement **full 4D parallelism** using `DeviceMesh` with axes `[dp, tp, pp, ep]`.
The product must equal `world_size`. Each axis is independently controlled by
`ParallelismConfig.{data,expert,tensor,pipeline}_parallel`.

**Axis priority and independence:**
- DP and EP are orthogonal: each DP replica contains all EP ranks. FSDP2 wraps along `dp`; all_to_all runs along `ep`.
- TP is within-EP: each EP rank applies TP to its expert weights. `ColumnParallelLinear` and `RowParallelLinear` use the TP process group.
- PP stages the full model depth-wise; each stage executes all four of the above.

**SwiGLU TP sharding rule:** Both `w_gate` and `w_up` are `ColumnParallelLinear`
so their element-wise product `silu(gate(x)) * up(x)` occurs in shard space
`[F//tp_size]`. `w_down` is `RowParallelLinear` to reconstruct `H`. This is
the only correct factorisation — applying ColumnParallel to only one of gate/up
would require an extra all_gather before the element-wise product.

**Expert weight exclusion from FSDP2:** Expert parameters are already sharded
across `ep_size` ranks. Applying FSDP2 along `dp` to expert weights would shard
them a second time, corrupting the expert assignment. `apply_fsdp2()` explicitly
excludes `DistributedMoELayer` expert weights.

## Consequences

**Positive:**
- Supports any factorisation of `world_size` into 4 axes.
- Each axis degrades gracefully to identity when size=1.
- `ParallelTopology` is a frozen dataclass — immutable after construction, safe to share across threads.
- Degenerate 1-rank topology works without `dist.init_process_group`, enabling full CPU development.

**Negative / trade-offs:**
- Four axes × multiple collective types = complex interaction surface. Every new feature must be tested at each relevant axis combination.
- PP requires careful 1F1B scheduling; activation memory during warmup phase scales with `pp_size`.
- TP requires `tp_size` to divide `hidden_dim` and `ffn_dim`; constraints enforced at topology construction.

## Currently validated combinations (June 2026)

| dp | ep | tp | pp | Status |
|----|----|----|-----|--------|
| 1  | 1  | 1  | 1  | ✅ CPU + GPU smoke |
| N  | 1  | 1  | 1  | ✅ FSDP2 unit tests |
| 1  | N  | 1  | 1  | ✅ EP all-to-all unit tests |
| 1  | 1  | 2  | 1  | ✅ 2-rank TP mp.spawn verified |
| 1  | 1  | 1  | 2  | ✅ 2-rank PP mp.spawn verified |
| N  | M  | 1  | 1  | ⚠️ Pending sustained cluster access (v0.4) |
