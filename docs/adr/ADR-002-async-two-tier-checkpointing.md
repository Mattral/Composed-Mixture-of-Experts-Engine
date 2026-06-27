# ADR-002: Async Two-Tier Checkpointing

**Status:** Accepted  
**Date:** June 2026  
**Deciders:** Min Htet Myet  
**Supersedes:** —

---

## Context

At hyperscale (10K+ GPUs), checkpointing is on the critical path for recovery
time. A checkpoint that takes 10 minutes to write to S3 means 10 minutes of
idle GPUs after every node failure. Two competing constraints:

1. **Durability**: checkpoints must survive a full node loss (NVMe alone is not durable).
2. **Speed**: the training loop must resume within seconds of a failure, not minutes.

Naive approaches:
- **Synchronous S3 write**: training pauses for the full upload duration (~2–10 min for large shards). Unacceptable.
- **Synchronous local write**: fast, but checkpoints lost if node dies before S3 upload.
- **In-memory only**: no durability at all.

## Decision

We implement **async two-tier checkpointing** (`AsyncCheckpointer`):

```
Training thread  →  [non-blocking]  →  Background I/O thread
                                           │
                                           ├── Tier 1: NVMe (fast, local)
                                           │    O_DIRECT, 256 MB chunks
                                           │    atomic rename (.tmp → final)
                                           │    prune oldest (keep `retention`)
                                           │
                                           └── Tier 2: S3 / MinIO (durable, remote)
                                                boto3 multipart upload
                                                .meta.json (step, rank, ts, hash)
                                                prune remote
```

The training thread pays only the D2H copy cost (~tens of ms per shard).
All disk and network I/O runs in `async_workers` background threads (default 4).

Atomic rename (`tmp → final`) on Tier 1 ensures every NVMe checkpoint is either
fully present or absent — no partial reads possible even if the node is killed
during the write.

After a node drop, `ElasticTrainerHarness.recover()` reads from Tier 1 (NVMe,
local, fast) rather than Tier 2 (S3, remote, slow). Tier 2 is the durable
fallback if the NVMe node itself is lost.

## Consequences

**Positive:**
- Training loop sees only the D2H copy cost, not the full I/O cost.
- Atomic rename guarantees checkpoint integrity even under SIGKILL.
- Fast recovery (seconds from NVMe vs minutes from S3).
- Scenario B (storage stall, 10s injected) passes at 100% — the queue drains without deadlock.

**Negative / trade-offs:**
- Recovery requires the same node (or a node with the same NVMe mount) to use the fast path.
- If the NVMe node is permanently lost, recovery falls back to S3 (slower).
- Background thread complexity adds a failure mode: if the background thread raises, it is logged and the run continues without checkpointing until the next interval.

## Alternatives rejected

**Fully synchronous checkpointing**: Pauses training for the full I/O duration. At 7B parameter scale with BF16 this is ~14 GB per shard × 4 ranks = 56 GB, taking 2–5 minutes on PCIe NVMe. Unacceptable at multi-node scale.

**Memory-mapped files**: Faster than O_DIRECT for small writes, but O_DIRECT avoids page cache pressure and is more predictable under sustained write load.

**Single S3 tier**: Network latency is 10–100× higher than NVMe. Fast recovery path requires local tier.
