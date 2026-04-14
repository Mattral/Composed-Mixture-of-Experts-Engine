# Quickstart

**Version:** v0.2  
**Last updated:** June 2026

Get from zero to a running MoE training step in under 5 minutes — no GPU required.

---

## Prerequisites

- Python ≥ 3.10
- pip ≥ 23
- 4 GB RAM (smoke config)

No CUDA, no GPU, no Docker needed for steps 1–5.

---

## Step 1 — Clone and install

```bash
git clone https://github.com/your-org/moe-engine
cd moe-engine/moe-engine
pip install -e ".[dev]"
```

**Verify:**

```bash
python -c "import pkg.kernels.moe_router; print('moe-engine OK')"
```

---

## Step 2 — Run the smoke test

```bash
python train.py --config configs/smoke.yaml --smoke
```

This runs 2 training steps on a tiny MoE model (hidden=32, 4 experts, 2 layers)
using the CPU fp64 reference kernel path. Expected output:

```
INFO ... step=0 loss=4.8802 MFU=0.00% (smooth=0.00%) | 1847 tok/s | step=8.7ms ...
INFO ... step=1 loss=4.8741 MFU=0.00% (smooth=0.00%) | 1923 tok/s | step=8.3ms ...
```

MFU is near zero because the CPU reference path is not GPU hardware. On H100
with the Triton kernel, expect 40–50% MFU at production config.

Outputs written to `/tmp/moe-engine/` (configurable in `configs/smoke.yaml`):
- `logs/step.jsonl` — structured per-step telemetry
- `ckpts/` — checkpoint shards

---

## Step 3 — Run the test suite

```bash
pytest tests/ -v --ignore=tests/test_chaos.py
```

Expected: **123 passed, 1 skipped** in ~45 seconds on a modern laptop.
The single skip is the GPU-only Triton kernel path.

---

## Step 4 — Inspect the telemetry

```bash
# Read the last emitted step record
tail -1 /tmp/moe-engine/logs/step.jsonl | python -m json.tool
```

You will see a record like:

```json
{
  "step": 1,
  "loss": 4.8741,
  "mfu": 0.0003,
  "tokens_per_sec": 1923,
  "wall_clock_ms": 8.3,
  "kernel": {
    "sram_bytes_per_block": 16384,
    "tokens_per_expert_mean": 4.0,
    "tokens_per_expert_std": 1.41,
    "used_triton": false
  },
  "collective": {
    "all_to_all_dispatch_ms": 0.0,
    "all_to_all_combine_ms": 0.0
  },
  "routing": {
    "expert_load_imbalance": 1.12,
    "router_z_loss": 2.87
  },
  "infra": {
    "async_ckpt_commit_ms": 0.0,
    "active_nodes": 1,
    "ep_world_size": 1,
    "lr": 0.0003
  },
  "rank": 0,
  "ts": 1748901234.56
}
```

Key v0.2 additions: `routing.expert_load_imbalance` (1.0 = perfect balance)
and `routing.router_z_loss` (auxiliary regularisation signal).

---

## Step 5 — Run the benchmark suite

```bash
python benchmarks/run_benchmark.py --json /tmp/bench.json
python -c "
import json
r = json.load(open('/tmp/bench.json'))
for x in r:
    if x['passed']:
        print(f\"{x['name']:30s} {x['batch_ms_mean']:7.2f}ms  {x['tokens_per_sec']/1e6:5.2f}M tok/s\")
"
```

---

## Step 6 — Validate the router numerics

```bash
python tests/run_numerics_tests.py
```

This runs 30 parametrised forward/backward tolerance checks across
`H ∈ {64,128,256,512}`, `E ∈ {8…256}`, `K ∈ {1,2,4}`, all against the
fp64 reference at `atol=rtol=1e-5`.

---

## Step 7 — Run a multi-GPU training step (requires GPU)

```bash
torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node=4 \
  train.py --config configs/default.yaml --max-steps 10 --profile
```

The `--profile` flag writes a per-step benchmark JSON to `benchmarks/`:

```bash
ls benchmarks/run_*.json | tail -1 | xargs python -c "
import json, sys
r = json.load(open(sys.argv[1]))
print(f\"Steps: {r['steps']}\")
print(f\"MFU mean: {r['mfu_mean']:.2%}\")
print(f\"Tokens/sec: {r['tokens_per_sec_mean']:,.0f}\")
"
```

---

## Step 8 — Docker smoke (no GPU required)

```bash
docker build -f deploy/docker/Dockerfile -t moe-engine:v0.2 .
docker compose -f deploy/docker/docker-compose.yml run --rm smoke
```

---

## Step 9 — Monitor with TensorBoard

```bash
tensorboard --logdir /tmp/moe-engine/logs/tb
# Open http://localhost:6006
```

Scalars available: `loss`, `mfu`, `tokens_per_sec`, `collective/dispatch_ms`,
`collective/combine_ms`, `memory/peak_allocated_gb`, `routing/expert_load_imbalance`,
`routing/router_z_loss`.

---

## Step 10 — Multi-node cluster launch

When you have cluster access:

```bash
NUM_NODES=2 \
GPUS_PER_NODE=8 \
RDZV_ENDPOINT=head-node:29500 \
RDZV_ID=moe-run-001 \
CONFIG=configs/default.yaml \
bash scripts/launch.sh
```

For > 100 nodes, set `elastic.rdzv_backend: etcd` in your config.
See `docs/deployment.md` for the full deployment reference.

---

## What to read next

| Document | Purpose |
|---|---|
| `docs/ARCHITECTURE.md` | System component map, 4D mesh, kernel design |
| `docs/DESIGN.md` | Why each design choice was made; tradeoffs considered |
| `docs/SYSTEM_DESIGN.md` | Full component API reference |
| `docs/testing.md` | Test suite guide; how to write new tests |
| `docs/deployment.md` | Docker, Kubernetes, multi-node operations |
| `docs/CONTRIBUTING.md` | Contribution workflow and standards |
| `benchmarks/BENCHMARKS.md` | Benchmark methodology and results |
| `RESULTS.md` | Reproducible results and telemetry samples |
| `roadmap.md` | Honest status, known deficiencies, v0.3 plan |
