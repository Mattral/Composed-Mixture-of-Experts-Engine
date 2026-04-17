# Setup and Operations

**Version:** v0.3  
**Last updated:** June 2026

---

## Install

```bash
git clone https://github.com/your-org/moe-engine
cd moe-engine/moe-engine

# Standard install (editable, with dev tools)
pip install -e ".[dev]"

# With Prometheus monitoring support
pip install -e ".[all]"
```

**Manual install from requirements.txt:**

```bash
pip install -r requirements.txt
# GPU kernel (optional — CPU fallback always available)
pip install triton>=3.0.0
# Remote checkpoint tier (optional)
pip install boto3 botocore
# Prometheus metrics endpoint (optional)
pip install prometheus-client
```

**Verify install:**

```bash
python -c "import torch, pkg.kernels.moe_router; print('OK', torch.__version__)"
```

---

## Configuration Reference

Training is configured via YAML. Two canonical configs ship with the repo:

| Config | Purpose |
|---|---|
| `configs/smoke.yaml` | CPU-only 2-step smoke test (hidden=32, layers=2, experts=4) |
| `configs/default.yaml` | H100-scale production config (hidden=4096, experts=64, dtype=bfloat16) |

### All configuration keys

```yaml
model:
  hidden_dim: 4096          # transformer hidden dimension H
  num_layers: 32            # transformer block count
  num_experts: 64           # total expert count E
  top_k: 2                  # active experts per token K
  ffn_dim: 14336            # expert intermediate dimension F
  vocab_size: 128256        # vocabulary size
  sequence_length: 4096     # tokens per sample
  capacity_factor: 1.25     # expert buffer capacity (overflow tokens dropped)
  dtype: bfloat16           # float32 | bfloat16 | float16

training:
  global_batch_size: 4096   # total tokens per step across all ranks
  micro_batch_size: 2       # per-rank batch size before gradient accumulation
  gradient_accumulation_steps: 4   # [v0.2] accumulate before optimizer step
  learning_rate: 3.0e-4
  weight_decay: 0.1
  grad_clip: 1.0
  max_steps: 100000
  log_interval: 10          # emit telemetry every N steps
  ckpt_interval: 500        # save checkpoint every N steps
  warmup_steps: 2000        # [v0.2] linear LR warmup steps

parallelism:
  data_parallel: 8          # FSDP2 sharding axis
  expert_parallel: 8        # EP all-to-all axis
  tensor_parallel: 1        # ColumnParallel / RowParallel axis
  pipeline_parallel: 1      # 1F1B pipeline stage count

checkpoint:
  local_dir: /mnt/nvme/ckpts      # NVMe staging directory
  remote_uri: s3://bucket/path    # s3:// or file:// remote tier
  async_workers: 4               # background I/O thread count
  retention: 10                  # number of checkpoints to retain

elastic:
  min_nodes: 4
  max_nodes: 64
  rdzv_backend: c10d            # c10d (≤100 nodes) | etcd (>100 nodes)
  rdzv_endpoint: head-node:29500
  health_check_interval_s: 5
  drop_grace_period_s: 30

telemetry:
  log_dir: /var/log/moe-engine
  tensorboard_dir: /var/log/moe-engine/tb
  json_path: /var/log/moe-engine/step.jsonl
  hardware_peak_tflops: 989.0   # H100 SXM5 BF16; adjust to your hardware
  mfu_target: 0.55
```

### Parallelism constraint

The product of all parallelism axes must equal `WORLD_SIZE`:

```
dp_size × tp_size × pp_size × ep_size == WORLD_SIZE
```

`train.py` automatically clamps axes to valid values if the product exceeds
world size (useful when running at reduced scale for testing).

---

## Running the Smoke Test (CPU, no GPU required)

```bash
cd moe-engine
python train.py --config configs/smoke.yaml --smoke
# With WandB (requires WANDB_API_KEY):
# python train.py --config configs/smoke.yaml --smoke --wandb-project moe-engine
```

Expected output: 2 training steps, structured JSON telemetry at
`/tmp/moe-engine/logs/step.jsonl` (or as configured), checkpoint at
`/tmp/moe-engine/ckpts/`.

With benchmark profiling:

```bash
python train.py --config configs/smoke.yaml --smoke
# With WandB (requires WANDB_API_KEY):
# python train.py --config configs/smoke.yaml --smoke --wandb-project moe-engine --profile
# Writes: benchmarks/run_<timestamp>_rank0.json
```

---

## Running the Full Test Suite

```bash
# Full suite (CPU, no GPU needed), ~45s:
pytest tests/ -v --ignore=tests/test_chaos.py

# Single test file:
pytest tests/test_tensor_parallel.py -v

# Numerics-only (Triton vs fp64 reference, 30 configs):
python tests/run_numerics_tests.py

# Benchmark suite (CPU sweep):
python benchmarks/run_benchmark.py --json /tmp/bench.json

# Benchmark suite (GPU sweep, requires CUDA + Triton):
python benchmarks/run_benchmark.py --cuda --json /tmp/bench_gpu.json
```

---

## Single-Node Multi-GPU Launch

```bash
torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node=8 \
  train.py --config configs/default.yaml --profile
```

With Prometheus metrics:

```bash
torchrun --standalone --nproc_per_node=8 \
  train.py --config configs/default.yaml --prometheus-port 9102
# → /metrics available at http://localhost:9102/metrics
```

---

## Multi-Node Launch (TorchElastic)

Use `scripts/launch.sh` which wraps `torchrun`:

```bash
NUM_NODES=16 \
GPUS_PER_NODE=8 \
RDZV_ENDPOINT=head-node:29500 \
RDZV_ID=moe-run-001 \
MAX_RESTARTS=10 \
CONFIG=configs/default.yaml \
S3_ENDPOINT_URL=http://minio.internal:9000 \
AWS_ACCESS_KEY_ID=<key> \
AWS_SECRET_ACCESS_KEY=<secret> \
bash scripts/launch.sh
```

For >100 nodes, set `elastic.rdzv_backend: etcd` in your config and point
`RDZV_ENDPOINT` at your etcd cluster. `ElasticTrainerHarness` selects the
backend automatically based on this config value.

---

## Docker

```bash
# Build image
docker build -f deploy/docker/Dockerfile -t moe-engine:v0.3 .

# CPU smoke test (no GPU required)
docker compose -f deploy/docker/docker-compose.yml run --rm smoke

# 4-GPU training run
docker compose -f deploy/docker/docker-compose.yml run --rm train-4gpu

# 8-GPU training run with monitoring stack
docker compose -f deploy/docker/docker-compose.yml --profile monitoring up -d train-8gpu prometheus grafana

# Run test suite inside container
docker compose -f deploy/docker/docker-compose.yml run --rm test
```

---

## Kubernetes

```bash
# Create namespace + PVC + config
kubectl apply -f deploy/k8s/namespace.yaml
kubectl apply -f deploy/k8s/pvc.yaml
kubectl apply -f deploy/k8s/configmap.yaml

# Single-node 8-GPU job
kubectl apply -f deploy/k8s/training-job.yaml
kubectl logs -n moe-engine -l job-name=moe-training -f

# Multi-node (16 × 8GPU = 128 GPUs total) with etcd rendezvous
kubectl apply -f deploy/k8s/training-job-multinode.yaml

# Monitor (Prometheus scrapes :9102/metrics from training pods)
kubectl port-forward -n moe-engine svc/prometheus 9090:9090
```

For S3 credentials, create a secret before applying jobs:

```bash
kubectl create secret generic moe-engine-s3 -n moe-engine \
  --from-literal=endpoint=http://minio.internal:9000 \
  --from-literal=access_key_id=<key> \
  --from-literal=secret_access_key=<secret>
```

---

## Checkpointing and Recovery

Checkpoints are written asynchronously in the background. The training thread
blocks only for the D2H copy of the parameter shard. The background thread:

1. Writes 256 MB chunks to `checkpoint.local_dir` (NVMe) with `O_DIRECT`.
2. Atomically renames `tmp_<step>_<rank>` → `step=<N>/rank=<R>.pt`.
3. Writes `.meta.json` with step, rank, timestamp, hostname.
4. Copies shard to `checkpoint.remote_uri` (S3 or file://).
5. Prunes checkpoints older than `retention` steps.

On resume, `ElasticTrainerHarness` discovers the latest step via
`AsyncCheckpointer.latest_step()` and loads before entering the training loop.

If a remote URI is not configured, training continues with local-only
checkpointing (no S3 upload). Loss of the node means loss of the checkpoint.

---

## Chaos and Failure Testing

```bash
# Baseline (no fault, verifies clean recovery path):
GLOO_SOCKET_IFNAME=lo pytest tests/test_chaos.py -v -k "baseline"

# Scenario B — storage stall (10s injected I/O latency, ✅ passes reliably):
GLOO_SOCKET_IFNAME=lo pytest tests/test_chaos.py -v -k "scenario_b"

# Scenario A — node kill + recovery (⚠️ ~85% pass rate; Gloo race):
CHAOS_FAULT_TOLERANT=1 GLOO_SOCKET_IFNAME=lo \
  pytest tests/test_chaos.py -v -k "scenario_a"

# Manual failure injection (SIGKILL a rank):
bash scripts/simulate_node_failure.sh -r 2,3
```

---

## Observability

### Telemetry JSON

Each step emits one JSONL record to `telemetry.json_path`. Key fields:

| Field | Description |
|---|---|
| `mfu` | Model FLOPs Utilization (sparse-aware, 0–1) |
| `tokens_per_sec` | Training throughput |
| `collective.all_to_all_dispatch_ms` | EP dispatch latency (CUDA event) |
| `collective.all_to_all_combine_ms` | EP combine latency (CUDA event) |
| `collective.expert_compute_ms` | Expert FFN wall-clock (v0.3) |
| `collective.comm_compute_overlap_ratio` | dispatch_ms / expert_compute_ms (v0.3) |
| `memory.peak_allocated_gb` | Peak CUDA memory (torch.cuda.memory_stats) |
| `routing.expert_load_imbalance` | max/mean dispatch ratio (1.0 = perfect) |
| `routing.router_z_loss` | Auxiliary regularisation signal |
| `infra.async_ckpt_commit_ms` | Checkpoint background commit time |
| `infra.lr` | Current learning rate (v0.2: cosine schedule) |

### TensorBoard

```bash
tensorboard --logdir <telemetry.tensorboard_dir>
```

All numeric sub-fields are emitted as scalar summaries under their section
prefix (e.g., `collective/all_to_all_dispatch_ms`).

### Prometheus

Start training with `--prometheus-port 9102`. Scrape at `http://host:9102/metrics`.
Eight gauges: loss, mfu, tokens/sec, dispatch_ms, combine_ms, peak_memory_gb,
expert_load_imbalance, router_z_loss.

---

## Operational Notes

- Set `TORCH_NCCL_ASYNC_ERROR_HANDLING=1` and `TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=30`
  in all cluster environments. The Docker image and K8s ConfigMap set these
  automatically.
- Do not store S3 credentials in config YAML. Use environment variables or
  Kubernetes secrets only.
- For `tp_size > 1`, ensure `sequence_length % tp_size == 0` (assertion in
  `scatter_to_sequence_parallel`).
- Monitor `routing.expert_load_imbalance` per step. Values above 1.5 sustained
  over many steps indicate routing collapse; add a z-loss auxiliary term
  (weight ~1e-3) to your loss function.
- `hardware_peak_tflops` in the telemetry config is used for MFU calculation.
  Incorrect values produce misleading MFU numbers. See GPU spec sheets:
  H100 SXM5 BF16 = 989 TFLOPS; A100 SXM4 BF16 = 312 TFLOPS.
