# Deployment Guide

**Version:** v0.3  
**Last updated:** June 2026

---

## Supported Deployment Modes

| Mode | Use case | Entry point |
|---|---|---|
| CPU single-process | Development, CI, smoke testing | `python train.py --smoke` |
| Single-node multi-GPU | Integration testing, small runs | `torchrun --standalone` |
| Multi-node (c10d rendezvous) | ≤ 100 nodes | `scripts/launch.sh` + c10d |
| Multi-node (etcd rendezvous) | > 100 nodes / production | `scripts/launch.sh` + etcd |
| Docker (single-node) | Reproducible smoke + integration | `docker compose run` |
| Kubernetes (single-node Job) | Cloud GPU clusters | `deploy/k8s/training-job.yaml` |
| Kubernetes (multi-node Indexed Job) | 16-node+ production | `deploy/k8s/training-job-multinode.yaml` |

---

## Runtime Requirements

### Python packages (minimum)

```
torch >= 2.5.0
triton >= 3.0.0       # GPU kernel path; CPU fallback always available
numpy >= 1.26.0
pyyaml >= 6.0.1
tensorboard >= 2.16.0
psutil >= 5.9.0
```

### Optional packages

```
boto3, botocore        # S3/MinIO remote checkpoint tier
prometheus-client      # /metrics endpoint (--prometheus-port)
```

### Infrastructure

| Resource | Required | Notes |
|---|---|---|
| GPU | No (CPU fallback) | CUDA 12.4+ for Triton kernel path |
| NVMe storage | Recommended | Local checkpoint staging; HDD works but is slower |
| Rendezvous endpoint | Multi-node only | c10d: head-node TCP; etcd: dedicated service |
| S3 / object storage | No | Remote checkpoint durability tier |
| etcd | > 100 nodes only | `elastic.rdzv_backend: etcd` in config |

---

## Environment Variables

### Required for multi-node

| Variable | Purpose |
|---|---|
| `WORLD_SIZE` | Total GPU count across all nodes |
| `RANK` | Global rank of this process |
| `MASTER_ADDR` | Rendezvous host address |
| `MASTER_PORT` | Rendezvous port |

`torchrun` sets these automatically. Only set manually for custom launchers.

### For `scripts/launch.sh`

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `NUM_NODES` | ✅ | — | Number of nodes |
| `GPUS_PER_NODE` | ✅ | — | GPUs per node |
| `RDZV_ENDPOINT` | ✅ | — | `host:port` for rendezvous |
| `RDZV_ID` | ✅ | — | Unique run identifier |
| `MAX_RESTARTS` | No | `10` | TorchElastic max restarts |
| `CONFIG` | No | `configs/default.yaml` | Path to YAML config |

### For S3 / MinIO checkpoint tier

| Variable | Purpose |
|---|---|
| `S3_ENDPOINT_URL` | MinIO or custom S3 endpoint |
| `AWS_ACCESS_KEY_ID` | Access key |
| `AWS_SECRET_ACCESS_KEY` | Secret key |

### For NCCL tuning (set by Docker/K8s manifests automatically)

```bash
TORCH_NCCL_ASYNC_ERROR_HANDLING=1
TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=30
TORCH_NCCL_TRACE_BUFFER_SIZE=1048576
NCCL_IB_DISABLE=0
NCCL_NET_GDR_LEVEL=5     # GPUDirect RDMA level
OMP_NUM_THREADS=8
```

### For chaos testing

| Variable | Purpose |
|---|---|
| `CHAOS_FAULT_TOLERANT` | Set to `1` to enable exponential-backoff retries in `_safe_all_reduce` |
| `CHAOS_LATENCY_STEP` | Step at which to inject storage stall |
| `CHAOS_LATENCY_SECONDS` | Duration of injected stall (Scenario B) |
| `GLOO_SOCKET_IFNAME` | Set to `lo` for local single-machine chaos tests |

---

## Local Development

### Minimal smoke test (no GPU, no Docker)

```bash
cd moe-engine
pip install -e ".[dev]"
python train.py --config configs/smoke.yaml --smoke
```

### Single-node 8-GPU with profiling

```bash
torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node=8 \
  train.py \
    --config configs/default.yaml \
    --profile \
    --prometheus-port 9102
```

---

## Docker Deployment

### Build the image

```bash
docker build -f deploy/docker/Dockerfile -t moe-engine:v0.3 moe-engine/
```

The `Dockerfile` uses a two-stage build:
- **Builder:** installs all Python deps including Triton.
- **Runtime:** minimal image with only installed packages and source copied in.

Key environment variables are baked into the runtime stage (NCCL tuning, OMP,
PYTHONUNBUFFERED). A HEALTHCHECK pings the package import to verify the image is
functional.

### Run with docker compose

```bash
# CPU smoke test (no GPU required)
docker compose -f deploy/docker/docker-compose.yml run --rm smoke

# 4-GPU training run
docker compose -f deploy/docker/docker-compose.yml run --rm train-4gpu

# 8-GPU with full monitoring stack (Prometheus + Grafana)
docker compose -f deploy/docker/docker-compose.yml \
  --profile monitoring up -d train-8gpu prometheus grafana

# Run test suite inside container
docker compose -f deploy/docker/docker-compose.yml run --rm test
```

The `docker-compose.yml` mounts the source tree read-only and a named volume
`moe_checkpoints` for persistent checkpoint storage.

### Access monitoring

```
Prometheus:  http://localhost:9090
Grafana:     http://localhost:3000  (admin / admin)
Metrics:     http://localhost:9102/metrics  (from training container)
```

---

## Kubernetes Deployment

### Single-node 8-GPU Job

```bash
kubectl apply -f deploy/k8s/namespace.yaml
kubectl apply -f deploy/k8s/pvc.yaml         # ReadWriteMany PVC for checkpoints
kubectl apply -f deploy/k8s/configmap.yaml   # NCCL + rendezvous env vars
kubectl apply -f deploy/k8s/training-job.yaml

# Monitor
kubectl logs -n moe-engine -l job-name=moe-training -f

# Watch pod status
kubectl get pods -n moe-engine -w
```

The Job spec (`deploy/k8s/training-job.yaml`):
- Requests 8× `nvidia.com/gpu` and `64` CPUs.
- Mounts `/dev/shm` (64 GiB `Memory` emptyDir) for NCCL shared memory.
- Mounts the PVC at `/mnt/ckpts`.
- Exposes Prometheus metrics on container port 9102.
- Has a liveness probe: `python -c "import torch; torch.zeros(1)"`.

### Multi-node Indexed Job (16 nodes × 8 GPUs = 128 GPUs)

```bash
kubectl apply -f deploy/k8s/training-job-multinode.yaml
```

The multi-node Job (`training-job-multinode.yaml`):
- Uses `completionMode: Indexed` so each pod gets a stable index.
- Runs `torchrun --nnodes=16 --nproc_per_node=8` with etcd rendezvous at
  `etcd-headless.moe-engine:2379`.
- `--max_restarts=10` allows TorchElastic to restart failed pods up to 10 times.
- The PVC is `ReadWriteMany` (requires EFS/GlusterFS/Longhorn — set
  `storageClassName` in `pvc.yaml` appropriately for your cluster).

### S3 credentials as a Kubernetes secret

```bash
kubectl create secret generic moe-engine-s3 -n moe-engine \
  --from-literal=endpoint=http://minio.internal:9000 \
  --from-literal=access_key_id=<key> \
  --from-literal=secret_access_key=<secret>
```

The Job spec reads these via `secretKeyRef` into environment variables.
If the secret does not exist, the S3 keys are optional and local-only
checkpointing is used.

---

## Multi-Node via `scripts/launch.sh`

```bash
# 32-node, 8 GPU/node = 256 GPUs total
NUM_NODES=32 \
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

For > 100 nodes, switch to etcd rendezvous:

```bash
# In configs/default.yaml:
elastic:
  rdzv_backend: etcd
  rdzv_endpoint: etcd.internal:2379
```

`ElasticTrainerHarness` reads `rdzv_backend` from config and selects the
appropriate TorchElastic handler automatically.

---

## Rendezvous Backends

### c10d (default, ≤ 100 nodes)

- Uses a TCP store on the head node.
- No external dependency.
- `connectFullMesh` scales as O(N²) — not suitable above ~100 nodes.
- After a rank failure, re-formation is subject to the Gloo socket race
  (see `roadmap.md §Known Deficiencies`). Use `CHAOS_FAULT_TOLERANT=1` and
  `MAX_RESTARTS` tuned to your node kill rate.

### etcd (> 100 nodes / production)

- Distributed key-value store; handles rendezvous durably across restarts.
- `ElasticTrainerHarness._init_etcd_rendezvous` creates an etcd-backed
  `DynamicRendezvousHandler` with epoch tracking.
- Deploy etcd as a StatefulSet or use a managed etcd service.
- Point `elastic.rdzv_endpoint` at the etcd client endpoint (`host:2379`).
- TorchElastic's etcd backend handles node joins and departures atomically —
  no Gloo socket race.

---

## Checkpoint Deployment Notes

The elastic checkpoint stack writes in two tiers:

| Tier | Path | Performance | Durability |
|---|---|---|---|
| Local NVMe | `checkpoint.local_dir` | 3–7 GB/s sequential | Node-local |
| Remote S3/MinIO | `checkpoint.remote_uri` | ~1 GB/s network | Durable |

On resume, `ElasticTrainerHarness` reads from the local tier (fastest). If
the node is replaced after a failure, the new node loads from the remote tier.

**PVC sizing:** For a model with P parameters at bf16, each rank's shard is
approximately `P / (dp_size × ep_size) × 2 bytes`. The PVC should hold at
least `retention × world_size × shard_size` bytes. For a 70B model at 128
GPUs with retention=10: ~1.1 TB.

---

## Deployment Validation Checklist

Before a production run:

```bash
# 1. Smoke test the image / environment
python train.py --config configs/smoke.yaml --smoke

# 2. Run the full test suite
pytest tests/ -v --ignore=tests/test_chaos.py

# 3. Verify checkpoint round-trip
pytest tests/test_elastic.py tests/test_elastic_v02.py -v

# 4. Verify telemetry fields
pytest tests/test_telemetry.py tests/test_smoke_e2e.py -v

# 5. Run benchmark suite to establish baseline
python benchmarks/run_benchmark.py --json baseline.json

# 6. Chaos Scenario B (storage stall — must pass before production)
GLOO_SOCKET_IFNAME=lo pytest tests/test_chaos.py -v -k "scenario_b" -m chaos
```
