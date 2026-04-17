#!/usr/bin/env bash
# scripts/launch.sh
# =================
# Elastic torchrun launcher for moe-engine.
#
# Supports single-node, multi-node bare-metal, Kubernetes/Kubeflow, and Slurm.
# The rendezvous backend is selected from the CONFIG file's elastic.rdzv_backend
# key, so the same script works for both c10d (<=100 nodes) and etcd (>100 nodes).
#
# Required environment
# --------------------
#   NUM_NODES        total node count (default: 1)
#   GPUS_PER_NODE    GPUs per node    (default: 8)
#   RDZV_ENDPOINT    host:port for rendezvous (default: localhost:29400)
#   CONFIG           path to YAML config      (default: configs/default.yaml)
#
# Optional environment
# --------------------
#   RDZV_ID          unique run ID (auto-generated from timestamp if absent)
#   MAX_RESTARTS     TorchElastic max restarts (default: 10)
#   RDZV_BACKEND     override rendezvous backend: c10d | etcd | etcd-v2
#                    (if unset, read from CONFIG elastic.rdzv_backend)
#   S3_ENDPOINT_URL  MinIO / custom S3 endpoint
#   AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY
#   PROMETHEUS_PORT  expose /metrics on this port (default: disabled)

set -euo pipefail

NUM_NODES="${NUM_NODES:-1}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
RDZV_ID="${RDZV_ID:-moe-engine-$(date +%Y%m%d-%H%M%S)}"
RDZV_ENDPOINT="${RDZV_ENDPOINT:-localhost:29400}"
MAX_RESTARTS="${MAX_RESTARTS:-10}"
CONFIG="${CONFIG:-configs/default.yaml}"

# Resolve rendezvous backend: explicit env var > config file > default
if [[ -z "${RDZV_BACKEND:-}" ]]; then
  if command -v python3 >/dev/null 2>&1 && [[ -f "$CONFIG" ]]; then
    RDZV_BACKEND=$(python3 -c "
import yaml, sys
try:
    cfg = yaml.safe_load(open('$CONFIG'))
    print(cfg.get('elastic', {}).get('rdzv_backend', 'c10d'))
except Exception:
    print('c10d')
" 2>/dev/null || echo "c10d")
  else
    RDZV_BACKEND="c10d"
  fi
fi

# Change to repo root so relative paths in configs work.
cd "$(dirname "$0")/.."

echo "[launch] NUM_NODES=${NUM_NODES} GPUS_PER_NODE=${GPUS_PER_NODE}"
echo "[launch] RDZV_BACKEND=${RDZV_BACKEND} RDZV_ENDPOINT=${RDZV_ENDPOINT}"
echo "[launch] RDZV_ID=${RDZV_ID} MAX_RESTARTS=${MAX_RESTARTS}"
echo "[launch] CONFIG=${CONFIG}"

EXTRA_ARGS=()
if [[ -n "${PROMETHEUS_PORT:-}" ]]; then
  EXTRA_ARGS+=("--prometheus-port" "${PROMETHEUS_PORT}")
fi

exec torchrun \
  --nnodes="${NUM_NODES}" \
  --nproc_per_node="${GPUS_PER_NODE}" \
  --max_restarts="${MAX_RESTARTS}" \
  --rdzv_id="${RDZV_ID}" \
  --rdzv_backend="${RDZV_BACKEND}" \
  --rdzv_endpoint="${RDZV_ENDPOINT}" \
  --rdzv_conf="timeout=900" \
  train.py \
    --config "${CONFIG}" \
    "${EXTRA_ARGS[@]}"
