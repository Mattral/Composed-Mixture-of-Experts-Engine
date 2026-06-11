# ============================================================
# moe-engine — multi-stage Docker image
# ============================================================
# Stage 1: build deps
# Stage 2: minimal runtime
#
# Build:
#   docker build -f deploy/docker/Dockerfile -t moe-engine:v0.2 .
#
# Run single-GPU smoke test:
#   docker run --rm --gpus all moe-engine:v0.2 \
#       python train.py --config configs/smoke.yaml --smoke
#
# Run 8-GPU torchrun:
#   docker run --rm --gpus all --ipc=host moe-engine:v0.2 \
#       torchrun --standalone --nproc_per_node=8 \
#       train.py --config configs/default.yaml
# ============================================================

ARG PYTORCH_VERSION=2.5.1
ARG CUDA_VERSION=12.4.1
ARG PYTHON_VERSION=3.11

# -------- builder --------------------------------------------------
FROM pytorch/pytorch:${PYTORCH_VERSION}-cuda${CUDA_VERSION}-cudnn9-devel AS builder

WORKDIR /build

# System deps: git for any source installs, build-essential for Triton
RUN apt-get update && apt-get install -y --no-install-recommends \
        git build-essential curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt \
 && pip install --no-cache-dir triton>=3.0.0

# -------- runtime --------------------------------------------------
FROM pytorch/pytorch:${PYTORCH_VERSION}-cuda${CUDA_VERSION}-cudnn9-runtime AS runtime

WORKDIR /workspace/moe-engine

# Copy installed Python packages from builder
COPY --from=builder /opt/conda/lib/python3.11/site-packages \
                    /opt/conda/lib/python3.11/site-packages

# Copy source
COPY . .

# Environment — NCCL tuning for H100/A100 NVLink clusters
ENV TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
    TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=30 \
    TORCH_NCCL_TRACE_BUFFER_SIZE=1048576 \
    GLOO_SOCKET_IFNAME=eth0 \
    NCCL_IB_DISABLE=0 \
    NCCL_NET_GDR_LEVEL=5 \
    OMP_NUM_THREADS=8 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Health-check: import the package
HEALTHCHECK --interval=60s --timeout=10s --retries=3 \
    CMD python -c "import pkg.kernels.moe_router; print('OK')"

# Default: run the smoke test
CMD ["python", "train.py", "--config", "configs/smoke.yaml", "--smoke"]
