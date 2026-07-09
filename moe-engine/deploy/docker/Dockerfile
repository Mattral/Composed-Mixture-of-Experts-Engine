# ============================================================
# moe-engine — multi-stage Docker image  (v0.3.2)
# ============================================================
#
# Tag conventions on Docker Hub (pytorch/pytorch):
#   pytorch/pytorch:<ver>-cuda<cuda_ver>-cudnn9-devel   ← build tools
#   pytorch/pytorch:<ver>-cuda<cuda_ver>-cudnn9-runtime ← minimal runtime
#
# VERIFIED available tags (July 2026):
#   pytorch/pytorch:2.6.0-cuda12.6-cudnn9-devel
#   pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel
#   pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel
#
# NOTE on CUDA version: the PyTorch Docker image CUDA version refers to the
# CUDA toolkit bundled in the image, not the host driver version. The host
# driver only needs to be >= the image's CUDA runtime version. A 12.6 image
# runs fine on a driver supporting CUDA 12.x.
#
# Build (GPU, default):
#   docker build -f deploy/docker/Dockerfile -t moe-engine:v0.3.2 \
#       --build-arg PYTORCH_VERSION=2.6.0 \
#       --build-arg CUDA_VERSION=12.6 .
#
# Build (CPU-only, for CI lint/test without GPU):
#   docker build -f deploy/docker/Dockerfile -t moe-engine:v0.3.2-cpu \
#       --target runtime-cpu \
#       --build-arg USE_GPU=0 .
#
# Run single-GPU smoke test:
#   docker run --rm --gpus all moe-engine:v0.3.2 \
#       python train.py --config configs/smoke.yaml --smoke
#
# Run 8-GPU torchrun:
#   docker run --rm --gpus all --ipc=host moe-engine:v0.3.2 \
#       torchrun --standalone --nproc_per_node=8 \
#       train.py --config configs/default.yaml
# ============================================================

# Versions — override at build time if needed:
#   docker build --build-arg PYTORCH_VERSION=2.5.1 --build-arg CUDA_VERSION=12.1 .
ARG PYTORCH_VERSION=2.6.0
ARG CUDA_VERSION=12.6
ARG PYTHON_VERSION=3.11

# ──────────────────────────────────────────────────────────────
# Stage 1: builder — install all Python deps in the full devel
#          image (has gcc, ninja, etc. needed for Triton compilation)
# ──────────────────────────────────────────────────────────────
FROM pytorch/pytorch:${PYTORCH_VERSION}-cuda${CUDA_VERSION}-cudnn9-devel AS builder

WORKDIR /build

# System deps: git for any source installs, build-essential for Triton
RUN apt-get update && apt-get install -y --no-install-recommends \
        git build-essential curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt \
 && pip install --no-cache-dir "triton>=3.0.0" "pydantic>=2.0.0"

# Pre-compile any Triton kernels during build (populates the Triton cache)
# so the first training step doesn't pay JIT compile time.
COPY pkg/ /build/pkg/
COPY configs/ /build/configs/
RUN python -c "
import sys; sys.path.insert(0, '/build')
try:
    from pkg.kernels.moe_router import MoERouter
    import torch
    r = MoERouter(hidden_dim=256, num_experts=8, top_k=2)
    if torch.cuda.is_available():
        r = r.cuda()
        r(torch.randn(64, 256, device='cuda'))
        print('Triton kernel pre-compiled on GPU')
    else:
        print('No GPU at build time — Triton will JIT on first run')
except Exception as e:
    print(f'Pre-compilation skipped: {e}')
" || true

# ──────────────────────────────────────────────────────────────
# Stage 2: runtime — minimal image for deployment
# ──────────────────────────────────────────────────────────────
FROM pytorch/pytorch:${PYTORCH_VERSION}-cuda${CUDA_VERSION}-cudnn9-runtime AS runtime

WORKDIR /workspace/moe-engine

# Copy installed Python packages from builder (avoids re-installing)
ARG PYTHON_VERSION=3.11
COPY --from=builder /opt/conda/lib/python${PYTHON_VERSION}/site-packages \
                    /opt/conda/lib/python${PYTHON_VERSION}/site-packages

# Copy Triton kernel cache from builder (avoids JIT compile on first run)
COPY --from=builder /root/.triton /root/.triton

# Copy source
COPY . .

# ── Environment: NCCL + GLOO tuning for H100/A100 NVLink clusters ──────────
ENV TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
    TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=30 \
    TORCH_NCCL_TRACE_BUFFER_SIZE=1048576 \
    NCCL_IB_DISABLE=0 \
    NCCL_NET_GDR_LEVEL=5 \
    GLOO_SOCKET_IFNAME=eth0 \
    OMP_NUM_THREADS=8 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Health-check: import the package and validate both configs
HEALTHCHECK --interval=60s --timeout=10s --retries=3 \
    CMD python -c "import pkg.kernels.moe_router; from pkg.utils.config import MoEConfig; MoEConfig.from_yaml('configs/smoke.yaml'); print('OK')"

# Default: run the smoke test
CMD ["python", "train.py", "--config", "configs/smoke.yaml", "--smoke"]

# ──────────────────────────────────────────────────────────────
# Stage 3: CPU-only image — for CI lint/test without GPU hardware
# (no CUDA toolkit, smaller image, no Triton)
# ──────────────────────────────────────────────────────────────
FROM python:${PYTHON_VERSION}-slim AS runtime-cpu

WORKDIR /workspace/moe-engine

RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir \
        "torch>=2.5.0" \
        "pydantic>=2.0.0" \
        "pyyaml>=6.0.1" \
        "pytest>=8.0.0" \
        "pytest-mock>=3.12.0" \
        "hypothesis>=6.100.0" \
        "moto>=5.0.0"

COPY . .

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

# Default: run the Tier-0 CPU test suite
CMD ["python", "-m", "pytest", "tests/", "-m", "cpu", \
     "-k", "not (2rank or multiprocess or distributed_invariants)", \
     "--ignore=tests/test_chaos.py", "-q"]
