"""
moe-engine: A Composed Mixture-of-Experts Engine.

Package layout
--------------
pkg/
  distributed/    4D parallelism primitives (mesh, TP, EP, PP, DP, MoE layer)
  elastic/        Fault tolerance and async checkpointing
  kernels/        Triton-fused MoE router kernel
  models/         Model architecture definitions (decoupled from training)
  telemetry/      Structured logging, MFU accounting, WandB integration
  utils/          Config (Pydantic), MFU math, misc utilities

Quick start
-----------
    from pkg.utils.config import MoEConfig
    from pkg.distributed import build_topology, DistributedMoELayer
    from pkg.models import ToyMoEModel

    cfg      = MoEConfig.from_yaml("configs/smoke.yaml")
    topology = build_topology(dp_size=1, ep_size=1)
    model    = ToyMoEModel(cfg, topology)
"""

__version__ = "0.3.3"

__all__ = ["__version__"]
