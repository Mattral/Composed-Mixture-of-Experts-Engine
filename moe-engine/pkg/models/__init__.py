"""
pkg/models
==========

Model architecture definitions, decoupled from the training loop.

Having model code here means the same definitions can be used for:
  - Training (via train.py)
  - Evaluation / inference (without pulling in the training harness)
  - Model analysis and parameter counting
  - Export / conversion scripts

Submodules
----------
moe.py  — RMSNorm, ToyMoEBlock, ToyMoEModel
"""

from pkg.models.moe import RMSNorm, ToyMoEBlock, ToyMoEModel
from pkg.models.registry import (
    ModelRegistry,
    build_model_from_config,
    list_registered_models,
    register_model,
)

__all__ = [
    "RMSNorm",
    "ToyMoEBlock",
    "ToyMoEModel",
    "register_model",
    "build_model_from_config",
    "list_registered_models",
    "ModelRegistry",
]
