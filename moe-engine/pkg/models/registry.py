"""
pkg/models/registry.py
=======================

Model registry and factory pattern for moe-engine.

Provides a lightweight decorator-based registry so new model architectures
can be added without modifying ``train.py`` or other call sites.

How frontier labs do it
-----------------------
Production training frameworks (Megatron-LM, torchTitan, MegaScale) use a
registry pattern so the config file selects the model class by name. This
decouples the training loop from specific model implementations — the same
``train.py`` can train any registered model by setting ``model.arch`` in the
YAML config.

Usage
-----
Registering a model::

    from pkg.models.registry import register_model

    @register_model("toy_moe")
    class ToyMoEModel(nn.Module):
        def __init__(self, cfg, topology): ...

Building from config::

    from pkg.models.registry import build_model_from_config
    from pkg.utils.config import MoEConfig
    from pkg.distributed.mesh import build_topology

    cfg = MoEConfig.from_yaml("configs/smoke.yaml")
    topo = build_topology(dp_size=1, ep_size=1)
    model = build_model_from_config(cfg, topo)   # dispatches to registered class

Public API
----------
    register_model(name)          — decorator to register a model class
    build_model_from_config(cfg, topology)  — factory that uses cfg.model.arch
    list_registered_models()      — returns all registered model names
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, Optional, Type

import torch.nn as nn

from pkg.utils.config import MoEConfig

__all__ = [
    "register_model",
    "build_model_from_config",
    "list_registered_models",
    "ModelRegistry",
]

log = logging.getLogger(__name__)

# Global registry: name → class
_REGISTRY: Dict[str, Type[nn.Module]] = {}


class ModelRegistry:
    """Class-level interface to the model registry.

    Prefer the module-level functions ``register_model``,
    ``build_model_from_config``, and ``list_registered_models`` for typical use.
    This class is provided for programmatic access (e.g. testing or tooling).
    """

    @staticmethod
    def register(name: str, cls: Type[nn.Module]) -> None:
        """Register ``cls`` under ``name``. Raises if name already taken."""
        if name in _REGISTRY:
            raise ValueError(
                f"Model '{name}' is already registered as {_REGISTRY[name].__qualname__}. "
                "Use a unique name or explicitly replace the registration."
            )
        _REGISTRY[name] = cls
        log.debug("Registered model '%s' → %s", name, cls.__qualname__)

    @staticmethod
    def get(name: str) -> Type[nn.Module]:
        """Return the class registered under ``name``.

        Raises
        ------
        KeyError
            With a clear message listing all registered names.
        """
        if name not in _REGISTRY:
            available = ", ".join(f"'{k}'" for k in sorted(_REGISTRY))
            raise KeyError(
                f"No model registered under name '{name}'. "
                f"Available: [{available}]. "
                "Register your model with @register_model('name') before building."
            )
        return _REGISTRY[name]

    @staticmethod
    def list() -> list[str]:
        """Return sorted list of all registered model names."""
        return sorted(_REGISTRY.keys())


def register_model(name: str) -> Callable[[Type[nn.Module]], Type[nn.Module]]:
    """Decorator to register a model class under a given name.

    The class must accept ``(cfg: MoEConfig, topology: ParallelTopology)``
    as its first two constructor arguments.

    Parameters
    ----------
    name : str
        Registry key. Must be unique. Use ``snake_case``.

    Returns
    -------
    The class unchanged (decorator is transparent).

    Example
    -------
    ::

        @register_model("my_moe_v2")
        class MyMoEv2(nn.Module):
            def __init__(self, cfg: MoEConfig, topology: ParallelTopology):
                super().__init__()
                ...
    """
    def decorator(cls: Type[nn.Module]) -> Type[nn.Module]:
        ModelRegistry.register(name, cls)
        return cls
    return decorator


def build_model_from_config(
    cfg: MoEConfig,
    topology: "ParallelTopology",  # forward ref to avoid circular import
    arch: Optional[str] = None,
) -> nn.Module:
    """Instantiate and return a model from the registry.

    Dispatches to the class registered under ``arch`` (or ``cfg.model.arch``
    if the config has that field, falling back to ``"toy_moe"``).

    The model is moved to ``topology.device`` before returning.

    Parameters
    ----------
    cfg : MoEConfig
        Typed, validated config.
    topology : ParallelTopology
        Distributed topology (device, mesh, rank coordinates).
    arch : str, optional
        Override the architecture name. If None, uses ``cfg.model.arch``
        if that attribute exists, else defaults to ``"toy_moe"``.

    Returns
    -------
    nn.Module  on ``topology.device``

    Raises
    ------
    KeyError
        If the architecture name is not registered.
    """
    if arch is None:
        arch = getattr(cfg.model, "arch", "toy_moe")

    cls = ModelRegistry.get(arch)
    log.info("Building model '%s' (%s) on %s", arch, cls.__qualname__, topology.device)
    model = cls(cfg, topology)
    model = model.to(topology.device)
    return model


def list_registered_models() -> list[str]:
    """Return the names of all registered models."""
    return ModelRegistry.list()


# ---------------------------------------------------------------------------
# Register built-in models on import
# ---------------------------------------------------------------------------
# This import triggers the @register_model("toy_moe") decorator in moe.py.
# Keep this at the bottom to avoid circular imports.
def _register_builtins() -> None:
    """Register all built-in model classes. Called once on module import."""
    try:
        import pkg.models.moe as _moe_module  # noqa: F401  (side-effect: registration)
        log.debug("Built-in models registered: %s", ModelRegistry.list())
    except ImportError as exc:
        log.warning("Could not register built-in models: %s", exc)


_register_builtins()
