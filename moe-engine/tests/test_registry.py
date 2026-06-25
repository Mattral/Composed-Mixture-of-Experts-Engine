"""
tests/test_registry.py
=======================

Tests for ``pkg/models/registry.py`` — model registry and factory pattern.

Coverage:
- Built-in ``toy_moe`` is always registered
- ``@register_model`` decorator works and is transparent (returns class)
- Duplicate registration raises with informative message
- ``build_model_from_config`` dispatches to correct class
- Unknown arch raises ``KeyError`` listing available names
- ``list_registered_models`` returns sorted names
- ``ModelRegistry`` class-level API is consistent with module-level functions
- Registered model is callable with ``(cfg, topology)`` signature
- Model is moved to correct device on construction
- Custom models can override ``toy_moe`` via explicit ``arch`` argument
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

import pkg.models.registry as _registry_module
from pkg.distributed.mesh import build_topology
from pkg.models.registry import (
    ModelRegistry,
    build_model_from_config,
    list_registered_models,
    register_model,
)
from pkg.utils.config import MoEConfig

pytestmark = pytest.mark.cpu


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def smoke_cfg() -> MoEConfig:
    return MoEConfig.from_yaml("configs/smoke.yaml")


@pytest.fixture
def cpu_topology():
    return build_topology(dp_size=1, ep_size=1)


# ===========================================================================
# Built-in registration
# ===========================================================================


class TestBuiltinRegistration:
    def test_toy_moe_always_registered(self):
        assert "toy_moe" in list_registered_models(), (
            "'toy_moe' must always be registered after `import pkg.models`"
        )

    def test_list_returns_sorted(self):
        names = list_registered_models()
        assert names == sorted(names), "list_registered_models() must return sorted list"

    def test_list_returns_list_type(self):
        assert isinstance(list_registered_models(), list)

    def test_registry_class_list_matches_module_fn(self):
        assert ModelRegistry.list() == list_registered_models()


# ===========================================================================
# @register_model decorator
# ===========================================================================


class TestRegisterDecorator:
    def test_decorator_is_transparent(self):
        """Decorated class must be unchanged (same type, same methods)."""

        @register_model("__test_transparent_cls__")
        class MyModel(nn.Module):
            def __init__(self, cfg, topology):
                super().__init__()

            def forward(self, x):
                return x

        assert MyModel.__name__ == "MyModel"
        assert issubclass(MyModel, nn.Module)
        _registry_module._REGISTRY.pop("__test_transparent_cls__", None)  # cleanup

    def test_decorator_registers_name(self, smoke_cfg, cpu_topology):
        """Registered name must appear in list immediately after decoration."""

        @register_model("__test_register_name__")
        class MinModel(nn.Module):
            def __init__(self, cfg, topology):
                super().__init__()
                self.linear = nn.Linear(cfg.model.hidden_dim, 1)

            def forward(self, x):
                return self.linear(x)

        assert "__test_register_name__" in list_registered_models()
        _registry_module._REGISTRY.pop("__test_register_name__", None)

    def test_duplicate_name_raises_value_error(self):
        """Registering the same name twice must raise ValueError."""

        @register_model("__test_duplicate__")
        class First(nn.Module):
            def __init__(self, cfg, topology):
                super().__init__()

        with pytest.raises(ValueError, match="already registered"):

            @register_model("__test_duplicate__")
            class Second(nn.Module):
                def __init__(self, cfg, topology):
                    super().__init__()

        _registry_module._REGISTRY.pop("__test_duplicate__", None)

    def test_error_message_mentions_existing_class(self):
        """Duplicate error message must name the class that holds the slot."""

        @register_model("__test_dup_msg__")
        class FirstHolder(nn.Module):
            def __init__(self, cfg, topology):
                super().__init__()

        try:

            @register_model("__test_dup_msg__")
            class Second(nn.Module):
                def __init__(self, cfg, topology):
                    super().__init__()
        except ValueError as exc:
            assert "FirstHolder" in str(exc)
        finally:
            _registry_module._REGISTRY.pop("__test_dup_msg__", None)


# ===========================================================================
# build_model_from_config
# ===========================================================================


class TestBuildModelFromConfig:
    def test_builds_toy_moe_by_default(self, smoke_cfg, cpu_topology):
        from pkg.models.moe import ToyMoEModel

        model = build_model_from_config(smoke_cfg, cpu_topology)
        assert isinstance(model, ToyMoEModel)

    def test_explicit_arch_overrides_default(self, smoke_cfg, cpu_topology):
        """Passing arch='toy_moe' explicitly must also work."""
        from pkg.models.moe import ToyMoEModel

        model = build_model_from_config(smoke_cfg, cpu_topology, arch="toy_moe")
        assert isinstance(model, ToyMoEModel)

    def test_model_on_correct_device(self, smoke_cfg, cpu_topology):
        model = build_model_from_config(smoke_cfg, cpu_topology)
        for param in model.parameters():
            assert param.device == cpu_topology.device, (
                f"Param on {param.device}, expected {cpu_topology.device}"
            )

    def test_unknown_arch_raises_key_error(self, smoke_cfg, cpu_topology):
        with pytest.raises(KeyError):
            build_model_from_config(smoke_cfg, cpu_topology, arch="nonexistent_xyz")

    def test_key_error_lists_available_names(self, smoke_cfg, cpu_topology):
        try:
            build_model_from_config(smoke_cfg, cpu_topology, arch="bad_arch")
        except KeyError as exc:
            assert "toy_moe" in str(exc), "Error message must list available registered models"

    def test_custom_model_dispatched_correctly(self, smoke_cfg, cpu_topology):
        """Registering a custom model and building it via arch name."""

        class TinyLinear(nn.Module):
            def __init__(self, cfg, topology):
                super().__init__()
                self.linear = nn.Linear(cfg.model.hidden_dim, cfg.model.vocab_size)

            def forward(self, x):
                return self.linear(x.float())

        ModelRegistry.register("__test_custom_dispatch__", TinyLinear)
        try:
            model = build_model_from_config(
                smoke_cfg, cpu_topology, arch="__test_custom_dispatch__"
            )
            assert isinstance(model, TinyLinear)
        finally:
            _registry_module._REGISTRY.pop("__test_custom_dispatch__", None)

    def test_built_model_forward_runs(self, smoke_cfg, cpu_topology):
        model = build_model_from_config(smoke_cfg, cpu_topology)
        ids = torch.randint(0, smoke_cfg.model.vocab_size, (2, smoke_cfg.model.sequence_length))
        out = model(ids)
        assert out.shape == (2, smoke_cfg.model.sequence_length, smoke_cfg.model.vocab_size)

    def test_built_model_backward_runs(self, smoke_cfg, cpu_topology):
        model = build_model_from_config(smoke_cfg, cpu_topology)
        ids = torch.randint(0, smoke_cfg.model.vocab_size, (1, smoke_cfg.model.sequence_length))
        loss = model(ids).sum()
        loss.backward()
        # Verify at least one parameter received a gradient
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert len(grads) > 0, "No parameters received gradients"


# ===========================================================================
# ModelRegistry class-level API
# ===========================================================================


class TestModelRegistryClass:
    def test_get_returns_class(self):
        cls = ModelRegistry.get("toy_moe")
        assert isinstance(cls, type)
        assert issubclass(cls, nn.Module)

    def test_get_unknown_raises_key_error(self):
        with pytest.raises(KeyError):
            ModelRegistry.get("__definitely_not_registered__")

    def test_register_and_get_round_trip(self, smoke_cfg, cpu_topology):
        class RoundTripModel(nn.Module):
            def __init__(self, cfg, topology):
                super().__init__()

            def forward(self, x):
                return x

        ModelRegistry.register("__round_trip__", RoundTripModel)
        try:
            retrieved = ModelRegistry.get("__round_trip__")
            assert retrieved is RoundTripModel
        finally:
            _registry_module._REGISTRY.pop("__round_trip__", None)

    def test_list_includes_newly_registered(self):
        class NewModel(nn.Module):
            def __init__(self, cfg, topology):
                super().__init__()

        ModelRegistry.register("__new_model_list_test__", NewModel)
        try:
            assert "__new_model_list_test__" in ModelRegistry.list()
        finally:
            _registry_module._REGISTRY.pop("__new_model_list_test__", None)
