"""
tests/test_config.py
====================

Tests for pkg/utils/config.py — the Pydantic MoEConfig hierarchy.

Coverage targets
----------------
- All six sub-configs (ModelConfig, TrainingConfig, ParallelismConfig,
  CheckpointConfig, ElasticConfig, TelemetryConfig)
- Cross-field validators: top_k <= num_experts, warmup < max_steps,
  min_nodes <= max_nodes, hidden_dim % 8 == 0
- Bad-value validators: dtype, hidden_dim alignment
- From-file: smoke.yaml, default.yaml, FileNotFoundError
- From-dict: valid, invalid, missing nested keys
- Environment variable overrides: MOE_<SECTION>__<KEY>=value
- to_yaml / from_yaml round-trip (idempotent)
- Legacy load_config shim: .raw, .model dict, .typed()
- ConfigValidationError messages contain field path
- parallelism.world_size property

Every test is CPU-only and needs no GPU or dist.
"""

from __future__ import annotations

import os
import pathlib

import pytest

from pkg.utils.config import (
    ConfigValidationError,
    MoEConfig,
    ParallelismConfig,
    load_config,
)

pytestmark = pytest.mark.cpu

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SMOKE = pathlib.Path(__file__).parent.parent / "configs" / "smoke.yaml"
_DEFAULT = pathlib.Path(__file__).parent.parent / "configs" / "default.yaml"
_LARGE_SCALE = pathlib.Path(__file__).parent.parent / "configs" / "large_scale.yaml"


def _minimal_dict(**overrides) -> dict:
    """Return the smallest valid raw dict, optionally overriding model fields."""
    d = {
        "model": {
            "hidden_dim": 64,
            "num_layers": 2,
            "num_experts": 8,
            "top_k": 2,
            "capacity_factor": 1.25,
            "ffn_dim": 128,
            "vocab_size": 256,
            "sequence_length": 16,
            "dtype": "float32",
        },
        "training": {
            "global_batch_size": 8,
            "micro_batch_size": 2,
            "learning_rate": 3e-4,
            "weight_decay": 0.1,
            "grad_clip": 1.0,
            "max_steps": 10,
            "log_interval": 1,
            "ckpt_interval": 5,
            "warmup_steps": 2,
            "gradient_accumulation_steps": 1,
        },
        "parallelism": {
            "data_parallel": 1,
            "expert_parallel": 1,
            "tensor_parallel": 1,
            "pipeline_parallel": 1,
        },
        "checkpoint": {
            "local_dir": "/tmp/test",
            "remote_uri": "file:///tmp/remote",
            "async_workers": 1,
            "retention": 2,
        },
        "elastic": {
            "min_nodes": 1,
            "max_nodes": 4,
            "rdzv_backend": "c10d",
            "rdzv_endpoint": "localhost:29400",
            "health_check_interval_s": 1.0,
            "drop_grace_period_s": 5.0,
        },
        "telemetry": {
            "log_dir": "/tmp/logs",
            "tensorboard_dir": "/tmp/logs/tb",
            "json_path": "/tmp/logs/step.jsonl",
            "mfu_target": 0.55,
            "hardware_peak_tflops": 989.0,
        },
    }
    d["model"].update(overrides)
    return d


# ===========================================================================
# From-file loading
# ===========================================================================


class TestFromYaml:
    def test_smoke_yaml_loads(self):
        """smoke.yaml must load and have toy dimensions."""
        cfg = MoEConfig.from_yaml(_SMOKE)
        assert cfg.model.hidden_dim == 32
        assert cfg.model.num_experts == 4
        assert cfg.model.top_k == 2
        assert cfg.parallelism.world_size == 1

    def test_default_yaml_loads(self):
        """default.yaml must load and have production-scale dimensions."""
        cfg = MoEConfig.from_yaml(_DEFAULT)
        assert cfg.model.hidden_dim == 4096
        assert cfg.model.num_experts == 64
        assert cfg.model.dtype == "bfloat16"

    def test_large_scale_yaml_loads(self):
        """large_scale.yaml must load and demonstrate fine-grained MoE (P2.2)."""
        cfg = MoEConfig.from_yaml(_LARGE_SCALE)
        assert cfg.model.num_experts == 256
        assert cfg.model.top_k == 8
        assert cfg.model.capacity_dropping is True
        assert cfg.training.z_loss_weight > 0.0
        assert cfg.parallelism.world_size == 8 * 16  # dp=8, ep=16
        # top_k must still respect the top_k <= num_experts invariant
        assert cfg.model.top_k <= cfg.model.num_experts

    def test_file_not_found_raises(self):
        with pytest.raises(FileNotFoundError, match="not found"):
            MoEConfig.from_yaml("/nonexistent/path/config.yaml")

    def test_invalid_yaml_raises_config_error(self, tmp_path):
        bad = tmp_path / "bad.yaml"
        bad.write_text("- this is a list not a dict\n")
        with pytest.raises(ConfigValidationError, match="did not parse to a dict"):
            MoEConfig.from_yaml(bad)


# ===========================================================================
# From-dict loading and validation
# ===========================================================================


class TestFromDict:
    def test_valid_minimal_dict(self):
        cfg = MoEConfig.from_dict(_minimal_dict())
        assert cfg.model.hidden_dim == 64
        assert cfg.model.top_k == 2

    def test_top_k_gt_num_experts_raises(self):
        with pytest.raises(ConfigValidationError) as exc:
            MoEConfig.from_dict(_minimal_dict(top_k=16, num_experts=8))
        assert "top_k" in str(exc.value)
        assert "num_experts" in str(exc.value)

    def test_top_k_equals_num_experts_allowed(self):
        cfg = MoEConfig.from_dict(_minimal_dict(top_k=8, num_experts=8))
        assert cfg.model.top_k == 8

    def test_hidden_dim_not_multiple_of_8_raises(self):
        with pytest.raises(ConfigValidationError):
            MoEConfig.from_dict(_minimal_dict(hidden_dim=33))

    def test_hidden_dim_multiple_of_8_but_not_64_allowed(self):
        """32 is valid — smoke.yaml uses it."""
        cfg = MoEConfig.from_dict(_minimal_dict(hidden_dim=32))
        assert cfg.model.hidden_dim == 32

    def test_bad_dtype_raises(self):
        with pytest.raises(ConfigValidationError, match="dtype"):
            MoEConfig.from_dict(_minimal_dict(dtype="float8"))

    def test_all_valid_dtypes_accepted(self):
        for dtype in ("float32", "bfloat16", "float16"):
            cfg = MoEConfig.from_dict(_minimal_dict(dtype=dtype))
            assert cfg.model.dtype == dtype

    def test_warmup_ge_max_steps_raises(self):
        d = _minimal_dict()
        d["training"]["warmup_steps"] = 100
        d["training"]["max_steps"] = 50
        with pytest.raises(ConfigValidationError, match="warmup_steps"):
            MoEConfig.from_dict(d)

    def test_min_nodes_gt_max_nodes_raises(self):
        d = _minimal_dict()
        d["elastic"]["min_nodes"] = 10
        d["elastic"]["max_nodes"] = 5
        with pytest.raises(ConfigValidationError, match="min_nodes"):
            MoEConfig.from_dict(d)

    def test_bad_remote_uri_raises(self):
        d = _minimal_dict()
        d["checkpoint"]["remote_uri"] = "ftp://not-valid"
        with pytest.raises(ConfigValidationError, match="remote_uri"):
            MoEConfig.from_dict(d)

    def test_error_message_contains_field_path(self):
        """ConfigValidationError must name the offending field."""
        try:
            MoEConfig.from_dict(_minimal_dict(top_k=99, num_experts=4))
        except ConfigValidationError as e:
            msg = str(e)
            assert "top_k" in msg, f"Field path missing from error: {msg}"


# ===========================================================================
# Sub-config properties
# ===========================================================================


class TestParallelismConfig:
    def test_world_size_product(self):
        p = ParallelismConfig(
            data_parallel=4, expert_parallel=8, tensor_parallel=2, pipeline_parallel=2
        )
        assert p.world_size == 4 * 8 * 2 * 2

    def test_single_rank(self):
        p = ParallelismConfig(
            data_parallel=1, expert_parallel=1, tensor_parallel=1, pipeline_parallel=1
        )
        assert p.world_size == 1


# ===========================================================================
# Environment variable overrides
# ===========================================================================


class TestEnvOverrides:
    def test_hidden_dim_override(self):
        os.environ["MOE_MODEL__HIDDEN_DIM"] = "128"
        try:
            cfg = MoEConfig.from_yaml(_SMOKE)
            assert cfg.model.hidden_dim == 128, (
                f"Expected 128 from env override, got {cfg.model.hidden_dim}"
            )
        finally:
            del os.environ["MOE_MODEL__HIDDEN_DIM"]

    def test_learning_rate_override(self):
        os.environ["MOE_TRAINING__LEARNING_RATE"] = "1e-5"
        try:
            cfg = MoEConfig.from_yaml(_SMOKE)
            assert abs(cfg.training.learning_rate - 1e-5) < 1e-12
        finally:
            del os.environ["MOE_TRAINING__LEARNING_RATE"]

    def test_parallelism_override(self):
        os.environ["MOE_PARALLELISM__DATA_PARALLEL"] = "4"
        try:
            cfg = MoEConfig.from_yaml(_SMOKE)
            assert cfg.parallelism.data_parallel == 4
        finally:
            del os.environ["MOE_PARALLELISM__DATA_PARALLEL"]

    def test_unrelated_env_var_ignored(self):
        os.environ["SOME_OTHER_VAR"] = "999"
        try:
            cfg = MoEConfig.from_yaml(_SMOKE)
            assert cfg.model.hidden_dim == 32  # unchanged
        finally:
            del os.environ["SOME_OTHER_VAR"]

    def test_env_override_with_invalid_value_raises(self):
        """An env override that violates a constraint must raise ConfigValidationError."""
        os.environ["MOE_MODEL__TOP_K"] = "999"
        try:
            with pytest.raises(ConfigValidationError):
                MoEConfig.from_yaml(_SMOKE)
        finally:
            del os.environ["MOE_MODEL__TOP_K"]


# ===========================================================================
# Serialisation round-trip
# ===========================================================================


class TestSerialisationRoundTrip:
    def test_to_dict_from_dict_roundtrip(self):
        cfg = MoEConfig.from_yaml(_DEFAULT)
        d = cfg.to_dict()
        cfg2 = MoEConfig.from_dict(d)
        assert cfg2.model.hidden_dim == cfg.model.hidden_dim
        assert cfg2.model.num_experts == cfg.model.num_experts
        assert cfg2.parallelism.data_parallel == cfg.parallelism.data_parallel
        assert cfg2.training.learning_rate == pytest.approx(cfg.training.learning_rate)

    def test_to_yaml_from_yaml_roundtrip(self, tmp_path):
        cfg = MoEConfig.from_yaml(_DEFAULT)
        out = tmp_path / "roundtrip.yaml"
        cfg.to_yaml(out)
        cfg2 = MoEConfig.from_yaml(out)
        assert cfg2.model.hidden_dim == cfg.model.hidden_dim
        assert cfg2.model.dtype == cfg.model.dtype
        assert cfg2.checkpoint.remote_uri == cfg.checkpoint.remote_uri
        assert cfg2.telemetry.hardware_peak_tflops == pytest.approx(
            cfg.telemetry.hardware_peak_tflops
        )

    def test_to_yaml_creates_parent_dirs(self, tmp_path):
        cfg = MoEConfig.from_yaml(_SMOKE)
        nested = tmp_path / "deep" / "nested" / "config.yaml"
        cfg.to_yaml(nested)
        assert nested.exists()
        cfg2 = MoEConfig.from_yaml(nested)
        assert cfg2.model.hidden_dim == cfg.model.hidden_dim


# ===========================================================================
# Legacy load_config shim
# ===========================================================================


class TestLegacyShim:
    def test_raw_dict_access(self):
        legacy = load_config(_SMOKE)
        assert isinstance(legacy.raw, dict)
        assert legacy.raw["model"]["hidden_dim"] == 32

    def test_section_dict_access(self):
        legacy = load_config(_SMOKE)
        assert legacy.model["hidden_dim"] == 32
        assert legacy.training["learning_rate"] == pytest.approx(3e-4)
        assert legacy.parallelism["data_parallel"] == 1
        assert legacy.checkpoint["remote_uri"].startswith("file://")

    def test_typed_upgrade(self):
        legacy = load_config(_SMOKE)
        cfg = legacy.typed()
        assert isinstance(cfg, MoEConfig)
        assert cfg.model.hidden_dim == 32

    def test_raw_triggers_deprecation_warning(self):
        cfg = MoEConfig.from_yaml(_SMOKE)
        with pytest.warns(DeprecationWarning, match="deprecated"):
            _ = cfg.raw


# ===========================================================================
# Defaults are sensible for production
# ===========================================================================


class TestDefaultsAreSane:
    def test_default_yaml_capacity_factor_ge_1(self):
        cfg = MoEConfig.from_yaml(_DEFAULT)
        assert cfg.model.capacity_factor >= 1.0

    def test_default_yaml_mfu_target_in_range(self):
        cfg = MoEConfig.from_yaml(_DEFAULT)
        assert 0.0 < cfg.telemetry.mfu_target <= 1.0

    def test_default_yaml_h100_peak_tflops(self):
        cfg = MoEConfig.from_yaml(_DEFAULT)
        assert cfg.telemetry.hardware_peak_tflops == pytest.approx(989.0)

    def test_smoke_yaml_single_node(self):
        cfg = MoEConfig.from_yaml(_SMOKE)
        assert cfg.elastic.min_nodes == 1
        assert cfg.elastic.max_nodes == 1

    def test_large_scale_yaml_capacity_factor_ge_1(self):
        cfg = MoEConfig.from_yaml(_LARGE_SCALE)
        assert cfg.model.capacity_factor >= 1.0

    def test_large_scale_yaml_mfu_target_in_range(self):
        cfg = MoEConfig.from_yaml(_LARGE_SCALE)
        assert 0.0 < cfg.telemetry.mfu_target <= 1.0

    def test_large_scale_yaml_more_experts_than_default(self):
        """large_scale.yaml exists specifically to exercise E >> 64 (P2.2)."""
        default_cfg = MoEConfig.from_yaml(_DEFAULT)
        large_cfg = MoEConfig.from_yaml(_LARGE_SCALE)
        assert large_cfg.model.num_experts > default_cfg.model.num_experts

    def test_default_checkpoint_retention_positive(self):
        cfg = MoEConfig.from_yaml(_DEFAULT)
        assert cfg.checkpoint.retention >= 1
