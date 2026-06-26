"""
pkg/utils/config.py
===================

Strongly-typed, validated configuration system for moe-engine.

Replaces the old flat-dict approach with a Pydantic v2 model hierarchy.
Errors are caught at load time with clear, actionable messages rather than
surfacing as silent KeyErrors or shape mismatches mid-training.

Usage
-----
    from pkg.utils.config import MoEConfig

    cfg = MoEConfig.from_yaml("configs/default.yaml")
    cfg = MoEConfig.from_yaml("configs/smoke.yaml")

    # All fields are typed and validated:
    hidden_dim: int = cfg.model.hidden_dim
    lr: float       = cfg.training.learning_rate

    # Environment variable overrides (useful for cluster launches):
    #   MOE_TRAINING__LEARNING_RATE=1e-4 -> cfg.training.learning_rate == 1e-4

Public API
----------
    MoEConfig         - root config (composed of sub-configs below)
    ModelConfig       - model architecture parameters
    TrainingConfig    - optimiser, schedule, gradient settings
    ParallelismConfig - 4D parallelism sizing
    CheckpointConfig  - NVMe + S3 checkpointing parameters
    ElasticConfig     - TorchElastic / fault-tolerance parameters
    TelemetryConfig   - logging, tensorboard, WandB, MFU target

    MoEConfig.from_yaml(path)  - load + validate from YAML file
    MoEConfig.from_dict(d)     - load + validate from raw dict
    MoEConfig.to_yaml(path)    - serialise to YAML (round-trips cleanly)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Literal, Optional

import yaml

try:
    from pydantic import BaseModel, Field, field_validator, model_validator
    from pydantic import ValidationError as _PydanticValidationError
    _HAS_PYDANTIC = True
except ImportError:
    _HAS_PYDANTIC = False
    _PydanticValidationError = Exception  # type: ignore

    # Minimal shim so module imports without pydantic
    class BaseModel:  # type: ignore
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
        def model_dump(self):
            return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

    def Field(default=None, **kwargs):  # type: ignore
        return default

    def field_validator(*args, **kwargs):  # type: ignore
        def decorator(fn):
            return fn
        return decorator

    def model_validator(**kwargs):  # type: ignore
        def decorator(fn):
            return fn
        return decorator


__all__ = [
    "MoEConfig",
    "ModelConfig",
    "TrainingConfig",
    "ParallelismConfig",
    "CheckpointConfig",
    "ElasticConfig",
    "TelemetryConfig",
    "ConfigValidationError",
    "load_config",
]


class ConfigValidationError(ValueError):
    """Raised when a config file fails schema validation.

    The message includes the path to the offending field and a human-readable
    description of the constraint that was violated.
    """


# ===========================================================================
# Sub-config models
# ===========================================================================

class ModelConfig(BaseModel):
    """Architecture hyperparameters."""
    hidden_dim: int = Field(4096, description="Token embedding / hidden dimension (H).")
    num_layers: int = Field(32, description="Number of Transformer + MoE blocks.")
    num_experts: int = Field(64, description="Total number of experts (E).")
    top_k: int = Field(2, description="Active experts per token (K).")
    capacity_factor: float = Field(1.25, description="Expert buffer capacity factor.")
    ffn_dim: int = Field(14336, description="Expert FFN intermediate dimension (F).")
    vocab_size: int = Field(128256, description="Vocabulary size.")
    sequence_length: int = Field(4096, description="Input sequence length (S).")
    dtype: str = Field("bfloat16", description="Training precision.")

    if _HAS_PYDANTIC:
        @model_validator(mode="after")
        def _cross_validate(self) -> "ModelConfig":
            if self.top_k > self.num_experts:
                raise ValueError(
                    f"top_k ({self.top_k}) must be <= num_experts ({self.num_experts})"
                )
            # hidden_dim divisibility: recommend multiples of 64 for Triton efficiency,
            # but do NOT hard-error on smoke/test configs (hidden_dim=32 is valid for the
            # fp64 reference path; Triton kernel pads to the next power-of-2 block).
            if self.hidden_dim % 8 != 0:
                raise ValueError(
                    f"hidden_dim ({self.hidden_dim}) must be divisible by 8 "
                    "(minimum alignment for mixed-precision tensor operations)."
                )
            valid_dtypes = {"float32", "bfloat16", "float16"}
            if self.dtype not in valid_dtypes:
                raise ValueError(
                    f"dtype must be one of {valid_dtypes}, got {self.dtype!r}"
                )
            return self


class TrainingConfig(BaseModel):
    """Optimizer and learning-rate schedule parameters."""
    global_batch_size: int = Field(4096)
    micro_batch_size: int = Field(4)
    learning_rate: float = Field(3e-4)
    weight_decay: float = Field(0.1)
    grad_clip: float = Field(1.0)
    max_steps: int = Field(100_000)
    log_interval: int = Field(10)
    ckpt_interval: int = Field(500)
    warmup_steps: int = Field(2000)
    gradient_accumulation_steps: int = Field(4)

    if _HAS_PYDANTIC:
        @model_validator(mode="after")
        def _warmup_le_max_steps(self) -> "TrainingConfig":
            if self.warmup_steps >= self.max_steps:
                raise ValueError(
                    f"warmup_steps ({self.warmup_steps}) must be < max_steps ({self.max_steps})"
                )
            return self


class ParallelismConfig(BaseModel):
    """4D parallelism sizing."""
    data_parallel: int = Field(8, description="Data parallel degree (DP).")
    expert_parallel: int = Field(8, description="Expert parallel degree (EP).")
    tensor_parallel: int = Field(1, description="Tensor parallel degree (TP).")
    pipeline_parallel: int = Field(1, description="Pipeline parallel degree (PP).")

    @property
    def world_size(self) -> int:
        """Total GPU count implied by this config."""
        return (
            self.data_parallel
            * self.expert_parallel
            * self.tensor_parallel
            * self.pipeline_parallel
        )


class CheckpointConfig(BaseModel):
    """Two-tier async checkpointing config."""
    local_dir: str = Field("/mnt/nvme/moe-engine/ckpts")
    remote_uri: str = Field("s3://moe-engine-ckpts/run-001/")
    async_workers: int = Field(4)
    retention: int = Field(8)

    if _HAS_PYDANTIC:
        @field_validator("remote_uri")
        @classmethod
        def _valid_uri(cls, v: str) -> str:
            if not (v.startswith("s3://") or v.startswith("file://")):
                raise ValueError(
                    f"remote_uri must start with 's3://' or 'file://', got: {v!r}"
                )
            return v


class ElasticConfig(BaseModel):
    """TorchElastic / fault-tolerance parameters."""
    min_nodes: int = Field(8)
    max_nodes: int = Field(256)
    rdzv_backend: str = Field("c10d")
    rdzv_endpoint: str = Field("etcd-headless:2379")
    health_check_interval_s: float = Field(5.0)
    drop_grace_period_s: float = Field(30.0)

    if _HAS_PYDANTIC:
        @model_validator(mode="after")
        def _min_le_max(self) -> "ElasticConfig":
            if self.min_nodes > self.max_nodes:
                raise ValueError(
                    f"min_nodes ({self.min_nodes}) must be <= max_nodes ({self.max_nodes})"
                )
            return self


class TelemetryConfig(BaseModel):
    """Logging, metrics, and MFU accounting."""
    log_dir: str = Field("/var/log/moe-engine")
    tensorboard_dir: str = Field("/var/log/moe-engine/tb")
    json_path: str = Field("/var/log/moe-engine/step.jsonl")
    mfu_target: float = Field(0.55, description="Target MFU fraction (0.55 = 55% of peak).")
    hardware_peak_tflops: float = Field(
        989.0,
        description="Peak TFLOPS for MFU denominator. H100 SXM5 BF16 = 989 TFLOPS.",
    )


# ===========================================================================
# Root config
# ===========================================================================

class MoEConfig(BaseModel):
    """Root configuration for moe-engine.

    Composes all sub-configs. Instantiate via :meth:`from_yaml` or
    :meth:`from_dict` for validated loading with clear error messages.

    Example
    -------
    >>> cfg = MoEConfig.from_yaml("configs/smoke.yaml")
    >>> cfg.model.hidden_dim
    32
    >>> cfg.parallelism.world_size
    1
    """

    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    parallelism: ParallelismConfig = Field(default_factory=ParallelismConfig)
    checkpoint: CheckpointConfig = Field(default_factory=CheckpointConfig)
    elastic: ElasticConfig = Field(default_factory=ElasticConfig)
    telemetry: TelemetryConfig = Field(default_factory=TelemetryConfig)

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str | Path) -> "MoEConfig":
        """Load and validate a config from a YAML file.

        Raises
        ------
        FileNotFoundError
            If *path* does not exist.
        ConfigValidationError
            If the YAML content fails schema validation, with a detailed
            message identifying the offending field(s).
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(
                f"Config file not found: {p.resolve()}\n"
                "Check the path or run from the moe-engine/ directory."
            )
        with p.open("r") as f:
            raw = yaml.safe_load(f)
        if not isinstance(raw, dict):
            raise ConfigValidationError(
                f"Config file {p} did not parse to a dict (got {type(raw).__name__}). "
                "Is the YAML valid?"
            )
        return cls.from_dict(raw, source=str(p))

    @classmethod
    def from_dict(cls, d: Dict[str, Any], source: str = "<dict>") -> "MoEConfig":
        """Load and validate from a raw dictionary.

        Also applies environment variable overrides of the form::

            MOE_TRAINING__LEARNING_RATE=1e-4
            MOE_MODEL__HIDDEN_DIM=1024
            MOE_PARALLELISM__DATA_PARALLEL=16
        """
        d = _apply_env_overrides(d)
        try:
            if _HAS_PYDANTIC:
                return cls(**d)
            else:
                # Minimal non-pydantic path: construct sub-objects manually
                return cls(
                    model=ModelConfig(**d.get("model", {})),
                    training=TrainingConfig(**d.get("training", {})),
                    parallelism=ParallelismConfig(**d.get("parallelism", {})),
                    checkpoint=CheckpointConfig(**d.get("checkpoint", {})),
                    elastic=ElasticConfig(**d.get("elastic", {})),
                    telemetry=TelemetryConfig(**d.get("telemetry", {})),
                )
        except _PydanticValidationError as exc:
            lines = [f"Config validation failed (source: {source}):"]
            for err in exc.errors():
                loc = " -> ".join(str(x) for x in err["loc"])
                lines.append(f"  [{loc}]  {err['msg']}")
                if "input" in err:
                    lines.append(f"    got: {err['input']!r}")
            raise ConfigValidationError("\n".join(lines)) from exc
        except (TypeError, KeyError) as exc:
            raise ConfigValidationError(
                f"Unexpected error loading config from {source}: {exc}\n"
                "Check that all required fields are present in the YAML."
            ) from exc

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_yaml(self, path: str | Path) -> None:
        """Serialise config to YAML. Round-trips cleanly through from_yaml."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w") as f:
            yaml.safe_dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    def to_dict(self) -> Dict[str, Any]:
        """Return a plain-dict representation suitable for JSON/YAML."""
        if _HAS_PYDANTIC:
            return self.model_dump()
        return {
            "model": self.model.__dict__,
            "training": self.training.__dict__,
            "parallelism": self.parallelism.__dict__,
            "checkpoint": self.checkpoint.__dict__,
            "elastic": self.elastic.__dict__,
            "telemetry": self.telemetry.__dict__,
        }

    # ------------------------------------------------------------------
    # Backward-compatibility shim
    # ------------------------------------------------------------------

    @property
    def raw(self) -> Dict[str, Any]:
        """Return raw dict for code still using cfg.raw["model"]["hidden_dim"].

        .. deprecated::
           Access fields directly: ``cfg.model.hidden_dim``
        """
        import warnings
        warnings.warn(
            "MoEConfig.raw is deprecated; access fields directly: cfg.model.hidden_dim",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.to_dict()


# ===========================================================================
# Legacy load_config shim - keeps old train.py call sites working
# ===========================================================================

class _LegacyConfig:
    """Wrapper that makes MoEConfig look like the old Config dataclass.

    Old: cfg = load_config(path); cfg.raw["model"]["hidden_dim"]
    New: cfg = MoEConfig.from_yaml(path); cfg.model.hidden_dim
    """
    def __init__(self, moe_cfg: MoEConfig):
        self._cfg = moe_cfg
        d = moe_cfg.to_dict()
        self.model = d["model"]
        self.training = d["training"]
        self.parallelism = d["parallelism"]
        self.checkpoint = d["checkpoint"]
        self.elastic = d["elastic"]
        self.telemetry = d["telemetry"]
        self.raw = d

    def typed(self) -> MoEConfig:
        """Upgrade to the modern typed config."""
        return self._cfg


def load_config(path: str | Path) -> _LegacyConfig:
    """Load config — legacy entry point preserved for backward compatibility.

    New callers should use ``MoEConfig.from_yaml(path)`` directly.
    """
    return _LegacyConfig(MoEConfig.from_yaml(path))


# ===========================================================================
# Environment variable override helper
# ===========================================================================

def _apply_env_overrides(d: Dict[str, Any]) -> Dict[str, Any]:
    """Apply ``MOE_<SECTION>__<KEY>=value`` overrides from the environment.

    Examples::

        MOE_TRAINING__LEARNING_RATE=1e-4   -> d["training"]["learning_rate"] = 1e-4
        MOE_MODEL__HIDDEN_DIM=1024         -> d["model"]["hidden_dim"] = 1024
        MOE_PARALLELISM__DATA_PARALLEL=16  -> d["parallelism"]["data_parallel"] = 16

    Values are parsed with ``yaml.safe_load`` so booleans, ints, and floats
    are converted from their string representations correctly.
    """
    import copy
    d = copy.deepcopy(d)
    prefix = "MOE_"
    for key, val in os.environ.items():
        if not key.startswith(prefix):
            continue
        rest = key[len(prefix):].lower()
        if "__" not in rest:
            continue
        section, _, field_name = rest.partition("__")
        if section not in d or not isinstance(d[section], dict):
            continue
        d[section][field_name] = yaml.safe_load(val)
    return d
