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
from typing import Any, Dict

import yaml

try:
    from pydantic import BaseModel, Field, field_validator, model_validator
    from pydantic import ValidationError as _PydanticValidationError
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "moe-engine requires pydantic>=2.0.0 for its configuration system. "
        "Config validation is core functionality, not optional — a silently "
        "degraded validator is more dangerous than a hard import error, "
        "since it would let invalid configs (top_k > num_experts, bad dtypes, "
        "negative learning rates, etc.) reach training undetected.\n\n"
        "Install with:\n"
        "    pip install -e '.[dev]'\n"
        "or:\n"
        "    pip install pydantic>=2.0.0\n\n"
        f"Original error: {exc}"
    ) from exc

_HAS_PYDANTIC = True  # retained for any external code that still checks this flag


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
    capacity_dropping: bool = Field(
        False,
        description=(
            "Enable Switch Transformer / GShard-style capacity dropping. "
            "When True, each expert accepts at most "
            "ceil(capacity_factor * N*K/E) tokens; overflow tokens are "
            "dropped (zero combine weight for that slot). When False "
            "(default), all tokens are always processed regardless of "
            "load imbalance."
        ),
    )
    ffn_dim: int = Field(14336, description="Expert FFN intermediate dimension (F).")
    vocab_size: int = Field(128256, description="Vocabulary size.")
    sequence_length: int = Field(4096, description="Input sequence length (S).")
    dtype: str = Field("bfloat16", description="Training precision.")

    @model_validator(mode="after")
    def _cross_validate(self) -> "ModelConfig":
        if self.top_k > self.num_experts:
            raise ValueError(f"top_k ({self.top_k}) must be <= num_experts ({self.num_experts})")
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
            raise ValueError(f"dtype must be one of {valid_dtypes}, got {self.dtype!r}")
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
    z_loss_weight: float = Field(
        0.0,
        description=(
            "Auxiliary router z-loss weight (Switch Transformer style). "
            "0.0 = disabled. Typical value: 1e-3. "
            "Override via MOE_TRAINING__Z_LOSS_WEIGHT=1e-3."
        ),
    )

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

    @field_validator("remote_uri")
    @classmethod
    def _valid_uri(cls, v: str) -> str:
        if not (v.startswith("s3://") or v.startswith("file://")):
            raise ValueError(f"remote_uri must start with 's3://' or 'file://', got: {v!r}")
        return v


class ElasticConfig(BaseModel):
    """TorchElastic / fault-tolerance parameters."""

    min_nodes: int = Field(8)
    max_nodes: int = Field(256)
    rdzv_backend: str = Field("c10d")
    rdzv_endpoint: str = Field("etcd-headless:2379")
    health_check_interval_s: float = Field(5.0)
    drop_grace_period_s: float = Field(30.0)

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
            return cls(**d)
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
        return self.model_dump()

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

    Values are parsed with :func:`_coerce_env_value`, which tries native
    ``int``/``float`` coercion before falling back to ``yaml.safe_load`` for
    booleans, null, and other YAML scalars. This avoids a well-known PyYAML
    footgun: ``yaml.safe_load("1e-5")`` returns the **string** ``"1e-5"``,
    not the float ``1e-05``, because YAML 1.1's float grammar requires a
    decimal point or an explicit ``+``/``-`` exponent sign in certain
    positions (``yaml.safe_load("1.0e-5")`` works fine). Since
    ``MOE_TRAINING__LEARNING_RATE=1e-5`` is exactly the kind of value an
    operator would set on a command line, this footgun must not bite at the
    config layer.
    """
    import copy

    d = copy.deepcopy(d)
    prefix = "MOE_"
    for key, val in os.environ.items():
        if not key.startswith(prefix):
            continue
        rest = key[len(prefix) :].lower()
        if "__" not in rest:
            continue
        section, _, field_name = rest.partition("__")
        if section not in d or not isinstance(d[section], dict):
            continue
        d[section][field_name] = _coerce_env_value(val)
    return d


def _coerce_env_value(val: str) -> Any:
    """Coerce an environment-variable string into the most natural Python type.

    Tries, in order:
      1. ``int(val)``     — e.g. "16", "-4"
      2. ``float(val)``   — e.g. "1e-5", "3.14", "1.0e-5", "-2.5e3"
      3. ``yaml.safe_load(val)`` — booleans ("true"/"false"), "null", quoted
         strings, and any other YAML scalar.
      4. The raw string, unchanged, if all of the above fail.

    This is deliberately more permissive than raw ``yaml.safe_load`` because
    YAML 1.1's float grammar rejects exponential notation without a decimal
    point or explicit sign (``"1e-5"`` parses as a string under
    ``yaml.safe_load`` alone), which would silently break common
    command-line float overrides like ``MOE_TRAINING__LEARNING_RATE=1e-5``.
    """
    try:
        return int(val)
    except ValueError:
        pass
    try:
        return float(val)
    except ValueError:
        pass
    return yaml.safe_load(val)
