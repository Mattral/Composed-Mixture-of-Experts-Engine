#!/usr/bin/env python
"""
scripts/cli.py
==============

moe-engine command-line interface.

Commands
--------
    moe train      --config <yaml>  [options]   Launch training
    moe benchmark  [options]                     Run benchmark sweep
    moe validate   <config_path>...              Validate YAML configs
    moe info                                     Print environment info

Install the CLI into your env:

    pip install -e ".[dev]"
    # Then: moe --help

Or run directly:

    python scripts/cli.py train --config configs/smoke.yaml --smoke

P0.3 requirement from MOE_instructions v2.1:
  "Add a proper CLI (using typer or click) for common operations."
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running from the repo root without installing
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

try:
    import typer

    _HAS_TYPER = True
except ImportError:
    _HAS_TYPER = False


# ---------------------------------------------------------------------------
# If typer is not installed, emit a clear install message and exit.
# ---------------------------------------------------------------------------
if not _HAS_TYPER:
    print(
        "ERROR: typer is required for the moe-engine CLI.\n"
        "Install it with:\n"
        "    pip install typer\n"
        "or:\n"
        "    pip install -e '.[dev]'\n",
        file=sys.stderr,
    )
    sys.exit(1)


import subprocess  # noqa: E402
from typing import Optional  # noqa: E402

app = typer.Typer(
    name="moe",
    help=(
        "moe-engine CLI — fault-tolerant MoE training at hyperscale.\n\n"
        "Run 'moe COMMAND --help' for command-specific options."
    ),
    add_completion=False,
    rich_markup_mode="markdown",
)


# ===========================================================================
# moe train
# ===========================================================================


@app.command("train")
def train(
    config: Path = typer.Option(
        ...,
        "--config",
        "-c",
        help="Path to YAML config file (e.g. configs/smoke.yaml).",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    max_steps: Optional[int] = typer.Option(
        None,
        "--max-steps",
        help="Override max_steps from config.",
    ),
    smoke: bool = typer.Option(
        False,
        "--smoke",
        help="Minimal smoke run: toy model, 5 steps, no GPU required.",
    ),
    profile: bool = typer.Option(
        False,
        "--profile",
        help="Write a benchmark JSON to benchmarks/ on exit.",
    ),
    wandb_project: Optional[str] = typer.Option(
        None,
        "--wandb-project",
        help="WandB project name (requires WANDB_API_KEY env var).",
    ),
    no_wandb: bool = typer.Option(
        False,
        "--no-wandb",
        help="Disable WandB logging even if WANDB_API_KEY is set.",
    ),
    nproc: int = typer.Option(
        1,
        "--nproc",
        help="Number of processes (uses torchrun when nproc > 1).",
    ),
) -> None:
    """Launch moe-engine training.

    Single-process:

        moe train --config configs/smoke.yaml --smoke

    Multi-GPU (4 processes):

        moe train --config configs/default.yaml --nproc 4

    For large multi-node jobs, use torchrun directly with your cluster's
    rdzv endpoint (the CLI's --nproc > 1 path uses --standalone mode for
    local multi-GPU only).
    """
    # Validate config before launching anything
    # Validate config before launching
    from pkg.utils.config import ConfigValidationError, MoEConfig

    try:
        MoEConfig.from_yaml(config)
        typer.echo(f"Config {config} validated OK.")
    except (ConfigValidationError, FileNotFoundError) as exc:
        typer.echo(f"Config validation failed: {exc}", err=True)
        raise typer.Exit(1)

    cmd: list[str] = []

    if nproc > 1:
        cmd += [
            "torchrun",
            "--standalone",
            f"--nproc_per_node={nproc}",
        ]

    cmd += [
        sys.executable,
        str(_ROOT / "train.py"),
        "--config",
        str(config),
    ]

    if max_steps is not None:
        cmd += ["--max-steps", str(max_steps)]
    if smoke:
        cmd.append("--smoke")
    if profile:
        cmd.append("--profile")
    if wandb_project:
        cmd += ["--wandb-project", wandb_project]
    if no_wandb:
        cmd.append("--no-wandb")

    # For single-process, replace the process entirely (no subprocess overhead).
    if nproc == 1:
        import train as _train_module  # noqa: F401

        sys.argv = ["train.py"] + cmd[cmd.index(str(_ROOT / "train.py")) + 1 :]
        _train_module.main()
    else:
        typer.echo(f"Launching: {' '.join(cmd)}")
        result = subprocess.run(cmd)
        raise typer.Exit(result.returncode)


# ===========================================================================
# moe benchmark
# ===========================================================================


@app.command("benchmark")
def benchmark(
    cuda: bool = typer.Option(
        False,
        "--cuda",
        help="Run GPU benchmark (requires CUDA + Triton).",
    ),
    json_out: Path = typer.Option(
        Path("/tmp/moe_bench.json"),
        "--json",
        help="Output path for JSON results.",
    ),
    csv_out: Optional[Path] = typer.Option(
        None,
        "--csv",
        help="Output path for CSV results (optional).",
    ),
) -> None:
    """Run the moe-engine micro-benchmark suite.

    CPU sweep (no GPU required):

        moe benchmark

    GPU sweep (requires CUDA + Triton, e.g. on T4 or H100):

        moe benchmark --cuda --json benchmarks/gpu_results.json

    Results are written to the JSON path and optionally a CSV.
    """
    cmd = [
        sys.executable,
        str(_ROOT / "benchmarks" / "run_benchmark.py"),
        "--json",
        str(json_out),
    ]
    if cuda:
        cmd.append("--cuda")
    if csv_out:
        cmd += ["--csv", str(csv_out)]

    typer.echo(f"Running benchmark → {json_out}")
    result = subprocess.run(cmd, cwd=str(_ROOT))
    if result.returncode == 0:
        typer.echo(f"Done. Results written to {json_out}")
    raise typer.Exit(result.returncode)


# ===========================================================================
# moe validate
# ===========================================================================


@app.command("validate")
def validate(
    paths: list[Path] = typer.Argument(
        ...,
        help="One or more YAML config files or directories containing *.yaml files.",
    ),
) -> None:
    """Validate one or more moe-engine YAML config files.

    Reports field-level validation errors with the offending field path
    and a clear description of the constraint that was violated.

    Examples:

        moe validate configs/smoke.yaml

        moe validate configs/

        moe validate configs/smoke.yaml configs/default.yaml
    """
    from pkg.utils.config import ConfigValidationError, MoEConfig

    all_paths: list[Path] = []
    for p in paths:
        if p.is_dir():
            found = sorted(p.glob("*.yaml")) + sorted(p.glob("*.yml"))
            if not found:
                typer.echo(f"  [WARN] No *.yaml files found in {p}", err=True)
            all_paths.extend(found)
        elif p.is_file():
            all_paths.append(p)
        else:
            typer.echo(f"  [ERROR] Path not found: {p}", err=True)
            raise typer.Exit(1)

    if not all_paths:
        typer.echo("No config files to validate.", err=True)
        raise typer.Exit(1)

    typer.echo(f"\nValidating {len(all_paths)} config(s):\n")
    failed = 0
    for p in all_paths:
        try:
            cfg = MoEConfig.from_yaml(p)
            typer.echo(
                f"  \033[92m[OK]\033[0m  {p.name:<30s} "
                f"H={cfg.model.hidden_dim} E={cfg.model.num_experts} "
                f"K={cfg.model.top_k} world={cfg.parallelism.world_size} "
                f"dtype={cfg.model.dtype}"
            )
        except (ConfigValidationError, FileNotFoundError) as exc:
            typer.echo(f"  \033[91m[FAIL]\033[0m {p.name}:", err=True)
            for line in str(exc).splitlines():
                typer.echo(f"         {line}", err=True)
            failed += 1

    typer.echo()
    if failed == 0:
        typer.echo(f"\033[92mAll {len(all_paths)} config(s) valid.\033[0m")
    else:
        typer.echo(
            f"\033[91m{failed} config(s) failed, {len(all_paths) - failed} passed.\033[0m",
            err=True,
        )
        raise typer.Exit(1)


# ===========================================================================
# moe info
# ===========================================================================


@app.command("info")
def info() -> None:
    """Print moe-engine environment information.

    Useful for bug reports and reproducing results. Shows:
    Python version, PyTorch version, CUDA availability, Triton version,
    and the moe-engine package version.
    """

    typer.echo("\n=== moe-engine environment ===\n")

    # Python
    typer.echo(f"  Python:    {sys.version.split()[0]}")

    # PyTorch
    try:
        import torch

        typer.echo(f"  PyTorch:   {torch.__version__}")
        if torch.cuda.is_available():
            typer.echo(
                f"  CUDA:      {torch.version.cuda}  (device: {torch.cuda.get_device_name(0)})"
            )
        else:
            typer.echo("  CUDA:      not available")
    except ImportError:
        typer.echo("  PyTorch:   NOT INSTALLED")

    # Triton
    try:
        import triton

        typer.echo(f"  Triton:    {triton.__version__}")
    except ImportError:
        typer.echo("  Triton:    not installed (GPU path unavailable)")

    # Pydantic
    try:
        import pydantic

        typer.echo(f"  Pydantic:  {pydantic.__version__}")
    except ImportError:
        typer.echo("  Pydantic:  not installed (config validation unavailable)")

    # moe-engine
    try:
        import pkg as _pkg

        typer.echo(f"  moe-engine: {_pkg.__version__}")
    except (ImportError, AttributeError):
        typer.echo("  moe-engine: (version unknown)")

    # Config validation
    typer.echo()
    typer.echo("  Configs:")
    for yaml_path in sorted((_ROOT / "configs").glob("*.yaml")):
        try:
            from pkg.utils.config import MoEConfig

            cfg = MoEConfig.from_yaml(yaml_path)
            typer.echo(
                f"    \033[92m[OK]\033[0m {yaml_path.name}: "
                f"H={cfg.model.hidden_dim} E={cfg.model.num_experts}"
            )
        except Exception as exc:
            typer.echo(f"    \033[91m[FAIL]\033[0m {yaml_path.name}: {exc}")

    typer.echo()


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    app()
