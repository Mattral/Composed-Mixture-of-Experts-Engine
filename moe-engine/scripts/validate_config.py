#!/usr/bin/env python
"""
scripts/validate_config.py
===========================

Validate one or more moe-engine YAML config files at import time.

Usage
-----
    python scripts/validate_config.py configs/              # validate all *.yaml in dir
    python scripts/validate_config.py configs/smoke.yaml    # single file
    python scripts/validate_config.py configs/smoke.yaml configs/default.yaml

Exit code 0 on success, 1 on any validation failure.
Prints a compact summary for every file processed.

Used by:
    make validate-config
    pre-commit hooks (optional)
    CI lint job
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running from any working directory
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from pkg.utils.config import ConfigValidationError, MoEConfig  # noqa: E402


def validate_path(p: Path) -> bool:
    """Validate a single YAML file. Returns True on success, False on failure."""
    try:
        cfg = MoEConfig.from_yaml(p)
        print(
            f"  [\033[92mOK\033[0m]  {p.name:<30s} "
            f"H={cfg.model.hidden_dim:<5d} "
            f"E={cfg.model.num_experts:<3d} "
            f"K={cfg.model.top_k:<2d} "
            f"world_size={cfg.parallelism.world_size:<4d} "
            f"dtype={cfg.model.dtype}"
        )
        return True
    except FileNotFoundError as exc:
        print(f"  [\033[91mFAIL\033[0m] {p.name}: {exc}")
        return False
    except ConfigValidationError as exc:
        # Print the full multi-line error with indentation
        lines = str(exc).splitlines()
        print(f"  [\033[91mFAIL\033[0m] {p.name}:")
        for line in lines:
            print(f"         {line}")
        return False
    except Exception as exc:
        print(f"  [\033[91mFAIL\033[0m] {p.name}: Unexpected error: {exc}")
        return False


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python scripts/validate_config.py <config_file_or_dir> [...]")
        print("Example: python scripts/validate_config.py configs/")
        return 1

    paths: list[Path] = []
    for arg in sys.argv[1:]:
        p = Path(arg)
        if p.is_dir():
            found = sorted(p.glob("*.yaml")) + sorted(p.glob("*.yml"))
            if not found:
                print(f"  [WARN] No *.yaml files found in {p}")
            paths.extend(found)
        elif p.is_file():
            paths.append(p)
        else:
            print(f"  [\033[91mFAIL\033[0m] Path not found: {p}")
            return 1

    if not paths:
        print("No config files to validate.")
        return 1

    print(f"\nValidating {len(paths)} config file(s):\n")
    results = [validate_path(p) for p in paths]
    passed = sum(results)
    failed = len(results) - passed

    print(f"\n{'─' * 60}")
    if failed == 0:
        print(f"\033[92m  All {passed} config(s) valid.\033[0m\n")
        return 0
    else:
        print(f"\033[91m  {failed} config(s) FAILED, {passed} passed.\033[0m\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
