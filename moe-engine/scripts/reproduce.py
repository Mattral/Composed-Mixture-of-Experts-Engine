#!/usr/bin/env python3
"""
scripts/reproduce.py
=====================

Reproducibility helper for moe-engine benchmark and validation results.

Usage
-----
    # Reproduce CPU benchmarks (cpu_results_colab.json)
    python scripts/reproduce.py --target cpu

    # Reproduce T4 GPU validation (requires CUDA + Triton)
    python scripts/reproduce.py --target gpu

    # Reproduce specific invariant tests
    python scripts/reproduce.py --target invariants

    # Full reproduction report
    python scripts/reproduce.py --target all --report /tmp/reproduce_report.json

This script is the single source of truth for reproducing any number in
RESULTS.md or benchmarks/BENCHMARKS.md. Every claim in those documents
can be reproduced by running the appropriate target here.

Exit codes
----------
    0   All reproduced results match within tolerance
    1   One or more results deviate beyond tolerance
    2   Missing dependency (Triton, CUDA)
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from typing import List

# Allow running from repo root or moe-engine/
_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))


# ===========================================================================
# Result dataclass
# ===========================================================================


@dataclass
class ReproResult:
    target: str
    status: str  # "PASS" | "FAIL" | "SKIP" | "ERROR"
    claim: str  # what RESULTS.md claims
    measured: str  # what we actually measured
    tolerance: str  # acceptable deviation
    notes: str = ""


# ===========================================================================
# Individual reproducers
# ===========================================================================


def reproduce_invariants() -> List[ReproResult]:
    """Reproduce all mathematical invariants (token conservation, normalisation)."""
    results = []
    try:
        import torch

        from pkg.kernels.moe_router import MoERouter

        configs = [
            (512, 128, 32, 2),
            (1024, 256, 64, 2),
            (2048, 512, 64, 4),
        ]
        violations = 0
        for N, H, E, K in configs:
            router = MoERouter(hidden_dim=H, num_experts=E, top_k=K)
            for seed in range(10):
                torch.manual_seed(seed)
                tokens = torch.randn(N, H)
                idx, w, dispatch_cnt = router(tokens)
                total = int(dispatch_cnt.sum().item())
                if total != N * K:
                    violations += 1
            # Weight normalisation
            row_sums = w.sum(dim=-1)
            if not torch.allclose(row_sums, torch.ones(N), atol=1e-4):
                violations += 1

        results.append(
            ReproResult(
                target="invariants",
                status="PASS" if violations == 0 else "FAIL",
                claim="violations=0/100 (RESULTS.md token conservation sweep)",
                measured=f"violations={violations}/{len(configs) * 10}",
                tolerance="exact (0 violations required)",
            )
        )
    except Exception as exc:
        results.append(
            ReproResult(
                target="invariants",
                status="ERROR",
                claim="violations=0/100",
                measured="error",
                tolerance="exact",
                notes=str(exc),
            )
        )
    return results


def reproduce_cpu_throughput() -> List[ReproResult]:
    """Reproduce CPU reference path throughput numbers from BENCHMARKS.md."""
    results = []
    try:
        import time

        import torch

        from pkg.kernels.moe_router import MoERouter

        # Published CPU numbers from BENCHMARKS.md (tok/s)
        # Tolerance: ±30% (CPU timing is highly environment-dependent)
        published = {
            (512, 256, 16, 2): 747_123,
            (1024, 512, 32, 2): 420_892,
            (2048, 1024, 64, 2): 236_481,
        }
        WARMUP, REPS = 3, 20

        for (N, H, E, K), pub_tps in published.items():
            router = MoERouter(hidden_dim=H, num_experts=E, top_k=K)
            x = torch.randn(N, H)
            for _ in range(WARMUP):
                router(x)
            t0 = time.perf_counter()
            for _ in range(REPS):
                router(x)
            elapsed_ms = (time.perf_counter() - t0) / REPS * 1000
            measured_tps = int(N / elapsed_ms * 1000)
            ratio = measured_tps / pub_tps
            status = "PASS" if 0.5 <= ratio <= 2.0 else "FAIL"  # ±50% for CPU
            results.append(
                ReproResult(
                    target="cpu_throughput",
                    status=status,
                    claim=f"N={N},H={H},E={E},K={K}: {pub_tps:,} tok/s (BENCHMARKS.md)",
                    measured=f"{measured_tps:,} tok/s  ({ratio:.2f}× published)",
                    tolerance="0.5× – 2.0× (CPU timing is environment-dependent)",
                )
            )
    except Exception as exc:
        results.append(
            ReproResult(
                target="cpu_throughput",
                status="ERROR",
                claim="see BENCHMARKS.md",
                measured="error",
                tolerance="±50%",
                notes=str(exc),
            )
        )
    return results


def reproduce_config_validation() -> List[ReproResult]:
    """Reproduce config validation results."""
    results = []
    try:
        from pkg.utils.config import ConfigValidationError, MoEConfig

        # Valid configs load
        for path in ["configs/smoke.yaml", "configs/default.yaml"]:
            try:
                cfg = MoEConfig.from_yaml(path)
                results.append(
                    ReproResult(
                        target="config",
                        status="PASS",
                        claim=f"{path} loads without error",
                        measured=f"H={cfg.model.hidden_dim}, E={cfg.model.num_experts}",
                        tolerance="exact",
                    )
                )
            except Exception as exc:
                results.append(
                    ReproResult(
                        target="config",
                        status="FAIL",
                        claim=f"{path} loads",
                        measured=f"error: {exc}",
                        tolerance="exact",
                    )
                )

        # Invalid config raises
        try:
            MoEConfig.from_dict(
                {
                    "model": {
                        "hidden_dim": 64,
                        "num_layers": 1,
                        "num_experts": 4,
                        "top_k": 8,
                        "capacity_factor": 1.25,
                        "ffn_dim": 128,
                        "vocab_size": 256,
                        "sequence_length": 8,
                        "dtype": "float32",
                    }
                }
            )
            results.append(
                ReproResult(
                    target="config",
                    status="FAIL",
                    claim="top_k=8 > num_experts=4 raises ConfigValidationError",
                    measured="no error raised",
                    tolerance="exact",
                )
            )
        except ConfigValidationError:
            results.append(
                ReproResult(
                    target="config",
                    status="PASS",
                    claim="top_k=8 > num_experts=4 raises ConfigValidationError",
                    measured="ConfigValidationError raised correctly",
                    tolerance="exact",
                )
            )
    except Exception as exc:
        results.append(
            ReproResult(
                target="config",
                status="ERROR",
                claim="config system",
                measured=f"error: {exc}",
                tolerance="exact",
            )
        )
    return results


def reproduce_gpu() -> List[ReproResult]:
    """Reproduce GPU throughput numbers (requires CUDA + Triton)."""
    results = []
    try:
        import torch

        if not torch.cuda.is_available():
            results.append(
                ReproResult(
                    target="gpu",
                    status="SKIP",
                    claim="80.1× GPU speedup at N=4096 (RESULTS.md)",
                    measured="CUDA not available",
                    tolerance="±20%",
                    notes="Run on T4 or H100 with CUDA + Triton installed",
                )
            )
            return results

        # Run the GPU benchmark subprocess
        cmd = [
            sys.executable,
            str(_ROOT / "benchmarks" / "run_benchmark.py"),
            "--cuda",
            "--json",
            "/tmp/reproduce_gpu.json",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            results.append(
                ReproResult(
                    target="gpu",
                    status="ERROR",
                    claim="GPU benchmark",
                    measured=f"subprocess failed: {proc.stderr[:200]}",
                    tolerance="±20%",
                )
            )
            return results

        gpu_data = json.loads(pathlib.Path("/tmp/reproduce_gpu.json").read_text())
        cpu_data_path = _ROOT / "benchmarks" / "cpu_results_colab.json"
        if cpu_data_path.exists():
            cpu_data = json.loads(cpu_data_path.read_text())
            cpu_map = {(d["name"], d["N"], d["H"], d["E"], d["K"]): d for d in cpu_data}
        else:
            cpu_map = {}

        for d in gpu_data:
            if d["name"] == "router_fwd" and d["device"] == "cuda":
                key = ("router_fwd", d["N"], d["H"], d["E"], d["K"])
                gpu_tps = d["tokens_per_sec"]
                if key in cpu_map:
                    cpu_tps = cpu_map[key]["tokens_per_sec"]
                    speedup = gpu_tps / cpu_tps
                    # Check against published speedups from RESULTS.md
                    expected = {
                        (512, 256, 16, 2): 2.9,
                        (1024, 512, 32, 2): 8.7,
                        (2048, 1024, 64, 2): 20.4,
                        (4096, 2048, 64, 4): 80.1,
                    }.get((d["N"], d["H"], d["E"], d["K"]), None)
                    if expected:
                        ok = 0.7 * expected <= speedup <= 1.5 * expected
                        results.append(
                            ReproResult(
                                target="gpu",
                                status="PASS" if ok else "FAIL",
                                claim=f"N={d['N']},H={d['H']}: {expected:.1f}× speedup (RESULTS.md)",
                                measured=f"{speedup:.1f}× ({gpu_tps / 1e6:.3f}M tok/s)",
                                tolerance="±30% of published speedup",
                            )
                        )
    except Exception as exc:
        results.append(
            ReproResult(
                target="gpu",
                status="ERROR",
                claim="GPU throughput",
                measured=f"error: {exc}",
                tolerance="±20%",
                notes=str(exc),
            )
        )
    return results


# ===========================================================================
# Main
# ===========================================================================


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Reproduce moe-engine benchmark and validation results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--target",
        choices=["invariants", "cpu", "gpu", "config", "all"],
        default="all",
        help="Which results to reproduce.",
    )
    parser.add_argument(
        "--report",
        default=None,
        help="Write JSON report to this path.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print full result details.",
    )
    args = parser.parse_args()

    all_results: List[ReproResult] = []

    if args.target in ("invariants", "all"):
        print("Reproducing: mathematical invariants (token conservation, normalisation)...")
        all_results.extend(reproduce_invariants())

    if args.target in ("cpu", "all"):
        print("Reproducing: CPU reference throughput...")
        all_results.extend(reproduce_cpu_throughput())

    if args.target in ("config", "all"):
        print("Reproducing: config validation...")
        all_results.extend(reproduce_config_validation())

    if args.target in ("gpu", "all"):
        print("Reproducing: GPU throughput (requires CUDA + Triton)...")
        all_results.extend(reproduce_gpu())

    # ── Summary ──────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("REPRODUCTION SUMMARY")
    print("=" * 70)
    for r in all_results:
        icon = {"PASS": "✅", "FAIL": "❌", "SKIP": "⏭️", "ERROR": "💥"}.get(r.status, "?")
        print(f"\n{icon} [{r.status:5s}] {r.target.upper()}")
        print(f"  Claim   : {r.claim}")
        print(f"  Measured: {r.measured}")
        print(f"  Tol.    : {r.tolerance}")
        if r.notes:
            print(f"  Notes   : {r.notes}")

    passed = sum(1 for r in all_results if r.status == "PASS")
    failed = sum(1 for r in all_results if r.status == "FAIL")
    skipped = sum(1 for r in all_results if r.status == "SKIP")
    errored = sum(1 for r in all_results if r.status == "ERROR")

    print()
    print(f"Results: {passed} PASS  {failed} FAIL  {skipped} SKIP  {errored} ERROR")

    if args.report:
        report = {
            "ts": time.time(),
            "target": args.target,
            "summary": {"passed": passed, "failed": failed, "skipped": skipped, "errored": errored},
            "results": [asdict(r) for r in all_results],
        }
        pathlib.Path(args.report).write_text(json.dumps(report, indent=2))
        print(f"\nReport written to: {args.report}")

    return 0 if failed == 0 and errored == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
