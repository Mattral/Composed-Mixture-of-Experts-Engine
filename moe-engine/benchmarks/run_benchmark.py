#!/usr/bin/env python3
"""
benchmarks/run_benchmark.py
===========================

Reproducible micro-benchmark for the moe-engine router and MoE layer.

Measures:
  1. Router forward + backward throughput (tokens/sec)
  2. DistributedMoELayer single-process forward latency
  3. All-to-all overhead vs expert compute ratio

Run:
  python benchmarks/run_benchmark.py                  # CPU sweep
  python benchmarks/run_benchmark.py --cuda           # GPU sweep (requires CUDA)
  python benchmarks/run_benchmark.py --csv result.csv # Save results

Exit 0 = all benchmarks passed; prints structured JSON summary to stdout.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from pkg.kernels.moe_router import MoERouter, moe_topk_route
from pkg.distributed.parallel_mesh import DistributedMoELayer, build_topology
from pkg.utils.mfu import compute_mfu_detailed


@dataclass
class BenchResult:
    name: str
    device: str
    N: int
    H: int
    E: int
    K: int
    batch_ms_mean: float
    batch_ms_std: float
    tokens_per_sec: float
    mfu_estimate: float          # rough MFU vs H100 peak
    passed: bool
    notes: str = ""


def _time_fn(fn, warmup: int = 3, iters: int = 20, sync_cuda: bool = False):
    """Time a zero-arg callable, return (mean_ms, std_ms)."""
    for _ in range(warmup):
        fn()
    if sync_cuda and torch.cuda.is_available():
        torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        if sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    mean = sum(times) / len(times)
    variance = sum((t - mean) ** 2 for t in times) / len(times)
    std = variance ** 0.5
    return mean, std


def bench_router_forward(
    N: int, H: int, E: int, K: int, device: str, iters: int = 20
) -> BenchResult:
    """Benchmark the router forward pass throughput."""
    tokens = torch.randn(N, H, device=device)
    gate_w = torch.randn(H, E, device=device) * (H ** -0.5)

    def fwd():
        return moe_topk_route(tokens, gate_w, K, force_reference=(device == "cpu"))

    mean_ms, std_ms = _time_fn(fwd, sync_cuda=(device != "cpu"))
    tps = (N / (mean_ms / 1000))

    # Rough MFU: router is ~N*H*E FLOPs (matmul only), divide by H100 peak
    router_flops = 2 * N * H * E
    mfu = compute_mfu_detailed(
        batch_tokens=N, param_dense=H * E, param_expert=0,
        num_experts=E, top_k=K, world_size=1,
        hardware_peak_tflops=989.0,
        step_time_sec=mean_ms / 1000,
    ).mfu

    return BenchResult(
        name="router_fwd", device=device, N=N, H=H, E=E, K=K,
        batch_ms_mean=mean_ms, batch_ms_std=std_ms,
        tokens_per_sec=tps, mfu_estimate=mfu, passed=True,
    )


def bench_router_fwd_bwd(
    N: int, H: int, E: int, K: int, device: str, iters: int = 20
) -> BenchResult:
    """Benchmark router forward + backward combined."""
    tokens = torch.randn(N, H, device=device, requires_grad=True)
    gate_w = torch.randn(H, E, device=device, requires_grad=True)
    gate_w.data *= H ** -0.5

    def fwd_bwd():
        idx, w = moe_topk_route(tokens, gate_w, K, force_reference=(device == "cpu"))
        loss = w.sum()
        loss.backward()
        if tokens.grad is not None:
            tokens.grad.zero_()
        if gate_w.grad is not None:
            gate_w.grad.zero_()

    mean_ms, std_ms = _time_fn(fwd_bwd, sync_cuda=(device != "cpu"))
    tps = N / (mean_ms / 1000)
    mfu = compute_mfu_detailed(
        batch_tokens=N, param_dense=H * E * 3, param_expert=0,
        num_experts=E, top_k=K, world_size=1,
        hardware_peak_tflops=989.0,
        step_time_sec=mean_ms / 1000,
    ).mfu

    return BenchResult(
        name="router_fwd_bwd", device=device, N=N, H=H, E=E, K=K,
        batch_ms_mean=mean_ms, batch_ms_std=std_ms,
        tokens_per_sec=tps, mfu_estimate=mfu, passed=True,
    )


def bench_moe_layer(
    B: int, S: int, H: int, F: int, E: int, K: int, device: str, iters: int = 10
) -> BenchResult:
    """Benchmark a full DistributedMoELayer (single-process, no collectives)."""
    topo = build_topology(dp_size=1, ep_size=1, device_type=device)
    layer = DistributedMoELayer(
        hidden_dim=H, ffn_dim=F, num_experts=E, top_k=K, topology=topo
    ).to(device)

    x = torch.randn(B, S, H, device=device)

    def fwd():
        with torch.no_grad():
            return layer(x)

    mean_ms, std_ms = _time_fn(fwd, sync_cuda=(device != "cpu"), iters=iters)
    N = B * S
    tps = N / (mean_ms / 1000)

    # Expert FFN FLOPs per active token: 3 GEMMs of size H*F (SwiGLU)
    expert_flops_per_token = K * 6 * H * F
    mfu = compute_mfu_detailed(
        batch_tokens=N, param_dense=0,
        param_expert=3 * H * F,  # SwiGLU: gate + up + down
        num_experts=E, top_k=K, world_size=1,
        hardware_peak_tflops=989.0,
        step_time_sec=mean_ms / 1000,
    ).mfu

    return BenchResult(
        name="moe_layer_fwd", device=device, N=N, H=H, E=E, K=K,
        batch_ms_mean=mean_ms, batch_ms_std=std_ms,
        tokens_per_sec=tps, mfu_estimate=mfu, passed=True,
        notes=f"B={B} S={S} F={F}",
    )


def bench_token_conservation(
    N: int, H: int, E: int, K: int, device: str
) -> BenchResult:
    """Validate token conservation invariant holds across 100 random seeds."""
    violations = 0
    for seed in range(100):
        torch.manual_seed(seed)
        tokens = torch.randn(N, H, device=device)
        router = MoERouter(H, E, K)
        idx, w, cnt = router(tokens)
        if int(cnt.sum().item()) != N * K:
            violations += 1

    return BenchResult(
        name="token_conservation_sweep", device=device, N=N, H=H, E=E, K=K,
        batch_ms_mean=0.0, batch_ms_std=0.0,
        tokens_per_sec=0.0, mfu_estimate=0.0,
        passed=(violations == 0),
        notes=f"violations={violations}/100",
    )


def run_suite(device: str) -> List[BenchResult]:
    results = []
    print(f"\n{'='*60}")
    print(f"  moe-engine benchmark suite  |  device={device}")
    print(f"{'='*60}\n")

    configs = [
        # (N,    H,   E,  K)  — varied scale
        (512,  256,  16, 2),
        (1024, 512,  32, 2),
        (2048, 1024, 64, 2),
        (4096, 2048, 64, 4),
    ]

    for N, H, E, K in configs:
        tag = f"N={N} H={H} E={E} K={K}"
        try:
            r = bench_router_forward(N, H, E, K, device)
            print(f"  router_fwd   {tag}  {r.batch_ms_mean:.2f}±{r.batch_ms_std:.2f}ms  "
                  f"{r.tokens_per_sec/1e6:.2f}M tok/s")
            results.append(r)
        except Exception as exc:
            print(f"  router_fwd   {tag}  FAILED: {exc}")
            results.append(BenchResult("router_fwd", device, N, H, E, K,
                                        0, 0, 0, 0, False, str(exc)))

        try:
            r = bench_router_fwd_bwd(N, H, E, K, device)
            print(f"  router_fwd_bwd {tag}  {r.batch_ms_mean:.2f}±{r.batch_ms_std:.2f}ms  "
                  f"{r.tokens_per_sec/1e6:.2f}M tok/s")
            results.append(r)
        except Exception as exc:
            print(f"  router_fwd_bwd {tag}  FAILED: {exc}")
            results.append(BenchResult("router_fwd_bwd", device, N, H, E, K,
                                        0, 0, 0, 0, False, str(exc)))

    # MoE layer benchmarks
    moe_configs = [
        # (B, S, H,   F,    E,  K)
        (2,  16, 128, 256,  8,  2),
        (2,  32, 256, 512,  16, 2),
        (4,  16, 512, 1024, 32, 2),
    ]
    for B, S, H, F, E, K in moe_configs:
        tag = f"B={B} S={S} H={H} F={F} E={E} K={K}"
        try:
            r = bench_moe_layer(B, S, H, F, E, K, device)
            print(f"  moe_layer    {tag}  {r.batch_ms_mean:.2f}±{r.batch_ms_std:.2f}ms  "
                  f"{r.tokens_per_sec/1e3:.1f}k tok/s")
            results.append(r)
        except Exception as exc:
            print(f"  moe_layer    {tag}  FAILED: {exc}")
            results.append(BenchResult("moe_layer_fwd", device, B * S, H, E, K,
                                        0, 0, 0, 0, False, str(exc)))

    # Token conservation
    r = bench_token_conservation(512, 128, 32, 2, device)
    print(f"  token_conservation  {r.notes}  {'PASS' if r.passed else 'FAIL'}")
    results.append(r)

    n_pass = sum(1 for r in results if r.passed)
    print(f"\n  {n_pass}/{len(results)} benchmarks passed")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", action="store_true",
                        help="Run GPU benchmarks (requires CUDA)")
    parser.add_argument("--csv", type=str, default=None,
                        help="Save results to CSV file")
    parser.add_argument("--json", type=str, default=None,
                        help="Save results to JSON file")
    args = parser.parse_args()

    all_results: List[BenchResult] = []

    all_results.extend(run_suite("cpu"))

    if args.cuda:
        if not torch.cuda.is_available():
            print("ERROR: --cuda requested but no CUDA device found")
            sys.exit(1)
        all_results.extend(run_suite("cuda"))

    # JSON output
    json_data = [asdict(r) for r in all_results]
    if args.json:
        Path(args.json).write_text(json.dumps(json_data, indent=2))
        print(f"\nJSON saved to {args.json}")
    else:
        print("\n" + json.dumps(json_data, indent=2))

    # CSV output
    if args.csv and all_results:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(asdict(all_results[0]).keys()))
            w.writeheader()
            for r in all_results:
                w.writerow(asdict(r))
        print(f"CSV saved to {args.csv}")

    failed = [r for r in all_results if not r.passed]
    if failed:
        print(f"\n{len(failed)} benchmark(s) FAILED:")
        for r in failed:
            print(f"  {r.name} ({r.notes})")
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
