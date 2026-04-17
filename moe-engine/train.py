"""
train.py
========

Unified training entrypoint for moe-engine v0.3.

Launch via TorchElastic:

    torchrun \\
      --nnodes=$NUM_NODES \\
      --nproc_per_node=$GPUS_PER_NODE \\
      --rdzv_id=moe-run-001 \\
      --rdzv_backend=c10d \\
      --rdzv_endpoint=$RDZV_ENDPOINT \\
      --max_restarts=10 \\
      train.py --config configs/default.yaml

v0.2 additions
--------------
* Routing quality telemetry: expert_load_imbalance, router_z_loss per step.
* Comm/compute overlap ratio in collective telemetry block (v0.3).
* ``--profile`` flag: emits a structured benchmark summary to benchmarks/
* ``compute_mfu_detailed`` for accurate dense + sparse FLOP accounting.
* Warm-up LR schedule with cosine decay.
* Gradient accumulation support (``gradient_accumulation_steps`` config key).
* Rank-0 console summary with smoothed MFU.
* WandB integration via StructuredLogger (WANDB_API_KEY env var).
* ``--wandb-project`` flag for WandB project name.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.distributed as dist

from pkg.distributed.parallel_mesh import (
    DistributedMoELayer,
    ParallelTopology,
    build_topology,
    apply_fsdp2,
)
from pkg.elastic.fault_monitor import (
    ElasticConfig,
    ElasticTrainerHarness,
)
from pkg.telemetry.logger import StructuredLogger, StepRecord
from pkg.utils.config import load_config
from pkg.utils.mfu import MFUAccountant, compute_moe_flops, compute_mfu_detailed


# ----------------------------------------------------------------------
# Toy test model: stack of (RMSNorm + MoEBlock).
# ----------------------------------------------------------------------
class _RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        v = x.float()
        norm = v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + self.eps)
        return (norm * self.weight).to(x.dtype)


class _ToyMoEBlock(nn.Module):
    def __init__(self, model_cfg, topo: ParallelTopology, dtype: torch.dtype):
        super().__init__()
        H = model_cfg["hidden_dim"]
        self.norm = _RMSNorm(H)
        self.moe = DistributedMoELayer(
            hidden_dim=H,
            ffn_dim=model_cfg["ffn_dim"],
            num_experts=model_cfg["num_experts"],
            top_k=model_cfg["top_k"],
            topology=topo,
            capacity_factor=model_cfg["capacity_factor"],
            dtype=dtype,
        )

    def forward(self, x):
        return x + self.moe(self.norm(x))


class _ToyMoEModel(nn.Module):
    def __init__(self, cfg, topo: ParallelTopology):
        super().__init__()
        dtype_map = {
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
        }
        dtype = dtype_map[cfg["model"]["dtype"]]
        H = cfg["model"]["hidden_dim"]
        self.embed = nn.Embedding(cfg["model"]["vocab_size"], H, dtype=dtype)
        self.blocks = nn.ModuleList([
            _ToyMoEBlock(cfg["model"], topo, dtype)
            for _ in range(cfg["model"]["num_layers"])
        ])
        self.norm = _RMSNorm(H)
        self.lm_head = nn.Linear(H, cfg["model"]["vocab_size"], bias=False, dtype=dtype)

    def forward(self, ids):
        x = self.embed(ids)
        for blk in self.blocks:
            x = blk(x)
        return self.lm_head(self.norm(x))


# ----------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--profile", action="store_true",
                   help="Write a benchmark JSON to benchmarks/ on exit")
    p.add_argument("--prometheus-port", type=int, default=0,
                   help="Port for Prometheus /metrics endpoint (0=disabled)")
    p.add_argument("--wandb-project", type=str, default=None,
                   help="WandB project name (requires WANDB_API_KEY env var)")
    p.add_argument("--no-wandb", action="store_true",
                   help="Disable WandB logging even if WANDB_API_KEY is set")
    return p.parse_args()


def _get_lr(step: int, warmup_steps: int, max_steps: int, lr: float) -> float:
    """Linear warm-up + cosine decay."""
    import math
    if step < warmup_steps:
        return lr * (step + 1) / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
    return lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    args = parse_args()
    cfg = load_config(args.config).raw

    # ----------------------------------------------------------------
    # Process group bootstrap.
    # ----------------------------------------------------------------
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(
            backend="nccl" if torch.cuda.is_available() else "gloo",
            world_size=world_size, rank=rank,
        )

    dp = cfg["parallelism"]["data_parallel"]
    ep = cfg["parallelism"]["expert_parallel"]
    tp = cfg["parallelism"]["tensor_parallel"]
    pp = cfg["parallelism"]["pipeline_parallel"]

    if world_size < dp * tp * pp * ep:
        dp = max(1, world_size // (tp * pp * ep))
        if dp * tp * pp * ep > world_size:
            ep = max(1, world_size // (tp * pp))
            dp = max(1, world_size // (tp * pp * ep))

    topology = build_topology(
        dp_size=dp, ep_size=ep, tp_size=tp, pp_size=pp,
        device_type="cuda" if torch.cuda.is_available() else "cpu",
    )

    # ----------------------------------------------------------------
    # Model.
    # ----------------------------------------------------------------
    if args.smoke:
        cfg["model"]["hidden_dim"] = 64
        cfg["model"]["num_layers"] = 2
        cfg["model"]["ffn_dim"] = 128
        cfg["model"]["num_experts"] = 4
        cfg["model"]["sequence_length"] = 16
        cfg["model"]["vocab_size"] = 256
        cfg["training"]["micro_batch_size"] = 2

    model = _ToyMoEModel(cfg, topology)
    model = apply_fsdp2(model, topology,
                        mixed_precision_dtype=torch.bfloat16
                        if cfg["model"]["dtype"] == "bfloat16" else None)
    model = model.to(topology.device)

    lr = cfg["training"]["learning_rate"]
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr,
        weight_decay=cfg["training"]["weight_decay"],
        betas=(0.9, 0.95),
    )

    # ----------------------------------------------------------------
    # Telemetry + MFU.
    # ----------------------------------------------------------------
    wandb_kwargs = {}
    if args.wandb_project:
        wandb_kwargs["project"] = args.wandb_project
    logger = StructuredLogger(
        json_path=cfg["telemetry"]["json_path"],
        tensorboard_dir=cfg["telemetry"]["tensorboard_dir"],
        rank=topology.rank,
        prometheus_port=args.prometheus_port,
        wandb_enabled=not args.no_wandb,
        wandb_init_kwargs=wandb_kwargs if wandb_kwargs else None,
    )
    # Log hyperparameters to WandB run config (no-op if WandB inactive)
    logger.log_config(cfg.raw)
    mfu_acct = MFUAccountant(
        peak_tflops=cfg["telemetry"]["hardware_peak_tflops"],
        mfu_target=cfg["telemetry"]["mfu_target"],
        smoothing_window=50,
    )
    mfu_acct.configure(
        flops_per_token=compute_moe_flops(
            hidden_dim=cfg["model"]["hidden_dim"],
            num_layers=cfg["model"]["num_layers"],
            ffn_dim=cfg["model"]["ffn_dim"],
            num_experts=cfg["model"]["num_experts"],
            top_k=cfg["model"]["top_k"],
            seq_length=cfg["model"]["sequence_length"],
            batch_tokens=1,
            vocab_size=cfg["model"]["vocab_size"],
        )
    )

    # ----------------------------------------------------------------
    # Elastic harness.
    # ----------------------------------------------------------------
    el_cfg = ElasticConfig(
        local_ckpt_dir=cfg["checkpoint"]["local_dir"],
        remote_uri=cfg["checkpoint"]["remote_uri"],
        s3_endpoint=os.environ.get("S3_ENDPOINT_URL"),
        retention=cfg["checkpoint"]["retention"],
        async_workers=cfg["checkpoint"]["async_workers"],
        health_interval_s=cfg["elastic"]["health_check_interval_s"],
        drop_grace_s=cfg["elastic"]["drop_grace_period_s"],
        min_nodes=cfg["elastic"]["min_nodes"],
    )
    harness = ElasticTrainerHarness(el_cfg, topology)
    harness.install_signal_handlers()

    latest = harness.async_ckpt.latest_step()
    start_step = 0
    if latest is not None:
        harness.async_ckpt.load(model, optimizer, latest, rank=topology.rank)
        start_step = latest + 1
        logging.info("Resumed at step %d", start_step)

    # ----------------------------------------------------------------
    # Training loop.
    # ----------------------------------------------------------------
    max_steps = args.max_steps if args.max_steps is not None else cfg["training"]["max_steps"]
    if args.smoke:
        max_steps = min(max_steps, 5)

    warmup_steps = cfg["training"].get("warmup_steps", 0)
    grad_accum = cfg["training"].get("gradient_accumulation_steps", 1)

    B = cfg["training"]["micro_batch_size"]
    S = cfg["model"]["sequence_length"]
    V = cfg["model"]["vocab_size"]

    # Profiling: capture per-step stats for --profile flag
    profile_records: list[dict] = []

    for step in range(start_step, max_steps):
        # LR schedule
        step_lr = _get_lr(step, warmup_steps, max_steps, lr)
        for pg in optimizer.param_groups:
            pg["lr"] = step_lr

        mfu_acct.start_step()
        optimizer.zero_grad(set_to_none=True)

        # Gradient accumulation
        total_loss = 0.0
        for _acc_step in range(grad_accum):
            ids = torch.randint(0, V, (B, S), device=topology.device)
            targets = torch.randint(0, V, (B, S), device=topology.device)
            logits = model(ids)
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, V).float(), targets.view(-1),
            ) / grad_accum
            loss.backward()
            total_loss += float(loss.detach().item())

        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["grad_clip"])
        optimizer.step()

        mfu_res = mfu_acct.end_step(tokens=B * S * grad_accum)

        # Telemetry envelope
        rec = StepRecord(
            step=step,
            loss=total_loss,
            mfu=mfu_res.mfu,
            tokens_per_sec=mfu_res.tokens_per_sec,
            wall_clock_ms=mfu_res.step_ms,
            kernel={},
            collective={},
            memory={},
            infra={
                "async_ckpt_commit_ms": harness.async_ckpt.last_commit_ms,
                "active_nodes": topology.world_size,
                "ep_world_size": topology.ep_size,
                "lr": step_lr,
                "grad_accum": grad_accum,
            },
            routing={},
        )

        # Pull router + MoE telemetry from first block
        try:
            first_router = model.blocks[0].moe.router
            first_moe = model.blocks[0].moe
            if first_router.last_profile is not None:
                p = first_router.last_profile
                rec.kernel = {
                    "sram_bytes_per_block": p.sram_bytes_per_block,
                    "achieved_bw_gbps": p.achieved_bandwidth_gbps,
                    "tokens_per_expert_mean": p.tokens_per_expert_mean,
                    "tokens_per_expert_std": p.tokens_per_expert_std,
                    "used_triton": p.used_triton,
                }
                rec.routing = {
                    "expert_load_imbalance": p.expert_load_imbalance,
                    "router_z_loss": p.router_z_loss,
                }
            rec.collective = {
                "all_to_all_dispatch_ms": first_moe.last_dispatch_ms,
                "all_to_all_combine_ms": first_moe.last_combine_ms,
                "expert_compute_ms": first_moe.last_expert_compute_ms,
                "comm_compute_overlap_ratio": first_moe.last_overlap_ratio,
            }
        except (AttributeError, IndexError):
            pass

        if step % cfg["training"]["log_interval"] == 0:
            logger.emit(rec)
            if topology.rank == 0:
                logging.info(
                    "step=%d loss=%.4f %s",
                    step, total_loss, mfu_acct.summary_str(),
                )

        if step > 0 and step % cfg["training"]["ckpt_interval"] == 0:
            harness.checkpoint(model, optimizer, step)

        if step > 0 and step % 50 == 0:
            dead = harness.health_check()
            if dead:
                logging.warning("Rank drop: %s; entering recovery", dead)
                topology = harness.recover(model, optimizer,
                                           num_experts=cfg["model"]["num_experts"])

        if args.profile:
            profile_records.append({
                "step": step,
                "step_ms": mfu_res.step_ms,
                "mfu": mfu_res.mfu,
                "tokens_per_sec": mfu_res.tokens_per_sec,
                "loss": total_loss,
                "dispatch_ms": rec.collective.get("all_to_all_dispatch_ms", 0.0),
                "combine_ms": rec.collective.get("all_to_all_combine_ms", 0.0),
                "load_imbalance": rec.routing.get("expert_load_imbalance", 1.0),
            })

    # ----------------------------------------------------------------
    # Shutdown.
    # ----------------------------------------------------------------
    harness.checkpoint(model, optimizer, max_steps)
    harness.shutdown()
    logger.close()

    # --profile: write benchmark JSON
    if args.profile and topology.rank == 0 and profile_records:
        bench_dir = Path("benchmarks")
        bench_dir.mkdir(exist_ok=True)
        ts = int(time.time())
        bench_path = bench_dir / f"run_{ts}_rank0.json"
        # Compute summary statistics
        step_ms_all = [r["step_ms"] for r in profile_records]
        mfu_all = [r["mfu"] for r in profile_records]
        tps_all = [r["tokens_per_sec"] for r in profile_records]
        n = len(step_ms_all)
        summary = {
            "config": {
                "hidden_dim": cfg["model"]["hidden_dim"],
                "num_experts": cfg["model"]["num_experts"],
                "top_k": cfg["model"]["top_k"],
                "num_layers": cfg["model"]["num_layers"],
                "dtype": cfg["model"]["dtype"],
                "world_size": world_size,
                "dp": dp, "ep": ep, "tp": tp, "pp": pp,
            },
            "steps": n,
            "mfu_mean": sum(mfu_all) / n,
            "mfu_p50": sorted(mfu_all)[n // 2],
            "step_ms_mean": sum(step_ms_all) / n,
            "tokens_per_sec_mean": sum(tps_all) / n,
            "per_step": profile_records,
        }
        bench_path.write_text(json.dumps(summary, indent=2))
        logging.info("Benchmark written to %s", bench_path)

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
