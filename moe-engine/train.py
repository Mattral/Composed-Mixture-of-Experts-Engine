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

v0.3 changes
------------
* Model definition extracted to pkg/models/moe.py (no longer entangled here).
* Configuration uses MoEConfig (Pydantic-validated) via load_config shim.
  New callers can use ``MoEConfig.from_yaml(path)`` directly.
* Imports consolidated to public pkg.distributed API surface.
* Cleaner separation of bootstrap / model / telemetry / training loop.

v0.2 additions (retained)
--------------------------
* Routing quality telemetry: expert_load_imbalance, router_z_loss per step.
* Comm/compute overlap ratio in collective telemetry block.
* --profile flag: structured benchmark JSON to benchmarks/.
* compute_mfu_detailed for accurate dense + sparse FLOP accounting.
* Warm-up LR schedule with cosine decay.
* Gradient accumulation (gradient_accumulation_steps config key).
* WandB integration (WANDB_API_KEY env var + --wandb-project flag).
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist

from pkg.distributed import (
    build_topology,
    apply_fsdp2,
)
from pkg.elastic.fault_monitor import (
    ElasticConfig as _ElasticRuntimeConfig,
    ElasticTrainerHarness,
)
from pkg.models import ToyMoEModel
from pkg.models.moe import build_model
from pkg.telemetry.logger import StructuredLogger, StepRecord
from pkg.utils.config import MoEConfig, load_config
from pkg.utils.mfu import MFUAccountant, compute_moe_flops


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="moe-engine training entrypoint",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", required=True, help="Path to YAML config file.")
    p.add_argument("--max-steps", type=int, default=None,
                   help="Override max_steps from config.")
    p.add_argument("--smoke", action="store_true",
                   help="Minimal smoke-test run (2 steps, tiny dimensions).")
    p.add_argument("--profile", action="store_true",
                   help="Write a benchmark JSON to benchmarks/ on exit.")
    p.add_argument("--prometheus-port", type=int, default=0,
                   help="Port for Prometheus /metrics endpoint (0=disabled).")
    p.add_argument("--wandb-project", type=str, default=None,
                   help="WandB project name (requires WANDB_API_KEY env var).")
    p.add_argument("--no-wandb", action="store_true",
                   help="Disable WandB logging even if WANDB_API_KEY is set.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------

def _get_lr(step: int, warmup_steps: int, max_steps: int, lr: float) -> float:
    """Linear warm-up + cosine decay."""
    if step < warmup_steps:
        return lr * (step + 1) / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
    return lr * 0.5 * (1.0 + math.cos(math.pi * progress))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    args = parse_args()

    # ----------------------------------------------------------------
    # Config loading (Pydantic-validated).
    # ----------------------------------------------------------------
    # load_config returns a _LegacyConfig shim that also exposes .raw
    # for backward-compatible dict access.  New code should use
    # MoEConfig.from_yaml(args.config) directly.
    legacy_cfg = load_config(args.config)
    cfg = legacy_cfg.typed()   # typed MoEConfig

    # Smoke-test overrides on the typed config's raw dict representation.
    if args.smoke:
        cfg = MoEConfig.from_dict({
            **cfg.to_dict(),
            "model": {
                **cfg.to_dict()["model"],
                "hidden_dim": 64,
                "num_layers": 2,
                "ffn_dim": 128,
                "num_experts": 4,
                "sequence_length": 16,
                "vocab_size": 256,
            },
            "training": {
                **cfg.to_dict()["training"],
                "micro_batch_size": 2,
                "max_steps": 5,
                "gradient_accumulation_steps": 1,
                "warmup_steps": 0,
            },
        })

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

    par = cfg.parallelism
    dp, ep, tp, pp = (
        par.data_parallel, par.expert_parallel,
        par.tensor_parallel, par.pipeline_parallel,
    )

    # Auto-reduce parallelism if world_size is smaller than config demands.
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
    model = build_model(cfg, topology)
    model = apply_fsdp2(
        model, topology,
        mixed_precision_dtype=(
            torch.bfloat16 if cfg.model.dtype == "bfloat16" else None
        ),
    )

    lr = cfg.training.learning_rate
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr,
        weight_decay=cfg.training.weight_decay,
        betas=(0.9, 0.95),
    )

    # ----------------------------------------------------------------
    # Telemetry + MFU.
    # ----------------------------------------------------------------
    wandb_kwargs = {}
    if args.wandb_project:
        wandb_kwargs["project"] = args.wandb_project
    logger = StructuredLogger(
        json_path=cfg.telemetry.json_path,
        tensorboard_dir=cfg.telemetry.tensorboard_dir,
        rank=topology.rank,
        prometheus_port=args.prometheus_port,
        wandb_enabled=not args.no_wandb,
        wandb_init_kwargs=wandb_kwargs if wandb_kwargs else None,
    )
    logger.log_config(cfg.to_dict())

    mfu_acct = MFUAccountant(
        peak_tflops=cfg.telemetry.hardware_peak_tflops,
        mfu_target=cfg.telemetry.mfu_target,
        smoothing_window=50,
    )
    mfu_acct.configure(
        flops_per_token=compute_moe_flops(
            hidden_dim=cfg.model.hidden_dim,
            num_layers=cfg.model.num_layers,
            ffn_dim=cfg.model.ffn_dim,
            num_experts=cfg.model.num_experts,
            top_k=cfg.model.top_k,
            seq_length=cfg.model.sequence_length,
            batch_tokens=1,
            vocab_size=cfg.model.vocab_size,
        )
    )

    # ----------------------------------------------------------------
    # Elastic harness.
    # ----------------------------------------------------------------
    el_cfg = _ElasticRuntimeConfig(
        local_ckpt_dir=cfg.checkpoint.local_dir,
        remote_uri=cfg.checkpoint.remote_uri,
        s3_endpoint=os.environ.get("S3_ENDPOINT_URL"),
        retention=cfg.checkpoint.retention,
        async_workers=cfg.checkpoint.async_workers,
        health_interval_s=cfg.elastic.health_check_interval_s,
        drop_grace_s=cfg.elastic.drop_grace_period_s,
        min_nodes=cfg.elastic.min_nodes,
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
    max_steps = (
        args.max_steps if args.max_steps is not None else cfg.training.max_steps
    )
    warmup_steps = cfg.training.warmup_steps
    grad_accum = cfg.training.gradient_accumulation_steps

    B = cfg.training.micro_batch_size
    S = cfg.model.sequence_length
    V = cfg.model.vocab_size

    profile_records: list = []

    for step in range(start_step, max_steps):
        step_lr = _get_lr(step, warmup_steps, max_steps, lr)
        for pg in optimizer.param_groups:
            pg["lr"] = step_lr

        mfu_acct.start_step()
        optimizer.zero_grad(set_to_none=True)

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

        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip)
        optimizer.step()

        mfu_res = mfu_acct.end_step(tokens=B * S * grad_accum)

        # -- Telemetry envelope --
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

        # Pull router + MoE telemetry from first block.
        try:
            first_router = model.blocks[0].moe.router
            first_moe = model.blocks[0].moe
            if first_router.last_profile is not None:
                p_info = first_router.last_profile
                rec.kernel = {
                    "sram_bytes_per_block": p_info.sram_bytes_per_block,
                    "achieved_bw_gbps": p_info.achieved_bandwidth_gbps,
                    "tokens_per_expert_mean": p_info.tokens_per_expert_mean,
                    "tokens_per_expert_std": p_info.tokens_per_expert_std,
                    "used_triton": p_info.used_triton,
                }
                rec.routing = {
                    "expert_load_imbalance": p_info.expert_load_imbalance,
                    "router_z_loss": p_info.router_z_loss,
                }
            rec.collective = {
                "all_to_all_dispatch_ms": first_moe.last_dispatch_ms,
                "all_to_all_combine_ms": first_moe.last_combine_ms,
                "expert_compute_ms": first_moe.last_expert_compute_ms,
                "comm_compute_overlap_ratio": first_moe.last_overlap_ratio,
            }
        except (AttributeError, IndexError):
            pass

        if step % cfg.training.log_interval == 0:
            logger.emit(rec)
            if topology.rank == 0:
                logging.info(
                    "step=%d loss=%.4f %s",
                    step, total_loss, mfu_acct.summary_str(),
                )

        if step > 0 and step % cfg.training.ckpt_interval == 0:
            harness.checkpoint(model, optimizer, step)

        if step > 0 and step % 50 == 0:
            dead = harness.health_check()
            if dead:
                logging.warning("Rank drop: %s; entering recovery", dead)
                topology = harness.recover(
                    model, optimizer, num_experts=cfg.model.num_experts
                )

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

    if args.profile and topology.rank == 0 and profile_records:
        bench_dir = Path("benchmarks")
        bench_dir.mkdir(exist_ok=True)
        ts = int(time.time())
        bench_path = bench_dir / f"run_{ts}_rank0.json"
        step_ms_all = [r["step_ms"] for r in profile_records]
        mfu_all = [r["mfu"] for r in profile_records]
        tps_all = [r["tokens_per_sec"] for r in profile_records]
        n = len(step_ms_all)
        summary = {
            "config": {
                "hidden_dim": cfg.model.hidden_dim,
                "num_experts": cfg.model.num_experts,
                "top_k": cfg.model.top_k,
                "num_layers": cfg.model.num_layers,
                "dtype": cfg.model.dtype,
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
