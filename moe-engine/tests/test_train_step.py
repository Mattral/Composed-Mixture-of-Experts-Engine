"""
tests/test_train_step.py
========================

``train.py`` itself, under tensor parallelism (F18):

* Every rank of a TP group must train on the **same** tokens (TP layers
  split the model, not the data). Different data ranks (DP x EP) must get
  different tokens, reproducibly. ``train.py`` drew ``torch.randint`` per
  rank with no seeding, so TP peers saw different batches.
* One ``train.py`` step at tp = 2 (two micro-batches, router auxiliary
  losses, global-norm clipping, an optimizer step) must match a dense
  single-device reference built from ``tests/_differential.py``.

Multi-process Gloo on CPU; runs in the CI differential job, not Tier-0.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pkg.distributed.grad_sync import broadcast_replicated_params  # noqa: E402
from pkg.distributed.mesh import build_topology  # noqa: E402
from pkg.models.moe import ToyMoEModel  # noqa: E402
from pkg.utils.config import MoEConfig  # noqa: E402
from tests._differential import (  # noqa: E402
    check,
    compare,
    dense_aux_sums,
    dense_moe,
    failures_for,
    run_multiprocess,
)

pytestmark = pytest.mark.cpu


def _assert_ok(results, kind):
    fails = failures_for(results, kind)
    if fails:
        pytest.fail("\n".join(fails), pytrace=False)


def _train_api():
    """train.py's batch and step functions, or the ImportError text."""
    try:
        from train import micro_batch, train_step
    except ImportError as exc:
        return None, None, repr(exc)
    return micro_batch, train_step, ""


# ===========================================================================
# TP ranks see identical batches
# ===========================================================================


def _batch_worker(rank: int, ws: int, cfg: dict) -> list:
    topo = build_topology(
        dp_size=cfg["dp"], ep_size=cfg["ep"], tp_size=cfg["tp"], device_type="cpu"
    )
    micro_batch, _, err = _train_api()
    if micro_batch is None:
        return [check("batch", "train.micro_batch", False, err)]
    a = micro_batch(11, 3, 1, topo, 2, 8, 128)
    again = micro_batch(11, 3, 1, topo, 2, 8, 128)
    other = micro_batch(11, 3, 0, topo, 2, 8, 128)
    everyone = [None] * ws
    dist.all_gather_object(everyone, ((topo.dp_rank, topo.ep_rank), a))
    me = (topo.dp_rank, topo.ep_rank)
    peers = [b for d, b in everyone if d == me]
    others = [b for d, b in everyone if d != me]
    same = all(torch.equal(b[0], a[0]) and torch.equal(b[1], a[1]) for b in peers)
    distinct = all(not torch.equal(b[0], a[0]) for b in others)
    return [
        check("batch", "TP peers get identical ids and targets", same, f"{len(peers)} peers"),
        check("batch", "other data ranks get different ids", distinct, f"{len(others)} others"),
        check("batch", "reproducible", torch.equal(a[0], again[0]), ""),
        check("batch", "micro-batches differ", not torch.equal(a[0], other[0]), ""),
    ]


@pytest.mark.parametrize("dp,ep,tp", [(2, 1, 2), (1, 2, 2)], ids=["dp2-tp2", "ep2-tp2"])
def test_tp_ranks_get_identical_batches(dp, ep, tp):
    _assert_ok(run_multiprocess(_batch_worker, dp * ep * tp, dict(dp=dp, ep=ep, tp=tp)), "batch")


# ===========================================================================
# One train.py step at tp = 2 against a dense reference
# ===========================================================================


def _rmsnorm(x, w, eps=1e-5):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


def _step_worker(rank: int, ws: int, cfg_in: dict) -> list:
    LR, G = 0.1, 2
    cfg = MoEConfig.from_yaml(ROOT / "configs" / "smoke.yaml")
    cfg.parallelism.tensor_parallel = ws
    cfg.training.gradient_accumulation_steps = G
    cfg.training.grad_clip = cfg_in["clip"]
    m = cfg.model
    H, E, K, L, V = m.hidden_dim, m.num_experts, m.top_k, m.num_layers, m.vocab_size
    topo = build_topology(dp_size=1, ep_size=1, tp_size=ws, device_type="cpu")
    torch.manual_seed(500 + rank)
    model = ToyMoEModel(cfg, topo)
    broadcast_replicated_params(model, topo)

    micro_batch, train_step, err = _train_api()
    if train_step is None:
        return [check("step", "train.train_step", False, err)]
    batches = [micro_batch(7, 0, a, topo, 2, m.sequence_length, V) for a in range(G)]

    # ---- Dense reference: full experts from both TP shards. ----
    def is_expert(n):
        return ".moe.experts." in n

    shards = [None] * ws
    dist.all_gather_object(
        shards, {n: p.detach().clone() for n, p in model.named_parameters() if is_expert(n)}
    )
    rp = {
        n: p.detach().clone().requires_grad_(True)
        for n, p in model.named_parameters()
        if not is_expert(n)
    }
    full = {}
    for n in shards[0]:
        dim = 1 if ".w_down." in n else 0
        full[n] = torch.cat([s[n] for s in shards], dim).requires_grad_(True)
    experts = [
        [
            [full[f"blocks.{blk}.moe.experts.{e}.{w}.weight"] for w in ("w_gate", "w_up", "w_down")]
            for e in range(E)
        ]
        for blk in range(L)
    ]
    tw = cfg.training
    loss_ref, n_tokens = 0.0, 0
    for ids, targets in batches:
        x = F.embedding(ids, rp["embed.weight"])
        for blk in range(L):
            h = _rmsnorm(x, rp[f"blocks.{blk}.norm.weight"]).reshape(-1, H)
            gate = rp[f"blocks.{blk}.moe.router.gate_w"]
            z, lb = dense_aux_sums(h, gate, K)
            loss_ref = loss_ref + tw.z_loss_weight * z + tw.load_balance_weight * lb
            x = x + dense_moe(h, gate, experts[blk], K).reshape(x.shape)
        logits = F.linear(_rmsnorm(x, rp["norm.weight"]), rp["lm_head.weight"])
        loss_ref = loss_ref + F.cross_entropy(
            logits.reshape(-1, V), targets.reshape(-1), reduction="sum"
        )
        n_tokens += targets.numel()
    (loss_ref / n_tokens).backward()
    dense = list(rp.values()) + list(full.values())
    norm = float(torch.sqrt(sum((t.grad.double() ** 2).sum() for t in dense)))
    coef = min(1.0, tw.grad_clip / (norm + 1e-6))
    chunk = m.ffn_dim // ws
    rows = slice(topo.tp_rank * chunk, (topo.tp_rank + 1) * chunk)
    expected = {}
    with torch.no_grad():
        for n, p in model.named_parameters():
            t = full[n] if is_expert(n) else rp[n]
            after = t - LR * coef * t.grad
            if is_expert(n):
                after = after[:, rows] if ".w_down." in n else after[rows]
            expected[n] = after

    # ---- train.py's step ----
    opt = torch.optim.SGD(model.parameters(), lr=LR)
    train_step(model, opt, topo, batches, cfg)
    checks = [
        check("step", "clipping active", coef < 1.0 or cfg_in["clip"] > 1e3, f"coef={coef:.4f}")
    ]
    for n, p in model.named_parameters():
        checks.append(compare("step", f"{n} after train_step", p.detach(), expected[n]))
    return checks


@pytest.mark.parametrize("clip", [0.05, 1e9], ids=["clipped", "unclipped"])
def test_tp2_train_step_matches_dense(clip):
    """train.py's step at tp = 2 with 2 micro-batches and the router
    auxiliary losses on, with global-norm clipping active and inactive."""
    _assert_ok(run_multiprocess(_step_worker, 2, dict(clip=clip)), "step")
