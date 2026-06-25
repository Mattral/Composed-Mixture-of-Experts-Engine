"""
pkg/models/moe.py
=================

Toy MoE model for smoke-testing and single-GPU benchmarking.

Extracted from train.py (where it was entangled with the training loop) so
the same model can be instantiated independently for evaluation, conversion,
analysis, or alternate training loops.

Architecture
------------
    Embedding → N × (RMSNorm + DistributedMoEBlock) → RMSNorm → LM Head

This is a minimal transformer-like stack where every FFN block is replaced
with a full MoE block.  It is intentionally toy-scale and not meant to
match any specific production model.

Public API
----------
    RMSNorm       — Root Mean Square layer normalisation
    ToyMoEBlock   — (RMSNorm + DistributedMoELayer) with residual connection
    ToyMoEModel   — Full model: embed + blocks + norm + lm_head
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from pkg.distributed.mesh import ParallelTopology
from pkg.distributed.moe_layer import DistributedMoELayer
from pkg.utils.config import MoEConfig

__all__ = [
    "RMSNorm",
    "ToyMoEBlock",
    "ToyMoEModel",
]


# Registry import (lazy - avoids circular import at module level)
def _get_register_model():
    from pkg.models.registry import register_model

    return register_model


_DTYPE_MAP: Dict[str, torch.dtype] = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


# ===========================================================================
# Building blocks
# ===========================================================================


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalisation.

    Equivalent to LayerNorm without the mean subtraction, following the
    LLaMA / Mistral convention.

    Parameters
    ----------
    dim : int    Feature dimension to normalise over.
    eps : float  Numerical stability epsilon (default 1e-5).
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def extra_repr(self) -> str:
        return f"dim={self.weight.shape[0]}, eps={self.eps}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        v = x.float()
        norm = v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + self.eps)
        return (norm * self.weight).to(x.dtype)


class ToyMoEBlock(nn.Module):
    """Single MoE transformer block: RMSNorm + DistributedMoELayer + residual.

    Parameters
    ----------
    hidden_dim : int
    ffn_dim : int
    num_experts : int
    top_k : int
    topology : ParallelTopology
    capacity_factor : float
    dtype : torch.dtype
    """

    def __init__(
        self,
        hidden_dim: int,
        ffn_dim: int,
        num_experts: int,
        top_k: int,
        topology: ParallelTopology,
        capacity_factor: float = 1.25,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.norm = RMSNorm(hidden_dim)
        self.moe = DistributedMoELayer(
            hidden_dim=hidden_dim,
            ffn_dim=ffn_dim,
            num_experts=num_experts,
            top_k=top_k,
            topology=topology,
            capacity_factor=capacity_factor,
            dtype=dtype,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.moe(self.norm(x))


@_get_register_model()("toy_moe")
class ToyMoEModel(nn.Module):
    """Complete toy MoE language model.

    Architecture::

        Embedding(vocab_size, H)
        N × ToyMoEBlock(H, F, E, K)
        RMSNorm(H)
        Linear(H, vocab_size)  [LM head, weight-tied with embedding is optional]

    Parameters
    ----------
    cfg : MoEConfig
        Typed configuration.  Use ``MoEConfig.from_yaml(...)`` to construct.
    topology : ParallelTopology
        Distributed topology for the model's MoE and TP layers.
    """

    def __init__(self, cfg: MoEConfig, topology: ParallelTopology):
        super().__init__()
        m = cfg.model
        dtype = _DTYPE_MAP.get(m.dtype, torch.float32)
        H = m.hidden_dim

        self.embed = nn.Embedding(m.vocab_size, H, dtype=dtype)
        self.blocks = nn.ModuleList(
            [
                ToyMoEBlock(
                    hidden_dim=H,
                    ffn_dim=m.ffn_dim,
                    num_experts=m.num_experts,
                    top_k=m.top_k,
                    topology=topology,
                    capacity_factor=m.capacity_factor,
                    dtype=dtype,
                )
                for _ in range(m.num_layers)
            ]
        )
        self.norm = RMSNorm(H)
        self.lm_head = nn.Linear(H, m.vocab_size, bias=False, dtype=dtype)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        ids : LongTensor  ``[B, S]``

        Returns
        -------
        Tensor  ``[B, S, vocab_size]``  (logits)
        """
        x = self.embed(ids)
        for blk in self.blocks:
            x = blk(x)
        return self.lm_head(self.norm(x))

    def count_parameters(self) -> Dict[str, int]:
        """Return a parameter count breakdown.

        Returns
        -------
        dict with keys:
            total          — all parameters
            embed          — embedding + lm_head
            moe_router     — all router gate matrices
            moe_experts    — all expert FFN parameters
            other          — norms and any remaining parameters
        """
        total = sum(p.numel() for p in self.parameters())
        embed = sum(p.numel() for p in self.embed.parameters()) + sum(
            p.numel() for p in self.lm_head.parameters()
        )
        router = sum(p.numel() for blk in self.blocks for p in blk.moe.router.parameters())
        experts = sum(p.numel() for blk in self.blocks for p in blk.moe.experts.parameters())
        return {
            "total": total,
            "embed": embed,
            "moe_router": router,
            "moe_experts": experts,
            "other": total - embed - router - experts,
        }


# ---------------------------------------------------------------------------
# Factory helper
# ---------------------------------------------------------------------------


def build_model(cfg: MoEConfig, topology: ParallelTopology) -> ToyMoEModel:
    """Construct and move :class:`ToyMoEModel` to the topology's device.

    This is the recommended entry point for training scripts.

    Parameters
    ----------
    cfg : MoEConfig
    topology : ParallelTopology

    Returns
    -------
    ToyMoEModel  on ``topology.device``
    """
    model = ToyMoEModel(cfg, topology)
    model = model.to(topology.device)
    return model
