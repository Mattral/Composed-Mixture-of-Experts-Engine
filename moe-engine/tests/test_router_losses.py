"""
tests/test_router_losses.py
===========================

Single-process tests for the router's training signals:

* F23: ``MoERouter(bias=True)`` must keep the combine weights differentiable
  with respect to the gate matrix, the bias and the tokens. Compared against
  a dense fp64 softmax-top-k reference that imports nothing from ``pkg``.

Runs in Tier-0 (``-m cpu``); no process group needed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pkg.kernels.moe_router import MoERouter  # noqa: E402

pytestmark = pytest.mark.cpu


def _dense_biased_route(x, gate_w, bias, k):
    """Softmax over ``x @ gate_w + bias`` (fp64), top-k, renormalised."""
    probs = torch.softmax(x.double() @ gate_w.double() + bias.double(), dim=-1)
    vals, idx = torch.topk(probs, k=k, dim=-1)
    return idx, (vals / vals.sum(-1, keepdim=True)).to(x.dtype)


@pytest.mark.parametrize("k", [1, 2])
def test_biased_router_weights_are_differentiable(k):
    """F23: the bias path recomputed idx/w under no_grad, so w had no grad_fn."""
    torch.manual_seed(0)
    N, H, E = 16, 8, 6
    router = MoERouter(H, E, top_k=k, bias=True)
    with torch.no_grad():
        router.bias.copy_(torch.randn(E))  # non-zero, so the bias matters
    x = torch.randn(N, H, requires_grad=True)
    cot = torch.randn(N, k)

    idx, w, _ = router(x)
    (w * cot).sum().backward()

    xr = x.detach().clone().requires_grad_(True)
    gr = router.gate_w.detach().clone().requires_grad_(True)
    br = router.bias.detach().clone().requires_grad_(True)
    idx_ref, w_ref = _dense_biased_route(xr, gr, br, k)
    (w_ref * cot).sum().backward()

    assert torch.equal(idx, idx_ref)
    torch.testing.assert_close(w, w_ref, atol=1e-6, rtol=1e-5)
    for name, got, exp in (
        ("gate_w", router.gate_w.grad, gr.grad),
        ("bias", router.bias.grad, br.grad),
        ("tokens", x.grad, xr.grad),
    ):
        assert got is not None, f"{name}.grad is None"
        torch.testing.assert_close(got, exp, atol=1e-5, rtol=1e-4, msg=f"{name}.grad")
