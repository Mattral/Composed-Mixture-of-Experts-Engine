"""
tests/test_router_losses.py
===========================

Single-process tests for the router's training signals:

* F23: ``MoERouter(bias=True)`` must keep the combine weights differentiable
  with respect to the gate matrix, the bias and the tokens. Compared against
  a dense fp64 softmax-top-k reference that imports nothing from ``pkg``.
* F5: every MoE layer must produce tensor-valued auxiliary losses (router
  z-loss and a Switch-style load-balancing loss) that train the router, and
  the trainer must collect them from every layer. The per-layer terms are
  *sums* over the layer's tokens, normalised later by the global token count
  like the main loss (ADR-011); the load-balancing term uses the global
  expert fractions ``f_e`` (here, one process: the local ones).

Runs in Tier-0 (``-m cpu``); no process group needed. The multi-rank
behaviour (global ``f_e``, layout independence) is in
``tests/test_differential.py::test_layout_*[aux-*]``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pkg.distributed.mesh import build_topology  # noqa: E402
from pkg.distributed.moe_layer import DistributedMoELayer  # noqa: E402
from pkg.kernels.moe_router import MoERouter  # noqa: E402

pytestmark = pytest.mark.cpu

_F5 = (
    "F5: the router z-loss is a detached Python float and there is no "
    "load-balancing loss, so neither trains the router"
)


def _dense_aux_sums(x, gate_w, k):
    """Dense fp64 reference for one layer's auxiliary sums over its tokens:
    z = sum_t logsumexp(l_t)^2 and lb = E * sum_e f_e * sum_t p_te, with
    f_e = (top-k assignments to e) / (N * k)."""
    logits = x.double() @ gate_w.double()
    N, E = logits.shape
    probs = torch.softmax(logits, dim=-1)
    idx = torch.topk(probs, k=k, dim=-1).indices
    f = torch.bincount(idx.reshape(-1), minlength=E).double() / (N * k)
    z = (torch.logsumexp(logits, dim=-1) ** 2).sum()
    lb = E * (f * probs.sum(0)).sum()
    return z, lb


def _layer(seed=0, H=8, Fd=16, E=6, K=2):
    torch.manual_seed(seed)
    topo = build_topology(dp_size=1, ep_size=1, tp_size=1, device_type="cpu")
    try:
        return DistributedMoELayer(H, Fd, E, K, topo, aux_losses=True)
    except TypeError as exc:  # no aux_losses argument yet
        pytest.fail(f"DistributedMoELayer has no aux losses: {exc}")


@pytest.mark.xfail(strict=True, reason=_F5)
@pytest.mark.parametrize("k", [1, 2])
def test_layer_aux_sums_and_grads_match_dense(k):
    layer = _layer(K=k)
    x = torch.randn(20, 8)
    layer(x)
    z, lb = layer.aux_z_sum, layer.aux_lb_sum
    assert isinstance(z, torch.Tensor) and z.requires_grad
    assert isinstance(lb, torch.Tensor) and lb.requires_grad
    (z + lb).backward()

    gr = layer.router.gate_w.detach().clone().requires_grad_(True)
    z_ref, lb_ref = _dense_aux_sums(x, gr, k)
    (z_ref + lb_ref).backward()
    torch.testing.assert_close(z.double(), z_ref, atol=1e-4, rtol=1e-5)
    torch.testing.assert_close(lb.double(), lb_ref, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(
        layer.router.gate_w.grad, gr.grad, atol=1e-5, rtol=1e-4, msg="gate_w.grad"
    )


@pytest.mark.xfail(strict=True, reason=_F5)
def test_aux_losses_change_router_gradients():
    """M2 done-when (a): adding the auxiliary losses changes the router's
    gradient (a detached float could not)."""
    from pkg.distributed.moe_layer import collect_aux_losses

    layer = _layer()
    x, cot = torch.randn(20, 8), torch.randn(20, 8)
    (layer(x) * cot).sum().backward()
    g_main = layer.router.gate_w.grad.clone()
    layer.zero_grad()
    out = layer(x)
    z, lb = collect_aux_losses(layer)
    ((out * cot).sum() + 1e-2 * z + 1e-2 * lb).backward()
    assert not torch.allclose(layer.router.gate_w.grad, g_main)


class _TwoLayers(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = _layer(seed=1)
        self.b = _layer(seed=2)

    def forward(self, x):
        return self.b(self.a(x))


@pytest.mark.xfail(strict=True, reason=_F5)
def test_collect_aux_losses_sums_every_layer():
    """The trainer must see every MoE layer's auxiliary losses, not only the
    first block's (train.py read block 0 alone)."""
    from pkg.distributed.moe_layer import collect_aux_losses

    model = _TwoLayers()
    model(torch.randn(20, 8))
    z, lb = collect_aux_losses(model)
    torch.testing.assert_close(z, model.a.aux_z_sum + model.b.aux_z_sum)
    torch.testing.assert_close(lb, model.a.aux_lb_sum + model.b.aux_lb_sum)
    lb.backward()
    for name in ("a", "b"):
        g = getattr(model, name).router.gate_w.grad
        assert g is not None and g.abs().sum() > 0, f"layer {name}: no aux gradient"


@pytest.mark.xfail(strict=True, reason=_F5)
def test_training_config_aux_weights_and_model_wiring():
    """D4: load-balancing weight defaults to 1e-2, z-loss stays opt-in (0.0);
    the model enables the auxiliary losses on every layer when either
    weight is non-zero."""
    from pkg.models.moe import ToyMoEModel
    from pkg.utils.config import MoEConfig

    cfg = MoEConfig.from_yaml(ROOT / "configs" / "smoke.yaml")
    assert cfg.training.load_balance_weight == pytest.approx(1e-2)
    assert MoEConfig().training.z_loss_weight == 0.0
    topo = build_topology(dp_size=1, ep_size=1, tp_size=1, device_type="cpu")
    model = ToyMoEModel(cfg, topo)
    layers = [m for m in model.modules() if isinstance(m, DistributedMoELayer)]
    assert layers and all(m.aux_losses for m in layers)


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
