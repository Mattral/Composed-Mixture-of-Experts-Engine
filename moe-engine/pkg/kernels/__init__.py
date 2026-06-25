"""Hardware-aware Triton kernels for sparse MoE routing."""

from pkg.kernels.moe_router import (
    TRITON_AVAILABLE,
    MoERouter,
    MoERouterAutograd,
    MoERouterFunction,
    moe_topk_route,
)

__all__ = [
    "MoERouter",
    "moe_topk_route",
    "MoERouterFunction",
    "MoERouterAutograd",
    "TRITON_AVAILABLE",
]
