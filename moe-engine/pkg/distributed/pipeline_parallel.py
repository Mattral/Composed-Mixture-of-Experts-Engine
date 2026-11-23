"""
pkg/distributed/pipeline_parallel.py
======================================

Pipeline-parallel stage with real dist.send/recv inter-stage communication.

v0.3: replaced the v0.2 single-process scheduling shim with a full
multi-process PipelineStage class that supports real distributed execution.

Design decisions
----------------
Activation tagging
    Every micro-batch carries an int64 header sent immediately before the
    activation tensor: ``[stage_id, mb_index, dtype_code, ndim, *shape]``,
    padded to a fixed length.  The receiver checks the stage and micro-batch
    index and allocates its buffer from the header, so non-first stages need
    neither the input micro-batches nor the activation shape in advance.

Micro-batch count
    Only the first stage holds the micro-batches.  Before the schedule starts,
    ``run_1f1b_distributed`` passes the count ``m`` down the pipeline, stage by
    stage, so every stage runs the same number of forwards and backwards.

Buffer management
    Forward activations are stashed in ``_activation_stash`` (dict keyed by
    mb_index) between the forward and backward passes.
    Entries are deleted after their backward to bound memory usage.

Tag protocol
    Send:  [header: Tensor[_HEADER_LEN, int64]] then [activation: Tensor[...]]
    Recv:  mirrors send.
    Both use blocking send/recv for simplicity; a future optimisation can
    replace these with isend/irecv + handle list (NCCL async pipe).

Single-process fast path
    When world_size == 1 (all CI tests, smoke runs), dist.send/recv are not
    called.  The existing run_1f1b scheduling logic passes activations through
    Python object references.  This preserves the 13-test suite.

Public API
----------
    PipelineStage
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn

from pkg.distributed.mesh import ParallelTopology, pp_process_group

__all__ = ["PipelineStage"]


class PipelineStage:
    """Pipeline stage with real dist.send/recv communication in multi-process mode.

    Parameters
    ----------
    stage_id : int
        0-based index of this stage in the pipeline.
    num_stages : int
        Total number of pipeline stages (== pp_size).
    module : Optional[nn.Module]
        The layer(s) this stage executes.  None = passthrough (useful for
        testing the scheduling logic without a real model).
    topology : Optional[ParallelTopology]
        When provided and pp_size > 1 with dist initialised, real
        dist.send/recv are used.  Otherwise: single-process passthrough.
    """

    # Tags: SEND_TAG_BASE + mb_index for activations;
    #       GRAD_TAG_BASE + mb_index for gradients.
    _SEND_TAG_BASE = 1000
    _GRAD_TAG_BASE = 1500
    _COUNT_TAG = 999
    # Header: [stage_id, mb_index, dtype_code, ndim, d0 .. d{_MAX_NDIM-1}]
    _MAX_NDIM = 8
    _HEADER_LEN = 4 + _MAX_NDIM
    _DTYPES: Tuple[torch.dtype, ...] = (
        torch.float32,
        torch.float64,
        torch.float16,
        torch.bfloat16,
        torch.int64,
        torch.int32,
        torch.bool,
    )

    def __init__(
        self,
        stage_id: int,
        num_stages: int,
        module: Optional[nn.Module] = None,
        topology: Optional[ParallelTopology] = None,
    ):
        if stage_id < 0 or stage_id >= num_stages:
            raise ValueError(f"stage_id ({stage_id}) must be in [0, num_stages={num_stages})")
        self.stage_id = int(stage_id)
        self.num_stages = int(num_stages)
        self.module = module
        self.topology = topology
        self._pp_group: Optional[dist.ProcessGroup] = (
            pp_process_group(topology) if topology is not None else None
        )
        self._multi_process: bool = (
            topology is not None and topology.pp_size > 1 and dist.is_initialized()
        )
        # Forward activation stash: mb_index → (input_tensor, output_tensor)
        self._activation_stash: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}

    # ------------------------------------------------------------------
    # Stage position queries
    # ------------------------------------------------------------------

    @property
    def is_first(self) -> bool:
        return self.stage_id == 0

    @property
    def is_last(self) -> bool:
        return self.stage_id == self.num_stages - 1

    def _prev_rank(self) -> Optional[int]:
        """Global rank of the previous pipeline stage, or None if stage 0."""
        if self.topology is None or self.is_first:
            return None
        base = self.topology.rank - self.topology.ep_size
        return base if base >= 0 else None

    def _next_rank(self) -> Optional[int]:
        """Global rank of the next pipeline stage, or None if last stage."""
        if self.topology is None or self.is_last:
            return None
        base = self.topology.rank + self.topology.ep_size
        return base if base < self.topology.world_size else None

    # ------------------------------------------------------------------
    # Low-level communication helpers
    # ------------------------------------------------------------------

    def _send_activation(self, tensor: torch.Tensor, mb_index: int) -> None:
        """Send activation tensor to the next stage (tagged header + data)."""
        next_rank = self._next_rank()
        if next_rank is None:
            return
        if tensor.dim() > self._MAX_NDIM:
            raise ValueError(
                f"Stage {self.stage_id}: activation has {tensor.dim()} dims; "
                f"at most {self._MAX_NDIM} are supported"
            )
        if tensor.dtype not in self._DTYPES:
            raise ValueError(f"Stage {self.stage_id}: unsupported activation dtype {tensor.dtype}")
        header = torch.zeros(self._HEADER_LEN, dtype=torch.long)
        header[0] = self.stage_id
        header[1] = mb_index
        header[2] = self._DTYPES.index(tensor.dtype)
        header[3] = tensor.dim()
        if tensor.dim():
            header[4 : 4 + tensor.dim()] = torch.tensor(tensor.shape, dtype=torch.long)
        tag = self._SEND_TAG_BASE + mb_index
        dist.send(header, dst=next_rank, group=self._pp_group, tag=tag)
        dist.send(tensor.detach().contiguous(), dst=next_rank, group=self._pp_group, tag=tag + 1)

    def _recv_activation(
        self,
        shape: Optional[Tuple[int, ...]] = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        mb_index: int = 0,
    ) -> torch.Tensor:
        """Receive activation tensor from the previous stage.

        The buffer's shape and dtype come from the header the sender attached.
        ``shape`` and ``dtype`` are optional; if given, they must match it.
        """
        prev_rank = self._prev_rank()
        if prev_rank is None:
            raise RuntimeError(
                f"Stage {self.stage_id}: no previous stage to receive activation from"
            )
        tag = self._SEND_TAG_BASE + mb_index
        header = torch.zeros(self._HEADER_LEN, dtype=torch.long)
        dist.recv(header, src=prev_rank, group=self._pp_group, tag=tag)
        h = header.tolist()
        if h[0] != self.stage_id - 1 or h[1] != mb_index:
            raise RuntimeError(
                f"Stage {self.stage_id}: expected activation (stage {self.stage_id - 1}, "
                f"micro-batch {mb_index}), got header (stage {h[0]}, micro-batch {h[1]})"
            )
        recv_dtype = self._DTYPES[h[2]]
        recv_shape = tuple(h[4 : 4 + h[3]])
        if shape is not None and tuple(shape) != recv_shape:
            raise RuntimeError(
                f"Stage {self.stage_id}: expected activation shape {tuple(shape)}, "
                f"header says {recv_shape}"
            )
        if dtype is not None and dtype != recv_dtype:
            raise RuntimeError(
                f"Stage {self.stage_id}: expected activation dtype {dtype}, header says {recv_dtype}"
            )
        buf = torch.empty(recv_shape, dtype=recv_dtype, device=device or torch.device("cpu"))
        dist.recv(buf, src=prev_rank, group=self._pp_group, tag=tag + 1)
        return buf

    def _exchange_micro_batch_count(self, local_count: int) -> int:
        """Pass the first stage's micro-batch count down the pipeline.

        The first stage sends ``local_count``; every other stage receives the
        count from its predecessor and forwards it to its successor.
        """
        count = torch.tensor([local_count], dtype=torch.long)
        if not self.is_first:
            dist.recv(count, src=self._prev_rank(), group=self._pp_group, tag=self._COUNT_TAG)
        if not self.is_last:
            dist.send(count, dst=self._next_rank(), group=self._pp_group, tag=self._COUNT_TAG)
        return int(count.item())

    def _send_gradient(self, grad: torch.Tensor, mb_index: int) -> None:
        """Send gradient to the previous stage."""
        prev_rank = self._prev_rank()
        if prev_rank is None:
            return
        tag = self._GRAD_TAG_BASE + mb_index
        dist.send(grad.contiguous(), dst=prev_rank, group=self._pp_group, tag=tag)

    def _recv_gradient(
        self,
        shape: Tuple[int, ...],
        dtype: torch.dtype,
        device: torch.device,
        mb_index: int,
    ) -> torch.Tensor:
        """Receive gradient from the next stage."""
        next_rank = self._next_rank()
        if next_rank is None:
            raise RuntimeError(f"Stage {self.stage_id}: no next stage to receive gradient from")
        tag = self._GRAD_TAG_BASE + mb_index
        buf = torch.empty(shape, dtype=dtype, device=device)
        dist.recv(buf, src=next_rank, group=self._pp_group, tag=tag)
        return buf

    # ------------------------------------------------------------------
    # Single-step forward / backward (public)
    # ------------------------------------------------------------------

    def forward_step(self, micro_batch: torch.Tensor) -> torch.Tensor:
        """Execute forward work for one micro-batch.

        Multi-process: receive from predecessor (non-first stages), apply
        module, send to successor (non-last stages).

        Single-process: apply module directly (passthrough if module is None).
        """
        if self._multi_process:
            if not self.is_first:
                micro_batch = self._recv_activation(
                    device=micro_batch.device,
                    mb_index=0,  # simple single-stream; mb_index managed by run_1f1b
                )
            out = self.module(micro_batch) if self.module is not None else micro_batch
            if not self.is_last:
                self._send_activation(out, mb_index=0)
            return out

        return self.module(micro_batch) if self.module is not None else micro_batch

    def backward_step(self, grad_output: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """Passthrough — multi-process backward is handled inside run_1f1b_distributed."""
        return grad_output

    # ------------------------------------------------------------------
    # 1F1B schedule — single-process shim (test path)
    # ------------------------------------------------------------------

    def run_1f1b(self, micro_batches: List[torch.Tensor]) -> List[Optional[torch.Tensor]]:
        """Run the 1F1B schedule (single-process / scheduling verification).

        This is the test shim used by the 13-test pipeline suite.
        For real multi-process execution, use :meth:`run_1f1b_distributed`.

        Parameters
        ----------
        micro_batches : list of Tensor
            Input micro-batches.

        Returns
        -------
        list of Optional[Tensor]
            Per-micro-batch gradient (None if not computed yet in warmup).
        """
        p = self.num_stages
        m = len(micro_batches)
        if m == 0:
            return []

        activations: List[Optional[torch.Tensor]] = [None] * m
        grads: List[Optional[torch.Tensor]] = [None] * m

        # Warmup: issue min(p-1, m) forwards without any backward.
        for t in range(min(p - 1, m)):
            activations[t] = self.forward_step(micro_batches[t])

        # Steady-state: one forward + one backward per clock.
        steady_range = max(0, m - (p - 1))
        for t in range(steady_range):
            idx_fwd = t + (p - 1)
            if idx_fwd < m:
                activations[idx_fwd] = self.forward_step(micro_batches[idx_fwd])
            if activations[t] is not None:
                grads[t] = self.backward_step(torch.ones_like(activations[t]))  # type: ignore[arg-type]

        # Drain: remaining backwards.
        drain_start = max(0, m - (p - 1))
        for t in range(drain_start, m):
            if activations[t] is not None:
                grads[t] = self.backward_step(torch.ones_like(activations[t]))  # type: ignore[arg-type]

        return grads

    # ------------------------------------------------------------------
    # 1F1B schedule — multi-process distributed implementation (v0.3)
    # ------------------------------------------------------------------

    def run_1f1b_distributed(
        self,
        micro_batches: List[torch.Tensor],
        loss_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Run the full multi-process 1F1B interleaved schedule.

        v0.3 distributed implementation.  Replaces the single-process shim
        for production use when pp_size > 1.

        Parameters
        ----------
        micro_batches : list of Tensor
            Input micro-batches for the first stage.  Ignored on non-first
            stages: they receive activations from their predecessor, and the
            micro-batch count is passed down from the first stage.
        loss_fn : callable, optional
            Loss applied on the last stage.  Signature: ``loss_fn(output) → scalar``.
            Defaults to ``output.sum()``.

        Returns
        -------
        outputs : list of Tensor
            Model outputs (only meaningful on the last stage).
        losses : list of Tensor
            Per-micro-batch scalar losses (only on last stage; empty otherwise).

        Algorithm
        ---------
        Three phases — Warmup (p-1 forwards), Steady-state (1f+1b per clock),
        Drain (remaining backwards) — matching the Megatron-LM 1F1B schedule.
        """
        if not self._multi_process:
            raise RuntimeError(
                "run_1f1b_distributed requires pp_size > 1 and dist.is_initialized(). "
                "Use run_1f1b() for single-process scheduling."
            )

        p = self.num_stages
        m = self._exchange_micro_batch_count(len(micro_batches))
        if loss_fn is None:
            loss_fn = lambda out: out.sum()  # noqa: E731

        fwd_inputs: Dict[int, torch.Tensor] = {}
        fwd_outputs: Dict[int, torch.Tensor] = {}
        losses: List[torch.Tensor] = []
        outputs: List[torch.Tensor] = []

        device = self.topology.device if self.topology else torch.device("cpu")

        def _do_forward(mb_idx: int) -> torch.Tensor:
            if self.is_first:
                x = micro_batches[mb_idx]
            else:
                x = self._recv_activation(device=device, mb_index=mb_idx)
            if not self.is_first:
                # A received activation becomes a leaf so its gradient can be
                # sent back; stash the leaf itself, whose .grad backward fills.
                x = x.detach().requires_grad_(x.is_floating_point())
            fwd_inputs[mb_idx] = x
            out = self.module(x) if self.module is not None else x
            fwd_outputs[mb_idx] = out
            if not self.is_last:
                self._send_activation(out, mb_index=mb_idx)
            else:
                loss = loss_fn(out)
                losses.append(loss)
                outputs.append(out)
            return out

        def _send_input_grad(mb_idx: int, inp: Optional[torch.Tensor]) -> None:
            # Every non-first stage must answer each micro-batch with a
            # gradient, or its predecessor blocks in _recv_gradient.
            if self.is_first or inp is None:
                return
            grad = inp.grad if inp.grad is not None else torch.zeros_like(inp)
            self._send_gradient(grad, mb_index=mb_idx)

        def _do_backward(mb_idx: int) -> None:
            out = fwd_outputs.pop(mb_idx)
            inp = fwd_inputs.pop(mb_idx, None)
            if self.is_last:
                # Forwards run in micro-batch order, so losses[i] is micro-batch i.
                loss = losses[mb_idx]
                if loss.requires_grad:
                    loss.backward(retain_graph=False)
            else:
                grad_out = self._recv_gradient(
                    shape=out.shape, dtype=out.dtype, device=device, mb_index=mb_idx
                )
                if out.requires_grad:
                    out.backward(grad_out)
            _send_input_grad(mb_idx, inp)

        # Phase 1: Warmup — issue min(p-1, m) forwards.
        warmup_count = min(p - 1, m)
        for t in range(warmup_count):
            _do_forward(t)

        # Phase 2: Steady-state — one forward + one backward per clock.
        for t in range(m - warmup_count):
            fwd_idx = t + warmup_count
            if fwd_idx < m:
                _do_forward(fwd_idx)
            _do_backward(t)

        # Phase 3: Drain — remaining backwards.
        for t in range(m - warmup_count, m):
            _do_backward(t)

        return outputs, losses
