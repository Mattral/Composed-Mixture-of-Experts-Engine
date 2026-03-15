"""
tests/test_tensor_parallel.py
=============================

Unit tests for tensor parallelism layers (ColumnParallel, RowParallel).
"""

import math

import pytest
import torch
import torch.nn as nn

from pkg.distributed.parallel_mesh import ColumnParallelLinear, RowParallelLinear


def test_column_parallel_forward_shapes():
    """Verify ColumnParallel linear forward produces correct shapes."""
    batch_size, in_features, out_features = 32, 128, 256
    
    layer = ColumnParallelLinear(in_features, out_features, bias=True)
    x = torch.randn(batch_size, in_features)
    
    y = layer(x)
    
    assert y.shape == (batch_size, out_features)
    assert not torch.isnan(y).any()


def test_column_parallel_backward():
    """Verify gradients flow correctly through ColumnParallel."""
    batch_size, in_features, out_features = 16, 64, 128
    
    layer = ColumnParallelLinear(in_features, out_features, bias=True)
    x = torch.randn(batch_size, in_features, requires_grad=True)
    
    y = layer(x)
    loss = y.sum()
    loss.backward()
    
    assert x.grad is not None
    assert layer.weight.grad is not None
    assert layer.bias.grad is not None
    assert x.grad.shape == x.shape
    assert layer.weight.grad.shape == layer.weight.shape
    assert layer.bias.grad.shape == layer.bias.shape


def test_column_parallel_no_bias():
    """Verify ColumnParallel works correctly without bias."""
    batch_size, in_features, out_features = 32, 128, 256
    
    layer = ColumnParallelLinear(in_features, out_features, bias=False)
    assert layer.bias is None
    
    x = torch.randn(batch_size, in_features)
    y = layer(x)
    
    assert y.shape == (batch_size, out_features)


def test_row_parallel_forward_shapes():
    """Verify RowParallel linear forward produces correct shapes."""
    batch_size, in_features, out_features = 32, 128, 256
    
    layer = RowParallelLinear(in_features, out_features, bias=True)
    x = torch.randn(batch_size, in_features)
    
    y = layer(x)
    
    assert y.shape == (batch_size, out_features)
    assert not torch.isnan(y).any()


def test_row_parallel_backward():
    """Verify gradients flow correctly through RowParallel."""
    batch_size, in_features, out_features = 16, 64, 128
    
    layer = RowParallelLinear(in_features, out_features, bias=True)
    x = torch.randn(batch_size, in_features, requires_grad=True)
    
    y = layer(x)
    loss = y.sum()
    loss.backward()
    
    assert x.grad is not None
    assert layer.weight.grad is not None
    assert layer.bias.grad is not None


def test_row_parallel_no_bias():
    """Verify RowParallel works correctly without bias."""
    batch_size, in_features, out_features = 32, 128, 256
    
    layer = RowParallelLinear(in_features, out_features, bias=False)
    assert layer.bias is None
    
    x = torch.randn(batch_size, in_features)
    y = layer(x)
    
    assert y.shape == (batch_size, out_features)


def test_tensor_parallel_layers_numerically_correct():
    """Verify TP layers produce same output as standard nn.Linear (no actual sharding yet)."""
    batch_size, in_features, out_features = 16, 64, 128
    
    # Create standard linear layer
    std_layer = nn.Linear(in_features, out_features, bias=True)
    
    # Create TP layers
    col_layer = ColumnParallelLinear(in_features, out_features, bias=True)
    row_layer = RowParallelLinear(in_features, out_features, bias=True)
    
    # Copy weights for deterministic comparison
    col_layer.weight.data.copy_(std_layer.weight.data)
    col_layer.bias.data.copy_(std_layer.bias.data)
    row_layer.weight.data.copy_(std_layer.weight.data)
    row_layer.bias.data.copy_(std_layer.bias.data)
    
    x = torch.randn(batch_size, in_features)
    
    with torch.no_grad():
        y_std = std_layer(x)
        y_col = col_layer(x)
        y_row = row_layer(x)
    
    # Since TP layers have no actual parallelism wired in yet, they should
    # be identical to standard linear layers
    assert torch.allclose(y_col, y_std, atol=1e-6, rtol=1e-6)
    assert torch.allclose(y_row, y_std, atol=1e-6, rtol=1e-6)


def test_column_parallel_dtype_preservation():
    """Verify ColumnParallel preserves dtype correctly."""
    batch_size, in_features, out_features = 16, 64, 128
    
    for dtype in [torch.float32, torch.float64]:
        layer = ColumnParallelLinear(in_features, out_features, dtype=dtype)
        x = torch.randn(batch_size, in_features, dtype=dtype)
        
        y = layer(x)
        
        assert y.dtype == dtype


def test_row_parallel_dtype_preservation():
    """Verify RowParallel preserves dtype correctly."""
    batch_size, in_features, out_features = 16, 64, 128
    
    for dtype in [torch.float32, torch.float64]:
        layer = RowParallelLinear(in_features, out_features, dtype=dtype)
        x = torch.randn(batch_size, in_features, dtype=dtype)
        
        y = layer(x)
        
        assert y.dtype == dtype


def test_sequence_parallel_scatter_single_rank():
    """Verify scatter_to_sequence_parallel no-op on single TP rank."""
    from pkg.distributed.parallel_mesh import (
        scatter_to_sequence_parallel,
        build_topology,
    )
    
    batch_size, seq_len, hidden_dim = 4, 128, 64
    x = torch.randn(batch_size, seq_len, hidden_dim)
    
    # Create single-rank topology (tp_size=1)
    topo = build_topology(dp_size=1, tp_size=1, ep_size=1, device_type="cpu")
    
    x_scattered = scatter_to_sequence_parallel(x, topo)
    
    assert x_scattered.shape == x.shape
    assert torch.equal(x_scattered, x)


def test_sequence_parallel_gather_single_rank():
    """Verify gather_from_sequence_parallel no-op on single TP rank."""
    from pkg.distributed.parallel_mesh import (
        gather_from_sequence_parallel,
        build_topology,
    )
    
    batch_size, seq_len, hidden_dim = 4, 128, 64
    x = torch.randn(batch_size, seq_len, hidden_dim)
    
    # Create single-rank topology (tp_size=1)
    topo = build_topology(dp_size=1, tp_size=1, ep_size=1, device_type="cpu")
    
    x_gathered = gather_from_sequence_parallel(x, topo)
    
    assert x_gathered.shape == x.shape
    assert torch.equal(x_gathered, x)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
