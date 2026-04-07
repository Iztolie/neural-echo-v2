"""
Neural Echo V2 — Compressed Parameter Memory for Continual Learning.

Pure PyTorch implementation. No external dependencies.

Stores periodic snapshots of model parameters in INT8 format (4x compression)
and blends them with current parameters during training as a regularization
mechanism for continual learning.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional
from collections import deque
from .quantization import QuantizedSnapshot


class NeuralEchoV2:
    """
    Parameter snapshot memory with INT8 compression for continual learning.

    Periodically stores compressed snapshots of model parameters and blends
    them with current parameters to regularize training. Achieves 4x memory
    reduction via per-tensor affine quantization (FP32 → UINT8).

    Args:
        use_compression: If True, store snapshots as INT8 (4x less RAM).
        max_snapshots: Maximum number of historical snapshots to retain (FIFO).
        blend_weight: Weight for current parameters during blending.
            Historical average gets (1 - blend_weight).
        snapshot_interval: Store a snapshot every N training steps.

    Example::

        echo = NeuralEchoV2(use_compression=True, max_snapshots=10)
        for step in range(num_steps):
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            echo.train_step(model, loss, optimizer)
    """

    def __init__(
        self,
        use_compression: bool = True,
        max_snapshots: int = 10,
        blend_weight: float = 0.8,
        snapshot_interval: int = 10,
    ):
        self.use_compression = use_compression
        self.blend_weight = blend_weight
        self.snapshot_interval = snapshot_interval
        self.step_count = 0
        self.max_snapshots = max_snapshots
        self._snapshots: deque = deque(maxlen=max_snapshots)
        self._model_size = 0

    def store_snapshot(self, model: nn.Module, metadata=None):
        """Store a parameter snapshot, optionally compressed to INT8."""
        params = {n: p.data.clone().detach() for n, p in model.named_parameters()}
        self._model_size = sum(p.numel() for p in params.values())

        if self.use_compression:
            self._snapshots.append(QuantizedSnapshot(params))
        else:
            self._snapshots.append(params)

    def interpolate_parameters(self, model: nn.Module) -> None:
        """Blend current parameters with the historical snapshot average."""
        if not self._snapshots:
            return

        snapshots = []
        for snap in self._snapshots:
            if isinstance(snap, QuantizedSnapshot):
                snapshots.append(snap.dequantize())
            else:
                snapshots.append(snap)

        with torch.no_grad():
            for name, param in model.named_parameters():
                blended = param.data * self.blend_weight
                hist_weight = (1 - self.blend_weight) / len(snapshots)
                for snapshot in snapshots:
                    if name in snapshot:
                        blended += snapshot[name] * hist_weight
                param.data.copy_(blended)

    def train_step(self, model: nn.Module, loss: torch.Tensor, optimizer):
        """Execute one training step with parameter memory."""
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if self.step_count % self.snapshot_interval == 0:
            self.store_snapshot(model)
            self.interpolate_parameters(model)

        self.step_count += 1

    def get_stats(self) -> Dict:
        """Return memory and compression statistics."""
        stats = {
            'num_snapshots': len(self._snapshots),
            'blend_weight': self.blend_weight,
            'snapshot_interval': self.snapshot_interval,
            'total_steps': self.step_count,
            'compression_ratio': 1.0,
            'memory_saved_mb': 0.0,
            'actual_ram_bytes': 0,
            'uncompressed_ram_bytes': 0,
        }

        for snap in self._snapshots:
            if isinstance(snap, QuantizedSnapshot):
                stats['actual_ram_bytes'] += snap.memory_bytes()
                stats['uncompressed_ram_bytes'] += snap.uncompressed_bytes()
            else:
                nbytes = sum(p.nelement() * p.element_size() for p in snap.values())
                stats['actual_ram_bytes'] += nbytes
                stats['uncompressed_ram_bytes'] += nbytes

        if stats['actual_ram_bytes'] > 0:
            stats['compression_ratio'] = (
                stats['uncompressed_ram_bytes'] / stats['actual_ram_bytes']
            )
        stats['memory_saved_mb'] = (
            (stats['uncompressed_ram_bytes'] - stats['actual_ram_bytes'])
            / (1024 * 1024)
        )

        return stats
    
    def close(self):
        """Clean up resources"""
        if self.use_compression:
            self.memory.close()