"""
Neural Echo V2 — Compressed Parameter Memory for Continual Learning.
Pure PyTorch. No external dependencies.
"""

from .echo_memory import NeuralEchoV2
from .quantization import QuantizedSnapshot

__version__ = "2.0.0"
__all__ = ["NeuralEchoV2", "QuantizedSnapshot"]