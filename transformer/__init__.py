# -*- coding: utf-8 -*-
"""
Gauge-Theoretic Transformer Package
====================================

Implements Hamiltonian dynamics on SPD manifolds for transformer architectures.
"""

# Suppress noisy Triton warnings about missing CUDA binaries on Windows
# These occur because Triton looks for cuobjdump.exe and nvdisasm.exe
# which are only in the CUDA Toolkit (not required for PyTorch GPU usage)
import warnings
warnings.filterwarnings(
    "ignore",
    message="Failed to find cuobjdump",
    category=UserWarning,
    module="triton"
)
warnings.filterwarnings(
    "ignore",
    message="Failed to find nvdisasm",
    category=UserWarning,
    module="triton"
)

from .model import GaugeTransformerLM
from .train import Trainer, TrainingConfig
from .data import (
    create_dataloaders,
    create_char_dataloaders,
    create_byte_dataloaders,
)

__all__ = [
    'GaugeTransformerLM',
    'Trainer',
    'TrainingConfig',
    'create_dataloaders',
    'create_char_dataloaders',
    'create_byte_dataloaders',
]
