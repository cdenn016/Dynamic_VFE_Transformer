"""
Analysis Plotting Modules
==========================

Specialized plotting functions for different types of analysis:
- fields: Spatial field visualizations (mu, Sigma, phi)
- support: Support and overlap analysis
- softmax: Softmax weights visualization
- mu_tracking: Mu center tracking over time
"""

import numpy as np


def get_spatial_shape_from_system(system):
    """Infer spatial shape from first agent's support or base_manifold."""
    a0 = system.agents[0]
    if hasattr(a0, "support") and a0.support is not None:
        shape = getattr(a0.support, "base_shape", None)
        if shape is None:
            shape = getattr(a0.support, "mask", np.array([])).shape
        return tuple(shape)
    if hasattr(a0, "base_manifold"):
        return tuple(a0.base_manifold.shape)
    if hasattr(a0, "support_shape"):
        return tuple(a0.support_shape)
    return ()


__all__ = ['get_spatial_shape_from_system']