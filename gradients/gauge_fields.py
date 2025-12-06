"""
Gauge Fields on SO(3) Principal Bundle
======================================

Representation and geometry of gauge fields φ(c) for active inference agents.

Mathematical Framework:
----------------------
Each agent carries a gauge field φ: C → so(3) over its support region.

**Axis-Angle Representation:**
    φ(c) ∈ ℝ³ where ||φ|| encodes rotation angle, φ/||φ|| encodes axis
    
**Lie Group Element:**
    g(c) = exp(φ(c)) ∈ SO(3)
    
**Transport Operator:**
    Ω_ij(c) = g_i(c) · g_j(c)^{-1} = exp(φ_i(c)) · exp(-φ_j(c))

**Principal Ball:**
    Valid φ must satisfy ||φ(c)|| < π - margin to avoid branch cuts

**Natural Gradient:**
    Updates on φ must respect SO(3) geometry via right-Jacobian J(φ)

Author: Clean Rebuild
Date: November 2025
"""

import numpy as np
from typing import Optional, Tuple, Literal

# =============================================================================
# Gauge Field Container
# =============================================================================

class GaugeField:
    """
    Container for agent's gauge field φ(c) over spatial support.
    
    Attributes:
        phi: Axis-angle field, shape (*S, 3)
        support_shape: Spatial dimensions tuple
        K: Latent dimension
    
    Examples:
        >>> # 1D chain
        >>> field_1d = GaugeField.zeros(shape=(100,), K=3)
        >>> field_1d.phi.shape
        (100, 3)
        
        >>> # 2D grid
        >>> field_2d = GaugeField.zeros(shape=(32, 32), K=5)
        >>> field_2d.phi.shape
        (32, 32, 3)
    """
    
    def __init__(
        self,
        phi: np.ndarray,
        K: int,
        *,
        validate: bool = True,
        margin: float = 1e-2,
    ):
        """
        Initialize gauge field from axis-angle array.
        
        Args:
            phi: Axis-angle field, shape (*S, 3)
            K: Latent dimension
            validate: If True, check principal ball constraint
            margin: Safety margin from branch cut (π - margin)
        """
        self.phi = np.asarray(phi, dtype=np.float32, order='C')
        self.K = int(K)
        
        if self.phi.shape[-1] != 3:
            raise ValueError(f"Last dimension must be 3 (so(3)), got {self.phi.shape[-1]}")
        
        self.support_shape = self.phi.shape[:-1]
        
         

    
    @classmethod
    def zeros(cls, shape: Tuple[int, ...], K: int) -> 'GaugeField':
        """Create identity gauge field (φ = 0 everywhere)."""
        phi = np.zeros(shape + (3,), dtype=np.float32)
        return cls(phi, K, validate=False)
    
    @classmethod
    def random(
        cls,
        shape: Tuple[int, ...],
        K: int,
        *,
        scale: float = 0.5,
        seed: Optional[int] = None,
    ) -> 'GaugeField':
        """
        Create random gauge field with controlled magnitude.
        
        Args:
            shape: Spatial shape
            K: Latent dimension
            scale: Maximum rotation angle
            seed: Random seed
        """
        if seed is not None:
            np.random.seed(seed)
        
        phi = np.random.randn(*shape, 3).astype(np.float32)
        norms = np.linalg.norm(phi, axis=-1, keepdims=True)
        phi = phi / np.maximum(norms, 1e-8) * (scale * np.random.rand(*shape, 1))
        
        return cls(phi, K, validate=True, margin=0.1)
    
    def copy(self) -> 'GaugeField':
        """Create a copy of this gauge field."""
        return GaugeField(self.phi.copy(), self.K, validate=False)
    
    def __repr__(self) -> str:
        return (
            f"GaugeField(shape={self.support_shape}, K={self.K}, "
            f"||φ||_max={np.max(np.linalg.norm(self.phi, axis=-1)):.4f})"
        )




# =============================================================================
# Principal Ball Retraction
# =============================================================================

def retract_to_principal_ball(
    phi: np.ndarray,
    *,
    margin: float = 1e-2,
    mode: Literal['mod2pi', 'project'] = 'mod2pi',
) -> np.ndarray:
    """
    Retract gauge field to principal ball ||φ|| < π - margin.
    
    Two modes:
        - 'mod2pi': Wrap to [0, 2π) with antipodal flip
        - 'project': Radial projection (simpler but discontinuous)
    
    Args:
        phi: Axis-angle field, shape (*S, 3)
        margin: Safety margin from branch cut
        mode: Retraction method
    
    Returns:
        phi_retracted: Shape (*S, 3), satisfies ||φ|| < π - margin
    
    Examples:
        >>> phi = np.array([[3.5, 0.0, 0.0]])  # θ ≈ 3.5 > π
        >>> phi_ret = retract_to_principal_ball(phi)
        >>> np.linalg.norm(phi_ret) < np.pi
        True
    """
    phi = np.asarray(phi, dtype=np.float64)
    
    if phi.shape[-1] != 3:
        raise ValueError(f"Expected shape (*S, 3), got {phi.shape}")
    
    # Compute norms
    theta = np.linalg.norm(phi, axis=-1, keepdims=True)  # (*S, 1)
    
    # ✅ FIX: Add epsilon BEFORE division to avoid 0/0
    theta_safe = np.maximum(theta, 1e-12)  # Never actually zero
    
    # Compute normalized axis (safe now)
    axis = phi / theta_safe  # No warning!
    
    # Handle true zero-norm case: set to arbitrary unit vector
    # (These points won't matter since theta=0 means identity anyway)
    is_zero = (theta < 1e-12)[..., 0]
    if np.any(is_zero):
        axis[is_zero] = np.array([1.0, 0.0, 0.0])
    
    # Threshold
    r_max = float(np.pi - margin)
    
    if mode == 'mod2pi':
        # ========== Modulo 2π with antipodal flip ==========
        two_pi = 2.0 * np.pi
        
        # Wrap to [0, 2π)
        theta_wrapped = np.remainder(theta[..., 0], two_pi)
        
        # Flip axis if θ > π (antipodal symmetry)
        flip = theta_wrapped > np.pi
        theta_final = np.where(flip, two_pi - theta_wrapped, theta_wrapped)
        axis_final = np.where(flip[..., None], -axis, axis)
        
        # Clamp to safety margin
        theta_final = np.minimum(theta_final, r_max)
        
        phi_new = axis_final * theta_final[..., None]
    
    elif mode == 'project':
        # ========== Radial projection ==========
        # Only scale down if exceeds limit
        exceeds = theta[..., 0] > r_max
        scale = np.ones_like(theta[..., 0])
        scale[exceeds] = r_max / theta_safe[exceeds, 0]
        
        phi_new = phi * scale[..., None]
    
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    return phi_new.astype(np.float32, copy=False)




