# -*- coding: utf-8 -*-
"""
Created on Sat Nov  8 21:02:11 2025

@author: chris and christine
"""

# -*- coding: utf-8 -*-
"""
General Covariance Field Initialization
========================================

Tools for creating spatially varying, full covariance matrices Σ(c) that are:
- Positive-definite everywhere
- Spatially smooth (optional)
- Expressive (not just diagonal!)

Mathematical Framework:
----------------------
Covariance fields Σ: C → Sym⁺⁺(K) live on the manifold of positive-definite matrices.

Parametrizations:
1. **Cholesky**: Σ = L Lᵀ where L is lower-triangular with positive diagonal
2. **Low-rank + diagonal**: Σ = UUᵀ + D where U ∈ ℝᴷˣʳ, D diagonal
3. **Random SPD**: Generate via eigendecomposition with positive eigenvalues
4. **Smooth interpolation**: Spatially smooth variations via Gaussian processes

Author: Chris & Christine
Date: November 2025
"""

import numpy as np
from typing import Tuple, Optional, Literal, Callable
from scipy.ndimage import gaussian_filter
from agent.masking import SupportRegionSmooth


# =============================================================================
# Agent Integration
# =============================================================================



class CovarianceFieldInitializer:
    """
    Factory for initializing agent covariance fields with various strategies.
    This version automatically uses an agent's base manifold geometry.
    """

    def __init__(
        self,
        strategy: Literal[
            "constant", "random", "smooth", "gradient", "center", "random_centers"
        ] = "smooth",
        **kwargs,
    ):
        self.strategy = strategy
        self.kwargs = kwargs

    def generate_for_agent(
        self,
        agent,
        *,
        scale: float = 1.0,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """
        Geometry-aware covariance generator: uses agent.base_manifold.shape.

        Args:
            agent: The Agent instance (must have base_manifold and config.K)
            scale: Overall scale factor
            rng: Random generator
        """
        spatial_shape = agent.base_manifold.shape
        K = agent.config.K

        return self.generate(spatial_shape, K, scale=scale, rng=rng)

    # --- legacy internal API kept for backward compatibility ---------------
    def generate(
        self,
        spatial_shape: Tuple[int, ...],
        K: int,
        *,
        scale: float = 1.0,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """Generate covariance field given explicit spatial_shape."""
        if self.strategy == "constant":
            return generate_constant_field(spatial_shape, K, scale=scale, rng=rng)

        elif self.strategy == "random":
            return generate_random_spd_field(
                spatial_shape, K, scale=scale, rng=rng, **self.kwargs
            )

        elif self.strategy == "smooth":
            return generate_smooth_spd_field(
                spatial_shape, K, scale=scale, rng=rng, **self.kwargs
            )

        elif self.strategy in ["gradient", "center", "random_centers"]:
            return generate_structured_field(
                spatial_shape,
                K,
                structure_type=self.strategy,
                base_scale=scale,
                rng=rng,
                **self.kwargs,
            )

        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")


    


    @staticmethod
    def enforce_covariance_field(
        Sigma_raw: np.ndarray,
        support: 'SupportRegionSmooth',
        inside_scale: float = 1.0,
        outside_scale: float = 1e3,
        use_smooth_transition: bool = True
    ) -> np.ndarray:
        """
        Enforce covariance support constraints.
        
        Args:
            Sigma_raw: Raw covariance field, shape (*S, K, K)
            support: Support region with continuous mask
            inside_scale: Scale for covariance inside support
            outside_scale: Scale for covariance outside support (large!)
            use_smooth_transition: If True, smooth interpolation around threshold
        
        Behavior:
            MIN_MASK_FOR_NORMAL_COV controls WHERE the transition happens.
            use_smooth_transition controls HOW (sharp vs smooth).
            
            Without smooth transition:
                χ > threshold → normal Σ
                χ ≤ threshold → large Σ (sharp jump)
            
            With smooth transition:
                χ >> threshold → normal Σ
                χ around threshold → smooth interpolation
                χ << threshold → large Σ
        """
        spatial_shape = Sigma_raw.shape[:-2]
        K = Sigma_raw.shape[-1]
        
        # Get continuous mask [0, 1]
        chi = support.mask_continuous  # (*S,)
        threshold = support.config.min_mask_for_normal_cov  # e.g., 0.1
        
        # Define inside and outside covariances
        Sigma_inside = Sigma_raw
        Sigma_outside = outside_scale * np.eye(K, dtype=Sigma_raw.dtype)
        
        if not use_smooth_transition:
            # ========== SHARP TRANSITION AT THRESHOLD ==========
            inside_mask = chi > threshold
            
            Sigma_enforced = np.zeros_like(Sigma_raw)
            Sigma_enforced[inside_mask] = Sigma_raw[inside_mask]
            
            outside_mask = ~inside_mask
            Sigma_enforced[outside_mask] = Sigma_outside
        
        else:
            # ========== SMOOTH TRANSITION AROUND THRESHOLD ==========
            # Create smooth interpolation zone centered on threshold
            
            # Define transition width (as fraction of threshold)
            transition_width = 0.5 * threshold  # Adjustable!
            
            # Compute normalized position in transition zone
            # t = 0 when χ = threshold - width  (fully outside)
            # t = 1 when χ = threshold + width  (fully inside)
            lower_bound = threshold - transition_width
            upper_bound = threshold + transition_width
            
            t = (chi - lower_bound) / (2 * transition_width)
            t = np.clip(t, 0, 1)  # Clamp to [0, 1]
            
            # Apply smooth S-curve (smoothstep function)
            # This gives C¹ continuity (smooth first derivative)
            alpha = 3 * t**2 - 2 * t**3
            
            # For even smoother (C² continuity), use smootherstep:
            # alpha = 6 * t**5 - 15 * t**4 + 10 * t**3
            
            # Broadcast alpha to matrix dimensions
            alpha = alpha[..., None, None]  # (*S, 1, 1)
            
            # Interpolate: Σ(c) = α · Σ_inside + (1-α) · Σ_outside
            Sigma_enforced = (
                alpha * Sigma_inside +
                (1 - alpha) * Sigma_outside
            )
            
            # Ensure symmetry
            Sigma_enforced = 0.5 * (Sigma_enforced + np.swapaxes(Sigma_enforced, -1, -2))
        
        return Sigma_enforced.astype(np.float32)



# =============================================================================
# Core SPD Matrix Generation
# =============================================================================

def generate_constant_field_safe(
    spatial_shape: Tuple[int, ...],
    K: int,
    *,
    scale: float = 1.0,
    min_eigenvalue: float = 0.1,  # CRITICAL: prevent small eigenvalues
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Generate constant covariance field with guaranteed conditioning.
    
    For 0D agents, this is CRITICAL to avoid blow-up!
    """
    if rng is None:
        rng = np.random.default_rng()
    
    # Create well-conditioned base covariance
    if K == 1:
        # Scalar case
        Sigma_base = np.array([[scale]], dtype=np.float32)
    else:
        # Generate random SPD with controlled eigenvalues
        eigs = rng.uniform(min_eigenvalue, scale, size=K)
        eigs = np.sort(eigs)[::-1]  # Descending
        
        # Random rotation
        Q = _random_orthogonal(K, rng=rng)
        
        # Σ = Q Λ Qᵀ
        Sigma_base = Q @ np.diag(eigs) @ Q.T
        Sigma_base = 0.5 * (Sigma_base + Sigma_base.T)  # Ensure symmetry
    
    # Tile to spatial shape
    if len(spatial_shape) == 0:
        return Sigma_base.astype(np.float32)
    else:
        return np.tile(Sigma_base, (*spatial_shape, 1, 1)).astype(np.float32)




def generate_random_spd(
    K: int,
    *,
    min_eigenvalue: float = 0.1,
    max_eigenvalue: float = 10.0,
    scale: float = 1.0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Generate a single random positive-definite matrix.
    
    Method: Random rotation + specified eigenvalue spectrum.
    
    Args:
        K: Matrix dimension
        min_eigenvalue: Minimum eigenvalue (controls condition number)
        max_eigenvalue: Maximum eigenvalue
        scale: Overall scale factor
        rng: Random generator
    
    Returns:
        Sigma: Positive-definite matrix, shape (K, K)
    
    Examples:
        >>> Sigma = generate_random_spd(3, min_eigenvalue=0.5, max_eigenvalue=5.0)
        >>> eigs = np.linalg.eigvalsh(Sigma)
        >>> print(f"Eigenvalues: {eigs}")  # All positive, in [0.5, 5.0]
    """
    if rng is None:
        rng = np.random.default_rng()
    
    # Random eigenvalues
    eigs = rng.uniform(min_eigenvalue, max_eigenvalue, size=K)
    eigs = scale * np.sort(eigs)[::-1]  # Descending order
    
    # Random orthogonal matrix (rotation)
    Q = _random_orthogonal(K, rng=rng)
    
    # Σ = Q Λ Qᵀ
    Sigma = Q @ np.diag(eigs) @ Q.T
    
    # Ensure symmetry (numerical)
    Sigma = 0.5 * (Sigma + Sigma.T)
    
    return Sigma.astype(np.float32)


def cholesky_to_covariance(L: np.ndarray) -> np.ndarray:
    """
    Convert Cholesky factor L to covariance Σ = L Lᵀ.
    
    Args:
        L: Lower-triangular matrix with positive diagonal, shape (*S, K, K)
    
    Returns:
        Sigma: Covariance matrix, shape (*S, K, K)
    """
    return np.einsum('...ij,...kj->...ik', L, L, optimize=True).astype(np.float32)


def covariance_to_cholesky(Sigma: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Compute Cholesky factor L such that Σ = L Lᵀ.
    
    Args:
        Sigma: Covariance matrix, shape (*S, K, K)
        eps: Regularization for numerical stability
    
    Returns:
        L: Lower-triangular Cholesky factor, shape (*S, K, K)
    """
    # Regularize for stability
    Sigma = Sigma + eps * np.eye(Sigma.shape[-1])
    
    # Compute Cholesky
    try:
        L = np.linalg.cholesky(Sigma)
    except np.linalg.LinAlgError:
        # Fallback: eigendecomposition
        eigs, V = np.linalg.eigh(Sigma)
        eigs = np.maximum(eigs, eps)
        L = V @ np.diag(np.sqrt(eigs))
    
    return L.astype(np.float32)


# =============================================================================
# Spatial Covariance Field Generation
# =============================================================================

def generate_constant_field(
    spatial_shape: Tuple[int, ...],
    K: int,
    *,
    Sigma_base: Optional[np.ndarray] = None,
    scale: float = 1.0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Generate spatially constant covariance field (baseline).
    
    Args:
        spatial_shape: Shape of base manifold, e.g., (100,) for 1D, (32, 32) for 2D
        K: Latent dimension
        Sigma_base: Base covariance (if None, generate random SPD)
        scale: Scale factor
        rng: Random generator
    
    Returns:
        Sigma_field: Shape (*spatial_shape, K, K)
    """
    if Sigma_base is None:
        Sigma_base = generate_random_spd(K, scale=scale, rng=rng)
    
    return np.tile(Sigma_base, (*spatial_shape, 1, 1)).astype(np.float32)


def generate_random_spd_field(
    spatial_shape: Tuple[int, ...],
    K: int,
    *,
    min_eigenvalue: float = 0.1,
    max_eigenvalue: float = 10.0,
    scale: float = 1.0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Generate spatially varying random SPD field (independent at each point).
    
    Args:
        spatial_shape: Spatial dimensions
        K: Latent dimension
        min_eigenvalue: Minimum eigenvalue across all points
        max_eigenvalue: Maximum eigenvalue across all points
        scale: Overall scale
        rng: Random generator
    
    Returns:
        Sigma_field: Shape (*spatial_shape, K, K)
    
    Examples:
        >>> # 1D chain with 100 points, K=3
        >>> Sigma_field = generate_random_spd_field((100,), 3)
        >>> Sigma_field.shape
        (100, 3, 3)
        >>> 
        >>> # Check all positive-definite
        >>> for i in range(100):
        >>>     eigs = np.linalg.eigvalsh(Sigma_field[i])
        >>>     assert np.all(eigs > 0)
    """
    if rng is None:
        rng = np.random.default_rng()
    
    n_points = int(np.prod(spatial_shape))
    
    # Generate random SPD matrix at each point
    Sigma_flat = np.zeros((n_points, K, K), dtype=np.float32)
    for i in range(n_points):
        Sigma_flat[i] = generate_random_spd(
            K, 
            min_eigenvalue=min_eigenvalue,
            max_eigenvalue=max_eigenvalue,
            scale=scale,
            rng=rng
        )
    
    # Reshape to spatial
    Sigma_field = Sigma_flat.reshape(*spatial_shape, K, K)
    
    return Sigma_field


def generate_smooth_spd_field(
    spatial_shape: Tuple[int, ...],
    K: int,
    *,
    smoothness_scale: float = 5.0,
    min_eigenvalue: float = 0.1,
    scale: float = 1.0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Generate smooth SPD field with gentler trace control.
    
    Instead of pointwise renormalization (which destroys smoothness),
    apply a SMOOTH trace correction field.
    """
    if rng is None:
        rng = np.random.default_rng()
    
    # Generate and smooth as before
    Sigma_random = generate_random_spd_field(
        spatial_shape, K,
        min_eigenvalue=min_eigenvalue,
        max_eigenvalue=scale,
        scale=scale,
        rng=rng
    )
    
    L_field = covariance_to_cholesky(Sigma_random)
    
    L_smooth = np.zeros_like(L_field)
    for i in range(K):
        for j in range(K):
            L_smooth[..., i, j] = gaussian_filter(
                L_field[..., i, j],
                sigma=smoothness_scale,
                mode='wrap'
            )
    
    # Ensure triangular and positive diagonal
    for i in range(K):
        for j in range(i+1, K):
            L_smooth[..., i, j] = 0
        L_smooth[..., i, i] = np.abs(L_smooth[..., i, i]) + min_eigenvalue
    
    Sigma_smooth = cholesky_to_covariance(L_smooth)
    
    # GENTLE correction: Smooth the scaling field itself!
    spatial_shape_actual = Sigma_smooth.shape[:-2]
    
    # Compute trace at each point
    trace_field = np.zeros(spatial_shape_actual)
    for idx in np.ndindex(spatial_shape_actual):
        trace_field[idx] = np.trace(Sigma_smooth[idx])
    
    # Target trace (could be constant or varying)
    target_trace = K * scale
    
    # Compute scaling factor field
    scale_field = target_trace / (trace_field + 1e-8)
    
    # SMOOTH the scaling field (this preserves spatial smoothness!)
    scale_field_smooth = gaussian_filter(scale_field, sigma=smoothness_scale, mode='wrap')
    
    # Apply smooth scaling
    for idx in np.ndindex(spatial_shape_actual):
        Sigma_smooth[idx] *= scale_field_smooth[idx]
    
    return Sigma_smooth


def generate_structured_field(
    spatial_shape: Tuple[int, ...],
    K: int,
    *,
    structure_type: Literal['gradient', 'center', 'random_centers'] = 'gradient',
    base_scale: float = 1.0,
    variation_scale: float = 5.0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Generate covariance field with specific spatial structure.
    
    Structures:
    - 'gradient': Eigenvalues vary smoothly along one axis
    - 'center': Covariance increases toward center of domain
    - 'random_centers': Multiple centers with different covariances
    
    Args:
        spatial_shape: Spatial dimensions
        K: Latent dimension
        structure_type: Type of spatial structure
        base_scale: Base covariance scale
        variation_scale: Amount of spatial variation
        rng: Random generator
    
    Returns:
        Sigma_field: Structured SPD field, shape (*spatial_shape, K, K)
    """
    if rng is None:
        rng = np.random.default_rng()
    
    ndim = len(spatial_shape)
    
    if structure_type == 'gradient':
        # Eigenvalues vary along first axis
        return _generate_gradient_field(
            spatial_shape, K, base_scale, variation_scale, rng
        )
    
    elif structure_type == 'center':
        # Covariance increases toward center
        return _generate_center_field(
            spatial_shape, K, base_scale, variation_scale, rng
        )
    
    elif structure_type == 'random_centers':
        # Multiple random centers with different covariances
        return _generate_random_centers_field(
            spatial_shape, K, base_scale, variation_scale, rng
        )
    
    else:
        raise ValueError(f"Unknown structure_type: {structure_type}")


def _generate_gradient_field(
    spatial_shape: Tuple[int, ...],
    K: int,
    base_scale: float,
    variation_scale: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate field where eigenvalues vary along first axis."""
    
    # Coordinate along first axis: [0, 1]
    coords = np.linspace(0, 1, spatial_shape[0])
    
    # Eigenvalue profile: varies smoothly from base to base*variation
    eig_profile = base_scale * (1 + (variation_scale - 1) * coords)
    
    # Random rotation (constant across space)
    Q = _random_orthogonal(K, rng=rng)
    
    # Build field
    Sigma_field = np.zeros((*spatial_shape, K, K), dtype=np.float32)
    
    for idx in np.ndindex(spatial_shape):
        i = idx[0]  # First axis
        eigs = eig_profile[i] * np.ones(K)
        Sigma_field[idx] = Q @ np.diag(eigs) @ Q.T
    
    return Sigma_field


def _generate_center_field(
    spatial_shape: Tuple[int, ...],
    K: int,
    base_scale: float,
    variation_scale: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate field where covariance increases toward center."""
    
    # Compute distance from center at each point
    center = tuple(s // 2 for s in spatial_shape)
    
    # Distance field
    coords = np.meshgrid(*[np.arange(s) for s in spatial_shape], indexing='ij')
    dist_sq = sum((c - center[i])**2 for i, c in enumerate(coords))
    max_dist_sq = sum((s // 2)**2 for s in spatial_shape)
    
    # Normalized distance: 0 at center, 1 at edges
    dist_normalized = np.sqrt(dist_sq / max_dist_sq)
    
    # Scale profile: large at center, small at edges
    scale_profile = base_scale * (1 + (variation_scale - 1) * (1 - dist_normalized))
    
    # Random rotation (constant)
    Q = _random_orthogonal(K, rng=rng)
    
    # Build field
    Sigma_field = np.zeros((*spatial_shape, K, K), dtype=np.float32)
    
    for idx in np.ndindex(spatial_shape):
        scale = scale_profile[idx]
        eigs = scale * np.ones(K)
        Sigma_field[idx] = Q @ np.diag(eigs) @ Q.T
    
    return Sigma_field


def _generate_random_centers_field(
    spatial_shape: Tuple[int, ...],
    K: int,
    base_scale: float,
    variation_scale: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate field with multiple random centers."""
    
    n_centers = max(3, int(np.prod(spatial_shape) ** 0.3))  # Heuristic
    
    # Random center locations
    centers = [tuple(rng.integers(0, s) for s in spatial_shape) 
               for _ in range(n_centers)]
    
    # Random covariances at each center
    center_covs = [generate_random_spd(K, scale=base_scale * variation_scale, rng=rng)
                   for _ in range(n_centers)]
    
    # Interpolate via distance weighting
    Sigma_field = np.zeros((*spatial_shape, K, K), dtype=np.float32)
    
    coords = np.meshgrid(*[np.arange(s) for s in spatial_shape], indexing='ij')
    
    for idx in np.ndindex(spatial_shape):
        # Distance to each center
        dists = [np.sqrt(sum((coords[i][idx] - c[i])**2 for i in range(len(spatial_shape))))
                 for c in centers]
        
        # Softmax weights (closer = higher weight)
        weights = np.exp(-np.array(dists) / (0.2 * max(spatial_shape)))
        weights = weights / np.sum(weights)
        
        # Weighted average of covariances
        Sigma_avg = sum(w * cov for w, cov in zip(weights, center_covs))
        Sigma_field[idx] = Sigma_avg
    
    return Sigma_field


# =============================================================================
# Utilities
# =============================================================================

def _random_orthogonal(n: int, *, rng: np.random.Generator) -> np.ndarray:
    """Generate random orthogonal matrix via QR decomposition."""
    A = rng.standard_normal((n, n))
    Q, R = np.linalg.qr(A)
    
    # Adjust signs
    d = np.diagonal(R)
    Q = Q * np.sign(d)
    
    # Ensure det = +1
    if np.linalg.det(Q) < 0:
        Q[:, -1] *= -1
    
    return Q.astype(np.float32)
