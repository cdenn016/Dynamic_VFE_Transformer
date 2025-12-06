# -*- coding: utf-8 -*-
"""
Created on Sun Nov  9 17:55:51 2025

@author: chris and christine
"""

#!/usr/bin/env python3
"""
Clean Geometric Foundation for Multi-Agent Active Inference
===========================================================

CRITICAL DESIGN PRINCIPLES:
--------------------------
1. χ_i(c) ∈ [0,1] is a CONTINUOUS WEIGHT for integration
2. Boolean masks are DERIVED via thresholding (for computational gating only)
3. ALL spatial integrals explicitly weighted: ∫_C χ(c)·f(c) dc ≈ Σ_c χ(c)·f(c)
4. Energy terms handle χ weighting, NOT buried in field computations

Mathematical Framework:
----------------------
Base space: C with shape (*S) where S = (s₁, ..., sₐ) for d-dimensional manifold
Support: C_i ⊆ C characterized by continuous weight χ_i: C → [0,1]
Overlap: χ_ij(c) = χ_i(c)·χ_j(c) (pointwise product)
Integration: ∫_C χ(c)f(c)dc ≈ Σ_c χ(c)·f(c) (weighted sum)

Author: Chris 
Date: November 2025
"""

import numpy as np
from typing import Tuple, Optional, Dict, Literal, Union
from dataclasses import dataclass
from enum import Enum


# =============================================================================
# SECTION 1: Base Manifold (Shared Spatial Domain)
# =============================================================================

class TopologyType(Enum):
    """Topology types for base manifold."""
    FLAT = "flat"              # ℝᵈ (Euclidean, open boundaries)
    PERIODIC = "periodic"       # 𝕋ᵈ (torus, periodic boundaries)
    SPHERE = "sphere"          # Sᵈ (compact, no boundary)



@dataclass
class BaseManifold:
    """
    Base space C for the principal bundle.
    
    All agents are sections over this shared manifold.
    
    Attributes:
        shape: Spatial dimensions (*S) where S = () for 0D, (N,) for 1D, (H,W) for 2D, etc.
        topology: Geometric structure (flat, periodic, sphere)
        n_points: Total number of spatial points |C|
        ndim: Spatial dimensionality d = len(shape)
    """
    shape: Tuple[int, ...]
    topology: TopologyType = TopologyType.PERIODIC
    
    def __post_init__(self):
        self.n_points = int(np.prod(self.shape)) if self.shape else 1
        self.ndim = len(self.shape)
    
    @property
    def is_point(self) -> bool:
        """True if 0-dimensional (single point)."""
        return self.ndim == 0
    
    def __repr__(self) -> str:
        if self.is_point:
            return "BaseManifold(0D point)"
        return f"BaseManifold(shape={self.shape}, {self.topology.value}, {self.n_points} pts)"


# =============================================================================
# SECTION 2: Support Region with Continuous Weight χ(c)
# =============================================================================

@dataclass
class SupportRegion:
    """
    Support region C_i ⊆ C with continuous weight function χ_i(c) ∈ [0,1].
    
    CRITICAL DISTINCTION:
    --------------------
    - chi_weight: CONTINUOUS float ∈ [0,1] for INTEGRATION weighting
    - mask_bool: DERIVED boolean for computational gating
    
    The continuous weight χ(c) is the fundamental geometric object.
    Boolean masks are just a computational convenience.
    
    Attributes:
        base_manifold: The base space C
        chi_weight: Continuous support weight χ_i: C → [0,1], shape = base_manifold.shape
        threshold_computational: Threshold for deriving boolean masks
        threshold_integration: Numerical cutoff for treating χ ≈ 0 in sums
    """
    base_manifold: BaseManifold
    chi_weight: np.ndarray  # Shape: (*S,), float ∈ [0,1]
    threshold_computational: float = 1e-6  # For boolean gating
    threshold_integration: float = 1e-12   # For numerical integration cutoff
    
    def __post_init__(self):
        """Validate support region on construction."""
        # Handle 0D case
        if self.base_manifold.is_point:
            if not isinstance(self.chi_weight, np.ndarray):
                self.chi_weight = np.array(1.0, dtype=np.float32)
            return
        
        # Validate shape
        if self.chi_weight.shape != self.base_manifold.shape:
            raise ValueError(
                f"χ weight shape {self.chi_weight.shape} != "
                f"base manifold shape {self.base_manifold.shape}"
            )
        
        # Validate range
        if not (np.all(self.chi_weight >= 0) and np.all(self.chi_weight <= 1)):
            raise ValueError("χ weights must be in [0, 1]")
        
        # Ensure float32
        self.chi_weight = self.chi_weight.astype(np.float32)
    
    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------
    
    @property
    def n_active(self) -> int:
        """Number of points where χ > threshold (substantively active)."""
        return int(np.sum(self.get_mask_bool()))
    
    @property
    def effective_coverage(self) -> float:
        """Fraction of manifold covered: ∫χ(c)dc / |C|."""
        if self.base_manifold.is_point:
            return 1.0
        return float(np.sum(self.chi_weight)) / self.base_manifold.n_points
    
    @property
    def is_full_support(self) -> bool:
        """True if C_i = C (agent defined everywhere)."""
        return bool(np.all(self.chi_weight > self.threshold_computational))
    

    
    # -------------------------------------------------------------------------
    # Mask Generation (Derived from χ)
    # -------------------------------------------------------------------------
    
    def get_mask_bool(self, threshold: Optional[float] = None) -> np.ndarray:
        """
        Get BOOLEAN mask for computational gating.
        
        This is DERIVED from χ via thresholding, not fundamental.
        Use for: "Should I compute at this point?"
        
        Args:
            threshold: Override default computational threshold
        
        Returns:
            mask: Boolean array, True where χ > threshold
        """
        if self.base_manifold.is_point:
            return np.array(True)
        
        thresh = threshold if threshold is not None else self.threshold_computational
        return self.chi_weight > thresh
    
    # -------------------------------------------------------------------------
    # Overlap Operations
    # -------------------------------------------------------------------------
    
    def compute_overlap_continuous(self, other: 'SupportRegion') -> np.ndarray:
        """
        Compute CONTINUOUS overlap weight: χ_ij(c) = χ_i(c)·χ_j(c).
        
        This is the CORRECT mathematical object for integration.
        
        Mathematical form:
            χ_ij: C → [0,1]
            χ_ij(c) = χ_i(c) · χ_j(c)
        
        Physical interpretation:
            - χ_ij(c) = 0: No overlap at c
            - χ_ij(c) = 1: Both agents fully active at c
            - 0 < χ_ij(c) < 1: Soft boundary region
        
        Returns:
            chi_ij: Continuous overlap field, shape = base_manifold.shape
        
        Example:
            >>> chi_i = np.array([1.0, 0.8, 0.3, 0.0])
            >>> chi_j = np.array([0.0, 0.3, 0.8, 1.0])
            >>> chi_ij = support_i.compute_overlap_continuous(support_j)
            >>> # Result: [0.0, 0.24, 0.24, 0.0]
        """
        if self.base_manifold is not other.base_manifold:
            if self.base_manifold.shape != other.base_manifold.shape:
                raise ValueError("Cannot compute overlap: different base manifolds")
        
        if self.base_manifold.is_point:
            return np.array(1.0, dtype=np.float32)
        
        # Pointwise product (this is the mathematical definition)
        chi_ij = self.chi_weight * other.chi_weight
        return chi_ij.astype(np.float32)
    
    def has_overlap(self, other: 'SupportRegion', threshold: Optional[float] = None) -> bool:
        """
        Check if geometric overlap exists: ∃c : χ_ij(c) > threshold.
        
        Returns:
            True if any point has substantive overlap
        """
        chi_ij = self.compute_overlap_continuous(other)
        thresh = threshold if threshold is not None else self.threshold_computational
        return bool(np.any(chi_ij > thresh))
    
    def compute_overlap_volume(self, other: 'SupportRegion') -> float:
        """
        Compute overlap volume: ∫_C χ_ij(c) dc.
        
        Returns:
            volume: Total overlap weight
        """
        chi_ij = self.compute_overlap_continuous(other)
        return float(np.sum(chi_ij))
    
    # -------------------------------------------------------------------------
    # Representation
    # -------------------------------------------------------------------------
    
    def __repr__(self) -> str:
        if self.base_manifold.is_point:
            return "SupportRegion(0D, χ=1.0)"
        return (
            f"SupportRegion("
            f"shape={self.base_manifold.shape}, "
            f"active={self.n_active}/{self.base_manifold.n_points}, "
            f"coverage={self.effective_coverage:.1%})"
        )


# =============================================================================
# SECTION 4: Factory Functions for Common Support Patterns
# =============================================================================

def create_full_support(base_manifold: 'BaseManifold') -> 'SupportRegion':
    """
    Create full support: C_i = C, χ_i(c) = 1 everywhere.
    
    Args:
        base_manifold: Base space C
    
    Returns:
        support: Full support region
    
    Example:
        >>> manifold = BaseManifold(shape=(10,), topology=TopologyType.FLAT)
        >>> support = create_full_support(manifold)
        >>> assert support.is_full_support
    """
    if base_manifold.is_point:
        chi = np.array(1.0, dtype=np.float32)
    else:
        chi = np.ones(base_manifold.shape, dtype=np.float32)
    
    return SupportRegion(
        base_manifold=base_manifold,
        chi_weight=chi
    )


def create_gaussian_support(
    base_manifold: 'BaseManifold',
    center: Tuple[float, ...],
    radius: float,
    sigma_relative: float = 0.47,  # Default: ~10% at boundary
    cutoff_sigma: float = 3.0,
) -> 'SupportRegion':
    """
    Create Gaussian support: χ(c) = exp(-r²/2σ²) with hard cutoff.
    
    Mathematical form:
        r(c) = ||c - center||
        χ(c) = exp(-r²/2σ²)  if r ≤ cutoff_sigma·σ, else 0
    
    Args:
        base_manifold: Base space C
        center: Center point coordinates (length = ndim)
        radius: Support radius (distance units)
        sigma_relative: Gaussian width σ = sigma_relative·radius
        cutoff_sigma: Hard cutoff at cutoff_sigma·σ standard deviations
    
    Returns:
        support: Gaussian support region
    
    Example:
        >>> manifold = BaseManifold(shape=(50, 50), topology=TopologyType.PERIODIC)
        >>> support = create_gaussian_support(
        ...     manifold, center=(25.0, 25.0), radius=10.0
        ... )
    """
    if base_manifold.is_point:
        return create_full_support(base_manifold)
    
    if len(center) != base_manifold.ndim:
        raise ValueError(f"Center dim {len(center)} != manifold dim {base_manifold.ndim}")
    
    # Create coordinate grids
    coords = [np.arange(s, dtype=np.float32) for s in base_manifold.shape]
    grids = np.meshgrid(*coords, indexing='ij')
    
    # Compute squared distance from center
    r_squared = sum((grid - c)**2 for grid, c in zip(grids, center))
    r = np.sqrt(r_squared)
    
    # Gaussian parameters
    sigma = sigma_relative * radius
    cutoff_radius = cutoff_sigma * sigma
    
    # Gaussian with hard cutoff
    chi = np.where(
        r <= cutoff_radius,
        np.exp(-0.5 * r_squared / sigma**2),
        0.0
    ).astype(np.float32)
    
    return SupportRegion(
        base_manifold=base_manifold,
        chi_weight=chi
    )


def create_box_support(
    base_manifold: 'BaseManifold',
    center: Tuple[float, ...],
    half_widths: Tuple[float, ...],
    smoothness: float = 0.0,
) -> 'SupportRegion':
    """
    Create box/rectangular support with optional smooth boundaries.
    
    Args:
        base_manifold: Base space C
        center: Box center coordinates
        half_widths: Half-widths along each axis
        smoothness: Boundary smoothness (0 = hard, >0 = smooth transition)
    
    Returns:
        support: Box support region
    
    Example:
        >>> manifold = BaseManifold(shape=(50, 50), topology=TopologyType.FLAT)
        >>> support = create_box_support(
        ...     manifold, 
        ...     center=(25.0, 25.0), 
        ...     half_widths=(10.0, 15.0),
        ...     smoothness=2.0
        ... )
    """
    if base_manifold.is_point:
        return create_full_support(base_manifold)
    
    if len(center) != base_manifold.ndim:
        raise ValueError(f"Center dim mismatch")
    if len(half_widths) != base_manifold.ndim:
        raise ValueError(f"Half-widths dim mismatch")
    
    # Create coordinate grids
    coords = [np.arange(s, dtype=np.float32) for s in base_manifold.shape]
    grids = np.meshgrid(*coords, indexing='ij')
    
    # Start with all ones
    chi = np.ones(base_manifold.shape, dtype=np.float32)
    
    if smoothness == 0:
        # Hard box
        for grid, c, hw in zip(grids, center, half_widths):
            inside = (grid >= c - hw) & (grid <= c + hw)
            chi *= inside.astype(np.float32)
    else:
        # Smooth box using product of sigmoids
        for grid, c, hw in zip(grids, center, half_widths):
            # Distance from boundary
            dist = np.minimum(grid - (c - hw), (c + hw) - grid)
            # Smooth step function
            chi *= 0.5 * (1.0 + np.tanh(dist / smoothness))
    
    return SupportRegion(
        base_manifold=base_manifold,
        chi_weight=chi
    )


# =============================================================================
# SECTION 5: SupportPatterns Wrapper Class
# =============================================================================

class SupportPatterns:
    """
    Factory class for common support patterns.
    
    Provides convenient class-based API wrapping the factory functions.
    """
    
    @staticmethod
    def full(base_manifold: 'BaseManifold') -> 'SupportRegion':
        """Create full support: C_i = C, χ_i(c) = 1 everywhere."""
        return create_full_support(base_manifold)
    
    @staticmethod
    def gaussian(
        base_manifold: 'BaseManifold',
        center: Tuple[float, ...],
        radius: float,
        sigma_relative: float = 0.47,
        cutoff_sigma: float = 3.0,
    ) -> 'SupportRegion':
        """Create Gaussian support."""
        return create_gaussian_support(
            base_manifold, center, radius, sigma_relative, cutoff_sigma
        )
    
    @staticmethod
    def box(
        base_manifold: 'BaseManifold',
        center: Tuple[float, ...],
        half_widths: Tuple[float, ...],
        smoothness: float = 0.0,
    ) -> 'SupportRegion':
        """Create box support."""
        return create_box_support(base_manifold, center, half_widths, smoothness)
    
    @staticmethod
    def ball(
        base_manifold: 'BaseManifold',
        center: Union[Tuple[float, ...], float],
        radius: float,
        sigma_relative: float = 0.47,
    ) -> 'SupportRegion':
        """Alias for gaussian() - backward compatibility."""
        if isinstance(center, (int, float)):
            center = (center,)
        return create_gaussian_support(
            base_manifold, center, radius, sigma_relative
        )
    
    @staticmethod
    def point(
        base_manifold: 'BaseManifold',
        index: Union[Tuple[int, ...], int]
    ) -> 'SupportRegion':
        """Single point support at given index."""
        if base_manifold.is_point:
            return create_full_support(base_manifold)
        
        if isinstance(index, int) and base_manifold.ndim == 1:
            index = (index,)
        
        chi = np.zeros(base_manifold.shape, dtype=np.float32)
        chi[index] = 1.0
        
        return SupportRegion(
            base_manifold=base_manifold,
            chi_weight=chi
        )





# =============================================================================
# SECTION 3: Integration Primitives (ALL χ-weighted)
# =============================================================================

def spatial_integrate(
    field: np.ndarray,
    chi_weight: np.ndarray,
    *,
    threshold: Optional[float] = 1e-12
) -> float:
    """
    Compute weighted spatial integral: ∫_C χ(c)·f(c) dc ≈ Σ_c χ(c)·f(c).
    
    CRITICAL: This is the CORRECT way to integrate over a support region.
    
    Args:
        field: Spatial field f(c), shape (*S, ...) where *S matches chi_weight
        chi_weight: Continuous weight χ(c) ∈ [0,1], shape (*S,)
        threshold: Treat χ < threshold as zero (numerical stability)
    
    Returns:
        integral: Weighted sum over space
    
    Examples:
        >>> # Integrate energy density over agent support
        >>> energy = spatial_integrate(kl_field, chi_i)
        
        >>> # Integrate over overlap region
        >>> chi_ij = chi_i * chi_j
        >>> overlap_energy = spatial_integrate(beta_ij * kl_field, chi_ij)
    """
    # Handle 0D case
    if chi_weight.ndim == 0:
        if field.ndim == 0:
            return float(field)
        else:
            # field has trailing dimensions beyond spatial (e.g., vector/matrix)
            return float(np.sum(field))
    
    # Get spatial shape from chi_weight
    spatial_shape = chi_weight.shape
    spatial_ndim = len(spatial_shape)
    
    # Validate field has matching spatial dimensions
    if field.shape[:spatial_ndim] != spatial_shape:
        raise ValueError(
            f"Field spatial shape {field.shape[:spatial_ndim]} != "
            f"χ weight shape {spatial_shape}"
        )
    
    # Threshold χ for numerical stability (avoid multiplying huge field by tiny χ)
    chi_active = np.where(chi_weight > threshold, chi_weight, 0.0)
    
    # Broadcast χ to match field shape
    # chi_weight has shape (*S,)
    # field has shape (*S, ...) where ... are trailing dimensions (K, K×K, etc.)
    broadcast_shape = chi_weight.shape + (1,) * (field.ndim - spatial_ndim)
    chi_broadcast = chi_active.reshape(broadcast_shape)
    
    # Weighted field
    weighted_field = chi_broadcast * field
    
    # Sum over all dimensions (spatial + trailing)
    integral = float(np.sum(weighted_field))
    
    return integral


def broadcast_mask(
    mask: np.ndarray,
    target_shape: Tuple[int, ...],
    is_vector: bool = False
) -> np.ndarray:
    """Broadcast mask to target shape in dimension-agnostic way."""
    mask = np.asarray(mask)
    
    if mask.shape in ((), (1,)):
        scalar_val = float(mask.flat[0]) if mask.size > 0 else 1.0
        return np.full(target_shape, scalar_val, dtype=np.float32)
    
    if is_vector:
        if mask.ndim == len(target_shape) - 1:
            return mask[..., None] * np.ones(target_shape, dtype=np.float32)
    else:
        if mask.ndim == len(target_shape) - 2:
            return mask[..., None, None] * np.ones(target_shape, dtype=np.float32)
    
    raise ValueError(f"Mask shape {mask.shape} incompatible with target {target_shape}")



def spatial_integrate_vector(
    field: np.ndarray,
    chi_weight: np.ndarray,
    *,
    threshold: Optional[float] = 1e-12
) -> np.ndarray:
    """
    Compute weighted spatial integral of VECTOR field: ∫_C χ(c)·v(c) dc.
    
    Like spatial_integrate but preserves vector structure (last dimension).
    
    Args:
        field: Vector field v(c), shape (*S, K)
        chi_weight: Continuous weight χ(c) ∈ [0,1], shape (*S,)
        threshold: Treat χ < threshold as zero
    
    Returns:
        integral: Vector result, shape (K,)
    
    Example:
        >>> # Integrate mean field over support
        >>> mu_integrated = spatial_integrate_vector(mu_field, chi_i)
    """
    # Handle 0D case
    if chi_weight.ndim == 0:
        return field if field.ndim == 1 else field.ravel()
    
    spatial_shape = chi_weight.shape
    spatial_ndim = len(spatial_shape)
    
    # Validate
    if field.shape[:spatial_ndim] != spatial_shape:
        raise ValueError(f"Field spatial shape mismatch")
    
    # Get vector dimension K
    K = field.shape[-1]
    
    # Threshold and broadcast
    chi_active = np.where(chi_weight > threshold, chi_weight, 0.0)
    chi_broadcast = chi_active.reshape(*spatial_shape, 1)  # (*S, 1)
    
    # Weight and sum over spatial dimensions only
    weighted_field = chi_broadcast * field  # (*S, K)
    
    # Sum over spatial axes, keep vector structure
    spatial_axes = tuple(range(spatial_ndim))
    integral = np.sum(weighted_field, axis=spatial_axes)  # (K,)
    
    return integral


def spatial_integrate_matrix(
    field: np.ndarray,
    chi_weight: np.ndarray,
    *,
    threshold: Optional[float] = 1e-12
) -> np.ndarray:
    """
    Compute weighted spatial integral of MATRIX field: ∫_C χ(c)·M(c) dc.
    
    Like spatial_integrate but preserves matrix structure (last 2 dimensions).
    
    Args:
        field: Matrix field M(c), shape (*S, K, K)
        chi_weight: Continuous weight χ(c) ∈ [0,1], shape (*S,)
        threshold: Treat χ < threshold as zero
    
    Returns:
        integral: Matrix result, shape (K, K)
    
    Example:
        >>> # Integrate covariance field over support
        >>> Sigma_integrated = spatial_integrate_matrix(Sigma_field, chi_i)
    """
    # Handle 0D case
    if chi_weight.ndim == 0:
        return field if field.ndim == 2 else field.reshape(field.shape[-2:])
    
    spatial_shape = chi_weight.shape
    spatial_ndim = len(spatial_shape)
    
    # Validate
    if field.shape[:spatial_ndim] != spatial_shape:
        raise ValueError(f"Field spatial shape mismatch")
    
    # Get matrix dimensions K×K
    K = field.shape[-1]
    
    # Threshold and broadcast
    chi_active = np.where(chi_weight > threshold, chi_weight, 0.0)
    chi_broadcast = chi_active.reshape(*spatial_shape, 1, 1)  # (*S, 1, 1)
    
    # Weight and sum
    weighted_field = chi_broadcast * field  # (*S, K, K)
    
    # Sum over spatial axes only
    spatial_axes = tuple(range(spatial_ndim))
    integral = np.sum(weighted_field, axis=spatial_axes)  # (K, K)
    
    return integral





