# -*- coding: utf-8 -*-
"""
Complete Support Masking System with Smooth Boundaries
=======================================================

Implements geometrically correct support regions with:
1. Smooth Gaussian masks χ_i(c) ∈ [0,1]
2. Large covariance outside support (Σ → ∞·I)
3. Overlap thresholding
4. Proper field enforcement

Author: Chris
Date: November 2025
"""

import numpy as np
from typing import Tuple, Optional, Literal
from dataclasses import dataclass
from scipy.ndimage import gaussian_filter




@dataclass
class MaskConfig:
    """
    Configuration for support region masks.
    
    Geometric Interpretation:
    -------------------------
    - Hard mask: χ(c) = 1 inside, 0 outside (step function)
    - Smooth mask: χ(c) ∈ [0,1] with smooth transition (smooth_step)
    - Gaussian mask: χ(c) = exp(-r²/2σ²) with cutoff
    
    The mask defines where agent's section is "substantive":
    - χ(c) ≈ 1: Agent fully active
    - χ(c) ≈ 0: Agent effectively undefined
    - 0 < χ(c) < 1: Transition region
    """
    
    # Mask type
    mask_type: Literal['hard', 'smooth', 'gaussian'] = 'gaussian'
    
    # Smooth transition parameters
    smooth_width: float = 0.1  # For 'smooth' type (relative to radius)
    
    # Gaussian parameters
    gaussian_sigma: float = 0.47  # σ relative to radius gaussian_sigma = 1.0 / np.sqrt(-2 * np.log(OVERLAP_THRESHOLD))

    gaussian_cutoff_sigma: float = 3.0  # Hard cutoff at N*σ
    
    # Overlap detection
    overlap_threshold: float = 1e-1  # Min mask value to consider overlap
    min_mask_for_normal_cov: float = 1e-1  # Below this, use large Σ
    
    # Covariance outside support
    outside_cov_scale: float = 1e3  # Multiplier for diagonal Σ outside support
    use_smooth_cov_transition: bool = True  # Interpolate Σ at boundaries



# =============================================================================
# SECTION 2: Enhanced Support Region
# =============================================================================

class SupportRegionSmooth:
    """
    Support region with CONTINUOUS mask values χ_i(c) ∈ [0,1].
    
    Extends basic SupportRegion with smooth boundaries and overlap thresholding.
    
    Attributes:
        mask_binary: Boolean array (hard support boundary)
        mask_continuous: Float array in [0,1] (smooth weight)
        config: Mask configuration
        base_shape: Shape of base manifold
    """
    
    def __init__(
        self,
        mask_binary: np.ndarray,
        base_shape: Tuple[int, ...],
        config: Optional[MaskConfig] = None,
        mask_continuous: Optional[np.ndarray] = None,
    ):
        """
        Initialize smooth support region.
        
        Args:
            mask_binary: Hard boolean mask
            base_shape: Shape of base manifold
            config: Mask configuration (uses defaults if None)
            mask_continuous: Pre-computed continuous mask (optional)
        """
        self.base_shape = base_shape
        self.config = config if config is not None else MaskConfig()
        
        # Handle 0D case
        if len(base_shape) == 0:
            self.mask_binary = np.array(True)
            self.mask_continuous = np.array(1.0)
        else:
            if mask_binary.shape != base_shape:
                raise ValueError(
                    f"Mask shape {mask_binary.shape} != base shape {base_shape}"
                )
            self.mask_binary = mask_binary.astype(bool)
            
            # Generate continuous mask if not provided
            if mask_continuous is None:
                self.mask_continuous = self._generate_continuous_mask()
            else:
                self.mask_continuous = mask_continuous.astype(np.float32)
    
    def _generate_continuous_mask(self) -> np.ndarray:
        """
        Generate continuous mask from binary mask based on config.
        
        Returns:
            mask_continuous: Float array in [0,1]
        """
        if self.config.mask_type == 'hard':
            # Hard mask: just convert to float
            return self.mask_binary.astype(np.float32)
        
        elif self.config.mask_type == 'smooth':
            # Smooth transition at boundary using distance transform
            from scipy.ndimage import distance_transform_edt
            
            # Distance from boundary
            dist_inside = distance_transform_edt(self.mask_binary)
            dist_outside = distance_transform_edt(~self.mask_binary)
            
            # Signed distance (positive inside, negative outside)
            signed_dist = dist_inside - dist_outside
            
            # Smooth step function
            width = self.config.smooth_width * np.sqrt(np.sum(self.mask_binary))
            mask_smooth = 0.5 * (1.0 + np.tanh(signed_dist / width))
            
            return mask_smooth.astype(np.float32)
        
        elif self.config.mask_type == 'gaussian':
            # Gaussian decay from center
            
            # Find approximate center of mass
            if np.any(self.mask_binary):
                indices = np.argwhere(self.mask_binary)
                center = np.mean(indices, axis=0)
            else:
                center = np.array(self.base_shape) / 2
            
            # Distance from center
            coords = np.stack(
                np.meshgrid(
                    *[np.arange(s) for s in self.base_shape],
                    indexing='ij'
                ),
                axis=-1
            )
            dist = np.linalg.norm(coords - center, axis=-1)
            
            # Yoke σ to effective radius from area:
            #   R_eff = sqrt(area / π)
            #   choose σ so χ(R_eff) ≈ t_ref
            area = float(np.sum(self.mask_binary))
            if area > 0.0:
                radius = np.sqrt(area / np.pi)
            else:
                radius = min(self.base_shape) / 2.0  # fallback
            
            t_ref = max(self.config.overlap_threshold,
                       self.config.min_mask_for_normal_cov)
            
            if 0.0 < t_ref < 1.0:
                # t_ref = exp(-R^2 / (2 σ^2)) ⇒ σ = R / sqrt(-2 log t_ref)
                sigma = radius / np.sqrt(-2.0 * np.log(t_ref))
            else:
                # Degenerate thresholds: fall back to old behavior
                sigma = self.config.gaussian_sigma * radius
            
            mask_gauss = np.exp(-dist**2 / (2 * sigma**2))
            
            # Hard cutoff at N*σ
            cutoff_dist = self.config.gaussian_cutoff_sigma * sigma
            mask_gauss[dist > cutoff_dist] = 0.0
            
            return mask_gauss.astype(np.float32)
        
        else:
            raise ValueError(f"Unknown mask_type: {self.config.mask_type}")
    
    # =========================================================================
    # Properties
    # =========================================================================
    
    @property
    def n_active(self) -> int:
        """Number of active points (using threshold on continuous mask)."""
        if len(self.base_shape) == 0:
            return 1
        threshold = self.config.min_mask_for_normal_cov
        return int(np.sum(self.mask_continuous > threshold))
    
    @property
    def coverage(self) -> float:
        """Fraction of manifold covered (using continuous mask)."""
        if len(self.base_shape) == 0:
            return 1.0
        return float(np.sum(self.mask_continuous)) / self.mask_continuous.size
    
    @property
    def effective_coverage(self) -> float:
        """Coverage using threshold (only "substantive" points)."""
        if len(self.base_shape) == 0:
            return 1.0
        threshold = self.config.overlap_threshold
        return float(np.sum(self.mask_continuous > threshold)) / self.mask_continuous.size
    
    @property
    def mask(self) -> np.ndarray:
        """
        Backward compatibility: return continuous mask.
        
        NOTE: Despite the name, this returns CONTINUOUS mask χ_i(c) ∈ [0,1].
        For boolean computational mask, use get_computational_mask().
        """
        return self.mask_continuous
    
    @property
    def is_sparse(self) -> bool:
        """
        Backward compatibility: check if support is partial.
        
        Returns True if agent doesn't cover entire manifold.
        """
        return not self.is_full
    
    @property
    def is_full(self) -> bool:
        """True if C_i = C (full support)."""
        return np.all(self.mask_binary)
    
    @property
    def is_single_point(self) -> bool:
        """True if |C_i| = 1."""
        return self.n_active == 1
    
    
    # Add this to agent/masking.py SupportRegionSmooth class:

    @property
    def chi_weight(self) -> np.ndarray:
        """Compatibility alias for mask_continuous."""
        return self.mask_continuous
    
    @chi_weight.setter
    def chi_weight(self, value):
        """Setter for chi_weight compatibility."""
        self.mask_continuous = value.astype(np.float32) if hasattr(self, 'mask_continuous') else value
    
    @property  
    def base_manifold(self):
        """Create BaseManifold from base_shape for compatibility."""
        if not hasattr(self, 'base_shape') or self.base_shape is None:
            return None
        
        from geometry.geometry_base import BaseManifold, TopologyType
        
        if not hasattr(self, '_cached_base_manifold'):
            self._cached_base_manifold = BaseManifold(
                shape=self.base_shape,
                topology=TopologyType.PERIODIC
            )
        return self._cached_base_manifold
    
    @base_manifold.setter
    def base_manifold(self, value):
        """Setter for compatibility - no-op since base_manifold is derived."""
        pass  # Ignore assignments
    
    
    # =========================================================================
    # Computational Mask
    # =========================================================================
    
    def get_computational_mask(self, threshold: Optional[float] = None) -> np.ndarray:
        """
        Get BOOLEAN mask for computation.
        
        Thresholds the continuous mask to determine where agent is substantively defined.
        
        Returns:
            mask: Boolean array
        """
        if len(self.base_shape) == 0:
            return np.array(True)
        
        threshold = threshold or self.config.overlap_threshold
        return self.mask_continuous > threshold
    
    # =========================================================================
    # Overlap Operations
    # =========================================================================
    
    def compute_overlap_continuous(
        self,
        other: 'SupportRegionSmooth',
    ) -> np.ndarray:
        """
        Compute CONTINUOUS overlap field χ_ij(c) = χ_i(c) · χ_j(c).
        
        This is the CORRECT overlap for weighted integration.
        
        Mathematical form:
            χ_ij(c) = χ_i(c) · χ_j(c) ∈ [0,1]
        
        Physical interpretation:
            - χ_ij(c) = 0: No overlap at point c
            - χ_ij(c) = 1: Both agents fully active at c  
            - 0 < χ_ij(c) < 1: Soft boundary (one or both agents weak)
        
        Usage:
            - Store in system.overlap_masks for weighted integration
            - Threshold to get boolean computational mask when needed
        
        Returns:
            chi_ij: Continuous field in [0,1], shape = base_shape
        
        Example:
            >>> chi_i = np.array([1.0, 0.8, 0.3, 0.1])
            >>> chi_j = np.array([0.1, 0.3, 0.8, 1.0])
            >>> chi_ij = support_i.compute_overlap_continuous(support_j)
            >>> # Result: [0.1, 0.24, 0.24, 0.1]
            >>> # Note: max is 0.24, not 1.0!
        """
        if self.base_shape != other.base_shape:
            raise ValueError(
                f"Cannot compute overlap: different base shapes "
                f"{self.base_shape} vs {other.base_shape}"
            )
        
        if len(self.base_shape) == 0:
            # 0D: both agents at same point
            return np.array(1.0, dtype=np.float32)
        
        # Pointwise product of continuous masks
        chi_ij = self.mask_continuous * other.mask_continuous
        
        return chi_ij.astype(np.float32)
    
    def compute_overlap_mask(
        self,
        other: 'SupportRegionSmooth',
        threshold: Optional[float] = None,
    ) -> np.ndarray:
        """
        Compute BOOLEAN overlap mask via thresholding.
        
        Returns computational overlap domain (lens shape).
        
        χ_ij = {c : χ_i(c) > θ AND χ_j(c) > θ}
        
        Returns:
            chi_ij: Boolean mask, True where both agents substantively defined
        """
        if self.base_shape != other.base_shape:
            raise ValueError("Different base shapes")
        
        if len(self.base_shape) == 0:
            return np.array(True)
        
        threshold = threshold or self.config.overlap_threshold
        
        # Boolean intersection after thresholding
        mask_i = self.mask_continuous > threshold
        mask_j = other.mask_continuous > threshold
        
        return mask_i & mask_j
    
    def overlaps_with(
        self,
        other: 'SupportRegionSmooth',
        threshold: Optional[float] = None
    ) -> bool:
        """
        Check if geometric overlap exists.
        
        Returns True if any point has both agents substantively defined.
        """
        overlap_mask = self.compute_overlap_mask(other, threshold)
        return bool(np.any(overlap_mask))
    
    def compute_overlap_fraction(self, other: 'SupportRegionSmooth') -> float:
        """
        Compute overlap fraction: ∫χ_ij / ∫χ_i using CONTINUOUS overlap.
        
        Returns:
            fraction: Value in [0, 1]
        """
        if len(self.base_shape) == 0:
            return 1.0
        
        # Use continuous overlap for proper integration
        chi_ij_continuous = self.compute_overlap_continuous(other)
        overlap_integral = float(np.sum(chi_ij_continuous))
        self_integral = float(np.sum(self.mask_continuous))
        
        if self_integral < 1e-12:
            return 0.0
        
        return overlap_integral / self_integral

# =============================================================================
# SECTION 3: Factory Methods for Common Patterns
# =============================================================================

class SupportPatternsSmooth:
    """Factory for creating smooth support regions."""
    
    @staticmethod
    def circle(
        manifold_shape: Tuple[int, int],
        center: Tuple[float, float],
        radius: float,
        config: Optional[MaskConfig] = None
    ) -> "SupportRegionSmooth":
        """
        Create circular support region with smooth boundary.
    
        The Gaussian mask is *yoked* to the radius: the radius you pass in is
        approximately where χ(c) hits the overlap threshold, so the "support
        radius" matches the "mask radius" with no surprises.
        """
        if config is None:
            config = MaskConfig()
    
        H, W = manifold_shape
        cy, cx = center
    
        # Coordinate grid
        y, x = np.ogrid[:H, :W]
        dist = np.sqrt((x - cx)**2 + (y - cy)**2)
    
        # Default binary mask: inside (or just at) the radius
        # (we'll keep a tiny epsilon margin for numerical issues)
        eps = 1e-6
        cutoff_radius = radius * (1.0 + eps)
        mask_binary = dist <= cutoff_radius
    
        # Continuous mask
        mask_continuous = None
    
        if config.mask_type == "gaussian":
            # -------------------------------------------------
            # YOKE σ TO RADIUS:
            #   We choose σ so that χ(R) ≈ t_ref, where t_ref is
            #   the reference threshold used for "active" support.
            # -------------------------------------------------
            t_ref = max(config.overlap_threshold, config.min_mask_for_normal_cov)
    
            # Safety: fall back to old behavior if t_ref is degenerate
            if t_ref <= 0.0 or t_ref >= 1.0:
                sigma = config.gaussian_sigma * radius
            else:
                # t_ref = exp(-R^2 / (2 σ^2))  ⇒  σ = R / sqrt(-2 log t_ref)
                sigma = radius / np.sqrt(-2.0 * np.log(t_ref))
    
            mask_continuous = np.exp(-dist**2 / (2.0 * sigma**2))
    
            # Hard cutoff just (barely) outside the radius:
            # no long tails beyond the declared support radius.
            mask_continuous[dist > cutoff_radius] = 0.0
    
        # Other mask types fall back to default behavior (no continuous mask)
        return SupportRegionSmooth(
            mask_binary=mask_binary,
            base_shape=manifold_shape,
            config=config,
            mask_continuous=mask_continuous,
        )

    
    @staticmethod
    def rectangle(
        manifold_shape: Tuple[int, int],
        center: Tuple[float, float],
        width: float,
        height: float,
        config: Optional[MaskConfig] = None
    ) -> SupportRegionSmooth:
        """
        Create rectangular support region with smooth boundary.
        
        Args:
            manifold_shape: 2D grid shape (H, W)
            center: (y, x) center coordinates
            width: Rectangle width
            height: Rectangle height
            config: Mask configuration
        
        Returns:
            Smooth support region
        """
        if config is None:
            config = MaskConfig()
        
        H, W = manifold_shape
        cy, cx = center
        
        # Binary mask (hard rectangle)
        y, x = np.ogrid[:H, :W]
        mask_binary = (
            (np.abs(y - cy) <= height / 2) &
            (np.abs(x - cx) <= width / 2)
        )
        
        return SupportRegionSmooth(
            mask_binary=mask_binary,
            base_shape=manifold_shape,
            config=config
        )


# =============================================================================
# SECTION 4: Agent Field Enforcement
# =============================================================================

class FieldEnforcer:
    """
    Enforces support constraints on agent fields.
    
    Key operations:
    - Zero mean fields outside support
    - Large covariance outside support (Σ → ∞·I)
    - Smooth transitions at boundaries
    """
    @staticmethod
    def enforce_cholesky_field(
        L_raw: np.ndarray,
        support: 'SupportRegionSmooth',
        inside_scale: float = 1.0,
        outside_scale: float = 1e3,
        use_smooth_transition: bool = True
    ) -> np.ndarray:
        """
        Enforce support constraints on Cholesky factor field.
        
        This is MUCH cleaner than enforcing on Σ directly!
        
        Args:
            L_raw: Raw Cholesky factor field, shape (*S, K, K)
            support: Support region with continuous mask
            inside_scale: Scale for L inside support
            outside_scale: Scale for Σ outside (L will be sqrt of this)
            use_smooth_transition: Smooth interpolation at boundaries
        
        Returns:
            L_enforced: Cholesky field with proper support behavior
        
        Algorithm:
            Inside support: L(c) from initialization (smooth, well-conditioned)
            Outside support: L(c) = sqrt(outside_scale) * I
            
            Then Σ = LL^T automatically gives:
            - Inside: smooth covariance from initialization
            - Outside: large diagonal outside_scale * I
        """
       
        K = L_raw.shape[-1]
        
        chi = support.mask_continuous
        threshold = support.config.min_mask_for_normal_cov
        
        # Define inside and outside Cholesky factors
        L_inside = L_raw
        
        # Outside: L = sqrt(outside_scale) * I gives Σ = outside_scale * I
        L_outside = np.sqrt(outside_scale) * np.eye(K, dtype=L_raw.dtype)
        
        if not use_smooth_transition:
            # Sharp transition
            inside_mask = chi > threshold
            
            L_enforced = np.zeros_like(L_raw)
            
            if np.any(inside_mask):
                L_enforced[inside_mask] = L_inside[inside_mask]
            
            outside_mask = ~inside_mask
            if np.any(outside_mask):
                L_enforced[outside_mask] = L_outside
        
        else:
            # Smooth transition around threshold
            transition_width = 0.5 * threshold
            lower_bound = threshold - transition_width
            
            
            t = (chi - lower_bound) / (2 * transition_width)
            t = np.clip(t, 0, 1)
            
            # Smooth S-curve
            alpha = 3 * t**2 - 2 * t**3
            alpha = alpha[..., None, None]
            
            # Interpolate L directly
            L_enforced = alpha * L_inside + (1 - alpha) * L_outside
        
        # Ensure lower triangular with positive diagonal
        L_enforced = np.tril(L_enforced)  # Zero out upper triangle
        
        # Positive diagonal (critical!) but *idempotent*
        eps = 1e-6
        for k in range(K):
            diag_vals = L_enforced[..., k, k]
        
            # If already safely positive, leave it alone.
            # Only fix entries that are <= 0 or extremely tiny.
            diag_fixed = np.where(
                diag_vals <= eps,
                eps,          # clamp bad / tiny values up to eps
                diag_vals,    # keep existing good values
            )
            L_enforced[..., k, k] = diag_fixed
        
        return L_enforced.astype(np.float32)


    @staticmethod
    def enforce_mean_field(
        mu: np.ndarray,
        support: SupportRegionSmooth
    ) -> np.ndarray:
        """
        Enforce μ(c) = 0 outside support using BOOLEAN gate.
        
        ✅ CORRECT: Use threshold to gate, keep full values inside
        ❌ WRONG: Multiply by continuous mask (changes field values)
        
        Args:
            mu: Mean field, shape (*spatial, K)
            support: Support region
        
        Returns:
            mu_enforced: Full values inside, zero outside
        """
        if len(support.base_shape) == 0:
            return mu
        
        # Get BOOLEAN mask
        mask = support.get_computational_mask()
        
        # Gate field: full value OR zero
        mask_expanded = mask[..., None]  # (*spatial, 1)
        return np.where(mask_expanded, mu, 0.0)
    
    @staticmethod
    def enforce_covariance_field(
        Sigma: np.ndarray,
        support: SupportRegionSmooth,
        inside_scale: float = 1.0,
        outside_scale: float = 1e5,
        use_smooth_transition: bool = False  # Usually False for computation
    ) -> np.ndarray:
        """
        BROKEN~!!~!
        Args:
            Sigma: Covariance field
            support: Support region  
            inside_scale: Scale inside support
            outside_scale: Scale outside (large!)
            use_smooth_transition: If True, smooth interpolation (initialization only)
        """
        if len(support.base_shape) == 0:
            return Sigma
        
        K = Sigma.shape[-1]
        spatial_shape = Sigma.shape[:-2]
        large_cov = outside_scale * np.eye(K, dtype=np.float32)
        
        if use_smooth_transition:
            # Smooth version (for initialization/regularization)
            mask = support.mask_continuous[..., None, None]
            large_cov_field = np.broadcast_to(large_cov, spatial_shape + (K, K))
            return mask * Sigma + (1 - mask) * large_cov_field
        
        else:
            # Boolean gate (for computation)
            mask = support.get_computational_mask()
            
            Sigma_enforced = np.zeros_like(Sigma)
            Sigma_enforced[mask] = Sigma[mask]  # Full values where defined
            Sigma_enforced[~mask] = large_cov  # Large identity elsewhere
            
            return Sigma_enforced.astype(np.float32)
    
    @staticmethod
    def enforce_gauge_field(
        phi: np.ndarray,
        support: SupportRegionSmooth,
        fill_value: float = 0.0
    ) -> np.ndarray:
        """
        Enforce φ = 0 outside support using BOOLEAN gate.
        
        Args:
            phi: Gauge field, shape (*spatial, 3)
            support: Support region
        
        Returns:
            phi_enforced: Full values inside, zero outside
        """
        if len(support.base_shape) == 0:
            return phi
        
        # Boolean gate
        mask = support.get_computational_mask()
        mask_expanded = mask[..., None]
        
        return np.where(mask_expanded, phi, fill_value)






