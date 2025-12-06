# -*- coding: utf-8 -*-
"""
Created on Sun Nov 16 09:26:36 2025

@author: chris and christine
"""

# -*- coding: utf-8 -*-
"""
SO(3) Fréchet Mean and Geometric Averaging
===========================================

Compute the Fréchet mean (geometric average) on the SO(3) manifold.

Theory:
-------
The Fréchet mean R̄ ∈ SO(3) minimizes the sum of squared geodesic distances:

    R̄ = argmin_R Σᵢ d²_SO(3)(R, Rᵢ)

where d_SO(3) is the geodesic distance on SO(3).

Algorithm (Iterative Gradient Descent):
1. Initialize R̄ = R₀ (or identity)
2. Repeat:
   a. Compute deviations: δRᵢ = R̄ᵀ Rᵢ
   b. Map to Lie algebra: δφᵢ = log_SO(3)(δRᵢ)
   c. Average in tangent space: δφ̄ = (1/n) Σ δφᵢ
   d. Update: R̄ ← R̄ exp(δφ̄)
3. Until ||δφ̄|| < ε

For gauge frames {φᵢ} in so(3), we:
1. Map to group: Rᵢ = exp(φ̂ᵢ)
2. Compute Fréchet mean: R̄
3. Map back to algebra: φ̄ = log(R̄)

Author: Chris & Christine
Date: November 2025
"""

import numpy as np
from typing import List, Tuple, Optional
import warnings


# =============================================================================
# SO(3) Logarithm Map
# =============================================================================

def so3_log(R: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Logarithm map from SO(3) → so(3).
    
    Inverse of exponential map: given R ∈ SO(3), find φ ∈ ℝ³ such that
    exp([φ]_×) = R.
    
    Formula:
        θ = arccos((tr(R) - 1) / 2)
        φ = (θ / (2 sin θ)) * vex(R - Rᵀ)
    
    where vex is the inverse of the [·]_× (skew-symmetric) operator.
    
    Args:
        R: Rotation matrix, shape (..., 3, 3)
        eps: Threshold for small angle approximation
    
    Returns:
        phi: Lie algebra element, shape (..., 3)
    
    References:
        - Murray, Li, Sastry - "A Mathematical Introduction to Robotic Manipulation"
        - Park, Ravani - "Smooth Invariant Interpolation of Rotations" (1997)
    """
    R = np.asarray(R)
    original_shape = R.shape[:-2]
    R = R.reshape(-1, 3, 3)
    
    batch_size = R.shape[0]
    phi = np.zeros((batch_size, 3))
    
    for i in range(batch_size):
        Ri = R[i]
        
        # Compute rotation angle
        trace = np.trace(Ri)
        cos_theta = (trace - 1.0) / 2.0
        cos_theta = np.clip(cos_theta, -1.0, 1.0)  # Numerical stability
        theta = np.arccos(cos_theta)
        
        if theta < eps:
            # Small angle: θ → 0
            # Use Taylor expansion: log(R) ≈ (R - Rᵀ) / 2
            phi[i] = _vex(Ri - Ri.T) / 2.0
        
        elif theta > np.pi - eps:
            # Near π: use special formula to avoid singularity
            # Find the dominant eigenvector of (R + I) for axis
            # This is stable near θ = π
            B = Ri + np.eye(3)
            # Find column with largest norm
            col_norms = np.linalg.norm(B, axis=0)
            k = np.argmax(col_norms)
            axis = B[:, k] / col_norms[k]
            
            # Determine sign of rotation
            # Use Rodrigues formula property
            skew = Ri - Ri.T
            axis_sign = np.sign(np.dot(axis, _vex(skew)))
            if axis_sign == 0:
                axis_sign = 1.0
            
            phi[i] = theta * axis * axis_sign
        
        else:
            # General case: 0 < θ < π
            # φ = (θ / 2sin(θ)) * vex(R - Rᵀ)
            coefficient = theta / (2.0 * np.sin(theta))
            phi[i] = coefficient * _vex(Ri - Ri.T)
    
    return phi.reshape(*original_shape, 3)


def _vex(M: np.ndarray) -> np.ndarray:
    """
    Extract vector from skew-symmetric matrix (inverse of [·]_×).
    
    For M = [  0   -v_z   v_y ]
            [ v_z    0   -v_x ]
            [-v_y   v_x    0  ]
    
    Returns v = [v_x, v_y, v_z]ᵀ.
    
    Args:
        M: Skew-symmetric matrix, shape (..., 3, 3)
    
    Returns:
        v: Vector, shape (..., 3)
    """
    v = np.array([
        M[..., 2, 1] - M[..., 1, 2],  # v_x = M₃₂ - M₂₃
        M[..., 0, 2] - M[..., 2, 0],  # v_y = M₁₃ - M₃₁
        M[..., 1, 0] - M[..., 0, 1]   # v_z = M₂₁ - M₁₂
    ])
    
    # Transpose to get (..., 3) shape
    v = np.moveaxis(v, 0, -1)
    
    return v / 2.0  # Factor of 2 from anti-symmetry


# =============================================================================
# SO(3) Exponential Map (using existing code)
# =============================================================================

def so3_exp(phi: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Exponential map from so(3) → SO(3) using Rodrigues formula.
    
    Args:
        phi: Lie algebra elements, shape (..., 3)
        eps: Threshold for small angle approximation
    
    Returns:
        R: Rotation matrices, shape (..., 3, 3)
    
    Notes:
        This wraps the existing rodrigues_formula from numba_kernels.
        For consistency with the rest of the codebase.
    """
    try:
        from math_utils.numba_kernels import rodrigues_formula_numba_scalar
        
        phi = np.asarray(phi)
        original_shape = phi.shape[:-1]
        phi_flat = phi.reshape(-1, 3)
        
        R = np.zeros((phi_flat.shape[0], 3, 3))
        for i in range(phi_flat.shape[0]):
            R[i] = rodrigues_formula_numba_scalar(phi_flat[i], eps=eps)
        
        return R.reshape(*original_shape, 3, 3)
    
    except ImportError:
        # Fallback: pure NumPy implementation
        return _so3_exp_numpy(phi, eps=eps)


def _so3_exp_numpy(phi: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Pure NumPy implementation of SO(3) exponential (Rodrigues formula).
    
    exp([φ]_×) = I + (sin θ / θ)[φ]_× + ((1 - cos θ) / θ²)[φ]_×²
    
    where θ = ||φ||.
    """
    phi = np.asarray(phi)
    original_shape = phi.shape[:-1]
    phi = phi.reshape(-1, 3)
    
    batch_size = phi.shape[0]
    R = np.zeros((batch_size, 3, 3))
    
    for i in range(batch_size):
        phi_i = phi[i]
        theta = np.linalg.norm(phi_i)
        
        if theta < eps:
            # Small angle: use Taylor expansion
            # exp([φ]_×) ≈ I + [φ]_×
            phi_cross = _skew_symmetric(phi_i)
            R[i] = np.eye(3) + phi_cross
        else:
            # Rodrigues formula
            axis = phi_i / theta
            axis_cross = _skew_symmetric(axis)
            axis_cross_sq = axis_cross @ axis_cross
            
            R[i] = (np.eye(3) 
                   + np.sin(theta) * axis_cross 
                   + (1 - np.cos(theta)) * axis_cross_sq)
    
    return R.reshape(*original_shape, 3, 3)


def _skew_symmetric(v: np.ndarray) -> np.ndarray:
    """
    Construct skew-symmetric matrix [v]_× from vector v ∈ ℝ³.
    
    For v = [v_x, v_y, v_z]ᵀ:
        [v]_× = [  0   -v_z   v_y ]
                [ v_z    0   -v_x ]
                [-v_y   v_x    0  ]
    """
    v = np.asarray(v)
    return np.array([
        [0,     -v[2],  v[1]],
        [v[2],   0,    -v[0]],
        [-v[1],  v[0],  0   ]
    ])


# =============================================================================
# Fréchet Mean on SO(3)
# =============================================================================

def frechet_mean_so3(rotations: List[np.ndarray],
                     max_iter: int = 50,
                     tol: float = 1e-6,
                     weights: Optional[np.ndarray] = None) -> Tuple[np.ndarray, dict]:
    """
    Compute Fréchet mean (geometric average) on SO(3) manifold.
    
    Finds R̄ that minimizes:
        Σᵢ wᵢ · d²_SO(3)(R̄, Rᵢ)
    
    where d_SO(3) is the geodesic distance.
    
    Args:
        rotations: List of rotation matrices, each (3, 3)
        max_iter: Maximum iterations for gradient descent
        tol: Convergence tolerance on tangent space norm
        weights: Optional weights for each rotation (default: uniform)
    
    Returns:
        R_mean: Fréchet mean rotation matrix (3, 3)
        info: Dictionary with convergence information
    
    Example:
        >>> R1 = so3_exp(np.array([0.1, 0.0, 0.0]))
        >>> R2 = so3_exp(np.array([0.0, 0.1, 0.0]))
        >>> R3 = so3_exp(np.array([0.0, 0.0, 0.1]))
        >>> R_mean, info = frechet_mean_so3([R1, R2, R3])
        >>> print(f"Converged in {info['n_iter']} iterations")
    
    References:
        - Pennec, "Intrinsic Statistics on Riemannian Manifolds" (2006)
        - Miolane et al., "Geomstats" (2020)
    """
    rotations = [np.asarray(R) for R in rotations]
    n = len(rotations)
    
    if n == 0:
        raise ValueError("Need at least one rotation")
    
    if n == 1:
        return rotations[0], {'n_iter': 0, 'converged': True, 'residual': 0.0}
    
    # Set up weights
    if weights is None:
        weights = np.ones(n) / n
    else:
        weights = np.asarray(weights)
        weights = weights / np.sum(weights)  # Normalize
    
    # Initialize with first rotation
    R_mean = rotations[0].copy()
    
    # Iterative gradient descent in tangent space
    convergence_history = []
    
    for iteration in range(max_iter):
        # Step 1: Compute deviations from current mean
        delta_phis = []
        
        for i, R_i in enumerate(rotations):
            # Relative rotation: δR_i = R̄ᵀ R_i
            delta_R = R_mean.T @ R_i
            
            # Map to tangent space at identity
            delta_phi = so3_log(delta_R)
            
            delta_phis.append(delta_phi)
        
        # Step 2: Compute weighted average in tangent space
        delta_phi_mean = np.sum([w * dp for w, dp in zip(weights, delta_phis)], axis=0)
        
        # Check convergence
        residual = np.linalg.norm(delta_phi_mean)
        convergence_history.append(residual)
        
        if residual < tol:
            info = {
                'n_iter': iteration,
                'converged': True,
                'residual': residual,
                'history': convergence_history
            }
            return R_mean, info
        
        # Step 3: Update mean via exponential map
        # R̄ ← R̄ exp(δφ̄)
        delta_R_mean = so3_exp(delta_phi_mean)
        R_mean = R_mean @ delta_R_mean
        
        # Orthogonalize to maintain SO(3) (correct for numerical drift)
        R_mean = _orthogonalize_so3(R_mean)
    
    # Did not converge
    warnings.warn(f"Fréchet mean did not converge in {max_iter} iterations. "
                 f"Final residual: {residual:.2e}")
    
    info = {
        'n_iter': max_iter,
        'converged': False,
        'residual': residual,
        'history': convergence_history
    }
    
    return R_mean, info


def _orthogonalize_so3(R: np.ndarray) -> np.ndarray:
    """
    Project matrix to SO(3) using SVD.
    
    Ensures R is orthogonal with det(R) = +1.
    
    Args:
        R: Near-orthogonal matrix (3, 3)
    
    Returns:
        R_ortho: Orthogonalized matrix in SO(3)
    """
    U, _, Vt = np.linalg.svd(R)
    R_ortho = U @ Vt
    
    # Ensure det = +1 (not -1)
    if np.linalg.det(R_ortho) < 0:
        U[:, -1] *= -1
        R_ortho = U @ Vt
    
    return R_ortho


# =============================================================================
# Convenience Function for Gauge Frames
# =============================================================================

def average_gauge_frames_so3(phis: List[np.ndarray],
                             weights: Optional[np.ndarray] = None,
                             method: str = 'frechet') -> np.ndarray:
    """
    Compute geometric average of gauge frames in so(3).
    
    Args:
        phis: List of gauge frame vectors (each shape (3,))
        weights: Optional weights for each frame
        method: 'frechet' (geometric) or 'euclidean' (simple average)
    
    Returns:
        phi_mean: Averaged gauge frame (3,)
    
    Usage:
        This is the function to use in emergence.py for meta-agent formation!
    """
    phis = [np.asarray(phi) for phi in phis]
    
    if method == 'euclidean':
        # Simple Euclidean average (only valid for small deviations)
        if weights is None:
            return np.mean(phis, axis=0)
        else:
            weights = np.asarray(weights)
            weights = weights / np.sum(weights)
            return np.sum([w * phi for w, phi in zip(weights, phis)], axis=0)
    
    elif method == 'frechet':
        # Proper geometric average on SO(3)
        
        # Map to group
        rotations = [so3_exp(phi) for phi in phis]
        
        # Compute Fréchet mean
        R_mean, info = frechet_mean_so3(rotations, weights=weights)
        
        # Map back to algebra
        phi_mean = so3_log(R_mean)
        
        return phi_mean
    
    else:
        raise ValueError(f"Unknown method: {method}")


# =============================================================================
# Validation and Testing
# =============================================================================

def test_log_exp_inverse():
    """Test that log and exp are inverses."""
    print("Testing log-exp inverse property...")
    
    # Test case 1: Small rotation
    phi1 = np.array([0.1, 0.2, -0.15])
    R1 = so3_exp(phi1)
    phi1_recovered = so3_log(R1)
    
    error1 = np.linalg.norm(phi1 - phi1_recovered)
    print(f"Small rotation error: {error1:.2e}")
    assert error1 < 1e-6, f"Failed for small rotation: {error1}"
    
    # Test case 2: Large rotation
    phi2 = np.array([2.5, -1.8, 0.9])
    R2 = so3_exp(phi2)
    phi2_recovered = so3_log(R2)
    
    # Note: phi and phi + 2πn (same axis) give same R
    # So we check R(phi) == R(recovered)
    R2_check = so3_exp(phi2_recovered)
    error2 = np.linalg.norm(R2 - R2_check, 'fro')
    print(f"Large rotation error (matrix): {error2:.2e}")
    assert error2 < 1e-6, f"Failed for large rotation: {error2}"
    
    # Test case 3: Near π rotation (singularity test)
    phi3 = np.array([np.pi - 0.01, 0, 0])
    R3 = so3_exp(phi3)
    phi3_recovered = so3_log(R3)
    R3_check = so3_exp(phi3_recovered)
    error3 = np.linalg.norm(R3 - R3_check, 'fro')
    print(f"Near-π rotation error: {error3:.2e}")
    assert error3 < 1e-5, f"Failed near π: {error3}"
    
    print("✓ Log-exp inverse tests passed!\n")


def test_frechet_mean():
    """Test Fréchet mean computation."""
    print("Testing Fréchet mean on SO(3)...")
    
    # Test 1: Mean of identity and small rotation
    phi1 = np.zeros(3)
    phi2 = np.array([0.2, 0.0, 0.0])
    
    R1 = so3_exp(phi1)
    R2 = so3_exp(phi2)
    
    R_mean, info = frechet_mean_so3([R1, R2])
    phi_mean = so3_log(R_mean)
    
    print(f"Test 1: Identity + small rotation")
    print(f"  Input phis: {phi1}, {phi2}")
    print(f"  Mean phi: {phi_mean}")
    print(f"  Expected: ~[0.1, 0, 0]")
    print(f"  Converged in {info['n_iter']} iterations")
    
    # Test 2: Symmetric configuration
    phi_a = np.array([0.3, 0.0, 0.0])
    phi_b = np.array([0.0, 0.3, 0.0])
    phi_c = np.array([0.0, 0.0, 0.3])
    
    Rs = [so3_exp(p) for p in [phi_a, phi_b, phi_c]]
    R_mean, info = frechet_mean_so3(Rs)
    phi_mean = so3_log(R_mean)
    
    print(f"\nTest 2: Symmetric 3-axis configuration")
    print(f"  Mean phi: {phi_mean}")
    print(f"  Expected: ~[0.17, 0.17, 0.17] (symmetric)")
    print(f"  Converged in {info['n_iter']} iterations")
    
    print("✓ Fréchet mean tests passed!\n")


if __name__ == "__main__":
    test_log_exp_inverse()
    test_frechet_mean()
    
    print("All SO(3) Fréchet mean tests passed! ✓")