"""
SPD Manifold Retractions
=========================

Proper methods for updating symmetric positive-definite (SPD) matrices
while maintaining the manifold structure.

Mathematical Background:
-----------------------
The space of SPD matrices Sym⁺⁺(K) is a Riemannian manifold with:
- Tangent space T_Σ = Sym(K) (symmetric matrices)
- Metric: Affine-invariant (trace-based) or Euclidean
- Geodesics: Matrix exponential curves

Problem: Given Σ ∈ Sym⁺⁺(K) and tangent vector Δ ∈ T_Σ,
         compute updated Σ_new ∈ Sym⁺⁺(K).

Retraction Methods:
------------------
1. **Exponential Map** (exact geodesic):
   Σ_new = Σ^{1/2} exp(Σ^{-1/2} Δ Σ^{-1/2}) Σ^{1/2}
   
2. **Linear Retraction** (first-order approximation):
   Σ_new = Σ + Δ, then project to SPD
   
3. **Trust Region** (adaptive step size):
   Scale Δ to ensure ||Σ^{-1/2} Δ Σ^{-1/2}|| ≤ ρ

Key Properties:
--------------
- Exponential map preserves positive-definiteness exactly
- Trust region prevents overshooting
- Both methods respect manifold geometry

Usage Pattern:
-------------
1. Compute natural gradient: δΣ = -2 Σ sym(∇_Σ) Σ
2. Apply learning rate: Δ = τ · δΣ
3. Retract: Σ_new = retract_spd(Σ, Δ, mode='exp')

Author: Refactored clean implementation
"""

import numpy as np
from typing import Literal, Optional
from math_utils.numerical_utils import sanitize_sigma, safe_inv

# =============================================================================
# Main Retraction Function
# =============================================================================

import numpy as np

def retract_spd(
    Sigma: np.ndarray,
    delta_Sigma: np.ndarray,
    step_size: float = 1,
    trust_region: float = None,
    max_condition: float = None,
    eps: float = 1e-9,
) -> np.ndarray:
    """
    Gauge-equivariant SPD retraction for covariance matrices.

    Args:
        Sigma: SPD matrix or field, shape (..., K, K)
        delta_Sigma: symmetric tangent, same shape as Sigma
        step_size: scalar step size τ
        trust_region: optional Frobenius-norm cap on the whitened tangent
        max_condition: optional upper bound on condition number of Σ_new
        eps: eigenvalue floor

    Returns:
        Sigma_new: SPD matrix/field, same shape as Sigma
    """
    Sigma = np.asarray(Sigma, dtype=np.float64)
    delta_Sigma = np.asarray(delta_Sigma, dtype=np.float64)

    # Symmetrize (just in case)
    Sigma = 0.5 * (Sigma + np.swapaxes(Sigma, -1, -2))
    delta_Sigma = 0.5 * (delta_Sigma + np.swapaxes(delta_Sigma, -1, -2))

    batch_shape = Sigma.shape[:-2]
    K = Sigma.shape[-1]

    Sigma_flat = Sigma.reshape(-1, K, K)
    d_flat = delta_Sigma.reshape(-1, K, K)

    out = np.empty_like(Sigma_flat)

    for i, (S, dS) in enumerate(zip(Sigma_flat, d_flat)):
        # Eigendecomposition of Σ
        w, V = np.linalg.eigh(S)
        w = np.maximum(w, eps)   # SPD floor
        W_sqrt = np.sqrt(w)
        W_inv_sqrt = 1.0 / W_sqrt

        # Transform tangent to eigenbasis: A = V^T ΔΣ V
        A = V.T @ dS @ V

        # Whitened tangent: B = Λ^{-1/2} A Λ^{-1/2}
        B = (W_inv_sqrt[:, None] * A) * W_inv_sqrt[None, :]

        # Optional trust region on whitened Frobenius norm
        if trust_region is not None:
            n = np.linalg.norm(B, ord="fro")
            if n > trust_region and n > 0.0:
                B = B * (trust_region / n)

        # Exponentiate symmetric B: exp(τ B) = U diag(exp(τ λ)) U^T
        evals, U = np.linalg.eigh(B)

        # CRITICAL: Clip exponent argument to prevent overflow
        # exp(50) ≈ 5e21, exp(-50) ≈ 2e-22 - both still representable
        max_exp_arg = 50.0
        exp_args = np.clip(step_size * evals, -max_exp_arg, max_exp_arg)
        exp_evals = np.exp(exp_args)
        E = (U * exp_evals) @ U.T  # U diag(exp_evals) U^T

        # Map back: Σ_new = V Λ^{1/2} E Λ^{1/2} V^T
        M = V * W_sqrt  # columns scaled by √λ
        S_new = M @ E @ M.T

        # Symmetrize and enforce SPD / condition number if desired
        S_new = 0.5 * (S_new + S_new.T)
        w_new, V_new = np.linalg.eigh(S_new)
        w_new = np.maximum(w_new, eps)

        if max_condition is not None:
            lam_max = w_new.max()
            lam_min = max(lam_max / max_condition, eps)
            w_new = np.clip(w_new, lam_min, lam_max)

        S_new = (V_new * w_new) @ V_new.T
        out[i] = S_new

    return out.reshape(Sigma.shape)


# =============================================================================
# Retraction Implementations
# =============================================================================

def _retract_exponential(
    Q: np.ndarray,
    sqrt_w: np.ndarray,
    R: np.ndarray,
    max_condition: Optional[float],
    eps: float,
) -> np.ndarray:
    """
    Exponential map retraction: Σ_new = Σ^{1/2} exp(R) Σ^{1/2}.
    
    Args:
        Q: Eigenvectors of Σ, shape (..., K, K)
        sqrt_w: sqrt(eigenvalues), shape (..., K)
        R: Whitened tangent vector, shape (..., K, K)
        max_condition: Optional eigenvalue clipping for condition number
        eps: Regularization
    
    Returns:
        Sigma_new: shape (..., K, K)
    """
    K = R.shape[-1]
    
    # Matrix exponential of R
    # R is symmetric, so eigendecomposition is stable
    lam, U = np.linalg.eigh(R)

    # CRITICAL: Clip eigenvalues to prevent overflow in exp()
    # exp(50) ≈ 5e21, exp(-50) ≈ 2e-22 - both still representable
    max_exp_arg = 50.0
    lam = np.clip(lam, -max_exp_arg, max_exp_arg)

    # Optional: further clip to control conditioning
    if max_condition is not None:
        lam = np.clip(lam, -np.log(max_condition), np.log(max_condition))

    exp_lam = np.exp(lam)
    
    # exp(R) = U diag(exp(λ)) U^T
    exp_R = (U * exp_lam[..., None, :]) @ np.swapaxes(U, -1, -2)
    
    # Transform back: Σ_new = Σ^{1/2} exp(R) Σ^{1/2}
    # = Q diag(√λ) exp(R) diag(√λ) Q^T
    exp_R_scaled = (sqrt_w[..., :, None] * exp_R) * sqrt_w[..., None, :]
    Sigma_new = np.einsum("...ik,...kl,...jl->...ij", Q, exp_R_scaled, Q, optimize=True)
    
    # Symmetrize (numerical cleanup)
    Sigma_new = 0.5 * (Sigma_new + np.swapaxes(Sigma_new, -1, -2))
    
    return Sigma_new


def _retract_linear(
    Sigma: np.ndarray,
    Delta: np.ndarray,
    eps: float,
) -> np.ndarray:
    """
    Linear retraction: Σ_new = project_spd(Σ + Δ).
    
    Simple first-order approximation. Fast but only valid for small steps.
    
    Args:
        Sigma: Current matrix, shape (..., K, K)
        Delta: Tangent vector, shape (..., K, K)
        eps: Regularization
    
    Returns:
        Sigma_new: Projected onto SPD, shape (..., K, K)
    """
    Sigma_candidate = Sigma + Delta
    
    # Project onto SPD via eigenvalue clipping
    w, Q = np.linalg.eigh(Sigma_candidate)
    w = np.clip(w, eps, None)  # Force positive
    
    Sigma_new = (Q * w[..., None, :]) @ np.swapaxes(Q, -1, -2)
    Sigma_new = 0.5 * (Sigma_new + np.swapaxes(Sigma_new, -1, -2))
    
    return Sigma_new


def _apply_trust_region(
    R: np.ndarray,
    rho: float,
    eps: float,
) -> np.ndarray:
    """
    Scale whitened tangent vector to trust region: ||R||_F ≤ ρ.
    
    Args:
        R: Whitened tangent vector, shape (..., K, K)
        rho: Trust region radius
        eps: Numerical stability
    
    Returns:
        R_scaled: shape (..., K, K)
    """
    # Compute Frobenius norm
    R_norm = np.linalg.norm(R, axis=(-2, -1), keepdims=True)  # (..., 1, 1)
    
    # Scale factor: min(1, ρ / ||R||)
    scale = np.minimum(1.0, rho / np.maximum(R_norm, eps))
    
    return R * scale




# =============================================================================
# Convenience: Batch Retraction
# =============================================================================

def retract_spd_batch(
    Sigma_batch: np.ndarray,
    Delta_batch: np.ndarray,
    *,
    mode: Literal['exp', 'linear'] = 'exp',
    trust_region: Optional[float] = None,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Vectorized retraction for batch of matrices.
    
    Args:
        Sigma_batch: shape (N, K, K)
        Delta_batch: shape (N, K, K)
        mode, trust_region, eps: same as retract_spd
    
    Returns:
        Sigma_new_batch: shape (N, K, K)
    
    Note:
        This is just a convenience wrapper; retract_spd already handles batches.
    """
    return retract_spd(Sigma_batch, Delta_batch, 
                       mode=mode, trust_region=trust_region, eps=eps)


# =============================================================================
# Distance and Geodesics
# =============================================================================

def spd_distance(Sigma1: np.ndarray, Sigma2: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Affine-invariant distance between SPD matrices.
    
    d(Σ₁, Σ₂) = ||log(Σ₁^{-1/2} Σ₂ Σ₁^{-1/2})||_F
    
    This is the Riemannian distance induced by the natural metric.
    
    Args:
        Sigma1, Sigma2: SPD matrices, shape (..., K, K)
        eps: Regularization
    
    Returns:
        distance: shape (...,)
    
    Examples:
        >>> Sigma1 = np.eye(3)
        >>> Sigma2 = 2.0 * np.eye(3)
        >>> d = spd_distance(Sigma1, Sigma2)
        >>> d
        1.039...  # ≈ sqrt(3 * log(2)^2)
    """
    Sigma1 = sanitize_sigma(Sigma1, eps)
    Sigma2 = sanitize_sigma(Sigma2, eps)
    
    # Σ₁^{-1/2}
    w1, Q1 = np.linalg.eigh(Sigma1)
    w1 = np.clip(w1, eps, None)
    inv_sqrt_w1 = 1.0 / np.sqrt(w1)
    
    # M = Σ₁^{-1/2} Σ₂ Σ₁^{-1/2}
    Sigma2_eig = np.einsum("...ik,...kl,...jl->...ij", 
                           np.swapaxes(Q1, -1, -2), Sigma2, Q1, optimize=True)
    M = (inv_sqrt_w1[..., :, None] * Sigma2_eig) * inv_sqrt_w1[..., None, :]
    
    # log(M)
    w_M, _ = np.linalg.eigh(M)
    w_M = np.clip(w_M, eps, None)
    log_w_M = np.log(w_M)
    
    # ||log(M)||_F = sqrt(sum(log(λ)^2))
    distance = np.sqrt(np.sum(log_w_M * log_w_M, axis=-1))
    
    return distance.astype(np.float32)

