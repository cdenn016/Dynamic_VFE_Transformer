# -*- coding: utf-8 -*-
"""
Created on Sat Nov  8 12:22:11 2025

@author: chris and christine
"""

"""
Fisher-Rao Metric and Natural Gradients
========================================

Natural gradient descent on the statistical manifold of Gaussian distributions.

The Fisher-Rao metric provides the proper Riemannian structure on the space
of probability distributions. For multivariate Gaussians N(μ, Σ), this gives:

Metric Structure:
----------------
The Fisher information matrix for N(μ, Σ) in natural coordinates is:

    G = [Σ^{-1}              0        ]
        [   0     ½(Σ^{-1} ⊗ Σ^{-1}) ]

where ⊗ is Kronecker product. This makes (μ, Σ) coordinates "orthogonal".

Natural Gradient Formulas:
-------------------------
Given Euclidean gradients ∇_μ and ∇_Σ, the natural gradients are:

    **Mean update:**
        δμ = -G_μμ^{-1} ∇_μ = -Σ ∇_μ

    **Covariance update (tangent space):**
        δΣ = -2 Σ sym(∇_Σ) Σ

where sym(M) = ½(M + M^T).

Geometric Interpretation:
------------------------
- Euclidean gradient: steepest descent in coordinate space
- Natural gradient: steepest descent in *probability distribution* space
- Fisher-Rao metric: intrinsic distance between distributions = KL divergence

Key Property:
------------
Natural gradients are **parametrization-invariant**: the update direction
is the same regardless of how we coordinatize the distribution.

Usage in Optimization:
---------------------
1. Compute Euclidean gradients ∇_μ, ∇_Σ (e.g., via math.gradients)
2. Apply natural gradient projection (this module)
3. Update parameters: μ ← μ + τ·δμ, Σ ← retract_Σ(Σ, τ·δΣ)

Author: Refactored clean implementation
"""

from gradients.gradient_terms import grad_self_wrt_q
import numpy as np
from math_utils.numerical_utils import sanitize_sigma, safe_inv
from typing import Tuple
# =============================================================================
# Natural Gradient Projection
# =============================================================================





def natural_gradient_gaussian(
    mu: np.ndarray,
    Sigma: np.ndarray,
    grad_mu_euclidean: np.ndarray,
    grad_Sigma_euclidean: np.ndarray,
    *,
    eps: float = 1e-8,
    assume_symmetric: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project Euclidean gradients to natural gradients via Fisher-Rao metric.
    
    Transforms coordinate-dependent gradients into geometry-aware updates
    on the statistical manifold.
    
    Args:
        mu: Current mean parameters, shape (..., K)
        Sigma: Current covariance parameters, shape (..., K, K)
        grad_mu_euclidean: Euclidean gradient ∂L/∂μ, shape (..., K)
        grad_Sigma_euclidean: Euclidean gradient ∂L/∂Σ, shape (..., K, K)
        eps: Regularization for numerical stability
        assume_symmetric: If True, skip symmetrizing grad_Sigma
    
    Returns:
        delta_mu: Natural gradient for μ, shape (..., K)
        delta_Sigma: Natural gradient for Σ, shape (..., K, K)
    
    Formulas:
        δμ = -Σ ∇_μ
        δΣ = -2 Σ sym(∇_Σ) Σ
    
    Notes:
        - Returned deltas are coordinate updates, not tangent vectors
        - For Σ update, must use proper retraction (see retraction.py)
        - Works for any spatial dimensionality (0D, 1D, ND)
    
    Examples:
        >>> from math.gradients import grad_self_wrt_q
        >>> # Compute Euclidean gradients
        >>> g_mu, g_Sigma = grad_self_wrt_q(mu_q, Sigma_q, mu_p, Sigma_p)
        >>> 
        >>> # Project to natural gradients
        >>> delta_mu, delta_Sigma = natural_gradient_gaussian(
        ...     mu_q, Sigma_q, g_mu, g_Sigma
        ... )
        >>> 
        >>> # Update (with learning rate τ)
        >>> mu_new = mu_q + tau * delta_mu
        >>> Sigma_new = retract_sigma(Sigma_q, tau * delta_Sigma)
    """
    # Convert to float64 for numerical stability
    mu = np.asarray(mu, dtype=np.float64)
    Sigma = np.asarray(Sigma, dtype=np.float64)
    grad_mu = np.asarray(grad_mu_euclidean, dtype=np.float64)
    grad_Sigma = np.asarray(grad_Sigma_euclidean, dtype=np.float64)
    
    # Validate shapes
    if mu.shape[-1] != Sigma.shape[-1]:
        raise ValueError(f"Dimension mismatch: mu has K={mu.shape[-1]}, "
                        f"Sigma has K={Sigma.shape[-1]}")
    
    if grad_mu.shape != mu.shape:
        raise ValueError(f"grad_mu shape {grad_mu.shape} != mu shape {mu.shape}")
    
    if grad_Sigma.shape != Sigma.shape:
        raise ValueError(f"grad_Sigma shape {grad_Sigma.shape} != Sigma shape {Sigma.shape}")
    
    # Ensure Sigma is symmetric and regularized
    Sigma = sanitize_sigma(Sigma, eps)
    
    # Symmetrize gradient if needed
    if not assume_symmetric:
        grad_Sigma = 0.5 * (grad_Sigma + np.swapaxes(grad_Sigma, -1, -2))
    
    # ========== Natural gradient for μ ==========
    # δμ = -Σ ∇_μ
    delta_mu = -np.einsum("...ij,...j->...i", Sigma, grad_mu, optimize=True)
    
    # ========== Natural gradient for Σ ==========
    # δΣ = -2 Σ sym(∇_Σ) Σ
    # Step 1: Σ ∇_Σ
    tmp = np.einsum("...ij,...jk->...ik", Sigma, grad_Sigma, optimize=True)
    
    # Step 2: (Σ ∇_Σ) Σ
    delta_Sigma = -2.0 * np.einsum("...ij,...jk->...ik", tmp, Sigma, optimize=True)
    
    # Symmetrize (should be symmetric, but enforce numerically)
    delta_Sigma = 0.5 * (delta_Sigma + np.swapaxes(delta_Sigma, -1, -2))
    
    # Check for NaN/Inf
    if not (np.all(np.isfinite(delta_mu)) and np.all(np.isfinite(delta_Sigma))):
        raise FloatingPointError("Natural gradient contains NaN or Inf")
    
    return (delta_mu.astype(np.float32, copy=False),
            delta_Sigma.astype(np.float32, copy=False))


# =============================================================================
# Batch Natural Gradient (Efficient for Multiple Distributions)
# =============================================================================

def natural_gradient_batch(
    mu_batch: np.ndarray,
    Sigma_batch: np.ndarray,
    grad_mu_batch: np.ndarray,
    grad_Sigma_batch: np.ndarray,
    *,
    eps: float = 1e-8,
    assume_symmetric: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Vectorized natural gradient for batch of distributions.
    
    More efficient than calling natural_gradient_gaussian in a loop.
    
    Args:
        mu_batch: Means, shape (N, K)
        Sigma_batch: Covariances, shape (N, K, K)
        grad_mu_batch: Euclidean gradients, shape (N, K)
        grad_Sigma_batch: Euclidean gradients, shape (N, K, K)
        eps: Regularization
        assume_symmetric: Skip symmetrizing gradients
    
    Returns:
        delta_mu: Natural gradients, shape (N, K)
        delta_Sigma: Natural gradients, shape (N, K, K)
    
    Note:
        This is just a convenience wrapper around natural_gradient_gaussian
        which already handles arbitrary batch shapes via broadcasting.
    """
    return natural_gradient_gaussian(
        mu_batch, Sigma_batch,
        grad_mu_batch, grad_Sigma_batch,
        eps=eps,
        assume_symmetric=assume_symmetric
    )


# =============================================================================
# Inverse Natural Gradient (Euclidean from Natural)
# =============================================================================

def euclidean_from_natural(
    mu: np.ndarray,
    Sigma: np.ndarray,
    delta_mu_natural: np.ndarray,
    delta_Sigma_natural: np.ndarray,
    *,
    eps: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Invert natural gradient projection: natural → Euclidean.
    
    Given natural gradient updates, recover the original Euclidean gradients.
    Useful for diagnostics or when converting between representations.
    
    Args:
        mu, Sigma: Current distribution parameters
        delta_mu_natural: Natural gradient δμ = -Σ ∇_μ
        delta_Sigma_natural: Natural gradient δΣ = -2 Σ sym(∇_Σ) Σ
        eps: Regularization
    
    Returns:
        grad_mu_euclidean: Recovered ∇_μ
        grad_Sigma_euclidean: Recovered ∇_Σ
    
    Formulas:
        ∇_μ = -Σ^{-1} δμ
        ∇_Σ = -½ Σ^{-1} δΣ Σ^{-1}
    """
    Sigma = sanitize_sigma(Sigma, eps)
    Sigma_inv = safe_inv(Sigma, eps)
    
    # Recover ∇_μ = -Σ^{-1} δμ
    grad_mu = -np.einsum("...ij,...j->...i", Sigma_inv, delta_mu_natural, optimize=True)
    
    # Recover ∇_Σ = -½ Σ^{-1} δΣ Σ^{-1}
    tmp = np.einsum("...ij,...jk->...ik", Sigma_inv, delta_Sigma_natural, optimize=True)
    grad_Sigma = -0.5 * np.einsum("...ij,...jk->...ik", tmp, Sigma_inv, optimize=True)
    grad_Sigma = 0.5 * (grad_Sigma + np.swapaxes(grad_Sigma, -1, -2))
    
    return grad_mu.astype(np.float32), grad_Sigma.astype(np.float32)



