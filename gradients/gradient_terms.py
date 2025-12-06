"""
Euclidean Gradients of KL Divergence
====================================

Compute gradients of KL(q || p) with respect to Gaussian parameters.
These are Euclidean (coordinate) gradients before natural gradient projection.

Mathematical Formulas:
---------------------
For KL(q || p) where q = N(μ_q, Σ_q) and p = N(μ_p, Σ_p):

**Gradients w.r.t. q (first argument):**
    ∂KL/∂μ_q = Σ_p^{-1}(μ_q - μ_p)
    ∂KL/∂Σ_q = ½(Σ_p^{-1} - Σ_q^{-1})

**Gradients w.r.t. p (second argument):**
    ∂KL/∂μ_p = -Σ_p^{-1}(μ_q - μ_p)
    ∂KL/∂Σ_p = ½[Σ_p^{-1} - Σ_p^{-1}Σ_qΣ_p^{-1} - Σ_p^{-1}(Δμ)(Δμ)^TΣ_p^{-1}]

where Δμ = μ_q - μ_p.

Note: These are tangent vectors in Euclidean coordinates.
      For Riemannian optimization, apply Fisher-Rao metric (natural gradient).

Architecture:
------------
- All operations dimension-agnostic (0D, 1D, ND)
- Uses Cholesky decomposition for numerical stability
- Optionally accepts pre-computed inverses for efficiency
- Returns float32 for memory efficiency

Author: Refactored clean implementation
"""

import numpy as np
from typing import Optional, Tuple
from math_utils.numerical_utils import sanitize_sigma, safe_inv

# =============================================================================
# Main Gradient Functions
# =============================================================================
def cholesky_gradient(grad_Sigma: np.ndarray, L: np.ndarray) -> np.ndarray:
    """
    Convert gradient w.r.t. Σ to gradient w.r.t. Cholesky factor L.
    
    For Σ = LL^T where L is lower triangular:
        ∂S/∂L = 2 * (∂S/∂Σ) @ L
    
    Then enforce lower triangular structure.
    
    Mathematical Derivation:
    -----------------------
    Σ = LL^T
    dΣ = dL L^T + L dL^T
    
    For symmetric grad_Sigma:
    tr(grad_Sigma^T dΣ) = tr(grad_Sigma^T dL L^T) + tr(grad_Sigma^T L dL^T)
                        = tr(L^T grad_Sigma^T dL) + tr(dL^T grad_Sigma^T L)
                        = tr(L^T grad_Sigma dL) + tr(L grad_Sigma dL)  [if grad_Sigma symmetric]
                        = 2 tr(L grad_Sigma dL)
                        = 2 tr(grad_Sigma L dL)  [cyclic property]
    
    Therefore: ∂S/∂L = 2 * grad_Sigma @ L
    
    Args:
        grad_Sigma: Gradient w.r.t. Σ (symmetric), shape (*S, K, K)
        L: Cholesky factor (lower triangular), shape (*S, K, K)
    
    Returns:
        grad_L: Gradient w.r.t. L (lower triangular), shape (*S, K, K)
    
    Examples:
        >>> # 0D case
        >>> Sigma = np.array([[2., 1.], [1., 2.]])
        >>> L = np.linalg.cholesky(Sigma)
        >>> grad_Sigma = np.array([[1., 0.], [0., 1.]])  # dS/dΣ = I
        >>> grad_L = cholesky_gradient(grad_Sigma, L)
        >>> # grad_L is lower triangular
        >>> assert np.allclose(grad_L, np.tril(grad_L))
        
        >>> # Spatial case
        >>> Sigma_field = ...  # (10, 10, 3, 3)
        >>> L_field = agent.L_q  # (10, 10, 3, 3)
        >>> grad_Sigma_field = ...  # (10, 10, 3, 3)
        >>> grad_L_field = cholesky_gradient(grad_Sigma_field, L_field)
        >>> # Each slice is lower triangular
    """
    # Matrix multiply: 2 * grad_Sigma @ L
    grad_L = 2.0 * np.einsum('...ij,...jk->...ik', grad_Sigma, L, optimize=True)
    
    # Enforce lower triangular structure
    # (L has zeros in upper triangle, gradient should too)
    grad_L = np.tril(grad_L)
    
    return grad_L.astype(np.float32)




def grad_self_wrt_q(
    mu_q: np.ndarray,
    Sigma_q: np.ndarray,
    mu_p: np.ndarray,
    Sigma_p: np.ndarray,
    *,
    eps: float = 1e-8,
    Sigma_p_inv: Optional[np.ndarray] = None,
    Sigma_q_inv: Optional[np.ndarray] = None,
    assume_sanitized: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute ∂KL(q||p)/∂(μ_q, Σ_q).
    
    Returns the Euclidean gradients of KL divergence with respect to 
    the first argument (q distribution).
    
    Args:
        mu_q: Mean of q, shape (..., K)
        Sigma_q: Covariance of q, shape (..., K, K)
        mu_p: Mean of p, shape (..., K)
        Sigma_p: Covariance of p, shape (..., K, K)
        eps: Regularization for matrix inversion
        Sigma_p_inv: Optional pre-computed Σ_p^{-1}, shape (..., K, K)
        Sigma_q_inv: Optional pre-computed Σ_q^{-1}, shape (..., K, K)
        assume_sanitized: If True, skip symmetrization/regularization
    
    Returns:
        grad_mu: ∂KL/∂μ_q, shape (..., K)
        grad_Sigma: ∂KL/∂Σ_q, shape (..., K, K), symmetric
    
    Formula:
        grad_μ_q = Σ_p^{-1}(μ_q - μ_p)
        grad_Σ_q = ½(Σ_p^{-1} - Σ_q^{-1})
    
    Examples:
        >>> mu_q = np.array([0., 0.])
        >>> Sigma_q = np.eye(2)
        >>> mu_p = np.array([1., 0.])
        >>> Sigma_p = np.eye(2)
        >>> grad_mu, grad_Sigma = grad_self_wrt_q(mu_q, Sigma_q, mu_p, Sigma_p)
        >>> grad_mu
        array([-1.,  0.])  # Points from μ_q toward μ_p
    """
    # Convert to float64 for stability
    mu_q = np.asarray(mu_q, dtype=np.float64)
    mu_p = np.asarray(mu_p, dtype=np.float64)
    Sigma_q = np.asarray(Sigma_q, dtype=np.float64)
    Sigma_p = np.asarray(Sigma_p, dtype=np.float64)
    
    # Ensure symmetric + regularized if needed
    if not assume_sanitized:
        Sigma_q =sanitize_sigma(Sigma_q, eps)
        Sigma_p = sanitize_sigma(Sigma_p, eps)
    
    # Compute Σ_p^{-1} if not provided
    if Sigma_p_inv is None:
        Sigma_p_inv = safe_inv(Sigma_p, eps)
    else:
        Sigma_p_inv = np.asarray(Sigma_p_inv, dtype=np.float64)
    
    # ========== Gradient w.r.t. μ_q ==========
    delta_mu = mu_q - mu_p                                      # (..., K)
    grad_mu = np.einsum("...ij,...j->...i", 
                        Sigma_p_inv, delta_mu, optimize=True)  # (..., K)
    
    # ========== Gradient w.r.t. Σ_q ==========
    # Compute Σ_q^{-1} if not provided
    if Sigma_q_inv is None:
        Sigma_q_inv = safe_inv(Sigma_q, eps)
    else:
        Sigma_q_inv = np.asarray(Sigma_q_inv, dtype=np.float64)
    
    grad_Sigma = 0.5 * (Sigma_p_inv - Sigma_q_inv)            # (..., K, K)
    
    # Symmetrize (should already be symmetric, but enforce numerically)
    grad_Sigma = 0.5 * (grad_Sigma + np.swapaxes(grad_Sigma, -1, -2))
    
    return (grad_mu.astype(np.float32, copy=False),
            grad_Sigma.astype(np.float32, copy=False))


def grad_self_wrt_p(
    mu_q: np.ndarray,
    Sigma_q: np.ndarray,
    mu_p: np.ndarray,
    Sigma_p: np.ndarray,
    *,
    eps: float = 1e-8,
    Sigma_p_inv: Optional[np.ndarray] = None,
    assume_sanitized: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute ∂KL(q||p)/∂(μ_p, Σ_p).
    
    Returns the Euclidean gradients of KL divergence with respect to 
    the second argument (p distribution).
    
    Args:
        mu_q: Mean of q, shape (..., K)
        Sigma_q: Covariance of q, shape (..., K, K)
        mu_p: Mean of p, shape (..., K)
        Sigma_p: Covariance of p, shape (..., K, K)
        eps: Regularization for matrix inversion
        Sigma_p_inv: Optional pre-computed Σ_p^{-1}, shape (..., K, K)
        assume_sanitized: If True, skip symmetrization/regularization
    
    Returns:
        grad_mu: ∂KL/∂μ_p, shape (..., K)
        grad_Sigma: ∂KL/∂Σ_p, shape (..., K, K), symmetric
    
    Formula:
        grad_μ_p = -Σ_p^{-1}(μ_q - μ_p)
        grad_Σ_p = ½[Σ_p^{-1} - v v^T - Σ_p^{-1}Σ_qΣ_p^{-1}]
        where v = Σ_p^{-1}Δμ
    

    """
    # Convert to float64 for stability
    mu_q = np.asarray(mu_q, dtype=np.float64)
    mu_p = np.asarray(mu_p, dtype=np.float64)
    Sigma_q = np.asarray(Sigma_q, dtype=np.float64)
    Sigma_p = np.asarray(Sigma_p, dtype=np.float64)
    
    # Ensure symmetric + regularized if needed
    if not assume_sanitized:
        Sigma_q = sanitize_sigma(Sigma_q, eps)
        Sigma_p = sanitize_sigma(Sigma_p, eps)
    
    # Compute Σ_p^{-1} if not provided
    if Sigma_p_inv is None:
        Sigma_p_inv = safe_inv(Sigma_p, eps)
    else:
        Sigma_p_inv = np.asarray(Sigma_p_inv, dtype=np.float64)
    
    # ========== Gradient w.r.t. μ_p ==========
    delta_mu = mu_q - mu_p                                      # (..., K)
    v = np.einsum("...ij,...j->...i", 
                  Sigma_p_inv, delta_mu, optimize=True)        # (..., K)
    grad_mu = -v                                                # (..., K)
    
    # ========== Gradient w.r.t. Σ_p ==========
    # Term 1: Σ_p^{-1}
    # Term 2: -v v^T = -Σ_p^{-1}(Δμ)(Δμ)^TΣ_p^{-1}
    vvT = np.einsum("...i,...j->...ij", v, v, optimize=True)  # (..., K, K)
    
    # Term 3: -Σ_p^{-1}Σ_qΣ_p^{-1}
    tmp = np.einsum("...ij,...jk->...ik", 
                    Sigma_p_inv, Sigma_q, optimize=True)       # (..., K, K)
    term3 = np.einsum("...ij,...jk->...ik", 
                      tmp, Sigma_p_inv, optimize=True)         # (..., K, K)
    
    grad_Sigma = 0.5 * (Sigma_p_inv - vvT - term3)            # (..., K, K)
    
    # Symmetrize
    grad_Sigma = 0.5 * (grad_Sigma + np.swapaxes(grad_Sigma, -1, -2))
    
    return (grad_mu.astype(np.float32, copy=False),
            grad_Sigma.astype(np.float32, copy=False))


# =============================================================================
# Convenience: All Four Gradients at Once
# =============================================================================



def grad_kl_source(
    
    mu_i: np.ndarray,
    Sigma_i: np.ndarray,
    mu_j_t: np.ndarray,
    Sigma_j_t: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    ∂KL(i || j')/∂(μ_i, Σ_i) where j' is transported.
    
    Dimension-agnostic.
    """
    mu_i = np.asarray(mu_i, dtype=np.float64)
    mu_j_t = np.asarray(mu_j_t, dtype=np.float64)
    Sigma_i = np.asarray(Sigma_i, dtype=np.float64)
    Sigma_j_t = np.asarray(Sigma_j_t, dtype=np.float64)
    
    Sigma_j_t_inv = safe_inv(Sigma_j_t, eps = 1e-8)
    
    delta = mu_i - mu_j_t
    grad_mu = np.einsum('...ij,...j->...i', Sigma_j_t_inv, delta, optimize=True)
    
    Sigma_i_inv = safe_inv(Sigma_i, eps = 1e-8)
    grad_Sigma = 0.5 * (Sigma_j_t_inv - Sigma_i_inv)
    grad_Sigma = 0.5 * (grad_Sigma + np.swapaxes(grad_Sigma, -1, -2))
    
    return grad_mu.astype(np.float32), grad_Sigma.astype(np.float32)

def grad_kl_target(
    
    mu_i: np.ndarray,
    Sigma_i: np.ndarray,
    mu_j_t: np.ndarray,
    Sigma_j_t: np.ndarray,
    Omega_ij: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    ∂KL(i || j')/∂(μ_j, Σ_j) via backprop through transport.
    
    Dimension-agnostic.
    """
    mu_i = np.asarray(mu_i, dtype=np.float64)
    mu_j_t = np.asarray(mu_j_t, dtype=np.float64)
    Sigma_i = np.asarray(Sigma_i, dtype=np.float64)
    Sigma_j_t = np.asarray(Sigma_j_t, dtype=np.float64)
    Omega_ij = np.asarray(Omega_ij, dtype=np.float64)
    
    Sigma_j_t_inv = safe_inv(Sigma_j_t, eps = 1e-8)
    
    delta = mu_i - mu_j_t
    
    # Chain rule for μ: ∂L/∂μ_j' · ∂μ_j'/∂μ_j
    grad_mu_prime = -np.einsum('...ij,...j->...i', Sigma_j_t_inv, delta, optimize=True)
    grad_mu_j = np.einsum('...ji,...j->...i', Omega_ij, grad_mu_prime, optimize=True)
    
    # Chain rule for Σ (more complex)
    v = np.einsum('...ij,...j->...i', Sigma_j_t_inv, delta, optimize=True)
    vvT = np.einsum('...i,...j->...ij', v, v, optimize=True)
   
    
    term1 = -np.einsum('...ij,...jk,...kl->...il',
                      Sigma_j_t_inv, Sigma_i, Sigma_j_t_inv,
                      optimize=True)
    grad_Sigma_prime = 0.5 * (term1 - vvT + Sigma_j_t_inv)
    
    # Transform back: Ω^T grad Ω
    grad_Sigma_j = np.einsum('...ji,...jk,...kl->...il',
                            Omega_ij, grad_Sigma_prime,
                            np.swapaxes(Omega_ij, -1, -2),
                            optimize=True)
    grad_Sigma_j = 0.5 * (grad_Sigma_j + np.swapaxes(grad_Sigma_j, -1, -2))
    
    return grad_mu_j.astype(np.float32), grad_Sigma_j.astype(np.float32)


def grad_kl_wrt_transport(
    mu_i: np.ndarray,
    Sigma_i: np.ndarray,
    mu_j: np.ndarray,
    Sigma_j: np.ndarray,
    Omega_ij: np.ndarray,
    *,
    eps: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    KL(N(μ_i, Σ_i) || N(Ω μ_j, Ω Σ_j Ωᵀ)) gradients w.r.t. Ω.

    Returns:
        grad_mu_factor: for the mean transport term, shape (..., K, K)
        grad_Sigma_factor: covariance factor (here set to 0 for clarity)
    """
    mu_i = np.asarray(mu_i, dtype=np.float64)
    Sigma_i = np.asarray(Sigma_i, dtype=np.float64)
    mu_j = np.asarray(mu_j, dtype=np.float64)
    Sigma_j = np.asarray(Sigma_j, dtype=np.float64)
    Omega_ij = np.asarray(Omega_ij, dtype=np.float64)

    # Sanitize
    Sigma_i = sanitize_sigma(Sigma_i, eps)
    Sigma_j = sanitize_sigma(Sigma_j, eps)

    # Transport
    mu_j_transported = np.einsum('...ij,...j->...i', Omega_ij, mu_j, optimize=True)
    Sigma_j_transported = np.einsum(
        '...ij,...jk,...lk->...il',
        Omega_ij, Sigma_j, Omega_ij,
        optimize=True
    )
    Sigma_j_transported = sanitize_sigma(Sigma_j_transported, eps)

    # Inverse of transported target covariance
    Sigma_j_t_inv = safe_inv(Sigma_j_transported, eps)

    # Mean gradient: ∂KL/∂μ'_j = Σ'_j⁻¹(μ'_j - μ_i)
    diff = mu_j_transported - mu_i
    grad_wrt_mu_transported = np.einsum(
        '...ij,...j->...i',
        Sigma_j_t_inv, diff,
        optimize=True
    )  # (..., K)

    # Chain rule to Ω via μ'_j = Ω μ_j
    grad_mu_factor = np.einsum(
        '...i,...j->...ij',
        grad_wrt_mu_transported, mu_j,
        optimize=True
    )  # (..., K, K)

    # For debugging / cleanliness, drop the approximate Σ term
    grad_Sigma_factor = np.zeros_like(grad_mu_factor)

    return (
        grad_mu_factor.astype(np.float32, copy=False),
        grad_Sigma_factor.astype(np.float32, copy=False),
    )




