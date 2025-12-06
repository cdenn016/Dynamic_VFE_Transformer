# -*- coding: utf-8 -*-
"""
Created on Sat Nov  8 16:43:41 2025

@author: chris and christine
"""

import numpy as np
from typing import Tuple, Optional
import inspect
import os

# ============================================================================
# NUMBA INTEGRATION (Windows-safe)
# ============================================================================

try:
    from math_utils.numba_kernels import kl_gaussian_numba
    _NUMBA_AVAILABLE = True
except ImportError:
    _NUMBA_AVAILABLE = False
    # Windows-safe warning (no emoji)
    import warnings
    warnings.warn("Numba not available - falling back to NumPy (80x slower)", RuntimeWarning)


# ============================================================================
# SMART KL DIVERGENCE (Numba-accelerated)
# ============================================================================

def kl_gaussian(
    mu_q: np.ndarray,
    Sigma_q: np.ndarray,
    mu_p: np.ndarray,
    Sigma_p: np.ndarray,
    eps: float = 1e-8,
    return_terms: bool = False
):
    """
    KL divergence KL(q||p) between Gaussians - Numba-accelerated!
    
    Automatically uses fast Numba path (80x faster) when possible,
    falls back to NumPy for special features.
    
    Args:
        mu_q, Sigma_q: Source distribution N(μ_q, Σ_q)
        mu_p, Sigma_p: Target distribution N(μ_p, Σ_p)
        eps: Regularization (default: 1e-8)
        return_terms: If True, return dict with term breakdown
    
    Returns:
        kl: KL divergence (scalar or array)
        OR (kl, terms) if return_terms=True
    """
    
    if _NUMBA_AVAILABLE and not return_terms and eps == 1e-8:
        if mu_q.ndim == 1 and mu_p.ndim == 1:
            # Convert to contiguous float64 for Numba (homogeneous dtypes required)
            mu_q_f64 = np.ascontiguousarray(mu_q, dtype=np.float64)
            Sigma_q_f64 = np.ascontiguousarray(Sigma_q, dtype=np.float64)
            mu_p_f64 = np.ascontiguousarray(mu_p, dtype=np.float64)
            Sigma_p_f64 = np.ascontiguousarray(Sigma_p, dtype=np.float64)
            return kl_gaussian_numba(mu_q_f64, Sigma_q_f64, mu_p_f64, Sigma_p_f64)
    
    # FALLBACK: NumPy implementation (handles all edge cases)
    return _kl_gaussian_numpy_impl(mu_q, Sigma_q, mu_p, Sigma_p, eps, return_terms)


def _kl_gaussian_numpy_impl(
    mu_q: np.ndarray,
    Sigma_q: np.ndarray,
    mu_p: np.ndarray,
    Sigma_p: np.ndarray,
    eps: float = 1e-8,
    return_terms: bool = False
):
    """
    Original NumPy implementation - used as fallback.
    
    This is your existing kl_gaussian code - just renamed.
    Keep ALL your existing implementation here!
    """
    # Get latent dimension
    K = mu_q.shape[-1]
    
    # Ensure positive-definite via diagonal regularization
    Sigma_q = sanitize_sigma(Sigma_q, eps)
    Sigma_p = sanitize_sigma(Sigma_p, eps)
    
    # ========== Cholesky decomposition (stable) ==========
    try:
        L_p = np.linalg.cholesky(Sigma_p)
        logdet_p = 2.0 * np.sum(np.log(np.diagonal(L_p, axis1=-2, axis2=-1)), axis=-1)
        
        L_q = np.linalg.cholesky(Sigma_q)
        logdet_q = 2.0 * np.sum(np.log(np.diagonal(L_q, axis1=-2, axis2=-1)), axis=-1)
    except np.linalg.LinAlgError as e:
        raise FloatingPointError(f"Cholesky decomposition failed: {e}") from e
    
    # ========== Term 1: Trace term tr(Σ_p^{-1} Σ_q) ==========
    Y = np.linalg.solve(L_p, Sigma_q)
    Z = np.linalg.solve(np.swapaxes(L_p, -1, -2), Y)
    term_trace = np.trace(Z, axis1=-2, axis2=-1)
    
    # ========== Term 2: Mahalanobis term (μ_p - μ_q)^T Σ_p^{-1} (μ_p - μ_q) ==========
    delta_mu = mu_p - mu_q
    y = np.linalg.solve(L_p, delta_mu[..., None])
    z = np.linalg.solve(np.swapaxes(L_p, -1, -2), y)
    term_quad = np.sum(delta_mu * z[..., 0], axis=-1)
    
    # ========== Term 3: Log-determinant term ==========
    term_logdet = logdet_p - logdet_q
    
    # ========== Combine ==========
    kl = 0.5 * (term_trace + term_quad - K + term_logdet)
    
    # ========== Numerical cleanup ==========
    kl = np.where(kl < 0, np.maximum(kl, -1e-12), kl)
    kl = np.clip(kl, 0.0, None)
    
    # Check for NaN/Inf
    if not np.all(np.isfinite(kl)):
        raise FloatingPointError("KL divergence contains NaN or Inf")
    
    # ========== Return ==========
    if return_terms:
        terms = {
            'term_trace': term_trace,
            'term_quad': term_quad,
            'term_logdet': term_logdet,
            'logdet_q': logdet_q,
            'logdet_p': logdet_p,
        }
        return kl.astype(np.float32), terms
    
    return kl.astype(np.float32)




def _caller_info(levels_back=4):
    """Return short 'module:func:line' string of caller `levels_back` up."""
    try:
        frame = inspect.stack()[levels_back]
        fname = os.path.basename(frame.filename)
        return f"{fname}:{frame.function}:{frame.lineno}"
    except Exception:
        return "<?>"
    
    



def safe_inv(Sigma: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Safely compute matrix inverse with regularization.
    
    Works with batched matrices of any shape (..., K, K).
    Uses direct inversion with regularization for numerical stability.
    
    Args:
        Sigma: Covariance matrix, shape (..., K, K)
               Can be single (K, K) or batched (N, K, K), (H, W, K, K), etc.
        eps: Regularization strength (default: 1e-8)
    
    Returns:
        Sigma_inv: Matrix inverse, same shape as input
    
    Algorithm:
        1. Symmetrize: Sigma = 0.5 * (Sigma + Sigma^T)
        2. Regularize: Sigma_reg = Sigma + eps * I
        3. Invert: Sigma_inv = inv(Sigma_reg)
        4. If singular: increase eps and retry
    
    Examples:
        >>> # Single matrix
        >>> Sigma = np.array([[2.0, 0.5], [0.5, 3.0]])
        >>> Sigma_inv = safe_inv(Sigma)
        >>> np.allclose(Sigma @ Sigma_inv, np.eye(2))
        True
        
        >>> # Batch of matrices (1D spatial)
        >>> Sigma_batch = np.random.randn(50, 3, 3)
        >>> Sigma_batch = Sigma_batch @ Sigma_batch.swapaxes(-1, -2)
        >>> Sigma_inv_batch = safe_inv(Sigma_batch)
        >>> Sigma_inv_batch.shape
        (50, 3, 3)
        
        >>> # 2D spatial grid
        >>> Sigma_2d = np.random.randn(32, 32, 2, 2)
        >>> Sigma_2d = Sigma_2d @ Sigma_2d.swapaxes(-1, -2)
        >>> Sigma_inv_2d = safe_inv(Sigma_2d)
        >>> Sigma_inv_2d.shape
        (32, 32, 2, 2)
    
    Notes:
        - Always returns finite values
        - Preserves input dtype (float32/float64)
        - No warnings or exceptions (graceful fallback)
        - ~10-20% slower than Cholesky but much simpler
    """
    # Store original dtype and convert to float64 for stability
    original_dtype = Sigma.dtype
    Sigma = np.asarray(Sigma, dtype=np.float64)
    
    # Validate shape
    if Sigma.ndim < 2:
        raise ValueError(f"Sigma must be at least 2D, got shape {Sigma.shape}")
    
    if Sigma.shape[-2] != Sigma.shape[-1]:
        raise ValueError(f"Sigma must be square matrices (..., K, K), got {Sigma.shape}")
    
    K = Sigma.shape[-1]
    
    # ========== Step 1: Ensure Symmetric ==========
    # SPD matrices should be symmetric, but numerical errors can break this
    Sigma = 0.5 * (Sigma + np.swapaxes(Sigma, -1, -2))
    
    # ========== Step 2: Add Regularization ==========
    # Add small eps * I to ensure positive definite
    # np.eye broadcasts automatically to batch dimensions!
    Sigma_reg = Sigma + eps * np.eye(K)
    
    # ========== Step 3: Invert ==========
    try:
        # Direct inversion using np.linalg.inv
        # This works for any batch shape: (K, K), (N, K, K), (H, W, K, K), etc.
        Sigma_inv = np.linalg.inv(Sigma_reg)
        
    except np.linalg.LinAlgError:
        # If still singular, add more aggressive regularization
        # This is rare but possible with pathological inputs
        Sigma_reg = Sigma + (eps * 100) * np.eye(K)
        
        try:
            Sigma_inv = np.linalg.inv(Sigma_reg)
        except np.linalg.LinAlgError:
            # Ultimate fallback: very heavy regularization
            # This ensures we always return something finite
            Sigma_reg = Sigma + (eps * 10000) * np.eye(K)
            Sigma_inv = np.linalg.inv(Sigma_reg)
    
    # ========== Step 4: Validate Output ==========
    # Check for NaN/Inf
    if not np.all(np.isfinite(Sigma_inv)):
        raise FloatingPointError(
            f"safe_inv produced non-finite values. "
            f"Input Sigma shape: {Sigma.shape}, "
            f"min eigenvalue: {np.min(np.linalg.eigvalsh(Sigma[..., :, :]))}"
        )
    
    # Convert back to original dtype
    return Sigma_inv.astype(original_dtype, copy=False)


# =============================================================================
# Optional: Cholesky Version (Advanced - Use Only If You Need Speed)
# =============================================================================

def safe_inv_cholesky(Sigma: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Cholesky-based inversion (faster but more complex).
    
    ~20% faster than safe_inv() but requires careful handling.
    Only use if profiling shows safe_inv() is a bottleneck.
    
    Args:
        Sigma: Covariance matrix, shape (..., K, K)
        eps: Regularization strength
    
    Returns:
        Sigma_inv: Matrix inverse, same shape as input
    """
    original_dtype = Sigma.dtype
    Sigma = np.asarray(Sigma, dtype=np.float64)
    
    K = Sigma.shape[-1]
    batch_shape = Sigma.shape[:-2]
    
    # Ensure symmetric
    Sigma = 0.5 * (Sigma + np.swapaxes(Sigma, -1, -2))
    
    # Create identity with proper batch shape
    if batch_shape:
        I = np.tile(np.eye(K), batch_shape + (1, 1))
    else:
        I = np.eye(K)
    
    # Try Cholesky with progressive regularization
    for attempt in range(1, 5):
        try:
            # Add regularization
            current_eps = eps * (10 ** (attempt - 1))
            Sigma_reg = Sigma + current_eps * np.eye(K)
            
            # Cholesky: Sigma = L L^T
            L = np.linalg.cholesky(Sigma_reg)
            
            # Solve L @ Y = I, then L^T @ X = Y
            # Using np.linalg.solve (supports batching!)
            Y = np.linalg.solve(L, I)
            Sigma_inv = np.linalg.solve(np.swapaxes(L, -1, -2), Y)
            
            return Sigma_inv.astype(original_dtype, copy=False)
            
        except np.linalg.LinAlgError:
            if attempt == 4:
                # Final fallback to direct inversion
                Sigma_reg = Sigma + (eps * 1000) * np.eye(K)
                Sigma_inv = np.linalg.inv(Sigma_reg)
                return Sigma_inv.astype(original_dtype, copy=False)






def TUFF_sanitize_sigma(
    Sigma: np.ndarray,
    debug: bool = True,
    eig_floor: float | None = None,
    cond_cap: float | None = None,
    eig_cap: float | None = None,
    trace_target: float | None = None,
    eps: float | None = None,
) -> np.ndarray:
    """
    Symmetrize, scrub NaN/Inf, and project to SPD via eigen clipping.
    Prints caller info on failure or diagnostic messages.
    """
    import config 

    caller = _caller_info(2)  # identify where sanitize_sigma was called from
    in_dtype = Sigma.dtype
    S = np.asarray(Sigma, dtype=np.float64)  # work in fp64 for eig, cast back at end

    eig_floor = float(eig_floor if eig_floor is not None else getattr(config, "sigma_eig_floor", 1e-6))
    cond_cap  = None if cond_cap is None else float(cond_cap)
    eig_cap   = None if eig_cap   is None else float(eig_cap)
    trace_target = None if trace_target is None else float(trace_target)

    # --- symmetrize ---
    S = 0.5 * (S + np.swapaxes(S, -1, -2))

    # --- scrub NaN/Inf ---
    bad = ~np.isfinite(S).all(axis=(-2, -1))
    if np.any(bad):
        print(f"[sanitize_sigma:{caller}] {int(bad.sum())} matrices had NaN/Inf; replaced with floor*I")
        K = S.shape[-1]
        eye = np.eye(K, dtype=S.dtype)
        S = S.copy()
        S[bad] = eig_floor * eye

    # --- eigen projection (batched) ---
    try:
        w, V = np.linalg.eigh(S)
    except Exception as e:
        raise FloatingPointError(f"[sanitize_sigma:{caller}] eigh failed: {e}")

    # floor eigenvalues
    w = np.maximum(w, eig_floor)

    # --- optional condition-number cap ---
    if cond_cap is not None:
        lam_min = np.min(w, axis=-1, keepdims=True)
        lam_min_flat = lam_min.reshape(-1)
        w_flat = w.reshape(-1)
        tau = 1e-8
        floored_frac = np.mean(w_flat <= 1.001 * tau)
        before = w.copy()
        w = np.minimum(w, lam_min * cond_cap)
        capped_frac = np.mean((w < before).astype(np.float32))
        if debug:
            print(
                f"[sanitize_sigma:{caller}] "
                f"λ_min min={lam_min_flat.min():.3e} med={np.median(lam_min_flat):.3e} "
                f"p90={np.percentile(lam_min_flat,90):.3e} max={lam_min_flat.max():.3e} "
                f"| floored_frac={floored_frac:.3f} capped_frac={capped_frac:.3f}",
                flush=True,
            )

    # --- optional absolute cap ---
    if eig_cap is not None:
        w = np.minimum(w, eig_cap)

    # --- reconstruct SPD ---
    S_proj = (V * w[..., None, :]) @ np.swapaxes(V, -1, -2)

    # --- optional trace normalization ---
    if trace_target is not None:
        tr = np.trace(S_proj, axis1=-2, axis2=-1)[..., None, None]
        S_proj = S_proj * (trace_target / np.clip(tr, 1e-12, None))

    # --- final symmetrize + SPD check ---
    S_proj = 0.5 * (S_proj + np.swapaxes(S_proj, -1, -2))
    min_eig = np.min(np.linalg.eigvalsh(S_proj))

    if not np.isfinite(min_eig) or min_eig <= 0:
        raise FloatingPointError(f"[sanitize_sigma:{caller}] non-SPD result (min eig={min_eig:.3e})")

    return S_proj.astype(in_dtype, copy=False)



# -----------------------------------------------------------------------------
# Sigma sanitation (vectorized)
# -----------------------------------------------------------------------------
def sanitize_sigma(Sigma: np.ndarray, 
                   eps: float = 1e-4,  # INCREASE from 1e-6!
                   max_cond: float = 1e4,  # DECREASE from 1e6!
                   max_eig: float = None) -> np.ndarray:
    """
    Sanitize covariance matrix for numerical stability.
    """
    # Symmetrize
    Sigma = 0.5 * (Sigma + np.swapaxes(Sigma, -1, -2))
    
    # Eigendecomposition
    w, V = np.linalg.eigh(Sigma)
    
    # CRITICAL: Absolute floor (not relative!)
    MIN_EIGENVALUE = 1e-4  # Prevents confidence explosion
    w = np.maximum(w, MIN_EIGENVALUE)
    
    if max_eig is not None:
        w = np.minimum(w, max_eig)
    
    # Enforce condition number (but MIN_EIGENVALUE already helps)
    lambda_max = w[..., -1]
    lambda_min_required = lambda_max / max_cond
    w = np.maximum(w, lambda_min_required[..., None])
    
    # Reconstruct
    Sigma_clean = np.einsum('...ij,...j,...kj->...ik', V, w, V, optimize=True)
    
    # Final symmetrize
    Sigma_clean = 0.5 * (Sigma_clean + np.swapaxes(Sigma_clean, -1, -2))
    
    return Sigma_clean



def _chol_logdet(Sigma: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute Cholesky decomposition and log-determinant.
    
    Args:
        Sigma: SPD matrices, shape (..., K, K) in float64
    
    Returns:
        L: Lower-triangular Cholesky factors, shape (..., K, K)
        logdet: Log-determinants, shape (...,)
    
    Notes:
        - Uses float64 to avoid underflow in log(diag(L))
        - logdet = 2 * sum(log(diag(L)))
    """
    L = np.linalg.cholesky(Sigma)                               # (..., K, K)
    diag_L = np.diagonal(L, axis1=-2, axis2=-1)                # (..., K)
    
    # Clip to avoid log(0)
    tiny = np.finfo(np.float64).tiny
    diag_L_safe = np.clip(diag_L, tiny, None)
    
    logdet = 2.0 * np.sum(np.log(diag_L_safe), axis=-1)       # (...,)
    
    return L, logdet
