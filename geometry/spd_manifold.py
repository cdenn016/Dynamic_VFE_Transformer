# -*- coding: utf-8 -*-
"""
Created on Sat Nov 22 13:58:52 2025

@author: chris and christine
"""

"""
Symmetric Positive Definite (SPD) Manifold Geometry
====================================================

Geometry of the manifold of symmetric positive definite matrices.

The space SPD(n) = {Σ ∈ ℝⁿˣⁿ : Σ = Σᵀ, Σ ≻ 0} is a Riemannian manifold
with important applications in:
- Covariance matrices (Gaussian beliefs)
- Diffusion tensors (imaging)
- Kernel matrices (machine learning)

Riemannian Metric (Fisher-Rao / Affine-Invariant)
--------------------------------------------------
For tangent vectors Σ̇₁, Σ̇₂ ∈ T_Σ SPD(n) ≅ Sym(n):

    g_Σ(Σ̇₁, Σ̇₂) = (1/4) tr(Σ⁻¹ Σ̇₁ Σ⁻¹ Σ̇₂)

This metric is:
- Affine-invariant: g(AΣAᵀ) = g(Σ) under A ∈ GL(n)
- Fisher information metric for Gaussian families
- Unique (up to scaling) invariant metric

Geodesics
---------
    Σ(t) = Σ₀^{1/2} exp(t Σ₀^{-1/2} log(Σ₀⁻¹ Σ₁) Σ₀^{-1/2}) Σ₀^{1/2}

Geodesic Equation (Euler-Lagrange)
----------------------------------
    Σ̈ = Σ (Σ⁻¹Σ̇)² Σ - (1/2)[Σ̇Σ⁻¹Σ̇ + (Σ̇Σ⁻¹Σ̇)ᵀ]

Exponential Map
---------------
    exp_Σ(V) = Σ^{1/2} exp(Σ^{-1/2} V Σ^{-1/2}) Σ^{1/2}

Logarithm Map
-------------
    log_Σ(Λ) = Σ^{1/2} log(Σ^{-1/2} Λ Σ^{-1/2}) Σ^{1/2}

References:
    - Pennec et al. (2006) "A Riemannian Framework for Tensor Computing"
    - Bhatia (2007) "Positive Definite Matrices"

Author: Chris
Date: November 2025
"""

import numpy as np
from typing import Tuple, Optional
from scipy.linalg import expm, logm, sqrtm


def is_spd(Sigma: np.ndarray, tol: float = 1e-10) -> bool:
    """
    Check if matrix is symmetric positive definite.

    Args:
        Sigma: (n, n) matrix
        tol: Tolerance for symmetry and positivity

    Returns:
        True if Sigma ∈ SPD(n)
    """
    # Check symmetric
    if not np.allclose(Sigma, Sigma.T, atol=tol):
        return False

    # Check positive definite (all eigenvalues > 0)
    eigvals = np.linalg.eigvalsh(Sigma)
    return np.all(eigvals > tol)


def fisher_rao_metric(
    Sigma: np.ndarray,
    Sigma_dot_1: np.ndarray,
    Sigma_dot_2: np.ndarray
) -> float:
    """
    Fisher-Rao / Affine-Invariant metric on SPD(n).

    g_Σ(Σ̇₁, Σ̇₂) = (1/4) tr(Σ⁻¹ Σ̇₁ Σ⁻¹ Σ̇₂)

    Args:
        Sigma: Base point Σ ∈ SPD(n)
        Sigma_dot_1, Sigma_dot_2: Tangent vectors ∈ Sym(n)

    Returns:
        g: Metric value (scalar)
    """
    Sigma_inv = np.linalg.inv(Sigma)
    A = Sigma_inv @ Sigma_dot_1 @ Sigma_inv @ Sigma_dot_2
    return 0.25 * np.trace(A)


def spd_kinetic_energy(Sigma: np.ndarray, Sigma_dot: np.ndarray) -> float:
    """
    Kinetic energy on SPD manifold.

    T = (1/2) g(Σ̇, Σ̇) = (1/8) tr(Σ⁻¹ Σ̇ Σ⁻¹ Σ̇)

    Args:
        Sigma: Covariance Σ ∈ SPD(n)
        Sigma_dot: Velocity Σ̇ ∈ Sym(n)

    Returns:
        T: Kinetic energy
    """
    return fisher_rao_metric(Sigma, Sigma_dot, Sigma_dot)


def spd_geodesic_equation(
    Sigma: np.ndarray,
    Sigma_dot: np.ndarray
) -> np.ndarray:
    """
    Geodesic equation (acceleration) on SPD manifold.

    From Euler-Lagrange: d/dt(∂L/∂Σ̇) - ∂L/∂Σ = 0

    Result:
        Σ̈ = Σ (Σ⁻¹Σ̇)² Σ - (1/2)[Σ̇Σ⁻¹Σ̇ + (Σ̇Σ⁻¹Σ̇)ᵀ]

    Args:
        Sigma: Position Σ ∈ SPD(n)
        Sigma_dot: Velocity Σ̇ ∈ Sym(n)

    Returns:
        Sigma_ddot: Acceleration Σ̈ ∈ Sym(n)
    """
    Sigma_inv = np.linalg.inv(Sigma)

    # A = Σ⁻¹Σ̇
    A = Sigma_inv @ Sigma_dot

    # First term: Σ A² Σ
    term1 = Sigma @ (A @ A) @ Sigma

    # Second term: (1/2)[Σ̇Σ⁻¹Σ̇ + (Σ̇Σ⁻¹Σ̇)ᵀ]
    B = Sigma_dot @ Sigma_inv @ Sigma_dot
    term2 = 0.5 * (B + B.T)

    Sigma_ddot = term1 - term2

    return Sigma_ddot


def spd_exponential_map(
    Sigma: np.ndarray,
    V: np.ndarray
) -> np.ndarray:
    """
    Exponential map: exp_Σ: T_Σ SPD → SPD.

    exp_Σ(V) = Σ^{1/2} exp(Σ^{-1/2} V Σ^{-1/2}) Σ^{1/2}

    Maps tangent vector V at Σ to a point on the manifold.

    Args:
        Sigma: Base point Σ ∈ SPD(n)
        V: Tangent vector ∈ Sym(n)

    Returns:
        Sigma_new: exp_Σ(V) ∈ SPD(n)
    """
    Sigma_sqrt = sqrtm(Sigma).real
    Sigma_inv_sqrt = np.linalg.inv(Sigma_sqrt)

    # W = Σ^{-1/2} V Σ^{-1/2}
    W = Sigma_inv_sqrt @ V @ Sigma_inv_sqrt

    # exp_Σ(V) = Σ^{1/2} exp(W) Σ^{1/2}
    Sigma_new = Sigma_sqrt @ expm(W) @ Sigma_sqrt

    return Sigma_new


def spd_logarithm_map(
    Sigma: np.ndarray,
    Lambda: np.ndarray
) -> np.ndarray:
    """
    Logarithm map: log_Σ: SPD → T_Σ SPD.

    log_Σ(Λ) = Σ^{1/2} log(Σ^{-1/2} Λ Σ^{-1/2}) Σ^{1/2}

    Inverse of exponential map. Maps manifold point to tangent space.

    Args:
        Sigma: Base point Σ ∈ SPD(n)
        Lambda: Target point Λ ∈ SPD(n)

    Returns:
        V: Tangent vector log_Σ(Λ) ∈ Sym(n)
    """
    Sigma_sqrt = sqrtm(Sigma).real
    Sigma_inv_sqrt = np.linalg.inv(Sigma_sqrt)

    # W = Σ^{-1/2} Λ Σ^{-1/2}
    W = Sigma_inv_sqrt @ Lambda @ Sigma_inv_sqrt

    # log_Σ(Λ) = Σ^{1/2} log(W) Σ^{1/2}
    V = Sigma_sqrt @ logm(W).real @ Sigma_sqrt

    return V


def spd_geodesic(
    Sigma_0: np.ndarray,
    Sigma_1: np.ndarray,
    t: float
) -> np.ndarray:
    """
    Geodesic from Σ₀ to Σ₁ evaluated at t ∈ [0, 1].

    γ(t) = Σ₀^{1/2} exp(t Σ₀^{-1/2} log(Σ₀⁻¹ Σ₁) Σ₀^{-1/2}) Σ₀^{1/2}

    Equivalently:
        γ(t) = exp_{Σ₀}(t · log_{Σ₀}(Σ₁))

    Args:
        Sigma_0: Start point Σ₀ ∈ SPD(n)
        Sigma_1: End point Σ₁ ∈ SPD(n)
        t: Parameter ∈ [0, 1]

    Returns:
        Sigma_t: γ(t) ∈ SPD(n)
    """
    # V = log_{Σ₀}(Σ₁)
    V = spd_logarithm_map(Sigma_0, Sigma_1)

    # γ(t) = exp_{Σ₀}(t V)
    Sigma_t = spd_exponential_map(Sigma_0, t * V)

    return Sigma_t


def spd_distance(Sigma_1: np.ndarray, Sigma_2: np.ndarray) -> float:
    """
    Riemannian distance on SPD manifold.

    d(Σ₁, Σ₂) = ||log_{Σ₁}(Σ₂)||_Σ₁
              = ||log(Σ₁^{-1/2} Σ₂ Σ₁^{-1/2})||_F

    where ||·||_F is Frobenius norm.

    Args:
        Sigma_1, Sigma_2: Points ∈ SPD(n)

    Returns:
        d: Geodesic distance
    """
    V = spd_logarithm_map(Sigma_1, Sigma_2)
    return np.linalg.norm(V, 'fro')


def spd_mean(Sigmas: list, max_iter: int = 100, tol: float = 1e-8) -> np.ndarray:
    """
    Fréchet mean (Karcher mean) on SPD manifold.

    Σ̄ = argmin_Σ Σ_i d²(Σ, Σᵢ)

    Computed iteratively:
        Σ_{k+1} = exp_{Σ_k}((1/N) Σ_i log_{Σ_k}(Σᵢ))

    Args:
        Sigmas: List of SPD matrices
        max_iter: Maximum iterations
        tol: Convergence tolerance

    Returns:
        Sigma_mean: Fréchet mean Σ̄ ∈ SPD(n)
    """
    # Initialize at Euclidean mean
    Sigma_mean = sum(Sigmas) / len(Sigmas)

    for iteration in range(max_iter):
        # Compute tangent vectors to all points
        V_sum = np.zeros_like(Sigma_mean)
        for Sigma_i in Sigmas:
            V_sum += spd_logarithm_map(Sigma_mean, Sigma_i)

        V_avg = V_sum / len(Sigmas)

        # Check convergence
        if np.linalg.norm(V_avg, 'fro') < tol:
            break

        # Update mean
        Sigma_mean = spd_exponential_map(Sigma_mean, V_avg)

    return Sigma_mean


def spd_parallel_transport(
    V: np.ndarray,
    Sigma_0: np.ndarray,
    Sigma_1: np.ndarray
) -> np.ndarray:
    """
    Parallel transport of tangent vector along geodesic.

    Transport V ∈ T_{Σ₀} SPD to T_{Σ₁} SPD along the geodesic.

    For SPD manifold with affine-invariant metric:
        Γ(V) = E V Eᵀ
    where E = (Σ₁ Σ₀⁻¹)^{1/2}

    Args:
        V: Tangent vector at Σ₀
        Sigma_0: Start point
        Sigma_1: End point

    Returns:
        V_transported: Tangent vector at Σ₁
    """
    # E = (Σ₁ Σ₀⁻¹)^{1/2}
    Sigma_0_inv = np.linalg.inv(Sigma_0)
    E = sqrtm(Sigma_1 @ Sigma_0_inv).real

    # Γ(V) = E V Eᵀ
    V_transported = E @ V @ E.T

    return V_transported


def project_to_tangent_space(
    Sigma: np.ndarray,
    M: np.ndarray
) -> np.ndarray:
    """
    Project arbitrary matrix M to tangent space T_Σ SPD ≅ Sym(n).

    Tangent space is the space of symmetric matrices.

    Args:
        Sigma: Base point (not actually needed for SPD)
        M: Arbitrary matrix

    Returns:
        M_sym: Symmetric part (1/2)(M + Mᵀ)
    """
    return 0.5 * (M + M.T)


class SPDManifold:
    """
    SPD manifold with Riemannian operations.

    Convenience wrapper for all SPD geometry operations.
    """

    def __init__(self, n: int):
        """
        Args:
            n: Dimension (SPD(n) = n×n matrices)
        """
        self.n = n
        self.dim = n * (n + 1) // 2  # Intrinsic dimension

    def metric(self, Sigma: np.ndarray, V1: np.ndarray, V2: np.ndarray) -> float:
        """Riemannian metric."""
        return fisher_rao_metric(Sigma, V1, V2)

    def exp(self, Sigma: np.ndarray, V: np.ndarray) -> np.ndarray:
        """Exponential map."""
        return spd_exponential_map(Sigma, V)

    def log(self, Sigma: np.ndarray, Lambda: np.ndarray) -> np.ndarray:
        """Logarithm map."""
        return spd_logarithm_map(Sigma, Lambda)

    def geodesic(self, Sigma_0: np.ndarray, Sigma_1: np.ndarray, t: float) -> np.ndarray:
        """Geodesic curve."""
        return spd_geodesic(Sigma_0, Sigma_1, t)

    def distance(self, Sigma_1: np.ndarray, Sigma_2: np.ndarray) -> float:
        """Geodesic distance."""
        return spd_distance(Sigma_1, Sigma_2)

    def geodesic_equation(self, Sigma: np.ndarray, Sigma_dot: np.ndarray) -> np.ndarray:
        """Geodesic equation (acceleration)."""
        return spd_geodesic_equation(Sigma, Sigma_dot)

    def kinetic_energy(self, Sigma: np.ndarray, Sigma_dot: np.ndarray) -> float:
        """Kinetic energy."""
        return spd_kinetic_energy(Sigma, Sigma_dot)

    def random_point(self) -> np.ndarray:
        """Generate random SPD matrix."""
        A = np.random.randn(self.n, self.n)
        return A @ A.T + np.eye(self.n)

    def random_tangent(self, Sigma: np.ndarray) -> np.ndarray:
        """Generate random tangent vector."""
        M = np.random.randn(self.n, self.n)
        return project_to_tangent_space(Sigma, M)