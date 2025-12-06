# -*- coding: utf-8 -*-
"""
Pullback Geometry: "It From Bit" Construction
==============================================

Implements Wheeler's "it from bit" vision through mathematical pullback:
geometric structure emerges from informational dynamics.

Mathematical Framework
---------------------
Agents maintain probability distributions q(c), p(c) as smooth fields over
a base manifold C. These fields induce Riemannian metrics on C via pullback
of the Fisher-Rao metric from the statistical fiber.

The Fisher-Rao Metric (Statistical Fiber)
----------------------------------------
The space of probability distributions B carries the Fisher-Rao metric g_B.
For Gaussian distributions N(μ, Σ), this takes the form:

    g_B(δq, δq) = δμ^T Σ^{-1} δμ + (1/2)tr(Σ^{-1} δΣ Σ^{-1} δΣ)

This metric is intrinsic to the space of distributions, measuring statistical
distinguishability via Fisher information.

Induced Metrics via Pullback
----------------------------
Each smooth section σ: C → B induces a metric on the base manifold:

    G(c) = σ* g_B

For belief section σ^(q), the induced metric components are:

    G^(q)_μν(c) = E_{q(c)}[(∂_μ log q)(∂_ν log q)]

Similarly for prior section σ^(p):

    G^(p)_μν(c) = E_{p(c)}[(∂_μ log p)(∂_ν log p)]

Geometric Interpretation: "It From Bit"
--------------------------------------
These metrics measure how rapidly statistical fields vary across C:
- Large G_μν: beliefs change rapidly → short information distance
- Small G_μν: beliefs nearly constant → large information distance

The metric is NOT put in by hand but emerges from information-processing
dynamics. What appears as spatial distance IS information distance.

For Gaussian Distributions
--------------------------
For q(c) = N(μ(c), Σ(c)), the induced metric has explicit form:

    G_μν(c) = (∂_μ μ)^T Σ^{-1} (∂_ν μ) + (1/2)tr(Σ^{-1}(∂_μ Σ)Σ^{-1}(∂_ν Σ))

Isotropic case Σ(c) = σ² I (constant):

    G_μν(c) = (1/σ²) (∂_μ μ) · (∂_ν μ)

This is a conformal metric with conformal factor 1/σ². High certainty
(small σ) magnifies distances; high uncertainty (large σ) compresses them.

Dual Geometries
---------------
- Epistemic G^(q): belief geometry, dynamical, reflects current uncertainty
- Ontological G^(p): prior geometry, quasi-static, reflects world model

We conjecture G^(p) represents the agent's perceived geometry of reality.
Different agents with different priors perceive different geometries on
the same underlying base manifold C.

Author: Chris & Christine
Date: November 2025
"""

import numpy as np
from typing import Tuple, Optional, Literal
from dataclasses import dataclass

from math_utils.numerical_utils import sanitize_sigma, safe_inv


# =============================================================================
# Fisher-Rao Metric on Statistical Fiber
# =============================================================================

def fisher_rao_metric_gaussian(
    mu: np.ndarray,
    Sigma: np.ndarray,
    delta_mu: np.ndarray,
    delta_Sigma: np.ndarray,
    *,
    eps: float = 1e-8
) -> float:
    """
    Fisher-Rao metric on the space of Gaussian distributions.

    Computes the intrinsic distance between nearby Gaussian distributions
    via the Fisher information metric.

    Args:
        mu: Mean parameters, shape (..., K)
        Sigma: Covariance parameters, shape (..., K, K)
        delta_mu: Tangent vector in mean space, shape (..., K)
        delta_Sigma: Tangent vector in covariance space, shape (..., K, K)
        eps: Regularization for numerical stability

    Returns:
        g_B(δq, δq): Fisher-Rao norm squared, scalar or shape (...)

    Formula:
        g_B(δq, δq) = δμ^T Σ^{-1} δμ + (1/2)tr(Σ^{-1} δΣ Σ^{-1} δΣ)

    Examples:
        >>> mu = np.array([0., 0.])
        >>> Sigma = np.eye(2)
        >>> delta_mu = np.array([0.1, 0.1])
        >>> delta_Sigma = np.zeros((2, 2))
        >>> g = fisher_rao_metric_gaussian(mu, Sigma, delta_mu, delta_Sigma)
        >>> # g ≈ 0.02 = 0.1² + 0.1²
    """
    Sigma = sanitize_sigma(Sigma, eps)
    Sigma_inv = safe_inv(Sigma, eps)

    # Mean contribution: δμ^T Σ^{-1} δμ
    mean_term = np.einsum(
        "...i,...ij,...j->...",
        delta_mu, Sigma_inv, delta_mu,
        optimize=True
    )

    # Covariance contribution: (1/2)tr(Σ^{-1} δΣ Σ^{-1} δΣ)
    # Step 1: Σ^{-1} δΣ
    tmp1 = np.einsum("...ij,...jk->...ik", Sigma_inv, delta_Sigma, optimize=True)
    # Step 2: (Σ^{-1} δΣ) Σ^{-1}
    tmp2 = np.einsum("...ij,...jk->...ik", tmp1, Sigma_inv, optimize=True)
    # Step 3: tr((Σ^{-1} δΣ Σ^{-1}) δΣ)
    cov_term = 0.5 * np.einsum("...ij,...ji->...", tmp2, delta_Sigma, optimize=True)

    return (mean_term + cov_term).astype(np.float32)


def fisher_rao_distance_gaussian(
    mu1: np.ndarray,
    Sigma1: np.ndarray,
    mu2: np.ndarray,
    Sigma2: np.ndarray,
    *,
    eps: float = 1e-8
) -> float:
    """
    Approximate Fisher-Rao distance between two Gaussian distributions.

    Uses linearized approximation: d(q1, q2) ≈ sqrt(g_B(Δq, Δq))
    where Δq = q2 - q1.

    Args:
        mu1, Sigma1: First Gaussian N(μ₁, Σ₁)
        mu2, Sigma2: Second Gaussian N(μ₂, Σ₂)
        eps: Regularization

    Returns:
        Approximate Fisher-Rao distance (scalar or batched)

    Note:
        This is the linearized approximation. For exact distance, use
        the Riemannian geodesic distance (not implemented here).
    """
    delta_mu = mu2 - mu1
    delta_Sigma = Sigma2 - Sigma1

    # Use mean of the two covariances as reference point
    Sigma_avg = 0.5 * (Sigma1 + Sigma2)

    g_squared = fisher_rao_metric_gaussian(
        mu1, Sigma_avg, delta_mu, delta_Sigma, eps=eps
    )

    return np.sqrt(np.maximum(g_squared, 0.0)).astype(np.float32)


# =============================================================================
# Spatial Derivatives (Finite Differences)
# =============================================================================

def compute_spatial_gradients(
    field: np.ndarray,
    dx: float = 1.0,
    *,
    axis: Optional[int] = None,
    periodic: bool = True
) -> np.ndarray:
    """
    Compute spatial gradients of a field using finite differences.

    Args:
        field: Scalar, vector, or tensor field, shape (*spatial, ...)
        dx: Grid spacing (assumed uniform)
        axis: If provided, compute gradient along this spatial axis only
              If None, compute gradients along all spatial axes
        periodic: Use periodic boundary conditions

    Returns:
        Gradients:
            If axis is None: shape (n_spatial_dims, *spatial, ...)
            If axis is int: shape (*spatial, ...)

    Examples:
        >>> # 1D scalar field
        >>> field = np.sin(np.linspace(0, 2*np.pi, 100))
        >>> grad = compute_spatial_gradients(field, dx=2*np.pi/100)
        >>> # grad[0] ≈ cos(x)

        >>> # 2D vector field
        >>> mu_field = np.random.randn(32, 32, 3)  # (H, W, K)
        >>> grads = compute_spatial_gradients(mu_field)  # (2, H, W, K)
        >>> grad_x, grad_y = grads  # Each has shape (H, W, K)
    """
    ndim_spatial = field.ndim if field.ndim > 0 else 0

    if axis is not None:
        # Single axis gradient
        if periodic:
            grad = np.gradient(field, dx, axis=axis, edge_order=1)
        else:
            grad = np.gradient(field, dx, axis=axis, edge_order=2)
        return grad.astype(np.float32)
    else:
        # All spatial axes gradients
        # Determine number of spatial dimensions
        # Assume trailing dimensions are field components (K, K×K, etc.)
        # For now, assume all leading dimensions are spatial
        if ndim_spatial == 0:
            raise ValueError("Cannot compute spatial gradients of 0D field")

        # Compute gradients along all axes
        grads = []
        for ax in range(ndim_spatial):
            if periodic:
                g = np.gradient(field, dx, axis=ax, edge_order=1)
            else:
                g = np.gradient(field, dx, axis=ax, edge_order=2)
            grads.append(g)

        return np.stack(grads, axis=0).astype(np.float32)


# =============================================================================
# Induced Metrics via Pullback
# =============================================================================

@dataclass
class InducedMetric:
    """
    Induced metric tensor on base manifold via pullback.

    Attributes:
        G: Metric tensor, shape (*spatial, n_spatial_dims, n_spatial_dims)
        spatial_shape: Shape of the base manifold grid
        n_spatial_dims: Dimensionality of base manifold
        metric_type: "belief" or "prior"
        eigenvalues: Eigenvalues of G at each point, shape (*spatial, n_spatial_dims)
        eigenvectors: Eigenvectors of G, shape (*spatial, n_spatial_dims, n_spatial_dims)
    """
    G: np.ndarray
    spatial_shape: Tuple[int, ...]
    n_spatial_dims: int
    metric_type: Literal["belief", "prior"]
    eigenvalues: Optional[np.ndarray] = None
    eigenvectors: Optional[np.ndarray] = None

    def compute_spectral_decomposition(self, eps: float = 1e-10):
        """
        Compute eigenvalue decomposition: G = Σ λ_a (e_a ⊗ e_a).

        Updates self.eigenvalues and self.eigenvectors in place.

        Args:
            eps: Regularization for numerical stability
        """
        # Ensure G is symmetric
        G_sym = 0.5 * (self.G + np.swapaxes(self.G, -1, -2))

        # Compute eigendecomposition
        # np.linalg.eigh returns eigenvalues in ascending order
        eigvals, eigvecs = np.linalg.eigh(G_sym)

        # Sort in descending order
        idx = np.argsort(eigvals, axis=-1)[..., ::-1]

        # Gather sorted eigenvalues using take_along_axis
        self.eigenvalues = np.take_along_axis(eigvals, idx, axis=-1)

        # For eigenvectors, we need to reorder columns
        # eigvecs has shape (..., n_dims, n_dims)
        # idx has shape (..., n_dims)
        # We want eigvecs[..., :, idx[..., j]] for each j

        # Expand idx to match eigenvectors shape for indexing columns
        idx_expanded = idx[..., None, :]  # (..., 1, n_dims)

        # Use take_along_axis on the last axis (columns)
        self.eigenvectors = np.take_along_axis(eigvecs, idx_expanded, axis=-1)

    def get_observable_sector(
        self,
        threshold: float,
        *,
        relative: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract observable sector based on eigenvalue threshold.

        Args:
            threshold: Eigenvalue threshold Λ_obs
            relative: If True, threshold is relative to max eigenvalue
                     If False, threshold is absolute

        Returns:
            observable_mask: Boolean mask, shape (*spatial, n_spatial_dims)
            observable_eigenvalues: Filtered eigenvalues

        Definition:
            D_obs = {e_a : λ_a > Λ_obs}
        """
        if self.eigenvalues is None:
            self.compute_spectral_decomposition()

        if relative:
            # Threshold relative to maximum eigenvalue at each point
            max_eigval = np.max(self.eigenvalues, axis=-1, keepdims=True)
            obs_mask = self.eigenvalues > threshold * max_eigval
        else:
            # Absolute threshold
            obs_mask = self.eigenvalues > threshold

        return obs_mask, np.where(obs_mask, self.eigenvalues, 0.0)

    def get_three_sector_decomposition(
        self,
        lambda_obs: float,
        lambda_dark: float,
        *,
        relative: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Decompose into observable, dark, and internal sectors.

        Args:
            lambda_obs: Observable threshold
            lambda_dark: Dark threshold (< lambda_obs)
            relative: Thresholds relative to max eigenvalue

        Returns:
            obs_mask: Observable sector mask
            dark_mask: Dark sector mask
            internal_mask: Internal sector mask

        Definitions:
            D_obs = {e_a : λ_a > Λ_obs}
            D_dark = {e_a : Λ_dark < λ_a ≤ Λ_obs}
            D_internal = {e_a : λ_a ≤ Λ_dark}
        """
        if self.eigenvalues is None:
            self.compute_spectral_decomposition()

        if relative:
            max_eigval = np.max(self.eigenvalues, axis=-1, keepdims=True)
            thresh_obs = lambda_obs * max_eigval
            thresh_dark = lambda_dark * max_eigval
        else:
            thresh_obs = lambda_obs
            thresh_dark = lambda_dark

        obs_mask = self.eigenvalues > thresh_obs
        dark_mask = (self.eigenvalues > thresh_dark) & (self.eigenvalues <= thresh_obs)
        internal_mask = self.eigenvalues <= thresh_dark

        return obs_mask, dark_mask, internal_mask

    def volume_element(self, eps: float = 1e-10) -> np.ndarray:
        """
        Compute volume element sqrt(det G) at each point.

        Returns:
            vol: Volume element, shape (*spatial,)

        Formula:
            dV = sqrt(det G) dx^1 ... dx^n
        """
        # det(G) = product of eigenvalues
        if self.eigenvalues is None:
            self.compute_spectral_decomposition()

        det_G = np.prod(self.eigenvalues, axis=-1)
        return np.sqrt(np.maximum(det_G, eps)).astype(np.float32)


def pullback_metric_gaussian(
    mu_field: np.ndarray,
    Sigma_field: np.ndarray,
    dx: float = 1.0,
    *,
    metric_type: Literal["belief", "prior"] = "belief",
    periodic: bool = True,
    eps: float = 1e-8
) -> InducedMetric:
    """
    Compute induced metric on base manifold via pullback.

    For a Gaussian field q(c) = N(μ(c), Σ(c)), computes the pullback of
    the Fisher-Rao metric from the statistical fiber to the base manifold.

    Args:
        mu_field: Mean field μ(c), shape (*spatial, K)
        Sigma_field: Covariance field Σ(c), shape (*spatial, K, K)
        dx: Grid spacing
        metric_type: "belief" or "prior"
        periodic: Periodic boundary conditions
        eps: Regularization

    Returns:
        InducedMetric object containing:
            G: Metric tensor, shape (*spatial, n_spatial_dims, n_spatial_dims)

    Formula:
        G_μν(c) = (∂_μ μ)^T Σ^{-1} (∂_ν μ)
                  + (1/2)tr(Σ^{-1}(∂_μ Σ)Σ^{-1}(∂_ν Σ))

    Examples:
        >>> # 1D Gaussian field
        >>> x = np.linspace(0, 2*np.pi, 64)
        >>> mu_field = np.stack([np.sin(x), np.cos(x)], axis=-1)  # (64, 2)
        >>> Sigma_field = np.repeat(np.eye(2)[None], 64, axis=0)  # (64, 2, 2)
        >>> metric = pullback_metric_gaussian(mu_field, Sigma_field, dx=2*np.pi/64)
        >>> # metric.G has shape (64, 1, 1) for 1D base manifold

        >>> # 2D Gaussian field
        >>> mu_field = np.random.randn(32, 32, 3)  # (H, W, K)
        >>> Sigma_field = np.repeat(np.eye(3)[None, None], 32, axis=0)
        >>> Sigma_field = np.repeat(Sigma_field, 32, axis=1)  # (H, W, K, K)
        >>> metric = pullback_metric_gaussian(mu_field, Sigma_field)
        >>> # metric.G has shape (32, 32, 2, 2) for 2D base manifold
    """
    # Determine spatial shape and dimensions
    spatial_shape = mu_field.shape[:-1]  # All but last dim (K)
    n_spatial_dims = len(spatial_shape)
    K = mu_field.shape[-1]

    if n_spatial_dims == 0:
        raise ValueError("Cannot compute pullback metric for 0D base manifold")

    # Validate shapes
    if Sigma_field.shape != (*spatial_shape, K, K):
        raise ValueError(
            f"Shape mismatch: mu_field has spatial_shape={spatial_shape} and K={K}, "
            f"but Sigma_field has shape {Sigma_field.shape}. "
            f"Expected Sigma_field shape: {(*spatial_shape, K, K)}"
        )

    # Sanitize covariance
    Sigma_field = sanitize_sigma(Sigma_field, eps)
    Sigma_inv_field = safe_inv(Sigma_field, eps)

    # Compute spatial gradients of μ(c)
    # grad_mu has shape (n_spatial_dims, *spatial, K)
    grad_mu = []
    for axis in range(n_spatial_dims):
        g = compute_spatial_gradients(mu_field, dx=dx, axis=axis, periodic=periodic)
        # Validate gradient shape
        expected_grad_shape = (*spatial_shape, K)
        if g.shape != expected_grad_shape:
            raise ValueError(
                f"Gradient computation error: expected shape {expected_grad_shape}, "
                f"got {g.shape} for axis {axis}. "
                f"mu_field shape: {mu_field.shape}"
            )
        grad_mu.append(g)
    grad_mu = np.stack(grad_mu, axis=0)  # (n_spatial_dims, *spatial, K)

    # Compute spatial gradients of Σ(c)
    # grad_Sigma has shape (n_spatial_dims, *spatial, K, K)
    grad_Sigma = []
    for axis in range(n_spatial_dims):
        g = compute_spatial_gradients(Sigma_field, dx=dx, axis=axis, periodic=periodic)
        grad_Sigma.append(g)
    grad_Sigma = np.stack(grad_Sigma, axis=0)  # (n_spatial_dims, *spatial, K, K)

    # Initialize metric tensor G
    G = np.zeros((*spatial_shape, n_spatial_dims, n_spatial_dims), dtype=np.float32)

    # Compute metric components G_μν
    for mu_idx in range(n_spatial_dims):
        for nu_idx in range(n_spatial_dims):
            # Mean contribution: (∂_μ μ)^T Σ^{-1} (∂_ν μ)
            mean_term = np.einsum(
                "...i,...ij,...j->...",
                grad_mu[mu_idx],
                Sigma_inv_field,
                grad_mu[nu_idx],
                optimize=True
            )

            # Covariance contribution: (1/2)tr(Σ^{-1}(∂_μ Σ)Σ^{-1}(∂_ν Σ))
            # Step 1: Σ^{-1} (∂_μ Σ)
            tmp1 = np.einsum(
                "...ij,...jk->...ik",
                Sigma_inv_field,
                grad_Sigma[mu_idx],
                optimize=True
            )
            # Step 2: (Σ^{-1} ∂_μ Σ) Σ^{-1}
            tmp2 = np.einsum(
                "...ij,...jk->...ik",
                tmp1,
                Sigma_inv_field,
                optimize=True
            )
            # Step 3: tr((Σ^{-1} ∂_μ Σ Σ^{-1}) (∂_ν Σ))
            cov_term = 0.5 * np.einsum(
                "...ij,...ji->...",
                tmp2,
                grad_Sigma[nu_idx],
                optimize=True
            )

            G[..., mu_idx, nu_idx] = mean_term + cov_term

    # Ensure symmetry (should be symmetric, but enforce numerically)
    G = 0.5 * (G + np.swapaxes(G, -1, -2))

    return InducedMetric(
        G=G.astype(np.float32),
        spatial_shape=spatial_shape,
        n_spatial_dims=n_spatial_dims,
        metric_type=metric_type
    )


def pullback_metric_gaussian_isotropic(
    mu_field: np.ndarray,
    sigma_field: np.ndarray,
    dx: float = 1.0,
    *,
    metric_type: Literal["belief", "prior"] = "belief",
    periodic: bool = True,
    eps: float = 1e-8
) -> InducedMetric:
    """
    Simplified pullback for isotropic Gaussian: Σ(c) = σ²(c) I.

    When covariance is isotropic and constant, the metric reduces to:

        G_μν(c) = (1/σ²) (∂_μ μ) · (∂_ν μ)

    This is a conformal metric with conformal factor 1/σ².

    Args:
        mu_field: Mean field μ(c), shape (*spatial, K)
        sigma_field: Standard deviation field σ(c), shape (*spatial,) or scalar
        dx: Grid spacing
        metric_type: "belief" or "prior"
        periodic: Periodic boundary conditions
        eps: Regularization

    Returns:
        InducedMetric with conformal metric tensor

    Examples:
        >>> # Constant uncertainty
        >>> mu_field = np.random.randn(64, 3)
        >>> sigma = 0.5
        >>> metric = pullback_metric_gaussian_isotropic(mu_field, sigma)
        >>> # G_μν ∝ (∂_μ μ) · (∂_ν μ) with factor 1/0.5² = 4
    """
    spatial_shape = mu_field.shape[:-1]
    n_spatial_dims = len(spatial_shape)
    K = mu_field.shape[-1]

    if n_spatial_dims == 0:
        raise ValueError("Cannot compute pullback metric for 0D base manifold")

    # Broadcast sigma to spatial shape if needed
    if np.isscalar(sigma_field):
        sigma_field = np.full(spatial_shape, sigma_field, dtype=np.float32)
    elif sigma_field.shape != spatial_shape:
        raise ValueError(f"sigma_field shape {sigma_field.shape} != spatial shape {spatial_shape}")

    # Conformal factor: 1/σ²
    conformal_factor = 1.0 / (sigma_field**2 + eps)

    # Compute spatial gradients of μ(c)
    grad_mu = []
    for axis in range(n_spatial_dims):
        g = compute_spatial_gradients(mu_field, dx=dx, axis=axis, periodic=periodic)
        grad_mu.append(g)
    grad_mu = np.stack(grad_mu, axis=0)  # (n_spatial_dims, *spatial, K)

    # Initialize metric tensor
    G = np.zeros((*spatial_shape, n_spatial_dims, n_spatial_dims), dtype=np.float32)

    # Compute G_μν = (1/σ²) (∂_μ μ) · (∂_ν μ)
    for mu_idx in range(n_spatial_dims):
        for nu_idx in range(n_spatial_dims):
            dot_product = np.einsum(
                "...i,...i->...",
                grad_mu[mu_idx],
                grad_mu[nu_idx],
                optimize=True
            )
            G[..., mu_idx, nu_idx] = conformal_factor * dot_product

    # Ensure symmetry
    G = 0.5 * (G + np.swapaxes(G, -1, -2))

    return InducedMetric(
        G=G.astype(np.float32),
        spatial_shape=spatial_shape,
        n_spatial_dims=n_spatial_dims,
        metric_type=metric_type
    )


# =============================================================================
# Agent Pullback Metrics
# =============================================================================

def agent_induced_metrics(
    agent,
    dx: float = 1.0,
    *,
    compute_belief: bool = True,
    compute_prior: bool = True,
    periodic: bool = True,
    eps: float = 1e-8
) -> Tuple[Optional[InducedMetric], Optional[InducedMetric]]:
    """
    Compute both belief-induced and prior-induced metrics for an agent.

    Args:
        agent: Agent object with mu_q, Sigma_q, mu_p, Sigma_p fields
        dx: Grid spacing
        compute_belief: Whether to compute G^(q)
        compute_prior: Whether to compute G^(p)
        periodic: Periodic boundary conditions
        eps: Regularization

    Returns:
        (G_belief, G_prior): Tuple of InducedMetric objects
            Either can be None if not requested

    Examples:
        >>> from agent.agents import Agent
        >>> from config import AgentConfig
        >>>
        >>> config = AgentConfig(spatial_shape=(64,), K=3)
        >>> agent = Agent(0, config)
        >>> # ... initialize agent fields ...
        >>>
        >>> G_q, G_p = agent_induced_metrics(agent)
        >>> # G_q: epistemic geometry (belief)
        >>> # G_p: ontological geometry (prior)
    """
    G_belief = None
    G_prior = None

    if compute_belief:
        G_belief = pullback_metric_gaussian(
            agent.mu_q,
            agent.Sigma_q,
            dx=dx,
            metric_type="belief",
            periodic=periodic,
            eps=eps
        )

    if compute_prior:
        G_prior = pullback_metric_gaussian(
            agent.mu_p,
            agent.Sigma_p,
            dx=dx,
            metric_type="prior",
            periodic=periodic,
            eps=eps
        )

    return G_belief, G_prior


# =============================================================================
# Utilities
# =============================================================================

def metric_trace(G: np.ndarray) -> np.ndarray:
    """
    Compute trace of metric tensor at each point.

    Args:
        G: Metric tensor, shape (*spatial, n_dims, n_dims)

    Returns:
        tr(G): Trace, shape (*spatial,)
    """
    return np.trace(G, axis1=-2, axis2=-1).astype(np.float32)


def metric_determinant(G: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """
    Compute determinant of metric tensor at each point.

    Args:
        G: Metric tensor, shape (*spatial, n_dims, n_dims)
        eps: Regularization

    Returns:
        det(G): Determinant, shape (*spatial,)
    """
    det = np.linalg.det(G)
    return np.maximum(det, eps).astype(np.float32)


def metric_inverse(G: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Compute inverse metric tensor at each point.

    Args:
        G: Metric tensor, shape (*spatial, n_dims, n_dims)
        eps: Regularization

    Returns:
        G^{-1}: Inverse metric, shape (*spatial, n_dims, n_dims)
    """
    # Regularize
    G_reg = G + eps * np.eye(G.shape[-1])

    # Compute inverse
    try:
        G_inv = np.linalg.inv(G_reg)
    except np.linalg.LinAlgError:
        # Fallback to pseudoinverse
        G_inv = np.linalg.pinv(G_reg)

    return G_inv.astype(np.float32)