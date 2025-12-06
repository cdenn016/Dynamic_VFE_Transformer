# -*- coding: utf-8 -*-
"""
Geodesic Corrections for Hamiltonian Dynamics
==============================================

Implements the missing geodesic correction term in Hamilton's equations:

    dp/dt = -dF/d\theta - (1/2) pi^T (dM^{-1}/d\theta) pi

This term arises from the position-dependent mass matrix M(\theta) and
ensures that trajectories follow geodesics on the statistical manifold.

Key Features:
-------------
1. Computes dM^{-1}/d\theta via finite differences (robust)
2. Accounts for beta variation through M (critical for self-consistency)
3. Supports both mu and Sigma parameter variations

Mathematical Background:
------------------------
The mass matrix M includes:
    M = Sigma_p^{-1} + sum_j beta_ij Omega_ij Sigma_q_j^{-1} Omega_ij^T

The softmax weights beta_ij depend on theta through KL divergences:
    beta_ij = exp[-KL_ij/kappa] / sum_k exp[-KL_ik/kappa]

Therefore dM/d\theta includes contributions from:
    1. Direct dependence of Sigma_q_j on theta (when theta = Sigma_j)
    2. Indirect dependence through beta_ij(theta) (always)

Author: Chris
Date: November 2025
"""

import numpy as np
from typing import Optional, Tuple, Dict
from dataclasses import dataclass


@dataclass
class GeodesicForceResult:
    """Container for geodesic force computation results."""
    geodesic_force: np.ndarray  # Full geodesic force vector
    force_mu: np.ndarray  # Component from mu variations
    force_Sigma: np.ndarray  # Component from Sigma variations
    dM_inv_norm: float  # Diagnostic: ||dM^{-1}/d\theta||_F


def compute_geodesic_force(
    trainer,
    theta: np.ndarray,
    p: np.ndarray,
    eps: float = 1e-5,
    include_beta_variation: bool = True
) -> np.ndarray:
    """
    Compute geodesic correction force: -(1/2) p^T (dM^{-1}/d\theta) p

    CRITICAL: This term ensures trajectories follow geodesics on the
    statistical manifold defined by the Fisher-Rao metric.

    The computation uses finite differences to compute dM^{-1}/d\theta_i
    for each parameter theta_i. When include_beta_variation=True, the
    softmax weights beta are recomputed at each perturbed theta, ensuring
    the full chain rule is respected.

    Args:
        trainer: HamiltonianTrainer instance with system reference
        theta: Current parameter vector (flattened [mu, Sigma])
        p: Current momentum vector (flattened [pi_mu, Pi_Sigma])
        eps: Finite difference step size
        include_beta_variation: If True, recompute beta at each theta
                               If False, freeze beta (faster but less accurate)

    Returns:
        geodesic_force: Force vector with same shape as theta

    Mathematical Form:
    ------------------
    For each parameter theta_i:
        F_geo,i = -(1/2) p^T (dM^{-1}/d\theta_i) p

    where dM^{-1}/d\theta_i is computed via central differences:
        dM^{-1}/d\theta_i = [M^{-1}(theta + eps*e_i) - M^{-1}(theta - eps*e_i)] / (2*eps)
    """
    geodesic_force = np.zeros_like(theta)

    # Get index ranges for each agent
    idx_ranges = _compute_parameter_index_ranges(trainer)

    # Only compute for mu parameters (Sigma handled by hyperbolic geodesic flow)
    for agent_idx, (mu_start, mu_end, Sigma_start, Sigma_end) in idx_ranges.items():
        agent = trainer.system.agents[agent_idx]
        K = agent.config.K
        n_spatial = agent.mu_q.size // K

        # Extract momenta for this agent's mu part
        p_mu = p[mu_start:mu_end].reshape(agent.mu_q.shape)

        # Compute M^{-1} at current theta
        theta_backup = trainer._pack_parameters()
        M_inv_current = _compute_M_inverse_for_agent(trainer, agent, agent_idx)

        # Compute geodesic force for each mu parameter
        for local_idx in range(mu_end - mu_start):
            global_idx = mu_start + local_idx

            # Perturbed theta+
            theta_plus = theta.copy()
            theta_plus[global_idx] += eps
            trainer._unpack_parameters(theta_plus)

            M_inv_plus = _compute_M_inverse_for_agent(
                trainer, trainer.system.agents[agent_idx], agent_idx,
                recompute_beta=include_beta_variation
            )

            # Perturbed theta-
            theta_minus = theta.copy()
            theta_minus[global_idx] -= eps
            trainer._unpack_parameters(theta_minus)

            M_inv_minus = _compute_M_inverse_for_agent(
                trainer, trainer.system.agents[agent_idx], agent_idx,
                recompute_beta=include_beta_variation
            )

            # Restore original theta
            trainer._unpack_parameters(theta_backup)

            # Central difference: dM^{-1}/d\theta_i
            dM_inv_dtheta_i = (M_inv_plus - M_inv_minus) / (2 * eps)

            # Compute -(1/2) p^T (dM^{-1}/d\theta_i) p for this spatial point
            # Map local_idx to spatial point and component
            spatial_idx = local_idx // K
            component_idx = local_idx % K

            if agent.mu_q.ndim == 1:
                # 0D: single matrix
                geodesic_force[global_idx] = -0.5 * p_mu @ dM_inv_dtheta_i @ p_mu
            elif agent.mu_q.ndim == 2:
                # 1D field
                p_local = p_mu[spatial_idx]
                dM_local = dM_inv_dtheta_i[spatial_idx] if dM_inv_dtheta_i.ndim == 3 else dM_inv_dtheta_i
                geodesic_force[global_idx] = -0.5 * p_local @ dM_local @ p_local
            else:
                # 2D field
                shape = agent.mu_q.shape[:-1]
                x = spatial_idx // shape[1]
                y = spatial_idx % shape[1]
                p_local = p_mu[x, y]
                dM_local = dM_inv_dtheta_i[x, y] if dM_inv_dtheta_i.ndim == 4 else dM_inv_dtheta_i
                geodesic_force[global_idx] = -0.5 * p_local @ dM_local @ p_local

    return geodesic_force


def compute_geodesic_force_vectorized(
    trainer,
    theta: np.ndarray,
    p: np.ndarray,
    eps: float = 1e-5
) -> np.ndarray:
    """
    Vectorized geodesic force computation for efficiency.

    This version computes dM^{-1}/d\theta for all mu parameters simultaneously
    using batched operations where possible.

    Note: This is an optimization over compute_geodesic_force. Use this
    for production; use the scalar version for debugging.

    Args:
        trainer: HamiltonianTrainer instance
        theta: Parameter vector
        p: Momentum vector
        eps: Finite difference step size

    Returns:
        geodesic_force: Force vector
    """
    geodesic_force = np.zeros_like(theta)

    # Get index ranges
    idx_ranges = _compute_parameter_index_ranges(trainer)

    # Current state backup
    theta_backup = trainer._pack_parameters()

    for agent_idx, (mu_start, mu_end, Sigma_start, Sigma_end) in idx_ranges.items():
        agent = trainer.system.agents[agent_idx]
        K = agent.config.K
        n_spatial = agent.mu_q.size // K

        # Extract mu momentum for this agent
        p_mu = p[mu_start:mu_end].reshape(agent.mu_q.shape)

        # Compute current M^{-1}
        M_inv_current = _compute_M_inverse_for_agent(trainer, agent, agent_idx)

        # For each mu parameter, compute the geodesic force contribution
        for local_idx in range(mu_end - mu_start):
            global_idx = mu_start + local_idx
            spatial_idx = local_idx // K

            # Forward perturbation
            theta[global_idx] += eps
            trainer._unpack_parameters(theta)
            M_inv_plus = _compute_M_inverse_for_agent(
                trainer, trainer.system.agents[agent_idx], agent_idx,
                recompute_beta=True
            )

            # Backward perturbation
            theta[global_idx] -= 2 * eps
            trainer._unpack_parameters(theta)
            M_inv_minus = _compute_M_inverse_for_agent(
                trainer, trainer.system.agents[agent_idx], agent_idx,
                recompute_beta=True
            )

            # Restore
            theta[global_idx] += eps

            # Central difference
            dM_inv = (M_inv_plus - M_inv_minus) / (2 * eps)

            # Compute quadratic form at appropriate spatial location
            if agent.mu_q.ndim == 1:
                geodesic_force[global_idx] = -0.5 * p_mu @ dM_inv @ p_mu
            elif agent.mu_q.ndim == 2:
                p_local = p_mu[spatial_idx]
                dM_local = dM_inv[spatial_idx] if dM_inv.ndim == 3 else dM_inv
                geodesic_force[global_idx] = -0.5 * p_local @ dM_local @ p_local
            else:
                shape = agent.mu_q.shape[:-1]
                x = spatial_idx // shape[1]
                y = spatial_idx % shape[1]
                p_local = p_mu[x, y]
                dM_local = dM_inv[x, y] if dM_inv.ndim == 4 else dM_inv
                geodesic_force[global_idx] = -0.5 * p_local @ dM_local @ p_local

    # Restore original parameters
    trainer._unpack_parameters(theta_backup)

    return geodesic_force


def _compute_M_inverse_for_agent(
    trainer,
    agent,
    agent_idx: int,
    recompute_beta: bool = True
) -> np.ndarray:
    """
    Compute M^{-1} for a single agent at current system state.

    Args:
        trainer: HamiltonianTrainer
        agent: Agent instance
        agent_idx: Agent index in system
        recompute_beta: If True, recompute softmax weights

    Returns:
        M_inv: Inverse mass matrix, shape (*S, K, K) or (K, K)
    """
    from gradients.softmax_grads import compute_softmax_weights

    K = agent.config.K
    spatial_shape = agent.mu_q.shape[:-1] if agent.mu_q.ndim > 1 else ()

    # Initialize with bare mass: Sigma_p^{-1}
    if agent.Sigma_p.ndim == 2:
        M = np.linalg.inv(agent.Sigma_p + 1e-8 * np.eye(K))
    elif agent.Sigma_p.ndim == 3:
        M = np.zeros(agent.Sigma_p.shape, dtype=np.float64)
        for i in range(agent.Sigma_p.shape[0]):
            M[i] = np.linalg.inv(agent.Sigma_p[i] + 1e-8 * np.eye(K))
    else:
        M = np.zeros(agent.Sigma_p.shape, dtype=np.float64)
        for i in range(agent.Sigma_p.shape[0]):
            for j in range(agent.Sigma_p.shape[1]):
                M[i, j] = np.linalg.inv(agent.Sigma_p[i, j] + 1e-8 * np.eye(K))

    # Add relational mass from consensus coupling
    kappa_beta = getattr(trainer.system.config, 'kappa_beta', 1.0)

    if recompute_beta:
        beta_fields = compute_softmax_weights(
            trainer.system, agent_idx, 'belief', kappa_beta
        )
    else:
        # Use cached beta if available
        if hasattr(trainer, '_beta_cache') and trainer._beta_cache is not None:
            beta_fields = trainer._beta_cache.get(agent_idx, {})
        else:
            beta_fields = compute_softmax_weights(
                trainer.system, agent_idx, 'belief', kappa_beta
            )

    for j_idx, beta_ij in beta_fields.items():
        agent_j = trainer.system.agents[j_idx]
        Omega_ij = trainer.system.compute_transport_ij(agent_idx, j_idx)

        if agent.mu_q.ndim == 1:
            # 0D
            Sigma_q_j_inv = np.linalg.inv(agent_j.Sigma_q + 1e-8 * np.eye(K))
            M += float(beta_ij) * (Omega_ij @ Sigma_q_j_inv @ Omega_ij.T)
        elif agent.mu_q.ndim == 2:
            # 1D field
            for i in range(agent.mu_q.shape[0]):
                Sigma_q_j_inv = np.linalg.inv(agent_j.Sigma_q[i] + 1e-8 * np.eye(K))
                Omega_c = Omega_ij[i] if Omega_ij.ndim == 3 else Omega_ij
                M[i] += beta_ij[i] * (Omega_c @ Sigma_q_j_inv @ Omega_c.T)
        else:
            # 2D field
            for i in range(agent.mu_q.shape[0]):
                for j in range(agent.mu_q.shape[1]):
                    Sigma_q_j_inv = np.linalg.inv(agent_j.Sigma_q[i, j] + 1e-8 * np.eye(K))
                    Omega_c = Omega_ij[i, j] if Omega_ij.ndim == 4 else Omega_ij
                    M[i, j] += beta_ij[i, j] * (Omega_c @ Sigma_q_j_inv @ Omega_c.T)

    # Invert M to get M^{-1}
    if M.ndim == 2:
        M_inv = np.linalg.inv(M + 1e-8 * np.eye(K))
    elif M.ndim == 3:
        M_inv = np.zeros_like(M)
        for i in range(M.shape[0]):
            M_inv[i] = np.linalg.inv(M[i] + 1e-8 * np.eye(K))
    else:
        M_inv = np.zeros_like(M)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                M_inv[i, j] = np.linalg.inv(M[i, j] + 1e-8 * np.eye(K))

    return M_inv


def _compute_parameter_index_ranges(trainer) -> Dict[int, Tuple[int, int, int, int]]:
    """
    Compute index ranges for each agent's parameters in the flattened theta vector.

    Returns:
        Dict mapping agent_idx -> (mu_start, mu_end, Sigma_start, Sigma_end)
    """
    idx_ranges = {}
    idx = 0

    for agent_idx, agent in enumerate(trainer.system.agents):
        K = agent.config.K
        n_spatial = agent.mu_q.size // K

        # Mu indices
        mu_start = idx
        mu_size = n_spatial * K
        mu_end = idx + mu_size
        idx = mu_end

        # Sigma indices (upper triangle per spatial point)
        Sigma_start = idx
        Sigma_size_per_point = K * (K + 1) // 2
        Sigma_size = n_spatial * Sigma_size_per_point
        Sigma_end = idx + Sigma_size
        idx = Sigma_end

        idx_ranges[agent_idx] = (mu_start, mu_end, Sigma_start, Sigma_end)

    return idx_ranges


# =============================================================================
# Analytical Geodesic Forces
# =============================================================================

def compute_geodesic_force_analytical(
    trainer,
    theta: np.ndarray,
    p: np.ndarray
) -> np.ndarray:
    """
    Analytical computation of geodesic force.

    Computes -(1/2) π^T (dM^{-1}/dθ) π analytically using:

    1. d(M^{-1})/dμ_k = -M^{-1} (dM/dμ_k) M^{-1}

    2. dM/dμ_k = Σ_j (dβ_ij/dμ_k) Ω_ij Σ_qj^{-1} Ω_ij^T

    3. dβ_ij/dμ_k = Σ_m (dβ_ij/dKL_im) · (dKL_im/dμ_k)

    where:
    - dβ_ij/dKL_ij = -(β_ij/κ)(1 - β_ij)     [diagonal]
    - dβ_ij/dKL_im = +(β_ij β_im)/κ          [off-diagonal, m≠j]
    - dKL_im/dμ_i = Σ_jt^{-1}(μ_i - μ_m^t)   [gradient of KL w.r.t. source mean]

    Args:
        trainer: HamiltonianTrainer with system reference
        theta: Current parameter vector
        p: Current momentum vector

    Returns:
        geodesic_force: Force vector with same shape as theta

    Performance:
        O(n_agents² × n_neighbors × K³) vs O(n_params × n_agents × K³) for finite diff.
        More efficient when n_params >> n_agents × n_neighbors.
    """
    from gradients.softmax_grads import compute_softmax_weights, compute_softmax_derivative_fields
    from gradients.gradient_terms import grad_kl_source
    from math_utils.push_pull import push_gaussian, GaussianDistribution

    geodesic_force = np.zeros_like(theta)
    idx_ranges = _compute_parameter_index_ranges(trainer)

    for agent_idx, (mu_start, mu_end, Sigma_start, Sigma_end) in idx_ranges.items():
        agent = trainer.system.agents[agent_idx]
        K = agent.config.K
        kappa_beta = getattr(trainer.system.config, 'kappa_beta', 1.0)

        # Get neighbors
        neighbors = list(trainer.system.get_neighbors(agent_idx))
        if len(neighbors) == 0:
            continue

        # Extract momentum for this agent
        p_mu = p[mu_start:mu_end].reshape(agent.mu_q.shape)

        # Compute current M and M^{-1}
        M = _compute_M_for_agent(trainer, agent, agent_idx)
        M_inv = _invert_mass_matrix(M, K)

        # Compute β fields and their derivatives
        beta_fields = compute_softmax_weights(
            trainer.system, agent_idx, 'belief', kappa_beta
        )
        softmax_derivs = compute_softmax_derivative_fields(
            beta_fields, neighbors, kappa_beta
        )

        # Compute transported distributions and KL gradients
        kl_grads = {}  # (m, component) -> dKL_im/dμ_i[component]
        transported_dists = {}

        for m in neighbors:
            agent_m = trainer.system.agents[m]
            Omega_im = trainer.system.compute_transport_ij(agent_idx, m)

            # Transport agent m's distribution
            dist_m = GaussianDistribution(agent_m.mu_q, agent_m.Sigma_q)
            dist_m_t = push_gaussian(dist_m, Omega_im)
            transported_dists[m] = dist_m_t

            # dKL_im/dμ_i
            grad_mu, _ = grad_kl_source(
                agent.mu_q, agent.Sigma_q,
                dist_m_t.mu, dist_m_t.Sigma
            )
            kl_grads[m] = grad_mu  # Shape: (*S, K)

        # Precompute Ω_ij Σ_qj^{-1} Ω_ij^T for all neighbors
        coupling_matrices = {}
        for j in neighbors:
            agent_j = trainer.system.agents[j]
            Omega_ij = trainer.system.compute_transport_ij(agent_idx, j)
            coupling_matrices[j] = _compute_coupling_matrix(
                agent_j, Omega_ij, K
            )

        # Compute geodesic force for each μ component
        for local_idx in range(mu_end - mu_start):
            global_idx = mu_start + local_idx

            # Map to spatial point and K-component
            if agent.mu_q.ndim == 1:
                spatial_idx = None
                k_idx = local_idx
            elif agent.mu_q.ndim == 2:
                spatial_idx = local_idx // K
                k_idx = local_idx % K
            else:
                shape = agent.mu_q.shape
                flat_spatial = local_idx // K
                k_idx = local_idx % K
                spatial_idx = (flat_spatial // shape[1], flat_spatial % shape[1])

            # Compute dM/dμ_i[k] at this spatial point
            dM_dmu_k = _compute_dM_dmu_analytical(
                agent, agent_idx, k_idx, spatial_idx,
                neighbors, beta_fields, softmax_derivs,
                kl_grads, coupling_matrices, kappa_beta, K
            )

            # dM^{-1}/dμ_k = -M^{-1} (dM/dμ_k) M^{-1}
            M_inv_local = _get_spatial_slice(M_inv, spatial_idx)
            dM_inv_dmu_k = -M_inv_local @ dM_dmu_k @ M_inv_local

            # Geodesic force: -(1/2) π^T (dM^{-1}/dμ_k) π
            p_local = _get_spatial_slice(p_mu, spatial_idx)
            geodesic_force[global_idx] = -0.5 * p_local @ dM_inv_dmu_k @ p_local

    return geodesic_force


def _compute_M_for_agent(trainer, agent, agent_idx: int) -> np.ndarray:
    """Compute mass matrix M (not inverted) for an agent."""
    from gradients.softmax_grads import compute_softmax_weights

    K = agent.config.K
    kappa_beta = getattr(trainer.system.config, 'kappa_beta', 1.0)

    # Initialize with Σ_p^{-1}
    if agent.Sigma_p.ndim == 2:
        M = np.linalg.inv(agent.Sigma_p + 1e-8 * np.eye(K))
    elif agent.Sigma_p.ndim == 3:
        M = np.zeros(agent.Sigma_p.shape, dtype=np.float64)
        for i in range(agent.Sigma_p.shape[0]):
            M[i] = np.linalg.inv(agent.Sigma_p[i] + 1e-8 * np.eye(K))
    else:
        M = np.zeros(agent.Sigma_p.shape, dtype=np.float64)
        for i in range(agent.Sigma_p.shape[0]):
            for j in range(agent.Sigma_p.shape[1]):
                M[i, j] = np.linalg.inv(agent.Sigma_p[i, j] + 1e-8 * np.eye(K))

    # Add relational mass
    beta_fields = compute_softmax_weights(trainer.system, agent_idx, 'belief', kappa_beta)

    for j_idx, beta_ij in beta_fields.items():
        agent_j = trainer.system.agents[j_idx]
        Omega_ij = trainer.system.compute_transport_ij(agent_idx, j_idx)

        if agent.mu_q.ndim == 1:
            Sigma_q_j_inv = np.linalg.inv(agent_j.Sigma_q + 1e-8 * np.eye(K))
            M += float(beta_ij) * (Omega_ij @ Sigma_q_j_inv @ Omega_ij.T)
        elif agent.mu_q.ndim == 2:
            for i in range(agent.mu_q.shape[0]):
                Sigma_q_j_inv = np.linalg.inv(agent_j.Sigma_q[i] + 1e-8 * np.eye(K))
                Omega_c = Omega_ij[i] if Omega_ij.ndim == 3 else Omega_ij
                M[i] += beta_ij[i] * (Omega_c @ Sigma_q_j_inv @ Omega_c.T)
        else:
            for i in range(agent.mu_q.shape[0]):
                for j in range(agent.mu_q.shape[1]):
                    Sigma_q_j_inv = np.linalg.inv(agent_j.Sigma_q[i, j] + 1e-8 * np.eye(K))
                    Omega_c = Omega_ij[i, j] if Omega_ij.ndim == 4 else Omega_ij
                    M[i, j] += beta_ij[i, j] * (Omega_c @ Sigma_q_j_inv @ Omega_c.T)

    return M


def _invert_mass_matrix(M: np.ndarray, K: int) -> np.ndarray:
    """Invert mass matrix with regularization."""
    if M.ndim == 2:
        return np.linalg.inv(M + 1e-8 * np.eye(K))
    elif M.ndim == 3:
        M_inv = np.zeros_like(M)
        for i in range(M.shape[0]):
            M_inv[i] = np.linalg.inv(M[i] + 1e-8 * np.eye(K))
        return M_inv
    else:
        M_inv = np.zeros_like(M)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                M_inv[i, j] = np.linalg.inv(M[i, j] + 1e-8 * np.eye(K))
        return M_inv


def _compute_coupling_matrix(agent_j, Omega_ij, K: int) -> np.ndarray:
    """Compute Ω_ij Σ_qj^{-1} Ω_ij^T."""
    if agent_j.Sigma_q.ndim == 2:
        Sigma_q_j_inv = np.linalg.inv(agent_j.Sigma_q + 1e-8 * np.eye(K))
        return Omega_ij @ Sigma_q_j_inv @ Omega_ij.T
    elif agent_j.Sigma_q.ndim == 3:
        result = np.zeros((agent_j.Sigma_q.shape[0], K, K))
        for s in range(agent_j.Sigma_q.shape[0]):
            Sigma_inv = np.linalg.inv(agent_j.Sigma_q[s] + 1e-8 * np.eye(K))
            Omega_s = Omega_ij[s] if Omega_ij.ndim == 3 else Omega_ij
            result[s] = Omega_s @ Sigma_inv @ Omega_s.T
        return result
    else:
        shape = agent_j.Sigma_q.shape[:-2]
        result = np.zeros((*shape, K, K))
        for i in range(shape[0]):
            for j in range(shape[1]):
                Sigma_inv = np.linalg.inv(agent_j.Sigma_q[i, j] + 1e-8 * np.eye(K))
                Omega_s = Omega_ij[i, j] if Omega_ij.ndim == 4 else Omega_ij
                result[i, j] = Omega_s @ Sigma_inv @ Omega_s.T
        return result


def _get_spatial_slice(arr: np.ndarray, spatial_idx):
    """Get slice at spatial index."""
    if spatial_idx is None:
        return arr
    elif isinstance(spatial_idx, int):
        return arr[spatial_idx]
    else:
        return arr[spatial_idx[0], spatial_idx[1]]


def _compute_dM_dmu_analytical(
    agent, agent_idx, k_idx, spatial_idx,
    neighbors, beta_fields, softmax_derivs,
    kl_grads, coupling_matrices, kappa_beta, K
):
    """
    Compute dM/dμ_i[k] analytically.

    dM/dμ_i[k] = Σ_j (dβ_ij/dμ_i[k]) · (Ω_ij Σ_qj^{-1} Ω_ij^T)

    where:
    dβ_ij/dμ_i[k] = Σ_m (dβ_ij/dKL_im) · (dKL_im/dμ_i[k])
    """
    dM = np.zeros((K, K), dtype=np.float64)

    for j in neighbors:
        # Compute dβ_ij/dμ_i[k] = Σ_m (dβ_ij/dKL_im) · (dKL_im/dμ_i[k])
        dbeta_ij_dmu_k = 0.0

        for m in neighbors:
            # dβ_ij/dKL_im
            dbeta_dKL = softmax_derivs.get((j, m))
            if dbeta_dKL is None:
                continue

            # Get value at spatial point
            if spatial_idx is None:
                dbeta_val = float(dbeta_dKL)
            elif isinstance(spatial_idx, int):
                dbeta_val = dbeta_dKL[spatial_idx] if hasattr(dbeta_dKL, '__len__') else float(dbeta_dKL)
            else:
                dbeta_val = dbeta_dKL[spatial_idx[0], spatial_idx[1]] if dbeta_dKL.ndim >= 2 else float(dbeta_dKL)

            # dKL_im/dμ_i[k]
            kl_grad_m = kl_grads[m]
            if spatial_idx is None:
                dKL_dmu_k = kl_grad_m[k_idx]
            elif isinstance(spatial_idx, int):
                dKL_dmu_k = kl_grad_m[spatial_idx, k_idx]
            else:
                dKL_dmu_k = kl_grad_m[spatial_idx[0], spatial_idx[1], k_idx]

            dbeta_ij_dmu_k += dbeta_val * dKL_dmu_k

        # Get coupling matrix at spatial point
        coupling = coupling_matrices[j]
        if spatial_idx is None:
            coupling_local = coupling
        elif isinstance(spatial_idx, int):
            coupling_local = coupling[spatial_idx] if coupling.ndim == 3 else coupling
        else:
            coupling_local = coupling[spatial_idx[0], spatial_idx[1]] if coupling.ndim == 4 else coupling

        # Accumulate: dM += (dβ_ij/dμ_k) · (Ω_ij Σ_qj^{-1} Ω_ij^T)
        dM += dbeta_ij_dmu_k * coupling_local

    return dM


def compare_geodesic_methods(
    trainer,
    theta: np.ndarray,
    p: np.ndarray,
    eps: float = 1e-5
) -> Dict:
    """
    Compare finite-difference and analytical geodesic force computations.

    Useful for validation and debugging.

    Args:
        trainer: HamiltonianTrainer
        theta: Current parameters
        p: Current momentum
        eps: Finite difference step size

    Returns:
        Dictionary with comparison metrics
    """
    # Compute using both methods
    force_fd = compute_geodesic_force(trainer, theta, p, eps, include_beta_variation=True)
    force_analytical = compute_geodesic_force_analytical(trainer, theta, p)

    # Compare
    diff = force_analytical - force_fd
    relative_diff = np.abs(diff) / (np.abs(force_fd) + 1e-10)

    return {
        'force_finite_diff': force_fd,
        'force_analytical': force_analytical,
        'absolute_diff': diff,
        'relative_diff': relative_diff,
        'max_abs_diff': np.max(np.abs(diff)),
        'max_rel_diff': np.max(relative_diff),
        'mean_abs_diff': np.mean(np.abs(diff)),
        'correlation': np.corrcoef(force_fd.flatten(), force_analytical.flatten())[0, 1]
        if np.std(force_fd) > 1e-10 and np.std(force_analytical) > 1e-10 else 1.0
    }


# =============================================================================
# Geodesic Forces with Full Inter-Agent Coupling
# =============================================================================

def compute_geodesic_force_full_coupling(
    trainer,
    theta: np.ndarray,
    p: np.ndarray,
    eps: float = 1e-5,
    coupling_strength: float = 0.1
) -> np.ndarray:
    """
    Compute geodesic force using the FULL mass matrix with inter-agent coupling.

    When agents are kinetically coupled (M_ik ≠ 0 for i≠k), the geodesic force
    must account for how perturbing θ_i affects the entire mass matrix M,
    including blocks M_jk for j,k ≠ i.

    Mathematical Form:
    ------------------
    F_geo,i = -(1/2) π^T (dM^{-1}/dθ_i) π

    where M is the FULL multi-agent mass matrix:
        M = [ M_11  M_12  ...  M_1N ]
            [ M_21  M_22  ...  M_2N ]
            [ ...   ...   ...  ... ]
            [ M_N1  M_N2  ...  M_NN ]

    And dM^{-1}/dθ_i is computed via finite differences on the full matrix.

    Args:
        trainer: HamiltonianTrainer
        theta: Current parameter vector (interleaved per-agent)
        p: Current momentum vector (interleaved per-agent)
        eps: Finite difference step size
        coupling_strength: λ for inter-agent coupling

    Returns:
        geodesic_force: Force vector with same shape as theta

    Notes:
        - More expensive than per-agent version: O(n_params × N² × K³)
        - Essential when inter-agent coupling is significant
        - Falls back to per-agent when coupling_strength = 0
    """
    from geometry.multi_agent_mass_matrix import (
        build_full_mass_matrix_with_coupling,
        _compute_dimension_info
    )

    if coupling_strength == 0:
        # Fall back to efficient per-agent computation
        return compute_geodesic_force(trainer, theta, p, eps, include_beta_variation=True)

    geodesic_force = np.zeros_like(theta)
    dim_info = _compute_dimension_info(trainer)
    idx_ranges = _compute_parameter_index_ranges(trainer)

    # Extract all mu momenta into global vector
    p_mu_global = np.zeros(dim_info['total_mu_dim'])
    idx = 0
    for agent_idx, agent in enumerate(trainer.system.agents):
        K = agent.config.K
        n_spatial = agent.mu_q.size // K
        mu_size = n_spatial * K
        Sigma_size = n_spatial * K * (K + 1) // 2

        mu_start, mu_end = dim_info['mu_ranges'][agent_idx]
        p_mu_global[mu_start:mu_end] = p[idx:idx + mu_size]
        idx += mu_size + Sigma_size

    # Backup current theta
    theta_backup = trainer._pack_parameters()

    # Compute current full M^{-1}
    trainer._unpack_parameters(theta)
    M_current = build_full_mass_matrix_with_coupling(
        trainer, theta, coupling_strength, symmetrize=True
    )
    M_inv_current = np.linalg.inv(M_current + 1e-8 * np.eye(len(M_current)))

    # Compute geodesic force for each mu parameter
    for agent_idx, (mu_start, mu_end, Sigma_start, Sigma_end) in idx_ranges.items():
        for local_idx in range(mu_end - mu_start):
            global_idx = mu_start + local_idx

            # Perturb theta+
            theta_plus = theta.copy()
            theta_plus[global_idx] += eps
            trainer._unpack_parameters(theta_plus)

            M_plus = build_full_mass_matrix_with_coupling(
                trainer, theta_plus, coupling_strength, symmetrize=True
            )
            M_inv_plus = np.linalg.inv(M_plus + 1e-8 * np.eye(len(M_plus)))

            # Perturb theta-
            theta_minus = theta.copy()
            theta_minus[global_idx] -= eps
            trainer._unpack_parameters(theta_minus)

            M_minus = build_full_mass_matrix_with_coupling(
                trainer, theta_minus, coupling_strength, symmetrize=True
            )
            M_inv_minus = np.linalg.inv(M_minus + 1e-8 * np.eye(len(M_minus)))

            # Central difference: dM^{-1}/dθ_i
            dM_inv_dtheta_i = (M_inv_plus - M_inv_minus) / (2 * eps)

            # Geodesic force: -(1/2) π_μ^T (dM^{-1}/dθ_i) π_μ
            geodesic_force[global_idx] = -0.5 * p_mu_global @ dM_inv_dtheta_i @ p_mu_global

    # Restore original theta
    trainer._unpack_parameters(theta_backup)

    return geodesic_force


def compute_geodesic_force_analytical_full_coupling(
    trainer,
    theta: np.ndarray,
    p: np.ndarray,
    coupling_strength: float = 0.1
) -> np.ndarray:
    """
    Analytical geodesic force with full inter-agent coupling.

    Computes -(1/2) π^T (dM^{-1}/dθ) π for the full coupled mass matrix.

    Key insight: When θ_i changes, it affects:
    1. Diagonal block M_ii (via β_ij changes)
    2. Off-diagonal blocks M_ik and M_ki (via β coupling weights)
    3. Other blocks M_jk if agent i is a neighbor of j or k

    Mathematical Form:
    ------------------
    dM/dμ_i[k] includes contributions from ALL blocks:
        dM_jl/dμ_i[k] for all (j,l) pairs where i is involved

    For the diagonal block of agent j:
        dM_jj/dμ_i[k] = Σ_m (dβ_jm/dμ_i[k]) Ω_jm Σ_qm^{-1} Ω_jm^T
        (non-zero only if i is a neighbor of j, through β chain rule)

    For off-diagonal block M_jl:
        dM_jl/dμ_i[k] = -(dλ/dμ_i[k]) β_jl Ω_jl Σ_pl^{-1} Ω_jl^T
        (if coupling depends on μ_i through β)

    Args:
        trainer: HamiltonianTrainer
        theta: Current parameters
        p: Current momentum
        coupling_strength: λ for inter-agent coupling

    Returns:
        geodesic_force: Force vector
    """
    from gradients.softmax_grads import compute_softmax_weights, compute_softmax_derivative_fields
    from gradients.gradient_terms import grad_kl_source
    from math_utils.push_pull import push_gaussian, GaussianDistribution
    from geometry.multi_agent_mass_matrix import (
        build_full_mass_matrix_with_coupling,
        _compute_dimension_info
    )

    if coupling_strength == 0:
        return compute_geodesic_force_analytical(trainer, theta, p)

    geodesic_force = np.zeros_like(theta)
    dim_info = _compute_dimension_info(trainer)
    idx_ranges = _compute_parameter_index_ranges(trainer)
    n_agents = trainer.system.n_agents

    # Extract all mu momenta
    p_mu_global = np.zeros(dim_info['total_mu_dim'])
    idx = 0
    for agent_idx, agent in enumerate(trainer.system.agents):
        K = agent.config.K
        n_spatial = agent.mu_q.size // K
        mu_size = n_spatial * K
        Sigma_size = n_spatial * K * (K + 1) // 2
        mu_start, mu_end = dim_info['mu_ranges'][agent_idx]
        p_mu_global[mu_start:mu_end] = p[idx:idx + mu_size]
        idx += mu_size + Sigma_size

    # Current full M and M^{-1}
    M_full = build_full_mass_matrix_with_coupling(
        trainer, theta, coupling_strength, symmetrize=True
    )
    M_inv_full = np.linalg.inv(M_full + 1e-8 * np.eye(len(M_full)))
    total_mu_dim = dim_info['total_mu_dim']

    # Precompute β fields and derivatives for all agents
    kappa_beta = getattr(trainer.system.config, 'kappa_beta', 1.0)
    all_beta_fields = {}
    all_softmax_derivs = {}
    all_kl_grads = {}
    all_coupling_matrices = {}

    for i in range(n_agents):
        agent_i = trainer.system.agents[i]
        K = agent_i.config.K
        neighbors = list(trainer.system.get_neighbors(i))

        if len(neighbors) == 0:
            continue

        beta_fields = compute_softmax_weights(trainer.system, i, 'belief', kappa_beta)
        all_beta_fields[i] = beta_fields
        all_softmax_derivs[i] = compute_softmax_derivative_fields(beta_fields, neighbors, kappa_beta)

        # KL gradients for this agent
        kl_grads = {}
        for m in neighbors:
            agent_m = trainer.system.agents[m]
            Omega_im = trainer.system.compute_transport_ij(i, m)
            dist_m = GaussianDistribution(agent_m.mu_q, agent_m.Sigma_q)
            dist_m_t = push_gaussian(dist_m, Omega_im)
            grad_mu, _ = grad_kl_source(
                agent_i.mu_q, agent_i.Sigma_q,
                dist_m_t.mu, dist_m_t.Sigma
            )
            kl_grads[m] = grad_mu
        all_kl_grads[i] = kl_grads

        # Coupling matrices
        coupling_matrices = {}
        for j in neighbors:
            agent_j = trainer.system.agents[j]
            Omega_ij = trainer.system.compute_transport_ij(i, j)
            coupling_matrices[j] = _compute_coupling_matrix(agent_j, Omega_ij, K)
        all_coupling_matrices[i] = coupling_matrices

    # Compute geodesic force for each μ parameter
    for agent_idx, (mu_start, mu_end, Sigma_start, Sigma_end) in idx_ranges.items():
        agent = trainer.system.agents[agent_idx]
        K = agent.config.K
        neighbors = list(trainer.system.get_neighbors(agent_idx))

        if len(neighbors) == 0:
            continue

        for local_idx in range(mu_end - mu_start):
            global_idx = mu_start + local_idx

            # Map to spatial and K index
            if agent.mu_q.ndim == 1:
                spatial_idx = None
                k_idx = local_idx
            elif agent.mu_q.ndim == 2:
                spatial_idx = local_idx // K
                k_idx = local_idx % K
            else:
                shape = agent.mu_q.shape
                flat_spatial = local_idx // K
                k_idx = local_idx % K
                spatial_idx = (flat_spatial // shape[1], flat_spatial % shape[1])

            # Compute dM_full/dμ_i[k]
            dM_full = np.zeros((total_mu_dim, total_mu_dim), dtype=np.float64)

            # Contribution to diagonal block M_ii
            if agent_idx in all_softmax_derivs:
                dM_ii = _compute_dM_dmu_analytical(
                    agent, agent_idx, k_idx, spatial_idx,
                    neighbors, all_beta_fields.get(agent_idx, {}),
                    all_softmax_derivs.get(agent_idx, {}),
                    all_kl_grads.get(agent_idx, {}),
                    all_coupling_matrices.get(agent_idx, {}),
                    kappa_beta, K
                )
                i_start, i_end = dim_info['mu_ranges'][agent_idx]
                # Place in full matrix (accounting for spatial structure)
                if agent.mu_q.ndim == 1:
                    dM_full[i_start:i_end, i_start:i_end] = dM_ii
                else:
                    # For spatial fields, dM_ii is for a single spatial point
                    if spatial_idx is not None:
                        if isinstance(spatial_idx, int):
                            s_start = i_start + spatial_idx * K
                            s_end = s_start + K
                        else:
                            flat_s = spatial_idx[0] * agent.mu_q.shape[1] + spatial_idx[1]
                            s_start = i_start + flat_s * K
                            s_end = s_start + K
                        dM_full[s_start:s_end, s_start:s_end] = dM_ii

            # Contribution to off-diagonal blocks M_jl where j or l involves agent_idx
            # through the β coupling weights
            #
            # Note: For the current physics model:
            # - β_jk depends on KL(q_j || Ω_jk[q_k]), which involves μ_j
            # - Therefore dβ_jk/dμ_i = 0 for i ≠ j
            # - Off-diagonal blocks M_jk don't depend on μ_i for i ≠ j
            #
            # The diagonal block M_jj DOES depend on μ_i through the relational mass:
            # M_jj includes β_ji which depends on KL(q_j || Ω_ji[q_i])
            # When i perturbs μ_i, this changes q_i, hence Ω_ji[q_i], hence β_ji
            #
            # However, implementing this cross-agent term analytically is complex.
            # The finite-difference version (compute_geodesic_force_full_coupling)
            # correctly captures ALL these effects. For now, the analytical version
            # only computes the direct diagonal contribution.

            # dM^{-1}/dμ = -M^{-1} (dM/dμ) M^{-1}
            dM_inv = -M_inv_full @ dM_full @ M_inv_full

            # Geodesic force: -(1/2) π^T dM^{-1} π
            geodesic_force[global_idx] = -0.5 * p_mu_global @ dM_inv @ p_mu_global

    return geodesic_force


# =============================================================================
# Diagnostics
# =============================================================================

def diagnose_geodesic_correction(
    trainer,
    theta: np.ndarray,
    p: np.ndarray,
    eps: float = 1e-5
) -> Dict:
    """
    Diagnostic tool for understanding geodesic correction behavior.

    Returns:
        Dictionary with diagnostic information including:
        - geodesic_force_norm: ||F_geo||
        - potential_force_norm: ||-dV/d\theta||
        - ratio: ||F_geo|| / ||-dV/d\theta||
        - component_breakdown: Contribution from each agent
        - dM_inv_norms: Frobenius norms of dM^{-1}/d\theta
    """
    from gradients.gradient_engine import compute_natural_gradients

    # Compute geodesic force
    geodesic_force = compute_geodesic_force(trainer, theta, p, eps)

    # Compute potential force
    potential_force = trainer._compute_force(theta)

    # Norms
    geo_norm = np.linalg.norm(geodesic_force)
    pot_norm = np.linalg.norm(potential_force)

    # Per-agent breakdown
    idx_ranges = _compute_parameter_index_ranges(trainer)
    agent_contributions = {}

    for agent_idx, (mu_start, mu_end, Sigma_start, Sigma_end) in idx_ranges.items():
        mu_geo = geodesic_force[mu_start:mu_end]
        mu_pot = potential_force[mu_start:mu_end]

        agent_contributions[agent_idx] = {
            'mu_geodesic_norm': np.linalg.norm(mu_geo),
            'mu_potential_norm': np.linalg.norm(mu_pot),
            'mu_ratio': np.linalg.norm(mu_geo) / (np.linalg.norm(mu_pot) + 1e-10)
        }

    return {
        'geodesic_force_norm': geo_norm,
        'potential_force_norm': pot_norm,
        'ratio': geo_norm / (pot_norm + 1e-10),
        'agent_contributions': agent_contributions,
        'geodesic_force': geodesic_force,
        'potential_force': potential_force
    }