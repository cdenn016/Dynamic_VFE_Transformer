# -*- coding: utf-8 -*-
"""
Multi-Agent Mass Matrix Structure
=================================

Implements the full block-structured mass matrix for multi-agent Hamiltonian dynamics:

    M = [ M_{11}  M_{12}  ...  M_{1N} ]
        [ M_{21}  M_{22}  ...  M_{2N} ]
        [  ...     ...   ...   ...    ]
        [ M_{N1}  M_{N2}  ...  M_{NN} ]

where each block M_{ik} has further structure:

    M_{ik} = [ M_{ik}^{mu mu}   M_{ik}^{mu Sigma} ]
             [ M_{ik}^{Sigma mu} M_{ik}^{Sigma Sigma} ]

Block Definitions:
------------------
1. Diagonal blocks M_{ii}:
   - M_{ii}^{mu mu} = Sigma_p^{-1} + sum_j beta_ij Omega_ij Sigma_qj^{-1} Omega_ij^T
   - M_{ii}^{Sigma Sigma} = (1/2)(Sigma^{-1} ⊗ Sigma^{-1}) for SPD metric
   - M_{ii}^{mu Sigma} = 0 for standard Gaussian parametrization

2. Off-diagonal blocks M_{ik} for i != k:
   - Currently set to zero (no kinetic coupling between agents)
   - Cross-coupling happens through:
     a) The potential (free energy gradients)
     b) The geodesic correction (dM/dtheta includes dbeta/dtheta)

Physical Interpretation:
------------------------
- Diagonal blocks: Each agent's own "inertia" on the statistical manifold
- Off-diagonal blocks: Would represent kinetic coupling (momentum exchange)
- Current model: Coupling is purely in potential, not kinetic

Author: Chris
Date: November 2025
"""

import numpy as np
from typing import Optional, Dict, Tuple, List
from dataclasses import dataclass


@dataclass
class MassMatrixDiagnostics:
    """Diagnostic information about the mass matrix structure."""
    total_dim: int                    # Total dimension of theta
    n_agents: int                     # Number of agents
    per_agent_dims: Dict[int, int]    # agent_idx -> dimension
    condition_numbers: Dict[int, float]  # Per-agent condition numbers
    global_condition_number: float    # Full matrix condition number
    diagonal_dominance: float         # Ratio of diagonal to off-diagonal
    min_eigenvalue: float             # Smallest eigenvalue (stability check)


def build_full_mass_matrix(
    trainer,
    theta: np.ndarray,
    include_sigma_blocks: bool = False,
    include_off_diagonal: bool = False
) -> np.ndarray:
    """
    Build the full multi-agent mass matrix.

    This constructs the complete block-structured mass matrix M for all agents.
    Currently implements diagonal blocks only (off-diagonal = 0).

    Args:
        trainer: HamiltonianTrainer with system reference
        theta: Current parameter vector (for parameter-dependent mass)
        include_sigma_blocks: If True, include M^{Sigma Sigma} blocks
                             If False, only compute M^{mu mu} (faster)
        include_off_diagonal: If True, compute M_{ik} for i != k
                             Currently not implemented (always zero)

    Returns:
        M: Full mass matrix, shape (total_dim, total_dim)

    Notes:
        The returned matrix is symmetric positive definite.
        For efficiency, only the mu-part is typically needed since
        Sigma dynamics use hyperbolic geodesic flow (Sigma @ Pi @ Sigma).
    """
    from gradients.softmax_grads import compute_softmax_weights

    system = trainer.system
    n_agents = system.n_agents

    # First pass: compute total dimensions and index mapping
    dim_info = _compute_dimension_info(trainer)
    total_mu_dim = dim_info['total_mu_dim']
    total_sigma_dim = dim_info['total_sigma_dim']

    if include_sigma_blocks:
        total_dim = total_mu_dim + total_sigma_dim
    else:
        total_dim = total_mu_dim

    # Initialize full matrix
    M = np.zeros((total_dim, total_dim), dtype=np.float64)

    # Build diagonal blocks for each agent
    for agent_idx, agent in enumerate(system.agents):
        K = agent.config.K
        kappa_beta = getattr(system.config, 'kappa_beta', 1.0)

        # Get this agent's index range
        mu_start = dim_info['mu_ranges'][agent_idx][0]
        mu_end = dim_info['mu_ranges'][agent_idx][1]

        # Compute M_{ii}^{mu mu} (same as _compute_complete_mass_matrix)
        M_ii_mu = _compute_mu_block(trainer, agent, agent_idx, kappa_beta)

        # Handle different spatial dimensions
        if agent.mu_q.ndim == 1:
            # 0D: single K x K block
            M[mu_start:mu_end, mu_start:mu_end] = M_ii_mu
        else:
            # 1D or 2D: block diagonal over spatial points
            n_spatial = agent.mu_q.size // K
            for s in range(n_spatial):
                s_start = mu_start + s * K
                s_end = s_start + K
                M_s = M_ii_mu[s] if M_ii_mu.ndim == 3 else M_ii_mu
                if M_ii_mu.ndim == 4:
                    # 2D field: need to flatten spatial indices
                    shape = agent.mu_q.shape[:-1]
                    i, j = s // shape[1], s % shape[1]
                    M_s = M_ii_mu[i, j]
                M[s_start:s_end, s_start:s_end] = M_s

        # Sigma blocks (if requested)
        if include_sigma_blocks:
            sigma_start = dim_info['sigma_ranges'][agent_idx][0]
            sigma_end = dim_info['sigma_ranges'][agent_idx][1]

            # M^{Sigma Sigma} = (1/2)(Sigma^{-1} ⊗ Sigma^{-1})
            M_ii_sigma = _compute_sigma_block(agent)

            if agent.Sigma_q.ndim == 2:
                # 0D: single block
                M[sigma_start:sigma_end, sigma_start:sigma_end] = M_ii_sigma
            else:
                # Higher D: block diagonal
                K_sigma = K * (K + 1) // 2
                n_spatial = agent.mu_q.size // K
                for s in range(n_spatial):
                    s_start = sigma_start + s * K_sigma
                    s_end = s_start + K_sigma
                    M_s = M_ii_sigma[s] if M_ii_sigma.ndim == 3 else M_ii_sigma
                    M[s_start:s_end, s_start:s_end] = M_s

    # Off-diagonal blocks (future extension)
    if include_off_diagonal:
        # Currently not implemented - would require defining M_{ik} physics
        pass

    return M


def build_mu_mass_matrix(
    trainer,
    theta: np.ndarray,
    include_off_diagonal: bool = False
) -> np.ndarray:
    """
    Build just the mu-part of the mass matrix.

    This is more efficient than build_full_mass_matrix when only
    mu dynamics are needed (Sigma uses hyperbolic geodesic flow).

    Args:
        trainer: HamiltonianTrainer
        theta: Current parameters
        include_off_diagonal: Include M_{ik}^{mu mu} for i != k

    Returns:
        M_mu: Mass matrix for mu coordinates only
    """
    return build_full_mass_matrix(
        trainer, theta,
        include_sigma_blocks=False,
        include_off_diagonal=include_off_diagonal
    )


def compute_velocity_full_coupling(
    trainer,
    theta: np.ndarray,
    p: np.ndarray,
    include_inter_agent: bool = False,
    coupling_strength: float = 0.1
) -> np.ndarray:
    """
    Compute velocity with full mass matrix coupling.

    This version uses the complete multi-agent mass matrix
    and its inverse to compute velocities. When include_inter_agent=True,
    agent i's velocity depends on all agents' momenta.

    IMPORTANT: The theta/p ordering is interleaved per-agent:
        [mu_0, Sigma_0, mu_1, Sigma_1, ..., mu_N, Sigma_N]
    NOT [all_mu, all_Sigma]!

    Args:
        trainer: HamiltonianTrainer
        theta: Position [mu, Sigma] flattened (interleaved per agent)
        p: Momentum [pi_mu, Pi_Sigma] flattened (interleaved per agent)
        include_inter_agent: If True, use full M^{-1} including off-diagonal
        coupling_strength: λ for inter-agent coupling (only used if include_inter_agent=True)

    Returns:
        dtheta_dt: Velocity vector

    Notes:
        With include_inter_agent=False, this is equivalent to block-diagonal mode.
        With include_inter_agent=True, the full mass matrix is used:
            dμ/dt = M^{-1} @ π_μ  (globally coupled)
    """
    dtheta_dt = np.zeros_like(theta)

    if include_inter_agent:
        # Full coupling mode: build global mass matrix and invert
        # Step 1: Build full mass matrix with inter-agent coupling
        M_full = build_full_mass_matrix_with_coupling(
            trainer, theta, coupling_strength, symmetrize=True
        )

        # Step 2: Extract all mu momenta and compute global velocity
        dim_info = _compute_dimension_info(trainer)

        # Gather all mu momenta into a single vector
        p_mu_global = np.zeros(dim_info['total_mu_dim'])
        idx = 0
        for agent_idx, agent in enumerate(trainer.system.agents):
            K = agent.config.K
            n_spatial = agent.mu_q.size // K
            mu_size = n_spatial * K
            Sigma_size_per_point = K * (K + 1) // 2

            # Extract mu momentum for this agent from interleaved p
            p_mu_global[dim_info['mu_ranges'][agent_idx][0]:
                       dim_info['mu_ranges'][agent_idx][1]] = p[idx:idx + mu_size]
            idx += mu_size + n_spatial * Sigma_size_per_point

        # Step 3: Solve M @ v = p for v (more stable than explicit inverse)
        try:
            M_inv = np.linalg.inv(M_full + 1e-8 * np.eye(len(M_full)))
            dmu_global = M_inv @ p_mu_global / trainer.mass_scale
        except np.linalg.LinAlgError:
            # Fall back to pseudo-inverse if singular
            dmu_global = np.linalg.lstsq(M_full, p_mu_global, rcond=None)[0] / trainer.mass_scale

        # Step 4: Scatter mu velocities back to dtheta_dt
        idx = 0
        for agent_idx, agent in enumerate(trainer.system.agents):
            K = agent.config.K
            n_spatial = agent.mu_q.size // K
            mu_size = n_spatial * K
            Sigma_size_per_point = K * (K + 1) // 2

            # Place mu velocity
            mu_start, mu_end = dim_info['mu_ranges'][agent_idx]
            dtheta_dt[idx:idx + mu_size] = dmu_global[mu_start:mu_end]
            idx += mu_size

            # Sigma velocity (still uses hyperbolic flow, unchanged)
            for s in range(n_spatial):
                Pi_upper = p[idx:idx + Sigma_size_per_point]

                # Reconstruct symmetric matrix
                Pi_mat = np.zeros((K, K))
                upper_indices = np.triu_indices(K)
                Pi_mat[upper_indices] = Pi_upper
                Pi_mat = Pi_mat + Pi_mat.T - np.diag(np.diag(Pi_mat))

                # Get Sigma for this spatial point
                if agent.Sigma_q.ndim == 2:
                    Sigma = agent.Sigma_q
                elif agent.Sigma_q.ndim == 3:
                    Sigma = agent.Sigma_q[s]
                else:
                    shape = agent.Sigma_q.shape[:-2]
                    i, j = s // shape[1], s % shape[1]
                    Sigma = agent.Sigma_q[i, j]

                # Hyperbolic geodesic flow
                dSigma_dt = Sigma @ Pi_mat @ Sigma / trainer.mass_scale
                dtheta_dt[idx:idx + Sigma_size_per_point] = dSigma_dt[upper_indices]
                idx += Sigma_size_per_point

        return dtheta_dt

    # Block-diagonal mode: same as _compute_velocity_hyperbolic
    # Process each agent separately
    idx = 0

    for agent_idx, agent in enumerate(trainer.system.agents):
        K = agent.config.K
        n_spatial = agent.mu_q.size // K
        mu_size = n_spatial * K
        Sigma_size_per_point = K * (K + 1) // 2

        # --- mu part ---
        pi_mu = p[idx:idx + mu_size].reshape(agent.mu_q.shape)

        # Compute mass matrix for this agent
        kappa_beta = getattr(trainer.system.config, 'kappa_beta', 1.0)
        M = _compute_mu_block(trainer, agent, agent_idx, kappa_beta)

        # Compute velocity: dmu/dt = M^{-1} @ pi_mu
        dmu_dt = np.zeros_like(pi_mu)

        if agent.mu_q.ndim == 1:
            # 0D particle
            M_inv = np.linalg.inv(M + 1e-8 * np.eye(K))
            dmu_dt = M_inv @ pi_mu / trainer.mass_scale
        elif agent.mu_q.ndim == 2:
            # 1D field
            for i in range(agent.mu_q.shape[0]):
                M_inv = np.linalg.inv(M[i] + 1e-8 * np.eye(K))
                dmu_dt[i] = M_inv @ pi_mu[i] / trainer.mass_scale
        else:
            # 2D field
            for i in range(agent.mu_q.shape[0]):
                for j in range(agent.mu_q.shape[1]):
                    M_inv = np.linalg.inv(M[i, j] + 1e-8 * np.eye(K))
                    dmu_dt[i, j] = M_inv @ pi_mu[i, j] / trainer.mass_scale

        dtheta_dt[idx:idx + mu_size] = dmu_dt.flatten()
        idx += mu_size

        # --- Sigma part ---
        for s in range(n_spatial):
            Pi_upper = p[idx:idx + Sigma_size_per_point]

            # Reconstruct symmetric matrix
            Pi_mat = np.zeros((K, K))
            upper_indices = np.triu_indices(K)
            Pi_mat[upper_indices] = Pi_upper
            Pi_mat = Pi_mat + Pi_mat.T - np.diag(np.diag(Pi_mat))

            # Get Sigma for this spatial point
            if agent.Sigma_q.ndim == 2:
                Sigma = agent.Sigma_q
            elif agent.Sigma_q.ndim == 3:
                Sigma = agent.Sigma_q[s]
            else:
                shape = agent.Sigma_q.shape[:-2]
                i, j = s // shape[1], s % shape[1]
                Sigma = agent.Sigma_q[i, j]

            # Hyperbolic geodesic flow
            dSigma_dt = Sigma @ Pi_mat @ Sigma / trainer.mass_scale

            # Pack back
            dtheta_dt[idx:idx + Sigma_size_per_point] = dSigma_dt[upper_indices]
            idx += Sigma_size_per_point

    return dtheta_dt


def diagnose_mass_matrix(
    trainer,
    theta: np.ndarray
) -> MassMatrixDiagnostics:
    """
    Compute diagnostic information about the mass matrix.

    Useful for understanding the conditioning and structure of M.

    Args:
        trainer: HamiltonianTrainer
        theta: Current parameters

    Returns:
        MassMatrixDiagnostics with condition numbers, eigenvalues, etc.
    """
    dim_info = _compute_dimension_info(trainer)

    # Build full mu mass matrix
    M_mu = build_mu_mass_matrix(trainer, theta)

    # Compute eigenvalues
    eigenvalues = np.linalg.eigvalsh(M_mu)
    min_eig = eigenvalues.min()
    max_eig = eigenvalues.max()

    global_cond = max_eig / (min_eig + 1e-10)

    # Per-agent condition numbers
    per_agent_conds = {}
    per_agent_dims = {}

    for agent_idx, (start, end) in dim_info['mu_ranges'].items():
        M_ii = M_mu[start:end, start:end]
        eigs = np.linalg.eigvalsh(M_ii)
        per_agent_conds[agent_idx] = eigs.max() / (eigs.min() + 1e-10)
        per_agent_dims[agent_idx] = end - start

    # Diagonal dominance: ratio of diagonal to off-diagonal norm
    diag_norm = np.linalg.norm(np.diag(M_mu))
    off_diag = M_mu - np.diag(np.diag(M_mu))
    off_diag_norm = np.linalg.norm(off_diag)
    diag_dominance = diag_norm / (off_diag_norm + 1e-10)

    return MassMatrixDiagnostics(
        total_dim=len(M_mu),
        n_agents=trainer.system.n_agents,
        per_agent_dims=per_agent_dims,
        condition_numbers=per_agent_conds,
        global_condition_number=global_cond,
        diagonal_dominance=diag_dominance,
        min_eigenvalue=min_eig
    )


# =============================================================================
# Internal Helper Functions
# =============================================================================

def _compute_dimension_info(trainer) -> Dict:
    """Compute dimension information for all agents."""
    mu_ranges = {}
    sigma_ranges = {}
    total_mu_dim = 0
    total_sigma_dim = 0

    for agent_idx, agent in enumerate(trainer.system.agents):
        K = agent.config.K
        n_spatial = agent.mu_q.size // K

        # Mu dimensions
        mu_start = total_mu_dim
        mu_size = n_spatial * K
        mu_end = mu_start + mu_size
        mu_ranges[agent_idx] = (mu_start, mu_end)
        total_mu_dim = mu_end

        # Sigma dimensions
        sigma_start = total_sigma_dim
        sigma_size = n_spatial * K * (K + 1) // 2
        sigma_end = sigma_start + sigma_size
        sigma_ranges[agent_idx] = (sigma_start, sigma_end)
        total_sigma_dim = sigma_end

    return {
        'mu_ranges': mu_ranges,
        'sigma_ranges': sigma_ranges,
        'total_mu_dim': total_mu_dim,
        'total_sigma_dim': total_sigma_dim
    }


def _compute_mu_block(trainer, agent, agent_idx: int, kappa_beta: float) -> np.ndarray:
    """
    Compute M_{ii}^{mu mu} for a single agent.

    M_{ii}^{mu mu} = Sigma_p^{-1} + sum_j beta_ij Omega_ij Sigma_qj^{-1} Omega_ij^T

    This is the same computation as _compute_complete_mass_matrix in
    hamiltonian_trainer.py, extracted for modularity.
    """
    from gradients.softmax_grads import compute_softmax_weights

    K = agent.config.K

    # Initialize with bare mass: Sigma_p^{-1}
    M = np.zeros(agent.Sigma_p.shape, dtype=np.float64)

    if agent.Sigma_p.ndim == 2:
        M = np.linalg.inv(agent.Sigma_p + 1e-8 * np.eye(K))
    elif agent.Sigma_p.ndim == 3:
        for i in range(agent.Sigma_p.shape[0]):
            M[i] = np.linalg.inv(agent.Sigma_p[i] + 1e-8 * np.eye(K))
    else:
        for i in range(agent.Sigma_p.shape[0]):
            for j in range(agent.Sigma_p.shape[1]):
                M[i, j] = np.linalg.inv(agent.Sigma_p[i, j] + 1e-8 * np.eye(K))

    # Add relational mass from consensus coupling
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


def _compute_sigma_block(agent) -> np.ndarray:
    """
    Compute M_{ii}^{Sigma Sigma} for the SPD manifold metric.

    For the affine-invariant metric on SPD:
        ds^2 = tr(Sigma^{-1} dSigma Sigma^{-1} dSigma)

    The corresponding mass matrix in upper-triangle coordinates is:
        M^{Sigma Sigma} = (1/2)(Sigma^{-1} ⊗ Sigma^{-1})

    But since we use geodesic flow dSigma/dt = Sigma @ Pi @ Sigma,
    the "mass" is implicit and M^{Sigma Sigma} is not directly used.
    This function is provided for completeness.
    """
    K = agent.config.K
    Sigma_size = K * (K + 1) // 2

    if agent.Sigma_q.ndim == 2:
        # 0D: single matrix
        Sigma_inv = np.linalg.inv(agent.Sigma_q + 1e-8 * np.eye(K))
        # Kronecker product gives full metric
        M_full = 0.5 * np.kron(Sigma_inv, Sigma_inv)
        # Project to upper triangle
        M = _project_to_upper_triangle(M_full, K)
        return M
    else:
        # Higher D: per spatial point
        n_spatial = agent.mu_q.size // K
        M = np.zeros((n_spatial, Sigma_size, Sigma_size))
        for s in range(n_spatial):
            if agent.Sigma_q.ndim == 3:
                Sigma_inv = np.linalg.inv(agent.Sigma_q[s] + 1e-8 * np.eye(K))
            else:
                shape = agent.Sigma_q.shape[:-2]
                i, j = s // shape[1], s % shape[1]
                Sigma_inv = np.linalg.inv(agent.Sigma_q[i, j] + 1e-8 * np.eye(K))
            M_full = 0.5 * np.kron(Sigma_inv, Sigma_inv)
            M[s] = _project_to_upper_triangle(M_full, K)
        return M


def _project_to_upper_triangle(M_full: np.ndarray, K: int) -> np.ndarray:
    """Project full K^2 x K^2 metric to upper-triangle coordinates."""
    # Upper triangle indices
    upper_idx = np.triu_indices(K)
    n_upper = len(upper_idx[0])

    # Build projection matrix
    P = np.zeros((n_upper, K * K))
    for idx, (i, j) in enumerate(zip(*upper_idx)):
        flat_idx = i * K + j
        P[idx, flat_idx] = 1.0 if i == j else np.sqrt(2)  # Scale off-diagonal

    # Project: M_upper = P @ M_full @ P.T
    M_upper = P @ M_full @ P.T
    return M_upper


# =============================================================================
# Inter-Agent Coupling
# =============================================================================

def compute_inter_agent_coupling_block(
    trainer,
    agent_i_idx: int,
    agent_k_idx: int,
    coupling_strength: float = 0.1
) -> np.ndarray:
    """
    Compute M_{ik}^{mu mu} for inter-agent kinetic coupling.

    Physical Interpretation:
    ------------------------
    The inter-agent coupling creates momentum exchange between aligned agents.
    When agents i and k are in consensus (high β_ik), their momenta become
    coupled - a push on agent i partially accelerates agent k.

    Mathematical Form:
    ------------------
    M_ik^{μμ} = -λ * β_ik * Ω_ik * Σ_pk^{-1} * Ω_ik^T

    where:
    - λ is the coupling strength (small to maintain positive definiteness)
    - β_ik is the softmax coupling weight (high when agents agree)
    - Ω_ik is the transport operator from k to i's frame
    - Σ_pk^{-1} is agent k's prior precision (their "inertia")

    The negative sign ensures the full mass matrix M remains SPD when λ < 1.
    This creates attractive coupling: aligned agents share kinetic energy.

    Args:
        trainer: HamiltonianTrainer
        agent_i_idx: First agent index
        agent_k_idx: Second agent index
        coupling_strength: λ parameter (default 0.1, must be < 1 for SPD)

    Returns:
        M_ik: Inter-agent coupling block, shape (dim_i, dim_k)

    Notes:
        - M_ik ≠ M_ki in general (asymmetric coupling)
        - For symmetric M, use (M_ik + M_ki.T) / 2 in full matrix
        - Coupling vanishes when β_ik → 0 (no consensus)
    """
    from gradients.softmax_grads import compute_softmax_weights

    if agent_i_idx == agent_k_idx:
        raise ValueError("Use _compute_mu_block for diagonal blocks")

    agent_i = trainer.system.agents[agent_i_idx]
    agent_k = trainer.system.agents[agent_k_idx]

    K_i = agent_i.config.K
    K_k = agent_k.config.K

    if K_i != K_k:
        raise ValueError("Inter-agent coupling requires same latent dimension")

    K = K_i  # Same for both

    # Get softmax coupling weight β_ik
    kappa_beta = getattr(trainer.system.config, 'kappa_beta', 1.0)
    beta_fields = compute_softmax_weights(trainer.system, agent_i_idx, 'belief', kappa_beta)

    # Check if k is a neighbor of i
    if agent_k_idx not in beta_fields:
        # No coupling if not neighbors
        dim_i = agent_i.mu_q.size
        dim_k = agent_k.mu_q.size
        return np.zeros((dim_i, dim_k), dtype=np.float64)

    beta_ik = beta_fields[agent_k_idx]

    # Get transport operator
    Omega_ik = trainer.system.compute_transport_ij(agent_i_idx, agent_k_idx)

    # Compute coupling block based on spatial structure
    if agent_i.mu_q.ndim == 1:
        # 0D: single K x K block
        Sigma_pk_inv = np.linalg.inv(agent_k.Sigma_p + 1e-8 * np.eye(K))
        M_ik = -coupling_strength * float(beta_ik) * (Omega_ik @ Sigma_pk_inv @ Omega_ik.T)
        return M_ik

    elif agent_i.mu_q.ndim == 2:
        # 1D field: block diagonal coupling
        n_spatial = agent_i.mu_q.shape[0]
        dim_i = agent_i.mu_q.size
        dim_k = agent_k.mu_q.size

        M_ik = np.zeros((dim_i, dim_k), dtype=np.float64)

        for s in range(n_spatial):
            Sigma_pk_inv = np.linalg.inv(agent_k.Sigma_p[s] + 1e-8 * np.eye(K))
            Omega_s = Omega_ik[s] if Omega_ik.ndim == 3 else Omega_ik
            beta_s = beta_ik[s] if hasattr(beta_ik, '__len__') else float(beta_ik)

            block = -coupling_strength * beta_s * (Omega_s @ Sigma_pk_inv @ Omega_s.T)

            # Place in block-diagonal position
            i_start, i_end = s * K, (s + 1) * K
            k_start, k_end = s * K, (s + 1) * K
            M_ik[i_start:i_end, k_start:k_end] = block

        return M_ik

    else:
        # 2D field
        shape = agent_i.mu_q.shape[:-1]
        n_spatial = shape[0] * shape[1]
        dim_i = agent_i.mu_q.size
        dim_k = agent_k.mu_q.size

        M_ik = np.zeros((dim_i, dim_k), dtype=np.float64)

        for s in range(n_spatial):
            x, y = s // shape[1], s % shape[1]
            Sigma_pk_inv = np.linalg.inv(agent_k.Sigma_p[x, y] + 1e-8 * np.eye(K))
            Omega_s = Omega_ik[x, y] if Omega_ik.ndim == 4 else Omega_ik
            beta_s = beta_ik[x, y] if beta_ik.ndim >= 2 else float(beta_ik)

            block = -coupling_strength * beta_s * (Omega_s @ Sigma_pk_inv @ Omega_s.T)

            i_start, i_end = s * K, (s + 1) * K
            k_start, k_end = s * K, (s + 1) * K
            M_ik[i_start:i_end, k_start:k_end] = block

        return M_ik


def build_full_mass_matrix_with_coupling(
    trainer,
    theta: np.ndarray,
    coupling_strength: float = 0.1,
    symmetrize: bool = True
) -> np.ndarray:
    """
    Build full mass matrix including inter-agent kinetic coupling.

    This constructs M with both diagonal (M_ii) and off-diagonal (M_ik) blocks.

    Args:
        trainer: HamiltonianTrainer
        theta: Current parameters
        coupling_strength: λ for inter-agent coupling (0 = no coupling)
        symmetrize: If True, symmetrize off-diagonal blocks

    Returns:
        M: Full mass matrix with inter-agent coupling
    """
    from gradients.softmax_grads import compute_softmax_weights

    system = trainer.system
    n_agents = system.n_agents

    # Compute dimensions
    dim_info = _compute_dimension_info(trainer)
    total_mu_dim = dim_info['total_mu_dim']

    # Initialize with diagonal blocks
    M = build_mu_mass_matrix(trainer, theta, include_off_diagonal=False)

    # Add off-diagonal blocks
    if coupling_strength > 0:
        for i in range(n_agents):
            for k in range(n_agents):
                if i == k:
                    continue

                M_ik = compute_inter_agent_coupling_block(
                    trainer, i, k, coupling_strength
                )

                # Get index ranges
                i_start, i_end = dim_info['mu_ranges'][i]
                k_start, k_end = dim_info['mu_ranges'][k]

                # Place block
                M[i_start:i_end, k_start:k_end] = M_ik

        # Symmetrize if requested (for SPD guarantee)
        if symmetrize:
            M = (M + M.T) / 2

    return M