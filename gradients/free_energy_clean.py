#!/usr/bin/env python3
"""
Complete Free Energy Functional with Clean χ-Weighted Integration
==================================================================

Implements all energy terms with EXPLICIT χ weighting:

S = Σ_i ∫ χ_i α KL(q||p)                    [Self-coupling]
  + Σ_ij ∫ χ_ij β_ij KL(q_i||Ω[q_j])        [Belief alignment]
  + Σ_ij ∫ χ_ij γ_ij KL(p_i||Ω[p_j])        [Prior alignment]
  - Σ_i ∫ χ_i E_q[log p(o|x)]               [Observations]

CRITICAL PRINCIPLES:
-------------------
1. ALL spatial integrals use spatial_integrate() from geometry_clean
2. Softmax β_ij(c) computed pointwise, NO χ in denominator
3. χ weights applied during integration, not in field computation
4. Overlap weights χ_ij = χ_i · χ_j for alignment terms

Author: Chris - Phase 2
Date: November 2025
"""

import numpy as np
from typing import Dict, Optional
from dataclasses import dataclass

# Phase 1 geometry (clean χ-weighted integration)
from geometry.geometry_base import spatial_integrate

# Validated math utilities
from math_utils.numerical_utils import kl_gaussian
from math_utils.push_pull import compute_kl_transported, GaussianDistribution


# Phase 2 softmax weights
from gradients.softmax_grads import compute_softmax_weights




@dataclass
class FreeEnergyBreakdown:
    """Container for free energy components."""
    self_energy: float
    belief_align: float
    prior_align: float
    observations: float
    total: float
    
    def __repr__(self) -> str:
        return (
            f"FreeEnergy(\n"
            f"  self={self.self_energy:.4f},\n"
            f"  belief_align={self.belief_align:.4f},\n"
            f"  prior_align={self.prior_align:.4f},\n"
            f"  observations={self.observations:.4f},\n"
            f"  total={self.total:.4f}\n"
            f")"
        )


# =============================================================================
# Energy Term 1: Self-Coupling
# =============================================================================

def compute_self_energy(
    agent,
    lambda_self: float = 1.0
) -> float:
    """
    Self-coupling energy: α ∫_C χ_i(c) · KL(q_i(c) || p_i(c)) dc
    
    Encourages beliefs q to match priors p.
    
    Args:
        agent: Agent object with mu_q, Sigma_q, mu_p, Sigma_p, support
        lambda_self: Self-coupling strength (default: 1.0)
    
    Returns:
        energy: Non-negative scalar (KL ≥ 0)
    
    Properties:
        - Energy = 0 when q = p everywhere
        - Energy > 0 otherwise (KL divergence non-negative)
        - Only integrates over C_i (agent's support)
    """
    # FIXED CODE:
    kl_field = np.asarray(
        kl_gaussian(
            agent.mu_q, agent.Sigma_q,
            agent.mu_p, agent.Sigma_p,
            eps=1e-6
        ),
        dtype=np.float32
    )  # Shape: (*S,) or () for 0D
    
    # Get support weight
    chi_i = agent.support.chi_weight  # Shape: (*S,)
    
    # χ-weighted integration: ∫ χ_i · KL dc
    energy = lambda_self * spatial_integrate(kl_field, chi_i)
    
    return float(energy)


# =============================================================================
# Energy Term 2: Belief Alignment
# =============================================================================

def compute_belief_alignment_energy(
    system,
    agent_idx_i: int,
    lambda_belief: Optional[float] = None
) -> float:
    """
    Belief alignment energy for agent i with all neighbors j:
    
    E_i = Σ_j λ ∫_C χ_ij(c) · β_ij(c) · KL(q_i(c) || Ω_ij[q_j](c)) dc
    
    Encourages belief alignment with neighbors, weighted by softmax β_ij.
    
    Args:
        system: MultiAgentSystem instance
        agent_idx_i: Index of agent i
        lambda_belief: Coupling strength (uses system.config if None)
    
    Returns:
        energy: Non-negative scalar
    
    Algorithm:
        1. Get neighbors with spatial overlap
        2. Compute pointwise softmax weights β_ij(c) (NO χ in denominator)
        3. For each neighbor j:
           a. Get overlap weight χ_ij = χ_i · χ_j
           b. Compute transported KL field
           c. Apply β_ij weight pointwise
           d. Integrate with χ_ij weight: ∫ χ_ij · (β_ij · KL) dc
        4. Sum over all neighbors
    """
    agent_i = system.agents[agent_idx_i]
    
    # Get coupling strength
    if lambda_belief is None:
        lambda_belief = system.config.lambda_belief_align
    
    # Get neighbors (agents with spatial overlap)
    neighbors = system.get_neighbors(agent_idx_i)
    
    if len(neighbors) == 0:
        return 0.0
    
    # Compute softmax weights β_ij(c) for all neighbors
    # CRITICAL: NO χ in denominator!
    beta_fields = compute_softmax_weights(
        system,
        agent_idx_i,
        mode='belief',
        kappa=system.config.kappa_beta
    )
    
    total_energy = 0.0
    
    for j in neighbors:
        agent_j = system.agents[j]
        
        # 1. Get continuous overlap weight: χ_ij = χ_i · χ_j
        chi_ij = agent_i.support.compute_overlap_continuous(agent_j.support)
        
        # 2. Get softmax weight field β_ij(c)
        beta_ij = beta_fields[j]  # Shape: (*S,)
        
        # 3. Compute transport operator Ω_ij
        Omega_ij = system.compute_transport_ij(agent_idx_i, j)
     
        
        # 4. Compute transported KL field: KL(q_i || Ω[q_j])
        # Wrap distributions in GaussianDistribution objects
        q_i = GaussianDistribution(agent_i.mu_q, agent_i.Sigma_q)
        q_j = GaussianDistribution(agent_j.mu_q, agent_j.Sigma_q)
        
        kl_field = np.asarray(
            compute_kl_transported(q_i, q_j, Omega_ij),
            dtype=np.float32
        )  # Shape: (*S,) or () for 0D

        
        # 5. CRITICAL: Apply β_ij weight pointwise
        weighted_field = beta_ij * kl_field  # Shape: (*S,)
        
        # 6. Integrate with χ_ij weight: ∫ χ_ij · (β_ij · KL) dc
        energy_ij = spatial_integrate(weighted_field, chi_ij)
        
        total_energy += energy_ij
    
    return lambda_belief * total_energy


# =============================================================================
# Energy Term 3: Prior Alignment
# =============================================================================

def compute_prior_alignment_energy(
    system,
    agent_idx_i: int,
    lambda_prior: Optional[float] = None
) -> float:
    """
    Prior alignment energy for agent i with all neighbors j:
    
    E_i = Σ_j λ ∫_C χ_ij(c) · γ_ij(c) · KL(p_i(c) || Ω_ij[p_j](c)) dc
    
    Identical structure to belief alignment, but using priors p instead of beliefs q.
    
    Args:
        system: MultiAgentSystem instance
        agent_idx_i: Index of agent i
        lambda_prior: Coupling strength (uses system.config if None)
    
    Returns:
        energy: Non-negative scalar
    """
    agent_i = system.agents[agent_idx_i]
    
    # Get coupling strength
    if lambda_prior is None:
        lambda_prior = system.config.lambda_prior_align
    
    # Get neighbors
    neighbors = system.get_neighbors(agent_idx_i)
    
    if len(neighbors) == 0:
        return 0.0
    
    # Compute softmax weights γ_ij(c) for all neighbors
    gamma_fields = compute_softmax_weights(
        system,
        agent_idx_i,
        mode='prior',
        kappa=system.config.kappa_gamma
    )
    
    total_energy = 0.0
    
    for j in neighbors:
        agent_j = system.agents[j]
        
        # Overlap weight
        chi_ij = agent_i.support.compute_overlap_continuous(agent_j.support)
        
        # Softmax weight
        gamma_ij = gamma_fields[j]
        
        # Transport
        Omega_ij = system.compute_transport_ij(agent_idx_i, j)
       
      
        
        # Transported KL for priors
        # Wrap distributions in GaussianDistribution objects
        p_i = GaussianDistribution(agent_i.mu_p, agent_i.Sigma_p)
        p_j = GaussianDistribution(agent_j.mu_p, agent_j.Sigma_p)
        
        kl_field = np.asarray(
            compute_kl_transported(p_i, p_j, Omega_ij),
            dtype=np.float32
        )  # Shape: (*S,) or () for 0D
        
        # Apply γ_ij weight and integrate
        weighted_field = gamma_ij * kl_field
        energy_ij = spatial_integrate(weighted_field, chi_ij)
        
        total_energy += energy_ij
    
    return lambda_prior * total_energy


# =============================================================================
# Energy Term 4: Observations
# =============================================================================

def compute_observation_energy(
    system,
    agent,
    lambda_obs: float = 1.0
) -> float:
    """
    Observation likelihood energy: -λ ∫_C χ_i(c) · E_q[log p(o_i(c)|x)] dc
    
    Encourages beliefs to explain observations.
    Observations can be sparse over support.
    
    Args:
        agent: Agent with observations attribute
        lambda_obs: Observation coupling strength
    
    Returns:
        energy: Can be negative (maximizing likelihood → minimizing free energy)
    
    Notes:
        - At unobserved points: log-likelihood = 0 (no contribution)
        - Negative sign: minimizing F = maximizing likelihood
        - Only integrates over C_i (agent's support)
    """
    # Check if agent has observations
    if not hasattr(agent, 'observations') or not agent.observations:
        return 0.0
    
    chi_i = agent.support.chi_weight
    spatial_shape = chi_i.shape
    
    # Initialize log-likelihood field (0 at unobserved points)
    log_lik_field = np.zeros(spatial_shape, dtype=np.float32)
    
    # Compute expected log-likelihood at observed points
    for coord, o_obs in agent.observations.items():
        # Get belief at this point
        if agent.base_manifold.is_point:
            mu_q = agent.mu_q
            Sigma_q = agent.Sigma_q
        else:
            mu_q = agent.mu_q[coord]
            Sigma_q = agent.Sigma_q[coord]
        
        log_lik = expected_log_likelihood_gaussian(
            o_obs, mu_q, Sigma_q,
            agent.C_obs, agent.R_obs
        )
        
        # 🔥 FIX: Use scalar assignment for 0D arrays
        if agent.base_manifold.is_point:
            log_lik_field[()] = log_lik  # Scalar assignment to 0D array
        else:
            log_lik_field[coord] = log_lik
    
    # Integrate with negative sign: -∫ χ_i · E[log p(o|x)] dc
    energy = -lambda_obs * spatial_integrate(log_lik_field, chi_i)
    
    return float(energy)







def expected_log_likelihood_gaussian(
    o_obs: np.ndarray,
    mu_q: np.ndarray,
    Sigma_q: np.ndarray,
    C: np.ndarray,
    R: np.ndarray
) -> float:
    """
    Expected log-likelihood E_q[log p(o|x)] for Gaussian observation model.
    
    Observation model: p(o|x) = N(o | Cx, R)
    Belief: q(x) = N(x | μ_q, Σ_q)
    
    Result:
        E_q[log p(o|x)] = -½[log|2πR| + tr(R^{-1}[R_o + (o - Cμ)(o - Cμ)^T])]
    
    where R_o = CΣ_qC^T (predicted observation covariance).
    
    Args:
        o_obs: Observation vector, shape (D,)
        mu_q: Belief mean, shape (K,)
        Sigma_q: Belief covariance, shape (K, K)
        C: Observation matrix, shape (D, K)
        R: Observation noise covariance, shape (D, D)
    
    Returns:
        log_lik: Expected log-likelihood (scalar)
    """
    D = o_obs.shape[0]
    
    # Predicted observation mean: C μ_q
    o_pred = C @ mu_q
    
    # Predicted observation covariance: C Σ_q C^T
    R_o = C @ Sigma_q @ C.T
    
    # Innovation: o - C μ_q
    innovation = o_obs - o_pred
    
    # Compute R^{-1}
    R_inv = np.linalg.inv(R + 1e-8 * np.eye(D))
    
    # Mahalanobis distance: (o - Cμ)^T R^{-1} (o - Cμ)
    mahal_innovation = innovation @ R_inv @ innovation
    
    # Trace term: tr(R^{-1} R_o)
    trace_term = np.trace(R_inv @ R_o)
    
    # Log determinant: log|2πR|
    sign, logdet_R = np.linalg.slogdet(R)
    log_det_term = D * np.log(2 * np.pi) + logdet_R
    
    # Expected log-likelihood
    log_lik = -0.5 * (log_det_term + trace_term + mahal_innovation)
    
    return float(log_lik)


# =============================================================================
# Total System Energy
# =============================================================================




   
def compute_total_free_energy(system) -> FreeEnergyBreakdown:

    """
    Compute complete free energy functional for multi-agent system.

    Returns all energy components separately for analysis.

    Args:
        system: MultiAgentSystem or MultiScaleSystem instance

    Returns:
        breakdown: FreeEnergyBreakdown with all components

    Usage:
        >>> energies = compute_total_free_energy(system)
        >>> print(f"Total energy: {energies.total:.4f}")
        >>> print(f"Self-coupling: {energies.self_energy:.4f}")
    """


    E_self = 0.0
    E_belief = 0.0
    E_prior = 0.0
    E_obs = 0.0
    E_obs_meta = 0.0  # Meta-agent observation of constituents (bottom-up coupling)

    # Get config (handle both system types)
    if hasattr(system, 'system_config'):
        config = system.system_config
    else:
        config = system.config

    # Get agent list (handle MultiScaleSystem vs MultiAgentSystem)
    if hasattr(system, 'get_all_active_agents'):
        # MultiScaleSystem
        agents = system.get_all_active_agents()
    else:
        # MultiAgentSystem
        agents = system.agents

    # (1) Self-coupling
    for agent in agents:
        E_self += compute_self_energy(agent, lambda_self=config.lambda_self)

    # (2) Belief alignment
    if config.has_belief_alignment:
        # For MultiScaleSystem, use agents list; for MultiAgentSystem, use range
        if hasattr(system, 'get_all_active_agents'):
            # MultiScaleSystem - iterate agents directly
            for i, agent in enumerate(agents):
                E_belief += compute_belief_alignment_energy(system, i)
        else:
            # MultiAgentSystem - use agent indices
            for i in range(system.n_agents):
                E_belief += compute_belief_alignment_energy(system, i)

    # (3) Prior alignment
    if config.has_prior_alignment:
        if hasattr(system, 'get_all_active_agents'):
            for i, agent in enumerate(agents):
                E_prior += compute_prior_alignment_energy(system, i)
        else:
            for i in range(system.n_agents):
                E_prior += compute_prior_alignment_energy(system, i)

    # (4) Observations (regular agents)
    if config.has_observations:
        for agent in agents:
            E_obs += compute_observation_energy(
                system,
                agent,
                lambda_obs=config.lambda_obs,
            )

    # (5) Meta-agent constituent observations (BOTTOM-UP COUPLING!)
    # This completes the renormalization group flow
    if hasattr(system, 'compute_observation_likelihood_meta'):
        lambda_obs_meta = getattr(config, 'lambda_obs_meta', 0.0)
        if lambda_obs_meta > 0:
            for agent in agents:
                # Only meta-agents observe their constituents
                if hasattr(agent, 'is_meta') and agent.is_meta and len(agent.constituents) > 0:
                    E_obs_meta += lambda_obs_meta * system.compute_observation_likelihood_meta(agent)

    E_total = E_self + E_belief + E_prior + E_obs + E_obs_meta

    return FreeEnergyBreakdown(
        self_energy=E_self,
        belief_align=E_belief,
        prior_align=E_prior,
        observations=E_obs + E_obs_meta,  # Combine regular + meta observations
        total=E_total,
    )



# =============================================================================
# Convenience Functions
# =============================================================================

def compute_agent_energy_contribution(system, agent_idx: int) -> Dict[str, float]:
    agent = system.agents[agent_idx]

    energies = {
        "self": compute_self_energy(agent, lambda_self=system.config.lambda_self),
        "belief_align": compute_belief_alignment_energy(system, agent_idx),
        "prior_align": compute_prior_alignment_energy(system, agent_idx),
        "observations": compute_observation_energy(
            system,
            agent,
            lambda_obs=system.config.lambda_obs,
        ),
    }

    energies["total"] = sum(energies.values())
    return energies
