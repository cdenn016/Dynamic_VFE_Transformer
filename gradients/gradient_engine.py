#!/usr/bin/env python3
"""
Clean Gradient Engine with χ-Weighted Accumulation
===================================================

Computes gradients ∂S/∂θ for all agent parameters with proper χ weighting.

CRITICAL PRINCIPLE:
------------------
Gradients must be χ-weighted DURING accumulation, not after!

Gradient structure:
    ∂S/∂θ_i = ∂E_self/∂θ_i + ∂E_belief/∂θ_i + ∂E_prior/∂θ_i + ∂E_obs/∂θ_i

Where θ_i represents:
    - μ_q, Σ_q: Belief parameters
    - μ_p, Σ_p: Prior parameters
    - φ: Gauge field (SO(3))

Author: Chris - Phase 3
Date: November 2025
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from joblib import Parallel, delayed, parallel_backend
from gradients.gradient_terms import (grad_kl_source, grad_kl_target, cholesky_gradient,
                                      grad_kl_wrt_transport, grad_self_wrt_q, grad_self_wrt_p )

from math_utils.fisher_metric import natural_gradient_gaussian

from math_utils.push_pull import GaussianDistribution
from math_utils.transport import compute_transport
                             
from gradients.softmax_grads import (
       compute_softmax_coupling_gradients_belief as _softmax_belief_clean,
       compute_softmax_coupling_gradients_prior as _softmax_prior_clean,
       compute_softmax_weights
   )
from math_utils.push_pull import push_gaussian
# =============================================================================
# Gradient Container
# =============================================================================

@dataclass
class AgentGradients:
    """
    Container for one agent's gradients.
    
    All gradients are spatial fields with shape matching agent's base manifold.
    
    Euclidean gradients (accumulated):
        grad_mu_q: ∂S/∂μ_q, shape (*S, K)
        grad_Sigma_q: ∂S/∂Σ_q, shape (*S, K, K)
        grad_mu_p: ∂S/∂μ_p, shape (*S, K)
        grad_Sigma_p: ∂S/∂Σ_p, shape (*S, K, K)
        grad_phi: ∂S/∂φ, shape (*S, 3)
    
    Natural gradients (computed after accumulation):
        delta_mu_q, delta_Sigma_q: Natural gradients for beliefs
        delta_mu_p, delta_Sigma_p: Natural gradients for priors
        delta_phi: Natural gradient for gauge field
    """
    # Euclidean gradients (accumulated)
    grad_mu_q: np.ndarray
    grad_Sigma_q: np.ndarray
    grad_mu_p: np.ndarray
    grad_Sigma_p: np.ndarray
    grad_phi: np.ndarray
    
    #natural gradients
    delta_L_q: np.ndarray = None     
    delta_L_p: np.ndarray = None     
    delta_mu_q: np.ndarray = None
    delta_Sigma_q: np.ndarray = None
    delta_mu_p: np.ndarray = None
    delta_Sigma_p: np.ndarray = None
    delta_phi: np.ndarray = None


# =============================================================================
# Gradient Term 4: Observation Likelihood ∂E_obs/∂θ
# =============================================================================

def compute_observation_gradients(system, agent) -> AgentGradients:
    """
    Compute gradients of observation energy: ∂/∂θ [-∫ χ_i E_q[log p(o|x)] dc]
    
    For Gaussian observation model p(o|x) = N(o | Cx, R):
        ∂E_q[log p]/∂μ_q = C^T R^{-1} (o - Cμ_q)
        ∂E_q[log p]/∂Σ_q = -½ C^T R^{-1} C
    
    Energy has negative sign, so gradients are negated.
    Only computed at observed points; zero elsewhere.
    
    Args:
        agent: Agent with observations attribute
    
    Returns:
        gradients: AgentGradients with observation contributions
    """

    spatial_shape = agent.support.base_manifold.shape
    K = agent.K

    grad_mu_q = np.zeros((*spatial_shape, K), dtype=np.float32)
    grad_Sigma_q = np.zeros((*spatial_shape, K, K), dtype=np.float32)

    # No observations → zero gradients
    if not hasattr(agent, "observations") or not agent.observations:
        return AgentGradients(
            grad_mu_q=grad_mu_q,
            grad_Sigma_q=grad_Sigma_q,
            grad_mu_p=np.zeros_like(grad_mu_q),
            grad_Sigma_p=np.zeros_like(grad_Sigma_q),
            grad_phi=np.zeros((*spatial_shape, 3), dtype=np.float32),
        )

    # Observation model parameters
    C = agent.C_obs
    R = agent.R_obs
    R_inv = np.linalg.inv(R + 1e-8 * np.eye(R.shape[0], dtype=R.dtype))

    # Energy prefactor λ_obs
    lambda_obs = getattr(system.config, "lambda_obs", 1.0)

    # ∂E_q[log p]/∂Σ_q = -½ C^T R^{-1} C
    grad_Sigma_obs_const = -0.5 * C.T @ R_inv @ C

    for coord, o_obs in agent.observations.items():
        if agent.base_manifold.is_point:
            mu_q = agent.mu_q
            chi = agent.support.chi_weight
        else:
            mu_q = agent.mu_q[coord]
            chi = agent.support.chi_weight[coord]

        o_pred = C @ mu_q
        innovation = o_obs - o_pred

        # ∂E_q[log p]/∂μ_q = C^T R^{-1} (o - Cμ_q)
        grad_mu_obs = C.T @ R_inv @ innovation

        # Energy is -λ_obs ∫ χ E[log p]; apply -λ_obs * χ
        factor = -lambda_obs * chi

        if agent.base_manifold.is_point:
            grad_mu_q = factor * grad_mu_obs
            grad_Sigma_q = factor * grad_Sigma_obs_const
        else:
            grad_mu_q[coord] = factor * grad_mu_obs
            grad_Sigma_q[coord] = factor * grad_Sigma_obs_const

    return AgentGradients(
        grad_mu_q=grad_mu_q,
        grad_Sigma_q=grad_Sigma_q,
        grad_mu_p=np.zeros((*spatial_shape, K), dtype=np.float32),
        grad_Sigma_p=np.zeros((*spatial_shape, K, K), dtype=np.float32),
        grad_phi=np.zeros((*spatial_shape, 3), dtype=np.float32),
    )


def compute_meta_observation_gradients(system, agent) -> AgentGradients:
    """
    Compute gradients for meta-agent observing constituents (BOTTOM-UP COUPLING).

    For meta-agents with constituents, this implements the observation
    likelihood where the meta-agent "observes" the renormalized state
    of its constituents:

        o_meta = weighted_average({q_constituent_i})
        E_obs_meta = λ * ||o_meta - μ_q_meta||² / (2R²)

    This completes the bidirectional renormalization group flow:
        - Top-down: Constituents get priors from meta-agent (update_prior_from_parent)
        - Bottom-up: Meta-agent observes constituents (this gradient)

    Like phonons tracking atom displacements in solid-state physics!

    Args:
        system: MultiScaleSystem with compute_observation_likelihood_meta method
        agent: Meta-agent (must have agent.is_meta=True and len(agent.constituents)>0)

    Returns:
        gradients: AgentGradients pulling meta-agent toward constituent consensus

    Physics:
        In a crystal, phonons (collective modes) track atom motion.
        Here, meta-agents (collective beliefs) track constituent beliefs.
    """
    spatial_shape = agent.support.base_manifold.shape
    K = agent.K

    # Zero gradients for non-meta-agents or meta-agents without constituents
    if not (hasattr(agent, 'is_meta') and agent.is_meta and len(agent.constituents) > 0):
        return AgentGradients(
            grad_mu_q=np.zeros((*spatial_shape, K), dtype=np.float32),
            grad_Sigma_q=np.zeros((*spatial_shape, K, K), dtype=np.float32),
            grad_mu_p=np.zeros((*spatial_shape, K), dtype=np.float32),
            grad_Sigma_p=np.zeros((*spatial_shape, K, K), dtype=np.float32),
            grad_phi=np.zeros((*spatial_shape, 3), dtype=np.float32),
        )

    # Generate synthetic observation from constituents
    o_meta = agent.generate_observations_from_constituents()

    # Get config
    if hasattr(system, 'system_config'):
        config = system.system_config
    else:
        config = system.config

    lambda_obs_meta = getattr(config, 'lambda_obs_meta', 1.0)

    if lambda_obs_meta == 0:
        # Coupling disabled
        return AgentGradients(
            grad_mu_q=np.zeros((*spatial_shape, K), dtype=np.float32),
            grad_Sigma_q=np.zeros((*spatial_shape, K, K), dtype=np.float32),
            grad_mu_p=np.zeros((*spatial_shape, K), dtype=np.float32),
            grad_Sigma_p=np.zeros((*spatial_shape, K, K), dtype=np.float32),
            grad_phi=np.zeros((*spatial_shape, 3), dtype=np.float32),
        )

    # Observation noise scale (from emergence.py:1143)
    # Using same R_scale as in compute_observation_likelihood_meta
    R_scale = getattr(config, 'obs_R_scale', 1.0)

    # Gradient: ∂E/∂μ_q = λ * (μ_q - o_meta) / R²
    # (Pulls meta-agent belief toward constituent consensus)
    residual = agent.mu_q - o_meta
    grad_mu_q = lambda_obs_meta * residual / (R_scale ** 2)

    # No covariance gradient for this simple observation model
    # (Could add if we want meta-agent uncertainty to track constituent spread)
    grad_Sigma_q = np.zeros((*spatial_shape, K, K), dtype=np.float32)

    return AgentGradients(
        grad_mu_q=grad_mu_q,
        grad_Sigma_q=grad_Sigma_q,
        grad_mu_p=np.zeros((*spatial_shape, K), dtype=np.float32),
        grad_Sigma_p=np.zeros((*spatial_shape, K, K), dtype=np.float32),
        grad_phi=np.zeros((*spatial_shape, 3), dtype=np.float32),
    )


# =============================================================================
# Gradient Term 5: Softmax Coupling (∂β/∂θ)·KL and (∂γ/∂θ)·KL
# =============================================================================

def compute_softmax_coupling_gradients_belief(
    system,
    agent_idx_i: int
) -> Dict[int, AgentGradients]:
    """
    Compute (∂β_ij/∂θ)·KL terms for belief alignment.
    
    Uses streamlined softmax_clean module - all logic is internal.
    """
    
    
    # Call streamlined version (handles everything internally)
    softmax_grads = _softmax_belief_clean(system, agent_idx_i, eps=1e-8)
    
    # Convert SoftmaxGradients -> AgentGradients for compatibility
    result = {}
    for agent_idx, grads in softmax_grads.items():
        agent = system.agents[agent_idx]
        result[agent_idx] = AgentGradients(
            grad_mu_q=grads.grad_mu_q,
            grad_Sigma_q=grads.grad_Sigma_q,
            grad_mu_p=grads.grad_mu_p,
            grad_Sigma_p=grads.grad_Sigma_p,
            grad_phi=grads.grad_phi if grads.grad_phi is not None else np.zeros_like(agent.gauge.phi)
        )
    
    return result


def compute_softmax_coupling_gradients_prior(
       system,
       agent_idx_i: int
   ) -> Dict[int, AgentGradients]:
       """Streamlined softmax coupling for priors."""
       
       # Call clean implementation
       softmax_grads = _softmax_prior_clean(system, agent_idx_i, eps=1e-8)
       
       # Convert SoftmaxGradients -> AgentGradients
       result = {}
       for agent_idx, grads in softmax_grads.items():
           agent = system.agents[agent_idx]
           result[agent_idx] = AgentGradients(
               grad_mu_q=grads.grad_mu_q,
               grad_Sigma_q=grads.grad_Sigma_q,
               grad_mu_p=grads.grad_mu_p,
               grad_Sigma_p=grads.grad_Sigma_p,
               grad_phi=grads.grad_phi if grads.grad_phi is not None 
                        else np.zeros_like(agent.gauge.phi)
           )
       return result





def contract_gauge_gradient(
    grad_mu_factor: np.ndarray,
    grad_Sigma_factor: np.ndarray,
    dOmega_dPhi: Tuple[np.ndarray, ...],
    *,
    weight: float = 1.0,
) -> np.ndarray:
    """
    Contract KL gradients with transport differential to get ∂L/∂φ.
    
    Computes: ∑_a weight · tr(∂KL/∂Ω : ∂Ω/∂φ^a)
    
    Args:
        grad_mu_factor: From grad_kl_wrt_transport, shape (..., K, K)
        grad_Sigma_factor: From grad_kl_wrt_transport, shape (..., K, K)
        dOmega_dPhi: Tuple of 3 arrays (dΩ/dφ^x, dΩ/dφ^y, dΩ/dφ^z), each (..., K, K)
        weight: Scalar weight (e.g., β_ij or γ_ij at a point, or spatially integrated)
    
    Returns:
        grad_phi: Gradient w.r.t. φ, shape (..., 3)
    """
    grad_mu_factor = np.asarray(grad_mu_factor, dtype=np.float64)
    grad_Sigma_factor = np.asarray(grad_Sigma_factor, dtype=np.float64)
    
    grad_phi_components = []
    
    for a in range(3):
        dOmega_a = np.asarray(dOmega_dPhi[a], dtype=np.float64)  # (..., K, K)
        
        # Contraction: tr(grad_factor · dΩ/dφ^a)
        # Mean contribution
        contrib_mu = np.einsum('...ij,...ij->...', grad_mu_factor, dOmega_a, optimize=True)
        
        # Sigma contribution
        contrib_Sigma = np.einsum('...ij,...ij->...', grad_Sigma_factor, dOmega_a, optimize=True)
        
        # Total for this component
        grad_phi_a = weight * (contrib_mu + contrib_Sigma)
        grad_phi_components.append(grad_phi_a)
    
    # Stack into (..., 3)
    grad_phi = np.stack(grad_phi_components, axis=-1)
    
    return grad_phi.astype(np.float32, copy=False)


# =============================================================================
# Full Gauge Gradient Computation
# =============================================================================

def compute_gauge_gradient_alignment(
    agent_i_mu: np.ndarray,
    agent_i_Sigma: np.ndarray,
    agent_j_mu: np.ndarray,
    agent_j_Sigma: np.ndarray,
    phi_i: np.ndarray,
    phi_j: np.ndarray,
    generators: np.ndarray,
    beta_ij: np.ndarray,  # Spatial field of weights
    *,
    direction: str = 'i',
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Compute analytical gauge gradient ∂/∂φ for alignment energy.
    
    For agent i:
        ∂/∂φ_i [β_ij(c) KL(q_i(c) || Ω_ij[q_j](c))]
    
    For agent j:
        ∂/∂φ_j [β_ij(c) KL(q_i(c) || Ω_ij[q_j](c))]
    
    This replaces finite differences with analytical chain rule.
    
    Args:
        agent_i_mu, agent_i_Sigma: Source agent parameters, shape (*S, K), (*S, K, K)
        agent_j_mu, agent_j_Sigma: Target agent parameters
        phi_i, phi_j: Gauge fields, shape (*S, 3)
        generators: SO(3) generators, shape (3, K, K)
        beta_ij: Spatial weight field, shape (*S,) - can be position-dependent
        direction: 'i' for ∂/∂φ_i or 'j' for ∂/∂φ_j
        eps: Numerical stability
    
    Returns:
        grad_phi: Gradient field, shape (*S, 3)
    """
    from math_utils.transport import compute_transport, compute_transport_differential
    
    # Compute transport operator
    Omega_ij = compute_transport(phi_i, phi_j, generators, validate=False)
    
    # Compute differential ∂Ω/∂φ
    dOmega_dPhi = compute_transport_differential(
        phi_i, phi_j, generators,
        direction=direction,
        exp_phi_i=None,
        exp_phi_j=None,
    )
    
    # Compute KL gradient w.r.t. Ω
    grad_mu_factor, grad_Sigma_factor = grad_kl_wrt_transport(
        agent_i_mu, agent_i_Sigma,
        agent_j_mu, agent_j_Sigma,
        Omega_ij,
        eps=eps,
    )
    
    # Initialize gradient
    spatial_shape = phi_i.shape[:-1]
    grad_phi = np.zeros((*spatial_shape, 3), dtype=np.float32)
    
    # Contract at each spatial point with weight β_ij(c)
    # Note: beta_ij can vary spatially!
    for idx in np.ndindex(spatial_shape):
        weight = float(beta_ij[idx])
        
        if abs(weight) < 1e-12:
            continue  # Skip if weight negligible
        
        # Extract local values
        gmu = grad_mu_factor[idx]  # (K, K)
        gSig = grad_Sigma_factor[idx]  # (K, K)
        dOm = tuple(dOmega_dPhi[a][idx] for a in range(3))  # 3x (K, K)
        
        # Contract
        grad_phi[idx] = contract_gauge_gradient(
            gmu, gSig, dOm, weight=weight
        )
    
    return grad_phi


# =============================================================================
# Gradient Term 1: Self-Coupling ∂E_self/∂θ
# =============================================================================

def compute_self_coupling_gradients(
    agent,
    alpha: float = 1.0
) -> AgentGradients:
    """
    Compute gradients of self-coupling energy: ∂/∂θ [α ∫ χ_i KL(q||p) dc]
    
    CRITICAL: Gradients already χ-weighted via spatial_integrate in energy.
    Here we compute local gradients, which will be χ-weighted during accumulation.
    
    Args:
        agent: Agent object
        alpha: Self-coupling strength
    
    Returns:
        gradients: AgentGradients with self-coupling contributions
    
    Mathematical form:
        ∂E_self/∂μ_q = α χ_i · ∂KL/∂μ_q
        ∂E_self/∂Σ_q = α χ_i · ∂KL/∂Σ_q
        ∂E_self/∂μ_p = α χ_i · ∂KL/∂μ_p (negative contribution)
        ∂E_self/∂Σ_p = α χ_i · ∂KL/∂Σ_p (negative contribution)
    """
    
    
    # Compute local KL gradients (Euclidean)
    # These are ∂KL(q||p)/∂q and ∂KL(q||p)/∂p at each point
    g_mu_q, g_Sigma_q = grad_self_wrt_q(
        agent.mu_q, agent.Sigma_q,
        agent.mu_p, agent.Sigma_p
    )
    
    g_mu_p, g_Sigma_p = grad_self_wrt_p(
        agent.mu_q, agent.Sigma_q,
        agent.mu_p, agent.Sigma_p
    )
    
    # Apply α and χ_i weighting
    chi_i = agent.support.chi_weight  # Shape: (*S,)
    
    # FIXED CODE:
    if agent.base_manifold.is_point:
        # 0D: chi is scalar, just multiply directly
        chi_scalar = float(chi_i)
        grad_mu_q = alpha * chi_scalar * g_mu_q
        grad_Sigma_q = alpha * chi_scalar * g_Sigma_q
        grad_mu_p = alpha * chi_scalar * g_mu_p
        grad_Sigma_p = alpha * chi_scalar * g_Sigma_p
    else:
        # ND: chi has spatial dimensions, broadcast properly
        chi_broadcast_vec = chi_i[..., np.newaxis]  # (*S, 1)
        chi_broadcast_mat = chi_i[..., np.newaxis, np.newaxis]  # (*S, 1, 1)
        grad_mu_q = alpha * chi_broadcast_vec * g_mu_q
        grad_Sigma_q = alpha * chi_broadcast_mat * g_Sigma_q
        grad_mu_p = alpha * chi_broadcast_vec * g_mu_p
        grad_Sigma_p = alpha * chi_broadcast_mat * g_Sigma_p
        
        
    
    # No gauge gradient from self-coupling
    grad_phi = np.zeros_like(agent.gauge.phi)
    
    return AgentGradients(
        grad_mu_q=grad_mu_q,
        grad_Sigma_q=grad_Sigma_q,
        grad_mu_p=grad_mu_p,
        grad_Sigma_p=grad_Sigma_p,
        grad_phi=grad_phi
    )


# =============================================================================
# Gradient Term 2: Belief Alignment ∂E_belief/∂θ
# =============================================================================

def compute_belief_alignment_gradients(
    system,
    agent_idx_i: int,
    lambda_belief: float = 1.0,
    lambda_phi: float = 1.0,
) -> Dict[int, AgentGradients]:
    """
    Compute gradients of belief alignment for agent i with all neighbors.
    
    Returns gradients for BOTH agent i and all neighbors j.
    
    
    Energy term:
        E = Σ_j λ ∫ χ_ij β_ij KL(q_i || Ω[q_j]) dc
    
    Gradients:
        ∂E/∂θ_i: Includes ∂β/∂θ_i terms (softmax coupling)
        ∂E/∂θ_j: Backprop through transport Ω[q_j]
        ∂E/∂φ_i, ∂E/∂φ_j: Chain rule through Ω = exp(φ_i)exp(-φ_j)
    
    Args:
        system: MultiAgentSystem
        agent_idx_i: Agent index
        lambda_belief: Coupling strength
    
    Returns:
        gradients: Dict mapping agent idx → AgentGradients
                   Contains contributions for agent i and all neighbors j
    """
    agent_i = system.agents[agent_idx_i]
    neighbors = system.get_neighbors(agent_idx_i)
    
    if len(neighbors) == 0:
        # Return zero gradients for agent i only
        spatial_shape = agent_i.support.base_manifold.shape
        K = agent_i.K
        return {
            agent_idx_i: AgentGradients(
                grad_mu_q=np.zeros((*spatial_shape, K), dtype=np.float32),
                grad_Sigma_q=np.zeros((*spatial_shape, K, K), dtype=np.float32),
                grad_mu_p=np.zeros((*spatial_shape, K), dtype=np.float32),
                grad_Sigma_p=np.zeros((*spatial_shape, K, K), dtype=np.float32),
                grad_phi=np.zeros((*spatial_shape, 3), dtype=np.float32)
            )
        }
    
    # Initialize gradient containers for all affected agents
    gradients = {}
    for idx in [agent_idx_i] + neighbors:
        agent = system.agents[idx]
        spatial_shape = agent.support.base_manifold.shape
        K = agent.K
        gradients[idx] = AgentGradients(
            grad_mu_q=np.zeros((*spatial_shape, K), dtype=np.float32),
            grad_Sigma_q=np.zeros((*spatial_shape, K, K), dtype=np.float32),
            grad_mu_p=np.zeros((*spatial_shape, K), dtype=np.float32),
            grad_Sigma_p=np.zeros((*spatial_shape, K, K), dtype=np.float32),
            grad_phi=np.zeros((*spatial_shape, 3), dtype=np.float32)
        )
    
    # Use cached attention weights if available (from transformer attention layer)
    # This ensures coherent information flow: attention selects context, FFN refines beliefs
    if hasattr(system, '_beta_cache') and system._beta_cache is not None:
        # Use precomputed β from attention layer (transformer mode)
        # _beta_cache is (N, N) array with β_ij weights for all pairs
        beta_fields = {}
        for j in neighbors:
            beta_fields[j] = system._beta_cache[agent_idx_i, j]
    else:
        # Compute softmax weights β_ij(c) from scratch (standalone mode)
        beta_fields = compute_softmax_weights(
            system, agent_idx_i, mode='belief', kappa=system.config.kappa_beta
        )
    
    # Process each neighbor
    for j in neighbors:
        agent_j = system.agents[j]
        
        # Get overlap weight
        chi_ij = agent_i.support.compute_overlap_continuous(agent_j.support)
        
        # Get softmax weight
        beta_ij = beta_fields[j]
        
        # Transport operator
                
        Omega_ij = compute_transport(
            agent_i.gauge.phi, agent_j.gauge.phi, agent_i.generators
        )
        
        # Transported distributions
        q_i = GaussianDistribution(agent_i.mu_q, agent_i.Sigma_q)
        q_j = GaussianDistribution(agent_j.mu_q, agent_j.Sigma_q)
        
        q_j_transported = push_gaussian(q_j, Omega_ij)

        g_mu_i, g_Sigma_i = grad_kl_source(
            agent_i.mu_q, agent_i.Sigma_q,
            q_j_transported.mu, q_j_transported.Sigma
        )
        
        # ∂KL(q_i || Ω[q_j])/∂q_j (target gradients via backprop)
        g_mu_j, g_Sigma_j = grad_kl_target(
            agent_i.mu_q, agent_i.Sigma_q,
            q_j_transported.mu, q_j_transported.Sigma,
            Omega_ij
        )
        
        # CRITICAL: Apply BOTH χ_ij and β_ij weights
       
        weight_field = lambda_belief * chi_ij * beta_ij
        
        if agent_i.base_manifold.is_point:
            # 0D: weight is scalar, multiply directly
            weight_scalar = float(weight_field)
            
            gradients[agent_idx_i].grad_mu_q += weight_scalar * g_mu_i
            gradients[agent_idx_i].grad_Sigma_q += weight_scalar * g_Sigma_i
            gradients[j].grad_mu_q += weight_scalar * g_mu_j
            gradients[j].grad_Sigma_q += weight_scalar * g_Sigma_j
        else:
            # ND: weight has spatial dimensions, broadcast properly
            weight_vec = weight_field[..., np.newaxis]  # (*S, 1)
            weight_mat = weight_field[..., np.newaxis, np.newaxis]  # (*S, 1, 1)
            
            gradients[agent_idx_i].grad_mu_q += weight_vec * g_mu_i
            gradients[agent_idx_i].grad_Sigma_q += weight_mat * g_Sigma_i
            gradients[j].grad_mu_q += weight_vec * g_mu_j
            gradients[j].grad_Sigma_q += weight_mat * g_Sigma_j
        
        
        if lambda_phi != 0.0:
            # Compute ∂E/∂φ_i and ∂E/∂φ_j via chain rule through Ω
            grad_phi_i = compute_gauge_gradient_alignment(
                agent_i.mu_q, agent_i.Sigma_q,
                agent_j.mu_q, agent_j.Sigma_q,
                agent_i.gauge.phi,
                agent_j.gauge.phi,
                agent_i.generators,
                beta_ij,
                direction='i',  # Gradient w.r.t. φ_i
                eps=1e-8,
            )
            
            grad_phi_j = compute_gauge_gradient_alignment(
                agent_i.mu_q, agent_i.Sigma_q,
                agent_j.mu_q, agent_j.Sigma_q,
                agent_i.gauge.phi,
                agent_j.gauge.phi,
                agent_i.generators,
                beta_ij,
                direction='j',  # Gradient w.r.t. φ_j
                eps=1e-8,
            )

            # Apply λ and χ_ij weights to gauge gradients
            # Weight and accumulate gauge gradients
            if agent_i.base_manifold.is_point:
                # 0D: scalar weight
                chi_ij_scalar = float(chi_ij)
                gradients[agent_idx_i].grad_phi += lambda_phi * chi_ij_scalar * grad_phi_i
                gradients[j].grad_phi += lambda_phi * chi_ij_scalar * grad_phi_j
            else:
                # ND: broadcast weight
                chi_ij_broadcast = chi_ij[..., np.newaxis]  # (*S, 1)
                gradients[agent_idx_i].grad_phi += lambda_phi * chi_ij_broadcast * grad_phi_i
                gradients[j].grad_phi += lambda_phi * chi_ij_broadcast * grad_phi_j
    
    
    
    return gradients
    
   


# =============================================================================
# Gradient Term 3: Prior Alignment ∂E_prior/∂θ
# =============================================================================

def compute_prior_alignment_gradients(
    system,
    agent_idx_i: int,
    lambda_prior: float = 1.0,
    lambda_phi: float = 1.0,
) -> Dict[int, AgentGradients]:
    """
    Compute gradients of prior alignment for agent i with all neighbors.
    
    Identical structure to belief alignment, but for priors p instead of beliefs q.
    
    Energy term:
        E = Σ_j λ ∫ χ_ij γ_ij KL(p_i || Ω[p_j]) dc
    
    Args:
        system: MultiAgentSystem
        agent_idx_i: Agent index
        lambda_prior: Coupling strength
    
    Returns:
        gradients: Dict mapping agent idx → AgentGradients
    """
    agent_i = system.agents[agent_idx_i]
    neighbors = system.get_neighbors(agent_idx_i)
    
    if len(neighbors) == 0:
        spatial_shape = agent_i.support.base_manifold.shape
        K = agent_i.K
        return {
            agent_idx_i: AgentGradients(
                grad_mu_q=np.zeros((*spatial_shape, K), dtype=np.float32),
                grad_Sigma_q=np.zeros((*spatial_shape, K, K), dtype=np.float32),
                grad_mu_p=np.zeros((*spatial_shape, K), dtype=np.float32),
                grad_Sigma_p=np.zeros((*spatial_shape, K, K), dtype=np.float32),
                grad_phi=np.zeros((*spatial_shape, 3), dtype=np.float32)
            )
        }
    
    # Initialize gradients
    gradients = {}
    for idx in [agent_idx_i] + neighbors:
        agent = system.agents[idx]
        spatial_shape = agent.support.base_manifold.shape
        K = agent.K
        gradients[idx] = AgentGradients(
            grad_mu_q=np.zeros((*spatial_shape, K), dtype=np.float32),
            grad_Sigma_q=np.zeros((*spatial_shape, K, K), dtype=np.float32),
            grad_mu_p=np.zeros((*spatial_shape, K), dtype=np.float32),
            grad_Sigma_p=np.zeros((*spatial_shape, K, K), dtype=np.float32),
            grad_phi=np.zeros((*spatial_shape, 3), dtype=np.float32)
        )
    
    # Compute softmax weights γ_ij(c)
    gamma_fields = compute_softmax_weights(
        system, agent_idx_i, mode='prior', kappa=system.config.kappa_gamma
    )
    
    # Process each neighbor
    for j in neighbors:
        agent_j = system.agents[j]
        
        # Overlap and softmax weights
        chi_ij = agent_i.support.compute_overlap_continuous(agent_j.support)
        gamma_ij = gamma_fields[j]
        
        Omega_ij = system.compute_transport_ij(agent_idx_i, j)
     
        # Transported distributions (priors)
        p_i = GaussianDistribution(agent_i.mu_p, agent_i.Sigma_p)
        p_j = GaussianDistribution(agent_j.mu_p, agent_j.Sigma_p)
        p_j_transported = push_gaussian(p_j, Omega_ij)
        
        # Local gradients
        g_mu_i, g_Sigma_i = grad_kl_source(
            agent_i.mu_p, agent_i.Sigma_p,
            p_j_transported.mu, p_j_transported.Sigma
        )
        
        g_mu_j, g_Sigma_j = grad_kl_target(
            agent_i.mu_p, agent_i.Sigma_p,
            p_j_transported.mu, p_j_transported.Sigma,
            Omega_ij
        )
        
        weight_field = lambda_prior * chi_ij * gamma_ij
        
        if agent_i.base_manifold.is_point:
            # 0D: weight is scalar, multiply directly
            weight_scalar = float(weight_field)
            
            gradients[agent_idx_i].grad_mu_p += weight_scalar * g_mu_i
            gradients[agent_idx_i].grad_Sigma_p += weight_scalar * g_Sigma_i
            gradients[j].grad_mu_p += weight_scalar * g_mu_j
            gradients[j].grad_Sigma_p += weight_scalar * g_Sigma_j
        else:
            # ND: weight has spatial dimensions, broadcast properly
            weight_vec = weight_field[..., np.newaxis]  # (*S, 1)
            weight_mat = weight_field[..., np.newaxis, np.newaxis]  # (*S, 1, 1)
            
            gradients[agent_idx_i].grad_mu_p += weight_vec * g_mu_i
            gradients[agent_idx_i].grad_Sigma_p += weight_mat * g_Sigma_i
            gradients[j].grad_mu_p += weight_vec * g_mu_j
            gradients[j].grad_Sigma_p += weight_mat * g_Sigma_j
        
        
        # ===== GAUGE FIELD GRADIENTS (for priors) =====
        if lambda_phi != 0.0:
            grad_phi_i = compute_gauge_gradient_alignment(
                agent_i.mu_p, agent_i.Sigma_p,
                agent_j.mu_p, agent_j.Sigma_p,
                agent_i.gauge.phi,
                agent_j.gauge.phi,
                agent_i.generators,
                gamma_ij,
                direction='i',
                eps=1e-8,
            )
            
            grad_phi_j = compute_gauge_gradient_alignment(
                agent_i.mu_p, agent_i.Sigma_p,
                agent_j.mu_p, agent_j.Sigma_p,
                agent_i.gauge.phi,
                agent_j.gauge.phi,
                agent_i.generators,
                gamma_ij,
                direction='j',
                eps=1e-8,
            )

            # Weight and accumulate gauge gradients
            if agent_i.base_manifold.is_point:
                # 0D: scalar weight
                chi_ij_scalar = float(chi_ij)
                gradients[agent_idx_i].grad_phi += lambda_phi * chi_ij_scalar * grad_phi_i
                gradients[j].grad_phi += lambda_phi * chi_ij_scalar * grad_phi_j
            else:
                # ND: broadcast weight
                chi_ij_broadcast = chi_ij[..., np.newaxis]  # (*S, 1)
                gradients[agent_idx_i].grad_phi += lambda_phi * chi_ij_broadcast * grad_phi_i
                gradients[j].grad_phi += lambda_phi * chi_ij_broadcast * grad_phi_j

    
    return gradients





# =============================================================================
# Natural Gradient Projection
# =============================================================================


def project_to_natural_gradients(
    agent,
    euclidean_grads: AgentGradients
) -> AgentGradients:
    """
    Project Euclidean gradients to natural gradients via Fisher-Rao metric.
    
    NEW: Returns gradients w.r.t. Cholesky factors L, not Σ.
    
    Args:
        agent: Agent with current parameters (now has L_q, L_p)
        euclidean_grads: Accumulated Euclidean gradients (w.r.t. Σ)
    
    Returns:
        natural_grads: AgentGradients with delta_L fields (not delta_Sigma!)
    """
    # =========================================================================
    # Step 1: Project to natural gradients w.r.t. Σ (Fisher-Rao metric)
    # =========================================================================
    # This uses the geometry of the SPD manifold
    
    delta_mu_q, delta_Sigma_q = natural_gradient_gaussian(
        agent.mu_q, agent.Sigma_q,  # Sigma_q computed from L_q
        euclidean_grads.grad_mu_q,
        euclidean_grads.grad_Sigma_q
    )
    
    delta_mu_p, delta_Sigma_p = natural_gradient_gaussian(
        agent.mu_p, agent.Sigma_p,  # Sigma_p computed from L_p
        euclidean_grads.grad_mu_p,
        euclidean_grads.grad_Sigma_p
    )
    
    # =========================================================================
    # Step 2: Convert natural Σ gradients → L gradients (Cholesky chain rule)
    # =========================================================================
    # For Σ = LL^T:
    #   ∂S/∂L = 2 * (∂S/∂Σ) @ L
    #   (then project to lower triangular)
    
    delta_L_q = cholesky_gradient(delta_Sigma_q, agent.L_q)
    delta_L_p = cholesky_gradient(delta_Sigma_p, agent.L_p)
    
    # =========================================================================
    # Step 3: Gauge field (unchanged)
    # =========================================================================
    
    lambda_phi = getattr(agent.config, "lambda_phi", 1.0)
    if lambda_phi == 0.0:
        delta_phi = np.zeros_like(euclidean_grads.grad_phi)
    else:
        delta_phi = -euclidean_grads.grad_phi
    
   
  
    
    # =========================================================================
    # Return: Now with delta_L instead of delta_Sigma!
    # =========================================================================
    
    # =========================================================================
    # Return: keep BOTH Σ- and L-based natural gradients
    # =========================================================================

    return AgentGradients(
        # Euclidean gradients for diagnostics
        grad_mu_q=euclidean_grads.grad_mu_q,
        grad_Sigma_q=euclidean_grads.grad_Sigma_q,
        grad_mu_p=euclidean_grads.grad_mu_p,
        grad_Sigma_p=euclidean_grads.grad_Sigma_p,
        grad_phi=euclidean_grads.grad_phi,

        # Natural gradients in mean / covariance coordinates
        delta_mu_q=delta_mu_q,
        delta_Sigma_q=delta_Sigma_q,
        delta_mu_p=delta_mu_p,
        delta_Sigma_p=delta_Sigma_p,

        # Natural gradients w.r.t. Cholesky factors (used by Trainer)
        delta_L_q=delta_L_q,
        delta_L_p=delta_L_p,

        # Gauge (unchanged)
        delta_phi=delta_phi,
    )


# =============================================================================
# Backward-compat shim: dict-based gradients for MultiAgentSystem.step()
# =============================================================================

def compute_all_gradients(system, n_jobs: Optional[int] = None,
                          backend: str = "loky") -> List[Dict[str, np.ndarray]]:
    """
    Backward-compatible wrapper used by MultiAgentSystem.compute_gradients().
    
    Returns:
        List[Dict[str, np.ndarray]] with keys:
            'delta_mu_q', 'delta_Sigma_q',
            'delta_mu_p', 'delta_Sigma_p',
            'delta_phi'
    """
    # Reuse the validated natural-gradient computation
    natural_grads: List[AgentGradients] = compute_natural_gradients(
        system, n_jobs=n_jobs
    )

    grad_dicts: List[Dict[str, np.ndarray]] = []

    for agent, g in zip(system.agents, natural_grads):
        # Fallbacks in case some fields are None (e.g. lambda_prior_align == 0)
        delta_mu_q = g.delta_mu_q if g.delta_mu_q is not None else np.zeros_like(agent.mu_q)
        delta_Sigma_q = (
            g.delta_Sigma_q
            if g.delta_Sigma_q is not None
            else np.zeros_like(agent.Sigma_q)
        )

        # Priors might be disabled
        if getattr(system.config, "lambda_prior_align", 0.0) > 0.0:
            delta_mu_p = g.delta_mu_p if g.delta_mu_p is not None else np.zeros_like(agent.mu_p)
            delta_Sigma_p = (
                g.delta_Sigma_p
                if g.delta_Sigma_p is not None
                else np.zeros_like(agent.Sigma_p)
            )
        else:
            # Explicit zeros if prior term is off
            delta_mu_p = np.zeros_like(agent.mu_p)
            delta_Sigma_p = np.zeros_like(agent.Sigma_p)

        # Gauge field: if training disabled, force zero
        trains_phi = getattr(system.config, "lambda_phi", 1.0) != 0.0
        if trains_phi and g.delta_phi is not None:
            delta_phi = g.delta_phi
        else:
            delta_phi = np.zeros_like(agent.phi)

        grad_dicts.append(
            {
                "delta_mu_q": delta_mu_q,
                "delta_Sigma_q": delta_Sigma_q,
                "delta_mu_p": delta_mu_p,
                "delta_Sigma_p": delta_Sigma_p,
                "delta_phi": delta_phi,
            }
        )

    return grad_dicts






# =============================================================================
# Parallel Gradient Computation
# =============================================================================

def _compute_agent_euclidean_gradients(system, agent_idx: int) -> AgentGradients:
    """..."""
    agent = system.agents[agent_idx]
    spatial_shape = agent.support.base_shape if hasattr(agent.support, 'base_shape') else agent.base_manifold.shape
    K = agent.config.K
    enable_p = getattr(system.config, "lambda_prior_align", 0.0) > 0.0
    # p-gradients are gated by lambda_prior_align via `enable_p`
    
    # Initialize accumulator
    total_grads = AgentGradients(
        grad_mu_q=np.zeros((*spatial_shape, K), dtype=np.float32),
        grad_Sigma_q=np.zeros((*spatial_shape, K, K), dtype=np.float32),
        grad_mu_p=np.zeros((*spatial_shape, K), dtype=np.float32),
        grad_Sigma_p=np.zeros((*spatial_shape, K, K), dtype=np.float32),
        grad_phi=np.zeros((*spatial_shape, 3), dtype=np.float32)
    )
    
    
        
    # (1) Self-coupling
    self_grads = compute_self_coupling_gradients(
        agent, alpha=system.config.lambda_self
    )
    total_grads.grad_mu_q += self_grads.grad_mu_q
    total_grads.grad_Sigma_q += self_grads.grad_Sigma_q
    
    if enable_p:
        total_grads.grad_mu_p += self_grads.grad_mu_p
        total_grads.grad_Sigma_p += self_grads.grad_Sigma_p
    

    
    # (2) Belief alignment
    belief_grads_dict = compute_belief_alignment_gradients(
        system,
        agent_idx,
        lambda_belief=system.config.lambda_belief_align,
        lambda_phi=getattr(system.config, "lambda_phi", 1.0),
    )
    
    
    for idx, grads in belief_grads_dict.items():
        if idx == agent_idx:
            total_grads.grad_mu_q += grads.grad_mu_q
            total_grads.grad_Sigma_q += grads.grad_Sigma_q
            total_grads.grad_phi += grads.grad_phi
 


    # (3) Prior alignment (guarded)
    if enable_p and len(system.get_neighbors(agent_idx)) > 0:
        prior_grads_dict = compute_prior_alignment_gradients(
            system,
            agent_idx,
            lambda_prior=system.config.lambda_prior_align,
            lambda_phi=getattr(system.config, "lambda_phi", 1.0),
        )
        
        for idx, grads in prior_grads_dict.items():
            if idx == agent_idx:
                total_grads.grad_mu_p += grads.grad_mu_p
                total_grads.grad_Sigma_p += grads.grad_Sigma_p

    
    # (4) Observations
    if getattr(system.config, "has_observations", False):
    
        obs_grads = compute_observation_gradients(system, agent)  # ✅ Correct args
        total_grads.grad_mu_q += obs_grads.grad_mu_q
        total_grads.grad_Sigma_q += obs_grads.grad_Sigma_q
        
    return total_grads




def _accumulate_coupling_gradients(system, agent_idx: int, euclidean_grads: List[AgentGradients]):
    """
    Accumulate coupling gradients that affect multiple agents.

    Handles the cross-agent gradient terms from softmax coupling.
    MUST respect lambda_belief_align and lambda_prior_align.
    """
    lambda_belief = getattr(system.config, "lambda_belief_align", 0.0)
    lambda_prior  = getattr(system.config, "lambda_prior_align", 0.0)

    # (5) Softmax coupling (beliefs): when this agent is SENDER j
    if lambda_belief > 0.0:
        softmax_belief_grads = compute_softmax_coupling_gradients_belief(system, agent_idx)
        
        for idx, grads in softmax_belief_grads.items():
            euclidean_grads[idx].grad_mu_q     += lambda_belief * grads.grad_mu_q
            euclidean_grads[idx].grad_Sigma_q  += lambda_belief * grads.grad_Sigma_q
            if hasattr(grads, "grad_mu_p") and grads.grad_mu_p is not None:
                euclidean_grads[idx].grad_mu_p += lambda_belief * grads.grad_mu_p
            if hasattr(grads, "grad_Sigma_p") and grads.grad_Sigma_p is not None:
                euclidean_grads[idx].grad_Sigma_p += lambda_belief * grads.grad_Sigma_p
            if hasattr(grads, "grad_phi") and grads.grad_phi is not None:
                euclidean_grads[idx].grad_phi += lambda_belief * grads.grad_phi

    # (6) Softmax coupling (priors): when this agent is SENDER j
    if lambda_prior > 0.0:
        softmax_prior_grads = compute_softmax_coupling_gradients_prior(system, agent_idx)
        
        for idx, grads in softmax_prior_grads.items():
            euclidean_grads[idx].grad_mu_p     += lambda_prior * grads.grad_mu_p
            euclidean_grads[idx].grad_Sigma_p  += lambda_prior * grads.grad_Sigma_p
            if hasattr(grads, "grad_mu_q") and grads.grad_mu_q is not None:
                euclidean_grads[idx].grad_mu_q += lambda_prior * grads.grad_mu_q
            if hasattr(grads, "grad_Sigma_q") and grads.grad_Sigma_q is not None:
                euclidean_grads[idx].grad_Sigma_q += lambda_prior * grads.grad_Sigma_q
            if hasattr(grads, "grad_phi") and grads.grad_phi is not None:
                euclidean_grads[idx].grad_phi += lambda_prior * grads.grad_phi





def compute_natural_gradients(
    system,
    n_jobs: Optional[int] = None,
    verbose: int = 0
) -> List[AgentGradients]:
    """
    Parallel gradient computation leveraging multi-core CPU.
    
    Strategy:
    ---------
    1. **Parallel phase**: Compute agent-local gradients (self + alignment as receiver)
    2. **Sequential phase**: Accumulate cross-agent coupling gradients
    3. **Parallel phase**: Project to natural gradients (Fisher-Rao metric)
    
    Args:
        system: MultiAgentSystem
        n_jobs: Number of parallel jobs
                None = auto-detect (use all cores)
                -1 = use all cores
                1 = sequential (for debugging)
                N = use N cores
        verbose: Verbosity level (0=silent, 10=debug)
    
    Returns:
        gradients: List[AgentGradients] with natural gradients
    
    Performance:
        Ryzen 9 9900X (12C/24T): Expected ~8-15x speedup vs sequential
    """
    n_agents = system.n_agents
    
    # Auto-detect cores if not specified
    if n_jobs is None:
        n_jobs = -1  # Use all available
    
    # ==========================================================================
    # PHASE 1: Parallel agent-local gradient computation
    # ==========================================================================
    # Each agent computes its own gradients independently
    # This is the bottleneck - perfect for parallelization!
    
    with parallel_backend('loky', n_jobs=n_jobs):
        euclidean_grads = Parallel(verbose=verbose)(
            delayed(_compute_agent_euclidean_gradients)(system, i)
            for i in range(n_agents)
        )
    
    # ==========================================================================
    # PHASE 2: Sequential accumulation of coupling gradients
    # ==========================================================================
    # Cross-agent terms from softmax coupling must be accumulated sequentially
    # (These are much faster than phase 1, so sequential is fine)
    
    for i in range(n_agents):
        _accumulate_coupling_gradients(system, i, euclidean_grads)
    
    # ==========================================================================
    # PHASE 3: Parallel projection to natural gradients
    # ==========================================================================
    # Fisher-Rao metric projection is independent per agent
    
    with parallel_backend('loky', n_jobs=n_jobs):
        natural_grads = Parallel(verbose=verbose)(
            delayed(project_to_natural_gradients)(system.agents[i], euclidean_grads[i])
            for i in range(n_agents)
        )
    
    return natural_grads






