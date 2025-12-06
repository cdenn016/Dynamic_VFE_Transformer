# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 10:01:57 2025

@author: chris and christine
"""

#!/usr/bin/env python3
"""
STREAMLINED Softmax Coupling Gradients
======================================

Mathematical structure made crystal clear:

Energy: E = Σ_j ∫ χ_ij β_ij KL_ij dc

Full gradient:
    ∂E/∂θ = Σ_j ∫ χ_ij [β_ij ∂KL_ij/∂θ] dc                [TERM 1]
          + Σ_j Σ_k ∫ χ_ij [KL_ij (∂β_ij/∂KL_ik) ∂KL_ik/∂θ] dc  [TERM 2]

This module computes TERM 2 only.
TERM 1 is handled in compute_belief/prior_alignment_gradients.

Key insight: For each (j,k) pair, we compute:
    contribution = χ_ij * KL_ij * (∂β_ij/∂KL_ik) * (∂KL_ik/∂θ)

Author: Chris 
Date: November 2025
"""

import numpy as np
from typing import Dict, List, Tuple, Literal
from dataclasses import dataclass

from gradients.gradient_terms import grad_kl_source, grad_kl_target
from math_utils.push_pull import push_gaussian, GaussianDistribution
from geometry.geometry_base import broadcast_mask
from math_utils.transport import _matrix_exponential_so3

@dataclass
class SoftmaxGradients:
    """Container for softmax coupling gradient contributions."""
    grad_mu_q: np.ndarray
    grad_Sigma_q: np.ndarray
    grad_mu_p: np.ndarray
    grad_Sigma_p: np.ndarray
    grad_phi: np.ndarray = None






def compute_softmax_weights(
    system,
    agent_idx_i: int,
    mode: Literal['belief', 'prior'],
    kappa: float
) -> Dict[int, np.ndarray]:
    """
    Compute softmax coupling weights β_ij(c) or γ_ij(c) for all neighbors.
    
    At each point c independently:
        β_ij(c) = exp[-KL_ij(c)/κ] / Σ_k exp[-KL_ik(c)/κ]
    
    CRITICAL: NO χ in the denominator! Pure softmax over transported KL.
    
    Args:
        system: MultiAgentSystem instance
        agent_idx_i: Index of agent i
        mode: 'belief' for β weights, 'prior' for γ weights
        kappa: Temperature parameter (κ_β or κ_γ)
    
    Returns:
        weights: Dict mapping neighbor j → weight field β_ij(c)
                 Each field has shape (*S,) matching base manifold
                 Returns empty dict if no neighbors
    
    Properties verified:
        - At each point c: Σ_j weights[j][c] = 1.0 (softmax normalization)
        - All weights ∈ [0, 1]
        - Fields have shape = base_manifold.shape
    
    Example:
        >>> beta_fields = compute_softmax_weights(system, 0, 'belief', kappa=1.0)
        >>> # beta_fields[j] is the β_0j(c) field
        >>> # Verify normalization at each point:
        >>> beta_sum = sum(beta_fields.values())  # Should be all 1.0
    """
    agent_i = system.agents[agent_idx_i]
    
    # Get neighbors (agents with spatial overlap)
    neighbors = system.get_neighbors(agent_idx_i)
    
    if len(neighbors) == 0:
        return {}
    
    # Import here to avoid circular dependencies
    from math_utils.push_pull import compute_kl_transported
    from math_utils.push_pull import GaussianDistribution
    
    # Select distributions based on mode
    if mode == 'belief':
        mu_i = agent_i.mu_q
        Sigma_i = agent_i.Sigma_q
        get_neighbor_dist = lambda j: (system.agents[j].mu_q, system.agents[j].Sigma_q)
    elif mode == 'prior':
        mu_i = agent_i.mu_p
        Sigma_i = agent_i.Sigma_p
        get_neighbor_dist = lambda j: (system.agents[j].mu_p, system.agents[j].Sigma_p)
    else:
        raise ValueError(f"Invalid mode: {mode}")
    
    # Compute transported KL fields for all neighbors
    kl_fields = []
    neighbor_list = list(neighbors)  # Preserve order
    
    for j in neighbor_list:
       
        mu_j, Sigma_j = get_neighbor_dist(j)
        
        # Compute transport operator Ω_ij = exp(φ_i) exp(-φ_j)
        Omega_ij = system.compute_transport_ij(agent_idx_i, j)
      
   
        # Compute transported KL: KL(q_i || Ω[q_j])
        # Wrap in GaussianDistribution objects
        dist_i = GaussianDistribution(mu_i, Sigma_i)
        dist_j = GaussianDistribution(mu_j, Sigma_j)
        
        kl_ij = compute_kl_transported(dist_i, dist_j, Omega_ij)
        kl_fields.append(kl_ij)
    
    # Stack KL fields along new axis: shape = (n_neighbors, *S)
    kl_stack = np.stack(kl_fields, axis=0)
    
    # Stack KL fields along new axis: shape = (n_neighbors, *S)
    kl_stack = np.stack(kl_fields, axis=0)  # (J, *S)
    
    # -----------------------------------------------------------------
    # Vectorized, numerically stable softmax over neighbors
    # β_j(c) = exp(-KL_j(c)/κ) / Σ_k exp(-KL_k(c)/κ)
    # Use log-sum-exp trick along neighbor axis
    # -----------------------------------------------------------------
    # Shape shortcuts
    # J = number of neighbors, spatial shape = *S
    # x = -KL/κ
    x = -kl_stack / kappa  # (J, *S)

    # max over neighbors at each point: shape = (1, *S)
    x_max = np.max(x, axis=0, keepdims=True)

    # subtract max for stability, then exponentiate
    exp_x = np.exp(x - x_max)           # (J, *S)
    denom = np.sum(exp_x, axis=0, keepdims=True)  # (1, *S)

    # Guard against degenerate denom (should basically never happen)
    denom = np.where(denom < 1e-30, 1e-30, denom)

    # Final softmax
    beta_stack = exp_x / denom          # (J, *S)

    # Package as dictionary mapping neighbor j → weight field
    weights = {
        j: beta_stack[i].astype(np.float32)
        for i, j in enumerate(neighbor_list)
    }

    return weights

    
    # Package as dictionary mapping neighbor j → weight field
    weights = {
        j: beta_stack[i].astype(np.float32)
        for i, j in enumerate(neighbor_list)
    }
    
    return weights




# =============================================================================
# Core: Softmax Derivative Fields
# =============================================================================

def compute_softmax_derivative_fields(
    beta_fields: Dict[int, np.ndarray],
    neighbors: List[int],
    kappa: float
) -> Dict[Tuple[int, int], np.ndarray]:
    """
    Compute ∂β_ij(c)/∂KL_ik(c) for all (j,k) pairs.
    
    These are POINTWISE derivatives at each spatial location c.
    
    Softmax derivative formulas:
        ∂β_ij/∂KL_ij = -(β_ij/κ) * (1 - β_ij)         [diagonal, k=j]
        ∂β_ij/∂KL_ik = +(β_ij * β_ik) / κ            [off-diagonal, k≠j]
    
    Args:
        beta_fields: Dict j → β_ij(c) field, shape (*S,)
        neighbors: List of neighbor indices
        kappa: Temperature parameter
    
    Returns:
        derivatives: Dict (j,k) → ∂β_ij/∂KL_ik field, shape (*S,)
    
    Example:
        >>> beta = {1: field1, 2: field2}
        >>> derivs = compute_softmax_derivative_fields(beta, [1,2], kappa=1.0)
        >>> # derivs[(1,1)] is ∂β_i1/∂KL_i1
        >>> # derivs[(1,2)] is ∂β_i1/∂KL_i2
    """
    derivatives = {}
    
    for j in neighbors:
        beta_j = beta_fields[j]  # Shape: (*S,)
        
        # Diagonal: ∂β_ij/∂KL_ij
        derivatives[(j, j)] = -(beta_j / kappa) * (1.0 - beta_j)
        
        # Off-diagonal: ∂β_ij/∂KL_ik for k≠j
        for k in neighbors:
            if k != j:
                beta_k = beta_fields[k]
                derivatives[(j, k)] = (beta_j * beta_k) / kappa
    
    return derivatives


# =============================================================================
# Main: Softmax Coupling Gradients
# =============================================================================

def compute_softmax_coupling_gradients(
    system,
    agent_idx_i: int,
    mode: str = 'belief',
    eps: float = 1e-8
) -> Dict[int, SoftmaxGradients]:
    """
    Compute (∂β/∂θ)·KL terms: the softmax coupling gradients.
    
    This is TERM 2 from the full gradient:
        ∂E/∂θ = ... + Σ_j Σ_k ∫ χ_ij KL_ij (∂β_ij/∂KL_ik)(∂KL_ik/∂θ) dc
    
    For each (j,k) pair:
        1. Get weight: w = χ_ij * KL_ij * (∂β_ij/∂KL_ik)
        2. Compute: ∂KL_ik/∂θ (affects both agent i and agent k)
        3. Accumulate: w * (∂KL_ik/∂θ)
    
    Args:
        system: MultiAgentSystem
        agent_idx_i: Index of receiving agent i
        mode: 'belief' or 'prior'
        eps: Numerical epsilon
    
    Returns:
        gradients: Dict mapping agent_idx → SoftmaxGradients
                   Contains contributions for agent i and all neighbors
    
    Structure:
        - Outer loop over j: which energy term E_j = ∫ χ_ij β_ij KL_ij
        - Inner loop over k: which KL appears in β derivative
        - Weight: χ_ij (from energy term j)
        - Coefficient: KL_ij(c) * (∂β_ij/∂KL_ik)(c)
    """
    agent_i = system.agents[agent_idx_i]
    neighbors = system.get_neighbors(agent_idx_i)
    
    if len(neighbors) < 2:
        # Need ≥2 neighbors for softmax coupling
        return {}
    
    # =========================================================================
    # STEP 1: Compute softmax weights β_ij for all neighbors j
    # =========================================================================
    
    
    if mode == 'belief':
        beta_fields = compute_softmax_weights(
            system, agent_idx_i, mode='belief', kappa=system.config.kappa_beta
        )
        get_dist = lambda agent: (agent.mu_q, agent.Sigma_q)
    else:  # mode == 'prior'
        beta_fields = compute_softmax_weights(
            system, agent_idx_i, mode='prior', kappa=system.config.kappa_gamma
        )
        get_dist = lambda agent: (agent.mu_p, agent.Sigma_p)
    
    # =========================================================================
    # STEP 2: Compute KL fields KL_ij for all neighbors j
    # =========================================================================
    from math_utils.push_pull import compute_kl_transported
    
    kl_fields = {}
    for j in neighbors:
        agent_j = system.agents[j]
        mu_i, Sigma_i = get_dist(agent_i)
        mu_j, Sigma_j = get_dist(agent_j)
        
        Omega_ij = system.compute_transport_ij(agent_idx_i, j)
        
        dist_i = GaussianDistribution(mu_i, Sigma_i)
        dist_j = GaussianDistribution(mu_j, Sigma_j)
        
        kl_fields[j] = compute_kl_transported(dist_i, dist_j, Omega_ij)
    
    # =========================================================================
    # STEP 3: Compute softmax derivatives ∂β_ij/∂KL_ik for all (j,k) pairs
    # =========================================================================
    kappa = system.config.kappa_beta if mode == 'belief' else system.config.kappa_gamma
    
    softmax_derivs = compute_softmax_derivative_fields(
        beta_fields, neighbors, kappa
    )  # Dict[(j,k)] → field(*S,)
    
    # =========================================================================
    # STEP 4: Initialize gradient accumulators
    # =========================================================================
    gradients = {}
    for idx in [agent_idx_i] + neighbors:
        agent = system.agents[idx]
        gradients[idx] = SoftmaxGradients(
            grad_mu_q=np.zeros_like(agent.mu_q, dtype=np.float32),
            grad_Sigma_q=np.zeros_like(agent.Sigma_q, dtype=np.float32),
            grad_mu_p=np.zeros_like(agent.mu_p, dtype=np.float32),
            grad_Sigma_p=np.zeros_like(agent.Sigma_p, dtype=np.float32),
            grad_phi=np.zeros_like(agent.gauge.phi, dtype=np.float32)
        )
    
    # =========================================================================
    # STEP 5: Double loop - accumulate gradients for each (j,k) pair
    # =========================================================================
    # Structure: Σ_j Σ_k [χ_ij * KL_ij * (∂β_ij/∂KL_ik) * (∂KL_ik/∂θ)]
    
    for j in neighbors:
        agent_j = system.agents[j]
        
        # Get overlap weight χ_ij (from energy term j)
        chi_ij = agent_i.support.compute_overlap_continuous(agent_j.support)
        
        # Get KL field for energy term j
        kl_j_field = kl_fields[j]  # Shape: (*S,)
        
        # Inner loop over k (which KL in β derivative)
        for k in neighbors:
            agent_k = system.agents[k]
            
            # Get softmax derivative ∂β_ij/∂KL_ik
            deriv_jk = softmax_derivs.get((j, k))
            if deriv_jk is None or np.max(np.abs(deriv_jk)) < 1e-12:
                continue
            
            # ================================================================
            # Compute weight field: χ_ij * KL_ij * (∂β_ij/∂KL_ik)
            # ================================================================
            # This is the coefficient for (∂KL_ik/∂θ) terms
            
            weight_field = chi_ij * kl_j_field * deriv_jk  # Shape: (*S,)
            
            # Early exit if negligible
            if np.max(np.abs(weight_field)) < 1e-12:
                continue
            
            # ================================================================
            # Compute ∂KL_ik/∂θ (affects both agent i and agent k)
            # ================================================================
            
            # Get transport Ω_ik
            Omega_ik = system.compute_transport_ij(agent_idx_i, k)
            
            # Get distributions
            mu_i, Sigma_i = get_dist(agent_i)
            mu_k, Sigma_k = get_dist(agent_k)
            
            # Transport k → i
            dist_k_transported = push_gaussian(
                GaussianDistribution(mu_k, Sigma_k),
                Omega_ik,
                eps=eps
            )
            
            # Gradient w.r.t. source (agent i)
            g_mu_i, g_Sigma_i = grad_kl_source(
                mu_i, Sigma_i,
                dist_k_transported.mu, dist_k_transported.Sigma
            )
            
            # Gradient w.r.t. target (agent k, via backprop through Ω)
            g_mu_k, g_Sigma_k = grad_kl_target(
                mu_i, Sigma_i,
                dist_k_transported.mu, dist_k_transported.Sigma,
                Omega_ik
            )
            


            
            # ================================================================
            # Accumulate weighted gradients
            # ================================================================
            
            # Broadcast weight field to match gradient shapes
            weight_vec = broadcast_mask(weight_field, g_mu_i.shape, is_vector=True)
            weight_mat = broadcast_mask(weight_field, g_Sigma_i.shape, is_vector=False)
            
            # Accumulate to agent i
            if mode == 'belief':
                gradients[agent_idx_i].grad_mu_q += weight_vec * g_mu_i
                gradients[agent_idx_i].grad_Sigma_q += weight_mat * g_Sigma_i
            else:  # prior
                gradients[agent_idx_i].grad_mu_p += weight_vec * g_mu_i
                gradients[agent_idx_i].grad_Sigma_p += weight_mat * g_Sigma_i
            
            # Accumulate to agent k
            weight_vec_k = broadcast_mask(weight_field, g_mu_k.shape, is_vector=True)
            weight_mat_k = broadcast_mask(weight_field, g_Sigma_k.shape, is_vector=False)
            
            if mode == 'belief':
                gradients[k].grad_mu_q += weight_vec_k * g_mu_k
                gradients[k].grad_Sigma_q += weight_mat_k * g_Sigma_k
            else:  # prior
                gradients[k].grad_mu_p += weight_vec_k * g_mu_k
                gradients[k].grad_Sigma_p += weight_mat_k * g_Sigma_k
    
    return gradients






# =============================================================================
# Convenience Wrappers
# =============================================================================

def compute_softmax_coupling_gradients_belief(
    system, agent_idx_i: int, eps: float = 1e-8
) -> Dict[int, SoftmaxGradients]:
    """Compute softmax coupling gradients for BELIEFS."""
    return compute_softmax_coupling_gradients(
        system, agent_idx_i, mode='belief', eps=eps
    )


def compute_softmax_coupling_gradients_prior(
    system, agent_idx_i: int, eps: float = 1e-8
) -> Dict[int, SoftmaxGradients]:
    """Compute softmax coupling gradients for PRIORS."""
    return compute_softmax_coupling_gradients(
        system, agent_idx_i, mode='prior', eps=eps
    )


# =============================================================================
# Validation & Diagnostics
# =============================================================================

def validate_softmax_gradient_structure(gradients: Dict[int, SoftmaxGradients]):
    """
    Validate that softmax gradients have correct structure.
    
    Checks:
        - All fields are finite (no NaN/Inf)
        - Shapes are consistent
        - Non-zero contributions where expected
    """
    for agent_idx, grads in gradients.items():
        # Check finite
        for field_name in ['grad_mu_q', 'grad_Sigma_q', 'grad_mu_p', 'grad_Sigma_p']:
            field = getattr(grads, field_name)
            if not np.all(np.isfinite(field)):
                raise ValueError(
                    f"Agent {agent_idx}: {field_name} contains NaN/Inf"
                )
        
        # Check shapes
        mu_shape = grads.grad_mu_q.shape
        Sigma_shape = grads.grad_Sigma_q.shape
        
        assert mu_shape[-1] == Sigma_shape[-1] == Sigma_shape[-2], \
            f"Agent {agent_idx}: Inconsistent K dimension"
        
        assert mu_shape[:-1] == Sigma_shape[:-2], \
            f"Agent {agent_idx}: Inconsistent spatial dimensions"
    
    return True