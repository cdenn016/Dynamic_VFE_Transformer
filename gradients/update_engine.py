# -*- coding: utf-8 -*-
"""
Created on Sun Nov 16 15:30:18 2025

@author: chris and christine
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared Gradient Update Engine
==============================

Pure update logic extracted from Trainer and HierarchicalEvolutionEngine.
Eliminates duplication and ensures both training paths use identical math.

This module contains ONLY the parameter update logic - no system-specific
orchestration, no energy computation, no gradient computation. Just the
pure mathematical transformations that map gradients to parameter updates.

Key principle: Both standard and hierarchical training should use the EXACT
same update equations. This module is the single source of truth.

Author: Chris
Date: November 2025
"""

import numpy as np
from typing import List, Optional
from dataclasses import dataclass

from gradients.retraction import retract_spd
from gradients.gauge_fields import retract_to_principal_ball
from agent.agents import Agent


class GradientApplier:
    """
    Pure gradient application logic shared between all trainers.

    This class contains NO state - all methods are static utilities.
    It's responsible for applying gradients to agent parameters using
    proper manifold-aware retractions.

    Usage:
        >>> # In Trainer.step()
        >>> gradients = compute_natural_gradients(system)
        >>> GradientApplier.apply_updates(system.agents, gradients, config)

        >>> # In HierarchicalEvolutionEngine.evolve_step()
        >>> active_agents = system.get_all_active_agents()
        >>> gradients = compute_natural_gradients(adapter)
        >>> GradientApplier.apply_updates(active_agents, gradients, hier_config)
    """

    @staticmethod
    def apply_updates(
        agents: List[Agent],
        gradients: List,  # List[AgentGradients]
        config,  # TrainingConfig or HierarchicalConfig
        *,
        verbose: bool = False
    ):
        """
        Apply natural gradient updates to all agents.

        Updates are applied using proper manifold retractions:
        - Means: Euclidean update (μ ← μ + lr·δμ)
        - Covariances: SPD manifold retraction (ensures positive-definite)
        - Gauge fields: SO(3) retraction to principal ball

        CRITICAL: Natural gradients are already negated (descent directions),
        so we use ADDITION: param_new = param + lr * delta

        Args:
            agents: List of agents to update
            gradients: List of gradient objects (same length as agents)
            config: Configuration with learning rates (lr_mu_q, lr_sigma_q, etc.)
            verbose: Print update norms for debugging

        Post-conditions:
            - All covariances remain positive-definite
            - Gauge fields remain in principal ball
            - Support constraints are enforced
            - Agent caches are invalidated
        """
        if len(agents) != len(gradients):
            raise ValueError(
                f"Mismatch: {len(agents)} agents but {len(gradients)} gradients"
            )

        for agent, grad in zip(agents, gradients):
            GradientApplier._update_single_agent(agent, grad, config, verbose)

    @staticmethod
    def _update_single_agent(agent: Agent, grad, config, verbose: bool):
        """
        Apply updates to a single agent.

        Internal method - use apply_updates() instead.
        """
        # Get learning rates from config (works for both TrainingConfig and HierarchicalConfig)
        lr_mu_q = getattr(config, 'lr_mu_q', 0.1)
        lr_sigma_q = getattr(config, 'lr_sigma_q', 0.001)
        lr_mu_p = getattr(config, 'lr_mu_p', 0.1)
        lr_sigma_p = getattr(config, 'lr_sigma_p', 0.001)
        lr_phi = getattr(config, 'lr_phi', 0.1)

        # Trust regions and condition limits for SPD manifold updates
        # These are proper manifold constraints, not arbitrary clipping
        trust_region = getattr(config, 'trust_region_sigma', None)
        max_condition = getattr(config, 'sigma_max_condition', None)
        gauge_margin = getattr(config, 'gauge_margin', 1e-2)
        retraction_mode = getattr(config, 'retraction_mode_phi', 'mod2pi')

        # -----------------------------------------------------------------
        # 1. Update belief mean μ_q (Euclidean)
        # -----------------------------------------------------------------
        if lr_mu_q != 0.0 and grad.delta_mu_q is not None:
            agent.mu_q = agent.mu_q + lr_mu_q * grad.delta_mu_q

        # -----------------------------------------------------------------
        # 2. Update belief covariance Σ_q (SPD manifold, GAUGE-COVARIANT)
        # -----------------------------------------------------------------
        if lr_sigma_q != 0.0 and grad.delta_Sigma_q is not None:
            Sigma_q_new = retract_spd(
                agent.Sigma_q,
                grad.delta_Sigma_q,
                step_size=lr_sigma_q,
                trust_region=trust_region,
                max_condition=max_condition,
            )
            agent.Sigma_q = Sigma_q_new.astype(np.float32)
            # Invalidate Cholesky cache (L_q computed on-demand)
            if hasattr(agent, '_L_q_cache'):
                agent._L_q_cache = None

        # -----------------------------------------------------------------
        # 3. Update prior mean μ_p (Euclidean)
        # -----------------------------------------------------------------
        if lr_mu_p != 0.0 and grad.delta_mu_p is not None:
            agent.mu_p = agent.mu_p + lr_mu_p * grad.delta_mu_p

        # -----------------------------------------------------------------
        # 4. Update prior covariance Σ_p (SPD manifold, GAUGE-COVARIANT)
        # -----------------------------------------------------------------
        if lr_sigma_p != 0.0 and grad.delta_Sigma_p is not None:
            Sigma_p_new = retract_spd(
                agent.Sigma_p,
                grad.delta_Sigma_p,
                step_size=lr_sigma_p,
                trust_region=trust_region,
                max_condition=max_condition,
            )
            agent.Sigma_p = Sigma_p_new.astype(np.float32)
            # Invalidate Cholesky cache (L_p computed on-demand)
            if hasattr(agent, '_L_p_cache'):
                agent._L_p_cache = None

        # -----------------------------------------------------------------
        # 5. Update gauge field φ (SO(3) → principal ball)
        # -----------------------------------------------------------------
        if lr_phi != 0.0 and grad.delta_phi is not None:
            phi_new = agent.gauge.phi + lr_phi * grad.delta_phi
            agent.gauge.phi = retract_to_principal_ball(
                phi_new,
                margin=gauge_margin,
                mode=retraction_mode,
            )

        # -----------------------------------------------------------------
        # 6. Post-update enforcement
        # -----------------------------------------------------------------
        # Re-enforce support constraints (ensures fields are zero outside support)
        if hasattr(agent, 'enforce_support_constraints'):
            agent.enforce_support_constraints()

        # Invalidate cached computations
        if hasattr(agent, 'invalidate_caches'):
            agent.invalidate_caches()

        # Optional: Verbose logging
        if verbose:
            mu_norm = np.linalg.norm(grad.delta_mu_q) if grad.delta_mu_q is not None else 0.0
            sigma_norm = np.linalg.norm(grad.delta_Sigma_q) if grad.delta_Sigma_q is not None else 0.0
            phi_norm = np.linalg.norm(grad.delta_phi) if grad.delta_phi is not None else 0.0
            print(f"  Agent {agent.agent_id}: |δμ|={mu_norm:.3e}, |δΣ|={sigma_norm:.3e}, |δφ|={phi_norm:.3e}")

    @staticmethod
    def apply_identical_priors_lock(agents: List[Agent]):
        """
        Enforce identical priors across all agents.

        Computes shared prior (μ_p, Σ_p) from average and applies to all agents.

        GAUGE-COVARIANT: We average Σ_p directly (not Cholesky factors).
        Averaging on the SPD manifold (arithmetic mean is SPD if all inputs are SPD).

        Args:
            agents: List of agents to synchronize

        Post-conditions:
            - All agents have identical μ_p
            - All agents have identical Σ_p
            - Cholesky caches invalidated
        """
        if len(agents) == 0:
            return

        # Compute average prior (using Σ_p, GAUGE-COVARIANT!)
        mu_p_sum = sum(agent.mu_p for agent in agents)
        Sigma_p_sum = sum(agent.Sigma_p for agent in agents)

        mu_p_shared = mu_p_sum / len(agents)
        Sigma_p_shared = Sigma_p_sum / len(agents)

        # Apply to all agents
        for agent in agents:
            agent.mu_p = mu_p_shared.copy()
            agent.Sigma_p = Sigma_p_shared.copy()

            # Invalidate Cholesky cache
            if hasattr(agent, '_L_p_cache'):
                agent._L_p_cache = None

            if hasattr(agent, 'invalidate_caches'):
                agent.invalidate_caches()

    @staticmethod
    def apply_identical_priors_lock_to_scale(system, scale: int = 0):
        """
        Apply identical priors lock to agents at a specific scale.

        Convenience method for hierarchical systems that need to lock
        only base agents (scale 0) while leaving meta-agents free.

        Args:
            system: MultiScaleSystem instance
            scale: Which scale to lock (default: 0 = base agents)
        """
        if not hasattr(system, 'agents'):
            raise ValueError("System must have 'agents' attribute (dict or list)")

        # Handle both MultiScaleSystem (dict) and MultiAgentSystem (list)
        if isinstance(system.agents, dict):
            agents_at_scale = system.agents.get(scale, [])
        else:
            agents_at_scale = system.agents if scale == 0 else []

        if len(agents_at_scale) > 0:
            GradientApplier.apply_identical_priors_lock(agents_at_scale)


# =============================================================================
# Convenience Functions
# =============================================================================

def apply_gradients_standard(system, gradients, config):
    """
    Apply gradients in standard training mode.

    Convenience wrapper for Trainer.

    Args:
        system: MultiAgentSystem instance
        gradients: List of AgentGradients
        config: TrainingConfig
    """
    GradientApplier.apply_updates(system.agents, gradients, config)

    # Post-processing: identical priors lock
    if getattr(config, 'identical_priors', 'off') == 'lock':
        GradientApplier.apply_identical_priors_lock(system.agents)


def apply_gradients_hierarchical(system, active_agents, gradients, config):
    """
    Apply gradients in hierarchical training mode.

    Convenience wrapper for HierarchicalEvolutionEngine.

    Args:
        system: MultiScaleSystem instance
        active_agents: List of active agents at all scales
        gradients: List of AgentGradients
        config: HierarchicalConfig
    """
    GradientApplier.apply_updates(active_agents, gradients, config)

    # Post-processing: identical priors lock (base agents only)
    if hasattr(system, 'system_config'):
        system_config = system.system_config
    else:
        system_config = config

    if getattr(system_config, 'identical_priors', 'off') == 'lock':
        GradientApplier.apply_identical_priors_lock_to_scale(system, scale=0)