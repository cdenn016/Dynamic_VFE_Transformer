# -*- coding: utf-8 -*-
"""
Multi-Agent System on Principal Bundle (CORRECT VERSION)
========================================================

Integration architecture:
-------------------------
Agents with smooth Gaussian boundaries χ_i(c) ∈ [0,1] require proper
integration weighting by the overlap product χ_{ij}(c) = χ_i(c) · χ_j(c).

Energy: ∫ χ_{ij}(c) · β_{ij}(c) · KL(...) dc
        ^^^^^^^^^^   ^^^^^^^^^^
        geometric    dynamic
        weight       coupling

Author: Chris & Christine  
Date: November 2025
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Literal

from agent.agents import Agent
from math_utils.transport import compute_transport
from math_utils.transport_cache import TransportCache
from geometry.connection import ConnectionField
from config import SystemConfig


# =============================================================================
# Multi-Agent System
# =============================================================================

class MultiAgentSystem:
    """
    System of multiple agents on principal bundle.
    
    Integration Architecture:
    ------------------------
    Agents have smooth support boundaries χ_i(c) ∈ [0,1] (Gaussian masks).
    
    Overlap weight:
        χ_{ij}(c) = χ_i(c) · χ_j(c) ∈ [0,1]
        
    Energy integrals:
        E = ∫ χ_{ij}(c) · β_{ij}(c) · KL(q_i || Ω_ij[q_j]) dc
    
    Where:
        - χ_{ij}: Geometric weight (how strongly both agents exist)
        - β_{ij}: Dynamic coupling (softmax from KL divergences)
        - Both are CONTINUOUS fields
    
    Storage:
        overlap_masks[(i,j)] = χ_i · χ_j  (CONTINUOUS float [0,1])
    
    Usage:
        chi_ij = system.get_overlap_mask(i, j)  # Continuous
        beta_ij = system.compute_softmax_weights(i, 'belief')[j]
        energy = np.sum(chi_ij * beta_ij * kl)  # Proper weighted integral
    
    Attributes:
        agents: List of Agent objects
        config: System configuration
        n_agents: Number of agents
        overlap_masks: Dict[(i,j) -> continuous float array [0,1]]
        transport_cache: Optional cache for Ω_ij operators
        connection: Optional ConnectionField for base manifold
    """
    

    def __init__(self, agents, config=None):
        """
        Initialize multi-agent system.
        
        Args:
            agents: List of Agent objects
            config: System configuration (uses defaults if None)
        """
        self.agents = agents
        self.config = config if config is not None else SystemConfig()
        self.n_agents = len(agents)
        
        if self.n_agents == 0:
            raise ValueError("Must have at least one agent")
        
        # Validate agents are compatible
        self._validate_agents()
        
        # Initialize connection field (optional)
        self.connection = self._initialize_connection()
        
        # Initialize transport cache (optional)
        if self.config.cache_transports:
            self.transport_cache = TransportCache(self, max_size=self.config.cache_size)
        else:
            self.transport_cache = None
        
        # Compute continuous overlap weights
        self.overlap_masks = self._compute_overlap_masks()
        
        # ✓ Observation model (initialized lazily)
        self.W_obs = None       # Observation matrix
        self.R_obs = None       # Observation covariance (not Lambda_obs!)
        self.x_true = None      # Ground truth (added by initialize_observation_model)
        # system.py  (in __init__ after self._validate_agents(), before printing summary)
        if getattr(self.config, "identical_priors", "off") in ("init_copy", "lock"):
            self._apply_identical_priors_now()

        
        
        # Print summary
        print("✓ Multi-agent system initialized:")
        print(f"  Agents: {self.n_agents}")
        print(f"  Total parameters: {self.count_total_parameters():,}")
        print(f"  Active overlaps: {len(self.overlap_masks)}")
        
        if self.connection is not None:
            print(f"  Connection: {self.config.connection_init_mode} "
                  f"(energy={self.connection.energy():.6f})")
        print()

    
    def _validate_agents(self) -> None:
        """Check agents are compatible."""
        # All agents must have same K
        K_values = [agent.K for agent in self.agents]
        if len(set(K_values)) > 1:
            raise ValueError(f"Agents have different K values: {K_values}")
        
        # All agents must have same spatial dimension
        dims = [len(agent.support_shape) for agent in self.agents]
        if len(set(dims)) > 1:
            raise ValueError(f"Agents have different spatial dimensions: {dims}")
    
    
    # =========================================================================
    # OVERLAP COMPUTATION (CONTINUOUS FOR PROPER INTEGRATION)
    # =========================================================================
    

    
    def _compute_overlap_masks(self) -> Dict[Tuple[int, int], np.ndarray]:
        """
        Compute continuous overlap weights χ_{ij}(c) = χ_i(c) · χ_j(c).
        
        FIXED: Properly handles both SupportRegion types:
        - SupportRegion (geometry_base.py): uses .chi_weight
        - SupportRegionSmooth (masking.py): uses .mask_continuous
        
        Physical Interpretation:
        -----------------------
        For agents with smooth Gaussian boundaries:
        - χ_i(c) = 1: Agent i fully active at point c
        - χ_i(c) = 0: Agent i absent at point c  
        - χ_i(c) ∈ (0,1): Agent i at boundary, partially defined
        
        The product χ_{ij}(c) = χ_i(c) · χ_j(c) gives the proper integration
        weight: energy contributions are weak where either agent is weak.
        
        Returns:
            overlap_masks: Dict mapping (i, j) → continuous array [0,1]
                          Only stores pairs with substantial overlap
        """
        from agent.masking import SupportRegionSmooth
        from geometry.geometry_base import SupportRegion
        
        overlap_masks = {}
        threshold = getattr(self.config, 'overlap_threshold', 1e-3)
        
        for i in range(self.n_agents):
            for j in range(self.n_agents):
                if i == j:
                    continue
                
                agent_i = self.agents[i]
                agent_j = self.agents[j]
                
                # Check if supports exist
                if not hasattr(agent_i, 'support') or agent_i.support is None:
                    continue
                if not hasattr(agent_j, 'support') or agent_j.support is None:
                    continue
                
                support_i = agent_i.support
                support_j = agent_j.support
                
                # ================================================================
                # CRITICAL FIX: Get continuous mask using correct attribute
                # ================================================================
                # SupportRegion (geometry_base.py) uses .chi_weight
                # SupportRegionSmooth (masking.py) uses .mask_continuous
                
                # Get continuous mask for agent i
                if isinstance(support_i, SupportRegionSmooth):
                    chi_i = support_i.mask_continuous
                elif isinstance(support_i, SupportRegion):
                    chi_i = support_i.chi_weight
                else:
                    # Fallback: try both attributes
                    chi_i = getattr(support_i, 'mask_continuous', 
                                   getattr(support_i, 'chi_weight', None))
                    if chi_i is None:
                        print(f"Warning: Agent {i} support has no continuous mask")
                        continue
                
                # Get continuous mask for agent j
                if isinstance(support_j, SupportRegionSmooth):
                    chi_j = support_j.mask_continuous
                elif isinstance(support_j, SupportRegion):
                    chi_j = support_j.chi_weight
                else:
                    # Fallback: try both attributes
                    chi_j = getattr(support_j, 'mask_continuous',
                                   getattr(support_j, 'chi_weight', None))
                    if chi_j is None:
                        print(f"Warning: Agent {j} support has no continuous mask")
                        continue
                
                # Quick check: is there any substantial overlap?
                # Max product gives upper bound on overlap
                max_overlap = np.max(chi_i) * np.max(chi_j)
                if max_overlap < threshold:
                    continue
                
                # ============================================================
                # COMPUTE CONTINUOUS OVERLAP (INTEGRATION WEIGHT)
                # ============================================================
                # This is the CORRECT method for integration:
                # χ_{ij}(c) = χ_i(c) · χ_j(c) ∈ [0,1]
                # 
                # Energy: ∫ χ_{ij} · β_{ij} · KL dc
                # ============================================================
                chi_ij = chi_i * chi_j
                
                # Check if there's actually substantial overlap after product
                if np.max(chi_ij) < threshold:
                    continue
                
                # Store continuous overlap
                overlap_masks[(i, j)] = chi_ij.astype(np.float32)
        
        return overlap_masks
    
    
    # ============================================================================
    # Alternative: Direct computation using SupportRegion methods
    # ============================================================================
    
    def _compute_overlap_masks_v2(self) -> Dict[Tuple[int, int], np.ndarray]:
        """
        Alternative implementation using SupportRegion.compute_overlap_continuous()
        
        This version uses the built-in method from geometry_base.py, which is
        cleaner and more robust.
        """
        from geometry.geometry_base import SupportRegion
        
        overlap_masks = {}
        threshold = getattr(self.config, 'overlap_threshold', 1e-3)
        
        for i in range(self.n_agents):
            for j in range(self.n_agents):
                if i == j:
                    continue
                
                agent_i = self.agents[i]
                agent_j = self.agents[j]
                
                # Check if supports exist
                if not hasattr(agent_i, 'support') or agent_i.support is None:
                    continue
                if not hasattr(agent_j, 'support') or agent_j.support is None:
                    continue
                
                support_i = agent_i.support
                support_j = agent_j.support
                
                # Use built-in overlap computation if available
                if isinstance(support_i, SupportRegion) and isinstance(support_j, SupportRegion):
                    # Check for overlap first
                    if not support_i.has_overlap(support_j, threshold=threshold):
                        continue
                    
                    # Compute continuous overlap field
                    chi_ij = support_i.compute_overlap_continuous(support_j)
                    overlap_masks[(i, j)] = chi_ij
                else:
                    # Fallback to manual computation
                    chi_i = getattr(support_i, 'chi_weight', 
                                   getattr(support_i, 'mask_continuous', None))
                    chi_j = getattr(support_j, 'chi_weight',
                                   getattr(support_j, 'mask_continuous', None))
                    
                    if chi_i is None or chi_j is None:
                        continue
                    
                    chi_ij = chi_i * chi_j
                    if np.max(chi_ij) >= threshold:
                        overlap_masks[(i, j)] = chi_ij.astype(np.float32)
        
        return overlap_masks
    
    
    # =========================================================================
    # OVERLAP QUERY API
    # =========================================================================
    
    def get_overlap_mask(self, i: int, j: int) -> Optional[np.ndarray]:
        """
        Get continuous overlap weight χ_{ij}(c) = χ_i(c) · χ_j(c).
        
        This is the CORRECT integration weight for energy computation.
        Values in [0,1] indicate how strongly both agents exist at each point.
        
        Physical Meaning:
            χ_{ij}(c) = 1.0: Both agents fully active → full interaction
            χ_{ij}(c) = 0.5: One or both at boundary → reduced interaction
            χ_{ij}(c) = 0.0: One or both absent → no interaction
        
        Returns:
            chi_ij: Continuous float array [0,1], or None if no overlap
        
        Usage - Energy Computation:
            >>> chi_ij = system.get_overlap_mask(i, j)  # Continuous [0,1]
            >>> beta_ij = system.compute_softmax_weights(i, 'belief')[j]
            >>> energy = np.sum(chi_ij * beta_ij * kl)  # Proper weighting
        
        Usage - Gradient Computation:
            >>> chi_ij = system.get_overlap_mask(i, j)
            >>> grad_weighted = chi_ij * grad  # Smooth boundary gradients
        """
        return self.overlap_masks.get((i, j), None)
    
    
    def get_overlap_mask_boolean(
        self, 
        i: int, 
        j: int, 
        threshold: Optional[float] = None
    ) -> Optional[np.ndarray]:
        """
        Get boolean mask for computational efficiency.
        
        Thresholds continuous overlap for hard gating:
        - Use for: "Does substantive overlap exist?"
        - Use for: Skipping negligible regions
        - Use for: Hard masking if needed
        
        Most energy computations should use continuous overlap directly.
        
        Args:
            i, j: Agent indices
            threshold: Threshold for boolean conversion (default: config.overlap_threshold)
        
        Returns:
            chi_ij_boolean: Boolean mask, or None if no overlap
        
        Example:
            >>> chi_bool = system.get_overlap_mask_boolean(i, j)
            >>> if chi_bool is not None and np.any(chi_bool):
            >>>     # Compute energy only in substantive region
        """
        chi_ij = self.get_overlap_mask(i, j)
        if chi_ij is None:
            return None
        
        threshold = threshold or self.config.overlap_threshold
        return chi_ij > threshold
    
    
    def has_overlap(self, i: int, j: int) -> bool:
        """
        Quick check if agents i and j have spatial overlap.
        
        Returns:
            True if any χ_{ij}(c) > 0, False otherwise
        """
        return (i, j) in self.overlap_masks
    
    
    def get_neighbors(self, agent_idx: int) -> List[int]:
        """
        Get indices of agents that overlap with given agent.
        
        Args:
            agent_idx: Index of agent
        
        Returns:
            neighbor_indices: List of j such that χ_{ij} exists
        
        Example:
            >>> neighbors = system.get_neighbors(0)
            >>> for j in neighbors:
            >>>     chi_ij = system.get_overlap_mask(0, j)
            >>>     # Compute interaction energy...
        """
        neighbors = []
        for j in range(self.n_agents):
            if j != agent_idx and self.has_overlap(agent_idx, j):
                neighbors.append(j)
        return neighbors
    
    
    def get_overlap_fraction(self, i: int, j: int) -> float:
        """
        Compute overlap fraction: ∫χ_{ij} / ∫χ_i
        
        Uses continuous overlap for proper integration measure.
        
        Returns:
            fraction: Value in [0, 1], or 0.0 if no overlap
        
        Interpretation:
            - 0.0: No overlap
            - 0.5: Half of agent i's "mass" overlaps with j
            - 1.0: Agent i completely contained in agent j
        """
        chi_ij = self.get_overlap_mask(i, j)
        if chi_ij is None:
            return 0.0
        
        agent_i = self.agents[i]
        
        # Integrate using continuous masks
        overlap_integral = float(np.sum(chi_ij))
        self_integral = float(np.sum(agent_i.support.mask_continuous))
        
        return overlap_integral / self_integral if self_integral > 0 else 0.0
    
    
    def count_active_interactions(self) -> int:
        """
        Count number of agent pairs with spatial overlap.
        
        Returns:
            n_interactions: Number of (i,j) pairs with χ_{ij} stored
        """
        return len(self.overlap_masks)
    


    def _shared_prior_from_agents(self):
        """Compute the shared prior (μ_p, Σ_p) given config.identical_priors_source."""
        if self.config.identical_priors_source == "mean":
            # average across agents - use Sigma_p (L_p is computed on-demand)
            mu_stack = np.stack([a.mu_p for a in self.agents], axis=0)
            Sigma_stack = np.stack([a.Sigma_p for a in self.agents], axis=0)
            mu_shared = mu_stack.mean(axis=0)
            Sigma_shared = Sigma_stack.mean(axis=0)
            return mu_shared, Sigma_shared
        else:
            # 'first'
            a0 = self.agents[0]
            return a0.mu_p.copy(), a0.Sigma_p.copy()

    def _apply_identical_priors_now(self):
        """Copy a shared prior into all agents."""
        mu_shared, Sigma_shared = self._shared_prior_from_agents()
        for a in self.agents:
            a.mu_p = mu_shared.copy()
            a.Sigma_p = Sigma_shared.copy()  # L_p is computed on-demand
        # invalidate any caches that depend on priors, if applicable
        for a in self.agents:
            a.invalidate_caches() if hasattr(a, "invalidate_caches") else None
    
    # =========================================================================
    # SOFTMAX WEIGHTS (CONTINUOUS COUPLING FIELDS)
    # =========================================================================
    
    def compute_softmax_weights(
        self,
        agent_idx_i: int,
        mode: Literal['belief', 'prior'],
        kappa: Optional[float] = None
    ) -> Dict[int, np.ndarray]:
        """
        Compute softmax coupling weights β_{ij}(c) or γ_{ij}(c) for agent i.
        
        These are CONTINUOUS fields providing dynamic coupling strength:
        
        β_{ij}(c) = exp[-KL(q_i || Ω_ij[q_j])/κ] / Σ_k exp[-KL(q_i || Ω_ik[q_k])/κ]
        
        Combined with geometric overlap χ_{ij}(c) in energy computation.
        
        Args:
            agent_idx_i: Index of agent i
            mode: 'belief' for β weights, 'prior' for γ weights
            kappa: Temperature (uses config values if None)
        
        Returns:
            weights: Dict mapping neighbor j → weight field β_{ij}(c)
                    Each field has shape (*spatial,) with values in [0,1]
                    Fields sum to 1 at each point c (softmax normalization)
        
        Example:
            >>> # Get both geometric and dynamic weights
            >>> chi_ij = system.get_overlap_mask(0, 1)  # Continuous geometric
            >>> beta_fields = system.compute_softmax_weights(0, 'belief')
            >>> beta_01 = beta_fields[1]  # Continuous dynamic
            >>> 
            >>> # Energy uses both
            >>> energy = np.sum(chi_ij * beta_01 * kl)
        """
        from gradients.softmax_grads import compute_softmax_weights
        
        # Use config values if not specified
        if kappa is None:
            kappa = (self.config.kappa_beta if mode == 'belief' 
                    else self.config.kappa_gamma)
        
        return compute_softmax_weights(
            system=self,
            agent_idx_i=agent_idx_i,
            mode=mode,
            kappa=kappa
        )
    
    
    # =========================================================================
    # TRANSPORT OPERATORS
    # =========================================================================
    
    def compute_transport_ij(self, i: int, j: int) -> np.ndarray:
        """
        Compute transport operator Ω_ij = exp(φ_i) exp(-φ_j).
        
        Maps distributions from j's gauge frame to i's gauge frame.
        Optional caching for performance.
        
        Args:
            i: Receiver agent index
            j: Sender agent index
        
        Returns:
            Omega_ij: Transport operator, shape (K, K) or (*S, K, K)
        """
        # Use cached version if available
        if hasattr(self, '_compute_transport_impl'):
            return self._compute_transport_impl(self, i, j)
        
        # Fallback: direct computation (no cache)
        agent_i = self.agents[i]
        agent_j = self.agents[j]
        
        return compute_transport(
            agent_i.gauge.phi,
            agent_j.gauge.phi,
            agent_i.generators,
            validate=False
        )
    
    
    # =========================================================================
    # CONNECTION FIELD
    # =========================================================================
    
    def _initialize_connection(self) -> Optional['ConnectionField']:
        """Initialize base manifold connection field."""
        if not self.config.use_connection:
            return None
        
        from geometry.connection import (
            initialize_flat_connection,
            initialize_random_connection,
            initialize_constant_connection,
        )
        
        spatial_shape = self.agents[0].support_shape
        K = self.agents[0].K
        mode = self.config.connection_init_mode
        
        if mode == 'flat':
            return initialize_flat_connection(
                support_shape=spatial_shape, K=K, N=3
            )
        elif mode == 'random':
            return initialize_random_connection(
                support_shape=spatial_shape, K=K, N=3,
                scale=self.config.connection_scale,
                seed=getattr(self.config, 'seed', None)
            )
        elif mode == 'constant':
            if self.config.connection_const is None:
                raise ValueError("connection_const required for mode='constant'")
            return initialize_constant_connection(
                support_shape=spatial_shape, K=K,
                A_const=self.config.connection_const
            )
        else:
            raise ValueError(f"Unknown connection_init_mode: {mode}")
    
    
    # =========================================================================
    # OBSERVATION MODEL
    # =========================================================================

    
    def initialize_observation_model(self, config):
        """
        Initialize observation model from clean config.
        
        Uses config fields directly - no more getattr() hacks!
        Generates observations from SHARED ground truth.
        """
       
        
        K = self.agents[0].mu_q.shape[-1]
        spatial_shape = self.agents[0].support_shape
        
        # Get RNG from config
        rng = config.get_obs_rng()
        
        # =========================================================================
        # 1. Observation matrix W_obs
        # =========================================================================
        self.W_obs = rng.normal(
            scale=config.obs_W_scale,  # ✓ Clean config access
            size=(config.D_x, K)
        ).astype(np.float32)
        
        # =========================================================================
        # 2. Observation noise covariance R_obs
        # =========================================================================

        A = rng.normal(
            scale=config.obs_R_scale,
            size=(config.D_x, config.D_x)
        ).astype(np.float32)
        
        # Create SPD matrix with minimum noise floor
        R = A.T @ A
        R += config.obs_noise_scale**2 * np.eye(config.D_x)  # Add noise floor
        R = R.astype(np.float32)
        
        # Optional: Scale to reasonable magnitude
        R /= config.D_x  # Keep variance per dimension ~ obs_noise_scale²
        
        self.R_obs = R
        
        
        
        # =========================================================================
        # 3. Generate SHARED ground truth
        # =========================================================================
        x_true = _generate_smooth_ground_truth(
            spatial_shape, 
            K, 
            n_modes=config.obs_ground_truth_modes,  # ✓ Clean config access
            amplitude=config.obs_ground_truth_amplitude,  # ✓ Clean config access
            rng=rng
        )
        
        self.x_true = x_true
        
        
        # =========================================================================
        # 4. Generate agent-specific observations from shared x_true
        # =========================================================================
        for agent in self.agents:
            chi = agent.support.chi_weight
            
            # Get ground truth at agent's location
            if getattr(agent.base_manifold, "is_point", False) or chi.size == 1:
                coord = ()
                x_true_at_coord = x_true
            else:
                flat_idx = int(np.argmax(chi))
                coord = np.unravel_index(flat_idx, chi.shape)
                x_true_at_coord = x_true[coord]
            
            # Generate observation: o = W @ x_true + noise + agent_bias
            y_true = self.W_obs @ x_true_at_coord
            
            # Add noise
            noise = rng.normal(
                scale=config.obs_noise_scale,  # ✓ Clean config access
                size=(config.D_x,)
            ).astype(np.float32)
            
            # Add agent-specific bias (CRITICAL for symmetry breaking!)
            agent.obs_bias = rng.normal(
                scale=config.obs_bias_scale,  # ✓ Clean config access
                size=(config.D_x,)
            ).astype(np.float32)
            
            observation = (y_true + noise + agent.obs_bias).astype(np.float32)
            
            # Store observation
            agent.observations = {coord: observation}
            agent.C_obs = self.W_obs
            agent.R_obs = self.R_obs
            
            # Legacy compatibility
            if len(spatial_shape) == 0:
                agent.x_obs = observation
            else:
                x_obs_full = np.zeros((*spatial_shape, config.D_x), dtype=np.float32)
                x_obs_full[coord] = observation
                agent.x_obs = x_obs_full
        
       
    
    
    
    
 
  
    
    def ensure_observation_model(self):
        """Initialize observation model if needed (idempotent)."""
        # Check for what actually exists after initialization
        if self.W_obs is not None and self.R_obs is not None:  # ✓ Check R_obs, not Lambda_obs
            return
        
        if self.config.lambda_obs == 0:
            return
        
        if not self.agents:
            return
        
        self.initialize_observation_model(self.config)

    
    

    # =========================================================================
    # FREE ENERGY AND GRADIENTS
    # =========================================================================
    
    def compute_free_energy(self) -> Dict[str, float]:
        """
        Compute total free energy and components.
        
        Delegates to free_energy module which uses continuous overlap weights.
        """
        from gradients.free_energy_clean import compute_total_free_energy
        return compute_total_free_energy(self)
    
    
    def compute_gradients(self) -> List[Dict[str, np.ndarray]]:
        """
        Compute natural gradients for all agents.
        
        Delegates to gradient_engine module which uses continuous overlap weights.
        """
        from gradients.gradient_engine import compute_all_gradients
        return compute_all_gradients(self)
    
    
    # =========================================================================
    # OPTIMIZATION STEP
    # =========================================================================
    
    def step(self) -> Dict[str, float]:
        """Perform one optimization step for all agents."""
        gradients = self.compute_gradients()
        
        trains_phi = getattr(self.config, "trains_phi", True)
        for agent, grad in zip(self.agents, gradients):
            agent.update_beliefs(grad['delta_mu_q'], grad['delta_Sigma_q'])
            agent.update_priors(grad['delta_mu_p'], grad['delta_Sigma_p'])
            if trains_phi:
                agent.update_gauge(grad['delta_phi'])
            # system.py (in step(), after applying updates to all agents)
            if getattr(self.config, "identical_priors", "off") == "lock":
                self._apply_identical_priors_now()

        
        return self.compute_free_energy()
    
    
    # =========================================================================
    # UTILITIES
    # =========================================================================
    
    def count_total_parameters(self) -> int:
        """Count total trainable parameters across all agents."""
        return sum(agent.count_parameters() for agent in self.agents)
    
    
    def summary(self) -> str:
        """Get system summary string."""
        energies = self.compute_free_energy()
        
        lines = [
            "Multi-Agent System:",
            f"  Agents: {self.n_agents}",
            f"  Total parameters: {self.count_total_parameters():,}",
            f"  Active overlaps: {len(self.overlap_masks)}",
            "",
            "Free Energy:",
            f"  Self: {energies['self']:.4f}",
            f"  Observations: {energies['observations']:.4f}",
            f"  Belief align: {energies['belief_align']:.4f}",
            f"  Prior align: {energies['prior_align']:.4f}",
            f"  Total: {energies['total']:.4f}",
        ]
        
        if self.transport_cache is not None:
            stats = self.transport_cache.stats()
            lines.extend([
                "",
                "Transport Cache:",
                f"  Size: {stats['size']}",
                f"  Hit rate: {stats['hit_rate']:.1%}",
            ])
        
        return "\n".join(lines)
    
    
def _generate_smooth_ground_truth(
    spatial_shape, K, n_modes=3, amplitude=1.0, rng=None
):
    """
    Generate smooth ground truth using sum of sinusoids.
    
    This creates the shared "reality" that all agents observe.
    """
    if rng is None:
        rng = np.random.default_rng()
    
    ndim = len(spatial_shape)
    
    # 0D case (point manifold)
    if ndim == 0:
        return rng.normal(size=(K,)).astype(np.float32)
    
    # Spatial case
    coords = np.meshgrid(
        *[np.linspace(0, 2*np.pi, n) for n in spatial_shape], 
        indexing='ij'
    )
    
    x_true = np.zeros((*spatial_shape, K), dtype=np.float32)
    
    for k in range(K):
        field_k = np.zeros(spatial_shape, dtype=np.float32)
        
        # Sum of sinusoids
        for _ in range(n_modes):
            freqs = rng.integers(1, 4, size=ndim)
            phases = rng.uniform(0, 2*np.pi, size=ndim)
            
            wave = np.ones(spatial_shape)
            for d in range(ndim):
                wave *= np.sin(freqs[d] * coords[d] + phases[d])
            
            field_k += wave
        
        # Normalize
        field_k = amplitude * field_k / (np.std(field_k) + 1e-8)
        x_true[..., k] = field_k
    
    return x_true
