# -*- coding: utf-8 -*-
"""
Meta-Agent Emergence and Hierarchical Structure
================================================

Creates meta-agents from consensus clusters through renormalization.

Theory:
-------
At each base point c ∈ C, agents at scale ζ can condense into meta-agents
at scale ζ+1 through renormalization:

    μ_M(c) = (1/n) Σᵢ Ωᵢⱼ[μᵢ(c)]     (gauge-transported average)
    Σ_M(c) = (1/n) Σᵢ Ωᵢⱼ[Σᵢ(c)]Ωᵢⱼᵀ
    φ_M(c) = average_SO3({φᵢ(c)})    (Fréchet mean on SO(3))

Multiple condensations can occur iteratively:
    ζ=0: 3000 agents → [partition] → ζ=1: 50 meta-agents
    ζ=1: 50 meta-agents → [partition] → ζ=2: 6 meta-agents
    etc.

No artificial timescale separation - dynamics emerge naturally!

Author: Chris & Christine
Date: November 2025
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum
from collections import defaultdict

# Import from existing codebase
from agent.agents import Agent
from geometry.geometry_base import BaseManifold, SupportRegion, create_full_support
from math_utils.transport import compute_transport
from math_utils.generators import generate_so3_generators
from math_utils.so3_frechet import average_gauge_frames_so3, so3_exp, so3_log
from config import AgentConfig
from agent.masking import MaskConfig

# Conditional import for spatial emergence (avoid circular import)
# Actual import done in methods that need it


# =============================================================================
# Scale Indexing
# =============================================================================

@dataclass
class ScaleIndex:
    """
    Index for agents at a specific (scale, position) in the hierarchy.
    
    At each base point c ∈ C, we can have multiple agents at each scale ζ.
    """
    scale: int          # ζ ∈ {0, 1, 2, ...}
    local_index: int    # Index within this scale
    
    def __repr__(self):
        return f"ζ{self.scale}[{self.local_index}]"


class ScaleLevel(Enum):
    """Hierarchical scale levels (for convenience)."""
    BASE = 0       # Individual agents (ζ=0)
    GROUP = 1      # First-order meta-agents (ζ=1)
    COMMUNITY = 2  # Second-order meta-agents (ζ=2)
    SOCIETY = 3    # Third-order meta-agents (ζ=3)
    
    def __int__(self):
        return self.value


@dataclass
class MetaAgentDescriptor:
    """
    Metadata for a meta-agent.
    
    Tracks hierarchical structure and emergence history.
    """
    scale_index: ScaleIndex           # Unique scale + local index
    constituent_indices: List[ScaleIndex]  # Lower-scale constituents
    emergence_time: int                    # Simulation step when formed
    
    # Coherence metrics
    belief_coherence: float           # Mean coherence: avg(C̄ᵢ)
    model_coherence: float            # Model coherence
    
    # Leadership structure
    leader_index: Optional[int] = None      # Index of dominant constituent
    leader_score: Optional[float] = None    # Leadership score L = χ² · C̄
    leadership_distribution: Optional[np.ndarray] = None  # L_i for all constituents
    
    def __repr__(self):
        leader_str = f", leader={self.leader_index}" if self.leader_index is not None else ""
        return (f"MetaAgent({self.scale_index}, "
                f"n_constituents={len(self.constituent_indices)}{leader_str})")


# =============================================================================
# Hierarchical Agent
# =============================================================================

class HierarchicalAgent(Agent):
    """
    Extended agent with hierarchical awareness.

    Can be either:
    - Base agent (scale ζ=0)
    - Meta-agent (scale ζ≥1) with renormalized fields

    Fields (μ_q, Σ_q, μ_p, Σ_p, φ) are smooth sections over support region.
    """

    def __init__(self,
                 scale: int,
                 local_index: int,
                 constituent_indices: Optional[List[ScaleIndex]] = None,
                 meta_descriptor: Optional[MetaAgentDescriptor] = None,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Scale identification
        self.scale = scale
        self.local_index = local_index
        self.scale_index = ScaleIndex(scale, local_index)

        # Hierarchical structure
        self.constituent_indices = constituent_indices or []
        self.is_meta = len(self.constituent_indices) > 0
        self.meta = meta_descriptor

        # Parent/child relationships for cross-scale dynamics
        self.parent_meta = None  # Meta-agent this agent belongs to (if any)
        self.constituents = []   # Constituent agents (if this is a meta-agent)

        # Ouroboros: Multi-scale hyperprior tower (non-Markovian memory)
        # hyperpriors[k] = prior from k+2 levels up (k=0 → grandparent, k=1 → great-grandparent, ...)
        # Standard: only parent_meta (Markov)
        # Tower: parent_meta + hyperpriors from all ancestors (non-Markov)
        self.hyperprior_mus = []     # List[μ_h^(k)] - hyperprior means from ancestors
        self.hyperprior_Sigmas = []  # List[Σ_h^(k)] - hyperprior covariances from ancestors

        # Activity status (can be deactivated when absorbed into meta-agent)
        self.is_active = True

        # Timescale separation (τ_ζ = 10^ζ bits)
        self.info_accumulator = 0.0  # Accumulated information change ΔI
        self.timescale_threshold = 10.0 ** scale  # Update threshold (bits)

    def update_prior_from_parent(self, enable_tower=False, max_depth=1, decay=0.3):
        """
        Top-down: Update prior from parent meta-agent's belief.

        Standard (Markov):
            p_i^(ζ) ← q_M^(ζ+1)

        Ouroboros Tower (Non-Markov):
            p_i^(ζ)     ← q_M^(ζ+1)      (parent - immediate prior)
            h_i^(ζ,0)   ← q_M^(ζ+2)      (grandparent - 1st hyperprior)
            h_i^(ζ,1)   ← q_M^(ζ+3)      (great-grandparent - 2nd hyperprior)
            ...

        This creates the downward flow of information: meta-agent beliefs
        become constituent priors. With tower enabled, information flows
        from ALL ancestral scales, not just immediate parent.

        Args:
            enable_tower: If True, collect hyperpriors from all ancestors
            max_depth: How many levels up to collect (1=standard, 2+=tower)
            decay: Exponential decay for hyperprior influence
        """
        if self.parent_meta is None or not self.parent_meta.is_active:
            return  # No parent or parent inactive

        # Standard: Prior from parent (ζ+1)
        omega = compute_transport(
            self.gauge.phi,
            self.parent_meta.gauge.phi,
            self.generators,
            validate=False
        )

        # Set prior = transported parent belief (handle spatial manifolds)
        if omega.ndim > 2:
            # Spatial manifold: use einsum
            mu_p_new = np.einsum('...ij,...j->...i', omega, self.parent_meta.mu_q)
            Sigma_p_new = np.einsum('...ik,...kl,...jl->...ij', omega, self.parent_meta.Sigma_q, omega)

            # CRITICAL: Only update prior where parent has support
            # Outside parent's support, renormalized values are defaults (0, I)
            # which causes huge KL if constituent's belief differs there
            parent_chi = self.parent_meta.support.chi_weight
            support_threshold = 0.01

            # Blend: use new prior where parent has support, keep old elsewhere
            blend_mask = parent_chi > support_threshold
            blend_weight = np.where(blend_mask, parent_chi, 0.0)[..., None]  # (*spatial, 1)

            # Smooth blending to avoid discontinuities
            self.mu_p = blend_weight * mu_p_new + (1 - blend_weight) * self.mu_p

            # For Sigma: expand blend_weight to matrix shape
            blend_weight_mat = blend_weight[..., None]  # (*spatial, 1, 1)
            self.Sigma_p = blend_weight_mat * Sigma_p_new + (1 - blend_weight_mat) * self.Sigma_p
        else:
            # Point manifold - no blending needed
            self.mu_p = omega @ self.parent_meta.mu_q
            self.Sigma_p = omega @ self.parent_meta.Sigma_q @ omega.T

        # Regularize to ensure SPD after transport
        # For spatial manifolds: transpose only last two axes
        if self.Sigma_p.ndim > 2:
            Sigma_p_T = np.swapaxes(self.Sigma_p, -1, -2)
        else:
            Sigma_p_T = self.Sigma_p.T

        self.Sigma_p = 0.5 * (self.Sigma_p + Sigma_p_T)  # Symmetrize

        # Regularize (broadcast identity for spatial manifolds)
        if self.Sigma_p.ndim > 2:
            I = np.eye(self.K)
            self.Sigma_p = self.Sigma_p + 1e-6 * I
        else:
            self.Sigma_p += 1e-6 * np.eye(self.K)

        # Ouroboros Tower: Collect hyperpriors from ancestors (ζ+2, ζ+3, ...)
        if enable_tower and max_depth > 1:
            self.hyperprior_mus = []
            self.hyperprior_Sigmas = []

            # Climb the tower: parent → grandparent → great-grandparent → ...
            current_ancestor = self.parent_meta
            for depth in range(1, max_depth):
                # Get next level up (grandparent, great-grandparent, ...)
                if current_ancestor is None or current_ancestor.parent_meta is None:
                    break  # Reached top of hierarchy

                current_ancestor = current_ancestor.parent_meta
                if not current_ancestor.is_active:
                    break  # Ancestor inactive

                # Transport ancestor's belief to this agent's frame
                omega_ancestor = compute_transport(
                    self.gauge.phi,
                    current_ancestor.gauge.phi,
                    self.generators,
                    validate=False
                )

                # Store hyperprior (handle both point and spatial manifolds)
                if omega_ancestor.ndim > 2:
                    # Spatial manifold: use einsum
                    mu_h = np.einsum('...ij,...j->...i', omega_ancestor, current_ancestor.mu_q)
                    Sigma_h = np.einsum('...ik,...kl,...jl->...ij', omega_ancestor, current_ancestor.Sigma_q, omega_ancestor)
                else:
                    # Point manifold
                    mu_h = omega_ancestor @ current_ancestor.mu_q
                    Sigma_h = omega_ancestor @ current_ancestor.Sigma_q @ omega_ancestor.T

                # Regularize
                # For spatial manifolds: transpose only last two axes
                if Sigma_h.ndim > 2:
                    Sigma_h_T = np.swapaxes(Sigma_h, -1, -2)
                else:
                    Sigma_h_T = Sigma_h.T

                Sigma_h = 0.5 * (Sigma_h + Sigma_h_T)

                # Regularize (broadcast identity for spatial manifolds)
                if Sigma_h.ndim > 2:
                    I = np.eye(self.K)
                    Sigma_h = Sigma_h + 1e-6 * I
                else:
                    Sigma_h += 1e-6 * np.eye(self.K)

                self.hyperprior_mus.append(mu_h)
                self.hyperprior_Sigmas.append(Sigma_h)

    def update_prior_from_global_state(self, system):
        """
        Self-referential closure: Top scale observes entire system.

        Wheeler's "it from bit" - the system observes itself observing itself.

        For agents at the top of the hierarchy (no parent), the prior
        comes from observing the ENTIRE system state - creating a
        strange loop where the system bootstraps its own reference frame.

        p_top ← statistics(ALL active agents at all scales)

        This creates a self-consistent fixed point:
        - Top observes collective → forms beliefs
        - Beliefs flow down as priors
        - Lower scales evolve under these priors
        - Evolution changes collective state
        - Top re-observes changed state
        - Loop continues → self-organizing!

        Args:
            system: MultiScaleSystem containing all agents
        """
        

        if self.parent_meta is not None:
            return  # Only for top-scale agents

        all_agents = system.get_all_active_agents()

        if len(all_agents) <= 1:
            return  # Need multiple agents to observe

        # Compute coherence-weighted average of all beliefs
        coherence_scores = []
        transported_beliefs = []

        for agent in all_agents:
            # Transport agent's belief to this (top) frame
            omega = compute_transport(
                self.gauge.phi,
                agent.gauge.phi,
                self.generators,
                validate=False
            )

            # Transport beliefs (handle both point and spatial manifolds)
            if omega.ndim > 2:
                # Spatial manifold: use einsum
                mu_transported = np.einsum('...ij,...j->...i', omega, agent.mu_q)
                Sigma_transported = np.einsum('...ik,...kl,...jl->...ij', omega, agent.Sigma_q, omega)
            else:
                # Point manifold
                mu_transported = omega @ agent.mu_q
                Sigma_transported = omega @ agent.Sigma_q @ omega.T

            # Regularize to ensure SPD after transport
            # For spatial manifolds: transpose only last two axes, not all
            if Sigma_transported.ndim > 2:
                Sigma_T = np.swapaxes(Sigma_transported, -1, -2)
            else:
                Sigma_T = Sigma_transported.T

            Sigma_transported = 0.5 * (Sigma_transported + Sigma_T)  # Symmetrize

            # Regularize (broadcast identity for spatial manifolds)
            if Sigma_transported.ndim > 2:
                # Spatial: add identity at each point
                I = np.eye(self.K)
                Sigma_transported = Sigma_transported + 1e-6 * I
            else:
                # Point manifold
                Sigma_transported += 1e-6 * np.eye(self.K)

            transported_beliefs.append((mu_transported, Sigma_transported))

            # Coherence = exp(-average KL to all others)
            # Agents coherent with collective get higher weight
            kl_sum = 0.0
            count = 0
            for other_agent in all_agents:
                if other_agent.agent_id == agent.agent_id:
                    continue

                omega_other = compute_transport(
                    self.gauge.phi,
                    other_agent.gauge.phi,
                    self.generators,
                    validate=False
                )

                # Transport other agent's beliefs
                if omega_other.ndim > 2:
                    # Spatial manifold: use einsum
                    mu_other = np.einsum('...ij,...j->...i', omega_other, other_agent.mu_q)
                    Sigma_other = np.einsum('...ik,...kl,...jl->...ij', omega_other, other_agent.Sigma_q, omega_other)
                else:
                    # Point manifold
                    mu_other = omega_other @ other_agent.mu_q
                    Sigma_other = omega_other @ other_agent.Sigma_q @ omega_other.T

                # Regularize to ensure SPD after transport
                # For spatial manifolds: transpose only last two axes
                if Sigma_other.ndim > 2:
                    Sigma_other_T = np.swapaxes(Sigma_other, -1, -2)
                else:
                    Sigma_other_T = Sigma_other.T

                Sigma_other = 0.5 * (Sigma_other + Sigma_other_T)  # Symmetrize

                # Regularize (broadcast identity for spatial manifolds)
                if Sigma_other.ndim > 2:
                    I = np.eye(self.K)
                    Sigma_other = Sigma_other + 1e-6 * I
                else:
                    Sigma_other += 1e-6 * np.eye(self.K)

                from math_utils.numerical_utils import kl_gaussian
                try:
                    kl = kl_gaussian(mu_transported, Sigma_transported,
                                   mu_other, Sigma_other)
                    # For spatial manifolds: kl is (*spatial,) array
                    # Aggregate to scalar for coherence score
                    if np.ndim(kl) > 0:
                        kl = float(np.mean(kl))  # Average KL over spatial points
                except np.linalg.LinAlgError:
                    # If still fails, skip this comparison
                    kl = 10.0  # Large default KL (low coherence)
                kl_sum += kl
                count += 1

            # Coherence score
            avg_kl = kl_sum / max(count, 1)
            coherence = np.exp(-avg_kl)
            coherence_scores.append(coherence)

        # Normalize weights
        coherence_scores = np.array(coherence_scores)
        weights = coherence_scores / (np.sum(coherence_scores) + 1e-10)

        # Compute weighted average → becomes top agent's prior
        # Use temporary variables to accumulate before setting (avoid triggering setter)
        mu_p_new = np.zeros_like(self.mu_q)
        Sigma_p_new = np.zeros_like(self.Sigma_q)

       
        for idx, ((mu, Sigma), w) in enumerate(zip(transported_beliefs, weights)):
            w_shape = getattr(w, 'shape', 'scalar')
            w_val = repr(w) if w_shape == 'scalar' else f"array shape={w.shape}"
            
            try:
                mu_p_new += w * mu
            except Exception as e:
                print(f"    ERROR in accumulation: {e}")
                print(f"    mu_p_new.shape={mu_p_new.shape}, w type={type(w).__name__}, mu type={type(mu).__name__}")
                raise
            Sigma_p_new += w * Sigma

        # Regularize to ensure SPD
        # For spatial manifolds: transpose only last two axes
        if Sigma_p_new.ndim > 2:
            Sigma_p_new_T = np.swapaxes(Sigma_p_new, -1, -2)
        else:
            Sigma_p_new_T = Sigma_p_new.T

        Sigma_p_new = 0.5 * (Sigma_p_new + Sigma_p_new_T)  # Symmetrize

        # Regularize (broadcast identity for spatial manifolds)
        if Sigma_p_new.ndim > 2:
            I = np.eye(self.K)
            Sigma_p_new = Sigma_p_new + 1e-6 * I
        else:
            Sigma_p_new += 1e-6 * np.eye(self.K)

        # Set once at the end (triggers Cholesky decomposition once)
        self.mu_p = mu_p_new
        self.Sigma_p = Sigma_p_new

    def generate_observations_from_constituents(self) -> Optional[np.ndarray]:
        """
        Bottom-up: Generate observations from constituent beliefs.

        o_M ← weighted_average({q_i : i ∈ constituents})

        This creates the upward flow: constituent statistics become
        meta-agent observations.

        Returns:
            Observation vector o_M, or None if not a meta-agent
        """
        if not self.is_meta or len(self.constituents) == 0:
            return None

        # Compute coherence-weighted average of constituent beliefs
        coherence_scores = self._compute_constituent_coherence()
        weights = coherence_scores / np.sum(coherence_scores)

        # Weighted average in this agent's frame
        o_meta = np.zeros_like(self.mu_q)
        for agent, w in zip(self.constituents, weights):
            if not agent.is_active:
                continue

            # Transport constituent belief to meta-agent frame
            omega = compute_transport(
                self.gauge.phi,
                agent.gauge.phi,
                self.generators,
                validate=False
            )

            o_meta += w * (omega @ agent.mu_q)

        return o_meta

    def _compute_constituent_coherence(self) -> np.ndarray:
        """Compute coherence scores for constituents."""
        if not self.is_meta:
            return np.array([])

        n = len(self.constituents)
        coherence_scores = np.ones(n)

        # Use cached coherence if available
        if self.meta and hasattr(self.meta, 'leadership_distribution'):
            # Normalize leadership scores as coherence proxy
            coherence_scores = self.meta.leadership_distribution
            coherence_scores = coherence_scores / np.sum(coherence_scores)

        return coherence_scores

    def should_update(self, delta_info: float) -> bool:
        """
        Check if agent should update based on information accumulation.

        Agents at scale ζ only update when accumulated ΔI ≥ 10^ζ bits.
        This creates emergent timescale separation.

        Args:
            delta_info: New information change (in bits)

        Returns:
            True if accumulated info exceeds threshold
        """
        self.info_accumulator += delta_info

        if self.info_accumulator >= self.timescale_threshold:
            self.info_accumulator = 0.0  # Reset accumulator
            return True

        return False

    def __repr__(self):
        status = "active" if self.is_active else "inactive"
        meta_str = f", meta" if self.is_meta else ""
        parent_str = f", parent={self.parent_meta.scale_index}" if self.parent_meta else ""
        return f"HAgent({self.scale_index}, {status}{meta_str}{parent_str})"


# =============================================================================
# Multi-Scale System
# =============================================================================

class MultiScaleSystem:
    """
    System with agents at multiple scales at each base point.
    
    Structure:
    - agents[ζ] = list of agents at scale ζ
    - At each c ∈ C, multiple agents can coexist at each scale
    
    Supports iterative condensation:
        Scale 0 → partition → Scale 1
        Scale 1 → partition → Scale 2
        etc.
    """
    
    def __init__(self, base_manifold: BaseManifold,
                 max_emergence_levels: Optional[int] = None,
                 max_meta_membership: Optional[int] = None,
                 max_total_agents: Optional[int] = None,
                 use_pointwise_emergence: bool = True,
                 min_consensus_volume: float = 0.3,
                 min_consensus_region_size: int = 4):
        self.base_manifold = base_manifold

        # agents[scale] = list of agents at that scale
        self.agents: Dict[int, List[HierarchicalAgent]] = defaultdict(list)

        # Track condensation history
        self.condensation_events = []
        self.current_time = 0

        # Hard cap on emergence levels to prevent performance degradation
        # None = unlimited (not recommended), otherwise max scale allowed
        # e.g., max_emergence_levels=3 allows scales 0, 1, 2, 3
        self.max_emergence_levels = max_emergence_levels

        # Cap on meta-agent membership to control exponential growth
        # None = unlimited, otherwise max constituents per meta-agent
        # e.g., max_meta_membership=10 limits each meta-agent to 10 constituents
        self.max_meta_membership = max_meta_membership

        # Hard cap on total agents across ALL scales
        # None = unlimited, otherwise max total agents in system
        # e.g., max_total_agents=1000 prevents system from exceeding 1000 agents total
        self.max_total_agents = max_total_agents

        # Pointwise emergence for spatial manifolds
        # If True: meta-agents form in connected regions where consensus holds locally
        # If False: require global consensus across entire manifold (stricter)
        self.use_pointwise_emergence = use_pointwise_emergence

        # Minimum consensus volume: meta-agent forms if consensus region covers
        # at least this fraction of the total manifold (e.g., 0.3 = 30%)
        self.min_consensus_volume = min_consensus_volume

        # Minimum connected region size: each consensus region must contain at least
        # this many spatial points to form a meta-agent (prevents tiny isolated regions)
        self.min_consensus_region_size = min_consensus_region_size
    
    def add_base_agent(self, agent_config: AgentConfig, agent_id: str = None) -> HierarchicalAgent:
        """
        Add a new scale-0 base agent.
        
        Args:
            agent_config: Configuration for the agent
            agent_id: Optional identifier
        
        Returns:
            Created HierarchicalAgent at scale 0
        """
        local_index = len(self.agents[0])

        if agent_id is None:
            agent_id = f"base_{local_index}"

        agent = HierarchicalAgent(
            scale=0,
            local_index=local_index,
            agent_id=agent_id,
            config=agent_config,
            rng=np.random.default_rng(local_index),
            base_manifold=self.base_manifold
        )

        # Initialize fields (same as Agent class would do)
        agent._initialize_belief_covariance()
        agent._initialize_prior_covariance()
        agent._initialize_gauge()

        self.agents[0].append(agent)
        return agent
    
    def form_meta_agents_at_scale(self,
                                  source_scale: int,
                                  partitions: List[List[int]],
                                  deactivate_constituents: bool = False) -> List[HierarchicalAgent]:
        """
        Form meta-agents at scale ζ+1 from partitions at scale ζ.

        Args:
            source_scale: Scale ζ of constituents
            partitions: List of constituent clusters (local indices at source_scale)
                       Each cluster becomes one meta-agent at scale ζ+1
            deactivate_constituents: Whether to freeze constituents (default False)
                False (default): Continuous flow - constituents keep evolving
                                 Like atoms in proteins - lower scales don't freeze!
                                 Meta-agent tracks renormalized statistics
                True: Categorical - constituents frozen (computational efficiency)

        Returns:
            List of newly formed meta-agents

        Physical Intuition:
            With continuous flow (default), emergence is like phase transitions in
            physics - water molecules don't stop moving when ice forms, they just
            organize into a crystal lattice. The molecules (constituents) keep
            evolving, while the lattice (meta-agent) tracks the renormalized order.

        Example:
            # Condense scale-0 agents [0,1,2] and [3,4] into two scale-1 meta-agents
            # Constituents remain active and evolving
            system.form_meta_agents_at_scale(
                source_scale=0,
                partitions=[[0, 1, 2], [3, 4]],
                deactivate_constituents=False  # Default: continuous flow
            )
        """
        target_scale = source_scale + 1
        source_agents = self.agents[source_scale]

        if not source_agents:
            raise ValueError(f"No agents at source scale {source_scale}")

        # Check emergence level cap
        if self.max_emergence_levels is not None and target_scale > self.max_emergence_levels:
            print(f"[Level Cap] Cannot form meta-agents at scale {target_scale} "
                  f"(max allowed: {self.max_emergence_levels}). Skipping condensation.")
            return []

        # Check total agent cap BEFORE creating new meta-agents
        if self.max_total_agents is not None:
            current_total = sum(len(agents) for agents in self.agents.values())
            n_new_meta = len([p for p in partitions if len(p) >= 2])  # Count valid partitions
            if current_total + n_new_meta > self.max_total_agents:
                print(f"[Agent Cap] Cannot form {n_new_meta} meta-agents: would exceed max_total_agents "
                      f"({current_total + n_new_meta} > {self.max_total_agents}). Skipping condensation.")
                return []

        new_meta_agents = []

        for partition in partitions:
            if len(partition) < 2:
                print(f"[Warning] Skipping singleton cluster: {partition}")
                continue

            # Check meta-agent membership cap
            if self.max_meta_membership is not None and len(partition) > self.max_meta_membership:
                # Trim partition to max size (keep most coherent members)
                # For now, just take first N members (could be improved to select by coherence)
                original_size = len(partition)
                partition = partition[:self.max_meta_membership]
                print(f"[Membership Cap] Trimmed partition from {original_size} to {self.max_meta_membership} constituents")

            # Get constituent agents
            constituents = [source_agents[i] for i in partition]

            # Verify all constituents are active
            inactive = [c for c in constituents if not c.is_active]
            if inactive:
                print(f"[Warning] Partition contains inactive agents: {inactive}")

            # Compute coherence scores ONCE (used for all renormalization)
            coherence_scores = self._compute_coherence_scores(constituents, field_type='belief')

            # Identify leader (agent with max χ² · C̄)
            leader_idx, leader_score, leadership_dist = self._identify_leader(constituents, coherence_scores)

            # Compute renormalized fields via coherence-weighted gauge transport
            mu_q, Sigma_q = self._renormalize_beliefs(constituents, coherence_scores)
            mu_p, Sigma_p = self._renormalize_models(constituents, coherence_scores)
            phi = self._average_gauge_frames(constituents, coherence_scores)

            # Create meta-agent configuration
            meta_index = len(self.agents[target_scale])
            meta_id = f"meta_{target_scale}_{meta_index}"

            constituent_scale_indices = [
                ScaleIndex(source_scale, i) for i in partition
            ]

            # Compute coherence metrics from scores
            belief_coherence = np.mean(coherence_scores)

            # Also compute model coherence separately
            model_coherence_scores = self._compute_coherence_scores(constituents, field_type='model')
            model_coherence = np.mean(model_coherence_scores)

            meta_descriptor = MetaAgentDescriptor(
                scale_index=ScaleIndex(target_scale, meta_index),
                constituent_indices=constituent_scale_indices,
                emergence_time=self.current_time,
                belief_coherence=belief_coherence,
                model_coherence=model_coherence,
                leader_index=leader_idx,
                leader_score=leader_score,
                leadership_distribution=leadership_dist
            )

            # Create meta-agent
            meta_agent = HierarchicalAgent(
                scale=target_scale,
                local_index=meta_index,
                constituent_indices=constituent_scale_indices,
                meta_descriptor=meta_descriptor,
                agent_id=meta_id,
                config=constituents[0].config,
                rng=np.random.default_rng(hash(meta_id) % 2**32),
                base_manifold=self.base_manifold
            )

            # Initialize field structures
            meta_agent.support = self._compute_meta_support(constituents, coherence_scores)
            meta_agent.generators = constituents[0].generators

            meta_agent._initialize_belief_covariance()
            meta_agent._initialize_prior_covariance()
            meta_agent._initialize_gauge()

            

            meta_agent.mu_q = mu_q
            

            meta_agent.Sigma_q = Sigma_q  # Triggers L_q recomputation via setter
            meta_agent.mu_p = mu_p
            meta_agent.Sigma_p = Sigma_p  # Triggers L_p recomputation via setter
            meta_agent.gauge.phi = phi

            # ========== NEW: Set up parent-child relationships ==========
            meta_agent.constituents = constituents.copy()
            for agent in constituents:
                agent.parent_meta = meta_agent

            # Add to system
            self.agents[target_scale].append(meta_agent)
            new_meta_agents.append(meta_agent)

            # Deactivate constituents if requested
            if deactivate_constituents:
                for agent in constituents:
                    agent.is_active = False

            # Record condensation event
            self.condensation_events.append({
                'time': self.current_time,
                'source_scale': source_scale,
                'target_scale': target_scale,
                'n_constituents': len(partition),
                'constituent_indices': constituent_scale_indices,
                'leader_index': leader_idx,
                'leader_score': leader_score,
                'coherence': {
                    'belief': belief_coherence,
                    'model': model_coherence
                }
            })

        print(f"[Condensation ζ={source_scale}→{target_scale}] "
              f"{len(partitions)} clusters → {len(new_meta_agents)} meta-agents")

        # Print leader info for each meta-agent
        for meta in new_meta_agents:
            leader_constituent_idx = meta.meta.constituent_indices[meta.meta.leader_index]
            print(f"  {meta.scale_index}: leader={leader_constituent_idx} "
                  f"(L={meta.meta.leader_score:.3f})")

        return new_meta_agents

    def re_renormalize_meta_agents(self) -> int:
        """
        Re-compute meta-agent fields from current constituent states.

        CRITICAL for proper hierarchical dynamics:
        Meta-agents do NOT receive direct gradient updates.
        Instead, their state is always derived from renormalization
        of their constituents (bottom-up information flow).

        This should be called after base agents update, so meta-agents
        reflect the evolved constituent beliefs.

        Returns:
            Number of meta-agents updated
        """
        n_updated = 0

        # Iterate over all scales above 0 (meta-agents)
        for scale in sorted(self.agents.keys()):
            if scale == 0:
                continue  # Base agents don't get re-renormalized

            for meta_agent in self.agents[scale]:
                if not meta_agent.is_active:
                    continue

                constituents = meta_agent.constituents
                if len(constituents) == 0:
                    continue

                # Re-compute coherence scores
                coherence_scores = self._compute_coherence_scores(
                    constituents, field_type='belief'
                )

                # Re-renormalize belief fields
                mu_q, Sigma_q = self._renormalize_beliefs(
                    constituents, coherence_scores
                )
                meta_agent.mu_q = mu_q.astype(np.float32)
                meta_agent.Sigma_q = Sigma_q.astype(np.float32)

                # Re-renormalize model/prior fields
                model_coherence = self._compute_coherence_scores(
                    constituents, field_type='model'
                )
                mu_p, Sigma_p = self._renormalize_models(
                    constituents, model_coherence
                )
                meta_agent.mu_p = mu_p.astype(np.float32)
                meta_agent.Sigma_p = Sigma_p.astype(np.float32)

                # Re-average gauge frames
                phi = self._average_gauge_frames(constituents, coherence_scores)
                meta_agent.gauge.phi = phi.astype(np.float32)

                n_updated += 1

        return n_updated

    # =========================================================================
    # Renormalization Methods
    # =========================================================================

    def _renormalize_beliefs(self,
                           constituents: List[HierarchicalAgent],
                           coherence_scores: Optional[np.ndarray] = None,
                           region_mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute renormalized belief fields via coherence-weighted gauge transport.

        Uses Presence × Coherence weighting:
            wᵢ(c) ∝ χᵢ(c) · C̄ᵢ
            μ_M(c) = Σᵢ wᵢ(c) · Ωᵢⱼ[μᵢ(c)]

        For 0D: χᵢ = 1, so weights are purely coherence-based
        For spatial: weights vary with location based on presence

        Args:
            constituents: List of constituent agents
            coherence_scores: Optional precomputed coherence scores
            region_mask: Optional (*spatial,) boolean mask. If provided, only
                        compute fields within the True region (efficiency optimization
                        for spatial emergence where meta-agent support is restricted)

        Returns:
            (mu_M, Sigma_M): Renormalized belief parameters
        """
        ref = constituents[0]
        K = ref.K

        # Compute coherence scores if not provided
        if coherence_scores is None:
            coherence_scores = self._compute_coherence_scores(constituents)

        if ref.base_manifold.is_point:
            # === 0D CASE (Transformers) ===
            # All χᵢ = 1, weight by coherence only

            mu_weighted = np.zeros(K)
            Sigma_weighted = np.zeros((K, K))
            total_weight = 0.0

            for i, agent in enumerate(constituents):
                C_i = coherence_scores[i]

                # Transport to reference frame
                omega = compute_transport(
                    ref.gauge.phi,
                    agent.gauge.phi,
                    ref.generators,
                    validate=False
                )

                # Accumulate weighted transported statistics
                mu_weighted += C_i * (omega @ agent.mu_q)
                Sigma_weighted += C_i * (omega @ agent.Sigma_q @ omega.T)
                total_weight += C_i

            # Normalize
            mu_M = mu_weighted / total_weight
            Sigma_M = Sigma_weighted / total_weight

            return mu_M, Sigma_M

        else:
            # === SPATIAL CASE ===
            # Weight by presence × coherence at each location

            spatial_shape = ref.base_manifold.shape
            mu_M = np.zeros((*spatial_shape, K))
            # Initialize Sigma_M to DIFFUSE prior outside support
            # Large variance = high uncertainty = "don't care about this region"
            # This prevents huge KL when constituent beliefs differ from default
            diffuse_variance = 1000.0
            Sigma_M = np.broadcast_to(diffuse_variance * np.eye(K), (*spatial_shape, K, K)).copy()

            # Precompute transports to reference frame
            transports = []
            for agent in constituents:
                omega = compute_transport(
                    ref.gauge.phi,
                    agent.gauge.phi,
                    ref.generators,
                    validate=False
                )
                transports.append(omega)

            # Determine which points to compute
            # If region_mask provided, only compute within that region (HUGE efficiency gain)
            if region_mask is not None:
                # Get indices where mask is True
                active_indices = np.argwhere(region_mask)
                n_active = len(active_indices)
                n_total = np.prod(spatial_shape)
                # Convert to tuples for indexing
                points_to_compute = [tuple(idx) for idx in active_indices]
            else:
                # Compute everywhere (original behavior)
                points_to_compute = list(np.ndindex(spatial_shape))

            # At each spatial location in the region
            for c_idx in points_to_compute:
                mu_weighted = np.zeros(K)
                Sigma_weighted = np.zeros((K, K))
                total_weight = 0.0

                for i, agent in enumerate(constituents):
                    # Presence at this location
                    chi_i = agent.support.chi_weight[c_idx]

                    # Skip if negligible presence
                    if chi_i < 1e-6:
                        continue

                    # Combined weight: presence × coherence
                    w_i = chi_i * coherence_scores[i]

                    # Transport and accumulate
                    # For spatial manifolds: omega has shape (*spatial, K, K)
                    omega = transports[i]

                    # Extract transport operator at this spatial location
                    if omega.ndim > 2:
                        omega_c = omega[c_idx]  # (K, K) at point c_idx
                    else:
                        # Should not happen in spatial case - but handle gracefully
                        # This means gauge frame is somehow not spatial
                        omega_c = omega  # (K, K) - use same transform everywhere

                    # Apply transport to beliefs at this point
                    mu_weighted += w_i * (omega_c @ agent.mu_q[c_idx])
                    Sigma_weighted += w_i * (omega_c @ agent.Sigma_q[c_idx] @ omega_c.T)
                    total_weight += w_i

                # Normalize
                if total_weight > 1e-6:
                    mu_M[c_idx] = mu_weighted / total_weight
                    Sigma_M[c_idx] = Sigma_weighted / total_weight
                # else: keep default (zeros for mu, identity for Sigma)

            return mu_M, Sigma_M
    
    def _renormalize_models(self,
                          constituents: List[HierarchicalAgent],
                          coherence_scores: Optional[np.ndarray] = None,
                          region_mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute renormalized model/prior fields via coherence-weighted transport.

        Uses same Presence × Coherence weighting as beliefs:
            wᵢ(c) ∝ χᵢ(c) · C̄ᵢ
            p_M(c) = Σᵢ wᵢ(c) · Ωᵢⱼ[pᵢ(c)]

        Args:
            constituents: List of constituent agents
            coherence_scores: Optional precomputed coherence scores
            region_mask: Optional (*spatial,) boolean mask. If provided, only
                        compute fields within the True region (efficiency optimization)

        Returns:
            (mu_M, Sigma_M): Renormalized model parameters
        """
        # Same mathematical procedure as beliefs, just different fields
        ref = constituents[0]
        K = ref.K

        if coherence_scores is None:
            coherence_scores = self._compute_coherence_scores(constituents, field_type='model')

        if ref.base_manifold.is_point:
            # 0D case
            mu_weighted = np.zeros(K)
            Sigma_weighted = np.zeros((K, K))
            total_weight = 0.0

            for i, agent in enumerate(constituents):
                C_i = coherence_scores[i]

                omega = compute_transport(
                    ref.gauge.phi,
                    agent.gauge.phi,
                    ref.generators,
                    validate=False
                )

                mu_weighted += C_i * (omega @ agent.mu_p)
                Sigma_weighted += C_i * (omega @ agent.Sigma_p @ omega.T)
                total_weight += C_i

            return mu_weighted / total_weight, Sigma_weighted / total_weight

        else:
            # Spatial case
            spatial_shape = ref.base_manifold.shape
            mu_M = np.zeros((*spatial_shape, K))
            # Initialize Sigma_M to DIFFUSE prior outside support
            # Large variance = high uncertainty = "don't care about this region"
            diffuse_variance = 1000.0
            Sigma_M = np.broadcast_to(diffuse_variance * np.eye(K), (*spatial_shape, K, K)).copy()

            transports = []
            for agent in constituents:
                omega = compute_transport(
                    ref.gauge.phi,
                    agent.gauge.phi,
                    ref.generators,
                    validate=False
                )
                transports.append(omega)

            # Determine which points to compute
            if region_mask is not None:
                active_indices = np.argwhere(region_mask)
                points_to_compute = [tuple(idx) for idx in active_indices]
            else:
                points_to_compute = list(np.ndindex(spatial_shape))

            for c_idx in points_to_compute:
                mu_weighted = np.zeros(K)
                Sigma_weighted = np.zeros((K, K))
                total_weight = 0.0

                for i, agent in enumerate(constituents):
                    chi_i = agent.support.chi_weight[c_idx]
                    if chi_i < 1e-6:
                        continue

                    w_i = chi_i * coherence_scores[i]
                    omega = transports[i]

                    # Extract transport operator at this spatial location
                    if omega.ndim > 2:
                        omega_c = omega[c_idx]  # (K, K) at point c_idx
                    else:
                        # Handle case where gauge frame is not spatial
                        omega_c = omega  # (K, K) - use same transform everywhere

                    # Apply transport to priors at this point
                    mu_weighted += w_i * (omega_c @ agent.mu_p[c_idx])
                    Sigma_weighted += w_i * (omega_c @ agent.Sigma_p[c_idx] @ omega_c.T)
                    total_weight += w_i

                if total_weight > 1e-6:
                    mu_M[c_idx] = mu_weighted / total_weight
                    Sigma_M[c_idx] = Sigma_weighted / total_weight
                # else: keep default (zeros for mu, identity for Sigma)

            return mu_M, Sigma_M
    
    def _average_gauge_frames(self,
                            constituents: List[HierarchicalAgent],
                            coherence_scores: Optional[np.ndarray] = None,
                            method: str = 'frechet',
                            region_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute average gauge frame on SO(3) with coherence weighting.

        Uses proper Fréchet mean (geometric average) on the SO(3) manifold:
        1. Map gauge frames to rotation matrices: Rᵢ = exp(φᵢ)
        2. Compute weighted Fréchet mean R̄
        3. Map back to Lie algebra: φ̄ = log(R̄)

        Weighting:
            - 0D: wᵢ ∝ C̄ᵢ (coherence only)
            - Spatial: wᵢ ∝ χᵢ(center) · C̄ᵢ (presence × coherence at center)

        Args:
            constituents: List of agents to average
            coherence_scores: Optional precomputed coherence scores
            method: 'frechet' (geometric, default) or 'euclidean' (fast approximation)
            region_mask: Optional (*spatial,) boolean mask. If provided, only
                        compute within the True region (efficiency optimization)

        Returns:
            phi_M: Averaged gauge frame in so(3), shape (*spatial, 3) or (3,)

        Notes:
            - Euclidean average only valid for small deviations (< 0.5 rad)
            - Fréchet mean is geometrically correct but ~10x slower
            - For meta-agent formation, geometric correctness is essential!
        """
        if coherence_scores is None:
            coherence_scores = self._compute_coherence_scores(constituents)

        # Extract gauge frames
        phis = [agent.gauge.phi for agent in constituents]

        # Check dimensionality
        ref_phi = phis[0]
        if ref_phi.ndim == 2:
            # Point manifold: phi is (K, K)
            # Compute global weights (coherence only, all χᵢ = 1)
            weights = coherence_scores.copy()
            weights = weights / np.sum(weights)

            # Check if all frames are nearly identical (fast path)
            phi_std = np.std(phis, axis=0)
            if np.max(phi_std) < 0.1 and method == 'frechet':
                # All frames within 0.1 rad - Euclidean average is accurate enough
                method = 'euclidean'

            # Compute average
            phi_avg = average_gauge_frames_so3(phis, weights=weights, method=method)
        else:
            # Spatial manifold: phi is (*spatial, 3) in axis-angle representation
            # NOT (*spatial, K, K) matrices! Matrices are computed on-the-fly via exp(phi)
            spatial_shape = ref_phi.shape[:-1]  # Extract spatial dimensions
            axis_angle_dim = ref_phi.shape[-1]  # Should be 3 for SO(3)
            phi_avg = np.zeros(ref_phi.shape)

            # Debug: Check all phis have correct shape
            for i, phi in enumerate(phis):
                if phi.shape != ref_phi.shape:
                    raise ValueError(
                        f"Gauge frame shape mismatch in meta-agent formation:\n"
                        f"  Reference (agent 0): {ref_phi.shape}\n"
                        f"  Agent {i}: {phi.shape}\n"
                        f"  Expected: (*spatial, 3) = {ref_phi.shape}\n"
                        f"  All constituents must have same spatial shape and gauge group!"
                    )

            # Determine which points to compute
            if region_mask is not None:
                active_indices = np.argwhere(region_mask)
                points_to_compute = [tuple(idx) for idx in active_indices]
            else:
                points_to_compute = list(np.ndindex(spatial_shape))

            # Loop over spatial points in region
            for idx in points_to_compute:
                # Extract axis-angle vectors (shape (3,)) at this spatial point from all agents
                phis_at_point = [phi[idx] for phi in phis]

                # Verify extraction worked correctly
                for i, phi_at_pt in enumerate(phis_at_point):
                    if phi_at_pt.shape != (axis_angle_dim,):
                        raise ValueError(
                            f"Gauge frame extraction failed at spatial index {idx}:\n"
                            f"  Agent {i}: got shape {phi_at_pt.shape}, expected ({axis_angle_dim},)\n"
                            f"  Full phi shape: {phis[i].shape}"
                        )

                # Compute weights at this spatial point: w_i(x) = χ_i(x) · C̄_i
                weights_at_point = np.array([
                    agent.support.chi_weight[idx] * coherence_scores[i]
                    for i, agent in enumerate(constituents)
                ])

                # Normalize
                weight_sum = np.sum(weights_at_point)
                if weight_sum > 1e-12:
                    weights_at_point = weights_at_point / weight_sum
                else:
                    # No support here - use uniform weights
                    weights_at_point = np.ones(len(constituents)) / len(constituents)

                # Check if frames at this point are nearly identical
                phi_std_point = np.std(phis_at_point, axis=0)
                method_point = method
                if np.max(phi_std_point) < 0.1 and method == 'frechet':
                    method_point = 'euclidean'

                # Compute average at this point with spatially-varying weights
                try:
                    phi_avg[idx] = average_gauge_frames_so3(phis_at_point, weights=weights_at_point, method=method_point)
                except Exception as e:
                    print(f"\n{'='*70}")
                    print(f"ERROR in gauge frame averaging at spatial point {idx}")
                    print(f"{'='*70}")
                    print(f"phis_at_point shapes: {[p.shape for p in phis_at_point]}")
                    print(f"phis_at_point types: {[type(p) for p in phis_at_point]}")
                    print(f"weights_at_point shape: {weights_at_point.shape}")
                    print(f"weights_at_point: {weights_at_point}")
                    print(f"method: {method_point}")
                    print(f"Full phis shapes: {[p.shape for p in phis]}")
                    print(f"spatial_shape: {spatial_shape}")
                    print(f"axis_angle_dim: {axis_angle_dim}")
                    for i, phi_pt in enumerate(phis_at_point):
                        print(f"  Agent {i} phi at {idx}: shape={phi_pt.shape}, value={phi_pt}")
                    raise

        return phi_avg
    
    def _compute_meta_support(self,
                             constituents: List[HierarchicalAgent],
                             coherence_scores: Optional[np.ndarray] = None) -> SupportRegion:
        """
        Compute meta-agent support via Presence × Coherence weighting.
        
        χ_M(c) = Σᵢ wᵢ(c) · χᵢ(c)
        where wᵢ(c) ∝ χᵢ(c) · C̄ᵢ
        
        This gives quadratic weighting: χ_M(c) ∝ Σᵢ χᵢ(c)² · C̄ᵢ
        
        Strong coherent agents dominate; weak agents contribute little.
        
        Args:
            constituents: List of constituent agents
            coherence_scores: Optional precomputed coherence scores
        
        Returns:
            support: SupportRegion for meta-agent
        """
        ref = constituents[0]
        
        if ref.base_manifold.is_point:
            return create_full_support(ref.base_manifold)
        
        # Compute coherence scores if not provided
        if coherence_scores is None:
            coherence_scores = self._compute_coherence_scores(constituents)
        
        spatial_shape = ref.base_manifold.shape
        chi_meta = np.zeros(spatial_shape, dtype=np.float32)
        
        # At each spatial location
        for c_idx in np.ndindex(spatial_shape):
            chi_values = []
            weights_unnorm = []
            
            for i, agent in enumerate(constituents):
                chi_i = agent.support.chi_weight[c_idx]
                C_i = coherence_scores[i]
                
                chi_values.append(chi_i)
                weights_unnorm.append(chi_i * C_i)  # Presence × Coherence
            
            # Normalize weights
            total_weight = sum(weights_unnorm)
            
            if total_weight > 1e-6:
                weights = [w / total_weight for w in weights_unnorm]
                chi_meta[c_idx] = sum(w * chi for w, chi in zip(weights, chi_values))
            else:
                # No substantive presence - leave as zero
                chi_meta[c_idx] = 0.0
        
        return SupportRegion(
            base_manifold=ref.base_manifold,
            chi_weight=chi_meta
        )
    
    def _compute_coherence(self,
                          constituents: List[HierarchicalAgent],
                          field_type: str = 'belief') -> float:
        """
        Compute coherence metric for constituent cluster.
        
        Measures how aligned the constituents are (lower KL = higher coherence).
        
        Args:
            constituents: List of agents in cluster
            field_type: 'belief' or 'model'
        
        Returns:
            coherence: Value in [0, 1] (1 = perfect consensus)
        """
        from math_utils.numerical_utils import kl_gaussian

        # Compute average pairwise KL divergence
        kl_sum = 0.0
        n_pairs = 0

        for i, agent_i in enumerate(constituents):
            for agent_j in constituents[i+1:]:
                if field_type == 'belief':
                    mu_i, Sigma_i = agent_i.mu_q, agent_i.Sigma_q
                    mu_j, Sigma_j = agent_j.mu_q, agent_j.Sigma_q
                else:
                    mu_i, Sigma_i = agent_i.mu_p, agent_i.Sigma_p
                    mu_j, Sigma_j = agent_j.mu_p, agent_j.Sigma_p

                # Symmetrized KL
                kl_ij = kl_gaussian(mu_i, Sigma_i, mu_j, Sigma_j)
                kl_ji = kl_gaussian(mu_j, Sigma_j, mu_i, Sigma_i)
                kl_sum += (kl_ij + kl_ji) / 2
                n_pairs += 1
        
        if n_pairs == 0:
            return 1.0
        
        avg_kl = kl_sum / n_pairs
        
        # Convert to coherence metric (exponential decay)
        coherence = np.exp(-avg_kl)
        
        return coherence
    
    def _compute_coherence_scores(self,
                                 constituents: List[HierarchicalAgent],
                                 field_type: str = 'belief') -> np.ndarray:
        """
        Compute coherence score for each agent with the cluster.
        
        C̄ᵢ = exp(-average_KL_with_others)
        
        Args:
            constituents: List of agents in cluster
            field_type: 'belief' or 'model' - which field to measure coherence
        
        Returns:
            scores: Array of coherence values in (0, 1], shape (n,)
        
        Notes:
            - Perfect consensus: C̄ᵢ = 1.0 (KL = 0)
            - Weak coherence: C̄ᵢ ≈ 0.0 (large KL)
            - Used to weight agents in meta-agent formation
        """
        from math_utils.numerical_utils import kl_gaussian

        n = len(constituents)
        coherence_scores = np.zeros(n)

        for i, agent_i in enumerate(constituents):
            kl_sum = 0.0
            count = 0

            for j, agent_j in enumerate(constituents):
                if i == j:
                    continue

                # Compute KL at representative location
                if agent_i.base_manifold.is_point:
                    # 0D case: use fields directly
                    if field_type == 'belief':
                        mu_i, Sigma_i = agent_i.mu_q, agent_i.Sigma_q
                        mu_j, Sigma_j = agent_j.mu_q, agent_j.Sigma_q
                    else:
                        mu_i, Sigma_i = agent_i.mu_p, agent_i.Sigma_p
                        mu_j, Sigma_j = agent_j.mu_p, agent_j.Sigma_p
                else:
                    # Spatial case: use center of support
                    center_idx = tuple(s//2 for s in agent_i.base_manifold.shape)

                    if field_type == 'belief':
                        mu_i = agent_i.mu_q[center_idx]
                        Sigma_i = agent_i.Sigma_q[center_idx]
                        mu_j = agent_j.mu_q[center_idx]
                        Sigma_j = agent_j.Sigma_q[center_idx]
                    else:
                        mu_i = agent_i.mu_p[center_idx]
                        Sigma_i = agent_i.Sigma_p[center_idx]
                        mu_j = agent_j.mu_p[center_idx]
                        Sigma_j = agent_j.Sigma_p[center_idx]

                # Symmetric KL divergence
                kl_ij = kl_gaussian(mu_i, Sigma_i, mu_j, Sigma_j)
                kl_ji = kl_gaussian(mu_j, Sigma_j, mu_i, Sigma_i)
                kl_sym = (kl_ij + kl_ji) / 2.0
                
                kl_sum += kl_sym
                count += 1
            
            # Average KL with all others
            avg_kl = kl_sum / count if count > 0 else 0.0
            
            # Coherence score via exponential decay
            # KL = 0 → C̄ = 1.0 (perfect coherence)
            # KL = 1 → C̄ ≈ 0.37
            # KL = 5 → C̄ ≈ 0.007 (weak coherence)
            coherence_scores[i] = np.exp(-avg_kl)
        
        return coherence_scores
    
    def _identify_leader(self,
                        constituents: List[HierarchicalAgent],
                        coherence_scores: np.ndarray) -> Tuple[int, float, np.ndarray]:
        """
        Identify the leader agent in the cluster.
        
        Leader = agent with max leadership score L_i = χ_i² · C̄_i
        
        For spatial cases, evaluates at center of support.
        For 0D (transformers), all χ_i = 1, so leader is most coherent agent.
        
        Args:
            constituents: List of agents in cluster
            coherence_scores: Precomputed coherence scores C̄_i
        
        Returns:
            leader_idx: Index of leader in constituents list
            leader_score: Leadership score L_leader
            leadership_distribution: All leadership scores [L_1, ..., L_n]
        
        Physical Meaning:
            The leader is the agent that most strongly "templates" the meta-agent.
            It has strong presence (high χ) and high coherence with the cluster.
        """
        n = len(constituents)
        leadership_scores = np.zeros(n)
        
        for i, agent in enumerate(constituents):
            if agent.base_manifold.is_point:
                # 0D: all χ = 1, leader is most coherent
                chi_i = 1.0
            else:
                # Spatial: evaluate at center
                center_idx = tuple(s//2 for s in agent.base_manifold.shape)
                chi_i = agent.support.chi_weight[center_idx]
            
            # Leadership score: L_i = χ_i² · C̄_i
            leadership_scores[i] = (chi_i ** 2) * coherence_scores[i]
        
        # Identify leader
        leader_idx = int(np.argmax(leadership_scores))
        leader_score = leadership_scores[leader_idx]
        
        return leader_idx, leader_score, leadership_scores
    
    # =========================================================================
    # System Queries
    # =========================================================================

    def get_active_agents_at_scale(self, scale: int, free_only: bool = False) -> List[HierarchicalAgent]:
        """
        Get active agents at a specific scale.

        Args:
            scale: Scale to query
            free_only: If True, only return agents NOT committed to a meta-agent
                      (i.e., agents with parent_meta=None)

        Returns:
            List of active (and optionally free) agents
        """
        if free_only:
            # Only agents not already in a meta-agent
            return [a for a in self.agents[scale] if a.is_active and a.parent_meta is None]
        else:
            # All active agents (including those in meta-agents)
            return [a for a in self.agents[scale] if a.is_active]

    def get_all_active_agents(self, free_only: bool = False) -> List[HierarchicalAgent]:
        """
        Get all active agents across all scales.

        Args:
            free_only: If True, only return agents NOT committed to a meta-agent

        Returns:
            List of active (and optionally free) agents
        """
        active = []
        for scale in sorted(self.agents.keys()):
            active.extend(self.get_active_agents_at_scale(scale, free_only=free_only))
        return active

    def max_scale(self) -> int:
        """Get maximum scale present in system."""
        return max(self.agents.keys()) if self.agents else 0

    # =========================================================================
    # Cross-Scale Dynamics
    # =========================================================================

    def update_cross_scale_priors(self, enable_tower=False, max_depth=1, decay=0.3):
        """
        Update all agent priors with self-referential closure.

        Two modes:
        1. Agents with parents: p_i^(ζ) ← q_M^(ζ+1) (from parent)
        2. Top-scale agents: p_top ← statistics(ALL agents) (strange loop!)

        With Ouroboros Tower enabled:
        - Agents receive priors from ALL ancestors, not just parent
        - Creates non-Markovian memory of entire ancestral lineage

        This creates bidirectional flow:
        - Top observes collective → forms prior
        - Prior flows down hierarchy (through entire tower!)
        - Lower scales evolve → change collective
        - Top re-observes → updates prior
        - LOOP: System observes itself!

        NOTE: Self-referential closure only activates when hierarchy exists (max_scale > 0).
        Before meta-agents form, agents keep their existing priors.

        Args:
            enable_tower: Enable multi-scale hyperprior propagation
            max_depth: How many ancestral levels to include
            decay: Exponential decay for hyperprior influence
        """
        n_updated_from_parent = 0
        n_updated_from_global = 0

        # Only apply self-referential closure when multiple scales exist
        has_hierarchy = self.max_scale() > 0

        for agent in self.get_all_active_agents():
            if agent.parent_meta is not None:
                # Regular hierarchical flow: prior from parent (+ tower if enabled)
                agent.update_prior_from_parent(
                    enable_tower=enable_tower,
                    max_depth=max_depth,
                    decay=decay
                )
                n_updated_from_parent += 1
            elif has_hierarchy:
                # Top-scale: self-referential closure (only when hierarchy exists!)
                # System observes itself observing itself!
                agent.update_prior_from_global_state(self)
                n_updated_from_global += 1
            # else: Keep existing prior (before meta-agents form)

        return {
            'from_parent': n_updated_from_parent,
            'from_global': n_updated_from_global,
            'total': n_updated_from_parent + n_updated_from_global
        }

    def compute_observation_likelihood_meta(self, meta_agent: HierarchicalAgent) -> float:
        """
        Compute observation likelihood for meta-agent from constituents.

        E_obs = - E_q[log p(o|q)] where o = aggregate({q_i})

        Args:
            meta_agent: Meta-agent to compute observation likelihood for

        Returns:
            Observation energy contribution
        """
        if not meta_agent.is_meta or len(meta_agent.constituents) == 0:
            return 0.0

        # Generate observation from constituents
        o_meta = meta_agent.generate_observations_from_constituents()
        if o_meta is None:
            return 0.0

        # Compute Gaussian observation likelihood
        # p(o|q) = N(o | C*mu_q, R) where C=I, R=small
        # For simplicity: - log p(o|q) ≈ ||o - mu_q||² / (2σ²)

        R_scale = 0.1  # Observation noise scale
        residual = o_meta - meta_agent.mu_q
        energy = 0.5 * np.sum(residual ** 2) / (R_scale ** 2)

        return energy

    def auto_detect_and_condense(self,
                                 scale: int,
                                 kl_threshold: float = 0.01,
                                 min_cluster_size: int = 2,
                                 deactivate_constituents: bool = False) -> List[HierarchicalAgent]:
        """
        Automatically detect consensus and form meta-agents.

        Uses ConsensusDetector to find agents that have reached epistemic death,
        then condenses them into meta-agents.

        Args:
            scale: Scale to check for consensus
            kl_threshold: KL divergence threshold for consensus
            min_cluster_size: Minimum agents in a cluster to form meta-agent
            deactivate_constituents: Whether to freeze constituents after condensation

        Returns:
            List of newly formed meta-agents
        """
        from meta.consensus import ConsensusDetector
        from math_utils.numerical_utils import kl_gaussian

        # CRITICAL: Only check "free" agents (not already in a meta-agent)
        # With continuous flow, constituents stay active but shouldn't
        # form new meta-agents (they're committed to their parent)
        agents_at_scale = self.get_active_agents_at_scale(scale, free_only=True)

        # Check if priors are identical (vanilla Active Inference mode)
        identical_priors = False
        if hasattr(self, 'system_config'):
            identical_priors_mode = getattr(self.system_config, "identical_priors", "off")
            identical_priors = identical_priors_mode in ("lock", "init_copy")

        if identical_priors:
            print(f"    ℹ️  Identical priors mode - checking belief consensus only")

        if len(agents_at_scale) < min_cluster_size:
            return []

        # DIAGNOSTIC: Show pairwise KL divergences
        if identical_priors:
            print(f"    Pairwise belief KL divergences (threshold={kl_threshold:.4f}):")
        else:
            print(f"    Pairwise KL divergences (threshold={kl_threshold:.4f}):")

        for i in range(min(len(agents_at_scale), 5)):  # Show first 5 agents
            for j in range(i+1, min(len(agents_at_scale), 5)):
                agent_i = agents_at_scale[i]
                agent_j = agents_at_scale[j]

                # Compute transport
                phi_i = agent_i.gauge.phi
                phi_j = agent_j.gauge.phi

                # Ensure phi has correct shape for spatial: (*spatial, 3)
                # For point manifolds: (3,) -> (1, 3)
                if phi_i.ndim == 1:
                    phi_i = phi_i[None, :]  # (3,) -> (1, 3)
                if phi_j.ndim == 1:
                    phi_j = phi_j[None, :]  # (3,) -> (1, 3)

                omega_ij = compute_transport(
                    phi_i,
                    phi_j,
                    agent_i.generators,
                    validate=False
                )

                # For point manifolds: omega_ij is (1, K, K), squeeze to (K, K)
                if omega_ij.shape[0] == 1 and agent_j.mu_q.ndim == 1:
                    omega_ij = omega_ij[0]  # (1, K, K) -> (K, K)

                # Check BELIEF consensus
                # For spatial: use einsum for proper broadcasting
                # omega_ij: (*spatial, K, K), mu_q: (*spatial, K) -> (*spatial, K)
                if omega_ij.ndim > 2:
                    mu_q_j_t = np.einsum('...ij,...j->...i', omega_ij, agent_j.mu_q)
                    Sigma_q_j_t = np.einsum('...ik,...kl,...jl->...ij', omega_ij, agent_j.Sigma_q, omega_ij)
                else:
                    # Point manifold case
                    mu_q_j_t = omega_ij @ agent_j.mu_q
                    Sigma_q_j_t = omega_ij @ agent_j.Sigma_q @ omega_ij.T

                # Sanitize transported covariance (gauge transformations can introduce numerical errors)
                from math_utils.numerical_utils import sanitize_sigma
                Sigma_q_j_t = sanitize_sigma(Sigma_q_j_t, eps=1e-6)

                try:
                    kl_belief = kl_gaussian(agent_i.mu_q, agent_i.Sigma_q, mu_q_j_t, Sigma_q_j_t)

                    # Handle spatial manifolds: kl_belief may be array (*spatial,)
                    if np.ndim(kl_belief) > 0:
                        # Only compute max/mean within OVERLAP region
                        chi_i = agent_i.support.chi_weight
                        chi_j = agent_j.support.chi_weight
                        overlap_mask = (chi_i > 0.01) & (chi_j > 0.01)

                        if np.any(overlap_mask):
                            kl_in_overlap = kl_belief[overlap_mask]
                            kl_belief_max = np.max(kl_in_overlap)
                            kl_belief_mean = np.mean(kl_in_overlap)
                        else:
                            kl_belief_max = np.inf
                            kl_belief_mean = np.inf

                        belief_ok = "✓" if kl_belief_max < kl_threshold else "✗"
                        kl_belief_str = f"mean={kl_belief_mean:.6f} max={kl_belief_max:.6f}"
                    else:
                        kl_belief_mean = float(kl_belief)
                        kl_belief_max = kl_belief_mean
                        belief_ok = "✓" if kl_belief < kl_threshold else "✗"
                        kl_belief_str = f"{kl_belief_mean:.6f}"
                except:
                    kl_belief_mean = float('inf')
                    kl_belief_max = float('inf')
                    belief_ok = "✗"
                    kl_belief_str = "inf"

                if identical_priors:
                    # Skip model check - priors are identical by construction
                    print(f"      {i}↔{j}: belief {kl_belief_str}{belief_ok}")
                else:
                    # Check MODEL consensus
                    # For spatial: use einsum for proper broadcasting
                    if omega_ij.ndim > 2:
                        mu_p_j_t = np.einsum('...ij,...j->...i', omega_ij, agent_j.mu_p)
                        Sigma_p_j_t = np.einsum('...ik,...kl,...jl->...ij', omega_ij, agent_j.Sigma_p, omega_ij)
                    else:
                        # Point manifold case
                        mu_p_j_t = omega_ij @ agent_j.mu_p
                        Sigma_p_j_t = omega_ij @ agent_j.Sigma_p @ omega_ij.T

                    # Sanitize transported covariance
                    Sigma_p_j_t = sanitize_sigma(Sigma_p_j_t, eps=1e-6)

                    try:
                        kl_model = kl_gaussian(agent_i.mu_p, agent_i.Sigma_p, mu_p_j_t, Sigma_p_j_t)

                        # Handle spatial manifolds: kl_model may be array (*spatial,)
                        if np.ndim(kl_model) > 0:
                            # Only compute max/mean within OVERLAP region
                            # (chi_i, chi_j, overlap_mask already computed above for beliefs)
                            if np.any(overlap_mask):
                                kl_in_overlap = kl_model[overlap_mask]
                                kl_model_max = np.max(kl_in_overlap)
                                kl_model_mean = np.mean(kl_in_overlap)
                            else:
                                kl_model_max = np.inf
                                kl_model_mean = np.inf

                            model_ok = "✓" if kl_model_max < kl_threshold else "✗"
                            kl_model_str = f"mean={kl_model_mean:.6f} max={kl_model_max:.6f}"
                        else:
                            kl_model_mean = float(kl_model)
                            kl_model_max = kl_model_mean
                            model_ok = "✓" if kl_model < kl_threshold else "✗"
                            kl_model_str = f"{kl_model_mean:.6f}"
                    except:
                        kl_model_mean = float('inf')
                        kl_model_max = float('inf')
                        model_ok = "✗"
                        kl_model_str = "inf"

                    # Epistemic death = BOTH consensus
                    both_ok = "✓✓" if (belief_ok == "✓" and model_ok == "✓") else "✗✗"
                    print(f"      {i}↔{j}: belief {kl_belief_str}{belief_ok} | model {kl_model_str}{model_ok} {both_ok}")

        # Create a temporary wrapper for consensus detection
        class AgentWrapper:
            """Wrapper to make agents compatible with ConsensusDetector."""
            def __init__(self, agents_list):
                self.agents = agents_list
                self.n_agents = len(agents_list)

        wrapper = AgentWrapper(agents_at_scale)

        # Detect consensus clusters
        if identical_priors:
            # Skip model consensus check - only check beliefs
            # (Models are identical by construction, so model KL after transport is meaningless)
            detector = ConsensusDetector(
                belief_threshold=kl_threshold,
                model_threshold=float('inf'),  # Always pass model check
                use_symmetric_kl=True
            )
        else:
            # Check both belief AND model consensus (epistemic death)
            detector = ConsensusDetector(
                belief_threshold=kl_threshold,
                model_threshold=kl_threshold,
                use_symmetric_kl=True
            )

        clusters = detector.find_consensus_clusters(wrapper)

        # Filter clusters by size
        valid_clusters = [c for c in clusters if len(c) >= min_cluster_size]

        if not valid_clusters:
            return []

        # Form meta-agents from consensus clusters
        new_meta_agents = self.form_meta_agents_at_scale(
            source_scale=scale,
            partitions=valid_clusters,
            deactivate_constituents=deactivate_constituents
        )

        if deactivate_constituents:
            mode_str = "categorical - constituents frozen"
        else:
            mode_str = "continuous flow - constituents evolving 🌊"
        print(f"[Auto-Condensation ζ={scale}] Detected {len(valid_clusters)} consensus clusters ({mode_str})")

        return new_meta_agents

    def auto_detect_and_condense_spatial(
        self,
        scale: int,
        kl_threshold: float = 0.1,
        min_region_size: int = 4,
        min_volume_fraction: float = 0.0,
        smooth_sigma: float = 1.5,
        deactivate_constituents: bool = False
    ) -> List[HierarchicalAgent]:
        """
        Detect and form meta-agents with spatial coherence (arbitrary dimensions).

        For spatial manifolds, this method:
        1. Finds agent clusters via consensus
        2. Computes spatial consensus fields for each cluster
        3. Identifies connected regions of consensus
        4. Forms meta-agents with support matching consensus geometry

        The resulting meta-agents have EMERGENT shapes determined by
        information topology, not predetermined geometry.

        Args:
            scale: Scale to check for consensus
            kl_threshold: KL divergence threshold for spatial consensus
            min_region_size: Minimum region size in points (absolute)
            min_volume_fraction: Minimum region size as fraction (relative)
            smooth_sigma: Gaussian smoothing width for boundaries (preserves smooth sections)
            deactivate_constituents: Whether to freeze constituents

        Returns:
            List of newly formed meta-agents (may be more than #clusters if
            a cluster's consensus region is spatially disconnected)

        Theory:
            Meta-agent shape is determined by the TOPOLOGY OF CONSENSUS:
            - Where do the constituent agents agree?
            - Connected regions become single meta-agents
            - Disconnected consensus → multiple meta-agents from one cluster

            Smoothing ensures meta-agents are SMOOTH SECTIONS (differentiable
            support functions), maintaining gauge-theoretic structure.
        """
        from meta.consensus import ConsensusDetector
        from meta.spatial_emergence import (
            SpatialEmergenceDetector,
            SpatialConsensusRegion
        )

        # Handle 0D case: fall back to standard method
        if self.base_manifold.is_point:
            print("[Spatial Emergence] 0D manifold - using standard condensation")
            return self.auto_detect_and_condense(
                scale=scale,
                kl_threshold=kl_threshold,
                deactivate_constituents=deactivate_constituents
            )

        # Get free agents at this scale
        agents_at_scale = self.get_active_agents_at_scale(scale, free_only=True)

        if len(agents_at_scale) < 2:
            return []

        # Create consensus detector
        detector = ConsensusDetector(
            belief_threshold=kl_threshold,
            model_threshold=kl_threshold,
            use_symmetric_kl=True
        )

        # Create spatial emergence detector
        spatial_detector = SpatialEmergenceDetector(
            consensus_detector=detector,
            kl_threshold=kl_threshold,
            min_region_size=min_region_size,
            min_volume_fraction=min_volume_fraction,
            smooth_sigma=smooth_sigma,
            connectivity='full',
            aggregation='mean',
            check_models=True
        )

        # Detect emergence regions
        # Need to create a wrapper for the spatial detector
        class AgentWrapper:
            def __init__(self, agents_list):
                self.agents = agents_list
                self.n_agents = len(agents_list)

        wrapper = AgentWrapper(agents_at_scale)
        regions = spatial_detector.detect_emergence_regions(wrapper, source_scale=0)

        if not regions:
            print(f"[Spatial Emergence ζ={scale}] No consensus regions detected")
            return []

        # Form meta-agents from spatial regions
        new_meta_agents = self._form_meta_agents_from_spatial_regions(
            source_scale=scale,
            regions=regions,
            agents_at_scale=agents_at_scale,
            spatial_detector=spatial_detector,
            deactivate_constituents=deactivate_constituents
        )

        n_clusters = len(set(tuple(r.cluster_indices) for r in regions))
        print(f"[Spatial Emergence ζ={scale}] {n_clusters} clusters → {len(regions)} regions → {len(new_meta_agents)} meta-agents")

        return new_meta_agents

    def _form_meta_agents_from_spatial_regions(
        self,
        source_scale: int,
        regions: List,  # List[SpatialConsensusRegion]
        agents_at_scale: List[HierarchicalAgent],
        spatial_detector,  # SpatialEmergenceDetector
        deactivate_constituents: bool = False
    ) -> List[HierarchicalAgent]:
        """
        Form meta-agents from detected spatial consensus regions.

        Each SpatialConsensusRegion becomes one meta-agent with support
        matching the smoothed consensus geometry.

        Args:
            source_scale: Scale of constituent agents
            regions: List of SpatialConsensusRegion from detection
            agents_at_scale: All agents at source scale
            spatial_detector: SpatialEmergenceDetector instance
            deactivate_constituents: Whether to freeze constituents

        Returns:
            List of newly formed meta-agents
        """
        target_scale = source_scale + 1
        source_agents = self.agents[source_scale]

        # Check emergence level cap
        if self.max_emergence_levels is not None and target_scale > self.max_emergence_levels:
            print(f"[Level Cap] Cannot form meta-agents at scale {target_scale}")
            return []

        new_meta_agents = []

        for region in regions:
            partition = region.cluster_indices

            if len(partition) < 2:
                continue

            # Get constituent agents
            constituents = [agents_at_scale[i] for i in partition]

            # Use region's coherence scores
            coherence_scores = region.coherence_scores
            if coherence_scores is None:
                coherence_scores = self._compute_coherence_scores(constituents, field_type='belief')

            # Identify leader
            leader_idx, leader_score, leadership_dist = self._identify_leader(
                constituents, coherence_scores
            )

            # Get region mask for efficient computation
            # Only compute fields within the consensus region (HUGE efficiency gain)
            region_mask = region.component_mask

            # Compute renormalized fields (restricted to consensus region)
            mu_q, Sigma_q = self._renormalize_beliefs(
                constituents, coherence_scores, region_mask=region_mask
            )
            mu_p, Sigma_p = self._renormalize_models(
                constituents, coherence_scores, region_mask=region_mask
            )
            phi = self._average_gauge_frames(
                constituents, coherence_scores, region_mask=region_mask
            )

            # Create meta-agent
            meta_index = len(self.agents[target_scale])
            meta_id = f"meta_{target_scale}_{meta_index}_r{region.component_id}"

            constituent_scale_indices = [
                ScaleIndex(source_scale, i) for i in partition
            ]

            belief_coherence = float(np.mean(coherence_scores))
            model_coherence_scores = self._compute_coherence_scores(constituents, field_type='model')
            model_coherence = float(np.mean(model_coherence_scores))

            meta_descriptor = MetaAgentDescriptor(
                scale_index=ScaleIndex(target_scale, meta_index),
                constituent_indices=constituent_scale_indices,
                emergence_time=self.current_time,
                belief_coherence=belief_coherence,
                model_coherence=model_coherence,
                leader_index=leader_idx,
                leader_score=leader_score,
                leadership_distribution=leadership_dist
            )

            meta_agent = HierarchicalAgent(
                scale=target_scale,
                local_index=meta_index,
                constituent_indices=constituent_scale_indices,
                meta_descriptor=meta_descriptor,
                agent_id=meta_id,
                config=constituents[0].config,
                rng=np.random.default_rng(hash(meta_id) % 2**32),
                base_manifold=self.base_manifold
            )

            # KEY DIFFERENCE: Use spatial support from consensus region
            meta_agent.support = spatial_detector.create_meta_agent_support(
                region=region,
                agents=agents_at_scale,
                weight_by_coherence=True
            )
            meta_agent.generators = constituents[0].generators

            meta_agent._initialize_belief_covariance()
            meta_agent._initialize_prior_covariance()
            meta_agent._initialize_gauge()

            meta_agent.mu_q = mu_q
            meta_agent.Sigma_q = Sigma_q
            meta_agent.mu_p = mu_p
            meta_agent.Sigma_p = Sigma_p
            meta_agent.gauge.phi = phi

            # Set up relationships
            meta_agent.constituents = constituents.copy()
            for agent in constituents:
                agent.parent_meta = meta_agent

            # Add to system
            self.agents[target_scale].append(meta_agent)
            new_meta_agents.append(meta_agent)

            # Optionally deactivate constituents
            if deactivate_constituents:
                for agent in constituents:
                    agent.is_active = False

            # Record condensation event with spatial info
            self.condensation_events.append({
                'time': self.current_time,
                'source_scale': source_scale,
                'target_scale': target_scale,
                'n_constituents': len(partition),
                'constituent_indices': constituent_scale_indices,
                'leader_index': leader_idx,
                'leader_score': leader_score,
                'coherence': {
                    'belief': belief_coherence,
                    'model': model_coherence
                },
                'spatial': {
                    'region_volume': region.volume,
                    'volume_fraction': region.volume_fraction,
                    'centroid': region.centroid,
                    'mean_consensus_strength': region.mean_consensus_strength
                }
            })

            print(f"  {meta_agent.scale_index}: {region.volume} pts "
                  f"({region.volume_fraction:.1%}) @ {tuple(int(x) for x in region.centroid)}")

        return new_meta_agents

    def summary(self) -> str:
        """Hierarchical structure summary."""
        lines = ["Multi-Scale System Structure"]
        lines.append("=" * 60)
        lines.append(f"Base manifold: {self.base_manifold}")
        lines.append("")

        for scale in sorted(self.agents.keys()):
            agents_at_scale = self.agents[scale]
            active_count = sum(1 for a in agents_at_scale if a.is_active)
            lines.append(f"Scale ζ={scale}: {active_count}/{len(agents_at_scale)} active")

            if scale > 0:  # Show meta-agent structure
                for agent in agents_at_scale[:5]:  # First 5
                    if agent.is_active and agent.is_meta:
                        lines.append(f"  {agent.scale_index}: "
                                   f"from {len(agent.constituent_indices)} constituents "
                                   f"(coherence: {agent.meta.belief_coherence:.3f})")

        lines.append("")
        lines.append(f"Total condensation events: {len(self.condensation_events)}")

        return "\n".join(lines)


# =============================================================================
# Analysis Functions
# =============================================================================

def analyze_hierarchical_structure(system: MultiScaleSystem) -> Dict:
    """
    Analyze the hierarchical structure and emergence patterns.
    
    Returns:
        Dictionary with structure metrics including leadership
    """
    metrics = {
        'n_scales': len(system.agents),
        'max_scale': system.max_scale(),
        'agents_per_scale': {},
        'active_per_scale': {},
        'total_agents': 0,
        'total_active': 0,
        'condensation_events': len(system.condensation_events),
        'mean_cluster_size': 0,
        'mean_coherence': 0,
        'leadership_stats': {}
    }
    
    # Count agents per scale
    cluster_sizes = []
    coherences = []
    leader_scores = []
    
    for scale, agents in system.agents.items():
        metrics['agents_per_scale'][scale] = len(agents)
        metrics['active_per_scale'][scale] = sum(1 for a in agents if a.is_active)
        metrics['total_agents'] += len(agents)
        metrics['total_active'] += sum(1 for a in agents if a.is_active)
        
        # Track meta-agent properties
        for agent in agents:
            if agent.is_meta:
                cluster_sizes.append(len(agent.meta.constituent_indices))
                coherences.append(agent.meta.belief_coherence)
                if agent.meta.leader_score is not None:
                    leader_scores.append(agent.meta.leader_score)
    
    if cluster_sizes:
        metrics['mean_cluster_size'] = np.mean(cluster_sizes)
        metrics['mean_coherence'] = np.mean(coherences)
    
    if leader_scores:
        metrics['leadership_stats'] = {
            'mean_score': np.mean(leader_scores),
            'min_score': np.min(leader_scores),
            'max_score': np.max(leader_scores),
            'std_score': np.std(leader_scores)
        }
    
    return metrics


def analyze_leadership_chains(system: MultiScaleSystem) -> Dict:
    """
    Trace leadership chains from base agents through meta-agents.
    
    Shows which base agents influence higher-scale structures.
    
    Returns:
        Dictionary mapping base agent indices to their influence chain
    """
    if 0 not in system.agents:
        return {}
    
    chains = {}
    base_agents = system.agents[0]
    
    # Initialize chains for each base agent
    for i, agent in enumerate(base_agents):
        chains[i] = {
            'scales_present': [0],
            'leadership_roles': [],  # (scale, meta_index, is_leader)
            'influence_score': 1.0 if agent.is_active else 0.0
        }
    
    # Trace through meta-agent hierarchy
    for scale in sorted(system.agents.keys()):
        if scale == 0:
            continue
        
        for meta_agent in system.agents[scale]:
            if not meta_agent.is_meta:
                continue
            
            # Get constituent indices at scale-1
            for constituent_idx in meta_agent.meta.constituent_indices:
                # This is tricky - need to map back to base agents
                # For now, record direct parent relationships
                pass
    
    return chains


def print_leadership_summary(system: MultiScaleSystem):
    """
    Print detailed leadership analysis.
    """
    print("\n" + "=" * 70)
    print("LEADERSHIP STRUCTURE ANALYSIS")
    print("=" * 70)
    
    for scale in sorted(system.agents.keys()):
        if scale == 0:
            continue  # Base agents have no leaders
        
        agents_at_scale = [a for a in system.agents[scale] if a.is_active and a.is_meta]
        
        if not agents_at_scale:
            continue
        
        print(f"\nScale ζ={scale}:")
        print("-" * 70)
        
        for agent in agents_at_scale:
            meta = agent.meta
            leader_idx = meta.leader_index
            leader_constituent = meta.constituent_indices[leader_idx]
            
            print(f"\n  Meta-agent {agent.scale_index}:")
            print(f"    Constituents: {meta.constituent_indices}")
            print(f"    Leader: {leader_constituent} (L={meta.leader_score:.4f})")
            print(f"    Leadership distribution:")
            
            # Show all leadership scores
            for i, (const_idx, L_i) in enumerate(zip(meta.constituent_indices, 
                                                     meta.leadership_distribution)):
                is_leader = "← LEADER" if i == leader_idx else ""
                pct = 100 * L_i / np.sum(meta.leadership_distribution)
                print(f"      {const_idx}: L={L_i:.4f} ({pct:.1f}%) {is_leader}")
            
            print(f"    Coherence: {meta.belief_coherence:.4f}")
    
    print("\n" + "=" * 70)