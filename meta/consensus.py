# -*- coding: utf-8 -*-
"""
Created on Sat Nov 15 11:32:57 2025

@author: chris and christine
"""

# -*- coding: utf-8 -*-
"""
Consensus Detection for Meta-Agent Emergence
=============================================

Detects when agents reach epistemic death (consensus) and are ready to form meta-agents.

Key concepts:
- Belief consensus: q_i ≈ Ω_ij[q_j] after transport
- Model consensus: p_i ≈ Ω_ij[p_j] after transport  
- Epistemic death: Both belief AND model consensus
- Consensus clusters: Groups of agents in mutual consensus

Author: Chris & Christine
Date: November 2025
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Set
from dataclasses import dataclass
from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

# Import from existing codebase
from math_utils.numerical_utils import kl_gaussian
from math_utils.transport import compute_transport


@dataclass
class ConsensusState:
    """Track consensus state of agent pair"""
    belief_consensus: bool = False
    model_consensus: bool = False
    belief_divergence: float = np.inf
    model_divergence: float = np.inf
    
    @property
    def is_epistemically_dead(self) -> bool:
        """Full consensus = epistemic death"""
        return self.belief_consensus and self.model_consensus


class ConsensusDetector:
    """
    Detect epistemic death and identify meta-agent candidates.
    
    Theory:
    -------
    Agents i,j reach consensus when:
        KL(q_i || Ω_ij[q_j]) < ε_belief  (belief consensus)
        KL(p_i || Ω_ij[p_j]) < ε_model   (model consensus)
    
    Where Ω_ij = exp(φ_i) exp(-φ_j) is the gauge transport operator.
    
    Epistemic death occurs when BOTH conditions hold, meaning agents
    share identical beliefs/models after accounting for gauge transformations.
    """
    
    def __init__(self, 
                 belief_threshold: float = 1e-3,
                 model_threshold: float = 1e-3,
                 use_symmetric_kl: bool = False,
                 cache_transport: bool = True):
        """
        Initialize consensus detector.
        
        Args:
            belief_threshold: KL divergence threshold for belief consensus
            model_threshold: KL divergence threshold for model consensus  
            use_symmetric_kl: Use symmetric KL (Jeffrey divergence) if True
            cache_transport: Cache transport operators for efficiency
        """
        
        self.belief_threshold = belief_threshold
        self.model_threshold = model_threshold
        self.use_symmetric_kl = use_symmetric_kl
        self.cache_transport = cache_transport
        self._transport_cache = {} if cache_transport else None
        
    def check_belief_consensus(self, 
                               agent_i, agent_j,
                               omega_ij: Optional[np.ndarray] = None) -> Tuple[bool, float]:
        """
        Check if agents have reached belief consensus.
        
        Args:
            agent_i, agent_j: Agent objects with mu_q, Sigma_q, gauge fields
            omega_ij: Pre-computed transport operator (optional)
            
        Returns:
            (consensus_reached, kl_divergence)
        """
        # Get or compute transport operator
        if omega_ij is None:
            omega_ij = self._get_transport(agent_i, agent_j)

        # Transport agent_j's belief to agent_i's frame
        # For spatial manifolds: use einsum for proper broadcasting
        # omega_ij: (*spatial, K, K), mu_q: (*spatial, K) -> (*spatial, K)
        if omega_ij.ndim > 2:
            mu_j_transported = np.einsum('...ij,...j->...i', omega_ij, agent_j.mu_q)
            Sigma_j_transported = np.einsum('...ik,...kl,...jl->...ij', omega_ij, agent_j.Sigma_q, omega_ij)
        else:
            # Point manifold case
            mu_j_transported = omega_ij @ agent_j.mu_q
            Sigma_j_transported = omega_ij @ agent_j.Sigma_q @ omega_ij.T

        # Sanitize transported covariance to ensure positive-definiteness
        # Gauge transformations can introduce small numerical errors
        from math_utils.numerical_utils import sanitize_sigma
        Sigma_j_transported = sanitize_sigma(Sigma_j_transported, eps=1e-6)
        
        # Compute KL divergence
        kl_div = kl_gaussian(
            agent_i.mu_q, agent_i.Sigma_q,
            mu_j_transported, Sigma_j_transported
        )

        if self.use_symmetric_kl:
            # Jeffrey divergence: (KL(i||j) + KL(j||i)) / 2
            kl_div_reverse = kl_gaussian(
                mu_j_transported, Sigma_j_transported,
                agent_i.mu_q, agent_i.Sigma_q
            )
            kl_div = (kl_div + kl_div_reverse) / 2

        # Handle spatial manifolds: kl_div may be array (*spatial,)
        # For consensus, use MAXIMUM KL within OVERLAP region only
        # (Outside overlap, KL is meaningless - one agent doesn't exist there)
        if np.ndim(kl_div) > 0:
            # Compute overlap mask: where BOTH agents have support
            chi_i = agent_i.support.chi_weight
            chi_j = agent_j.support.chi_weight
            overlap_mask = (chi_i > 0.01) & (chi_j > 0.01)

            if np.any(overlap_mask):
                kl_in_overlap = kl_div[overlap_mask]
                kl_div_max = np.max(kl_in_overlap)
                kl_div_scalar = float(np.mean(kl_in_overlap))
            else:
                # No overlap - can't compute consensus
                kl_div_max = np.inf
                kl_div_scalar = np.inf

            consensus = kl_div_max < self.belief_threshold
        else:
            kl_div_scalar = float(kl_div)
            consensus = kl_div < self.belief_threshold

        return consensus, kl_div_scalar

    def check_belief_consensus_spatial(self,
                                       agent_i, agent_j,
                                       omega_ij: Optional[np.ndarray] = None,
                                       active_mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Check belief consensus **pointwise** on spatial manifolds.

        Returns spatial maps showing where agents agree locally.

        Args:
            agent_i, agent_j: Agent objects with mu_q, Sigma_q, gauge fields
            omega_ij: Pre-computed transport operator (optional)
            active_mask: Optional (*spatial,) boolean. If provided, only compute
                        KL within True region (HUGE efficiency gain for sparse supports)

        Returns:
            (consensus_mask, kl_map):
                consensus_mask: (*spatial,) boolean array, True where KL < threshold
                kl_map: (*spatial,) float array, KL divergence at each point
        """
        # Get or compute transport operator
        if omega_ij is None:
            omega_ij = self._get_transport(agent_i, agent_j)

        spatial_shape = agent_i.mu_q.shape[:-1]
        K = agent_i.mu_q.shape[-1]

        # If no active_mask provided, compute overlap region
        if active_mask is None:
            # Default: compute everywhere (original behavior)
            active_mask = np.ones(spatial_shape, dtype=bool)

        n_active = np.sum(active_mask)

        # Fast path: if very few active points, use batch extraction
        if n_active < 0.3 * np.prod(spatial_shape) and n_active > 0:
            # Extract active points into batch arrays
            active_indices = np.argwhere(active_mask)

            # Extract mu and Sigma at active points
            mu_i_batch = np.array([agent_i.mu_q[tuple(idx)] for idx in active_indices])  # (N, K)
            Sigma_i_batch = np.array([agent_i.Sigma_q[tuple(idx)] for idx in active_indices])  # (N, K, K)
            mu_j_batch = np.array([agent_j.mu_q[tuple(idx)] for idx in active_indices])
            Sigma_j_batch = np.array([agent_j.Sigma_q[tuple(idx)] for idx in active_indices])

            # Extract transport at active points
            if omega_ij.ndim > 2:
                omega_batch = np.array([omega_ij[tuple(idx)] for idx in active_indices])  # (N, K, K)
            else:
                omega_batch = np.broadcast_to(omega_ij, (len(active_indices), K, K))

            # Transport agent_j's beliefs: batch einsum
            mu_j_transported = np.einsum('nij,nj->ni', omega_batch, mu_j_batch)
            Sigma_j_transported = np.einsum('nik,nkl,njl->nij', omega_batch, Sigma_j_batch, omega_batch)

            # Sanitize batch
            from math_utils.numerical_utils import sanitize_sigma
            Sigma_j_transported = sanitize_sigma(Sigma_j_transported, eps=1e-6)

            # Compute KL for batch
            kl_batch = kl_gaussian(mu_i_batch, Sigma_i_batch, mu_j_transported, Sigma_j_transported)

            if self.use_symmetric_kl:
                kl_reverse = kl_gaussian(mu_j_transported, Sigma_j_transported, mu_i_batch, Sigma_i_batch)
                kl_batch = (kl_batch + kl_reverse) / 2

            # Scatter back to full array
            kl_div = np.full(spatial_shape, np.inf, dtype=np.float32)
            for i, idx in enumerate(active_indices):
                kl_div[tuple(idx)] = kl_batch[i]

        else:
            # Original full-grid computation (for dense active regions)
            if omega_ij.ndim > 2:
                mu_j_transported = np.einsum('...ij,...j->...i', omega_ij, agent_j.mu_q)
                Sigma_j_transported = np.einsum('...ik,...kl,...jl->...ij', omega_ij, agent_j.Sigma_q, omega_ij)
            else:
                mu_j_transported = omega_ij @ agent_j.mu_q
                Sigma_j_transported = omega_ij @ agent_j.Sigma_q @ omega_ij.T

            from math_utils.numerical_utils import sanitize_sigma
            Sigma_j_transported = sanitize_sigma(Sigma_j_transported, eps=1e-6)

            kl_div = kl_gaussian(
                agent_i.mu_q, agent_i.Sigma_q,
                mu_j_transported, Sigma_j_transported
            )

            if self.use_symmetric_kl:
                kl_div_reverse = kl_gaussian(
                    mu_j_transported, Sigma_j_transported,
                    agent_i.mu_q, agent_i.Sigma_q
                )
                kl_div = (kl_div + kl_div_reverse) / 2

            # Set inactive points to inf
            if not np.all(active_mask):
                kl_div = np.where(active_mask, kl_div, np.inf)

        # Compute consensus mask pointwise
        consensus_mask = kl_div < self.belief_threshold

        return consensus_mask, kl_div

    def check_model_consensus(self,
                             agent_i, agent_j,
                             omega_ij: Optional[np.ndarray] = None) -> Tuple[bool, float]:
        """
        Check if agents have reached model consensus.
        
        Args:
            agent_i, agent_j: Agent objects with mu_p, Sigma_p, gauge fields
            omega_ij: Pre-computed transport operator (optional)
            
        Returns:
            (consensus_reached, kl_divergence)
        """
        # Get or compute transport operator
        if omega_ij is None:
            omega_ij = self._get_transport(agent_i, agent_j)

        # Transport agent_j's model to agent_i's frame
        # For spatial manifolds: use einsum for proper broadcasting
        if omega_ij.ndim > 2:
            mu_j_transported = np.einsum('...ij,...j->...i', omega_ij, agent_j.mu_p)
            Sigma_j_transported = np.einsum('...ik,...kl,...jl->...ij', omega_ij, agent_j.Sigma_p, omega_ij)
        else:
            # Point manifold case
            mu_j_transported = omega_ij @ agent_j.mu_p
            Sigma_j_transported = omega_ij @ agent_j.Sigma_p @ omega_ij.T

        # Sanitize transported covariance to ensure positive-definiteness
        from math_utils.numerical_utils import sanitize_sigma
        Sigma_j_transported = sanitize_sigma(Sigma_j_transported, eps=1e-6)
        
        # Compute KL divergence
        kl_div = kl_gaussian(
            agent_i.mu_p, agent_i.Sigma_p,
            mu_j_transported, Sigma_j_transported
        )

        if self.use_symmetric_kl:
            kl_div_reverse = kl_gaussian(
                mu_j_transported, Sigma_j_transported,
                agent_i.mu_p, agent_i.Sigma_p
            )
            kl_div = (kl_div + kl_div_reverse) / 2

        # Handle spatial manifolds: kl_div may be array (*spatial,)
        # For consensus, use MAXIMUM KL within OVERLAP region only
        if np.ndim(kl_div) > 0:
            # Compute overlap mask: where BOTH agents have support
            chi_i = agent_i.support.chi_weight
            chi_j = agent_j.support.chi_weight
            overlap_mask = (chi_i > 0.01) & (chi_j > 0.01)

            if np.any(overlap_mask):
                kl_in_overlap = kl_div[overlap_mask]
                kl_div_max = np.max(kl_in_overlap)
                kl_div_scalar = float(np.mean(kl_in_overlap))
            else:
                # No overlap - can't compute consensus
                kl_div_max = np.inf
                kl_div_scalar = np.inf

            consensus = kl_div_max < self.model_threshold
        else:
            kl_div_scalar = float(kl_div)
            consensus = kl_div < self.model_threshold

        return consensus, kl_div_scalar

    def check_model_consensus_spatial(self,
                                      agent_i, agent_j,
                                      omega_ij: Optional[np.ndarray] = None,
                                      active_mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Check model consensus **pointwise** on spatial manifolds.

        Args:
            agent_i, agent_j: Agent objects with mu_p, Sigma_p, gauge fields
            omega_ij: Pre-computed transport operator (optional)
            active_mask: Optional (*spatial,) boolean. If provided, only compute
                        KL within True region (efficiency optimization)

        Returns:
            (consensus_mask, kl_map):
                consensus_mask: (*spatial,) boolean array
                kl_map: (*spatial,) float array
        """
        # Get or compute transport operator
        if omega_ij is None:
            omega_ij = self._get_transport(agent_i, agent_j)

        spatial_shape = agent_i.mu_p.shape[:-1]
        K = agent_i.mu_p.shape[-1]

        # If no active_mask provided, compute everywhere
        if active_mask is None:
            active_mask = np.ones(spatial_shape, dtype=bool)

        n_active = np.sum(active_mask)

        # Fast path: sparse active region
        if n_active < 0.3 * np.prod(spatial_shape) and n_active > 0:
            active_indices = np.argwhere(active_mask)

            # Extract at active points
            mu_i_batch = np.array([agent_i.mu_p[tuple(idx)] for idx in active_indices])
            Sigma_i_batch = np.array([agent_i.Sigma_p[tuple(idx)] for idx in active_indices])
            mu_j_batch = np.array([agent_j.mu_p[tuple(idx)] for idx in active_indices])
            Sigma_j_batch = np.array([agent_j.Sigma_p[tuple(idx)] for idx in active_indices])

            if omega_ij.ndim > 2:
                omega_batch = np.array([omega_ij[tuple(idx)] for idx in active_indices])
            else:
                omega_batch = np.broadcast_to(omega_ij, (len(active_indices), K, K))

            # Transport
            mu_j_transported = np.einsum('nij,nj->ni', omega_batch, mu_j_batch)
            Sigma_j_transported = np.einsum('nik,nkl,njl->nij', omega_batch, Sigma_j_batch, omega_batch)

            from math_utils.numerical_utils import sanitize_sigma
            Sigma_j_transported = sanitize_sigma(Sigma_j_transported, eps=1e-6)

            kl_batch = kl_gaussian(mu_i_batch, Sigma_i_batch, mu_j_transported, Sigma_j_transported)

            if self.use_symmetric_kl:
                kl_reverse = kl_gaussian(mu_j_transported, Sigma_j_transported, mu_i_batch, Sigma_i_batch)
                kl_batch = (kl_batch + kl_reverse) / 2

            # Scatter back
            kl_div = np.full(spatial_shape, np.inf, dtype=np.float32)
            for i, idx in enumerate(active_indices):
                kl_div[tuple(idx)] = kl_batch[i]

        else:
            # Full grid computation
            if omega_ij.ndim > 2:
                mu_j_transported = np.einsum('...ij,...j->...i', omega_ij, agent_j.mu_p)
                Sigma_j_transported = np.einsum('...ik,...kl,...jl->...ij', omega_ij, agent_j.Sigma_p, omega_ij)
            else:
                mu_j_transported = omega_ij @ agent_j.mu_p
                Sigma_j_transported = omega_ij @ agent_j.Sigma_p @ omega_ij.T

            from math_utils.numerical_utils import sanitize_sigma
            Sigma_j_transported = sanitize_sigma(Sigma_j_transported, eps=1e-6)

            kl_div = kl_gaussian(
                agent_i.mu_p, agent_i.Sigma_p,
                mu_j_transported, Sigma_j_transported
            )

            if self.use_symmetric_kl:
                kl_div_reverse = kl_gaussian(
                    mu_j_transported, Sigma_j_transported,
                    agent_i.mu_p, agent_i.Sigma_p
                )
                kl_div = (kl_div + kl_div_reverse) / 2

            if not np.all(active_mask):
                kl_div = np.where(active_mask, kl_div, np.inf)

        consensus_mask = kl_div < self.model_threshold

        return consensus_mask, kl_div

    def check_full_consensus(self,
                            agent_i, agent_j) -> ConsensusState:
        """
        Check both belief and model consensus (epistemic death).
        
        Returns:
            ConsensusState with all consensus information
        """
        # Compute transport once for efficiency
        omega_ij = self._get_transport(agent_i, agent_j)
        
        # Check both types of consensus
        belief_consensus, belief_div = self.check_belief_consensus(
            agent_i, agent_j, omega_ij
        )
        model_consensus, model_div = self.check_model_consensus(
            agent_i, agent_j, omega_ij  
        )
        
        return ConsensusState(
            belief_consensus=belief_consensus,
            model_consensus=model_consensus,
            belief_divergence=belief_div,
            model_divergence=model_div
        )
    
    def find_consensus_clusters(self, system) -> List[List[int]]:
        """
        Find clusters of agents that have reached mutual consensus.
        
        Uses graph connectivity to find transitive consensus groups.
        
        Args:
            system: MultiAgentSystem object
            
        Returns:
            List of agent index clusters, e.g. [[0,1,2], [3,4], ...]
        """
        n_agents = system.n_agents
        
        # Build consensus adjacency matrix
        consensus_matrix = np.zeros((n_agents, n_agents), dtype=bool)
        
        for i in range(n_agents):
            for j in range(i+1, n_agents):
                # Check if i,j are in mutual consensus
                state_ij = self.check_full_consensus(
                    system.agents[i], system.agents[j]
                )
                
                if state_ij.is_epistemically_dead:
                    # Check reverse direction for mutual consensus
                    state_ji = self.check_full_consensus(
                        system.agents[j], system.agents[i]
                    )
                    if state_ji.is_epistemically_dead:
                        consensus_matrix[i, j] = True
                        consensus_matrix[j, i] = True
        
        # Find connected components (consensus clusters)
        sparse_matrix = csr_matrix(consensus_matrix)
        n_components, labels = connected_components(
            sparse_matrix, directed=False
        )
        
        # Group agents by cluster
        clusters = []
        for component_id in range(n_components):
            cluster = [i for i, label in enumerate(labels) if label == component_id]
            # Only return clusters with >1 agent (actual consensus groups)
            if len(cluster) > 1:
                clusters.append(cluster)
        
        return clusters
    
    def compute_consensus_matrix(self, system) -> np.ndarray:
        """
        Compute full consensus divergence matrix for analysis.
        
        Returns:
            Matrix where element (i,j) = KL(q_i || Ω_ij[q_j]) + KL(p_i || Ω_ij[p_j])
        """
        n_agents = system.n_agents
        consensus_matrix = np.zeros((n_agents, n_agents))
        
        for i in range(n_agents):
            for j in range(n_agents):
                if i == j:
                    continue
                    
                state = self.check_full_consensus(
                    system.agents[i], system.agents[j]
                )
                consensus_matrix[i, j] = (
                    state.belief_divergence + state.model_divergence
                )
        
        return consensus_matrix
    
    def identify_meta_agent_candidates(self, system,
                                      min_cluster_size: int = 2) -> List[Dict]:
        """
        Identify groups ready to form meta-agents.
        
        Returns:
            List of dicts with:
                - 'indices': constituent agent indices
                - 'belief_coherence': average belief consensus strength
                - 'model_coherence': average model consensus strength
                - 'scale': suggested hierarchical scale (0 for base agents)
        """
        clusters = self.find_consensus_clusters(system)
        
        candidates = []
        for cluster in clusters:
            if len(cluster) < min_cluster_size:
                continue
            
            # Compute average coherence within cluster
            belief_divs = []
            model_divs = []
            
            for i, idx_i in enumerate(cluster):
                for idx_j in cluster[i+1:]:
                    state = self.check_full_consensus(
                        system.agents[idx_i], system.agents[idx_j]
                    )
                    belief_divs.append(state.belief_divergence)
                    model_divs.append(state.model_divergence)
            
            candidates.append({
                'indices': cluster,
                'belief_coherence': 1.0 / (1.0 + np.mean(belief_divs)),
                'model_coherence': 1.0 / (1.0 + np.mean(model_divs)),
                'scale': 0  # Will be updated when hierarchies form
            })
        
        return candidates
    
    def _get_transport(self, agent_i, agent_j) -> np.ndarray:
        """Get or compute transport operator Ω_ij."""
        if self.cache_transport:
            key = (agent_i.agent_id, agent_j.agent_id)
            if key not in self._transport_cache:
                self._transport_cache[key] = compute_transport(
                    agent_i.gauge.phi,
                    agent_j.gauge.phi, 
                    agent_i.generators,
                    validate=False
                )
            return self._transport_cache[key]
        else:
            return compute_transport(
                agent_i.gauge.phi,
                agent_j.gauge.phi,
                agent_i.generators,
                validate=False
            )
    
    def clear_cache(self):
        """Clear transport operator cache."""
        if self._transport_cache:
            self._transport_cache.clear()


# =============================================================================
# Analysis Functions
# =============================================================================

def analyze_consensus_dynamics(system, detector: ConsensusDetector,
                              history: List[np.ndarray]) -> Dict:
    """
    Analyze how consensus evolves over time.
    
    Args:
        system: Current MultiAgentSystem state
        detector: ConsensusDetector instance
        history: List of consensus matrices over time
        
    Returns:
        Dictionary with analysis metrics
    """
    current_matrix = detector.compute_consensus_matrix(system)
    clusters = detector.find_consensus_clusters(system)
    
    # Compute metrics
    mean_divergence = np.mean(current_matrix[current_matrix > 0])
    min_divergence = np.min(current_matrix[current_matrix > 0]) if np.any(current_matrix > 0) else 0
    
    # Track convergence rate if history provided
    convergence_rate = 0
    if len(history) > 1:
        recent_change = np.mean(np.abs(current_matrix - history[-1]))
        old_change = np.mean(np.abs(history[-1] - history[-2])) if len(history) > 2 else recent_change
        convergence_rate = recent_change / (old_change + 1e-8)
    
    return {
        'n_clusters': len(clusters),
        'largest_cluster': max(len(c) for c in clusters) if clusters else 0,
        'mean_divergence': mean_divergence,
        'min_divergence': min_divergence,
        'convergence_rate': convergence_rate,
        'total_consensus_pairs': sum(len(c)*(len(c)-1)//2 for c in clusters),
        'clusters': clusters
    }


def visualize_consensus_matrix(consensus_matrix: np.ndarray,
                              clusters: Optional[List[List[int]]] = None) -> None:
    """
    Visualize consensus matrix as heatmap.
    (Requires matplotlib - skeleton for now)
    """
    try:
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(consensus_matrix, cmap='viridis_r', aspect='auto')
        ax.set_xlabel('Agent j')
        ax.set_ylabel('Agent i')
        ax.set_title('Consensus Matrix (KL Divergence after Transport)')
        plt.colorbar(im, ax=ax, label='KL Divergence')
        
        # Mark clusters if provided
        if clusters:
            for cluster in clusters:
                for i in cluster:
                    for j in cluster:
                        if i != j:
                            ax.add_patch(plt.Rectangle((j-0.4, i-0.4), 0.8, 0.8,
                                                     fill=False, edgecolor='red', linewidth=2))
        
        plt.tight_layout()
        plt.show()
    except ImportError:
        print("Matplotlib not available for visualization")