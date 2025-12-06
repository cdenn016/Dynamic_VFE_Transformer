# -*- coding: utf-8 -*-
"""
Created on Mon Nov 10 08:33:05 2025

@author: chris and christine
"""

# -*- coding: utf-8 -*-
"""
Enhanced μ(c) Tracking at Agent Centers
========================================

Tracks the evolution of belief means μ_q(c) at agent centers to study:
1. Gauge symmetry in vacuum theory (no observations):
   - All agents should have same ||μ|| but different components
   - Agents explore the gauge orbit
   
2. Symmetry breaking with observations:
   - ||μ|| becomes different across agents
   - System locks to specific gauge choice

Date: November 2025
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass, field


# =============================================================================
# Center Finding Utilities
# =============================================================================

def find_support_center(agent) -> Optional[Tuple[int, ...]]:
    """
    Find the center point of an agent's support.
    
    Strategy:
    1. For circular supports with stored center, use that
    2. Otherwise, use center of mass of support mask
    3. For 0D agents, return empty tuple
    
    Returns:
        center_idx: Spatial index tuple, or None if agent has no support
    """
    if not hasattr(agent, 'support') or agent.support is None:
        return None
    
    support = agent.support
    
    # 0D case: trivial center
    if hasattr(agent, 'base_manifold') and agent.base_manifold.is_point:
        return ()
    
    # Try to use stored center from circular support
    if hasattr(support, 'center') and support.center is not None:
        # Round to nearest integer indices
        center = tuple(int(round(c)) for c in support.center)
        return center
    
    # Fallback: center of mass of support mask
    if hasattr(support, 'mask_continuous'):
        mask = support.mask_continuous
    elif hasattr(support, 'mask'):
        mask = support.mask.astype(float)
    else:
        return None
    
    if mask.sum() == 0:
        return None
    
    # Compute center of mass
    ndim = mask.ndim
    coords = np.meshgrid(*[np.arange(s) for s in mask.shape], indexing='ij')
    
    center = []
    for dim in range(ndim):
        center_coord = (coords[dim] * mask).sum() / mask.sum()
        center.append(int(round(center_coord)))
    
    return tuple(center)


def extract_mu_at_center(agent) -> Optional[np.ndarray]:
    """
    Extract μ_q(c) at agent's center point.
    
    Returns:
        mu_center: μ_q values at center (shape: (K,)) or None
    """
    center_idx = find_support_center(agent)
    
    if center_idx is None:
        return None
    
    mu_q = agent.mu_q
    
    # 0D case: mu_q is already just (K,)
    if len(center_idx) == 0:
        return mu_q
    
    # Spatial case: extract mu_q[center_idx]
    try:
        mu_center = mu_q[center_idx]
        return mu_center
    except (IndexError, ValueError):
        return None


# =============================================================================
# Enhanced Tracking Data Structure
# =============================================================================

@dataclass
class MuCenterTracking:
    """
    Tracks μ(c) components and norms at agent centers over time.
    
    For each agent, tracks:
        - μ components: (n_steps, K) array
        - ||μ|| norm: (n_steps,) array
    
    This enables studying:
        - Gauge orbit exploration (vacuum theory)
        - Symmetry breaking (with observations)
    """
    
    n_agents: int
    K: int  # Latent dimension
    
    # Storage: List of arrays, one per agent
    # Each array has shape (n_steps, K) for components
    mu_components: List[List[np.ndarray]] = field(default_factory=list)
    mu_norms: List[List[float]] = field(default_factory=list)
    
    steps: List[int] = field(default_factory=list)
    
    def __post_init__(self):
        """Initialize storage for all agents."""
        self.mu_components = [[] for _ in range(self.n_agents)]
        self.mu_norms = [[] for _ in range(self.n_agents)]
    
    def record(self, step: int, system):
        """
        Record μ(c) at centers for all agents at current step.
        
        Args:
            step: Current training step
            system: MultiAgentSystem instance
        """
        self.steps.append(step)
        
        for i, agent in enumerate(system.agents):
            mu_center = extract_mu_at_center(agent)
            
            if mu_center is not None:
                # Store components
                self.mu_components[i].append(mu_center.copy())
                
                # Store norm
                norm = np.linalg.norm(mu_center)
                self.mu_norms[i].append(norm)
            else:
                # No valid center - store NaN
                self.mu_components[i].append(np.full(self.K, np.nan))
                self.mu_norms[i].append(np.nan)
    
    def get_component_array(self, agent_idx: int) -> np.ndarray:
        """
        Get μ components over time for one agent.
        
        Returns:
            components: (n_steps, K) array
        """
        return np.array(self.mu_components[agent_idx])
    
    def get_norm_array(self, agent_idx: int) -> np.ndarray:
        """
        Get ||μ|| over time for one agent.
        
        Returns:
            norms: (n_steps,) array
        """
        return np.array(self.mu_norms[agent_idx])
    
    def get_all_norms(self) -> np.ndarray:
        """
        Get ||μ|| over time for all agents.
        
        Returns:
            norms: (n_agents, n_steps) array
        """
        return np.array(self.mu_norms)
    
    def compute_norm_variance(self) -> np.ndarray:
        """
        Compute variance of ||μ|| across agents over time.
        
        This is a key diagnostic:
        - Vacuum theory (no obs): should stay near 0 (all same norm)
        - With observations: should increase (norms diverge)
        
        Returns:
            var_norm: (n_steps,) array of norm variances
        """
        norms = self.get_all_norms()  # (n_agents, n_steps)
        return np.var(norms, axis=0)
    
    def compute_mean_norm(self) -> np.ndarray:
        """
        Compute mean ||μ|| across agents over time.
        
        Returns:
            mean_norm: (n_steps,) array
        """
        norms = self.get_all_norms()  # (n_agents, n_steps)
        return np.mean(norms, axis=0)


# =============================================================================
# Integration with Trainer
# =============================================================================

def create_mu_tracker(system) -> MuCenterTracking:
    """
    Create a MuCenterTracking instance for a system.
    
    Args:
        system: MultiAgentSystem
    
    Returns:
        tracker: MuCenterTracking instance
    """
    n_agents = system.n_agents
    K = system.agents[0].config.K
    
    return MuCenterTracking(n_agents=n_agents, K=K)


# =============================================================================
# Usage Example
# =============================================================================

if __name__ == "__main__":
    print("="*70)
    print("ENHANCED μ(c) TRACKING MODULE")
    print("="*70)
    
    print("\n✅ Features:")
    print("  • Track μ_q(c) components at agent centers")
    print("  • Track ||μ_q(c)|| norms")
    print("  • Compute norm variance across agents (symmetry measure)")
    print("  • Compatible with 0D, 1D, 2D agents")
    
    print("\n✅ Key Physics:")
    print("  • Vacuum theory: Var(||μ||) ≈ 0 (gauge orbit)")
    print("  • With observations: Var(||μ||) > 0 (symmetry breaking)")
    
    print("\n✅ Usage:")
    print("  tracker = create_mu_tracker(system)")
    print("  # In training loop:")
    print("  tracker.record(step, system)")
    print("  # Analysis:")
    print("  var_norm = tracker.compute_norm_variance()")
    print("="*70)