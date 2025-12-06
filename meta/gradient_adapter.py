# -*- coding: utf-8 -*-
"""
Created on Wed Nov 19 21:20:57 2025

@author: chris and christine
"""

"""
Gradient System Adapter for Hierarchical Multi-Scale Systems

Provides a minimal adapter to make MultiScaleSystem compatible with
the gradient engine without re-initializing agent state.

This adapter:
- Exposes active agents as a flat list
- Provides spatial overlap information
- Computes transport operators between agents
- Preserves agent state (no re-initialization!)
"""

import numpy as np
from typing import List, Dict, Tuple


class GradientSystemAdapter:
    """
    Adapter to make MultiScaleSystem compatible with gradient engine.

    Provides the interface needed by compute_natural_gradients WITHOUT
    re-initializing agents (which would corrupt their state).

    CRITICAL: Respects spatial overlaps to match standard training behavior!
    """

    def __init__(self, agents_list: List, system_config):
        """
        Create adapter for gradient computation.

        Args:
            agents_list: List of active agents (from get_all_active_agents())
            system_config: SystemConfig with energy weights and parameters
        """
        from math_utils.transport import compute_transport

        self.agents = agents_list
        self.config = system_config
        self.n_agents = len(agents_list)
        self._compute_transport = compute_transport

        # Compute overlap relationships once (lightweight check)
        self._overlaps = self._compute_overlaps()

    def _compute_overlaps(self) -> Dict[Tuple[int, int], bool]:
        """
        Compute which agent pairs spatially overlap.

        Returns:
            Dict mapping (i, j) -> bool for overlap status
        """
        overlaps = {}

        # Check if we have a point manifold (all agents overlap)
        is_point_manifold = self._is_point_manifold()

        if is_point_manifold:
            # Point manifold: all agents overlap
            for i in range(self.n_agents):
                for j in range(self.n_agents):
                    if i != j:
                        overlaps[(i, j)] = True
        else:
            # Spatial manifold: check actual overlaps
            overlaps = self._compute_spatial_overlaps()

        return overlaps

    def _is_point_manifold(self) -> bool:
        """Check if agents exist on a point manifold (0-dimensional)."""
        if self.n_agents == 0:
            return False

        agent = self.agents[0]
        if hasattr(agent, 'base_manifold') and hasattr(agent.base_manifold, 'shape'):
            return agent.base_manifold.shape == ()

        return False

    def _compute_spatial_overlaps(self) -> Dict[Tuple[int, int], bool]:
        """
        Compute overlaps for spatial manifolds using support masks.

        CRITICAL: Matches MultiAgentSystem's two-check overlap logic:
        1. Upper bound check (max_i * max_j)
        2. Actual overlap check (max of products)
        """
        overlaps = {}
        overlap_threshold = getattr(self.config, 'overlap_threshold', 1e-3)

        for i in range(self.n_agents):
            for j in range(self.n_agents):
                if i == j:
                    continue

                agent_i = self.agents[i]
                agent_j = self.agents[j]

                # Check if both have supports
                if not (hasattr(agent_i, 'support') and hasattr(agent_j, 'support')):
                    # No support info - assume overlap
                    overlaps[(i, j)] = True
                    continue

                if agent_i.support is None or agent_j.support is None:
                    # Missing support - assume overlap
                    overlaps[(i, j)] = True
                    continue

                # Get masks
                chi_i = self._get_mask(agent_i.support)
                chi_j = self._get_mask(agent_j.support)

                if chi_i is None or chi_j is None:
                    # No mask - assume overlap
                    overlaps[(i, j)] = True
                    continue

                # CRITICAL: Match MultiAgentSystem's two-check logic
                # Check 1: Upper bound (product of maxes)
                max_overlap = np.max(chi_i) * np.max(chi_j)
                if max_overlap < overlap_threshold:
                    overlaps[(i, j)] = False
                    continue

                # Check 2: Actual overlap (max of products)
                chi_ij = chi_i * chi_j
                has_overlap = np.max(chi_ij) >= overlap_threshold
                overlaps[(i, j)] = has_overlap

        return overlaps

    @staticmethod
    def _get_mask(support):
        """Extract continuous mask from support (tries multiple attribute names)."""
        # Try different attribute names
        for attr in ['mask_continuous', 'chi_weight', 'mask']:
            if hasattr(support, attr):
                mask = getattr(support, attr)
                if mask is not None:
                    return mask
        return None

    def get_neighbors(self, agent_idx: int) -> List[int]:
        """
        Return agents that spatially overlap with given agent.

        Matches MultiAgentSystem.get_neighbors() behavior.

        Args:
            agent_idx: Index of agent to get neighbors for

        Returns:
            List of neighbor indices
        """
        neighbors = []
        for j in range(self.n_agents):
            # CRITICAL: Default to False (no overlap) like MultiAgentSystem.has_overlap
            if j != agent_idx and self._overlaps.get((agent_idx, j), False):
                neighbors.append(j)
        return neighbors

    def compute_transport_ij(self, i: int, j: int):
        """
        Compute transport operator Ω_ij = exp(φ_i) exp(-φ_j).

        Args:
            i: Source agent index
            j: Target agent index

        Returns:
            Transport operator (K x K matrix)
        """
        agent_i = self.agents[i]
        agent_j = self.agents[j]
        return self._compute_transport(
            agent_i.gauge.phi,
            agent_j.gauge.phi,
            agent_i.generators,
            validate=False
        )


def create_gradient_adapter(system):
    """
    Create GradientSystemAdapter for a MultiScaleSystem.

    Convenience function that handles getting active agents.

    Args:
        system: MultiScaleSystem instance

    Returns:
        GradientSystemAdapter ready for gradient computation
    """
    active_agents = system.get_all_active_agents()
    return GradientSystemAdapter(active_agents, system.system_config)