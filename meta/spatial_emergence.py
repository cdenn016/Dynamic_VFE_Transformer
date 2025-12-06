#!/usr/bin/env python3
"""
Spatial Emergence: Meta-Agent Formation in Arbitrary Dimensions
================================================================

Extends the 0D emergence framework to spatial manifolds of any dimension.
Meta-agents emerge as connected regions where constituent agents reach
consensus, with shapes determined by information topology rather than
predetermined geometry.

Key Principles:
--------------
1. Meta-agents are SMOOTH SECTIONS (same mathematical structure as base agents)
2. Shape is TOPOLOGICALLY EMERGENT from consensus geometry
3. SCALE-FREE: same mechanism works at any hierarchical level
4. INFORMATION-GEOMETRIC: KL divergence determines connectivity, not Euclidean distance

Algorithm:
---------
1. Find agent clusters via mutual consensus (existing machinery)
2. Compute SOFT consensus field: KL_cluster(c) = mean_{i<j} KL_ij(c)
3. Threshold to get consensus region: R = {c : KL_cluster(c) < ε}
4. SMOOTH the region boundary (preserves smooth section property)
5. Find CONNECTED COMPONENTS (dimension-agnostic)
6. Create meta-agents with COHERENCE-WEIGHTED support

Author:  Chris
Date: November 2025
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Union
from dataclasses import dataclass, field
from scipy.ndimage import label as find_connected_components
from scipy.ndimage import gaussian_filter, binary_dilation, generate_binary_structure

from geometry.geometry_base import BaseManifold, SupportRegion


# =============================================================================
# SECTION 1: Data Structures for Spatial Consensus
# =============================================================================

@dataclass
class SpatialConsensusRegion:
    """
    A connected region where a cluster of agents reaches consensus.

    This represents the geometric footprint of potential meta-agent emergence.
    Multiple SpatialConsensusRegions can arise from a single agent cluster
    if their consensus region is spatially disconnected.

    Attributes:
        cluster_indices: Indices of agents forming this consensus cluster
        component_id: Which connected component (0-indexed within cluster)
        component_mask: (*spatial,) boolean - this specific connected region
        consensus_field: (*spatial,) float - soft consensus strength (lower = better)
        smoothed_support: (*spatial,) float ∈ [0,1] - smoothed consensus weight
        volume: Number of points in the region
        volume_fraction: Fraction of manifold covered
        centroid: Center of mass coordinates
        coherence_scores: Per-agent coherence within this region
    """
    cluster_indices: List[int]
    component_id: int
    component_mask: np.ndarray
    consensus_field: np.ndarray
    smoothed_support: np.ndarray
    base_manifold: BaseManifold
    coherence_scores: Optional[np.ndarray] = None

    @property
    def volume(self) -> int:
        """Number of active points in region."""
        return int(np.sum(self.component_mask))

    @property
    def volume_fraction(self) -> float:
        """Fraction of manifold covered by this region."""
        return self.volume / self.base_manifold.n_points

    @property
    def centroid(self) -> Tuple[float, ...]:
        """Center of mass of the region (in grid coordinates)."""
        if self.base_manifold.is_point:
            return ()
        coords = np.argwhere(self.component_mask)
        if len(coords) == 0:
            # Fallback to manifold center
            return tuple(s / 2 for s in self.base_manifold.shape)
        return tuple(np.mean(coords, axis=0))

    @property
    def bounding_box(self) -> Tuple[Tuple[int, int], ...]:
        """Axis-aligned bounding box: ((min_0, max_0), (min_1, max_1), ...)."""
        if self.base_manifold.is_point:
            return ()
        coords = np.argwhere(self.component_mask)
        if len(coords) == 0:
            return tuple((0, s) for s in self.base_manifold.shape)
        mins = np.min(coords, axis=0)
        maxs = np.max(coords, axis=0)
        return tuple((int(mn), int(mx)) for mn, mx in zip(mins, maxs))

    @property
    def mean_consensus_strength(self) -> float:
        """Average KL divergence in the consensus region (lower = stronger consensus)."""
        if np.sum(self.component_mask) == 0:
            return np.inf
        return float(np.mean(self.consensus_field[self.component_mask]))

    def __repr__(self) -> str:
        return (
            f"SpatialConsensusRegion("
            f"agents={self.cluster_indices}, "
            f"component={self.component_id}, "
            f"volume={self.volume} ({self.volume_fraction:.1%}), "
            f"consensus={self.mean_consensus_strength:.4f})"
        )


# =============================================================================
# SECTION 2: Dimension-Agnostic Connectivity
# =============================================================================

def full_connectivity_kernel(ndim: int) -> np.ndarray:
    """
    Generate structuring element for full connectivity in any dimension.

    Full connectivity includes all diagonal neighbors, not just axis-aligned.
    This prevents artificial fragmentation of consensus regions.

    Args:
        ndim: Number of spatial dimensions

    Returns:
        structure: (3, 3, ..., 3) array of ones with shape (3,)*ndim

    Examples:
        1D: [1, 1, 1]                    # 2-connectivity (left-center-right)
        2D: [[1,1,1], [1,1,1], [1,1,1]]  # 8-connectivity (includes diagonals)
        3D: 3×3×3 cube of 1s             # 26-connectivity
        nD: 3^n hypercube of 1s          # Full (3^n - 1)-connectivity
    """
    if ndim == 0:
        return np.array(1)
    return np.ones((3,) * ndim, dtype=np.int32)


def axis_connectivity_kernel(ndim: int) -> np.ndarray:
    """
    Generate structuring element for axis-only connectivity.

    Only connects along coordinate axes (no diagonals).
    Stricter than full connectivity - may fragment more.

    Args:
        ndim: Number of spatial dimensions

    Returns:
        structure: Cross-shaped structuring element

    Examples:
        1D: [1, 1, 1]           # Same as full
        2D: [[0,1,0],           # 4-connectivity (no diagonals)
             [1,1,1],
             [0,1,0]]
        3D: 6-connectivity (faces only, no edges/corners)
    """
    if ndim == 0:
        return np.array(1)
    return generate_binary_structure(ndim, 1)


# =============================================================================
# SECTION 3: Soft Consensus Field Computation
# =============================================================================

def compute_cluster_consensus_field(
    cluster_indices: List[int],
    agents: List,  # List of Agent objects
    consensus_detector,  # ConsensusDetector instance
    check_models: bool = True,
    aggregation: str = 'mean'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute soft consensus field showing where cluster agents agree.

    For each point c on the manifold, computes aggregate KL divergence
    across all pairs of agents in the cluster.

    Args:
        cluster_indices: Indices of agents in this cluster
        agents: Full list of agent objects
        consensus_detector: ConsensusDetector for pairwise comparison
        check_models: Include model consensus (not just beliefs)
        aggregation: How to combine pairwise KL:
            'mean': Average over all pairs (default, soft)
            'max': Maximum over all pairs (strict, any disagreement kills consensus)
            'min': Minimum over all pairs (very soft, any agreement suffices)

    Returns:
        consensus_field: (*spatial,) float - aggregate KL at each point (lower = better)
        active_mask: (*spatial,) boolean - where cluster has support
    """
    cluster_agents = [agents[i] for i in cluster_indices]
    n_cluster = len(cluster_agents)

    if n_cluster < 2:
        raise ValueError("Cluster must have at least 2 agents")

    # Get spatial shape from first agent
    base_manifold = cluster_agents[0].base_manifold
    spatial_shape = base_manifold.shape

    # Handle 0D case
    if base_manifold.is_point:
        # Fall back to scalar consensus
        total_kl = 0.0
        n_pairs = 0
        for i in range(n_cluster):
            for j in range(i + 1, n_cluster):
                state = consensus_detector.check_full_consensus(
                    cluster_agents[i], cluster_agents[j]
                )
                total_kl += state.belief_divergence
                if check_models:
                    total_kl += state.model_divergence
                n_pairs += 1

        avg_kl = total_kl / n_pairs if n_pairs > 0 else 0.0
        return np.array(avg_kl), np.array(True)

    # Compute union of supports (active region)
    active_mask = np.zeros(spatial_shape, dtype=bool)
    for agent in cluster_agents:
        active_mask |= agent.support.get_mask_bool()

    # Collect pairwise KL fields
    kl_fields = []

    # Threshold for considering a point "active" in support
    support_threshold = 0.01

    for i in range(n_cluster):
        for j in range(i + 1, n_cluster):
            agent_i = cluster_agents[i]
            agent_j = cluster_agents[j]

            # Compute pairwise overlap mask (only compute KL where both have support)
            chi_i = agent_i.support.chi_weight
            chi_j = agent_j.support.chi_weight
            overlap_mask = (chi_i > support_threshold) & (chi_j > support_threshold)

            # Get transport operator
            omega_ij = consensus_detector._get_transport(agent_i, agent_j)

            # Belief KL field (only in overlap region)
            _, belief_kl = consensus_detector.check_belief_consensus_spatial(
                agent_i, agent_j, omega_ij=omega_ij, active_mask=overlap_mask
            )

            if check_models:
                # Model KL field (only in overlap region)
                _, model_kl = consensus_detector.check_model_consensus_spatial(
                    agent_i, agent_j, omega_ij=omega_ij, active_mask=overlap_mask
                )
                # Combined divergence
                pair_kl = belief_kl + model_kl
            else:
                pair_kl = belief_kl

            kl_fields.append(pair_kl)

    # Stack and aggregate
    kl_stack = np.stack(kl_fields, axis=-1)  # (*spatial, n_pairs)

    if aggregation == 'mean':
        consensus_field = np.mean(kl_stack, axis=-1)
    elif aggregation == 'max':
        consensus_field = np.max(kl_stack, axis=-1)
    elif aggregation == 'min':
        consensus_field = np.min(kl_stack, axis=-1)
    else:
        raise ValueError(f"Unknown aggregation: {aggregation}")

    return consensus_field, active_mask


def compute_coherence_scores_spatial(
    cluster_indices: List[int],
    agents: List,
    consensus_detector,
    region_mask: np.ndarray
) -> np.ndarray:
    """
    Compute per-agent coherence scores within a spatial region.

    Coherence measures how well each agent aligns with the cluster
    consensus within the specified region.

    C̄_i = exp(-mean_j KL(q_i || Ω_ij[q_j]))  averaged over region

    Args:
        cluster_indices: Indices of agents in cluster
        agents: Full list of agent objects
        consensus_detector: For pairwise comparison
        region_mask: (*spatial,) boolean - region to evaluate

    Returns:
        coherence_scores: (n_cluster,) float array, each in [0, 1]
    """
    cluster_agents = [agents[i] for i in cluster_indices]
    n_cluster = len(cluster_agents)

    coherence_scores = np.zeros(n_cluster)

    # Threshold for considering a point "active" in support
    support_threshold = 0.01

    for i, agent_i in enumerate(cluster_agents):
        # Average KL to all other agents in cluster
        total_kl = 0.0
        n_others = 0

        for j, agent_j in enumerate(cluster_agents):
            if i == j:
                continue

            # Compute pairwise overlap within region
            chi_i = agent_i.support.chi_weight
            chi_j = agent_j.support.chi_weight
            overlap_mask = (chi_i > support_threshold) & (chi_j > support_threshold) & region_mask

            omega_ij = consensus_detector._get_transport(agent_i, agent_j)
            _, kl_field = consensus_detector.check_belief_consensus_spatial(
                agent_i, agent_j, omega_ij=omega_ij, active_mask=overlap_mask
            )

            # Average KL within region
            if np.any(region_mask):
                regional_kl = np.mean(kl_field[region_mask])
            else:
                regional_kl = np.mean(kl_field)

            total_kl += regional_kl
            n_others += 1

        avg_kl = total_kl / n_others if n_others > 0 else 0.0
        coherence_scores[i] = np.exp(-avg_kl)

    return coherence_scores


# =============================================================================
# SECTION 4: Smooth Boundary Processing
# =============================================================================

def smooth_consensus_boundary(
    consensus_mask: np.ndarray,
    sigma: float = 1.5,
    threshold: float = 0.5
) -> np.ndarray:
    """
    Smooth binary consensus mask to ensure differentiable boundaries.

    This is CRITICAL for maintaining smooth section property of meta-agents.
    The Gaussian smoothing creates gradual transitions at region boundaries
    rather than sharp discontinuities.

    Args:
        consensus_mask: (*spatial,) boolean - raw consensus region
        sigma: Gaussian smoothing width (in grid units)
            - σ ~ 1-2 recommended for smooth transitions
            - σ = 0 gives hard boundaries (not smooth sections!)
        threshold: Threshold for final soft mask (0.5 = 50% level set)

    Returns:
        smoothed: (*spatial,) float ∈ [0,1] - smooth support weight
    """
    if consensus_mask.ndim == 0:
        # 0D: no smoothing needed
        return consensus_mask.astype(np.float32)

    if sigma <= 0:
        # No smoothing - hard boundaries
        return consensus_mask.astype(np.float32)

    # Convert to float for smoothing
    mask_float = consensus_mask.astype(np.float64)

    # Gaussian blur creates smooth transitions
    smoothed = gaussian_filter(mask_float, sigma=sigma, mode='nearest')

    # Normalize to [0, 1]
    smoothed = np.clip(smoothed, 0, 1)

    return smoothed.astype(np.float32)


def create_smooth_consensus_support(
    consensus_mask: np.ndarray,
    sigma: float = 1.5,
    boundary_falloff: str = 'gaussian'
) -> np.ndarray:
    """
    Create smooth support weight from consensus mask.

    Provides multiple options for how boundaries transition.

    Args:
        consensus_mask: (*spatial,) boolean - raw consensus region
        sigma: Smoothing width
        boundary_falloff: Type of boundary transition:
            'gaussian': Gaussian blur (default, most physical)
            'sigmoid': Sigmoid-based smooth step
            'linear': Linear interpolation to boundary

    Returns:
        support_weight: (*spatial,) float ∈ [0,1]
    """
    if consensus_mask.ndim == 0:
        return consensus_mask.astype(np.float32)

    if boundary_falloff == 'gaussian':
        return smooth_consensus_boundary(consensus_mask, sigma=sigma)

    elif boundary_falloff == 'sigmoid':
        # Distance transform + sigmoid
        from scipy.ndimage import distance_transform_edt

        # Signed distance: positive inside, negative outside
        dist_inside = distance_transform_edt(consensus_mask)
        dist_outside = distance_transform_edt(~consensus_mask)
        signed_dist = dist_inside - dist_outside

        # Sigmoid transition
        support = 1.0 / (1.0 + np.exp(-signed_dist / sigma))
        return support.astype(np.float32)

    elif boundary_falloff == 'linear':
        # Distance-based linear falloff
        from scipy.ndimage import distance_transform_edt

        dist_outside = distance_transform_edt(~consensus_mask)
        # Linear falloff over sigma distance
        support = np.where(
            consensus_mask,
            1.0,
            np.maximum(0, 1.0 - dist_outside / (3 * sigma))
        )
        return support.astype(np.float32)

    else:
        raise ValueError(f"Unknown boundary_falloff: {boundary_falloff}")


# =============================================================================
# SECTION 5: Connected Component Analysis
# =============================================================================

def find_consensus_components(
    consensus_field: np.ndarray,
    active_mask: np.ndarray,
    kl_threshold: float,
    min_region_size: int = 4,
    min_volume_fraction: float = 0.0,
    connectivity: str = 'full',
    smooth_sigma: float = 1.5
) -> List[Dict]:
    """
    Find connected components in thresholded consensus field.

    This is the core spatial analysis: given the soft consensus field,
    identify distinct connected regions that could become meta-agents.

    Args:
        consensus_field: (*spatial,) float - aggregate KL (lower = better)
        active_mask: (*spatial,) boolean - where agents have support
        kl_threshold: Maximum KL for consensus
        min_region_size: Minimum absolute size (points)
        min_volume_fraction: Minimum relative size (fraction of manifold)
        connectivity: 'full' (includes diagonals) or 'axis' (no diagonals)
        smooth_sigma: Smoothing for boundaries

    Returns:
        List of dicts with:
            'component_id': int
            'mask': (*spatial,) boolean
            'smoothed': (*spatial,) float ∈ [0,1]
            'size': int
            'volume_fraction': float
    """
    # Handle 0D case
    if consensus_field.ndim == 0:
        is_consensus = float(consensus_field) < kl_threshold
        return [{
            'component_id': 0,
            'mask': np.array(is_consensus),
            'smoothed': np.array(1.0 if is_consensus else 0.0, dtype=np.float32),
            'size': 1 if is_consensus else 0,
            'volume_fraction': 1.0 if is_consensus else 0.0
        }] if is_consensus else []

    # Threshold consensus field within active region
    consensus_mask = (consensus_field < kl_threshold) & active_mask

    # Get connectivity structure
    ndim = consensus_field.ndim
    if connectivity == 'full':
        structure = full_connectivity_kernel(ndim)
    else:
        structure = axis_connectivity_kernel(ndim)

    # Find connected components
    labeled, n_components = find_connected_components(consensus_mask, structure=structure)

    total_volume = consensus_field.size

    # Process each component
    components = []
    for comp_id in range(1, n_components + 1):  # Labels start at 1
        comp_mask = (labeled == comp_id)
        comp_size = int(np.sum(comp_mask))
        volume_frac = comp_size / total_volume

        # Check size thresholds
        if comp_size < min_region_size:
            continue
        if volume_frac < min_volume_fraction:
            continue

        # Smooth the boundary
        smoothed = smooth_consensus_boundary(comp_mask, sigma=smooth_sigma)

        components.append({
            'component_id': comp_id - 1,  # 0-indexed for output
            'mask': comp_mask,
            'smoothed': smoothed,
            'size': comp_size,
            'volume_fraction': volume_frac
        })

    return components


# =============================================================================
# SECTION 6: Main Emergence Detection Class
# =============================================================================

class SpatialEmergenceDetector:
    """
    Detect meta-agent emergence with spatial coherence in arbitrary dimensions.

    This class orchestrates the full pipeline from agent clustering to
    identification of connected consensus regions suitable for meta-agent
    formation.

    Theory:
    ------
    Meta-agents emerge where constituents reach epistemic death (consensus on
    both beliefs and models). In spatial manifolds, this creates REGIONS of
    consensus that may be irregularly shaped. The shape is EMERGENT from
    information topology, not predetermined.

    The resulting meta-agents are SMOOTH SECTIONS over their support regions,
    maintaining the gauge-theoretic structure at all hierarchical scales.
    """

    def __init__(
        self,
        consensus_detector,
        kl_threshold: float = 0.1,
        min_region_size: int = 4,
        min_volume_fraction: float = 0.0,
        smooth_sigma: float = 1.5,
        connectivity: str = 'full',
        aggregation: str = 'mean',
        check_models: bool = True
    ):
        """
        Initialize spatial emergence detector.

        Args:
            consensus_detector: ConsensusDetector instance for pairwise comparison
            kl_threshold: Maximum KL divergence for consensus
            min_region_size: Minimum region size in points (absolute)
            min_volume_fraction: Minimum region size as fraction (relative)
            smooth_sigma: Gaussian smoothing width for boundaries
            connectivity: 'full' or 'axis' connectivity for components
            aggregation: How to combine pairwise KL ('mean', 'max', 'min')
            check_models: Include model consensus (not just beliefs)
        """
        self.consensus_detector = consensus_detector
        self.kl_threshold = kl_threshold
        self.min_region_size = min_region_size
        self.min_volume_fraction = min_volume_fraction
        self.smooth_sigma = smooth_sigma
        self.connectivity = connectivity
        self.aggregation = aggregation
        self.check_models = check_models

    def detect_emergence_regions(
        self,
        system,  # MultiAgentSystem or MultiScaleSystem
        source_scale: int = 0
    ) -> List[SpatialConsensusRegion]:
        """
        Full pipeline: detect all consensus regions suitable for meta-agent formation.

        Args:
            system: Multi-agent system with agents at source_scale
            source_scale: Which hierarchical scale to analyze (0 = base agents)

        Returns:
            List of SpatialConsensusRegion objects, each representing a
            potential meta-agent with its spatial footprint
        """
        # Get agents at source scale
        if hasattr(system, 'get_active_agents_at_scale'):
            # MultiScaleSystem
            agents = system.get_active_agents_at_scale(source_scale)
            agent_list = list(agents.values()) if isinstance(agents, dict) else agents
        else:
            # Plain MultiAgentSystem
            agent_list = system.agents

        if len(agent_list) < 2:
            return []

        # Get base manifold
        base_manifold = agent_list[0].base_manifold

        # Step 1: Find agent clusters via consensus
        clusters = self.consensus_detector.find_consensus_clusters(system)

        if not clusters:
            return []

        # Step 2-5: Process each cluster
        all_regions = []

        for cluster_indices in clusters:
            regions = self._process_cluster(
                cluster_indices, agent_list, base_manifold
            )
            all_regions.extend(regions)

        return all_regions

    def _process_cluster(
        self,
        cluster_indices: List[int],
        agents: List,
        base_manifold: BaseManifold
    ) -> List[SpatialConsensusRegion]:
        """
        Process a single agent cluster to find consensus regions.

        Args:
            cluster_indices: Indices of agents in this cluster
            agents: Full list of agent objects
            base_manifold: Shared spatial manifold

        Returns:
            List of SpatialConsensusRegion objects for this cluster
        """
        # Compute soft consensus field
        consensus_field, active_mask = compute_cluster_consensus_field(
            cluster_indices=cluster_indices,
            agents=agents,
            consensus_detector=self.consensus_detector,
            check_models=self.check_models,
            aggregation=self.aggregation
        )

        # Find connected components
        components = find_consensus_components(
            consensus_field=consensus_field,
            active_mask=active_mask,
            kl_threshold=self.kl_threshold,
            min_region_size=self.min_region_size,
            min_volume_fraction=self.min_volume_fraction,
            connectivity=self.connectivity,
            smooth_sigma=self.smooth_sigma
        )

        # Create SpatialConsensusRegion for each component
        regions = []
        for comp in components:
            # Compute coherence scores within this region
            coherence_scores = compute_coherence_scores_spatial(
                cluster_indices=cluster_indices,
                agents=agents,
                consensus_detector=self.consensus_detector,
                region_mask=comp['mask']
            )

            region = SpatialConsensusRegion(
                cluster_indices=cluster_indices,
                component_id=comp['component_id'],
                component_mask=comp['mask'],
                consensus_field=consensus_field,
                smoothed_support=comp['smoothed'],
                base_manifold=base_manifold,
                coherence_scores=coherence_scores
            )
            regions.append(region)

        return regions

    def create_meta_agent_support(
        self,
        region: SpatialConsensusRegion,
        agents: List,
        weight_by_coherence: bool = True
    ) -> SupportRegion:
        """
        Create support region for meta-agent from consensus region.

        The meta-agent support is a coherence-weighted combination of
        constituent supports, restricted to and smoothed by the consensus region.

        χ_M(c) = R̃(c) · [Σᵢ Cᵢ·χᵢ(c)] / [Σᵢ Cᵢ·χᵢ(c)]_max

        where:
            R̃(c) = smoothed consensus region
            Cᵢ = coherence score of agent i
            χᵢ(c) = support of agent i

        Args:
            region: SpatialConsensusRegion from detection
            agents: Full list of agent objects
            weight_by_coherence: If True, weight by coherence scores

        Returns:
            SupportRegion for the meta-agent
        """
        cluster_agents = [agents[i] for i in region.cluster_indices]
        base_manifold = region.base_manifold

        # Handle 0D case
        if base_manifold.is_point:
            from geometry.geometry_base import create_full_support
            return create_full_support(base_manifold)

        spatial_shape = base_manifold.shape

        # Get coherence weights
        if weight_by_coherence and region.coherence_scores is not None:
            weights = region.coherence_scores
        else:
            weights = np.ones(len(cluster_agents))

        # Normalize weights
        weights = weights / (np.sum(weights) + 1e-12)

        # Compute weighted sum of constituent supports
        chi_combined = np.zeros(spatial_shape, dtype=np.float32)

        for agent, w in zip(cluster_agents, weights):
            chi_i = agent.support.chi_weight
            chi_combined += w * chi_i

        # Mask by smoothed consensus region
        chi_meta = chi_combined * region.smoothed_support

        # Normalize to [0, 1]
        max_chi = np.max(chi_meta)
        if max_chi > 1e-12:
            chi_meta = chi_meta / max_chi

        # Create support region
        support = SupportRegion(
            base_manifold=base_manifold,
            chi_weight=chi_meta
        )

        return support


# =============================================================================
# SECTION 7: Utility Functions
# =============================================================================

def visualize_consensus_regions_nd(
    regions: List[SpatialConsensusRegion],
    agents: Optional[List] = None,
    slice_dims: Optional[Tuple[int, int]] = None,
    slice_index: Optional[int] = None,
    save_path: Optional[str] = None
):
    """
    Visualize consensus regions (supports arbitrary dimensions via slicing).

    For dimensions > 2, takes a 2D slice through the data.

    Args:
        regions: List of SpatialConsensusRegion objects
        agents: Optional list of agents to overlay supports
        slice_dims: Which two dimensions to visualize (for ndim > 2)
        slice_index: Index along non-visualized dimensions
        save_path: Where to save figure (optional)
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    if not regions:
        print("No regions to visualize")
        return

    base_manifold = regions[0].base_manifold
    ndim = base_manifold.ndim

    if ndim == 0:
        print("Cannot visualize 0D manifold")
        return

    # For 1D: plot as line
    if ndim == 1:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Consensus field
        ax = axes[0]
        for i, region in enumerate(regions):
            x = np.arange(len(region.consensus_field))
            ax.plot(x, region.consensus_field, label=f'Cluster {i}')
            ax.fill_between(x, 0, region.smoothed_support * np.max(region.consensus_field),
                          alpha=0.3, label=f'Region {i}')
        ax.set_xlabel('Position')
        ax.set_ylabel('Consensus Field (KL)')
        ax.legend()
        ax.set_title('1D Consensus Regions')

        # Smoothed supports
        ax = axes[1]
        for i, region in enumerate(regions):
            ax.plot(region.smoothed_support, label=f'Region {i}')
        ax.set_xlabel('Position')
        ax.set_ylabel('Smoothed Support χ_M')
        ax.legend()
        ax.set_title('Meta-Agent Supports')

    # For 2D: plot as images
    elif ndim == 2:
        n_regions = len(regions)
        fig, axes = plt.subplots(2, n_regions, figsize=(4 * n_regions, 8))
        if n_regions == 1:
            axes = axes.reshape(2, 1)

        for i, region in enumerate(regions):
            # Consensus field
            ax = axes[0, i]
            im = ax.imshow(region.consensus_field.T, origin='lower', cmap='viridis_r')
            ax.contour(region.component_mask.T, levels=[0.5], colors='red', linewidths=2)
            ax.set_title(f'Cluster {region.cluster_indices}\nKL Field')
            plt.colorbar(im, ax=ax, label='KL')

            # Smoothed support
            ax = axes[1, i]
            ax.imshow(region.smoothed_support.T, origin='lower', cmap='Greys', vmin=0, vmax=1)
            ax.set_title(f'Smoothed Support\n{region.volume} pts ({region.volume_fraction:.1%})')

        plt.suptitle('2D Spatial Consensus Regions')

    # For higher dimensions: slice
    else:
        if slice_dims is None:
            slice_dims = (0, 1)  # Default: first two dimensions
        if slice_index is None:
            slice_index = base_manifold.shape[2] // 2 if ndim > 2 else 0

        print(f"Visualizing {ndim}D data: showing slice at dim[2]={slice_index}")

        # Take slice
        # This is simplified - proper implementation would handle arbitrary slices
        fig, axes = plt.subplots(1, len(regions), figsize=(4 * len(regions), 4))
        if len(regions) == 1:
            axes = [axes]

        for i, region in enumerate(regions):
            ax = axes[i]
            # Slice the smoothed support
            slicer = [slice(None)] * ndim
            for d in range(2, ndim):
                slicer[d] = slice_index
            sliced = region.smoothed_support[tuple(slicer)]

            ax.imshow(sliced.T, origin='lower', cmap='Greys', vmin=0, vmax=1)
            ax.set_title(f'Region {i} (slice)')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def summarize_emergence_detection(regions: List[SpatialConsensusRegion]) -> Dict:
    """
    Generate summary statistics for detected emergence regions.

    Args:
        regions: List of detected SpatialConsensusRegion objects

    Returns:
        Dictionary with summary statistics
    """
    if not regions:
        return {
            'n_regions': 0,
            'total_volume': 0,
            'mean_consensus_strength': np.inf,
            'clusters': []
        }

    return {
        'n_regions': len(regions),
        'total_volume': sum(r.volume for r in regions),
        'total_volume_fraction': sum(r.volume_fraction for r in regions),
        'mean_consensus_strength': np.mean([r.mean_consensus_strength for r in regions]),
        'region_sizes': [r.volume for r in regions],
        'clusters': [r.cluster_indices for r in regions],
        'centroids': [r.centroid for r in regions]
    }