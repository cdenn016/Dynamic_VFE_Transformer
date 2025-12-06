#!/usr/bin/env python3
"""
Geometry Tracker: Monitor Emergent Geometry During Simulation

Integrates pullback geometry tools into simulation loop to track:
- Epistemic geometry (G^q from beliefs) evolution
- Ontological geometry (G^p from priors) evolution
- Consensus geometry from multiple agents
- Eigenvalue sector decomposition (observable/dark/internal)
- Information-geometric curvature

Usage:
    tracker = GeometryTracker(agents, track_interval=10, dx=0.1)

    # In training loop:
    for step in range(n_steps):
        # ... gradient updates ...
        tracker.record(step, agents)

    # After training:
    tracker.save(output_dir / "geometry_history.pkl")
    tracker.plot_evolution(output_dir / "geometry_evolution.png")

Author: Chris & Christine
Date: November 2025
"""

import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Union
from dataclasses import dataclass, field
import pickle
import matplotlib.pyplot as plt

from geometry.pullback_metrics import (
    agent_induced_metrics,
    InducedMetric,
    pullback_metric_gaussian,
    pullback_metric_gaussian_isotropic
)
from geometry.gauge_consensus import (
    compute_consensus_metric,
    ConsensusMetric,
    gauge_average_metric_mc
)


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class GeometrySnapshot:
    """Single timestep snapshot of emergent geometry."""

    step: int

    # Per-agent metrics
    agent_metrics_belief: List[InducedMetric] = field(default_factory=list)
    agent_metrics_prior: List[InducedMetric] = field(default_factory=list)

    # Consensus metrics
    consensus_belief: Optional[ConsensusMetric] = None
    consensus_prior: Optional[ConsensusMetric] = None

    # Summary statistics (for efficient plotting)
    mean_eigenvalues_belief: Optional[np.ndarray] = None  # (n_dims,)
    mean_eigenvalues_prior: Optional[np.ndarray] = None

    # Sector dimensions (averaged over agents)
    mean_n_observable: float = 0.0
    mean_n_dark: float = 0.0
    mean_n_internal: float = 0.0

    # Volume elements (det(G)^{1/2})
    mean_volume_belief: float = 0.0
    mean_volume_prior: float = 0.0


@dataclass
class GeometryHistory:
    """Full history of geometry evolution."""

    snapshots: List[GeometrySnapshot] = field(default_factory=list)

    # Metadata
    n_agents: int = 0
    spatial_shape: tuple = ()
    K_latent: int = 0
    dx: float = 1.0
    track_interval: int = 1

    # Options
    enable_consensus: bool = True
    enable_gauge_averaging: bool = False

    def get_steps(self) -> np.ndarray:
        """Get array of timesteps."""
        return np.array([s.step for s in self.snapshots])

    def get_mean_observable_dims(self) -> np.ndarray:
        """Get average number of observable dimensions over time."""
        return np.array([s.mean_n_observable for s in self.snapshots])

    def get_mean_volume_belief(self) -> np.ndarray:
        """Get average belief geometry volume over time."""
        return np.array([s.mean_volume_belief for s in self.snapshots])

    def get_mean_volume_prior(self) -> np.ndarray:
        """Get average prior geometry volume over time."""
        return np.array([s.mean_volume_prior for s in self.snapshots])


# =============================================================================
# Geometry Tracker
# =============================================================================

class GeometryTracker:
    """
    Track emergent geometry during simulation.

    Computes pullback metrics from agent beliefs and priors, tracking:
    - Individual agent geometries (epistemic and ontological)
    - Consensus geometry (gauge-invariant collective metric)
    - Eigenvalue sector evolution (observable/dark/internal)
    - Volume elements and curvature

    Parameters
    ----------
    agents : list
        Initial agent list (to extract config)
    track_interval : int
        Record geometry every N steps
    dx : float or array
        Grid spacing for derivative computation
    enable_consensus : bool
        Compute consensus metrics (expensive for many agents)
    enable_gauge_averaging : bool
        Perform gauge averaging (very expensive, requires MC sampling)
    gauge_samples : int
        Number of MC samples for gauge averaging
    lambda_obs : float
        Observable sector threshold (relative to max eigenvalue)
    lambda_dark : float
        Dark sector threshold
    """

    def __init__(
        self,
        agents: List,
        track_interval: int = 10,
        dx: float = 1.0,
        enable_consensus: bool = True,
        enable_gauge_averaging: bool = False,
        gauge_samples: int = 50,
        lambda_obs: float = 0.1,
        lambda_dark: float = 0.01
    ):
        self.track_interval = track_interval
        self.dx = dx
        self.enable_consensus = enable_consensus
        self.enable_gauge_averaging = enable_gauge_averaging
        self.gauge_samples = gauge_samples
        self.lambda_obs = lambda_obs
        self.lambda_dark = lambda_dark

        # Extract config from first agent
        if len(agents) > 0:
            agent = agents[0]
            self.spatial_shape = agent.config.spatial_shape
            self.K_latent = agent.config.K
        else:
            self.spatial_shape = ()
            self.K_latent = 0

        # History
        self.history = GeometryHistory(
            n_agents=len(agents),
            spatial_shape=self.spatial_shape,
            K_latent=self.K_latent,
            dx=dx,
            track_interval=track_interval,
            enable_consensus=enable_consensus,
            enable_gauge_averaging=enable_gauge_averaging
        )

        # Cache
        self._last_step = -1

    def should_record(self, step: int) -> bool:
        """Check if we should record at this step."""
        return step % self.track_interval == 0 or step == 0

    def record(self, step: int, agents: List):
        """
        Record geometry snapshot at current step.

        Parameters
        ----------
        step : int
            Current training step
        agents : list
            Current agent list (may change with emergence)
        """
        if not self.should_record(step):
            return

        if step == self._last_step:
            return  # Already recorded

        self._last_step = step

        # Create snapshot
        snapshot = GeometrySnapshot(step=step)

        # Compute per-agent metrics
        agent_metrics_belief = []
        agent_metrics_prior = []

        for agent in agents:
            try:
                G_belief, G_prior = agent_induced_metrics(
                    agent,
                    dx=self.dx,
                    periodic=True,
                    eps=1e-8
                )

                # Compute spectral decomposition
                G_belief.compute_spectral_decomposition()
                G_prior.compute_spectral_decomposition()

                agent_metrics_belief.append(G_belief)
                agent_metrics_prior.append(G_prior)

            except Exception as e:
               # print(f"  ⚠️  Warning: Failed to compute metrics for agent {agent.agent_id}: {e}")
                continue

        snapshot.agent_metrics_belief = agent_metrics_belief
        snapshot.agent_metrics_prior = agent_metrics_prior

        # Compute consensus metrics (if multiple agents)
        if self.enable_consensus and len(agents) > 1:
            try:
                snapshot.consensus_belief = compute_consensus_metric(
                    agents,
                    metric_type="belief",
                    gauge_average=self.enable_gauge_averaging,
                    n_samples_gauge=self.gauge_samples,
                    dx=self.dx,
                    weight_function=None  # Uniform weights
                )

                snapshot.consensus_prior = compute_consensus_metric(
                    agents,
                    metric_type="prior",
                    gauge_average=self.enable_gauge_averaging,
                    n_samples_gauge=self.gauge_samples,
                    dx=self.dx,
                    weight_function=None  # Uniform weights
                )
            except Exception as e:
                c=3 #print(f"  ⚠️  Warning: Failed to compute consensus metrics: {e}")

        # Compute summary statistics
        if agent_metrics_belief:
            self._compute_summary_stats(snapshot, agent_metrics_belief, agent_metrics_prior)

        # Store snapshot
        self.history.snapshots.append(snapshot)

    def _compute_summary_stats(
        self,
        snapshot: GeometrySnapshot,
        metrics_belief: List[InducedMetric],
        metrics_prior: List[InducedMetric]
    ):
        """Compute summary statistics for snapshot."""
        n_agents = len(metrics_belief)

        # Average eigenvalues (over agents and spatial points)
        eigvals_belief_all = []
        eigvals_prior_all = []

        for G_b, G_p in zip(metrics_belief, metrics_prior):
            if G_b.eigenvalues is not None:
                # Average over spatial points: (*spatial, n_dims) -> (n_dims,)
                eigvals_belief_all.append(np.mean(G_b.eigenvalues, axis=tuple(range(len(self.spatial_shape)))))
            if G_p.eigenvalues is not None:
                eigvals_prior_all.append(np.mean(G_p.eigenvalues, axis=tuple(range(len(self.spatial_shape)))))

        if eigvals_belief_all:
            snapshot.mean_eigenvalues_belief = np.mean(eigvals_belief_all, axis=0)
        if eigvals_prior_all:
            snapshot.mean_eigenvalues_prior = np.mean(eigvals_prior_all, axis=0)

        # Sector dimensions (averaged over agents)
        n_observable_list = []
        n_dark_list = []
        n_internal_list = []

        for G_b in metrics_belief:
            try:
                obs_mask, dark_mask, int_mask = G_b.get_three_sector_decomposition(
                    lambda_obs=self.lambda_obs,
                    lambda_dark=self.lambda_dark,
                    relative=True
                )
                # Count True values (averaged over spatial points)
                n_observable_list.append(np.mean(np.sum(obs_mask, axis=-1)))
                n_dark_list.append(np.mean(np.sum(dark_mask, axis=-1)))
                n_internal_list.append(np.mean(np.sum(int_mask, axis=-1)))
            except:
                pass

        if n_observable_list:
            snapshot.mean_n_observable = np.mean(n_observable_list)
            snapshot.mean_n_dark = np.mean(n_dark_list)
            snapshot.mean_n_internal = np.mean(n_internal_list)

        # Volume elements (averaged over agents and spatial points)
        volumes_belief = []
        volumes_prior = []

        for G_b, G_p in zip(metrics_belief, metrics_prior):
            try:
                vol_b = G_b.volume_element()
                volumes_belief.append(np.mean(vol_b))
            except:
                pass

            try:
                vol_p = G_p.volume_element()
                volumes_prior.append(np.mean(vol_p))
            except:
                pass

        if volumes_belief:
            snapshot.mean_volume_belief = np.mean(volumes_belief)
        if volumes_prior:
            snapshot.mean_volume_prior = np.mean(volumes_prior)

    def save(self, path: Union[str, Path]):
        """Save history to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'wb') as f:
            pickle.dump(self.history, f)

        print(f"✓ Saved geometry history: {path}")

    def load(self, path: Union[str, Path]):
        """Load history from disk."""
        with open(path, 'rb') as f:
            self.history = pickle.load(f)

        print(f"✓ Loaded geometry history: {path}")

    def plot_evolution(self, save_path: Optional[Union[str, Path]] = None):
        """
        Plot geometry evolution over time.

        Creates multi-panel figure showing:
        - Observable sector dimensions
        - Volume elements
        - Top eigenvalues
        """
        if not self.history.snapshots:
            print("No snapshots to plot")
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        steps = self.history.get_steps()

        # Panel 1: Observable sector dimensions
        ax = axes[0, 0]
        n_obs = self.history.get_mean_observable_dims()
        ax.plot(steps, n_obs, 'b-', linewidth=2, marker='o', markersize=4)
        ax.set_xlabel('Step')
        ax.set_ylabel('# Observable Dimensions')
        ax.set_title('Observable Sector Evolution')
        ax.grid(True, alpha=0.3)

        # Panel 2: Volume elements
        ax = axes[0, 1]
        vol_belief = self.history.get_mean_volume_belief()
        vol_prior = self.history.get_mean_volume_prior()
        ax.plot(steps, vol_belief, 'b-', linewidth=2, label='Belief (epistemic)', marker='o', markersize=4)
        ax.plot(steps, vol_prior, 'r-', linewidth=2, label='Prior (ontological)', marker='s', markersize=4)
        ax.set_xlabel('Step')
        ax.set_ylabel('Volume Element √det(G)')
        ax.set_title('Geometry Volume Evolution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')

        # Panel 3: Top 3 eigenvalues (belief)
        ax = axes[1, 0]
        first_eigvals = self.history.snapshots[0].mean_eigenvalues_belief
        if first_eigvals is not None and len(first_eigvals) > 0:
            for i in range(min(3, len(first_eigvals))):
                eigvals = [s.mean_eigenvalues_belief[i] if s.mean_eigenvalues_belief is not None else 0
                          for s in self.history.snapshots]
                ax.plot(steps, eigvals, linewidth=2, marker='o', markersize=3, label=f'λ_{i+1}')
        ax.set_xlabel('Step')
        ax.set_ylabel('Eigenvalue')
        ax.set_title('Top Eigenvalues (Belief Geometry)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')

        # Panel 4: Sector decomposition
        ax = axes[1, 1]
        n_obs_arr = np.array([s.mean_n_observable for s in self.history.snapshots])
        n_dark_arr = np.array([s.mean_n_dark for s in self.history.snapshots])
        n_int_arr = np.array([s.mean_n_internal for s in self.history.snapshots])

        ax.plot(steps, n_obs_arr, 'g-', linewidth=2, marker='o', markersize=4, label='Observable')
        ax.plot(steps, n_dark_arr, 'orange', linewidth=2, marker='s', markersize=4, label='Dark')
        ax.plot(steps, n_int_arr, 'purple', linewidth=2, marker='^', markersize=4, label='Internal')
        ax.set_xlabel('Step')
        ax.set_ylabel('# Dimensions')
        ax.set_title('Three-Sector Decomposition')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Saved geometry evolution plot: {save_path}")
            plt.close()
        else:
            plt.show()

    def plot_consensus_evolution(self, save_path: Optional[Union[str, Path]] = None):
        """Plot consensus geometry evolution (if tracked)."""
        if not self.enable_consensus:
            print("Consensus tracking not enabled")
            return

        if not any(s.consensus_belief is not None for s in self.history.snapshots):
            print("No consensus metrics available")
            return

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        steps = self.history.get_steps()

        # Consensus volume elements
        ax = axes[0]
        vol_consensus_belief = []
        vol_consensus_prior = []

        for s in self.history.snapshots:
            if s.consensus_belief is not None:
                vol_b = np.mean(np.sqrt(np.maximum(np.linalg.det(s.consensus_belief.G_consensus), 1e-10)))
                vol_consensus_belief.append(vol_b)
            else:
                vol_consensus_belief.append(np.nan)

            if s.consensus_prior is not None:
                vol_p = np.mean(np.sqrt(np.maximum(np.linalg.det(s.consensus_prior.G_consensus), 1e-10)))
                vol_consensus_prior.append(vol_p)
            else:
                vol_consensus_prior.append(np.nan)

        ax.plot(steps, vol_consensus_belief, 'b-', linewidth=2, label='Belief', marker='o', markersize=4)
        ax.plot(steps, vol_consensus_prior, 'r-', linewidth=2, label='Prior', marker='s', markersize=4)
        ax.set_xlabel('Step')
        ax.set_ylabel('Consensus Volume √det(Ḡ)')
        ax.set_title('Consensus Geometry Volume')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')

        # Number of agents contributing
        ax = axes[1]
        n_agents_list = [s.consensus_belief.n_agents if s.consensus_belief is not None else 0
                        for s in self.history.snapshots]
        ax.plot(steps, n_agents_list, 'k-', linewidth=2, marker='o', markersize=4)
        ax.set_xlabel('Step')
        ax.set_ylabel('# Agents')
        ax.set_title('Agents Contributing to Consensus')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Saved consensus evolution plot: {save_path}")
            plt.close()
        else:
            plt.show()


# =============================================================================
# Helper Functions
# =============================================================================

def analyze_final_geometry(history: GeometryHistory, save_dir: Optional[Path] = None):
    """
    Analyze final geometry snapshot in detail.

    Generates comprehensive report and visualizations for the last timestep.
    """
    if not history.snapshots:
        print("No snapshots available")
        return

    final = history.snapshots[-1]

    print(f"\n{'='*70}")
    print(f"FINAL GEOMETRY ANALYSIS (Step {final.step})")
    print(f"{'='*70}")

    print(f"\nSummary Statistics:")
    print(f"  Observable dimensions:  {final.mean_n_observable:.1f}")
    print(f"  Dark dimensions:        {final.mean_n_dark:.1f}")
    print(f"  Internal dimensions:    {final.mean_n_internal:.1f}")
    print(f"  Total K:                {history.K_latent}")

    print(f"\nVolume Elements:")
    print(f"  Belief geometry:  {final.mean_volume_belief:.4e}")
    print(f"  Prior geometry:   {final.mean_volume_prior:.4e}")

    if final.mean_eigenvalues_belief is not None:
        print(f"\nTop 5 Eigenvalues (Belief):")
        for i, eigval in enumerate(final.mean_eigenvalues_belief[:5]):
            print(f"  λ_{i+1} = {eigval:.4e}")

    if final.consensus_belief is not None:
        print(f"\nConsensus Metrics:")
        print(f"  # Agents:         {final.consensus_belief.n_agents}")
        consensus_vol = np.mean(np.sqrt(np.maximum(np.linalg.det(final.consensus_belief.G_consensus), 1e-10)))
        print(f"  Consensus volume: {consensus_vol:.4e}")

    print(f"{'='*70}\n")

    # Save detailed plots if requested
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Eigenvalue spectrum
        fig, ax = plt.subplots(figsize=(10, 6))
        if final.mean_eigenvalues_belief is not None:
            ax.semilogy(range(1, len(final.mean_eigenvalues_belief) + 1),
                       final.mean_eigenvalues_belief, 'bo-', linewidth=2, markersize=6, label='Belief')
        if final.mean_eigenvalues_prior is not None:
            ax.semilogy(range(1, len(final.mean_eigenvalues_prior) + 1),
                       final.mean_eigenvalues_prior, 'rs-', linewidth=2, markersize=6, label='Prior')
        ax.set_xlabel('Eigenvalue Index')
        ax.set_ylabel('Eigenvalue')
        ax.set_title(f'Eigenvalue Spectrum (Step {final.step})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_dir / "final_eigenvalue_spectrum.png", dpi=150)
        plt.close()
        print(f"✓ Saved eigenvalue spectrum: {save_dir / 'final_eigenvalue_spectrum.png'}")