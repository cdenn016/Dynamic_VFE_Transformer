# -*- coding: utf-8 -*-
"""
Created on Fri Nov 21 14:26:38 2025

@author: chris and christine
"""

#!/usr/bin/env python3
"""
Visualization of agent fields on 2D spatial manifolds.

Supports imaging emergent agents at any hierarchical scale, showing:
- Belief means (mu_q) and covariances (Sigma_q)
- Prior means (mu_p) and covariances (Sigma_p)
- Observable fields (phi)

For multi-dimensional latent spaces (K > 1), can show:
- Individual components
- Norms/magnitudes
- Principal components

Author: Chris
Date: November 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass


@dataclass
class FieldSnapshot:
    """Single snapshot of agent fields at a given scale."""
    step: int
    scale: int
    n_agents: int

    # Field data: list of arrays per agent
    # Each array has shape (*spatial_shape, K) for means or (*spatial_shape, K, K) for covariances
    mu_q: Optional[List[np.ndarray]] = None
    Sigma_q: Optional[List[np.ndarray]] = None
    mu_p: Optional[List[np.ndarray]] = None
    Sigma_p: Optional[List[np.ndarray]] = None
    phi: Optional[List[np.ndarray]] = None

    # Metadata
    spatial_shape: Optional[Tuple[int, ...]] = None
    K: Optional[int] = None


class AgentFieldVisualizer:
    """
    Visualizes agent fields on 2D spatial manifolds across hierarchical scales.

    Features:
    - Image any scale (base agents or meta-agents)
    - Show any field (mu_q, Sigma_q, mu_p, Sigma_p, phi)
    - Handle multi-component latent spaces (K > 1)
    - Generate evolution movies over time
    - Side-by-side comparison of multiple agents
    """

    def __init__(
        self,
        output_dir: Path,
        scales_to_track: Optional[List[int]] = None,
        fields_to_track: Optional[List[str]] = None,
        latent_components: Optional[List[int]] = None,
        track_interval: int = 10,
    ):
        """
        Initialize visualizer.

        Args:
            output_dir: Directory to save visualizations
            scales_to_track: Which hierarchical scales to visualize (default: [0])
            fields_to_track: Which fields to track (default: ["mu_q", "phi"])
            latent_components: Which latent components to show (default: all)
            track_interval: Record every N steps
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.scales_to_track = scales_to_track if scales_to_track is not None else [0]
        self.fields_to_track = fields_to_track if fields_to_track is not None else ["mu_q", "phi"]
        self.latent_components = latent_components  # None = all
        self.track_interval = track_interval

        # History: scale -> list of snapshots
        self.history: Dict[int, List[FieldSnapshot]] = {scale: [] for scale in self.scales_to_track}

    def should_record(self, step: int) -> bool:
        """Check if should record this step."""
        return step % self.track_interval == 0

    def record(self, step: int, system) -> None:
        """
        Record agent fields from hierarchical system.

        Args:
            step: Current simulation step
            system: HierarchicalSystem with agents at multiple scales
        """
        for scale in self.scales_to_track:
            if scale not in system.agents:
                continue

            agents = system.agents[scale]
            if len(agents) == 0:
                continue

            # Extract spatial shape and K from first agent
            agent_0 = agents[0]
            if hasattr(agent_0, 'base_manifold') and agent_0.base_manifold.ndim > 0:
                spatial_shape = agent_0.base_manifold.shape
            else:
                # Point manifold - skip visualization
                continue

            K = agent_0.K

            # Create snapshot
            snapshot = FieldSnapshot(
                step=step,
                scale=scale,
                n_agents=len(agents),
                spatial_shape=spatial_shape,
                K=K
            )

            # Extract fields
            if "mu_q" in self.fields_to_track:
                snapshot.mu_q = [agent.mu_q.copy() for agent in agents]

            if "Sigma_q" in self.fields_to_track:
                snapshot.Sigma_q = [agent.Sigma_q.copy() for agent in agents]

            if "mu_p" in self.fields_to_track:
                snapshot.mu_p = [agent.mu_p.copy() for agent in agents]

            if "Sigma_p" in self.fields_to_track:
                snapshot.Sigma_p = [agent.Sigma_p.copy() for agent in agents]

            if "phi" in self.fields_to_track:
                snapshot.phi = [agent.phi.copy() for agent in agents]

            self.history[scale].append(snapshot)

    def plot_field_component(
        self,
        snapshot: FieldSnapshot,
        field_name: str,
        component_idx: int = 0,
        agent_idx: int = 0,
        ax: Optional[plt.Axes] = None,
        **kwargs
    ) -> plt.Axes:
        """
        Plot a single field component for one agent.

        Args:
            snapshot: FieldSnapshot containing field data
            field_name: "mu_q", "Sigma_q", "mu_p", "Sigma_p", or "phi"
            component_idx: Which latent component (for K > 1)
            agent_idx: Which agent to plot
            ax: Matplotlib axes (created if None)
            **kwargs: Additional kwargs for imshow (cmap, vmin, vmax, etc.)

        Returns:
            Matplotlib axes with plot
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 5))

        # Get field data
        field_data = getattr(snapshot, field_name)
        if field_data is None or agent_idx >= len(field_data):
            ax.text(0.5, 0.5, f"{field_name} not available",
                   ha='center', va='center', transform=ax.transAxes)
            return ax

        data = field_data[agent_idx]

        # Extract component
        if field_name in ["mu_q", "mu_p", "phi"]:
            # Shape: (*spatial, K)
            if data.ndim == 1:
                # Point manifold case
                img_data = data[component_idx:component_idx+1]
            elif data.ndim == 2:
                # 1D manifold
                img_data = data[:, component_idx]
            elif data.ndim == 3:
                # 2D manifold
                img_data = data[:, :, component_idx]
            else:
                raise ValueError(f"Unexpected data shape: {data.shape}")

        elif field_name in ["Sigma_q", "Sigma_p"]:
            # Shape: (*spatial, K, K) - show diagonal element
            if data.ndim == 2:
                # Point manifold
                img_data = np.diag(data)[component_idx:component_idx+1]
            elif data.ndim == 3:
                # 1D manifold
                img_data = data[:, component_idx, component_idx]
            elif data.ndim == 4:
                # 2D manifold
                img_data = data[:, :, component_idx, component_idx]
            else:
                raise ValueError(f"Unexpected covariance shape: {data.shape}")
        else:
            raise ValueError(f"Unknown field: {field_name}")

        # Plot
        if img_data.ndim == 1:
            # 1D plot
            ax.plot(img_data, **kwargs)
            ax.set_xlabel('Position')
            ax.set_ylabel('Value')
        else:
            # 2D heatmap
            im = ax.imshow(img_data, origin='lower', **kwargs)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            plt.colorbar(im, ax=ax)

        ax.set_title(f"{field_name}[{component_idx}] - Agent {agent_idx} (Scale {snapshot.scale})")

        return ax

    def plot_snapshot_grid(
        self,
        snapshot: FieldSnapshot,
        save_path: Optional[Path] = None,
        show_all_agents: bool = True,
        show_all_components: bool = False,
    ) -> None:
        """
        Create comprehensive grid visualization of a snapshot.

        Shows multiple agents and/or multiple latent components in a grid.

        Args:
            snapshot: FieldSnapshot to visualize
            save_path: Where to save (if None, generates default name)
            show_all_agents: Show all agents side-by-side (if False, only agent 0)
            show_all_components: Show all latent components (if False, only first 3)
        """
        # Determine what to show
        n_agents_to_show = snapshot.n_agents if show_all_agents else 1

        if show_all_components:
            components_to_show = list(range(snapshot.K)) if snapshot.K else [0]
        else:
            components_to_show = list(range(min(3, snapshot.K))) if snapshot.K else [0]

        # Filter to user-specified components if provided
        if self.latent_components is not None:
            components_to_show = [c for c in components_to_show if c in self.latent_components]

        n_fields = len(self.fields_to_track)
        n_components = len(components_to_show)

        # Create grid: rows = fields, cols = components × agents
        n_cols = n_components * n_agents_to_show
        n_rows = n_fields

        fig = plt.figure(figsize=(4 * n_cols, 3.5 * n_rows))
        gs = GridSpec(n_rows, n_cols, figure=fig, hspace=0.3, wspace=0.3)

        for row_idx, field_name in enumerate(self.fields_to_track):
            for agent_idx in range(n_agents_to_show):
                for comp_idx, component in enumerate(components_to_show):
                    col_idx = agent_idx * n_components + comp_idx

                    ax = fig.add_subplot(gs[row_idx, col_idx])

                    # Determine colormap
                    if field_name in ["Sigma_q", "Sigma_p"]:
                        cmap = "viridis"  # Covariances are positive
                    else:
                        cmap = "RdBu_r"  # Means can be positive/negative

                    self.plot_field_component(
                        snapshot,
                        field_name,
                        component_idx=component,
                        agent_idx=agent_idx,
                        ax=ax,
                        cmap=cmap,
                        aspect='auto'
                    )

        fig.suptitle(f"Scale {snapshot.scale} - Step {snapshot.step} - {snapshot.n_agents} agents",
                    fontsize=16, y=0.995)

        # Save
        if save_path is None:
            save_path = self.output_dir / f"fields_scale{snapshot.scale}_step{snapshot.step:04d}.png"
        else:
            save_path = Path(save_path)

        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

    def plot_all_snapshots(self, scale: int = 0) -> None:
        """Generate grid visualizations for all recorded snapshots at a scale."""
        if scale not in self.history or len(self.history[scale]) == 0:
            print(f"No snapshots recorded for scale {scale}")
            return

        scale_dir = self.output_dir / f"scale_{scale}"
        scale_dir.mkdir(exist_ok=True)

        print(f"\nGenerating field visualizations for scale {scale}...")
        for snapshot in self.history[scale]:
            save_path = scale_dir / f"step_{snapshot.step:04d}.png"
            self.plot_snapshot_grid(snapshot, save_path=save_path)
            print(f"  Saved: {save_path.name}")

    def plot_field_evolution(
        self,
        field_name: str,
        component_idx: int = 0,
        agent_idx: int = 0,
        scale: int = 0,
        save_path: Optional[Path] = None,
    ) -> None:
        """
        Create time evolution movie-style grid for a specific field component.

        Shows how a single field component evolves over all recorded steps.

        Args:
            field_name: Which field to show
            component_idx: Which latent component
            agent_idx: Which agent
            scale: Which hierarchical scale
            save_path: Where to save
        """
        if scale not in self.history or len(self.history[scale]) == 0:
            print(f"No snapshots recorded for scale {scale}")
            return

        snapshots = self.history[scale]
        n_snapshots = len(snapshots)

        # Layout: rows of 4
        n_cols = 4
        n_rows = (n_snapshots + n_cols - 1) // n_cols

        fig = plt.figure(figsize=(4 * n_cols, 3.5 * n_rows))
        gs = GridSpec(n_rows, n_cols, figure=fig, hspace=0.3, wspace=0.3)

        # Compute global vmin/vmax for consistent coloring
        all_data = []
        for snapshot in snapshots:
            field_data = getattr(snapshot, field_name)
            if field_data is not None and agent_idx < len(field_data):
                data = field_data[agent_idx]
                if field_name in ["mu_q", "mu_p", "phi"]:
                    all_data.append(data[..., component_idx])
                elif field_name in ["Sigma_q", "Sigma_p"]:
                    all_data.append(data[..., component_idx, component_idx])

        if len(all_data) > 0:
            all_data = np.concatenate([d.ravel() for d in all_data])
            vmin, vmax = np.percentile(all_data, [1, 99])
        else:
            vmin, vmax = None, None

        # Plot each snapshot
        for idx, snapshot in enumerate(snapshots):
            row = idx // n_cols
            col = idx % n_cols
            ax = fig.add_subplot(gs[row, col])

            cmap = "viridis" if field_name in ["Sigma_q", "Sigma_p"] else "RdBu_r"

            self.plot_field_component(
                snapshot,
                field_name,
                component_idx=component_idx,
                agent_idx=agent_idx,
                ax=ax,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                aspect='auto'
            )
            ax.set_title(f"Step {snapshot.step}")

        fig.suptitle(f"{field_name}[{component_idx}] Evolution - Agent {agent_idx}, Scale {scale}",
                    fontsize=16, y=0.995)

        # Save
        if save_path is None:
            save_path = self.output_dir / f"evolution_{field_name}_comp{component_idx}_agent{agent_idx}_scale{scale}.png"
        else:
            save_path = Path(save_path)

        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

    def plot_agent_comparison(
        self,
        field_name: str,
        component_idx: int = 0,
        scale: int = 0,
        step_idx: int = -1,
        save_path: Optional[Path] = None,
    ) -> None:
        """
        Compare all agents side-by-side for a specific field at a specific time.

        Args:
            field_name: Which field to compare
            component_idx: Which latent component
            scale: Which hierarchical scale
            step_idx: Which snapshot (-1 = final)
            save_path: Where to save
        """
        if scale not in self.history or len(self.history[scale]) == 0:
            print(f"No snapshots recorded for scale {scale}")
            return

        snapshot = self.history[scale][step_idx]
        n_agents = snapshot.n_agents

        # Layout
        n_cols = min(4, n_agents)
        n_rows = (n_agents + n_cols - 1) // n_cols

        fig = plt.figure(figsize=(4 * n_cols, 3.5 * n_rows))
        gs = GridSpec(n_rows, n_cols, figure=fig, hspace=0.3, wspace=0.3)

        # Compute global vmin/vmax
        field_data = getattr(snapshot, field_name)
        if field_data is not None:
            all_data = []
            for data in field_data:
                if field_name in ["mu_q", "mu_p", "phi"]:
                    all_data.append(data[..., component_idx])
                elif field_name in ["Sigma_q", "Sigma_p"]:
                    all_data.append(data[..., component_idx, component_idx])
            all_data = np.concatenate([d.ravel() for d in all_data])
            vmin, vmax = np.percentile(all_data, [1, 99])
        else:
            vmin, vmax = None, None

        # Plot each agent
        for agent_idx in range(n_agents):
            row = agent_idx // n_cols
            col = agent_idx % n_cols
            ax = fig.add_subplot(gs[row, col])

            cmap = "viridis" if field_name in ["Sigma_q", "Sigma_p"] else "RdBu_r"

            self.plot_field_component(
                snapshot,
                field_name,
                component_idx=component_idx,
                agent_idx=agent_idx,
                ax=ax,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                aspect='auto'
            )

        fig.suptitle(f"{field_name}[{component_idx}] - All Agents at Step {snapshot.step} (Scale {scale})",
                    fontsize=16, y=0.995)

        # Save
        if save_path is None:
            save_path = self.output_dir / f"comparison_{field_name}_comp{component_idx}_step{snapshot.step}_scale{scale}.png"
        else:
            save_path = Path(save_path)

        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

    def generate_summary_report(self) -> None:
        """Generate comprehensive visualization report for all tracked scales."""
        print(f"\n{'='*70}")
        print("GENERATING AGENT FIELD VISUALIZATIONS")
        print(f"{'='*70}\n")

        for scale in self.scales_to_track:
            if scale not in self.history or len(self.history[scale]) == 0:
                print(f"Scale {scale}: No data recorded")
                continue

            print(f"\nScale {scale}: {len(self.history[scale])} snapshots")

            # All snapshots
            self.plot_all_snapshots(scale=scale)

            # Field evolution for each tracked field
            for field_name in self.fields_to_track:
                # Show first few components
                n_components = self.history[scale][0].K if self.history[scale][0].K else 1
                components_to_show = list(range(min(3, n_components)))

                if self.latent_components is not None:
                    components_to_show = [c for c in components_to_show if c in self.latent_components]

                for component_idx in components_to_show:
                    self.plot_field_evolution(
                        field_name=field_name,
                        component_idx=component_idx,
                        agent_idx=0,
                        scale=scale
                    )

            # Agent comparison at final step
            final_snapshot = self.history[scale][-1]
            if final_snapshot.n_agents > 1:
                for field_name in self.fields_to_track:
                    self.plot_agent_comparison(
                        field_name=field_name,
                        component_idx=0,
                        scale=scale,
                        step_idx=-1
                    )

        print(f"\n{'='*70}")
        print(f"✓ Visualizations saved to: {self.output_dir}")
        print(f"{'='*70}\n")