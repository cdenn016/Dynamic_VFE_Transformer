"""
Energy Landscape and Thermodynamics Visualization

Visualizes the energy decomposition, non-equilibrium dynamics, and thermodynamic
properties of the hierarchical meta-agent system.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

from meta.participatory_diagnostics import ParticipatoryDiagnostics


class EnergyVisualizer:
    """Visualize energy landscapes and thermodynamic properties."""

    def __init__(self, diagnostics: ParticipatoryDiagnostics):
        """
        Initialize with diagnostics tracker.

        Args:
            diagnostics: ParticipatoryDiagnostics instance that has been tracking evolution
        """
        self.diagnostics = diagnostics

    def plot_energy_landscape(self,
                              figsize: Tuple[int, int] = (14, 8),
                              scales_per_figure: int = 4) -> List[plt.Figure]:
        """
        Plot multi-scale energy decomposition over time.

        Shows stacked contributions from:
        - E_self (intrinsic energy)
        - E_belief_align (lateral coherence)
        - E_prior_align (vertical alignment)

        Args:
            figsize: Figure size for each page
            scales_per_figure: Maximum number of scales per figure (default: 4)

        Returns:
            List of figures (one per group of scales)
        """
        if not self.diagnostics.scale_history:
            raise ValueError("No scale history recorded. Run diagnostics during evolution.")

        # Extract data from scale_history
        scale_energy = {}
        for scale, snapshots in self.diagnostics.scale_history.items():
            times = [snap.step for snap in snapshots]
            scale_energy[scale] = {
                'times': times,
                'self': [snap.total_self_energy for snap in snapshots],
                'belief': [snap.total_belief_align for snap in snapshots],
                'prior': [snap.total_prior_align for snap in snapshots],
                'total': [snap.total_energy for snap in snapshots]
            }

        if not scale_energy:
            raise ValueError("No scale data available")

        colors = {'self': '#FF6B6B', 'belief': '#4ECDC4', 'prior': '#45B7D1'}

        # Sort scales
        sorted_scales = sorted(scale_energy.items())
        n_total_scales = len(sorted_scales)

        # Break into groups of scales_per_figure
        figures = []
        for start_idx in range(0, n_total_scales, scales_per_figure):
            end_idx = min(start_idx + scales_per_figure, n_total_scales)
            scales_in_fig = sorted_scales[start_idx:end_idx]
            n_scales = len(scales_in_fig)

            # Create figure for this group
            fig, axes = plt.subplots(n_scales, 1, figsize=figsize, sharex=True)

            if n_scales == 1:
                axes = [axes]

            # Plot each scale in this group
            for idx, (scale, energies) in enumerate(scales_in_fig):
                ax = axes[idx]
                times = energies['times']

                # Stack plot for energy components
                ax.fill_between(times, 0, energies['self'],
                               label='E_self', alpha=0.8, color=colors['self'])
                ax.fill_between(times, energies['self'],
                               np.array(energies['self']) + np.array(energies['belief']),
                               label='E_belief_align', alpha=0.8, color=colors['belief'])
                ax.fill_between(times,
                               np.array(energies['self']) + np.array(energies['belief']),
                               energies['total'],
                               label='E_prior_align', alpha=0.8, color=colors['prior'])

                # Total energy line
                ax.plot(times, energies['total'], 'k-', linewidth=2, label='Total')

                ax.set_ylabel(f'Energy\n(ζ={scale})', fontsize=11, fontweight='bold')
                ax.legend(loc='upper right', fontsize=9, ncol=4)
                ax.grid(alpha=0.3, axis='y')

            axes[-1].set_xlabel('Step', fontsize=12)

            # Add page info if multiple figures
            if n_total_scales > scales_per_figure:
                page_num = (start_idx // scales_per_figure) + 1
                total_pages = (n_total_scales + scales_per_figure - 1) // scales_per_figure
                title = f'Energy Landscape (Scales {start_idx}-{end_idx-1}, Page {page_num}/{total_pages})'
            else:
                title = 'Multi-Scale Energy Landscape Decomposition'

            fig.suptitle(title, fontsize=16, fontweight='bold')
            plt.tight_layout()
            figures.append(fig)

        return figures

    def plot_energy_flow(self, figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
        """
        Plot energy flux (rate of change) across scales.

        Args:
            figsize: Figure size
        """
        if not self.diagnostics.scale_history:
            raise ValueError("No scale history recorded")

        # Compute energy flux for each scale
        scale_flux = {}

        for scale, snapshots in self.diagnostics.scale_history.items():
            if len(snapshots) < 2:
                continue

            times = []
            flux = []

            for i in range(1, len(snapshots)):
                prev_snap = snapshots[i - 1]
                curr_snap = snapshots[i]

                dt = curr_snap.step - prev_snap.step
                if dt == 0:
                    continue

                dE = curr_snap.total_energy - prev_snap.total_energy
                flux.append(dE / dt)
                times.append(curr_snap.step)

            if times:
                scale_flux[scale] = {'times': times, 'flux': flux}

        if not scale_flux:
            raise ValueError("Insufficient data to compute energy flux")

        fig, ax = plt.subplots(figsize=figsize)

        # Plot flux for each scale
        colors = plt.cm.viridis(np.linspace(0, 1, len(scale_flux)))

        for (scale, data), color in zip(sorted(scale_flux.items()), colors):
            ax.plot(data['times'], data['flux'], marker='o', label=f'ζ={scale}',
                   color=color, linewidth=2, alpha=0.8)

        ax.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5)
        ax.set_xlabel('Step', fontsize=12)
        ax.set_ylabel('Energy Flux (dE/dstep)', fontsize=12)
        ax.set_title('Energy Flow Across Scales', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_prior_evolution(self, figsize: Tuple[int, int] = (14, 6)) -> plt.Figure:
        """
        Plot prior change history (L2 norm from previous prior).

        Shows top-down information flow.

        Args:
            figsize: Figure size
        """
        if not self.diagnostics.prior_changes:
            raise ValueError("No prior changes recorded")

        # Organize by agent - prior_changes is List[Tuple[step, agent_id, change]]
        agent_changes = defaultdict(lambda: {'steps': [], 'changes': []})

        for step, agent_id, change in self.diagnostics.prior_changes:
            agent_changes[agent_id]['steps'].append(int(step))  # Ensure int
            agent_changes[agent_id]['changes'].append(float(change))  # Ensure float

        if not agent_changes:
            raise ValueError("No prior changes to plot")

        # Create plot
        fig, ax = plt.subplots(figsize=figsize)

        n_agents = len(agent_changes)
        max_labeled = 8  # Maximum agents to show in legend

        # Plot each agent's prior changes
        # Sort by string representation to handle mixed str/int agent IDs
        for idx, (agent_id, data) in enumerate(sorted(agent_changes.items(), key=lambda x: str(x[0]))):
            # Only add label for first few agents if too many
            if n_agents <= max_labeled:
                label = f'{agent_id}'
            elif idx < 3:  # First 3 agents
                label = f'{agent_id}'
            elif idx == 3:  # Indicate there are more
                label = f'... +{n_agents - 3} more agents'
            else:
                label = None  # No label for others

            ax.plot(data['steps'], data['changes'], marker='o' if n_agents <= max_labeled else None,
                   alpha=0.6 if n_agents <= max_labeled else 0.3,
                   label=label, linewidth=1.5 if n_agents <= max_labeled else 1.0,
                   markersize=4 if n_agents <= max_labeled else 0)

        # Compute and plot average
        all_steps = []
        all_changes = []
        for data in agent_changes.values():
            all_steps.extend(data['steps'])
            all_changes.extend(data['changes'])

        if all_steps:
            unique_steps = sorted(set(all_steps))
            avg_changes = []
            for step in unique_steps:
                changes_at_step = [all_changes[i] for i, s in enumerate(all_steps) if s == step]
                avg_changes.append(np.mean(changes_at_step))

            ax.plot(unique_steps, avg_changes, 'k-', linewidth=3,
                   label=f'Average ({n_agents} agents)', alpha=0.8)

        ax.set_xlabel('Step', fontsize=12)
        ax.set_ylabel('Prior Change (L2 norm)', fontsize=12)
        ax.set_title('Prior Evolution (Top-Down Information Flow)', fontsize=14, fontweight='bold')
        ax.set_yscale('log')

        # Smart legend placement and size
        if n_agents <= max_labeled:
            ax.legend(loc='upper right', fontsize=9, ncol=min(3, (n_agents + 1) // 3 + 1))
        else:
            ax.legend(loc='upper right', fontsize=9, ncol=1)

        ax.grid(alpha=0.3, which='both')

        plt.tight_layout()
        return fig

    def plot_non_equilibrium_indicators(self, figsize: Tuple[int, int] = (14, 10)) -> plt.Figure:
        """
        Plot non-equilibrium indicators dashboard.

        Shows:
        - Energy variance across scales
        - Gradient variance (from agent_history)
        - Energy flux magnitude
        - Non-equilibrium score

        Args:
            figsize: Figure size
        """
        if not self.diagnostics.scale_history:
            raise ValueError("No scale history recorded")

        # Get all unique timesteps
        all_steps = set()
        for snapshots in self.diagnostics.scale_history.values():
            all_steps.update(snap.step for snap in snapshots)
        steps = sorted(all_steps)

        # Energy variance across scales at each timestep
        energy_vars = []
        for step in steps:
            energies_at_step = []
            for scale_snapshots in self.diagnostics.scale_history.values():
                matching = [s.total_energy for s in scale_snapshots if s.step == step]
                if matching:
                    energies_at_step.append(matching[0])
            energy_vars.append(np.var(energies_at_step) if len(energies_at_step) > 1 else 0)

        # Gradient variance from tracked agents
        gradient_vars = []
        for step in steps:
            grad_norms = []
            for agent_history in self.diagnostics.agent_history.values():
                matching = [s.grad_mu_norm for s in agent_history if s.step == step]
                if matching:
                    grad_norms.append(matching[0])
            gradient_vars.append(np.var(grad_norms) if len(grad_norms) > 1 else 0)

        # Energy flux magnitude (average across scales)
        flux_magnitudes = []
        for i, step in enumerate(steps):
            if i == 0:
                flux_magnitudes.append(0)
                continue

            fluxes = []
            for scale_snapshots in self.diagnostics.scale_history.values():
                curr = [s for s in scale_snapshots if s.step == step]
                prev = [s for s in scale_snapshots if s.step == steps[i-1]]
                if curr and prev:
                    dE = curr[0].total_energy - prev[0].total_energy
                    dt = step - steps[i-1]
                    if dt > 0:
                        fluxes.append(abs(dE / dt))

            flux_magnitudes.append(np.mean(fluxes) if fluxes else 0)

        # Non-equilibrium score (0 = equilibrium, 1 = far from equilibrium)
        max_var = max(energy_vars) if energy_vars else 1
        eq_scores = [1 - np.exp(-v / (max_var + 1e-8)) for v in energy_vars]

        fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)

        # Energy variance
        axes[0].plot(steps, energy_vars, 'o-', color='#E74C3C', linewidth=2)
        axes[0].set_ylabel('Energy\nVariance', fontsize=11, fontweight='bold')
        axes[0].grid(alpha=0.3)
        axes[0].fill_between(steps, 0, energy_vars, alpha=0.3, color='#E74C3C')

        # Gradient variance
        axes[1].plot(steps, gradient_vars, 'o-', color='#3498DB', linewidth=2)
        axes[1].set_ylabel('Gradient\nVariance', fontsize=11, fontweight='bold')
        axes[1].grid(alpha=0.3)
        axes[1].fill_between(steps, 0, gradient_vars, alpha=0.3, color='#3498DB')

        # Energy flux magnitude
        axes[2].plot(steps, flux_magnitudes, 'o-', color='#2ECC71', linewidth=2)
        axes[2].set_ylabel('Energy Flux\nMagnitude', fontsize=11, fontweight='bold')
        axes[2].grid(alpha=0.3)
        axes[2].fill_between(steps, 0, flux_magnitudes, alpha=0.3, color='#2ECC71')

        # Non-equilibrium score
        axes[3].plot(steps, eq_scores, 'o-', color='#9B59B6', linewidth=2)
        axes[3].axhline(y=0.5, color='k', linestyle='--', alpha=0.5, label='Threshold')
        axes[3].fill_between(steps, 0, eq_scores, alpha=0.3, color='#9B59B6')
        axes[3].set_ylabel('Non-Equilibrium\nScore', fontsize=11, fontweight='bold')
        axes[3].set_xlabel('Step', fontsize=12)
        axes[3].legend(fontsize=9)
        axes[3].grid(alpha=0.3)

        fig.suptitle('Non-Equilibrium Dynamics Indicators', fontsize=16, fontweight='bold')

        plt.tight_layout()
        return fig

    def plot_energy_per_agent(self,
                             scale: int = 0,
                             step: Optional[int] = None,
                             figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
        """
        Plot energy decomposition for individual agents at a specific scale and step.

        Args:
            scale: Which scale to visualize
            step: Which step to visualize (None for latest)
            figsize: Figure size
        """
        if not self.diagnostics.agent_history:
            raise ValueError("No agent history recorded")

        # Find agents at the target scale and step
        agent_data = []

        for agent_id, history in self.diagnostics.agent_history.items():
            # Find snapshots at the target scale
            scale_snapshots = [s for s in history if s.scale == scale]
            if not scale_snapshots:
                continue

            # Get the snapshot at the target step (or latest)
            if step is None:
                snapshot = scale_snapshots[-1]
            else:
                matching = [s for s in scale_snapshots if s.step == step]
                if not matching:
                    continue
                snapshot = matching[0]

            agent_data.append((agent_id, snapshot))

        if not agent_data:
            raise ValueError(f"No agent data at scale {scale}" +
                           (f" and step {step}" if step is not None else ""))

        # Sort by agent_id for consistent ordering
        agent_data.sort(key=lambda x: x[0])

        agent_ids = [aid for aid, _ in agent_data]
        E_self = [snap.E_self for _, snap in agent_data]
        E_belief = [snap.E_belief_align for _, snap in agent_data]
        E_prior = [snap.E_prior_align for _, snap in agent_data]

        # Use the step from the first snapshot for title
        actual_step = agent_data[0][1].step

        fig, ax = plt.subplots(figsize=figsize)

        x = np.arange(len(agent_ids))
        width = 0.25

        ax.bar(x - width, E_self, width, label='E_self', color='#FF6B6B', alpha=0.8)
        ax.bar(x, E_belief, width, label='E_belief_align', color='#4ECDC4', alpha=0.8)
        ax.bar(x + width, E_prior, width, label='E_prior_align', color='#45B7D1', alpha=0.8)

        ax.set_xlabel('Agent', fontsize=12)
        ax.set_ylabel('Energy', fontsize=12)
        ax.set_title(f'Energy Decomposition per Agent (Scale {scale}, Step {actual_step})',
                    fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        # Handle both string and int agent IDs
        labels = []
        for aid in agent_ids:
            aid_str = str(aid)
            if '_' in aid_str:
                labels.append(aid_str.split('_')[-1][:8])
            else:
                labels.append(aid_str[:8])
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3, axis='y')

        plt.tight_layout()
        return fig

    def plot_interactive_energy_3d(self, scale: int = 0) -> Optional[go.Figure]:
        """
        Create 3D interactive energy landscape using Plotly.

        Shows energy evolution over time and agent space.

        Args:
            scale: Which scale to visualize

        Returns:
            Plotly figure or None if plotly not available
        """
        if not HAS_PLOTLY:
            print("Plotly not available. Install with: pip install plotly")
            return None

        if not self.diagnostics.agent_history:
            raise ValueError("No agent history recorded")

        # Collect data from agent_history
        agent_step_energy = defaultdict(lambda: defaultdict(float))
        all_steps = set()

        for agent_id, history in self.diagnostics.agent_history.items():
            scale_history = [s for s in history if s.scale == scale]
            for snapshot in scale_history:
                agent_step_energy[agent_id][snapshot.step] = snapshot.E_total
                all_steps.add(snapshot.step)

        if not agent_step_energy:
            raise ValueError(f"No agent-level data at scale {scale}")

        # Create meshgrid
        agent_ids = sorted(agent_step_energy.keys())
        steps = sorted(all_steps)

        Z = np.zeros((len(steps), len(agent_ids)))

        for i, agent_id in enumerate(agent_ids):
            for j, step in enumerate(steps):
                if step in agent_step_energy[agent_id]:
                    Z[j, i] = agent_step_energy[agent_id][step]

        # Create mesh
        agent_indices = list(range(len(agent_ids)))
        X, Y = np.meshgrid(agent_indices, steps)

        # Create 3D surface
        fig = go.Figure(data=[go.Surface(x=X, y=Y, z=Z, colorscale='Viridis')])

        fig.update_layout(
            title=f'Energy Landscape Evolution (Scale {scale})',
            scene=dict(
                xaxis_title='Agent Index',
                yaxis_title='Step',
                zaxis_title='Energy'
            ),
            width=900,
            height=700
        )

        return fig

    def create_energy_report(self, output_dir: str = './energy_analysis') -> Dict[str, str]:
        """
        Generate complete energy analysis report.

        Args:
            output_dir: Directory to save visualizations

        Returns:
            Dictionary mapping visualization names to file paths
        """
        from pathlib import Path

        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)

        saved_files = {}

        print("Generating energy visualizations...")

        try:
            figures = self.plot_energy_landscape()
            if len(figures) == 1:
                # Single figure - use simple name
                path = output_path / 'energy_landscape.png'
                figures[0].savefig(path, dpi=150, bbox_inches='tight')
                saved_files['energy_landscape'] = str(path)
                plt.close(figures[0])
                print(f"  ✓ Saved energy landscape to {path}")
            else:
                # Multiple figures - save with page numbers
                for i, fig in enumerate(figures):
                    path = output_path / f'energy_landscape_page{i+1}.png'
                    fig.savefig(path, dpi=150, bbox_inches='tight')
                    saved_files[f'energy_landscape_page{i+1}'] = str(path)
                    plt.close(fig)
                print(f"  ✓ Saved energy landscape ({len(figures)} pages) to {output_path}/energy_landscape_page*.png")
        except Exception as e:
            print(f"  ✗ Failed to generate energy landscape: {e}")

        try:
            fig = self.plot_energy_flow()
            path = output_path / 'energy_flow.png'
            fig.savefig(path, dpi=150, bbox_inches='tight')
            saved_files['energy_flow'] = str(path)
            plt.close(fig)
            print(f"  ✓ Saved energy flow to {path}")
        except Exception as e:
            print(f"  ✗ Failed to generate energy flow: {e}")

        try:
            fig = self.plot_prior_evolution()
            path = output_path / 'prior_evolution.png'
            fig.savefig(path, dpi=150, bbox_inches='tight')
            saved_files['prior_evolution'] = str(path)
            plt.close(fig)
            print(f"  ✓ Saved prior evolution to {path}")
        except Exception as e:
            print(f"  ✗ Failed to generate prior evolution: {e}")

        try:
            fig = self.plot_non_equilibrium_indicators()
            path = output_path / 'non_equilibrium_indicators.png'
            fig.savefig(path, dpi=150, bbox_inches='tight')
            saved_files['non_equilibrium'] = str(path)
            plt.close(fig)
            print(f"  ✓ Saved non-equilibrium indicators to {path}")
        except Exception as e:
            print(f"  ✗ Failed to generate non-equilibrium indicators: {e}")

        try:
            fig = self.plot_energy_per_agent(scale=0)
            path = output_path / 'energy_per_agent.png'
            fig.savefig(path, dpi=150, bbox_inches='tight')
            saved_files['energy_per_agent'] = str(path)
            plt.close(fig)
            print(f"  ✓ Saved energy per agent to {path}")
        except Exception as e:
            print(f"  ✗ Failed to generate energy per agent: {e}")

        print(f"\nEnergy analysis complete! Generated {len(saved_files)} outputs in {output_dir}")
        return saved_files