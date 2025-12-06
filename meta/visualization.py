"""
Visualization and Analysis Tools for Meta-Agent Hierarchies

This module provides comprehensive visualization and analysis capabilities for the
hierarchical meta-agent system, including:
- Interactive hierarchy graphs
- Consensus and coherence visualizations
- Multi-scale dynamics and energy landscapes
- Information flow diagrams
- Diagnostic dashboards
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import json
from pathlib import Path
from collections import defaultdict

try:
    import networkx as nx
except ImportError:
    nx = None

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

from meta.emergence import MultiScaleSystem, HierarchicalAgent, ScaleIndex, MetaAgentDescriptor


# ============================================================================
# Data Export and Extraction Utilities
# ============================================================================

@dataclass
class SystemSnapshot:
    """Complete snapshot of system state at a given timestep."""
    time: int
    agents_by_scale: Dict[int, List[Dict[str, Any]]]
    meta_agents: List[Dict[str, Any]]
    condensation_events: List[Dict[str, Any]]
    metrics: Dict[str, Any]


class MetaAgentAnalyzer:
    """Extract and analyze data from MultiScaleSystem for visualization."""

    def __init__(self, system: MultiScaleSystem):
        self.system = system
        self.snapshots: List[SystemSnapshot] = []

    def capture_snapshot(self) -> SystemSnapshot:
        """Capture current system state."""
        agents_by_scale = {}
        meta_agents = []

        for scale, agents in self.system.agents.items():
            agents_by_scale[scale] = []
            for agent in agents:
                agent_data = {
                    'scale': scale,
                    'local_index': agent.local_index,
                    'is_active': agent.is_active,
                    'is_meta': agent.is_meta,
                    'mu_q': agent.mu_q.tolist() if hasattr(agent.mu_q, 'tolist') else agent.mu_q,
                    'mu_p': agent.mu_p.tolist() if hasattr(agent.mu_p, 'tolist') else agent.mu_p,
                    'info_accumulator': agent.info_accumulator,
                }
                agents_by_scale[scale].append(agent_data)

                # Track meta-agent descriptors
                if agent.is_meta and hasattr(agent, 'meta') and agent.meta is not None:
                    desc = agent.meta
                    meta_agent_data = {
                        'scale': desc.scale_index.scale,
                        'local_index': desc.scale_index.local_index,
                        'constituents': [
                            {'scale': c.scale, 'local_index': c.local_index}
                            for c in desc.constituent_indices
                        ],
                        'emergence_time': desc.emergence_time,
                        'belief_coherence': desc.belief_coherence,
                        'model_coherence': desc.model_coherence,
                        'leader_index': desc.leader_index,
                        'leader_score': desc.leader_score,
                        'leadership_distribution': desc.leadership_distribution.tolist(),
                    }
                    meta_agents.append(meta_agent_data)

        # Compute system-level metrics
        metrics = self._compute_metrics()

        snapshot = SystemSnapshot(
            time=self.system.current_time,
            agents_by_scale=agents_by_scale,
            meta_agents=meta_agents,
            condensation_events=[dict(e) for e in self.system.condensation_events],
            metrics=metrics
        )

        self.snapshots.append(snapshot)
        return snapshot

    def _compute_metrics(self) -> Dict[str, Any]:
        """Compute aggregate metrics for the system."""
        metrics = {
            'n_agents_by_scale': {},
            'n_active_by_scale': {},
            'n_meta_by_scale': {},
            'total_agents': 0,
            'total_active': 0,
            'max_scale': 0,
        }

        for scale, agents in self.system.agents.items():
            n_agents = len(agents)
            n_active = sum(1 for a in agents if a.is_active)
            n_meta = sum(1 for a in agents if a.is_meta)

            metrics['n_agents_by_scale'][scale] = n_agents
            metrics['n_active_by_scale'][scale] = n_active
            metrics['n_meta_by_scale'][scale] = n_meta
            metrics['total_agents'] += n_agents
            metrics['total_active'] += n_active
            metrics['max_scale'] = max(metrics['max_scale'], scale)

        return metrics

    def get_consensus_matrix(self, scale: int = 0, metric: str = 'belief') -> np.ndarray:
        """
        Compute pairwise KL divergence matrix for agents at a given scale.

        Args:
            scale: Which scale to analyze
            metric: 'belief' (q) or 'prior' (p)

        Returns:
            NxN matrix of KL divergences
        """
        if scale not in self.system.agents:
            return np.array([])

        agents = self.system.agents[scale]
        n = len(agents)

        if n == 0:
            return np.array([])

        # Get means and covariances
        # For spatial manifolds: use center point for visualization
        is_spatial = not agents[0].base_manifold.is_point

        if is_spatial:
            center_idx = tuple(s//2 for s in agents[0].base_manifold.shape)
            if metric == 'belief':
                mus = np.array([a.mu_q[center_idx] for a in agents])
                Sigmas = np.array([a.Sigma_q[center_idx] for a in agents])
            else:  # prior
                mus = np.array([a.mu_p[center_idx] for a in agents])
                Sigmas = np.array([a.Sigma_p[center_idx] for a in agents])
        else:
            # Point manifold: use fields directly
            if metric == 'belief':
                mus = np.array([a.mu_q for a in agents])
                Sigmas = np.array([a.Sigma_q for a in agents])
            else:  # prior
                mus = np.array([a.mu_p for a in agents])
                Sigmas = np.array([a.Sigma_p for a in agents])

        # Compute pairwise KL divergences
        kl_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    kl_matrix[i, j] = self._kl_divergence(
                        mus[i], Sigmas[i], mus[j], Sigmas[j]
                    )

        return kl_matrix

    @staticmethod
    def _kl_divergence(mu1: np.ndarray, Sigma1: np.ndarray,
                       mu2: np.ndarray, Sigma2: np.ndarray) -> float:
        """Compute KL(N(mu1, Sigma1) || N(mu2, Sigma2))."""
        d = len(mu1)

        # Add small regularization for numerical stability
        Sigma1_reg = Sigma1 + 1e-8 * np.eye(d)
        Sigma2_reg = Sigma2 + 1e-8 * np.eye(d)

        try:
            Sigma2_inv = np.linalg.inv(Sigma2_reg)
            term1 = np.trace(Sigma2_inv @ Sigma1_reg)
            term2 = (mu2 - mu1).T @ Sigma2_inv @ (mu2 - mu1)
            term3 = np.log(np.linalg.det(Sigma2_reg) / np.linalg.det(Sigma1_reg))
            kl = 0.5 * (term1 + term2 - d + term3)
            return max(0, kl)  # Ensure non-negative
        except np.linalg.LinAlgError:
            return np.inf

    def get_hierarchy_edges(self) -> List[Tuple[str, str, Dict]]:
        """
        Extract parent-child relationships as graph edges.

        Returns:
            List of (parent_id, child_id, edge_data) tuples
        """
        edges = []

        for scale, agents in self.system.agents.items():
            for agent in agents:
                if agent.is_meta and hasattr(agent, 'meta') and agent.meta is not None:
                    desc = agent.meta
                    parent_id = f"s{desc.scale_index.scale}_i{desc.scale_index.local_index}"

                    for const_idx in desc.constituent_indices:
                        child_id = f"s{const_idx.scale}_i{const_idx.local_index}"
                        edge_data = {
                            'belief_coherence': desc.belief_coherence,
                            'model_coherence': desc.model_coherence,
                            'emergence_time': desc.emergence_time,
                        }
                        edges.append((parent_id, child_id, edge_data))

        return edges

    def export_to_json(self, filepath: str):
        """Export all snapshots to JSON file."""
        data = {
            'snapshots': [asdict(s) for s in self.snapshots],
            'system_config': {
                'max_emergence_levels': self.system.max_emergence_levels,
                'max_meta_membership': self.system.max_meta_membership,
                'max_total_agents': self.system.max_total_agents,
            }
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"Exported {len(self.snapshots)} snapshots to {filepath}")


# ============================================================================
# Hierarchy Visualization
# ============================================================================

class HierarchyVisualizer:
    """Visualize hierarchical structure of meta-agents."""

    def __init__(self, analyzer: MetaAgentAnalyzer):
        self.analyzer = analyzer

    def plot_hierarchy_tree(self,
                           snapshot_idx: int = -1,
                           layout: str = 'hierarchical',
                           figsize: Tuple[int, int] = (14, 10)) -> plt.Figure:
        """
        Plot hierarchy as a tree using matplotlib.

        Args:
            snapshot_idx: Which snapshot to visualize (-1 for latest)
            layout: 'hierarchical' or 'spring'
            figsize: Figure size
        """
        if not nx:
            raise ImportError("NetworkX required for hierarchy visualization")

        if not self.analyzer.snapshots:
            raise ValueError("No snapshots captured. Call capture_snapshot() first.")

        snapshot = self.analyzer.snapshots[snapshot_idx]

        # Build graph
        G = nx.DiGraph()

        # Add all agents as nodes
        for scale, agents in snapshot.agents_by_scale.items():
            for agent in agents:
                node_id = f"s{scale}_i{agent['local_index']}"
                G.add_node(node_id,
                          scale=scale,
                          local_index=agent['local_index'],
                          is_active=agent['is_active'],
                          is_meta=agent['is_meta'])

        # Add edges from meta-agents to constituents
        for meta in snapshot.meta_agents:
            parent_id = f"s{meta['scale']}_i{meta['local_index']}"
            for const in meta['constituents']:
                child_id = f"s{const['scale']}_i{const['local_index']}"
                G.add_edge(parent_id, child_id,
                          belief_coherence=meta['belief_coherence'],
                          model_coherence=meta['model_coherence'])

        # Create layout
        if layout == 'hierarchical':
            pos = self._hierarchical_layout(G)
        else:
            pos = nx.spring_layout(G, k=2, iterations=50)

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Draw nodes by scale
        scales = set(data['scale'] for _, data in G.nodes(data=True))
        colors = plt.cm.viridis(np.linspace(0, 1, len(scales)))

        for scale, color in zip(sorted(scales), colors):
            nodes = [n for n, d in G.nodes(data=True) if d['scale'] == scale]
            active_nodes = [n for n in nodes if G.nodes[n]['is_active']]
            inactive_nodes = [n for n in nodes if not G.nodes[n]['is_active']]

            # Active agents: solid circles
            if active_nodes:
                nx.draw_networkx_nodes(G, pos, nodelist=active_nodes,
                                      node_color=[color], node_size=500,
                                      alpha=0.9, ax=ax, label=f'Scale {scale} (active)')

            # Inactive agents: hollow circles
            if inactive_nodes:
                nx.draw_networkx_nodes(G, pos, nodelist=inactive_nodes,
                                      node_color='white', edgecolors=[color],
                                      node_size=500, alpha=0.5, ax=ax,
                                      linewidths=2)

        # Draw edges with varying thickness based on coherence
        edges = G.edges()
        if edges:
            edge_coherences = [G[u][v].get('belief_coherence', 0.5) for u, v in edges]
            widths = [1 + 3 * c for c in edge_coherences]
            nx.draw_networkx_edges(G, pos, width=widths, alpha=0.4,
                                  edge_color='gray', arrows=True,
                                  arrowsize=15, ax=ax)

        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold', ax=ax)

        ax.set_title(f'Hierarchical Meta-Agent Structure (t={snapshot.time})',
                    fontsize=16, fontweight='bold')
        ax.legend(loc='upper left', fontsize=10)
        ax.axis('off')

        plt.tight_layout()
        return fig

    @staticmethod
    def _hierarchical_layout(G: 'nx.DiGraph') -> Dict[str, Tuple[float, float]]:
        """Create hierarchical layout with scales on y-axis."""
        pos = {}

        # Group nodes by scale
        nodes_by_scale = defaultdict(list)
        for node, data in G.nodes(data=True):
            nodes_by_scale[data['scale']].append(node)

        # Position nodes
        for scale in sorted(nodes_by_scale.keys(), reverse=True):
            nodes = nodes_by_scale[scale]
            n = len(nodes)
            y = scale

            for i, node in enumerate(nodes):
                x = (i - n / 2) * 2
                pos[node] = (x, y)

        return pos

    def plot_interactive_hierarchy(self, snapshot_idx: int = -1) -> Optional[go.Figure]:
        """
        Create interactive hierarchy visualization using Plotly.

        Args:
            snapshot_idx: Which snapshot to visualize (-1 for latest)

        Returns:
            Plotly figure or None if plotly not available
        """
        if not HAS_PLOTLY:
            print("Plotly not available. Install with: pip install plotly")
            return None

        if not nx:
            raise ImportError("NetworkX required for hierarchy visualization")

        if not self.analyzer.snapshots:
            raise ValueError("No snapshots captured. Call capture_snapshot() first.")

        snapshot = self.analyzer.snapshots[snapshot_idx]

        # Build graph (same as matplotlib version)
        G = nx.DiGraph()

        for scale, agents in snapshot.agents_by_scale.items():
            for agent in agents:
                node_id = f"s{scale}_i{agent['local_index']}"
                G.add_node(node_id,
                          scale=scale,
                          local_index=agent['local_index'],
                          is_active=agent['is_active'],
                          is_meta=agent['is_meta'])

        for meta in snapshot.meta_agents:
            parent_id = f"s{meta['scale']}_i{meta['local_index']}"
            for const in meta['constituents']:
                child_id = f"s{const['scale']}_i{const['local_index']}"
                G.add_edge(parent_id, child_id,
                          belief_coherence=meta['belief_coherence'],
                          model_coherence=meta['model_coherence'])

        # Layout
        pos = self._hierarchical_layout(G)

        # Edge traces
        edge_traces = []
        for edge in G.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            coherence = edge[2].get('belief_coherence', 0.5)

            edge_trace = go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode='lines',
                line=dict(width=1 + 3 * coherence, color='rgba(125, 125, 125, 0.5)'),
                hoverinfo='none',
                showlegend=False
            )
            edge_traces.append(edge_trace)

        # Node traces (by scale)
        node_traces = []
        scales = sorted(set(data['scale'] for _, data in G.nodes(data=True)))
        colors = px.colors.sample_colorscale('Viridis', [i / (len(scales) - 1) for i in range(len(scales))])

        for scale, color in zip(scales, colors):
            nodes = [n for n, d in G.nodes(data=True) if d['scale'] == scale]

            x_vals = [pos[n][0] for n in nodes]
            y_vals = [pos[n][1] for n in nodes]
            active_status = [G.nodes[n]['is_active'] for n in nodes]
            meta_status = [G.nodes[n]['is_meta'] for n in nodes]

            hover_text = [
                f"Node: {n}<br>Scale: {scale}<br>Active: {active}<br>Meta: {meta}"
                for n, active, meta in zip(nodes, active_status, meta_status)
            ]

            node_trace = go.Scatter(
                x=x_vals,
                y=y_vals,
                mode='markers+text',
                text=nodes,
                textposition='top center',
                textfont=dict(size=8),
                hovertext=hover_text,
                hoverinfo='text',
                marker=dict(
                    size=15,
                    color=color,
                    opacity=[0.9 if a else 0.3 for a in active_status],
                    line=dict(width=2, color='white')
                ),
                name=f'Scale {scale}'
            )
            node_traces.append(node_trace)

        # Create figure
        fig = go.Figure(data=edge_traces + node_traces)

        fig.update_layout(
            title=f'Interactive Hierarchical Meta-Agent Structure (t={snapshot.time})',
            showlegend=True,
            hovermode='closest',
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, title='Scale'),
            height=700,
            template='plotly_white'
        )

        return fig


# ============================================================================
# Consensus and Coherence Visualization
# ============================================================================

class ConsensusVisualizer:
    """Visualize consensus formation and coherence dynamics."""

    def __init__(self, analyzer: MetaAgentAnalyzer):
        self.analyzer = analyzer

    def plot_consensus_matrix(self,
                             scale: int = 0,
                             metric: str = 'belief',
                             snapshot_idx: int = -1,
                             figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
        """
        Plot consensus matrix (pairwise KL divergences).

        Args:
            scale: Which scale to analyze
            metric: 'belief' or 'prior'
            snapshot_idx: Which snapshot (-1 for latest)
            figsize: Figure size
        """
        kl_matrix = self.analyzer.get_consensus_matrix(scale, metric)

        if kl_matrix.size == 0:
            raise ValueError(f"No agents at scale {scale}")

        snapshot = self.analyzer.snapshots[snapshot_idx]

        fig, ax = plt.subplots(figsize=figsize)

        # Plot heatmap
        im = ax.imshow(kl_matrix, cmap='RdYlGn_r', aspect='auto',
                      vmin=0, vmax=min(10, np.percentile(kl_matrix, 95)))

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('KL Divergence', rotation=270, labelpad=20, fontsize=12)

        # Labels
        n = kl_matrix.shape[0]
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels([f'A{i}' for i in range(n)], fontsize=8)
        ax.set_yticklabels([f'A{i}' for i in range(n)], fontsize=8)

        ax.set_title(f'Consensus Matrix - Scale {scale} ({metric.capitalize()}) at t={snapshot.time}',
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Agent Index', fontsize=12)
        ax.set_ylabel('Agent Index', fontsize=12)

        plt.tight_layout()
        return fig

    def plot_consensus_evolution(self,
                                scale: int = 0,
                                metric: str = 'belief',
                                figsize: Tuple[int, int] = (15, 5)) -> plt.Figure:
        """
        Animate consensus matrix evolution over snapshots.

        Args:
            scale: Which scale to analyze
            metric: 'belief' or 'prior'
            figsize: Figure size
        """
        if len(self.analyzer.snapshots) < 3:
            raise ValueError("Need at least 3 snapshots for evolution plot")

        # Select snapshots to display (beginning, middle, end)
        indices = [0, len(self.analyzer.snapshots) // 2, -1]

        fig, axes = plt.subplots(1, 3, figsize=figsize)

        for i, (ax, idx) in enumerate(zip(axes, indices)):
            # Temporarily set system to snapshot state (simplified - just get matrix)
            snapshot = self.analyzer.snapshots[idx]
            kl_matrix = self.analyzer.get_consensus_matrix(scale, metric)

            if kl_matrix.size == 0:
                ax.text(0.5, 0.5, f'No agents at scale {scale}',
                       ha='center', va='center', transform=ax.transAxes)
                continue

            im = ax.imshow(kl_matrix, cmap='RdYlGn_r', aspect='auto',
                          vmin=0, vmax=min(10, np.percentile(kl_matrix, 95)))

            ax.set_title(f't={snapshot.time}', fontsize=12, fontweight='bold')
            ax.set_xlabel('Agent')
            ax.set_ylabel('Agent' if i == 0 else '')

            if i == 2:
                plt.colorbar(im, ax=ax, label='KL Divergence')

        fig.suptitle(f'Consensus Matrix Evolution - Scale {scale} ({metric.capitalize()})',
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        return fig

    def plot_coherence_trajectories(self, figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
        """
        Plot belief and model coherence over time for all meta-agents.

        Args:
            figsize: Figure size
        """
        if not self.analyzer.snapshots:
            raise ValueError("No snapshots captured")

        # Collect coherence data for each meta-agent
        meta_data = defaultdict(lambda: {'times': [], 'belief_coh': [], 'model_coh': []})

        for snapshot in self.analyzer.snapshots:
            for meta in snapshot.meta_agents:
                meta_id = f"s{meta['scale']}_i{meta['local_index']}"
                meta_data[meta_id]['times'].append(snapshot.time)
                meta_data[meta_id]['belief_coh'].append(meta['belief_coherence'])
                meta_data[meta_id]['model_coh'].append(meta['model_coherence'])

        if not meta_data:
            raise ValueError("No meta-agents found in snapshots")

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)

        # Plot belief coherence
        for meta_id, data in meta_data.items():
            ax1.plot(data['times'], data['belief_coh'], marker='o',
                    label=meta_id, alpha=0.7, linewidth=2)

        ax1.axhline(y=0.95, color='r', linestyle='--', alpha=0.5, label='High coherence')
        ax1.set_ylabel('Belief Coherence', fontsize=12)
        ax1.set_title('Meta-Agent Coherence Trajectories', fontsize=14, fontweight='bold')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax1.grid(alpha=0.3)

        # Plot model coherence
        for meta_id, data in meta_data.items():
            ax2.plot(data['times'], data['model_coh'], marker='s',
                    label=meta_id, alpha=0.7, linewidth=2)

        ax2.axhline(y=0.95, color='r', linestyle='--', alpha=0.5, label='High coherence')
        ax2.set_xlabel('Time', fontsize=12)
        ax2.set_ylabel('Model Coherence', fontsize=12)
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax2.grid(alpha=0.3)

        plt.tight_layout()
        return fig


# ============================================================================
# Multi-Scale Dynamics Visualization
# ============================================================================

class DynamicsVisualizer:
    """Visualize multi-scale dynamics and population flows."""

    def __init__(self, analyzer: MetaAgentAnalyzer):
        self.analyzer = analyzer

    def plot_scale_occupancy(self, figsize: Tuple[int, int] = (14, 6)) -> plt.Figure:
        """
        Plot heatmap showing number of agents at each scale over time.

        Args:
            figsize: Figure size
        """
        if not self.analyzer.snapshots:
            raise ValueError("No snapshots captured")

        # Extract data
        times = [s.time for s in self.analyzer.snapshots]
        max_scale = max(s.metrics['max_scale'] for s in self.analyzer.snapshots)

        # Build matrix: rows=scales, cols=time
        occupancy_matrix = np.zeros((max_scale + 1, len(times)))

        for t_idx, snapshot in enumerate(self.analyzer.snapshots):
            for scale in range(max_scale + 1):
                occupancy_matrix[scale, t_idx] = snapshot.metrics['n_active_by_scale'].get(scale, 0)

        fig, ax = plt.subplots(figsize=figsize)

        # Plot heatmap
        im = ax.imshow(occupancy_matrix, aspect='auto', cmap='YlOrRd',
                      interpolation='nearest', origin='lower')

        # Overlay condensation events
        for event in self.analyzer.system.condensation_events:
            event_time_idx = None
            for i, t in enumerate(times):
                if t >= event['time']:
                    event_time_idx = i
                    break

            if event_time_idx is not None:
                target_scale = event['target_scale']
                ax.plot(event_time_idx, target_scale, 'w*', markersize=15,
                       markeredgecolor='black', markeredgewidth=1)

        # Colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Active Agents', rotation=270, labelpad=20, fontsize=12)

        # Labels
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('Scale (ζ)', fontsize=12)
        ax.set_title('Multi-Scale Agent Occupancy Heatmap\n(★ = condensation events)',
                    fontsize=14, fontweight='bold')

        # Set ticks
        ax.set_yticks(range(max_scale + 1))
        ax.set_yticklabels([f'ζ={i}' for i in range(max_scale + 1)])

        plt.tight_layout()
        return fig

    def plot_condensation_timeline(self, figsize: Tuple[int, int] = (14, 6)) -> plt.Figure:
        """
        Plot timeline of condensation events.

        Args:
            figsize: Figure size
        """
        events = self.analyzer.system.condensation_events

        if not events:
            raise ValueError("No condensation events recorded")

        fig, ax = plt.subplots(figsize=figsize)

        # Group events by target scale
        events_by_scale = defaultdict(list)
        for event in events:
            events_by_scale[event['target_scale']].append(event)

        # Plot events
        colors = plt.cm.viridis(np.linspace(0, 1, len(events_by_scale)))

        for (scale, scale_events), color in zip(sorted(events_by_scale.items()), colors):
            times = [e['time'] for e in scale_events]
            sizes = [e['n_constituents'] * 100 for e in scale_events]
            coherences = [e['coherence']['belief'] for e in scale_events]

            scatter = ax.scatter(times, [scale] * len(times), s=sizes, c=coherences,
                               cmap='RdYlGn', vmin=0, vmax=1, alpha=0.7,
                               edgecolors='black', linewidth=1.5,
                               label=f'Scale {scale}')

        # Colorbar for coherence
        sm = plt.cm.ScalarMappable(cmap='RdYlGn', norm=plt.Normalize(vmin=0, vmax=1))
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('Belief Coherence', rotation=270, labelpad=20, fontsize=12)

        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('Target Scale', fontsize=12)
        ax.set_title('Condensation Event Timeline\n(size = # constituents)',
                    fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3, axis='x')

        plt.tight_layout()
        return fig

    def plot_population_flows(self, figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
        """
        Plot stacked area chart showing agent population by scale over time.

        Args:
            figsize: Figure size
        """
        if not self.analyzer.snapshots:
            raise ValueError("No snapshots captured")

        times = [s.time for s in self.analyzer.snapshots]
        max_scale = max(s.metrics['max_scale'] for s in self.analyzer.snapshots)

        # Build data matrix
        populations = np.zeros((max_scale + 1, len(times)))
        for t_idx, snapshot in enumerate(self.analyzer.snapshots):
            for scale in range(max_scale + 1):
                populations[scale, t_idx] = snapshot.metrics['n_active_by_scale'].get(scale, 0)

        fig, ax = plt.subplots(figsize=figsize)

        # Stacked area plot
        colors = plt.cm.viridis(np.linspace(0, 1, max_scale + 1))
        ax.stackplot(times, populations, labels=[f'ζ={i}' for i in range(max_scale + 1)],
                    colors=colors, alpha=0.8)

        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('Active Agents', fontsize=12)
        ax.set_title('Population Flow Across Scales', fontsize=14, fontweight='bold')
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(alpha=0.3, axis='y')

        plt.tight_layout()
        return fig


# ============================================================================
# Utility Functions
# ============================================================================

def create_analysis_report(analyzer: MetaAgentAnalyzer,
                          output_dir: str = './meta_analysis') -> Dict[str, str]:
    """
    Generate complete analysis report with all visualizations.

    Args:
        analyzer: MetaAgentAnalyzer with captured snapshots
        output_dir: Directory to save visualizations

    Returns:
        Dictionary mapping visualization names to file paths
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    saved_files = {}

    # Initialize visualizers
    hierarchy_viz = HierarchyVisualizer(analyzer)
    consensus_viz = ConsensusVisualizer(analyzer)
    dynamics_viz = DynamicsVisualizer(analyzer)

    # Generate visualizations
    print("Generating visualizations...")

    try:
        # 1. Hierarchy tree
        fig = hierarchy_viz.plot_hierarchy_tree()
        path = output_path / 'hierarchy_tree.png'
        fig.savefig(path, dpi=150, bbox_inches='tight')
        saved_files['hierarchy_tree'] = str(path)
        plt.close(fig)
        print(f"  ✓ Saved hierarchy tree to {path}")
    except Exception as e:
        print(f"  ✗ Failed to generate hierarchy tree: {e}")

    try:
        # 2. Consensus matrix
        fig = consensus_viz.plot_consensus_matrix(scale=0)
        path = output_path / 'consensus_matrix.png'
        fig.savefig(path, dpi=150, bbox_inches='tight')
        saved_files['consensus_matrix'] = str(path)
        plt.close(fig)
        print(f"  ✓ Saved consensus matrix to {path}")
    except Exception as e:
        print(f"  ✗ Failed to generate consensus matrix: {e}")

    try:
        # 3. Scale occupancy heatmap
        fig = dynamics_viz.plot_scale_occupancy()
        path = output_path / 'scale_occupancy.png'
        fig.savefig(path, dpi=150, bbox_inches='tight')
        saved_files['scale_occupancy'] = str(path)
        plt.close(fig)
        print(f"  ✓ Saved scale occupancy to {path}")
    except Exception as e:
        print(f"  ✗ Failed to generate scale occupancy: {e}")

    try:
        # 4. Condensation timeline
        fig = dynamics_viz.plot_condensation_timeline()
        path = output_path / 'condensation_timeline.png'
        fig.savefig(path, dpi=150, bbox_inches='tight')
        saved_files['condensation_timeline'] = str(path)
        plt.close(fig)
        print(f"  ✓ Saved condensation timeline to {path}")
    except Exception as e:
        print(f"  ✗ Failed to generate condensation timeline: {e}")

    try:
        # 5. Coherence trajectories
        fig = consensus_viz.plot_coherence_trajectories()
        path = output_path / 'coherence_trajectories.png'
        fig.savefig(path, dpi=150, bbox_inches='tight')
        saved_files['coherence_trajectories'] = str(path)
        plt.close(fig)
        print(f"  ✓ Saved coherence trajectories to {path}")
    except Exception as e:
        print(f"  ✗ Failed to generate coherence trajectories: {e}")

    try:
        # 6. Population flows
        fig = dynamics_viz.plot_population_flows()
        path = output_path / 'population_flows.png'
        fig.savefig(path, dpi=150, bbox_inches='tight')
        saved_files['population_flows'] = str(path)
        plt.close(fig)
        print(f"  ✓ Saved population flows to {path}")
    except Exception as e:
        print(f"  ✗ Failed to generate population flows: {e}")

    # Export data
    try:
        json_path = output_path / 'snapshots.json'
        analyzer.export_to_json(str(json_path))
        saved_files['data'] = str(json_path)
        print(f"  ✓ Exported data to {json_path}")
    except Exception as e:
        print(f"  ✗ Failed to export data: {e}")

    print(f"\nAnalysis complete! Generated {len(saved_files)} outputs in {output_dir}")
    return saved_files