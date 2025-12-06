"""
Participatory Dynamics Diagnostics

Tracks and validates that participatory "it from bit" dynamics are working:
1. Individual agent energy evolution (self, belief align, prior align)
2. Per-scale energy aggregates
3. Prior evolution when meta-agents form
4. Non-equilibrium indicators (gradients, energy flux)
5. Cross-scale information flow

Author: Chris & Christine
Date: November 2025
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import matplotlib.pyplot as plt

from meta.emergence import MultiScaleSystem, HierarchicalAgent
from math_utils.numerical_utils import kl_gaussian


@dataclass
class AgentEnergySnapshot:
    """Energy decomposition for a single agent at one timestep"""
    step: int
    agent_id: str
    scale: int

    # Energy components
    E_self: float           # KL(q||p) - self-energy
    E_belief_align: float   # Belief alignment with neighbors
    E_prior_align: float    # Prior alignment with neighbors
    E_obs: float           # Observation energy (if applicable)
    E_total: float         # Total energy

    # Gradient info
    grad_mu_norm: float    # Gradient magnitude for mean
    grad_sigma_norm: float # Gradient magnitude for covariance

    # State info
    is_active: bool
    has_parent: bool
    num_constituents: int  # If meta-agent


@dataclass
class ScaleEnergySnapshot:
    """Aggregated energy for entire scale at one timestep"""
    step: int
    scale: int

    n_agents: int
    n_active: int

    # Energy aggregates
    total_energy: float
    avg_energy_per_agent: float

    # Component breakdown
    total_self_energy: float
    total_belief_align: float
    total_prior_align: float

    # Coherence
    avg_coherence: float
    coherence_std: float


class ParticipatoryDiagnostics:
    """
    Diagnostics for validating participatory dynamics

    Tracks:
    - Individual agent energies over time
    - Per-scale energy evolution
    - Prior changes when meta-agents form
    - Non-equilibrium indicators
    """

    def __init__(self,
                 system: MultiScaleSystem,
                 track_agent_ids: Optional[List[str]] = None,
                 compute_full_energies: bool = True):
        """
        Initialize diagnostics

        Args:
            system: MultiScaleSystem to monitor
            track_agent_ids: Specific agent IDs to track in detail (default: first 3 at scale-0)
            compute_full_energies: If True, compute belief/prior alignment energies (EXPENSIVE!)
                                   If False, only compute self-energy (fast, recommended)
        """
        self.system = system
        self.compute_full_energies = compute_full_energies

        # If not specified, track first 3 scale-0 agents
        if track_agent_ids is None:
            if 0 in system.agents and len(system.agents[0]) > 0:
                track_agent_ids = [a.agent_id for a in system.agents[0][:3]]
            else:
                track_agent_ids = []

        self.track_agent_ids = set(track_agent_ids)

        # Storage
        self.agent_history: Dict[str, List[AgentEnergySnapshot]] = defaultdict(list)
        self.scale_history: Dict[int, List[ScaleEnergySnapshot]] = defaultdict(list)

        # Prior change tracking (for validation)
        self._previous_priors: Dict[Tuple[int, int], np.ndarray] = {}
        self.prior_changes: List[Tuple[int, str, float]] = []  # (step, agent_id, KL_change)

        # Condensation events
        self.condensation_events: List[Tuple[int, int, int]] = []  # (step, scale, n_agents)

    def record_snapshot(self, step: int):
        """
        Record full system snapshot at current step

        Args:
            step: Current training step
        """
        # Record individual tracked agents
        # Use list() to avoid RuntimeError if dict changes during iteration
        for scale, agents in list(self.system.agents.items()):
            for agent in agents:
                if agent.agent_id in self.track_agent_ids:
                    snapshot = self._compute_agent_energy(agent, step)
                    self.agent_history[agent.agent_id].append(snapshot)

        # Record per-scale aggregates
        for scale, agents in list(self.system.agents.items()):
            snapshot = self._compute_scale_energy(scale, agents, step)
            self.scale_history[scale].append(snapshot)

        # Track prior changes
        self._track_prior_changes(step)

    def _compute_agent_energy(self, agent: HierarchicalAgent, step: int) -> AgentEnergySnapshot:
        """Compute energy decomposition for single agent"""
        from gradients.free_energy_clean import compute_self_energy

        config = self.system.system_config

        # Self-energy: KL(q||p) - always computed (fast)
        E_self = compute_self_energy(agent, lambda_self=config.lambda_self)

        # Alignment energies - only if requested (EXPENSIVE!)
        E_belief_align = 0.0
        E_prior_align = 0.0

        if self.compute_full_energies:
            from gradients.free_energy_clean import (
                compute_belief_alignment_energy,
                compute_prior_alignment_energy
            )
            from meta.gradient_adapter import create_gradient_adapter

            # Create adapter to make MultiScaleSystem compatible with energy functions
            # This provides flat agent list and proper neighbor lookup
            adapter = create_gradient_adapter(self.system)

            # Find agent index in flat active agent list
            try:
                agent_idx = adapter.agents.index(agent)
            except ValueError:
                agent_idx = None

            if agent_idx is not None:
                if config.has_belief_alignment:
                    E_belief_align = compute_belief_alignment_energy(adapter, agent_idx)

                if config.has_prior_alignment:
                    E_prior_align = compute_prior_alignment_energy(adapter, agent_idx)

        E_obs = 0.0  # Observation energy not tracked per agent
        E_total = E_self + E_belief_align + E_prior_align + E_obs

        # Gradient norms (proxy)
        grad_mu_norm = np.linalg.norm(agent.mu_q - agent.mu_p)
        grad_sigma_norm = np.linalg.norm(agent.L_q - agent.L_p)

        return AgentEnergySnapshot(
            step=step,
            agent_id=agent.agent_id,
            scale=agent.scale,
            E_self=E_self,
            E_belief_align=E_belief_align,
            E_prior_align=E_prior_align,
            E_obs=E_obs,
            E_total=E_total,
            grad_mu_norm=grad_mu_norm,
            grad_sigma_norm=grad_sigma_norm,
            is_active=agent.is_active,
            has_parent=(agent.parent_meta is not None),
            num_constituents=len(agent.constituents) if hasattr(agent, 'constituents') else 0
        )

    def _compute_scale_energy(self, scale: int, agents: List[HierarchicalAgent],
                             step: int) -> ScaleEnergySnapshot:
        """Compute aggregated energy for entire scale"""
        from gradients.free_energy_clean import compute_self_energy

        config = self.system.system_config
        n_agents = len(agents)
        n_active = sum(1 for a in agents if a.is_active)

        # Compute energies
        total_self = 0.0
        total_belief = 0.0
        total_prior = 0.0
        coherences = []

        # Build adapter once if needed for full energies (expensive!)
        adapter = None
        if self.compute_full_energies:
            from meta.gradient_adapter import create_gradient_adapter
            adapter = create_gradient_adapter(self.system)

        for agent in agents:
            if agent.is_active:
                # Self-energy - always computed (fast)
                E_self = compute_self_energy(agent, lambda_self=config.lambda_self)
                total_self += E_self

                # Alignment energies - only if requested (EXPENSIVE!)
                if self.compute_full_energies and adapter is not None:
                    from gradients.free_energy_clean import (
                        compute_belief_alignment_energy,
                        compute_prior_alignment_energy
                    )

                    # Find agent index in flat active agent list
                    try:
                        agent_idx = adapter.agents.index(agent)

                        if config.has_belief_alignment:
                            total_belief += compute_belief_alignment_energy(adapter, agent_idx)

                        if config.has_prior_alignment:
                            total_prior += compute_prior_alignment_energy(adapter, agent_idx)
                    except ValueError:
                        pass  # Agent not in active list

                # Coherence (if meta-agent)
                if hasattr(agent, 'meta') and agent.meta is not None:
                    coherences.append(agent.meta.belief_coherence)

        total_energy = total_self + total_belief + total_prior
        avg_energy = total_energy / max(n_active, 1)

        avg_coherence = np.mean(coherences) if coherences else 0.0
        coherence_std = np.std(coherences) if coherences else 0.0

        return ScaleEnergySnapshot(
            step=step,
            scale=scale,
            n_agents=n_agents,
            n_active=n_active,
            total_energy=total_energy,
            avg_energy_per_agent=avg_energy,
            total_self_energy=total_self,
            total_belief_align=total_belief,
            total_prior_align=total_prior,
            avg_coherence=avg_coherence,
            coherence_std=coherence_std
        )

    def _track_prior_changes(self, step: int):
        """Track how much priors change between steps"""

        # Use list() to avoid RuntimeError if dict changes during iteration
        for scale, agents in list(self.system.agents.items()):
            for i, agent in enumerate(agents):
                key = (scale, i)
                current_prior = agent.mu_p.copy()

                if key in self._previous_priors:
                    prev_prior = self._previous_priors[key]
                    # Simple L2 change (could use KL for distributions)
                    change = np.linalg.norm(current_prior - prev_prior)

                    if change > 1e-6:  # Threshold for significant change
                        self.prior_changes.append((step, agent.agent_id, change))

                self._previous_priors[key] = current_prior

    def detect_non_equilibrium(self, window: int = 10) -> Dict:
        """
        Detect if system is in non-equilibrium regime

        Args:
            window: Number of recent steps to analyze

        Returns:
            Dict with non-equilibrium indicators
        """
        if not self.scale_history or 0 not in self.scale_history:
            return {"status": "insufficient_data"}

        # Get recent energy history for scale-0
        scale_0_history = self.scale_history[0]
        if len(scale_0_history) < window + 1:
            return {"status": "insufficient_data"}

        recent = scale_0_history[-window:]
        energies = [s.total_energy for s in recent]

        # Check if energy is changing
        energy_std = np.std(energies)
        energy_trend = np.polyfit(range(len(energies)), energies, 1)[0]  # Slope

        # Non-equilibrium if energy still changing significantly
        is_non_eq = energy_std > 0.01 or abs(energy_trend) > 0.001

        return {
            "status": "ok",
            "is_non_equilibrium": is_non_eq,
            "energy_std": energy_std,
            "energy_trend": energy_trend,
            "recent_energies": energies
        }

    def validate_prior_evolution(self) -> Dict:
        """
        Validate that priors are evolving (participatory dynamics working)

        Returns:
            Dict with validation results
        """
        if not self.prior_changes:
            return {
                "status": "no_changes_detected",
                "n_changes": 0
            }

        # Group by agent
        agent_changes = defaultdict(list)
        for step, agent_id, change in self.prior_changes:
            agent_changes[agent_id].append((step, change))

        # Stats
        total_changes = len(self.prior_changes)
        n_agents_changing = len(agent_changes)
        avg_change = np.mean([c for _, _, c in self.prior_changes])

        return {
            "status": "ok",
            "n_changes": total_changes,
            "n_agents_with_changes": n_agents_changing,
            "avg_change_magnitude": avg_change,
            "changes_per_agent": {aid: len(changes) for aid, changes in agent_changes.items()}
        }

    def plot_agent_energies(self, save_path: Optional[str] = None):
        """Plot energy evolution for tracked agents"""

        if not self.agent_history:
            print("No agent history to plot")
            return

        fig, axes = plt.subplots(len(self.agent_history), 1,
                                figsize=(12, 4 * len(self.agent_history)),
                                squeeze=False)

        for idx, (agent_id, history) in enumerate(self.agent_history.items()):
            ax = axes[idx, 0]

            steps = [s.step for s in history]
            E_self = [s.E_self for s in history]
            grad_mu = [s.grad_mu_norm for s in history]

            # Plot energies
            ax2 = ax.twinx()
            ax.plot(steps, E_self, 'b-', label='Self-energy', linewidth=2)
            ax2.plot(steps, grad_mu, 'r--', label='Grad norm', alpha=0.7)

            ax.set_xlabel('Step')
            ax.set_ylabel('Self-energy (KL)', color='b')
            ax2.set_ylabel('Gradient norm', color='r')
            ax.set_title(f'Agent {agent_id} - Scale {history[0].scale}')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper left')
            ax2.legend(loc='upper right')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved agent energy plot: {save_path}")
        else:
            plt.show()

    def plot_scale_energies(self, save_path: Optional[str] = None):
        """Plot per-scale energy evolution"""

        if not self.scale_history:
            print("No scale history to plot")
            return

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        # Plot total energy per scale
        for scale, history in sorted(self.scale_history.items()):
            steps = [s.step for s in history]
            energies = [s.total_energy for s in history]
            ax1.plot(steps, energies, label=f'Scale {scale}', linewidth=2)

        ax1.set_xlabel('Step')
        ax1.set_ylabel('Total Energy')
        ax1.set_title('Energy Evolution per Scale')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot number of active agents per scale
        for scale, history in sorted(self.scale_history.items()):
            steps = [s.step for s in history]
            n_active = [s.n_active for s in history]
            ax2.plot(steps, n_active, 'o-', label=f'Scale {scale}', markersize=3)

        ax2.set_xlabel('Step')
        ax2.set_ylabel('Active Agents')
        ax2.set_title('Active Agents per Scale')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved scale energy plot: {save_path}")
        else:
            plt.show()

    def print_summary(self):
        """Print diagnostic summary"""

        print("\n" + "="*70)
        print("PARTICIPATORY DYNAMICS DIAGNOSTICS")
        print("="*70)

        # Non-equilibrium check
        non_eq = self.detect_non_equilibrium()
        print("\n1. NON-EQUILIBRIUM STATUS")
        print("-"*70)
        if non_eq['status'] == 'ok':
            status_str = "✓ NON-EQUILIBRIUM" if non_eq['is_non_equilibrium'] else "✗ EQUILIBRIUM"
            print(f"   Status: {status_str}")
            print(f"   Energy std: {non_eq['energy_std']:.6f}")
            print(f"   Energy trend: {non_eq['energy_trend']:.6f}")
        else:
            print(f"   Status: {non_eq['status']}")

        # Prior evolution check
        prior_val = self.validate_prior_evolution()
        print("\n2. PRIOR EVOLUTION")
        print("-"*70)
        print(f"   Status: {prior_val['status']}")
        if prior_val['status'] == 'ok':
            print(f"   Total changes: {prior_val['n_changes']}")
            print(f"   Agents changing: {prior_val['n_agents_with_changes']}")
            print(f"   Avg change: {prior_val['avg_change_magnitude']:.6f}")

        # Tracked agents summary
        print("\n3. TRACKED AGENTS")
        print("-"*70)
        for agent_id, history in self.agent_history.items():
            if history:
                latest = history[-1]
                print(f"   {agent_id}: E_self={latest.E_self:.4f}, "
                      f"grad={latest.grad_mu_norm:.4f}, "
                      f"active={latest.is_active}, "
                      f"has_parent={latest.has_parent}")

        # Per-scale summary
        print("\n4. PER-SCALE SUMMARY")
        print("-"*70)
        for scale in sorted(self.scale_history.keys()):
            history = self.scale_history[scale]
            if history:
                latest = history[-1]
                print(f"   Scale {scale}: {latest.n_active}/{latest.n_agents} active, "
                      f"E_total={latest.total_energy:.4f}, "
                      f"E_avg={latest.avg_energy_per_agent:.4f}")

        print("\n" + "="*70)