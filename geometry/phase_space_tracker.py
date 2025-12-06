# -*- coding: utf-8 -*-
"""
Created on Thu Nov 27 13:42:35 2025

@author: chris and christine
"""

"""
Phase Space Orbit Tracker for Hamiltonian Dynamics
===================================================

Enhanced tracking to detect stable orbits, quasi-periodic motion,
and other dynamical signatures in belief evolution.

Key Features:
- Full (μ, π_μ) phase space recording per agent
- Σ eigenvalue tracking (precision dynamics)
- Orbit detection via recurrence analysis
- Poincaré section computation
- Lyapunov exponent estimation

Author: Chris
Date: November 2025
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
from pathlib import Path


@dataclass
class PhaseSpaceSnapshot:
    """Single timestep snapshot of full phase space."""
    step: int
    t: float  # Continuous time
    
    # Per-agent phase space coordinates
    mu: Dict[int, np.ndarray] = field(default_factory=dict)      # agent_id -> μ
    pi_mu: Dict[int, np.ndarray] = field(default_factory=dict)   # agent_id -> π_μ
    
    # Covariance tracking (eigenvalues capture precision dynamics)
    sigma_eigenvalues: Dict[int, np.ndarray] = field(default_factory=dict)
    
    # Optional: gauge phase space
    phi: Dict[int, np.ndarray] = field(default_factory=dict)
    pi_phi: Dict[int, np.ndarray] = field(default_factory=dict)
    
    # Energy at this snapshot
    kinetic: float = 0.0
    potential: float = 0.0
    total: float = 0.0


@dataclass  
class OrbitDiagnostics:
    """Diagnostics for orbit detection."""
    # Recurrence statistics
    recurrence_times: List[float] = field(default_factory=list)
    mean_recurrence: float = 0.0
    recurrence_std: float = 0.0
    
    # Orbit classification
    is_periodic: bool = False
    is_quasi_periodic: bool = False
    is_chaotic: bool = False
    estimated_period: float = 0.0
    
    # Lyapunov exponent (positive = chaotic)
    lyapunov_exponent: float = 0.0
    
    # Phase space volume (Liouville theorem check)
    phase_volume_initial: float = 0.0
    phase_volume_final: float = 0.0
    volume_drift: float = 0.0


class PhaseSpaceTracker:
    """
    Track full phase space evolution during Hamiltonian dynamics.
    
    Records (μ, π_μ, Σ) trajectories for all agents and provides
    tools for orbit detection and dynamical analysis.
    """
    
    def __init__(
        self,
        track_interval: int = 1,
        max_snapshots: int = 10000,
        track_sigma_eigenvalues: bool = True,
        track_gauge: bool = False,
        recurrence_threshold: float = 0.1,  # Fraction of initial distance
    ):
        """
        Args:
            track_interval: Record every N steps
            max_snapshots: Maximum snapshots to store (circular buffer)
            track_sigma_eigenvalues: Whether to compute Σ eigenvalues
            track_gauge: Whether to track (φ, π_φ)
            recurrence_threshold: Distance threshold for recurrence detection
        """
        self.track_interval = track_interval
        self.max_snapshots = max_snapshots
        self.track_sigma_eigenvalues = track_sigma_eigenvalues
        self.track_gauge = track_gauge
        self.recurrence_threshold = recurrence_threshold
        
        self.snapshots: List[PhaseSpaceSnapshot] = []
        self.agent_ids: List[int] = []
        self._initialized = False
        
    def record(
        self,
        step: int,
        t: float,
        agents,  # List of Agent objects
        momenta: Dict[int, Dict[str, np.ndarray]],  # agent_id -> {'mu': π_μ, 'phi': π_φ}
        energies: Optional[Tuple[float, float, float]] = None,  # (T, V, H)
    ):
        """
        Record phase space snapshot.
        
        Args:
            step: Discrete step number
            t: Continuous time
            agents: List of Agent objects with mu_q, Sigma_q
            momenta: Dictionary mapping agent_id to momentum dict
            energies: Optional (kinetic, potential, total) tuple
        """
        if step % self.track_interval != 0:
            return
            
        # Initialize agent list on first call
        if not self._initialized:
            self.agent_ids = [a.agent_id for a in agents]
            self._initialized = True
        
        snapshot = PhaseSpaceSnapshot(
            step=step,
            t=t,
            kinetic=energies[0] if energies else 0.0,
            potential=energies[1] if energies else 0.0,
            total=energies[2] if energies else 0.0,
        )
        
        for agent in agents:
            aid = agent.agent_id
            
            # Position (belief mean)
            snapshot.mu[aid] = agent.mu_q.flatten().copy()
            
            # Momentum
            if aid in momenta and 'mu' in momenta[aid]:
                snapshot.pi_mu[aid] = momenta[aid]['mu'].flatten().copy()
            else:
                snapshot.pi_mu[aid] = np.zeros_like(snapshot.mu[aid])
            
            # Covariance eigenvalues (proxy for precision dynamics)
            if self.track_sigma_eigenvalues:
                try:
                    eigvals = np.linalg.eigvalsh(agent.Sigma_q)
                    snapshot.sigma_eigenvalues[aid] = np.sort(eigvals)[::-1]
                except:
                    snapshot.sigma_eigenvalues[aid] = np.array([1.0])
            
            # Gauge tracking
            if self.track_gauge and hasattr(agent, 'phi'):
                snapshot.phi[aid] = agent.phi.flatten().copy()
                if aid in momenta and 'phi' in momenta[aid]:
                    snapshot.pi_phi[aid] = momenta[aid]['phi'].flatten().copy()
                    
        # Circular buffer
        self.snapshots.append(snapshot)
        if len(self.snapshots) > self.max_snapshots:
            self.snapshots.pop(0)
    
    def get_trajectory(self, agent_id: int) -> Dict[str, np.ndarray]:
        """
        Get full trajectory for one agent.
        
        Returns:
            Dictionary with:
                'steps': (N,) step numbers
                't': (N,) continuous times
                'mu': (N, K) belief means
                'pi_mu': (N, K) momenta
                'sigma_eig': (N, K) covariance eigenvalues
                'energy': (N, 3) kinetic, potential, total
        """
        N = len(self.snapshots)
        if N == 0:
            return {}
        
        K = len(self.snapshots[0].mu.get(agent_id, []))
        
        result = {
            'steps': np.array([s.step for s in self.snapshots]),
            't': np.array([s.t for s in self.snapshots]),
            'mu': np.zeros((N, K)),
            'pi_mu': np.zeros((N, K)),
            'energy': np.zeros((N, 3)),
        }
        
        for i, snap in enumerate(self.snapshots):
            if agent_id in snap.mu:
                result['mu'][i] = snap.mu[agent_id]
            if agent_id in snap.pi_mu:
                result['pi_mu'][i] = snap.pi_mu[agent_id]
            result['energy'][i] = [snap.kinetic, snap.potential, snap.total]
            
        if self.track_sigma_eigenvalues and self.snapshots[0].sigma_eigenvalues:
            K_sig = len(self.snapshots[0].sigma_eigenvalues.get(agent_id, []))
            result['sigma_eig'] = np.zeros((N, K_sig))
            for i, snap in enumerate(self.snapshots):
                if agent_id in snap.sigma_eigenvalues:
                    result['sigma_eig'][i] = snap.sigma_eigenvalues[agent_id]
                    
        return result
    
    def compute_recurrence(self, agent_id: int, component: int = 0) -> List[int]:
        """
        Find recurrence times for a single phase space component.
        
        Uses the method of recurrence plots: find times when trajectory
        returns close to a previous point.
        
        Args:
            agent_id: Which agent
            component: Which μ component (0 = first)
            
        Returns:
            List of recurrence times (in steps)
        """
        traj = self.get_trajectory(agent_id)
        if 'mu' not in traj or len(traj['mu']) < 10:
            return []
        
        mu = traj['mu'][:, component]
        pi = traj['pi_mu'][:, component]
        
        # Normalize to unit scale
        mu_range = np.ptp(mu) + 1e-10
        pi_range = np.ptp(pi) + 1e-10
        mu_norm = (mu - mu.mean()) / mu_range
        pi_norm = (pi - pi.mean()) / pi_range
        
        # Initial point
        q0, p0 = mu_norm[0], pi_norm[0]
        threshold = self.recurrence_threshold
        
        recurrence_times = []
        in_neighborhood = True  # Start in neighborhood
        
        for i in range(1, len(mu_norm)):
            dist = np.sqrt((mu_norm[i] - q0)**2 + (pi_norm[i] - p0)**2)
            
            if dist > threshold and in_neighborhood:
                # Left neighborhood
                in_neighborhood = False
            elif dist <= threshold and not in_neighborhood:
                # Returned to neighborhood
                recurrence_times.append(i)
                in_neighborhood = True
                
        return recurrence_times
    
    def detect_orbit(self, agent_id: int) -> OrbitDiagnostics:
        """
        Analyze trajectory to classify orbit type.
        
        Returns:
            OrbitDiagnostics with classification and statistics
        """
        diag = OrbitDiagnostics()
        
        # Get recurrence times for first component
        rec_times = self.compute_recurrence(agent_id, component=0)
        
        if len(rec_times) < 2:
            # Not enough recurrences - might be transient or escaping
            return diag
            
        # Recurrence statistics
        periods = np.diff(rec_times)
        diag.recurrence_times = rec_times
        diag.mean_recurrence = np.mean(periods)
        diag.recurrence_std = np.std(periods)
        
        # Classification based on recurrence regularity
        cv = diag.recurrence_std / (diag.mean_recurrence + 1e-10)  # Coefficient of variation
        
        if cv < 0.1:
            # Very regular recurrence -> periodic
            diag.is_periodic = True
            diag.estimated_period = diag.mean_recurrence
        elif cv < 0.3:
            # Moderately regular -> quasi-periodic
            diag.is_quasi_periodic = True
            diag.estimated_period = diag.mean_recurrence
        else:
            # Irregular recurrence -> possibly chaotic
            diag.is_chaotic = True
            
        # Lyapunov exponent estimation (simplified)
        traj = self.get_trajectory(agent_id)
        if 'mu' in traj and len(traj['mu']) > 100:
            diag.lyapunov_exponent = self._estimate_lyapunov(traj['mu'], traj['pi_mu'])
            
        return diag
    
    def _estimate_lyapunov(
        self, 
        mu: np.ndarray, 
        pi_mu: np.ndarray,
        n_neighbors: int = 5,
    ) -> float:
        """
        Rough Lyapunov exponent estimate via neighbor divergence.
        
        Positive -> chaotic, Zero -> periodic/quasi-periodic, Negative -> stable fixed point
        """
        N, K = mu.shape
        if N < 100:
            return 0.0
            
        # Combine position and momentum
        phase = np.hstack([mu, pi_mu])  # (N, 2K)
        
        # Find divergence rate from initial point
        initial = phase[0]
        
        # Find points that pass close to initial
        distances = np.linalg.norm(phase - initial, axis=1)
        close_indices = np.where(distances < np.percentile(distances, 10))[0]
        close_indices = close_indices[close_indices > 10]  # Skip initial transient
        
        if len(close_indices) < 5:
            return 0.0
            
        # Track divergence after close approach
        divergence_rates = []
        for idx in close_indices[:20]:  # Use first 20 close approaches
            if idx + 50 < N:
                d0 = distances[idx]
                d_later = distances[idx + 50]
                if d0 > 1e-10:
                    rate = np.log(d_later / d0) / 50
                    divergence_rates.append(rate)
                    
        return np.median(divergence_rates) if divergence_rates else 0.0
    
    # ==========================================================================
    # Visualization
    # ==========================================================================
    
    def plot_phase_portrait(
        self,
        agent_id: int,
        components: Tuple[int, int] = (0, 1),
        ax: Optional[plt.Axes] = None,
        color_by_time: bool = True,
    ) -> plt.Axes:
        """
        Plot 2D phase portrait (μ_i vs π_μ_i).
        
        Args:
            agent_id: Which agent
            components: Which (position, momentum) components to plot
            ax: Matplotlib axes (created if None)
            color_by_time: Color trajectory by time
        """
        traj = self.get_trajectory(agent_id)
        if 'mu' not in traj:
            return ax
            
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))
            
        mu = traj['mu'][:, components[0]]
        pi = traj['pi_mu'][:, components[0]]
        
        if color_by_time:
            t = np.arange(len(mu))
            scatter = ax.scatter(mu, pi, c=t, cmap='viridis', s=1, alpha=0.7)
            plt.colorbar(scatter, ax=ax, label='Step')
        else:
            ax.plot(mu, pi, 'b-', alpha=0.5, linewidth=0.5)
            ax.plot(mu[0], pi[0], 'go', markersize=10, label='Start')
            ax.plot(mu[-1], pi[-1], 'ro', markersize=10, label='End')
            ax.legend()
            
        ax.set_xlabel(f'μ[{components[0]}]')
        ax.set_ylabel(f'π_μ[{components[0]}]')
        ax.set_title(f'Phase Portrait - Agent {agent_id}')
        ax.set_aspect('equal')
        ax.grid(alpha=0.3)
        
        return ax
    
    def plot_trajectory_components(
        self,
        agent_id: int,
        n_components: int = 3,
        save_path: Optional[Path] = None,
    ):
        """
        Plot μ and π_μ components over time.
        """
        traj = self.get_trajectory(agent_id)
        if 'mu' not in traj:
            return
            
        K = min(n_components, traj['mu'].shape[1])
        fig, axes = plt.subplots(K, 2, figsize=(14, 3*K))
        if K == 1:
            axes = axes.reshape(1, 2)
            
        t = traj['t']
        
        for i in range(K):
            # Position
            axes[i, 0].plot(t, traj['mu'][:, i], 'b-', linewidth=1)
            axes[i, 0].set_ylabel(f'μ[{i}]')
            axes[i, 0].grid(alpha=0.3)
            
            # Momentum
            axes[i, 1].plot(t, traj['pi_mu'][:, i], 'r-', linewidth=1)
            axes[i, 1].set_ylabel(f'π_μ[{i}]')
            axes[i, 1].grid(alpha=0.3)
            
        axes[0, 0].set_title('Position (μ)')
        axes[0, 1].set_title('Momentum (π_μ)')
        axes[-1, 0].set_xlabel('Time')
        axes[-1, 1].set_xlabel('Time')
        
        plt.suptitle(f'Phase Space Components - Agent {agent_id}')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
            plt.close()
        else:
            plt.show()
    
    def plot_sigma_evolution(
        self,
        agent_id: int,
        save_path: Optional[Path] = None,
    ):
        """
        Plot evolution of Σ eigenvalues (precision dynamics).
        """
        traj = self.get_trajectory(agent_id)
        if 'sigma_eig' not in traj:
            print("No sigma eigenvalue tracking enabled")
            return
            
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        t = traj['t']
        eigs = traj['sigma_eig']
        
        # Eigenvalue trajectories
        for i in range(eigs.shape[1]):
            axes[0].semilogy(t, eigs[:, i], label=f'λ_{i}')
        axes[0].set_xlabel('Time')
        axes[0].set_ylabel('Eigenvalue')
        axes[0].set_title('Σ Eigenvalues Over Time')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # Condition number
        cond = eigs[:, 0] / (eigs[:, -1] + 1e-10)
        axes[1].semilogy(t, cond, 'k-', linewidth=2)
        axes[1].set_xlabel('Time')
        axes[1].set_ylabel('Condition Number')
        axes[1].set_title('Σ Condition Number (λ_max/λ_min)')
        axes[1].grid(alpha=0.3)
        
        plt.suptitle(f'Precision Dynamics - Agent {agent_id}')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
            plt.close()
        else:
            plt.show()
    
    def plot_energy_exchange(self, save_path: Optional[Path] = None):
        """
        Plot kinetic/potential energy exchange (should oscillate for orbits).
        """
        if not self.snapshots:
            return
            
        t = np.array([s.t for s in self.snapshots])
        T = np.array([s.kinetic for s in self.snapshots])
        V = np.array([s.potential for s in self.snapshots])
        H = np.array([s.total for s in self.snapshots])
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # Energy components
        axes[0].plot(t, T, 'b-', label='Kinetic T', linewidth=1)
        axes[0].plot(t, V, 'r-', label='Potential V', linewidth=1)
        axes[0].plot(t, H, 'k--', label='Total H', linewidth=2)
        axes[0].set_xlabel('Time')
        axes[0].set_ylabel('Energy')
        axes[0].set_title('Energy Components')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # Energy drift
        H0 = H[0] if H[0] != 0 else 1.0
        drift = np.abs(H - H[0]) / np.abs(H0)
        axes[1].semilogy(t, drift + 1e-16, 'g-', linewidth=1)
        axes[1].axhline(0.01, color='orange', linestyle='--', label='1%')
        axes[1].axhline(0.001, color='red', linestyle='--', label='0.1%')
        axes[1].set_xlabel('Time')
        axes[1].set_ylabel('|H(t) - H(0)| / |H(0)|')
        axes[1].set_title('Energy Conservation')
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
            plt.close()
        else:
            plt.show()
    
    def generate_orbit_report(
        self,
        output_dir: Path,
        max_agents: int = 5,
    ):
        """
        Generate comprehensive orbit analysis report.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*60}")
        print("PHASE SPACE ORBIT ANALYSIS")
        print(f"{'='*60}")
        print(f"  Snapshots: {len(self.snapshots)}")
        print(f"  Agents: {len(self.agent_ids)}")
        
        # Global energy plot
        self.plot_energy_exchange(output_dir / "energy_exchange.png")
        print(f"  ✓ Saved energy_exchange.png")
        
        # Per-agent analysis
        for aid in self.agent_ids[:max_agents]:
            print(f"\n  Agent {aid}:")
            
            # Orbit diagnostics
            diag = self.detect_orbit(aid)
            
            orbit_type = "unknown"
            if diag.is_periodic:
                orbit_type = f"PERIODIC (T ≈ {diag.estimated_period:.1f})"
            elif diag.is_quasi_periodic:
                orbit_type = f"QUASI-PERIODIC (T ≈ {diag.estimated_period:.1f})"
            elif diag.is_chaotic:
                orbit_type = "CHAOTIC"
            
            print(f"    Orbit type: {orbit_type}")
            print(f"    Recurrences: {len(diag.recurrence_times)}")
            print(f"    Lyapunov exponent: {diag.lyapunov_exponent:.4f}")
            
            # Phase portrait
            fig, ax = plt.subplots(figsize=(8, 8))
            self.plot_phase_portrait(aid, ax=ax)
            fig.savefig(output_dir / f"phase_portrait_agent{aid}.png", dpi=150)
            plt.close()
            
            # Component trajectories
            self.plot_trajectory_components(
                aid, 
                save_path=output_dir / f"trajectory_agent{aid}.png"
            )
            
            # Sigma evolution
            if self.track_sigma_eigenvalues:
                self.plot_sigma_evolution(
                    aid,
                    save_path=output_dir / f"sigma_evolution_agent{aid}.png"
                )
                
        print(f"\n  ✓ Report saved to {output_dir}")
    
    def save(self, path: Path):
        """Save tracker state."""
        import pickle
        with open(path, 'wb') as f:
            pickle.dump({
                'snapshots': self.snapshots,
                'agent_ids': self.agent_ids,
                'track_interval': self.track_interval,
                'track_sigma_eigenvalues': self.track_sigma_eigenvalues,
                'track_gauge': self.track_gauge,
            }, f)
        print(f"✓ Saved phase space tracker to {path}")
    
    @classmethod
    def load(cls, path: Path) -> 'PhaseSpaceTracker':
        """Load tracker state."""
        import pickle
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        tracker = cls(
            track_interval=data['track_interval'],
            track_sigma_eigenvalues=data['track_sigma_eigenvalues'],
            track_gauge=data['track_gauge'],
        )
        tracker.snapshots = data['snapshots']
        tracker.agent_ids = data['agent_ids']
        tracker._initialized = True
        return tracker