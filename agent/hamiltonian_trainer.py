# -*- coding: utf-8 -*-
"""
Hamiltonian Trainer for Multi-Agent Belief Dynamics
===================================================

Integrates the Hamiltonian dynamics framework with the agent training system,
enabling underdamped dynamics as an alternative to gradient flow.

Comparison of Dynamics Regimes:
-------------------------------

**Overdamped (Gradient Flow - Current)**:
    θ̇ = -∇F(θ)
    - First-order dynamics
    - Dissipative (energy always decreases)
    - Fast convergence but no oscillations
    - Like particle moving through thick fluid

**Underdamped (Hamiltonian - New)**:
    dθ/dt = G^{-1} p
    dp/dt = -∇F - (curvature corrections)
    - Second-order dynamics
    - Nearly conservative (energy approximately conserved)
    - Orbital motion, can overshoot
    - Like particle moving in thin fluid

**Critical Damping (Optimal)**:
    Balance between overdamped and underdamped
    - Fastest convergence without oscillation
    - Requires tuning friction parameter

Author: Chris
"""

import numpy as np
from typing import List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from pathlib import Path
import pickle
import time

from gradients.free_energy_clean import compute_total_free_energy, FreeEnergyBreakdown
from config import TrainingConfig
from analysis.core.mu_tracking import create_mu_tracker, MuCenterTracking
from geometry.phase_space_tracker import PhaseSpaceTracker
from geometry.geodesic_corrections import compute_geodesic_force, diagnose_geodesic_correction
from geometry.multi_agent_mass_matrix import (
    build_full_mass_matrix,
    build_mu_mass_matrix,
    diagnose_mass_matrix,
    compute_velocity_full_coupling
)


@dataclass
class HamiltonianHistory:
    """
    Extended history for Hamiltonian dynamics.

    Includes both energy components AND phase space variables.
    """

    steps: List[int] = field(default_factory=list)

    # Energy components (same as TrainingHistory)
    total_energy: List[float] = field(default_factory=list)
    self_energy: List[float] = field(default_factory=list)
    belief_align: List[float] = field(default_factory=list)
    prior_align: List[float] = field(default_factory=list)
    observations: List[float] = field(default_factory=list)

    # Hamiltonian energy components (NEW)
    kinetic_energy: List[float] = field(default_factory=list)
    potential_energy: List[float] = field(default_factory=list)
    total_hamiltonian: List[float] = field(default_factory=list)

    # Phase space tracking (NEW)
    momentum_norms: List[float] = field(default_factory=list)  # ||p||
    velocity_norms: List[float] = field(default_factory=list)  # ||θ̇||

    # Energy conservation diagnostic (NEW)
    energy_drift: List[float] = field(default_factory=list)  # |H(t) - H(0)|/H(0)

    # Mu center tracking
    mu_tracker: Optional[MuCenterTracking] = None

    # Agent snapshots for metric evolution analysis
    agent_snapshots: List[dict] = field(default_factory=list)
    snapshot_steps: List[int] = field(default_factory=list)

    def record(self, step: int, energies: FreeEnergyBreakdown,
               hamiltonian_energies: Optional[Tuple[float, float, float]] = None,
               momentum_norm: float = 0.0,
               velocity_norm: float = 0.0,
               system = None):
        """
        Record metrics for current step.

        Args:
            step: Current step number
            energies: Free energy breakdown
            hamiltonian_energies: (T, V, H) kinetic, potential, total
            momentum_norm: ||p||
            velocity_norm: ||θ̇||
            system: Agent system for mu tracking
        """
        self.steps.append(step)

        # Standard energy components
        self.total_energy.append(energies.total)
        self.self_energy.append(energies.self_energy)
        self.belief_align.append(energies.belief_align)
        self.prior_align.append(energies.prior_align)
        self.observations.append(energies.observations)

        # Hamiltonian energies
        if hamiltonian_energies is not None:
            T, V, H = hamiltonian_energies
            self.kinetic_energy.append(T)
            self.potential_energy.append(V)
            self.total_hamiltonian.append(H)

            # Energy drift
            if len(self.total_hamiltonian) == 1:
                # First step - no drift yet
                self.energy_drift.append(0.0)
            else:
                H0 = self.total_hamiltonian[0]
                drift = abs(H - H0) / (abs(H0) + 1e-10)
                self.energy_drift.append(drift)

        # Phase space norms
        self.momentum_norms.append(momentum_norm)
        self.velocity_norms.append(velocity_norm)

        # Mu tracking
        if self.mu_tracker is None and system is not None:
            self.mu_tracker = create_mu_tracker(system)

        if self.mu_tracker is not None and system is not None:
            self.mu_tracker.record(step, system)

    def save_snapshot(self, step: int, system):
        """
        Save agent state snapshot for later metric analysis.

        Stores copies of all agent fields needed to compute pullback metrics.
        GAUGE-COVARIANT: Stores Σ directly (not Cholesky factors).

        Args:
            step: Current training step
            system: MultiAgentSystem to snapshot
        """
        snapshot = {
            'step': step,
            'agents': []
        }

        for agent in system.agents:
            agent_data = {
                'agent_id': agent.agent_id,
                'mu_q': agent.mu_q.copy(),
                'Sigma_q': agent.Sigma_q.copy(),  # Store Σ directly (gauge-covariant)
                'mu_p': agent.mu_p.copy(),
                'Sigma_p': agent.Sigma_p.copy(),  # Store Σ directly (gauge-covariant)
            }

            # Add gauge field if present
            if hasattr(agent, 'gauge') and hasattr(agent.gauge, 'phi'):
                agent_data['phi'] = agent.gauge.phi.copy()

            snapshot['agents'].append(agent_data)

        self.agent_snapshots.append(snapshot)
        self.snapshot_steps.append(step)


class HamiltonianTrainer:
    """
    Hamiltonian dynamics trainer for belief evolution.

    Implements second-order dynamics:
        dθ/dt = ∂H/∂p = G^{-1} p
        dp/dt = -∂H/∂θ = -∇V - (curvature)

    where:
        - θ: Belief parameters (flattened μ_q, L_q for all agents)
        - p: Conjugate momenta
        - G: Fisher metric (mass tensor)
        - V: Total free energy

    Comparison to GradientFlowTrainer:
        - Gradient flow: θ̇ = -∇V (first-order, dissipative)
        - Hamiltonian: Uses momentum, approximately conserves energy
    """

    def __init__(self, system, config: Optional[TrainingConfig] = None,
                 friction: float = 0.0,
                 mass_scale: float = 1.0,
                 track_phase_space: bool = True,
                 phase_space_output_dir: Optional[Path] = None,
                 enable_geodesic_correction: bool = True,
                 geodesic_eps: float = 1e-5):
        """
        Initialize Hamiltonian trainer.

        Args:
            system: MultiAgentSystem instance
            config: Training configuration
            friction: Friction coefficient γ (0 = conservative, >0 = damped)
                     0.0: Pure Hamiltonian (underdamped)
                     0.1: Light damping
                     1.0: Critical damping
                     10.0: Heavy damping (approaches overdamped)
            mass_scale: Overall mass scale (affects oscillation frequency)
            track_phase_space: Whether to enable PhaseSpaceTracker for orbit visualization
            phase_space_output_dir: Directory for phase space plots (default: ./phase_space_output)
            enable_geodesic_correction: Whether to include geodesic forces from dM/dθ
                                        True: Full Hamiltonian with curvature (geodesics)
                                        False: Ignore curvature (faster but less accurate)
            geodesic_eps: Finite difference step for geodesic force computation
        """
        self.system = system
        self.config = config or TrainingConfig()
        self.friction = friction
        self.mass_scale = mass_scale

        # Geodesic correction settings
        self.enable_geodesic_correction = enable_geodesic_correction
        self.geodesic_eps = geodesic_eps

        # Training state
        self.history = HamiltonianHistory()
        self.current_step = 0
        self.continuous_time = 0.0  # Track continuous time for phase space

        # Mu tracking
        self.history.mu_tracker = create_mu_tracker(system)

        # Phase space tracking for orbit visualization
        self.track_phase_space = track_phase_space
        self.phase_space_output_dir = phase_space_output_dir or Path("./phase_space_output")
        if track_phase_space:
            self.phase_space_tracker = PhaseSpaceTracker(
                track_interval=1,
                max_snapshots=10000,
                track_sigma_eigenvalues=True,
                track_gauge=False,
            )
        else:
            self.phase_space_tracker = None

        # Initialize phase space coordinates
        self._initialize_phase_space()

        # Track performance
        self._step_times = []

        print(f"Hamiltonian trainer initialized")
        print(f"  Friction: gamma = {friction}")
        print(f"  Mass scale: {mass_scale}")
        print(f"  Regime: {'Conservative' if friction < 0.01 else 'Damped'}")
        print(f"  Geodesic correction: {'ENABLED' if enable_geodesic_correction else 'DISABLED'}")
        if track_phase_space:
            print(f"  Phase space tracking: ENABLED")

    def _initialize_phase_space(self):
        """
        Initialize phase space coordinates (θ, p).

        θ: Flattened belief parameters
        p: Initial momenta (start from rest: p = 0)
        """
        # Flatten all agent parameters into θ
        self.theta = self._pack_parameters()

        # Initialize momenta to zero (start from rest)
        self.p = np.zeros_like(self.theta)

        print(f"  Phase space dimension: {len(self.theta)}")

    def _pack_parameters(self) -> np.ndarray:
        """
        Pack all agent parameters into flat vector θ ON SPD MANIFOLD.

        CRITICAL: For hyperbolic geometry, we parameterize by Σ_q (not L_q)!

        For each agent, includes:
            - mu_q (flattened)
            - Sigma_q (flattened, upper triangular part for symmetry)

        Returns:
            theta: Flattened parameter vector in SPD(n) × ℝ^K
        """
        params = []

        for agent in self.system.agents:
            # Flatten mu_q
            mu_flat = agent.mu_q.flatten()
            params.append(mu_flat)

            # Flatten Sigma_q (symmetric, so store upper triangle)
            K = agent.config.K
            Sigma_flat = agent.Sigma_q.reshape(-1, K, K)

            # For each spatial point, extract upper triangle (includes diagonal)
            for Sigma_mat in Sigma_flat:
                upper_indices = np.triu_indices(K)
                Sigma_upper = Sigma_mat[upper_indices]
                params.append(Sigma_upper)

        return np.concatenate(params)

    def _unpack_parameters(self, theta: np.ndarray):
        """
        Unpack flat vector θ back into agent parameters ON SPD MANIFOLD.

        CRITICAL: Updates Σ_q directly (gauge-covariant), invalidates L cache.

        Args:
            theta: Flattened parameter vector in SPD(n) × ℝ^K
        """
        idx = 0

        for agent in self.system.agents:
            K = agent.config.K
            n_spatial = agent.mu_q.size // K

            # Unpack mu_q
            mu_size = n_spatial * K
            mu_flat = theta[idx:idx + mu_size]
            agent.mu_q = mu_flat.reshape(agent.mu_q.shape)
            idx += mu_size

            # Unpack Sigma_q (symmetric upper triangle)
            Sigma_size = K * (K + 1) // 2  # Upper triangular elements per spatial point

            for i in range(n_spatial):
                Sigma_upper = theta[idx:idx + Sigma_size]
                idx += Sigma_size

                # Reconstruct symmetric matrix from upper triangular
                Sigma_mat = np.zeros((K, K))
                upper_indices = np.triu_indices(K)
                Sigma_mat[upper_indices] = Sigma_upper
                # Symmetrize (copy upper to lower)
                Sigma_mat = Sigma_mat + Sigma_mat.T - np.diag(np.diag(Sigma_mat))

                # Update agent's Sigma_q directly (GAUGE-COVARIANT!)
                if agent.Sigma_q.ndim == 2:
                    # 0D agent (particle)
                    agent.Sigma_q = Sigma_mat.astype(np.float32)
                elif agent.Sigma_q.ndim == 3:
                    # 1D grid
                    agent.Sigma_q[i] = Sigma_mat.astype(np.float32)
                else:
                    # 2D grid - need to map linear index to (x, y)
                    shape = agent.Sigma_q.shape[:2]
                    x = i // shape[1]
                    y = i % shape[1]
                    agent.Sigma_q[x, y] = Sigma_mat.astype(np.float32)

            # Invalidate Cholesky cache (L computed on-demand)
            if hasattr(agent, '_L_q_cache'):
                agent._L_q_cache = None
            if hasattr(agent, '_L_p_cache'):
                agent._L_p_cache = None

    def _compute_metric(self, theta: np.ndarray) -> np.ndarray:
        """
        Compute Fisher metric G(θ).

        For now, uses block diagonal approximation:
            G = diag(Σ_1^{-1}, Σ_2^{-1}, ..., Σ_N^{-1})

        Full implementation would include cross-agent coupling.

        Args:
            theta: Parameter vector

        Returns:
            G: Fisher metric matrix (d, d)
        """
        # Simplified: Block diagonal with each block = Σ^{-1}
        d = len(theta)
        G = np.zeros((d, d))

        idx = 0
        for agent in self.system.agents:
            K = agent.config.K
            n_spatial = agent.mu_q.size // K

            # Mu block: G_μ = Σ^{-1}
            for i in range(n_spatial):
                if agent.Sigma_q.ndim == 2:
                    Sigma = agent.Sigma_q
                elif agent.Sigma_q.ndim == 3:
                    Sigma = agent.Sigma_q[i]
                else:
                    shape = agent.Sigma_q.shape[:2]
                    x = i // shape[1]
                    y = i % shape[1]
                    Sigma = agent.Sigma_q[x, y]

                try:
                    Sigma_inv = np.linalg.inv(Sigma + 1e-6 * np.eye(K))
                except:
                    Sigma_inv = np.eye(K)

                # Fill diagonal block
                mu_block_size = K
                G[idx:idx+mu_block_size, idx:idx+mu_block_size] = Sigma_inv * self.mass_scale
                idx += mu_block_size

                # L block: Identity (simplified)
                L_block_size = K * (K + 1) // 2
                G[idx:idx+L_block_size, idx:idx+L_block_size] = np.eye(L_block_size) * self.mass_scale
                idx += L_block_size

        return G

    def _compute_potential(self, theta: np.ndarray) -> float:
        """
        Compute potential energy V(θ) = total free energy.

        Args:
            theta: Parameter vector

        Returns:
            V: Potential energy (free energy)
        """
        # Unpack theta into agent parameters
        theta_backup = self._pack_parameters()
        self._unpack_parameters(theta)

        # Compute free energy
        energies = compute_total_free_energy(self.system)
        V = energies.total

        # Restore original parameters
        self._unpack_parameters(theta_backup)

        return V

    def _compute_complete_mass_matrix(self, agent, agent_idx: int) -> np.ndarray:
        """
        Compute COMPLETE Fisher-Rao mass matrix with coupling terms.

        M_i = Σ_p^{-1} + Σ_j β_ij Ω_ij Σ_q_j^{-1} Ω_ij^T

        Components:
        - Σ_p^{-1}: Bare mass from prior precision (constant, quasi-static)
        - Σ_j β_ij Ω_ij Σ_q_j^{-1} Ω_ij^T: Relational mass from consensus coupling

        Args:
            agent: Agent whose mass matrix to compute
            agent_idx: Agent index in system

        Returns:
            M: Mass matrix per spatial point, shape (*S, K, K)
        """
        from gradients.softmax_grads import compute_softmax_weights

        K = agent.config.K
        spatial_shape = agent.mu_q.shape[:-1] if agent.mu_q.ndim > 1 else ()

        # Initialize with bare mass: Σ_p^{-1}
        M = np.zeros(agent.Sigma_p.shape, dtype=np.float64)

        # Bare mass (constant for quasi-static priors)
        if agent.Sigma_p.ndim == 2:
            # 0D: single Gaussian
            M = np.linalg.inv(agent.Sigma_p + 1e-8 * np.eye(K))
        elif agent.Sigma_p.ndim == 3:
            # 1D field
            for i in range(agent.Sigma_p.shape[0]):
                M[i] = np.linalg.inv(agent.Sigma_p[i] + 1e-8 * np.eye(K))
        else:
            # 2D field
            for i in range(agent.Sigma_p.shape[0]):
                for j in range(agent.Sigma_p.shape[1]):
                    M[i, j] = np.linalg.inv(agent.Sigma_p[i, j] + 1e-8 * np.eye(K))

        # Add relational mass from belief consensus coupling
        kappa_beta = getattr(self.system.config, 'kappa_beta', 1.0)
        beta_fields = compute_softmax_weights(self.system, agent_idx, 'belief', kappa_beta)

        for j_idx, beta_ij in beta_fields.items():
            agent_j = self.system.agents[j_idx]

            # Compute transport Ω_ij
            Omega_ij = self.system.compute_transport_ij(agent_idx, j_idx)

            # Add coupling: β_ij Ω_ij Σ_q_j^{-1} Ω_ij^T
            if agent.mu_q.ndim == 1:
                # 0D: scalar β, single matrix
                Sigma_q_j_inv = np.linalg.inv(agent_j.Sigma_q + 1e-8 * np.eye(K))
                M += float(beta_ij) * (Omega_ij @ Sigma_q_j_inv @ Omega_ij.T)

            elif agent.mu_q.ndim == 2:
                # 1D field: β_ij(c), Ω_ij(c), Σ_q_j(c) all spatial
                for i in range(agent.mu_q.shape[0]):
                    Sigma_q_j_inv = np.linalg.inv(agent_j.Sigma_q[i] + 1e-8 * np.eye(K))
                    Omega_c = Omega_ij[i] if Omega_ij.ndim == 3 else Omega_ij
                    M[i] += beta_ij[i] * (Omega_c @ Sigma_q_j_inv @ Omega_c.T)

            else:
                # 2D field
                for i in range(agent.mu_q.shape[0]):
                    for j in range(agent.mu_q.shape[1]):
                        Sigma_q_j_inv = np.linalg.inv(agent_j.Sigma_q[i, j] + 1e-8 * np.eye(K))
                        Omega_c = Omega_ij[i, j] if Omega_ij.ndim == 4 else Omega_ij
                        M[i, j] += beta_ij[i, j] * (Omega_c @ Sigma_q_j_inv @ Omega_c.T)

        return M.astype(np.float32)

    def _compute_velocity_hyperbolic(self, theta: np.ndarray, p: np.ndarray) -> np.ndarray:
        """
        Compute velocity dθ/dt with COMPLETE Hamilton's equations on product manifold.

        For μ part (COMPLETE Fisher-Rao): dμ/dt = M^{-1} π_μ
        For Σ part (Hyperbolic SPD): dΣ/dt = Σ Π_Σ Σ

        where M = Σ_p^{-1} + Σ_j β_ij Ω_ij Σ_q_j^{-1} Ω_ij^T includes:
        - Bare mass: Σ_p^{-1} (prior precision)
        - Relational mass: Σ_j β_ij Ω_ij Σ_q_j^{-1} Ω_ij^T (consensus coupling)

        The Σ equation is the geodesic flow on SPD(n) with affine-invariant metric,
        giving the manifold constant negative curvature κ = -1/4 (HYPERBOLIC!).

        Args:
            theta: [μ, Σ] parameters (flattened)
            p: [π_μ, Π_Σ] momenta (flattened)

        Returns:
            dtheta_dt: Velocity vector with COMPLETE Fisher-Rao + hyperbolic flow
        """
        dtheta_dt = np.zeros_like(theta)
        idx_theta = 0
        idx_p = 0

        for agent_idx, agent in enumerate(self.system.agents):
            K = agent.config.K
            n_spatial = agent.mu_q.size // K
            mu_size = n_spatial * K
            Sigma_size_per_point = K * (K + 1) // 2

            # --- μ part (COMPLETE Fisher-Rao metric with coupling) ---
            mu_flat = theta[idx_theta:idx_theta + mu_size]
            pi_mu = p[idx_p:idx_p + mu_size].reshape(agent.mu_q.shape)

            # Compute COMPLETE mass matrix M = Σ_p^{-1} + Σ_j β_ij Ω_ij Σ_q_j^{-1} Ω_ij^T
            M = self._compute_complete_mass_matrix(agent, agent_idx)

            # dμ/dt = M^{-1} π_μ (invert mass to get velocity)
            dmu_dt = np.zeros_like(pi_mu)

            if agent.mu_q.ndim == 1:
                # 0D particle: single Gaussian
                M_inv = np.linalg.inv(M + 1e-8 * np.eye(K))
                dmu_dt = M_inv @ pi_mu / self.mass_scale

            elif agent.mu_q.ndim == 2:
                # 1D field: invert M at each spatial point
                for i in range(agent.mu_q.shape[0]):
                    M_inv = np.linalg.inv(M[i] + 1e-8 * np.eye(K))
                    dmu_dt[i] = M_inv @ pi_mu[i] / self.mass_scale

            else:
                # 2D field: invert M at each spatial point
                for i in range(agent.mu_q.shape[0]):
                    for j in range(agent.mu_q.shape[1]):
                        M_inv = np.linalg.inv(M[i, j] + 1e-8 * np.eye(K))
                        dmu_dt[i, j] = M_inv @ pi_mu[i, j] / self.mass_scale

            dtheta_dt[idx_theta:idx_theta + mu_size] = dmu_dt.flatten()

            idx_theta += mu_size
            idx_p += mu_size

            # --- Σ part (HYPERBOLIC geodesic flow!) ---
            for i in range(n_spatial):
                # Extract Sigma and Pi_Sigma for this spatial point
                Sigma_upper = theta[idx_theta:idx_theta + Sigma_size_per_point]
                Pi_upper = p[idx_p:idx_p + Sigma_size_per_point]

                # Reconstruct symmetric matrices
                Sigma_mat = np.zeros((K, K))
                Pi_mat = np.zeros((K, K))
                upper_indices = np.triu_indices(K)

                Sigma_mat[upper_indices] = Sigma_upper
                Sigma_mat = Sigma_mat + Sigma_mat.T - np.diag(np.diag(Sigma_mat))

                Pi_mat[upper_indices] = Pi_upper
                Pi_mat = Pi_mat + Pi_mat.T - np.diag(np.diag(Pi_mat))

                # CRITICAL: Geodesic flow on hyperbolic SPD manifold
                # dΣ/dt = Σ Π Σ  (affine-invariant metric)
                dSigma_dt = Sigma_mat @ Pi_mat @ Sigma_mat / self.mass_scale

                # Pack back to upper triangle
                dSigma_upper = dSigma_dt[upper_indices]
                dtheta_dt[idx_theta:idx_theta + Sigma_size_per_point] = dSigma_upper

                idx_theta += Sigma_size_per_point
                idx_p += Sigma_size_per_point

        return dtheta_dt

    def _compute_force(self, theta: np.ndarray) -> np.ndarray:
        """
        Compute force -∇V(θ) using validated analytical gradients.

        CRITICAL: Uses gradient_engine for exact, symplectic-preserving gradients.
        Finite differences break energy conservation!

        Args:
            theta: Parameter vector

        Returns:
            force: -∇V (d,) in flattened Euclidean coordinates
        """
        from gradients.gradient_engine import compute_natural_gradients

        # Unpack theta into agent parameters
        theta_backup = self._pack_parameters()
        self._unpack_parameters(theta)

        # Compute analytical gradients (returns both Euclidean and natural)
        agent_grads = compute_natural_gradients(self.system, n_jobs=1)

        # Pack Euclidean gradients into flat vector
        grad_V = np.zeros_like(theta)
        idx = 0

        for agent, grads in zip(self.system.agents, agent_grads):
            K = agent.config.K
            n_spatial = agent.mu_q.size // K

            # --- μ gradients (Euclidean) ---
            grad_mu_flat = grads.grad_mu_q.flatten()
            mu_size = n_spatial * K
            grad_V[idx:idx + mu_size] = grad_mu_flat
            idx += mu_size

            # --- Σ gradients (Euclidean, upper triangle) ---
            Sigma_size_per_point = K * (K + 1) // 2
            grad_Sigma_reshaped = grads.grad_Sigma_q.reshape(-1, K, K)

            for i in range(n_spatial):
                Sigma_grad_mat = grad_Sigma_reshaped[i]
                # Extract upper triangle (symmetrize: (∂S/∂Σ)_ij + (∂S/∂Σ)_ji for i≠j)
                upper_indices = np.triu_indices(K)
                grad_upper = Sigma_grad_mat[upper_indices]

                # Symmetrize off-diagonal terms (gradient wrt symmetric matrix)
                for k, (i_idx, j_idx) in enumerate(zip(*upper_indices)):
                    if i_idx != j_idx:
                        grad_upper[k] += Sigma_grad_mat[j_idx, i_idx]

                grad_V[idx:idx + Sigma_size_per_point] = grad_upper
                idx += Sigma_size_per_point

        # Restore original parameters
        self._unpack_parameters(theta_backup)

        return -grad_V

    def _hamiltonian_equations(self, theta: np.ndarray, p: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute COMPLETE Hamilton's equations on product manifold R^K x SPD(n).

        CRITICAL: For Sigma parameters, uses geodesic flow on SPD(n):
            dSigma/dt = Sigma Pi Sigma  (affine-invariant metric, kappa = -1/4)
            dPi/dt = -dV/dSigma

        For mu parameters (COMPLETE Fisher-Rao with coupling):
            dmu/dt = M^{-1} pi_mu  where M = Sigma_p^{-1} + sum_j beta_ij Omega_ij Sigma_qj^{-1} Omega_ij^T
            dpi/dt = -dV/dmu - (1/2) pi^T (dM^{-1}/dmu) pi  [GEODESIC CORRECTION!]

        The geodesic correction term ensures trajectories follow geodesics on the
        statistical manifold defined by the position-dependent Fisher-Rao metric.

        Mass includes bare (Sigma_p^{-1}) + relational (sum_j beta_ij...) components.
        This provides natural damping through consensus coupling!

        If friction > 0, adds damping: dp/dt -= gamma*p

        Args:
            theta: Position [mu, Sigma] (flattened)
            p: Momentum [pi_mu, Pi_Sigma] (flattened)

        Returns:
            dtheta_dt: Velocity with COMPLETE Fisher-Rao (mu) + hyperbolic geodesic (Sigma)
            dp_dt: Force -dV + geodesic correction + optional friction
        """
        # Compute velocity using hyperbolic geometry for Sigma part
        dtheta_dt = self._compute_velocity_hyperbolic(theta, p)

        # dp/dt = -dV/dtheta (potential force)
        force = self._compute_force(theta)
        dp_dt = force

        # Add geodesic correction: -(1/2) p^T (dM^{-1}/dtheta) p
        # This term arises from the position-dependent mass matrix M(theta)
        # and ensures trajectories follow geodesics on the statistical manifold
        if self.enable_geodesic_correction:
            geodesic_force = compute_geodesic_force(
                self, theta, p,
                eps=self.geodesic_eps,
                include_beta_variation=True
            )
            dp_dt += geodesic_force

        # Add friction if specified
        if self.friction > 0:
            dp_dt -= self.friction * p

        return dtheta_dt, dp_dt

    def step(self, dt: float = 0.01) -> FreeEnergyBreakdown:
        """
        Perform one Hamiltonian evolution step.

        Uses leapfrog integrator (symplectic) for energy conservation.

        Args:
            dt: Time step size

        Returns:
            energies: Free energy breakdown after update
        """
        step_start = time.perf_counter()

        # Leapfrog integration (symplectic)
        # Reference: Neal (2011) "MCMC using Hamiltonian dynamics"

        # Half step for momentum
        _, dp_dt = self._hamiltonian_equations(self.theta, self.p)
        self.p = self.p + 0.5 * dt * dp_dt

        # Full step for position
        dtheta_dt, _ = self._hamiltonian_equations(self.theta, self.p)
        self.theta = self.theta + dt * dtheta_dt

        # Update agent parameters
        self._unpack_parameters(self.theta)

        # Half step for momentum
        _, dp_dt = self._hamiltonian_equations(self.theta, self.p)
        self.p = self.p + 0.5 * dt * dp_dt

        # Compute energies
        energies = compute_total_free_energy(self.system)

        # Compute Hamiltonian energies
        G = self._compute_metric(self.theta)
        try:
            G_inv = np.linalg.inv(G + 1e-6 * np.eye(len(G)))
        except:
            G_inv = np.eye(len(G)) / self.mass_scale

        T = 0.5 * self.p @ G_inv @ self.p  # Kinetic
        V = energies.total  # Potential
        H = T + V  # Total Hamiltonian

        # Record history
        if self.config.save_history:
            self.history.record(
                step=self.current_step,
                energies=energies,
                hamiltonian_energies=(T, V, H),
                momentum_norm=np.linalg.norm(self.p),
                velocity_norm=np.linalg.norm(dtheta_dt),
                system=self.system
            )

        # Record phase space for orbit visualization
        if self.phase_space_tracker is not None:
            momenta = self._build_momenta_dict()
            self.phase_space_tracker.record(
                step=self.current_step,
                t=self.continuous_time,
                agents=self.system.agents,
                momenta=momenta,
                energies=(T, V, H),
            )

        self.current_step += 1
        self.continuous_time += dt
        self._step_times.append(time.perf_counter() - step_start)

        return energies

    def _build_momenta_dict(self) -> dict:
        """
        Build momenta dictionary from flat momentum vector for PhaseSpaceTracker.

        Returns:
            Dict mapping agent_id -> {'mu': π_μ array}
        """
        momenta = {}
        idx = 0

        for agent in self.system.agents:
            K = agent.config.K
            n_spatial = agent.mu_q.size // K
            mu_size = n_spatial * K
            Sigma_size_per_point = K * (K + 1) // 2

            # Extract mu momentum
            pi_mu = self.p[idx:idx + mu_size].reshape(agent.mu_q.shape)
            momenta[agent.agent_id] = {'mu': pi_mu}

            idx += mu_size
            # Skip Sigma momenta (not currently tracked in PhaseSpaceTracker)
            idx += n_spatial * Sigma_size_per_point

        return momenta

    def train(self, n_steps: Optional[int] = None, dt: float = 0.01) -> HamiltonianHistory:
        """
        Run full Hamiltonian training loop.

        Args:
            n_steps: Number of steps (overrides config if provided)
            dt: Time step size

        Returns:
            history: Training history with Hamiltonian diagnostics
        """
        n_steps = n_steps or self.config.n_steps

        print("="*70)
        print("HAMILTONIAN BELIEF DYNAMICS")
        print("="*70)
        print(f"System: {self.system.n_agents} agents")
        print(f"Steps: {n_steps}")
        print(f"Time step: dt = {dt}")
        print(f"Friction: γ = {self.friction}")
        print(f"Regime: {'Conservative (Underdamped)' if self.friction < 0.01 else 'Damped'}")
        print("="*70)

        # Initial energy
        initial_energies = compute_total_free_energy(self.system)
        G = self._compute_metric(self.theta)
        G_inv = np.linalg.inv(G + 1e-6 * np.eye(len(G)))
        T0 = 0.5 * self.p @ G_inv @ self.p
        V0 = initial_energies.total
        H0 = T0 + V0

        print(f"\nInitial state:")
        print(f"  Free energy (V): {V0:.6f}")
        print(f"  Kinetic energy (T): {T0:.6f}")
        print(f"  Total Hamiltonian (H): {H0:.6f}")
        print()

        # Training loop
        try:
            for step in range(n_steps):
                energies = self.step(dt)

                # Logging
                if step % self.config.log_every == 0:
                    self._log_step(step, energies)

                # Snapshot saving for metric analysis
                if (self.config.save_snapshots and
                    step % self.config.snapshot_every == 0):
                    self.history.save_snapshot(step, self.system)

        except KeyboardInterrupt:
            print("\n⚠ Training interrupted by user")
        except Exception as e:
            print(f"\n⚠ Training stopped due to error: {e}")
            print("  Generating phase space report with data collected so far...")
            # Generate report even on crash
            if self.phase_space_tracker is not None and len(self.phase_space_tracker.snapshots) > 0:
                self.phase_space_tracker.generate_orbit_report(
                    output_dir=self.phase_space_output_dir,
                    max_agents=min(4, len(self.system.agents))
                )
            raise  # Re-raise the exception after saving plots

        # Final summary
        final_energies = compute_total_free_energy(self.system)
        G_final = self._compute_metric(self.theta)
        G_inv_final = np.linalg.inv(G_final + 1e-6 * np.eye(len(G_final)))
        T_final = 0.5 * self.p @ G_inv_final @ self.p
        V_final = final_energies.total
        H_final = T_final + V_final

        energy_conservation = abs(H_final - H0) / (abs(H0) + 1e-10)
        avg_step_time = np.mean(self._step_times[-100:])

        print("\n" + "="*70)
        print("HAMILTONIAN TRAINING COMPLETE")
        print("="*70)
        print(f"Final state:")
        print(f"  Free energy (V): {V_final:.6f} (Δ = {V_final - V0:+.6f})")
        print(f"  Kinetic energy (T): {T_final:.6f}")
        print(f"  Total Hamiltonian (H): {H_final:.6f}")
        print()
        print(f"Energy conservation:")
        print(f"  |H(T) - H(0)|/H(0) = {energy_conservation:.2e}")
        print(f"  Status: {'✓ GOOD' if energy_conservation < 0.01 else '⚠ DRIFT'}")
        print()
        print("Performance:")
        print(f"  Avg step time: {avg_step_time:.4f}s")
        print(f"  Steps/second: {1.0/avg_step_time:.2f}")
        print("="*70)

        # Generate phase space orbit report
        if self.phase_space_tracker is not None and len(self.phase_space_tracker.snapshots) > 0:
            print("\nGenerating phase space orbit analysis...")
            self.phase_space_tracker.generate_orbit_report(
                output_dir=self.phase_space_output_dir,
                max_agents=min(4, len(self.system.agents))
            )

        return self.history

    def _log_step(self, step: int, energies: FreeEnergyBreakdown):
        """Print progress for current step."""
        if len(self.history.total_hamiltonian) > 0:
            H = self.history.total_hamiltonian[-1]
            T = self.history.kinetic_energy[-1]
            V = self.history.potential_energy[-1]
            drift = self.history.energy_drift[-1]

            msg = f"Step {step:5d}: H = {H:8.4f} [T={T:.3f}, V={V:.3f}]"
            msg += f"  drift={drift:.2e}"

            # Add timing if available
            if self._step_times:
                recent_time = np.mean(self._step_times[-10:])
                msg += f"  ({recent_time:.3f}s)"

            print(msg)

    def generate_phase_space_report(self, output_dir: Optional[Path] = None, max_agents: int = 4):
        """
        Manually generate phase space orbit analysis report.

        Call this if you want to generate the report at a specific point
        during training or after loading a saved tracker.

        Args:
            output_dir: Directory for output plots (defaults to phase_space_output_dir)
            max_agents: Maximum number of agents to analyze
        """
        if self.phase_space_tracker is None:
            print("Phase space tracking not enabled")
            return

        if len(self.phase_space_tracker.snapshots) == 0:
            print("No phase space data recorded yet")
            return

        output_dir = output_dir or self.phase_space_output_dir
        self.phase_space_tracker.generate_orbit_report(
            output_dir=output_dir,
            max_agents=max_agents
        )

    def save_phase_space_tracker(self, path: Optional[Path] = None):
        """
        Save phase space tracker state for later analysis.

        Args:
            path: File path (defaults to phase_space_output_dir/tracker.pkl)
        """
        if self.phase_space_tracker is None:
            print("Phase space tracking not enabled")
            return

        if path is None:
            self.phase_space_output_dir.mkdir(parents=True, exist_ok=True)
            path = self.phase_space_output_dir / "tracker.pkl"

        self.phase_space_tracker.save(path)

    def diagnose_geodesic_correction(self) -> dict:
        """
        Diagnose the geodesic correction at current state.

        Returns detailed analysis of the geodesic correction term relative
        to the potential force term. This is useful for understanding whether
        the geodesic correction is significant for your system.

        Returns:
            Dictionary with diagnostic information:
            - geodesic_force_norm: ||F_geo||
            - potential_force_norm: ||-dV/dtheta||
            - ratio: How significant is geodesic vs potential force
            - agent_contributions: Per-agent breakdown
        """
        return diagnose_geodesic_correction(
            self, self.theta, self.p, eps=self.geodesic_eps
        )

    def diagnose_mass_matrix(self) -> dict:
        """
        Diagnose the mass matrix structure at current state.

        Returns information about conditioning, eigenvalues, and structure
        of the mass matrix. Useful for understanding numerical stability
        and the relative importance of coupling terms.

        Returns:
            Dictionary with:
            - total_dim: Total dimension of mu coordinates
            - n_agents: Number of agents
            - per_agent_dims: Dimension per agent
            - condition_numbers: Per-agent and global condition numbers
            - diagonal_dominance: How block-diagonal is the matrix
            - min_eigenvalue: Smallest eigenvalue (stability)
        """
        diag = diagnose_mass_matrix(self, self.theta)
        return {
            'total_dim': diag.total_dim,
            'n_agents': diag.n_agents,
            'per_agent_dims': diag.per_agent_dims,
            'condition_numbers': diag.condition_numbers,
            'global_condition_number': diag.global_condition_number,
            'diagonal_dominance': diag.diagonal_dominance,
            'min_eigenvalue': diag.min_eigenvalue
        }

    def compare_energy_conservation(self, n_steps: int = 100, dt: float = 0.01) -> dict:
        """
        Compare energy conservation with and without geodesic correction.

        Runs two short simulations from the same initial state:
        1. With geodesic correction enabled
        2. Without geodesic correction

        Returns comparison statistics for energy drift.

        Args:
            n_steps: Number of steps to run
            dt: Time step

        Returns:
            Dictionary with comparison results:
            - with_geo_drift: Energy drift with geodesic correction
            - without_geo_drift: Energy drift without geodesic correction
            - improvement_factor: How much better is with vs without
        """
        # Save current state
        theta_backup = self.theta.copy()
        p_backup = self.p.copy()
        geo_setting_backup = self.enable_geodesic_correction

        # Initial energy
        energies = compute_total_free_energy(self.system)
        G = self._compute_metric(self.theta)
        G_inv = np.linalg.inv(G + 1e-6 * np.eye(len(G)))
        T0 = 0.5 * self.p @ G_inv @ self.p
        V0 = energies.total
        H0 = T0 + V0

        results = {}

        # Run WITH geodesic correction
        self.enable_geodesic_correction = True
        self.theta = theta_backup.copy()
        self.p = p_backup.copy()
        self._unpack_parameters(self.theta)

        H_with = [H0]
        for _ in range(n_steps):
            dtheta_dt, dp_dt = self._hamiltonian_equations(self.theta, self.p)
            self.p = self.p + 0.5 * dt * dp_dt
            dtheta_dt, _ = self._hamiltonian_equations(self.theta, self.p)
            self.theta = self.theta + dt * dtheta_dt
            self._unpack_parameters(self.theta)
            _, dp_dt = self._hamiltonian_equations(self.theta, self.p)
            self.p = self.p + 0.5 * dt * dp_dt

            # Compute H
            energies = compute_total_free_energy(self.system)
            G = self._compute_metric(self.theta)
            G_inv = np.linalg.inv(G + 1e-6 * np.eye(len(G)))
            T = 0.5 * self.p @ G_inv @ self.p
            H_with.append(T + energies.total)

        with_geo_drift = [abs(H - H0) / (abs(H0) + 1e-10) for H in H_with]
        results['with_geo_drift'] = with_geo_drift[-1]
        results['with_geo_history'] = with_geo_drift

        # Run WITHOUT geodesic correction
        self.enable_geodesic_correction = False
        self.theta = theta_backup.copy()
        self.p = p_backup.copy()
        self._unpack_parameters(self.theta)

        H_without = [H0]
        for _ in range(n_steps):
            dtheta_dt, dp_dt = self._hamiltonian_equations(self.theta, self.p)
            self.p = self.p + 0.5 * dt * dp_dt
            dtheta_dt, _ = self._hamiltonian_equations(self.theta, self.p)
            self.theta = self.theta + dt * dtheta_dt
            self._unpack_parameters(self.theta)
            _, dp_dt = self._hamiltonian_equations(self.theta, self.p)
            self.p = self.p + 0.5 * dt * dp_dt

            # Compute H
            energies = compute_total_free_energy(self.system)
            G = self._compute_metric(self.theta)
            G_inv = np.linalg.inv(G + 1e-6 * np.eye(len(G)))
            T = 0.5 * self.p @ G_inv @ self.p
            H_without.append(T + energies.total)

        without_geo_drift = [abs(H - H0) / (abs(H0) + 1e-10) for H in H_without]
        results['without_geo_drift'] = without_geo_drift[-1]
        results['without_geo_history'] = without_geo_drift

        # Compute improvement factor
        improvement = results['without_geo_drift'] / (results['with_geo_drift'] + 1e-10)
        results['improvement_factor'] = improvement

        # Restore original state
        self.theta = theta_backup
        self.p = p_backup
        self._unpack_parameters(self.theta)
        self.enable_geodesic_correction = geo_setting_backup

        return results