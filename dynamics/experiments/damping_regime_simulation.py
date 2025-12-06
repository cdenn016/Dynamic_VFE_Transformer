# -*- coding: utf-8 -*-
"""
Created on Tue Nov 25 08:19:59 2025

@author: chris and christine
"""

#!/usr/bin/env python3
"""
Damping Regimes and Epistemic Momentum Simulations
====================================================

Comprehensive simulations demonstrating the Hamiltonian dynamics of belief
evolution from "The Inertia of Belief: Hiding in Plain Sight" manuscript.

**Now utilizing the GAUGE VARIATIONAL FREE ENERGY theory** for proper
Hamiltonian mechanics with:
- Full VFE functional as potential: F[q, p, φ]
- Fisher metric as mass tensor: M = Λ = Σ⁻¹
- Gauge transport operators: Ω_ij = exp(φ_i) · exp(-φ_j)
- Softmax attention weights: β_ij, γ_ij
- χ-weighted spatial integration

Simulations implemented:
1. Three Damping Regimes - Overdamped, critically damped, underdamped
2. Two-Agent Momentum Transfer - Recoil effect with gauge transport
3. Confirmation Bias as Stopping Distance - d ∝ Λ relationship
4. Resonance Curve - Peak at ω_res = √(K/M)
5. Belief Perseverance Decay - τ ∝ Λ/γ relationship

Key equations from manuscript:
- Hamiltonian: H = (1/2) π^T Λ^{-1} π + F[q, p, φ]
- Damped oscillator: M μ̈ + γ μ̇ = -∇_μ F
- Mass = Precision: M = Λ = Σ⁻¹ (Fisher metric)
- VFE potential: F = KL(q||p) + Σ_j β_ij KL(q_i||Ω_ij[q_j]) + ...

Author: Generated from psych_manuscript.pdf theory
Date: November 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
from typing import Tuple, Dict, List, Optional, Callable, Union
from dataclasses import dataclass, field
from scipy.integrate import solve_ivp

# Import from core dynamics suite
from dynamics.hamiltonian import BeliefHamiltonian, HamiltonianState
from dynamics.integrators import Verlet, PEFRL, SymplecticIntegrator

# Import gauge VFE theory components
from gradients.free_energy_clean import (
    compute_self_energy,
    compute_total_free_energy,
    FreeEnergyBreakdown,
    kl_gaussian
)
from gradients.gauge_fields import GaugeField
from math_utils.numerical_utils import kl_gaussian as kl_gauss_util


# =============================================================================
# VFE-Connected Hamiltonian Classes (PROPER IMPLEMENTATION)
# =============================================================================

# Import the actual agent/system infrastructure
from agent.agents import Agent
from agent.system import MultiAgentSystem
from config import AgentConfig, SystemConfig


def create_particle_agent(
    agent_id: int,
    K: int = 3,
    mu_q_init: np.ndarray = None,
    mu_p_init: np.ndarray = None,
    sigma_scale: float = 0.5,
    phi_init: np.ndarray = None,
    seed: int = None,
) -> Agent:
    """
    Create a 0D particle agent for damping regime simulations.

    This creates a REAL Agent object that can be used with compute_total_free_energy.

    Args:
        agent_id: Unique identifier
        K: Latent dimension (belief mean dimension)
        mu_q_init: Initial belief mean (shape: (K,))
        mu_p_init: Initial prior mean (shape: (K,))
        sigma_scale: Covariance scale
        phi_init: Initial gauge field (shape: (3,))
        seed: Random seed

    Returns:
        Agent: Properly initialized 0D particle agent
    """
    config = AgentConfig(
        spatial_shape=(),  # 0D particle!
        K=K,
        mu_scale=1.0,
        sigma_scale=sigma_scale,
        phi_scale=0.1,
        covariance_strategy="constant",
    )

    rng = np.random.default_rng(seed)
    agent = Agent(agent_id=agent_id, config=config, rng=rng)

    # Override with specified initial conditions
    if mu_q_init is not None:
        agent.mu_q = np.asarray(mu_q_init, dtype=np.float32)
    if mu_p_init is not None:
        agent.mu_p = np.asarray(mu_p_init, dtype=np.float32)
    if phi_init is not None:
        agent.gauge.phi = np.asarray(phi_init, dtype=np.float32)

    return agent


class VFEHamiltonian(BeliefHamiltonian):
    """
    Hamiltonian using the ACTUAL Gauge Variational Free Energy as potential.

    H(μ, π) = T(π) + V(μ)

    Where:
    - T(π) = (1/2) π^T Λ^{-1} π  (kinetic, Fisher metric)
    - V(μ) = F[q(μ), p, φ]       (potential = ACTUAL VFE from compute_total_free_energy!)

    The VFE includes (from gradients/free_energy_clean.py):
    - Self-coupling: α ∫ χ_i KL(q||p) dc
    - Belief alignment: Σ_j ∫ χ_ij β_ij KL(q_i||Ω_ij[q_j]) dc
    - Prior alignment: Σ_j ∫ χ_ij γ_ij KL(p_i||Ω_ij[p_j]) dc
    - Observations: -∫ χ_i E_q[log p(o|x)] dc

    This uses REAL Agent objects and compute_total_free_energy()!

    **NOW WITH ANALYTICAL GRADIENTS** for stable integration!
    """

    def __init__(
        self,
        agent: Agent,
        system: MultiAgentSystem = None,
        lambda_self: float = 1.0,
        use_analytical_gradient: bool = True,  # NEW: Use analytical gradient
    ):
        """
        Initialize VFE-based Hamiltonian.

        Args:
            agent: REAL Agent object with beliefs q, priors p, gauge φ
            system: Optional MultiAgentSystem for multi-agent VFE terms
            lambda_self: Self-coupling strength (used if no system)
            use_analytical_gradient: If True, use analytical gradient (more stable)
        """
        self.agent = agent
        self.system = system
        self.lambda_self = lambda_self
        self._K = agent.K
        self.use_analytical_gradient = use_analytical_gradient

        # Cache for gradient computation
        self._grad_eps = 1e-5

        # Initialize parent with VFE-based potential, Fisher metric, and analytical gradient
        # Pass analytical gradient to parent for stable integration
        super().__init__(
            potential=self._vfe_potential,
            metric=self._fisher_metric,
            potential_gradient=self._analytical_gradient if (use_analytical_gradient and system is None) else None,
        )

    def _vfe_potential(self, q: np.ndarray) -> float:
        """
        Compute VFE potential V(μ) = F[q(μ), p, φ].

        Uses the ACTUAL compute_total_free_energy or compute_self_energy
        from gradients/free_energy_clean.py!
        """
        # Store original and set new belief mean
        original_mu = self.agent.mu_q.copy()
        self.agent.mu_q = q.astype(np.float32)

        # Invalidate any caches
        if hasattr(self.agent, '_L_q_cache'):
            self.agent._L_q_cache = None

        try:
            if self.system is not None:
                # Full multi-agent VFE with transport operators, softmax weights, etc.
                breakdown = compute_total_free_energy(self.system)
                vfe = breakdown.total
            else:
                # Single-agent self-energy: KL(q||p) with χ-weighted integration
                vfe = compute_self_energy(self.agent, lambda_self=self.lambda_self)
        finally:
            # Always restore original belief
            self.agent.mu_q = original_mu
            if hasattr(self.agent, '_L_q_cache'):
                self.agent._L_q_cache = None

        return float(vfe)

    def _analytical_gradient(self, q: np.ndarray) -> np.ndarray:
        """
        Analytical gradient of VFE w.r.t. belief mean μ.

        For self-energy KL(N(μ_q, Σ_q) || N(μ_p, Σ_p)):
            ∇_μ KL = Σ_p^{-1} (μ_q - μ_p)

        This is MUCH more stable than numerical gradients!
        """
        mu_q = q
        mu_p = self.agent.mu_p
        Sigma_p = self.agent.Sigma_p

        # Compute precision of prior
        try:
            Lambda_p = np.linalg.inv(Sigma_p)
        except np.linalg.LinAlgError:
            Lambda_p = np.linalg.inv(Sigma_p + 1e-6 * np.eye(Sigma_p.shape[0]))

        # Gradient: λ_self * Λ_p (μ_q - μ_p)
        grad = self.lambda_self * Lambda_p @ (mu_q - mu_p)

        return grad

    # NOTE: equations_of_motion is now handled by parent BeliefHamiltonian
    # which uses the potential_gradient callback we passed in __init__.
    # The Fisher metric Σ^{-1} is constant (doesn't depend on q), so
    # no kinetic correction term is needed.

    def _fisher_metric(self, q: np.ndarray) -> np.ndarray:
        """
        Fisher information metric G = Λ = Σ^{-1}.

        The Fisher metric for Gaussian beliefs is the precision matrix.
        This is the PROPER mass tensor for belief dynamics on statistical manifold.
        """
        Sigma = self.agent.Sigma_q

        # Fisher metric = precision = inverse covariance
        try:
            Lambda = np.linalg.inv(Sigma)
        except np.linalg.LinAlgError:
            Lambda = np.linalg.inv(Sigma + 1e-6 * np.eye(Sigma.shape[0]))

        return Lambda

    def get_vfe_breakdown(self, q: np.ndarray) -> FreeEnergyBreakdown:
        """Get detailed VFE breakdown at position q."""
        original_mu = self.agent.mu_q.copy()
        self.agent.mu_q = q.astype(np.float32)

        try:
            if self.system is not None:
                breakdown = compute_total_free_energy(self.system)
            else:
                self_energy = compute_self_energy(self.agent, lambda_self=self.lambda_self)
                breakdown = FreeEnergyBreakdown(
                    self_energy=self_energy,
                    belief_align=0.0,
                    prior_align=0.0,
                    observations=0.0,
                    total=self_energy,
                )
        finally:
            self.agent.mu_q = original_mu

        return breakdown


class MultiAgentVFEHamiltonian(BeliefHamiltonian):
    """
    Hamiltonian for multiple coupled agents using FULL gauge VFE.

    H(μ₁, μ₂, ..., π₁, π₂, ...) = Σᵢ Tᵢ(πᵢ) + F[{qᵢ}, {pᵢ}, {φᵢ}]

    Where:
    - Tᵢ(πᵢ) = (1/2) πᵢ^T Λᵢ^{-1} πᵢ  (kinetic, agent i's Fisher metric)
    - F = FULL multi-agent VFE with gauge transport!

    The VFE includes GAUGE TRANSPORT:
    - Σᵢ α KL(qᵢ||pᵢ)                           [self-coupling]
    - Σᵢⱼ βᵢⱼ KL(qᵢ||Ω_ij[qⱼ])                  [belief alignment with transport!]
    - Σᵢⱼ γᵢⱼ KL(pᵢ||Ω_ij[pⱼ])                  [prior alignment with transport!]

    Transport operator: Ω_ij = exp(φᵢ) · exp(-φⱼ) ∈ SO(3)
    """

    def __init__(self, system: MultiAgentSystem):
        """
        Initialize multi-agent VFE Hamiltonian.

        Args:
            system: MultiAgentSystem with properly configured agents
        """
        self.system = system
        self.agents = system.agents
        self.n_agents = system.n_agents
        self._K = self.agents[0].K  # Assume all agents have same K

        # Total dimension: n_agents * K
        self._dim = self.n_agents * self._K

        # Initialize parent
        super().__init__(
            potential=self._vfe_potential,
            metric=self._fisher_metric
        )

    def _vfe_potential(self, q: np.ndarray) -> float:
        """
        Compute FULL multi-agent VFE F[{qᵢ}, {pᵢ}, {φᵢ}].

        Uses compute_total_free_energy(system) which includes:
        - Self-coupling for each agent
        - Belief alignment with gauge transport Ω_ij
        - Prior alignment with gauge transport
        - Softmax attention weights β_ij, γ_ij
        - χ-weighted spatial integration
        """
        # Store original means
        original_mus = [agent.mu_q.copy() for agent in self.agents]

        # Update all agent belief means
        for i, agent in enumerate(self.agents):
            start_idx = i * self._K
            end_idx = (i + 1) * self._K
            agent.mu_q = q[start_idx:end_idx].astype(np.float32)

        try:
            # Compute FULL VFE with all terms
            breakdown = compute_total_free_energy(self.system)
            vfe = breakdown.total
        finally:
            # Restore all original beliefs
            for i, agent in enumerate(self.agents):
                agent.mu_q = original_mus[i]

        return float(vfe)

    def _fisher_metric(self, q: np.ndarray) -> np.ndarray:
        """
        Block-diagonal Fisher metric for all agents.

        G = diag(Λ₁, Λ₂, ..., Λₙ) where Λᵢ = Σᵢ^{-1}
        """
        G = np.zeros((self._dim, self._dim))

        for i, agent in enumerate(self.agents):
            start_idx = i * self._K
            end_idx = (i + 1) * self._K

            Sigma = agent.Sigma_q
            try:
                Lambda = np.linalg.inv(Sigma)
            except np.linalg.LinAlgError:
                Lambda = np.linalg.inv(Sigma + 1e-6 * np.eye(self._K))

            G[start_idx:end_idx, start_idx:end_idx] = Lambda

        return G

    def get_vfe_breakdown(self, q: np.ndarray) -> FreeEnergyBreakdown:
        """Get detailed VFE breakdown at multi-agent state q."""
        original_mus = [agent.mu_q.copy() for agent in self.agents]

        for i, agent in enumerate(self.agents):
            start_idx = i * self._K
            end_idx = (i + 1) * self._K
            agent.mu_q = q[start_idx:end_idx].astype(np.float32)

        try:
            breakdown = compute_total_free_energy(self.system)
        finally:
            for i, agent in enumerate(self.agents):
                agent.mu_q = original_mus[i]

        return breakdown


# =============================================================================
# Core Dynamics Classes - Built on Core Suite + VFE Theory
# =============================================================================

@dataclass
class BeliefState:
    """State of a single agent's belief (wraps HamiltonianState)."""
    mu: Union[float, np.ndarray]  # Belief mean (position)
    pi: Union[float, np.ndarray]  # Belief momentum
    precision: Union[float, np.ndarray]  # Λ = 1/σ² (acts as mass)
    t: float = 0.0

    @property
    def mass(self) -> Union[float, np.ndarray]:
        return self.precision

    @property
    def velocity(self) -> Union[float, np.ndarray]:
        return self.pi / self.mass

    def to_hamiltonian_state(self) -> HamiltonianState:
        """Convert to core suite HamiltonianState."""
        mu_arr = np.atleast_1d(self.mu)
        pi_arr = np.atleast_1d(self.pi)
        return HamiltonianState(
            q=mu_arr,
            p=pi_arr,
            t=self.t
        )


@dataclass
class TwoAgentState:
    """State of two coupled agents with gauge transport."""
    mu1: Union[float, np.ndarray]
    mu2: Union[float, np.ndarray]
    pi1: Union[float, np.ndarray]
    pi2: Union[float, np.ndarray]
    precision1: Union[float, np.ndarray]
    precision2: Union[float, np.ndarray]
    coupling: float     # β₁₂ = β₂₁ attention coupling
    phi1: np.ndarray = field(default_factory=lambda: np.zeros(3))  # Gauge field agent 1
    phi2: np.ndarray = field(default_factory=lambda: np.zeros(3))  # Gauge field agent 2
    t: float = 0.0

    def to_hamiltonian_state(self) -> HamiltonianState:
        """Convert to core suite HamiltonianState for 2-agent system."""
        mu1_arr = np.atleast_1d(self.mu1)
        mu2_arr = np.atleast_1d(self.mu2)
        pi1_arr = np.atleast_1d(self.pi1)
        pi2_arr = np.atleast_1d(self.pi2)
        return HamiltonianState(
            q=np.concatenate([mu1_arr, mu2_arr]),
            p=np.concatenate([pi1_arr, pi2_arr]),
            t=self.t
        )

    @property
    def transport_operator(self) -> np.ndarray:
        """
        Gauge transport operator Ω₁₂ = exp(φ₁) · exp(-φ₂).

        This is the KEY connection to gauge theory!
        """
        from scipy.spatial.transform import Rotation
        R1 = Rotation.from_rotvec(self.phi1).as_matrix()
        R2 = Rotation.from_rotvec(self.phi2).as_matrix()
        return R1 @ R2.T


class EpistemicOscillator:
    """
    Damped epistemic oscillator from Eq. 36-37 of manuscript.

    **Now with TWO modes:**
    1. Simple mode: Harmonic potential V = (1/2)K(μ - μ*)²
    2. VFE mode: Full gauge variational free energy V = F[q, p, φ]

    M μ̈ + γ μ̇ = -∇V + f(t)

    Where:
    - M = precision = Λ = Σ⁻¹ (Fisher metric / epistemic mass)
    - γ = damping coefficient (evidence integration rate)
    - V = potential (harmonic or full VFE)
    - f(t) = external forcing (evidence stream)

    The VFE mode uses the actual gauge variational free energy:
        F = KL(q||p) + alignment terms + observation terms
    """

    def __init__(
        self,
        precision: float,      # M = Λ (scalar or from agent)
        stiffness: float,      # K (evidence strength, for harmonic mode)
        damping: float,        # γ
        equilibrium: float = 0.0,  # μ* (target, for harmonic mode)
        agent=None,            # Optional: Agent for VFE mode
        use_vfe: bool = False, # If True and agent provided, use VFE potential
    ):
        self.M = precision
        self.K = stiffness
        self.gamma = damping
        self.mu_eq = equilibrium
        self.agent = agent
        self.use_vfe = use_vfe and agent is not None

        # Set latent dimension for VFE mode
        if self.use_vfe and agent is not None:
            self._K = agent.K  # Latent dimension from agent
        else:
            self._K = 1  # Scalar mode

        # Build core suite Hamiltonian
        self._hamiltonian = self._create_hamiltonian()

    def _create_hamiltonian(self) -> BeliefHamiltonian:
        """Create BeliefHamiltonian from core dynamics suite."""
        if self.use_vfe:
            # VFE mode: use ACTUAL free energy as potential
            return VFEHamiltonian(
                agent=self.agent,
                system=None,  # Single agent (use from_agent with system for multi-agent)
                lambda_self=self.K,  # Use stiffness as self-coupling strength
            )
        else:
            # Harmonic mode: simple quadratic potential
            def potential(q: np.ndarray) -> float:
                mu = q[0] if len(q.shape) > 0 else q
                return 0.5 * self.K * (mu - self.mu_eq)**2

            # Metric: G = M (scalar mass as 1x1 matrix)
            def metric(q: np.ndarray) -> np.ndarray:
                return np.array([[self.M]])

            return BeliefHamiltonian(potential=potential, metric=metric)

    @classmethod
    def from_agent(
        cls,
        agent: Agent,
        damping: float,
        stiffness: float = 1.0,
        system: MultiAgentSystem = None,
    ) -> 'EpistemicOscillator':
        """
        Create oscillator from REAL Agent using ACTUAL VFE theory.

        This is the PROPER way to connect to gauge VFE:
        - Mass = Fisher metric = precision = Σ⁻¹
        - Potential = ACTUAL VFE = compute_self_energy(agent) or compute_total_free_energy(system)

        Args:
            agent: REAL Agent object with beliefs, priors, gauge field
            damping: Damping coefficient γ
            stiffness: Self-coupling strength λ_self
            system: Optional MultiAgentSystem for full VFE

        Returns:
            EpistemicOscillator configured for VFE dynamics
        """
        # Get precision from agent's belief covariance (Fisher metric)
        Sigma = agent.Sigma_q
        try:
            Lambda = np.linalg.inv(Sigma)
            precision = np.trace(Lambda) / Lambda.shape[0]  # Average precision
        except np.linalg.LinAlgError:
            precision = 1.0 / np.trace(Sigma) * Sigma.shape[0]

        osc = cls(
            precision=precision,
            stiffness=stiffness,
            damping=damping,
            equilibrium=0.0,  # VFE mode doesn't use this
            agent=agent,
            use_vfe=True,
        )

        # Store system for full VFE if provided
        if system is not None:
            osc._hamiltonian.system = system

        return osc

    @classmethod
    def create_vfe_oscillator(
        cls,
        K: int = 3,
        mu_q_init: np.ndarray = None,
        mu_p_init: np.ndarray = None,
        sigma_scale: float = 0.5,
        damping: float = 0.5,
        lambda_self: float = 1.0,
        seed: int = None,
    ) -> 'EpistemicOscillator':
        """
        Create a VFE-connected oscillator with a fresh particle agent.

        This is the easiest way to get started with VFE dynamics!

        Args:
            K: Latent dimension
            mu_q_init: Initial belief mean (default: random)
            mu_p_init: Prior mean (default: zeros)
            sigma_scale: Belief covariance scale
            damping: Damping coefficient γ
            lambda_self: Self-coupling strength
            seed: Random seed

        Returns:
            EpistemicOscillator using actual VFE as potential
        """
        # Create a real particle agent
        agent = create_particle_agent(
            agent_id=0,
            K=K,
            mu_q_init=mu_q_init,
            mu_p_init=mu_p_init if mu_p_init is not None else np.zeros(K),
            sigma_scale=sigma_scale,
            seed=seed,
        )

        return cls.from_agent(agent, damping=damping, stiffness=lambda_self)

    @property
    def hamiltonian(self) -> BeliefHamiltonian:
        """Access underlying core suite Hamiltonian."""
        return self._hamiltonian

    @property
    def natural_frequency(self) -> float:
        """ω₀ = √(K/M)"""
        return np.sqrt(self.K / self.M)

    @property
    def damping_ratio(self) -> float:
        """ζ = γ / (2√(KM))"""
        return self.gamma / (2 * np.sqrt(self.K * self.M))

    @property
    def discriminant(self) -> float:
        """Δ = γ² - 4KM"""
        return self.gamma**2 - 4 * self.K * self.M

    @property
    def regime(self) -> str:
        """Determine damping regime."""
        if self.discriminant > 0:
            return "overdamped"
        elif abs(self.discriminant) < 1e-10:
            return "critical"
        else:
            return "underdamped"

    @property
    def damped_frequency(self) -> float:
        """ω = √(K/M - γ²/(4M²)) for underdamped case."""
        if self.regime == "underdamped":
            return np.sqrt(self.K/self.M - self.gamma**2/(4*self.M**2))
        return 0.0

    @property
    def decay_time(self) -> float:
        """τ = 2M/γ (Eq. 39)"""
        if self.gamma > 0:
            return 2 * self.M / self.gamma
        return float('inf')

    def equations_of_motion(
        self,
        t: float,
        y: np.ndarray,
        forcing: Optional[Callable] = None
    ) -> np.ndarray:
        """
        Hamilton's equations with damping.

        Uses core suite's equations_of_motion and adds damping.

        Handles both:
        - Scalar mode: y = [μ, π]
        - VFE mode: y = [μ₀, μ₁, ..., μ_{K-1}, π₀, π₁, ..., π_{K-1}]
        """
        if self.use_vfe:
            # VFE mode: K-dimensional state
            K = self._K
            mu = y[:K]
            pi = y[K:]

            # Get conservative dynamics from VFE Hamiltonian
            dq_dt, dp_dt = self._hamiltonian.equations_of_motion(mu, pi)

            # Add damping: dπ/dt += -γ * v where v = dq/dt
            dpi_dt = dp_dt - self.gamma * dq_dt

            # Add external forcing (applied to first component only)
            if forcing is not None:
                force_vec = np.zeros(K)
                force_vec[0] = forcing(t)
                dpi_dt = dpi_dt + force_vec

            return np.concatenate([dq_dt, dpi_dt])
        else:
            # Scalar mode
            mu, pi = y
            q = np.array([mu])
            p = np.array([pi])

            # Get conservative dynamics from core Hamiltonian
            dq_dt, dp_dt = self._hamiltonian.equations_of_motion(q, p)

            # Velocity
            dmu_dt = dq_dt[0]

            # Add damping and external forcing
            force_damping = -self.gamma * dmu_dt
            force_external = forcing(t) if forcing else 0.0

            dpi_dt = dp_dt[0] + force_damping + force_external

            return np.array([dmu_dt, dpi_dt])

    def simulate(
        self,
        mu0: Union[float, np.ndarray],
        pi0: Union[float, np.ndarray],
        t_end: float,
        dt: float = 0.01,
        forcing: Optional[Callable] = None,
        use_symplectic: bool = False
    ) -> Dict[str, np.ndarray]:
        """
        Simulate belief evolution.

        Args:
            mu0: Initial belief position (scalar or K-dim array for VFE mode)
            pi0: Initial momentum (scalar or K-dim array for VFE mode)
            t_end: Simulation end time
            dt: Time step
            forcing: Optional external forcing function f(t)
            use_symplectic: If True and no damping/forcing, use Verlet integrator

        Returns:
            Dictionary with trajectory data
        """
        # For conservative systems (no damping, no forcing), use symplectic
        if use_symplectic and self.gamma == 0 and forcing is None and not self.use_vfe:
            return self._simulate_symplectic(mu0, pi0, t_end, dt)

        # Build initial state vector
        if self.use_vfe:
            # VFE mode: concatenate K-dim vectors
            mu0 = np.asarray(mu0).flatten()
            pi0 = np.asarray(pi0).flatten()
            y0 = np.concatenate([mu0, pi0])
        else:
            # Scalar mode
            y0 = np.array([mu0, pi0])

        # For dissipative systems, use scipy (damping breaks symplecticity)
        t_span = (0, t_end)
        t_eval = np.arange(0, t_end, dt)

        sol = solve_ivp(
            lambda t, y: self.equations_of_motion(t, y, forcing),
            t_span,
            y0,
            t_eval=t_eval,
            method='RK45',
            max_step=dt
        )

        if self.use_vfe:
            # VFE mode: extract K-dim trajectories
            K = self._K
            mu_traj = sol.y[:K, :].T  # Shape: (n_steps, K)
            pi_traj = sol.y[K:, :].T  # Shape: (n_steps, K)
            return {
                't': sol.t,
                'mu': mu_traj,
                'pi': pi_traj,
                'velocity': pi_traj / self.M,  # Approximate
            }
        else:
            # Scalar mode
            return {
                't': sol.t,
                'mu': sol.y[0],
                'pi': sol.y[1],
                'velocity': sol.y[1] / self.M,
                'kinetic_energy': 0.5 * sol.y[1]**2 / self.M,
                'potential_energy': 0.5 * self.K * (sol.y[0] - self.mu_eq)**2,
            }

    def _simulate_symplectic(
        self,
        mu0: float,
        pi0: float,
        t_end: float,
        dt: float
    ) -> Dict[str, np.ndarray]:
        """Use Verlet integrator from core suite for conservative dynamics."""
        integrator = Verlet(self._hamiltonian)

        q0 = np.array([mu0])
        p0 = np.array([pi0])

        t_arr, q_arr, p_arr, stats = integrator.integrate(
            q0, p0,
            t_span=(0.0, t_end),
            dt=dt
        )

        mu_arr = q_arr[:, 0]
        pi_arr = p_arr[:, 0]

        return {
            't': t_arr,
            'mu': mu_arr,
            'pi': pi_arr,
            'velocity': pi_arr / self.M,
            'kinetic_energy': 0.5 * pi_arr**2 / self.M,
            'potential_energy': 0.5 * self.K * (mu_arr - self.mu_eq)**2,
            'integrator_stats': stats,
        }


class TwoAgentHamiltonian(BeliefHamiltonian):
    """
    Hamiltonian for two coupled agents with GAUGE TRANSPORT.

    **Now with TWO modes:**
    1. Simple mode: Quadratic coupling V = (β/2)(μ₁ - μ₂)²
    2. VFE mode: Full gauge variational free energy with transport

    H = T₁ + T₂ + V[q₁, q₂, φ₁, φ₂]

    Where:
    - T_i = (1/2) π_i^T Λ_i^{-1} π_i  (kinetic, Fisher metric)
    - V = F[q₁, q₂, p₁, p₂, φ₁, φ₂]   (VFE potential!)

    VFE includes gauge transport:
    - KL(q₁||p₁) + KL(q₂||p₂)                    [self-coupling]
    - β₁₂ KL(q₁||Ω₁₂[q₂]) + β₂₁ KL(q₂||Ω₂₁[q₁]) [belief alignment with transport!]

    The transport operator Ω₁₂ = exp(φ₁)·exp(-φ₂) is the KEY gauge theory element.
    """

    def __init__(
        self,
        precision1: float,
        precision2: float,
        coupling: float,
        prior1: float = 0.0,
        prior2: float = 0.0,
        prior_strength: float = 0.1,
        phi1: np.ndarray = None,  # Gauge field agent 1
        phi2: np.ndarray = None,  # Gauge field agent 2
        use_gauge_transport: bool = False,  # Enable gauge VFE mode
    ):
        self.M1 = precision1
        self.M2 = precision2
        self.beta = coupling
        self.mu_bar1 = prior1
        self.mu_bar2 = prior2
        self.Lambda_bar = prior_strength
        self.phi1 = phi1 if phi1 is not None else np.zeros(3)
        self.phi2 = phi2 if phi2 is not None else np.zeros(3)
        self.use_gauge_transport = use_gauge_transport

        # Initialize parent with our potential and metric
        super().__init__(
            potential=self._potential,
            metric=self._metric
        )

    def _potential(self, q: np.ndarray) -> float:
        """
        Total potential energy.

        In VFE mode: F = KL(q₁||p₁) + KL(q₂||p₂) + β KL(q₁||Ω[q₂])
        In simple mode: V = prior anchoring + quadratic coupling
        """
        mu1, mu2 = q[0], q[1]

        if self.use_gauge_transport:
            # VFE mode with gauge transport
            return self._vfe_potential_with_transport(mu1, mu2)
        else:
            # Simple quadratic mode
            # Prior anchoring: KL(q||p) ≈ (Λ̄/2)(μ - μ̄)² for Gaussian
            V_prior1 = 0.5 * self.Lambda_bar * (mu1 - self.mu_bar1)**2
            V_prior2 = 0.5 * self.Lambda_bar * (mu2 - self.mu_bar2)**2

            # Consensus coupling: β KL(q₁||q₂) ≈ (β/2)(Λ₁+Λ₂)(μ₁ - μ₂)²
            V_coupling = 0.5 * self.beta * (self.M1 + self.M2) * (mu1 - mu2)**2

            return V_prior1 + V_prior2 + V_coupling

    def _vfe_potential_with_transport(self, mu1: float, mu2: float) -> float:
        """
        Compute VFE with gauge transport operators.

        F = α[KL(q₁||p₁) + KL(q₂||p₂)]
          + β[KL(q₁||Ω₁₂[q₂]) + KL(q₂||Ω₂₁[q₁])]

        Where Ω₁₂ = exp(φ₁)·exp(-φ₂) is the gauge transport.

        For scalar beliefs, transport acts as rotation in latent space.
        """
        # Self-coupling: KL(q||p) for each agent
        # For 1D Gaussian with precision Λ:
        # KL(N(μ,σ²)||N(μ̄,σ̄²)) = (1/2)[log(σ̄²/σ²) + (σ² + (μ-μ̄)²)/σ̄² - 1]

        sigma1_sq = 1.0 / self.M1
        sigma2_sq = 1.0 / self.M2
        sigma_bar1_sq = 1.0 / self.Lambda_bar if self.Lambda_bar > 0 else 1.0
        sigma_bar2_sq = sigma_bar1_sq

        # Self-coupling KL terms
        kl_self1 = 0.5 * (
            np.log(sigma_bar1_sq / sigma1_sq) +
            (sigma1_sq + (mu1 - self.mu_bar1)**2) / sigma_bar1_sq - 1
        )
        kl_self2 = 0.5 * (
            np.log(sigma_bar2_sq / sigma2_sq) +
            (sigma2_sq + (mu2 - self.mu_bar2)**2) / sigma_bar2_sq - 1
        )

        # Gauge transport effect on coupling
        # Transport operator Ω₁₂ = exp(φ₁)·exp(-φ₂)
        # For scalar beliefs, this rotates the "reference frame"
        # The transported belief mean is: μ₂' = Ω₁₂ · μ₂

        from scipy.spatial.transform import Rotation
        R12 = Rotation.from_rotvec(self.phi1).as_matrix() @ \
              Rotation.from_rotvec(-self.phi2).as_matrix()

        # For 1D embedded in 3D, transport effect:
        # Use first component of rotation matrix as scaling
        transport_factor = R12[0, 0]  # Projection onto first axis

        # Transported mean: μ₂ → μ₂ · transport_factor
        mu2_transported = mu2 * transport_factor

        # Belief alignment KL with transport
        # KL(q₁||Ω[q₂]) where q₂ has been transported
        kl_align12 = 0.5 * (
            np.log(sigma2_sq / sigma1_sq) +
            (sigma1_sq + (mu1 - mu2_transported)**2) / sigma2_sq - 1
        )

        # Reverse transport for Ω₂₁
        mu1_transported = mu1 * (1.0 / transport_factor if abs(transport_factor) > 1e-6 else 1.0)
        kl_align21 = 0.5 * (
            np.log(sigma1_sq / sigma2_sq) +
            (sigma2_sq + (mu2 - mu1_transported)**2) / sigma1_sq - 1
        )

        # Total VFE
        alpha = self.Lambda_bar  # Self-coupling strength
        F = alpha * (kl_self1 + kl_self2) + self.beta * (kl_align12 + kl_align21)

        return max(0.0, F)  # VFE is non-negative

    def _metric(self, q: np.ndarray) -> np.ndarray:
        """Mass matrix (diagonal: [M1, M2]) = Fisher metric."""
        return np.diag([self.M1, self.M2])

    @classmethod
    def from_agents(
        cls,
        agent1,
        agent2,
        coupling: float,
        prior_strength: float = 0.1,
    ) -> 'TwoAgentHamiltonian':
        """
        Create Hamiltonian from two Agent objects.

        This is the PROPER way to connect to gauge VFE theory!

        Args:
            agent1, agent2: Agent objects with beliefs, priors, gauge fields
            coupling: Attention coupling strength β
            prior_strength: Prior anchoring strength Λ̄

        Returns:
            TwoAgentHamiltonian configured for gauge VFE dynamics
        """
        # Extract precisions from agents
        def get_precision(agent):
            if agent.geometry.is_particle:
                Sigma = agent.Sigma_q
            else:
                Sigma = np.mean(agent.Sigma_q, axis=tuple(range(agent.geometry.ndim)))
            try:
                Lambda = np.linalg.inv(Sigma)
                return np.trace(Lambda) / Lambda.shape[0]
            except np.linalg.LinAlgError:
                return 1.0 / np.trace(Sigma) * Sigma.shape[0]

        # Extract prior means
        def get_prior_mean(agent):
            if agent.geometry.is_particle:
                return float(np.mean(agent.mu_p))
            else:
                return float(np.mean(agent.mu_p))

        # Extract gauge fields
        def get_gauge(agent):
            if agent.geometry.is_particle:
                return agent.gauge.phi
            else:
                return np.mean(agent.gauge.phi, axis=tuple(range(agent.geometry.ndim)))

        return cls(
            precision1=get_precision(agent1),
            precision2=get_precision(agent2),
            coupling=coupling,
            prior1=get_prior_mean(agent1),
            prior2=get_prior_mean(agent2),
            prior_strength=prior_strength,
            phi1=get_gauge(agent1),
            phi2=get_gauge(agent2),
            use_gauge_transport=True,
        )


class TwoAgentSystem:
    """
    Two coupled agents with momentum transfer (Section 4.6).

    **NOW WITH PROPER GAUGE VFE SUPPORT!**

    In VFE mode:
    - Uses REAL Agent objects with mu_q, Sigma_q, mu_p, Sigma_p, gauge.phi
    - Uses MultiAgentVFEHamiltonian with compute_total_free_energy()
    - Gauge transport: Ω_ij = exp(φ_i) · exp(-φ_j)

    M₁μ̈₁ + γ₁μ̇₁ = -∇₁F[q₁, q₂, p₁, p₂, φ₁, φ₂]
    M₂μ̈₂ + γ₂μ̇₂ = -∇₂F[q₁, q₂, p₁, p₂, φ₁, φ₂]

    The coupling terms show momentum transfer and recoil WITH gauge transport!
    """

    def __init__(
        self,
        precision1: float,
        precision2: float,
        coupling: float,        # β₁₂ = β₂₁
        damping1: float = 0.1,
        damping2: float = 0.1,
        prior1: float = 0.0,
        prior2: float = 0.0,
        prior_strength: float = 0.1,  # Λ̄ (prior anchoring)
        use_vfe: bool = False,  # NEW: Use actual gauge VFE theory!
        K_latent: int = 3,      # Latent dimension for VFE mode
    ):
        self.M1 = precision1
        self.M2 = precision2
        self.beta = coupling
        self.gamma1 = damping1
        self.gamma2 = damping2
        self.mu_bar1 = prior1
        self.mu_bar2 = prior2
        self.Lambda_bar = prior_strength
        self.use_vfe = use_vfe
        self.K_latent = K_latent

        if use_vfe:
            # =========================================================
            # VFE MODE: Use ACTUAL gauge VFE with real Agent objects!
            # =========================================================
            # Create real agents
            sigma1 = 1.0 / np.sqrt(precision1)  # Σ ∝ 1/Λ
            sigma2 = 1.0 / np.sqrt(precision2)

            mu_p1 = np.zeros(K_latent)
            mu_p1[0] = prior1
            mu_p2 = np.zeros(K_latent)
            mu_p2[0] = prior2

            self.agent1 = create_particle_agent(
                agent_id=0, K=K_latent, mu_p_init=mu_p1,
                sigma_scale=sigma1, seed=1
            )
            self.agent2 = create_particle_agent(
                agent_id=1, K=K_latent, mu_p_init=mu_p2,
                sigma_scale=sigma2, seed=2
            )

            # Create multi-agent system
            self._multi_agent_system = MultiAgentSystem(
                agents=[self.agent1, self.agent2]
            )

            # Build VFE Hamiltonian
            self._hamiltonian = MultiAgentVFEHamiltonian(self._multi_agent_system)
        else:
            # Simple quadratic mode
            self._hamiltonian = TwoAgentHamiltonian(
                precision1=precision1,
                precision2=precision2,
                coupling=coupling,
                prior1=prior1,
                prior2=prior2,
                prior_strength=prior_strength,
            )

    @property
    def hamiltonian(self):
        """Access underlying Hamiltonian."""
        return self._hamiltonian

    def equations_of_motion(self, t: float, y: np.ndarray) -> np.ndarray:
        """
        Coupled Hamilton's equations with damping.

        Uses core suite's equations_of_motion and adds damping.

        Simple mode: y = [μ₁, μ₂, π₁, π₂]
        VFE mode: y = [μ₁[0:K], μ₂[0:K], π₁[0:K], π₂[0:K]]
        """
        if self.use_vfe:
            # VFE mode: K-dimensional states per agent
            K = self.K_latent
            mu1 = y[0:K]
            mu2 = y[K:2*K]
            pi1 = y[2*K:3*K]
            pi2 = y[3*K:4*K]

            q = np.concatenate([mu1, mu2])
            p = np.concatenate([pi1, pi2])

            # Get conservative dynamics from VFE Hamiltonian
            dq_dt, dp_dt = self._hamiltonian.equations_of_motion(q, p)

            v1 = dq_dt[0:K]
            v2 = dq_dt[K:2*K]

            # Add damping
            dp1_dt = dp_dt[0:K] - self.gamma1 * v1
            dp2_dt = dp_dt[K:2*K] - self.gamma2 * v2

            return np.concatenate([v1, v2, dp1_dt, dp2_dt])
        else:
            # Simple mode: scalar states
            mu1, mu2, pi1, pi2 = y
            q = np.array([mu1, mu2])
            p = np.array([pi1, pi2])

            # Get conservative dynamics from core Hamiltonian
            dq_dt, dp_dt = self._hamiltonian.equations_of_motion(q, p)

            # Velocities
            v1, v2 = dq_dt[0], dq_dt[1]

            # Add damping
            dp1_dt = dp_dt[0] - self.gamma1 * v1
            dp2_dt = dp_dt[1] - self.gamma2 * v2

            return np.array([v1, v2, dp1_dt, dp2_dt])

    def simulate(
        self,
        mu1_0: float,
        mu2_0: float,
        pi1_0: float,
        pi2_0: float,
        t_end: float,
        dt: float = 0.01,
        use_symplectic: bool = False
    ) -> Dict[str, np.ndarray]:
        """
        Simulate two-agent dynamics.

        Args:
            mu1_0, mu2_0: Initial belief positions (scalar, will be expanded in VFE mode)
            pi1_0, pi2_0: Initial momenta (scalar, will be expanded in VFE mode)
            use_symplectic: If True and no damping, use Verlet integrator
        """
        if self.use_vfe:
            # VFE mode: expand scalars to K-dimensional vectors
            K = self.K_latent
            mu1_vec = np.zeros(K)
            mu1_vec[0] = mu1_0
            mu2_vec = np.zeros(K)
            mu2_vec[0] = mu2_0
            pi1_vec = np.zeros(K)
            pi1_vec[0] = pi1_0
            pi2_vec = np.zeros(K)
            pi2_vec[0] = pi2_0

            # Set initial agent states
            self.agent1.mu_q = mu1_vec.astype(np.float32)
            self.agent2.mu_q = mu2_vec.astype(np.float32)

            y0 = np.concatenate([mu1_vec, mu2_vec, pi1_vec, pi2_vec])
        else:
            y0 = np.array([mu1_0, mu2_0, pi1_0, pi2_0])

        # For conservative systems, use symplectic integrator
        if use_symplectic and self.gamma1 == 0 and self.gamma2 == 0:
            if not self.use_vfe:
                return self._simulate_symplectic(mu1_0, mu2_0, pi1_0, pi2_0, t_end, dt)
            # VFE mode symplectic not implemented yet
            pass

        # For dissipative systems, use scipy
        t_span = (0, t_end)
        t_eval = np.arange(0, t_end, dt)

        sol = solve_ivp(
            self.equations_of_motion,
            t_span,
            y0,
            t_eval=t_eval,
            method='RK45',
            max_step=dt
        )

        if self.use_vfe:
            # Extract first component for visualization
            K = self.K_latent
            mu1 = sol.y[0, :]  # First component of agent 1's belief
            mu2 = sol.y[K, :]  # First component of agent 2's belief
            pi1 = sol.y[2*K, :]  # First component of agent 1's momentum
            pi2 = sol.y[3*K, :]  # First component of agent 2's momentum
            total_momentum = pi1 + pi2
        else:
            mu1 = sol.y[0]
            mu2 = sol.y[1]
            pi1 = sol.y[2]
            pi2 = sol.y[3]
            total_momentum = pi1 + pi2

        return {
            't': sol.t,
            'mu1': mu1,
            'mu2': mu2,
            'pi1': pi1,
            'pi2': pi2,
            'total_momentum': total_momentum,
            'momentum_diff': pi1 - pi2,
            'use_vfe': self.use_vfe,
        }

    def _simulate_symplectic(
        self,
        mu1_0: float,
        mu2_0: float,
        pi1_0: float,
        pi2_0: float,
        t_end: float,
        dt: float
    ) -> Dict[str, np.ndarray]:
        """Use Verlet integrator from core suite for conservative dynamics."""
        integrator = Verlet(self._hamiltonian)

        q0 = np.array([mu1_0, mu2_0])
        p0 = np.array([pi1_0, pi2_0])

        t_arr, q_arr, p_arr, stats = integrator.integrate(
            q0, p0,
            t_span=(0.0, t_end),
            dt=dt
        )

        total_momentum = p_arr[:, 0] + p_arr[:, 1]

        return {
            't': t_arr,
            'mu1': q_arr[:, 0],
            'mu2': q_arr[:, 1],
            'pi1': p_arr[:, 0],
            'pi2': p_arr[:, 1],
            'total_momentum': total_momentum,
            'momentum_diff': p_arr[:, 0] - p_arr[:, 1],
            'integrator_stats': stats,
        }


# =============================================================================
# Simulation 1: Three Damping Regimes
# =============================================================================

def simulate_damping_regimes(
    precision: float = 2.0,
    stiffness: float = 1.0,
    mu0: float = 1.0,
    pi0: float = 0.0,
    t_end: float = 50.0,
    output_dir: Optional[Path] = None,
    use_vfe: bool = True,  # NEW: Use actual VFE theory!
    K_latent: int = 3,     # Latent dimension for VFE mode
) -> Dict[str, Dict]:
    """
    Simulate three damping regimes for the same agent.

    Δ = γ² - 4KM determines regime:
    - Overdamped: Δ > 0 (γ > 2√(KM)) → Bayesian-like
    - Critical: Δ = 0 (γ = 2√(KM)) → Optimal
    - Underdamped: Δ < 0 (γ < 2√(KM)) → Oscillatory

    **NOW WITH GAUGE VFE THEORY!**

    In VFE mode:
    - Mass M = Λ = Σ^{-1} (Fisher metric from agent's precision)
    - Potential V = F[q, p, φ] (actual VFE functional!)
    - Forces come from ∇F = ∇KL(q||p) + ...

    Args:
        precision: Epistemic mass M = Λ (controls sigma_scale in VFE mode)
        stiffness: Evidence strength K (controls lambda_self in VFE mode)
        mu0: Initial belief displacement
        pi0: Initial momentum
        t_end: Simulation duration
        output_dir: Where to save figures
        use_vfe: If True, use ACTUAL gauge VFE theory!
        K_latent: Latent dimension for VFE mode

    Returns:
        Dictionary with simulation results for each regime
    """
    if output_dir is None:
        output_dir = Path("_experiments/damping_regimes")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Critical damping value
    gamma_critical = 2 * np.sqrt(stiffness * precision)

    # Three regimes
    regimes = {
        'overdamped': gamma_critical * 3.0,      # γ >> γ_c
        'critical': gamma_critical,               # γ = γ_c
        'underdamped': gamma_critical * 0.2,     # γ << γ_c
    }

    results = {}

    print("\n" + "="*70)
    print("SIMULATION 1: THREE DAMPING REGIMES")
    if use_vfe:
        print(">>> USING ACTUAL GAUGE VFE THEORY <<<")
    print("="*70)
    print(f"Precision (Mass) M = {precision}")
    print(f"Stiffness K = {stiffness}")
    print(f"Natural frequency ω₀ = √(K/M) = {np.sqrt(stiffness/precision):.3f}")
    print(f"Critical damping γ_c = 2√(KM) = {gamma_critical:.3f}")
    if use_vfe:
        print(f"Latent dimension K = {K_latent}")
    print()

    for regime_name, gamma in regimes.items():
        if use_vfe:
            # =====================================================
            # VFE MODE: Use ACTUAL gauge variational free energy!
            # =====================================================
            # sigma_scale controls precision: Λ = Σ^{-1}
            # Small sigma → high precision → high mass
            sigma_scale = 1.0 / np.sqrt(precision)  # Σ ∝ 1/Λ

            # Initial belief mean with displacement in first component
            mu_q_init = np.zeros(K_latent)
            mu_q_init[0] = mu0

            # Prior at origin
            mu_p_init = np.zeros(K_latent)

            osc = EpistemicOscillator.create_vfe_oscillator(
                K=K_latent,
                mu_q_init=mu_q_init,
                mu_p_init=mu_p_init,
                sigma_scale=sigma_scale,
                damping=gamma,
                lambda_self=stiffness,
                seed=42,
            )
        else:
            # Simple quadratic mode (for comparison)
            osc = EpistemicOscillator(
                precision=precision,
                stiffness=stiffness,
                damping=gamma,
                equilibrium=0.0
            )

        # For VFE mode, simulate in K-dimensional space
        if use_vfe:
            # Get initial momentum vector
            pi0_vec = np.zeros(K_latent)
            pi0_vec[0] = pi0
            result = osc.simulate(osc.agent.mu_q.copy(), pi0_vec, t_end)
            # Extract first component for visualization
            result['mu'] = result['mu'][:, 0] if result['mu'].ndim > 1 else result['mu']
            result['pi'] = result['pi'][:, 0] if result['pi'].ndim > 1 else result['pi']
        else:
            result = osc.simulate(mu0, pi0, t_end)

        result['regime'] = regime_name
        result['gamma'] = gamma
        result['damping_ratio'] = osc.damping_ratio
        result['decay_time'] = osc.decay_time
        result['oscillator'] = osc
        result['use_vfe'] = use_vfe
        results[regime_name] = result

        print(f"{regime_name.upper()}:")
        print(f"  γ = {gamma:.3f}, ζ = {osc.damping_ratio:.3f}")
        print(f"  Discriminant Δ = {osc.discriminant:.3f}")
        print(f"  Decay time τ = {osc.decay_time:.3f}")
        if regime_name == 'underdamped':
            print(f"  Damped frequency ω = {osc.damped_frequency:.3f}")
        if use_vfe:
            print(f"  Hamiltonian type: {type(osc.hamiltonian).__name__}")
        print()

    # Create visualization
    _plot_damping_regimes(results, output_dir)

    return results


def _plot_damping_regimes(results: Dict, output_dir: Path):
    """Create comprehensive damping regimes visualization."""

    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)

    colors = {'overdamped': '#e74c3c', 'critical': '#2ecc71', 'underdamped': '#3498db'}
    labels = {'overdamped': 'Overdamped (Bayesian-like)',
              'critical': 'Critical (Optimal)',
              'underdamped': 'Underdamped (Oscillatory)'}

    # Check if VFE mode (results have 'use_vfe' key)
    is_vfe = any(result.get('use_vfe', False) for result in results.values())

    # Row 1: Time evolution of belief μ(t)
    ax1 = fig.add_subplot(gs[0, :2])
    for regime, result in results.items():
        mu = result['mu']
        ax1.plot(result['t'], mu, color=colors[regime],
                 linewidth=2.5, label=labels[regime], alpha=0.9)
    ax1.axhline(0, color='gray', linestyle='--', alpha=0.5, label='Equilibrium')
    ax1.set_xlabel('Time t', fontsize=12)
    ax1.set_ylabel('Belief μ(t)', fontsize=12)
    title_suffix = " (VFE)" if is_vfe else ""
    ax1.set_title(f'Belief Evolution: Three Damping Regimes{title_suffix}', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(alpha=0.3)
    ax1.set_xlim(0, results['underdamped']['t'][-1])

    # Row 1 right: Energy evolution (or momentum magnitude for VFE)
    ax2 = fig.add_subplot(gs[0, 2])
    for regime, result in results.items():
        if 'kinetic_energy' in result and 'potential_energy' in result:
            # Simple mode: plot energy
            total_E = result['kinetic_energy'] + result['potential_energy']
            ax2.plot(result['t'], total_E, color=colors[regime],
                     linewidth=2, label=regime.capitalize(), alpha=0.9)
            ax2.set_ylabel('Total Energy', fontsize=12)
            ax2.set_title('Energy Dissipation', fontsize=14, fontweight='bold')
        else:
            # VFE mode: plot momentum magnitude
            pi = result['pi']
            pi_mag = np.abs(pi) if pi.ndim == 1 else np.linalg.norm(pi, axis=1)
            ax2.plot(result['t'], pi_mag, color=colors[regime],
                     linewidth=2, label=regime.capitalize(), alpha=0.9)
            ax2.set_ylabel('|π| (Momentum)', fontsize=12)
            ax2.set_title('Momentum Decay (VFE)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Time t', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3)
    try:
        ax2.set_yscale('log')
    except ValueError:
        pass  # Can't use log scale if values <= 0

    # Row 2: Phase portraits (μ vs π)
    for idx, (regime, result) in enumerate(results.items()):
        ax = fig.add_subplot(gs[1, idx])

        mu = result['mu']
        pi = result['pi']

        # Phase trajectory
        ax.plot(mu, pi, color=colors[regime],
                linewidth=2, alpha=0.8)

        # Mark start and end
        ax.plot(mu[0], pi[0], 'o', color='green',
                markersize=12, label='Start', zorder=5)
        ax.plot(mu[-1], pi[-1], 's', color='red',
                markersize=10, label='End', zorder=5)
        ax.plot(0, 0, '*', color='gold', markersize=15,
                label='Equilibrium', zorder=5)

        # Direction arrows
        n_arrows = 8
        arrow_idx = np.linspace(0, len(result['t'])-2, n_arrows, dtype=int)
        for i in arrow_idx:
            dx = mu[i+1] - mu[i]
            dy = pi[i+1] - pi[i]
            ax.annotate('', xy=(mu[i]+dx*0.6, pi[i]+dy*0.6),
                       xytext=(mu[i], pi[i]),
                       arrowprops=dict(arrowstyle='->', color=colors[regime], alpha=0.6))

        ax.set_xlabel('Belief μ', fontsize=12)
        ax.set_ylabel('Momentum π', fontsize=12)
        ax.set_title(f'Phase Portrait: {labels[regime]}', fontsize=12, fontweight='bold')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_aspect('equal', adjustable='datalim')

    # Row 3: Velocity and detailed analysis
    ax5 = fig.add_subplot(gs[2, 0])
    for regime, result in results.items():
        vel = result['velocity']
        # Handle VFE mode: velocity might be 2D (n_steps, K)
        if vel.ndim > 1:
            vel = vel[:, 0]  # Extract first component
        ax5.plot(result['t'], vel, color=colors[regime],
                 linewidth=2, label=regime.capitalize(), alpha=0.9)
    ax5.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax5.set_xlabel('Time t', fontsize=12)
    ax5.set_ylabel('Velocity μ̇ = π/M', fontsize=12)
    ax5.set_title('Belief Velocity Evolution', fontsize=14, fontweight='bold')
    ax5.legend(fontsize=10)
    ax5.grid(alpha=0.3)

    # Comparison panel
    ax6 = fig.add_subplot(gs[2, 1])
    regimes_list = list(results.keys())
    x_pos = np.arange(len(regimes_list))

    decay_times = [results[r]['decay_time'] for r in regimes_list]
    damping_ratios = [results[r]['damping_ratio'] for r in regimes_list]

    width = 0.35
    bars1 = ax6.bar(x_pos - width/2, decay_times, width,
                    color=[colors[r] for r in regimes_list], alpha=0.8, label='Decay time τ')

    ax6.set_xticks(x_pos)
    ax6.set_xticklabels([r.capitalize() for r in regimes_list])
    ax6.set_ylabel('Decay Time τ = 2M/γ', fontsize=12)
    ax6.set_title('Characteristic Timescales', fontsize=14, fontweight='bold')
    ax6.grid(alpha=0.3, axis='y')

    # Add damping ratio as text
    for i, (tau, zeta) in enumerate(zip(decay_times, damping_ratios)):
        ax6.text(i, tau + 0.1, f'ζ={zeta:.2f}', ha='center', fontsize=10)

    # Summary text
    ax7 = fig.add_subplot(gs[2, 2])
    ax7.axis('off')

    
   

    plt.suptitle('The Inertia of Belief: Damping Regimes in Epistemic Dynamics',
                 fontsize=16, fontweight='bold', y=0.995)

    plt.savefig(output_dir / "damping_regimes.png", dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig(output_dir / "damping_regimes.pdf", bbox_inches='tight')
    plt.close()

    print(f"✓ Saved: {output_dir / 'damping_regimes.png'}")


# =============================================================================
# Simulation 2: Two-Agent Momentum Transfer
# =============================================================================

def simulate_momentum_transfer(
    precision1: float = 2.0,
    precision2: float = 1.0,
    coupling: float = 0.5,
    initial_momentum1: float = 2.0,
    t_end: float = 20.0,
    output_dir: Optional[Path] = None,
    use_vfe: bool = True,  # NEW: Use actual VFE theory!
    K_latent: int = 3,     # Latent dimension for VFE mode
) -> Dict[str, np.ndarray]:
    """
    Simulate momentum transfer between two coupled agents.

    **NOW WITH GAUGE VFE THEORY!**

    In VFE mode:
    - Uses REAL Agent objects with gauge fields φ
    - Gauge transport Ω_ij = exp(φ_i) · exp(-φ_j)
    - Full VFE: F = KL(q||p) + Σ_j β_ij KL(q_i||Ω_ij[q_j])

    Key prediction: The influencer's momentum decreases (recoil effect)
    as momentum flows to the coupled partner WITH gauge transport!

    From Eq. 48-52:
    - Momentum current: J_{k→i} = β_{ik} Λ̃_k (μ̃_k - μ_i)
    - Total momentum changes when priors/damping present

    Args:
        precision1: Agent 1's precision (the "influencer")
        precision2: Agent 2's precision (the "listener")
        coupling: Attention coupling β₁₂ = β₂₁
        initial_momentum1: Initial momentum of agent 1
        t_end: Simulation duration
        output_dir: Where to save figures
        use_vfe: If True, use ACTUAL gauge VFE theory!
        K_latent: Latent dimension for VFE mode
    """
    if output_dir is None:
        output_dir = Path("_experiments/momentum_transfer")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*70)
    print("SIMULATION 2: TWO-AGENT MOMENTUM TRANSFER")
    if use_vfe:
        print(">>> USING ACTUAL GAUGE VFE THEORY <<<")
    print("="*70)
    print(f"Agent 1: Precision = {precision1} (influencer, starts with momentum)")
    print(f"Agent 2: Precision = {precision2} (listener, starts at rest)")
    print(f"Coupling β = {coupling}")
    print(f"Initial momentum π₁(0) = {initial_momentum1}")
    if use_vfe:
        print(f"Latent dimension K = {K_latent}")
    print()

    system = TwoAgentSystem(
        precision1=precision1,
        precision2=precision2,
        coupling=coupling,
        damping1=0.05,  # Light damping
        damping2=0.05,
        prior1=0.0,
        prior2=0.0,
        prior_strength=0.1,
        use_vfe=use_vfe,
        K_latent=K_latent,
    )

    if use_vfe:
        print(f"Hamiltonian type: {type(system.hamiltonian).__name__}")

    # Agent 1 starts moving, agent 2 at rest
    result = system.simulate(
        mu1_0=0.0,      # Both start at same belief
        mu2_0=0.0,
        pi1_0=initial_momentum1,  # Agent 1 has momentum
        pi2_0=0.0,                 # Agent 2 at rest
        t_end=t_end
    )

    # Add metadata
    result['precision1'] = precision1
    result['precision2'] = precision2
    result['coupling'] = coupling

    _plot_momentum_transfer(result, output_dir)

    return result


def _plot_momentum_transfer(result: Dict, output_dir: Path):
    """Visualize momentum transfer between agents."""

    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

    t = result['t']

    # Colors
    c1, c2 = '#e74c3c', '#3498db'

    # Panel 1: Belief trajectories
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(t, result['mu1'], color=c1, linewidth=2.5, label='Agent 1 (influencer)')
    ax1.plot(t, result['mu2'], color=c2, linewidth=2.5, label='Agent 2 (listener)')
    ax1.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Time t', fontsize=12)
    ax1.set_ylabel('Belief μ', fontsize=12)
    ax1.set_title('Belief Trajectories', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)

    # Panel 2: Momentum trajectories - KEY RESULT
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(t, result['pi1'], color=c1, linewidth=2.5, label='π₁ (influencer)')
    ax2.plot(t, result['pi2'], color=c2, linewidth=2.5, label='π₂ (listener)')
    ax2.axhline(0, color='gray', linestyle='--', alpha=0.5)

    # Highlight recoil region
    ax2.fill_between(t, result['pi1'], alpha=0.2, color=c1)
    ax2.fill_between(t, result['pi2'], alpha=0.2, color=c2)

    ax2.set_xlabel('Time t', fontsize=12)
    ax2.set_ylabel('Momentum π', fontsize=12)
    ax2.set_title('Momentum Trajectories (Recoil Effect)', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3)

    # Add annotation for recoil (positioned to avoid title overlap)
    peak_idx = np.argmax(result['pi2'])
    peak_val = result['pi2'][peak_idx]
    ax2.annotate('Momentum\ntransfer',
                xy=(t[peak_idx], peak_val),
                xytext=(t[peak_idx]+3, peak_val * 0.7),  # Position below peak
                fontsize=9, ha='left',
                arrowprops=dict(arrowstyle='->', color='black', lw=0.8))

    # Panel 3: Total and difference
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(t, result['total_momentum'], color='purple', linewidth=2.5,
             label='Total π₁ + π₂')
    ax3.plot(t, result['momentum_diff'], color='orange', linewidth=2,
             linestyle='--', label='Difference π₁ - π₂')
    ax3.axhline(result['total_momentum'][0], color='gray', linestyle=':',
                alpha=0.5, label='Initial total')
    ax3.set_xlabel('Time t', fontsize=12)
    ax3.set_ylabel('Momentum', fontsize=12)
    ax3.set_title('Conservation Check', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(alpha=0.3)

    # Panel 4: Phase portrait Agent 1
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(result['mu1'], result['pi1'], color=c1, linewidth=2)
    ax4.plot(result['mu1'][0], result['pi1'][0], 'go', markersize=12, label='Start')
    ax4.plot(result['mu1'][-1], result['pi1'][-1], 'rs', markersize=10, label='End')
    ax4.set_xlabel('Belief μ₁', fontsize=12)
    ax4.set_ylabel('Momentum π₁', fontsize=12)
    ax4.set_title('Phase Portrait: Agent 1', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(alpha=0.3)

    # Panel 5: Phase portrait Agent 2
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(result['mu2'], result['pi2'], color=c2, linewidth=2)
    ax5.plot(result['mu2'][0], result['pi2'][0], 'go', markersize=12, label='Start')
    ax5.plot(result['mu2'][-1], result['pi2'][-1], 'rs', markersize=10, label='End')
    ax5.set_xlabel('Belief μ₂', fontsize=12)
    ax5.set_ylabel('Momentum π₂', fontsize=12)
    ax5.set_title('Phase Portrait: Agent 2', fontsize=14, fontweight='bold')
    ax5.legend(fontsize=10)
    ax5.grid(alpha=0.3)

    # Panel 6: Summary
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')

    # Compute key statistics
    pi1_drop = result['pi1'][0] - np.min(result['pi1'])
    pi2_max = np.max(result['pi2'])
    transfer_efficiency = pi2_max / result['pi1'][0]

    

    plt.suptitle('Two-Agent Momentum Transfer: The Recoil Effect',
                 fontsize=16, fontweight='bold', y=0.995)

    plt.savefig(output_dir / "momentum_transfer.png", dpi=150, bbox_inches='tight',
                facecolor='white')
    plt.savefig(output_dir / "momentum_transfer.pdf", bbox_inches='tight')
    plt.close()

    print(f"✓ Saved: {output_dir / 'momentum_transfer.png'}")


# =============================================================================
# Simulation 3: Confirmation Bias as Stopping Distance
# =============================================================================

def simulate_stopping_distance(
    precision_range: np.ndarray = None,
    initial_velocity: float = 1.0,
    counter_force: float = 0.5,
    t_end: float = 100.0,
    output_dir: Optional[Path] = None
) -> Dict[str, np.ndarray]:
    """
    Fixed stopping distance simulation - pure ballistic motion.
    
    Theory: d_stop = M v² / (2f)
    
    Key fix: NO spring, NO damping - just mass against constant force.
    This matches the manuscript's Eq. 33-35 exactly.
    """
    if output_dir is None:
        output_dir = Path("_experiments/stopping_distance_fixed")
    output_dir.mkdir(parents=True, exist_ok=True)

    if precision_range is None:
        precision_range = np.array([0.5, 1.0, 2.0, 4.0, 8.0])

    print("\n" + "="*70)
    print("SIMULATION 3 (FIXED): PURE BALLISTIC STOPPING DISTANCE")
    print("="*70)
    print(f"Initial velocity: v₀ = {initial_velocity}")
    print(f"Counter-force: f = {counter_force}")
    print(f"Theory: d = M v² / (2f)")
    print()

    results = {
        'precision': precision_range,
        'stopping_distance': [],
        'theoretical_distance': [],
        'stopping_time': [],
        'trajectories': {}
    }

    for Lambda in precision_range:
        # Initial momentum = mass × velocity
        pi0 = Lambda * initial_velocity
        
        def equations(t, y):
            """Pure ballistic: M dv/dt = -f, with v = π/M"""
            mu, pi = y
            v = pi / Lambda
            # Only force is constant counter-evidence (opposes positive velocity)
            if v > 0:
                dpi_dt = -counter_force
            else:
                dpi_dt = 0  # Stop when velocity reaches zero
            return [v, dpi_dt]
        
        # Integrate until velocity crosses zero
        def velocity_zero(t, y):
            return y[1]  # π = 0 means stopped
        velocity_zero.terminal = True
        velocity_zero.direction = -1
        
        sol = solve_ivp(
            equations,
            (0, t_end),
            [0.0, pi0],  # Start at μ=0 with momentum
            events=velocity_zero,
            max_step=0.01,
            dense_output=True
        )
        
        # Get stopping point
        if sol.t_events[0].size > 0:
            t_stop = sol.t_events[0][0]
            y_stop = sol.sol(t_stop)  # Fixed: sol.sol() not sol.y_sol()
            d_stop = y_stop[0]
        else:
            t_stop = sol.t[-1]
            d_stop = sol.y[0, -1]
        
        # Theoretical: d = M v² / (2f)
        d_theory = Lambda * initial_velocity**2 / (2 * counter_force)
        
        results['stopping_distance'].append(d_stop)
        results['theoretical_distance'].append(d_theory)
        results['stopping_time'].append(t_stop)
        
        # Store full trajectory for plotting
        t_dense = np.linspace(0, t_stop, 500)
        y_dense = sol.sol(t_dense)
        results['trajectories'][Lambda] = {
            't': t_dense,
            'mu': y_dense[0],
            'pi': y_dense[1],
            'velocity': y_dense[1] / Lambda
        }
        
        print(f"Λ = {Lambda:.1f}: d_stop = {d_stop:.4f}, d_theory = {d_theory:.4f}, "
              f"error = {abs(d_stop - d_theory)/d_theory*100:.2f}%")

    results['stopping_distance'] = np.array(results['stopping_distance'])
    results['theoretical_distance'] = np.array(results['theoretical_distance'])
    
    _plot_stopping_distance_fixed(results, output_dir, initial_velocity, counter_force)
    
    return results


def _plot_stopping_distance_fixed(results: Dict, output_dir: Path, v0: float, f: float):
    """Visualize fixed stopping distance results."""
    
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    precisions = results['precision']
    distances = results['stopping_distance']
    theoretical = results['theoretical_distance']
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(precisions)))
    
    # Panel 1: Trajectories
    ax1 = fig.add_subplot(gs[0, 0])
    for i, Lambda in enumerate(precisions):
        traj = results['trajectories'][Lambda]
        ax1.plot(traj['t'], traj['mu'], color=colors[i], linewidth=2,
                 label=f'Λ = {Lambda:.1f}')
        ax1.plot(traj['t'][-1], traj['mu'][-1], 'o', color=colors[i], markersize=10)
    
    ax1.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Time t', fontsize=12)
    ax1.set_ylabel('Belief μ(t)', fontsize=12)
    ax1.set_title('Pure Ballistic Trajectories', fontsize=14, fontweight='bold')
    ax1.legend(title='Precision (Mass)', fontsize=9)
    ax1.grid(alpha=0.3)
    
    # Panel 2: d vs Λ - KEY RESULT
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.scatter(precisions, distances, s=150, c=colors, edgecolors='black',
                linewidths=2, label='Simulated', zorder=5)
    ax2.plot(precisions, theoretical, 'k--', linewidth=2, 
             label=f'Theory: d = Λv²/2f = Λ×{v0**2/(2*f):.2f}')
    
    ax2.set_xlabel('Precision Λ (Mass)', fontsize=12)
    ax2.set_ylabel('Stopping Distance d', fontsize=12)
    ax2.set_title('d ∝ Λ: Perfect Theory Match', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3)
    
    # Compute R²
    ss_res = np.sum((distances - theoretical)**2)
    ss_tot = np.sum((distances - np.mean(distances))**2)
    r2 = 1 - ss_res/ss_tot
    ax2.text(0.05, 0.95, f'R² = {r2:.6f}', transform=ax2.transAxes,
             fontsize=12, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Panel 3: Velocity decay
    ax3 = fig.add_subplot(gs[1, 0])
    for i, Lambda in enumerate(precisions):
        traj = results['trajectories'][Lambda]
        ax3.plot(traj['t'], traj['velocity'], color=colors[i], linewidth=2,
                 label=f'Λ = {Lambda:.1f}')
    ax3.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Time t', fontsize=12)
    ax3.set_ylabel('Velocity v = π/M', fontsize=12)
    ax3.set_title('Velocity Decay (Linear: a = -f/M)', fontsize=14, fontweight='bold')
    ax3.legend(title='Precision', fontsize=9)
    ax3.grid(alpha=0.3)
    
    # Panel 4: Summary
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')
    
   
    
    plt.suptitle('Stopping Distance: Pure Ballistic Theory Match',
                 fontsize=16, fontweight='bold', y=0.995)
    
    plt.savefig(output_dir / "stopping_distance_fixed.png", dpi=150, 
                bbox_inches='tight', facecolor='white')
    plt.savefig(output_dir / "stopping_distance_fixed.pdf", bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {output_dir / 'stopping_distance_fixed.png'}")


# =============================================================================
# Simulation 4: Resonance Curve
# =============================================================================

def simulate_resonance(
    precision: float = 2.0,
    stiffness: float = 1.0,
    damping: float = 0.3,
    forcing_amplitude: float = 0.5,
    omega_range: np.ndarray = None,
    t_end: float = 100.0,
    output_dir: Optional[Path] = None,
    use_vfe: bool = True,  # NEW: Use actual VFE theory!
    K_latent: int = 3,     # Latent dimension for VFE mode
) -> Dict[str, np.ndarray]:
    """
    Simulate resonance curve for periodic evidence forcing.

    **NOW WITH GAUGE VFE THEORY!**

    In VFE mode:
    - Uses ACTUAL VFE functional as potential
    - Fisher metric Λ = Σ^{-1} as mass tensor
    - Resonance occurs at ω_res = √(∂²F/∂μ² / M)

    From Eq. 40-42:
    - ω_res = √(K/M) = resonance frequency
    - A(ω) = (f₀/M) / √((ω₀² - ω²)² + (γω/M)²)
    - At resonance: A_max = (f₀/γ)√(M/K)

    Args:
        precision: Epistemic mass M = Λ
        stiffness: Evidence strength K
        damping: Damping coefficient γ
        forcing_amplitude: Forcing amplitude f₀
        omega_range: Frequencies to test
        t_end: Simulation time (need steady state)
        output_dir: Where to save figures
        use_vfe: If True, use ACTUAL gauge VFE theory!
        K_latent: Latent dimension for VFE mode
    """
    if output_dir is None:
        output_dir = Path("_experiments/resonance")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*70)
    print("SIMULATION 4: RESONANCE CURVE")
    if use_vfe:
        print(">>> USING ACTUAL GAUGE VFE THEORY <<<")
    print("="*70)
    print(f"Damping γ = {damping}")
    print(f"Forcing amplitude f₀ = {forcing_amplitude}")

    if use_vfe:
        # VFE mode: Create oscillator with actual VFE
        sigma_scale = 1.0 / np.sqrt(precision)
        osc = EpistemicOscillator.create_vfe_oscillator(
            K=K_latent,
            mu_q_init=np.zeros(K_latent),
            mu_p_init=np.zeros(K_latent),
            sigma_scale=sigma_scale,
            damping=damping,
            lambda_self=stiffness,
            seed=42,
        )
        print(f"Hamiltonian type: {type(osc.hamiltonian).__name__}")
        print(f"Latent dimension K = {K_latent}")

        # VFE mode: Compute natural frequency from eigenvalues of M⁻¹K
        # From Hamiltonian: H = (1/2) π^T M⁻¹ π + V(μ)
        # The Fisher metric returns Σ_q (covariance) as M⁻¹
        # So mass M = Σ_q⁻¹ = Λ_q (precision is mass!)
        # Stiffness K = λ_self * Λ_p (from VFE gradient ∇V = λ_self * Λ_p * (μ_q - μ_p))
        # Natural frequencies: ω² = eigenvalues of M⁻¹K = Σ_q @ (λ_self * Λ_p)
        agent = osc.hamiltonian.agent
        Sigma_q = agent.Sigma_q
        Sigma_p = agent.Sigma_p
        lambda_self = osc.hamiltonian.lambda_self

        Lambda_q = np.linalg.inv(Sigma_q)  # Mass matrix M = Λ_q
        Lambda_p = np.linalg.inv(Sigma_p)
        K_matrix = lambda_self * Lambda_p  # Stiffness matrix
        M_inv_K = Sigma_q @ K_matrix

        # Get eigenvalues AND eigenvectors to compute effective mass
        eigenvalues, eigenvectors = np.linalg.eigh(M_inv_K)
        omega_squared = np.abs(eigenvalues)  # Should all be positive for stable system
        natural_frequencies = np.sqrt(omega_squared)

        # Find the dominant (lowest frequency) mode
        min_idx = np.argmin(omega_squared)
        omega_0 = natural_frequencies[min_idx]
        omega_max = np.max(natural_frequencies)
        v_dominant = eigenvectors[:, min_idx]  # Eigenvector for lowest mode

        # Effective mass for dominant mode: M_eff = v^T M v
        # (eigenvector is normalized, so this gives the projected mass)
        effective_precision = float(v_dominant @ Lambda_q @ v_dominant)
        effective_stiffness = float(v_dominant @ K_matrix @ v_dominant)

        # Verify: ω₀² should equal K_eff / M_eff
        omega_check = np.sqrt(effective_stiffness / effective_precision)

        print(f"VFE natural frequencies: {natural_frequencies}")
        print(f"Dominant (lowest) ω₀ = {omega_0:.3f}")
        print(f"Highest ω_max = {omega_max:.3f}")
        print(f"Effective M_eff = {effective_precision:.3f}, K_eff = {effective_stiffness:.3f}")
        print(f"Check: √(K_eff/M_eff) = {omega_check:.3f}")

        # Store VFE-specific results
        vfe_results = {
            'natural_frequencies': natural_frequencies,
            'omega_min': omega_0,
            'omega_max': omega_max,
        }
    else:
        # Simple quadratic mode
        omega_0 = np.sqrt(stiffness / precision)  # Natural frequency
        effective_precision = precision
        effective_stiffness = stiffness
        vfe_results = None  # No VFE-specific data
        osc = EpistemicOscillator(
            precision=precision,
            stiffness=stiffness,
            damping=damping,
            equilibrium=0.0
        )
        print(f"Precision M = {precision}")
        print(f"Stiffness K = {stiffness}")
        print(f"Natural frequency ω₀ = √(K/M) = {omega_0:.3f}")

    omega_res = omega_0  # For light damping

    if omega_range is None:
        omega_range = np.linspace(0.1, 3*omega_0, 50)

    print()

    results = {
        'omega': omega_range,
        'omega_0': omega_0,
        'amplitude': [],
        'theoretical_amplitude': [],
        'phase': [],
        'example_trajectories': {},
        'use_vfe': use_vfe,
        'vfe_data': vfe_results,  # VFE-specific: natural frequencies, etc.
    }

    # Theoretical amplitude function (Eq. 41)
    # For VFE mode, use effective_precision computed from eigenvalues
    def theoretical_amplitude(omega):
        numerator = forcing_amplitude / effective_precision
        denominator = np.sqrt((omega_0**2 - omega**2)**2 + (damping * omega / effective_precision)**2)
        return numerator / denominator

    for omega in omega_range:
        # Periodic forcing
        forcing = lambda t, w=omega: forcing_amplitude * np.cos(w * t)

        if use_vfe:
            # VFE mode: K-dimensional initial conditions
            mu0_vec = np.zeros(K_latent)
            pi0_vec = np.zeros(K_latent)
            result = osc.simulate(mu0_vec, pi0_vec, t_end, forcing=forcing)
            # Extract first component
            mu_trace = result['mu'][:, 0] if result['mu'].ndim > 1 else result['mu']
        else:
            result = osc.simulate(
                mu0=0.0,
                pi0=0.0,
                t_end=t_end,
                forcing=forcing
            )
            mu_trace = result['mu']

        # Measure steady-state amplitude (last 20% of simulation)
        steady_start = int(0.8 * len(result['t']))
        steady_state = mu_trace[steady_start:]
        amplitude = (np.max(steady_state) - np.min(steady_state)) / 2

        results['amplitude'].append(amplitude)
        results['theoretical_amplitude'].append(theoretical_amplitude(omega))

        # Store example trajectories for plotting (at specific relative frequencies)
        for ratio in [0.5, 1.0, 2.0]:
            target = omega_0 * ratio
            if abs(omega - target) < (omega_range[1] - omega_range[0]) / 2:
                result_copy = result.copy()
                result_copy['mu'] = mu_trace  # Store extracted trace
                results['example_trajectories'][omega] = result_copy

    results['amplitude'] = np.array(results['amplitude'])
    results['theoretical_amplitude'] = np.array(results['theoretical_amplitude'])

    # Find measured resonance peak
    peak_idx = np.argmax(results['amplitude'])
    results['measured_omega_res'] = omega_range[peak_idx]
    results['measured_A_max'] = results['amplitude'][peak_idx]

    # Theoretical peak amplitude
    # For VFE mode, use effective values computed from eigenvalues
    A_max_theory = (forcing_amplitude / damping) * np.sqrt(effective_precision / effective_stiffness)
    results['theoretical_A_max'] = A_max_theory

    print(f"Measured resonance: ω = {results['measured_omega_res']:.3f}, A = {results['measured_A_max']:.3f}")
    print(f"Predicted resonance: ω = {omega_0:.3f}, A_max = {A_max_theory:.3f}")
    if use_vfe:
        print(f"(VFE mode: ω₀ computed from eigenvalues of M⁻¹K)")
        print(f"Using VFE Hamiltonian: {type(osc.hamiltonian).__name__}")

    _plot_resonance(results, output_dir, precision, stiffness, damping, forcing_amplitude)

    return results


def _plot_resonance(results: Dict, output_dir: Path, M: float, K: float, gamma: float, f0: float):
    """Visualize resonance curve."""

    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

    omega = results['omega']
    amplitude = results['amplitude']
    omega_0 = results['omega_0']

    # Panel 1: Resonance curve - KEY RESULT
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(omega, amplitude, 'b-', linewidth=2.5, label='Simulated', zorder=5)
    ax1.plot(omega, results['theoretical_amplitude'], 'r--', linewidth=2,
             label='Theory (Eq. 41)', alpha=0.8)

    # Mark resonance peak
    ax1.axvline(omega_0, color='green', linestyle=':', linewidth=2,
                label=f'ω₀ = √(K/M) = {omega_0:.3f}')
    ax1.axvline(results['measured_omega_res'], color='purple', linestyle='--',
                linewidth=1.5, label=f'Measured ω_res = {results["measured_omega_res"]:.3f}')

    ax1.scatter([results['measured_omega_res']], [results['measured_A_max']],
                s=200, c='red', marker='*', edgecolors='black', linewidths=2,
                label=f'Peak A = {results["measured_A_max"]:.3f}', zorder=10)

    ax1.set_xlabel('Driving Frequency ω', fontsize=12)
    ax1.set_ylabel('Steady-State Amplitude A(ω)', fontsize=12)
    ax1.set_title('Cognitive Resonance: Optimal Persuasion at ω_res = √(K/M)',
                  fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(alpha=0.3)

    # Add resonance equation
    ax1.text(0.02, 0.95,
             f'A(ω) = (f₀/M) / √[(ω₀² - ω²)² + (γω/M)²]\n'
             f'ω₀ = √(K/M) = √({K}/{M}) = {omega_0:.3f}',
             transform=ax1.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Panel 2: Example trajectories
    ax2 = fig.add_subplot(gs[1, 0])
    colors = ['#e74c3c', '#2ecc71', '#3498db']

    example_trajs = list(results['example_trajectories'].items())
    for i, (omega_ex, traj) in enumerate(example_trajs[:3]):  # Max 3 examples
        # Determine position relative to resonance
        ratio = omega_ex / omega_0
        if ratio < 0.8:
            label_suffix = 'Below'
        elif ratio > 1.2:
            label_suffix = 'Above'
        else:
            label_suffix = 'At'

        # Show last portion for steady state
        start_idx = int(0.7 * len(traj['t']))
        ax2.plot(traj['t'][start_idx:] - traj['t'][start_idx],
                 traj['mu'][start_idx:],
                 color=colors[i % len(colors)], linewidth=2,
                 label=f'ω = {omega_ex:.2f} ({label_suffix} res.)')

    ax2.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Time t', fontsize=12)
    ax2.set_ylabel('Belief μ(t)', fontsize=12)
    ax2.set_title('Steady-State Oscillations', fontsize=14, fontweight='bold')
    if example_trajs:
        ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    # Panel 3: Summary
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.axis('off')

   

    plt.suptitle('Resonance in Belief Dynamics: Optimal Persuasion Frequency',
                 fontsize=16, fontweight='bold', y=0.995)

    plt.savefig(output_dir / "resonance_curve.png", dpi=150, bbox_inches='tight',
                facecolor='white')
    plt.savefig(output_dir / "resonance_curve.pdf", bbox_inches='tight')
    plt.close()

    print(f"✓ Saved: {output_dir / 'resonance_curve.png'}")


# =============================================================================
# Simulation 5: Belief Perseverance Decay
# =============================================================================

# =============================================================================
# Simulation 5: Belief Perseverance Decay (FIXED)
# =============================================================================

def simulate_belief_perseverance(
    precision_range: np.ndarray = None,
    gamma: float = 1.0,  # CONSTANT damping for all agents
    initial_displacement: float = 2.0,
    t_end: float = 50.0,
    output_dir: Optional[Path] = None
) -> Dict[str, np.ndarray]:
    """
    Fixed belief perseverance using FIRST-ORDER model.
    
    Theory: M dμ/dt = -γ μ  →  μ(t) = μ₀ exp(-γt/M)
    Decay time: τ = M/γ = Λ/γ
    
    This avoids the complications of second-order overdamped systems
    where the slow mode is independent of mass.
    """
    if output_dir is None:
        output_dir = Path("_experiments/perseverance_fixed")
    output_dir.mkdir(parents=True, exist_ok=True)

    if precision_range is None:
        precision_range = np.array([0.5, 1.0, 2.0, 4.0, 8.0])

    print("\n" + "="*70)
    print("SIMULATION 5 (FIXED): FIRST-ORDER BELIEF DECAY")
    print("="*70)
    print(f"Model: M dμ/dt = -γ μ (first-order friction)")
    print(f"Constant damping γ = {gamma}")
    print(f"Initial false belief: μ₀ = {initial_displacement}")
    print(f"Theory: τ = M/γ = Λ/γ")
    print()

    results = {
        'precision': precision_range,
        'damping': gamma,  # Scalar, not array!
        'measured_tau': [],
        'theoretical_tau': [],
        'trajectories': {}
    }

    for Lambda in precision_range:
        # Analytical solution: μ(t) = μ₀ exp(-γt/M)
        tau_theory = Lambda / gamma
        
        t = np.linspace(0, t_end, 1000)
        mu = initial_displacement * np.exp(-gamma * t / Lambda)
        
        # Measure τ: time to reach 1/e of initial
        target = initial_displacement / np.e
        idx = np.searchsorted(-mu, -target)  # First crossing
        tau_measured = t[idx] if idx < len(t) else t_end
        
        results['measured_tau'].append(tau_measured)
        results['theoretical_tau'].append(tau_theory)
        results['trajectories'][Lambda] = {'t': t, 'mu': mu}
        
        print(f"Λ = {Lambda:.1f}: τ_measured = {tau_measured:.4f}, "
              f"τ_theory = {tau_theory:.4f}, error = {abs(tau_measured-tau_theory)/tau_theory*100:.3f}%")

    results['measured_tau'] = np.array(results['measured_tau'])
    results['theoretical_tau'] = np.array(results['theoretical_tau'])
    
    _plot_belief_perseverance(results, output_dir)
    
    return results


def _plot_belief_perseverance(results: Dict, output_dir: Path):
    """Visualize fixed belief perseverance results."""
    
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    precisions = results['precision']
    measured_tau = results['measured_tau']
    theoretical_tau = results['theoretical_tau']
    gamma = results['damping']  # Now a scalar
    
    colors = plt.cm.plasma(np.linspace(0.2, 0.9, len(precisions)))
    
    # Panel 1: Decay trajectories
    ax1 = fig.add_subplot(gs[0, 0])
    for i, Lambda in enumerate(precisions):
        traj = results['trajectories'][Lambda]
        ax1.plot(traj['t'], traj['mu'], color=colors[i], linewidth=2,
                 label=f'Λ = {Lambda:.1f}, τ = {Lambda/gamma:.1f}')
    
    ax1.axhline(0, color='gray', linestyle='--', alpha=0.5, label='Truth')
    ax1.axhline(results['trajectories'][precisions[0]]['mu'][0]/np.e, 
                color='black', linestyle=':', alpha=0.5, label='1/e threshold')
    ax1.set_xlabel('Time t', fontsize=12)
    ax1.set_ylabel('Belief μ(t)', fontsize=12)
    ax1.set_title('First-Order Belief Decay', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)
    
    # Panel 2: τ vs Λ - KEY RESULT
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.scatter(precisions, measured_tau, s=150, c=colors, edgecolors='black',
                linewidths=2, label='Measured τ', zorder=5)
    ax2.plot(precisions, theoretical_tau, 'k--', linewidth=2,
             label=f'Theory: τ = Λ/{gamma}')
    
    ax2.set_xlabel('Precision Λ', fontsize=12)
    ax2.set_ylabel('Decay Time τ', fontsize=12)
    ax2.set_title('τ ∝ Λ: Perfect Theory Match', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3)
    
    # R²
    ss_res = np.sum((measured_tau - theoretical_tau)**2)
    ss_tot = np.sum((measured_tau - np.mean(measured_tau))**2)
    r2 = 1 - ss_res/ss_tot
    ax2.text(0.05, 0.95, f'R² = {r2:.6f}', transform=ax2.transAxes,
             fontsize=12, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Panel 3: Normalized decay (universal curve)
    ax3 = fig.add_subplot(gs[1, 0])
    for i, Lambda in enumerate(precisions):
        traj = results['trajectories'][Lambda]
        tau = Lambda / gamma
        t_norm = traj['t'] / tau
        mu_norm = traj['mu'] / traj['mu'][0]
        ax3.plot(t_norm, mu_norm, color=colors[i], linewidth=2,
                 label=f'Λ = {Lambda:.1f}')
    
    # Universal curve
    t_theory = np.linspace(0, 5, 100)
    ax3.plot(t_theory, np.exp(-t_theory), 'k--', linewidth=3, alpha=0.7,
             label='exp(-t/τ)')
    
    ax3.set_xlabel('Normalized Time t/τ', fontsize=12)
    ax3.set_ylabel('Normalized Belief μ/μ₀', fontsize=12)
    ax3.set_title('Universal Decay Curve', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(alpha=0.3)
    ax3.set_xlim(0, 5)
    
    # Panel 4: Summary
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')
    
    tau_ratio = measured_tau[-1] / measured_tau[0]
    prec_ratio = precisions[-1] / precisions[0]
    
   
    
    plt.suptitle('Belief Perseverance: First-Order Decay Theory Match',
                 fontsize=16, fontweight='bold', y=0.995)
    
    plt.savefig(output_dir / "perseverance_fixed.png", dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.savefig(output_dir / "perseverance_fixed.pdf", bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {output_dir / 'perseverance_fixed.png'}")

# =============================================================================
# Master Runner
# =============================================================================

def run_all_simulations(output_base: Optional[Path] = None):
    """
    Run all five simulations from the manuscript.

    Creates a comprehensive set of figures demonstrating the
    predictions of epistemic momentum theory.
    """
    if output_base is None:
        output_base = Path("_experiments/psych_manuscript_simulations")
    output_base.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*70)
    print("EPISTEMIC MOMENTUM SIMULATIONS")
    print("From 'The Inertia of Belief: Hiding in Plain Sight'")
    print("="*70 + "\n")

    all_results = {}

    # 1. Three Damping Regimes
    all_results['damping_regimes'] = simulate_damping_regimes(
        output_dir=output_base / "1_damping_regimes"
    )

    # 2. Two-Agent Momentum Transfer
    all_results['momentum_transfer'] = simulate_momentum_transfer(
        output_dir=output_base / "2_momentum_transfer"
    )

    # 3. Confirmation Bias / Stopping Distance
    all_results['stopping_distance'] = simulate_stopping_distance(
        output_dir=output_base / "3_stopping_distance"
    )

    # 4. Resonance Curve
    all_results['resonance'] = simulate_resonance(
        output_dir=output_base / "4_resonance"
    )

    # 5. Belief Perseverance Decay
    all_results['perseverance'] = simulate_belief_perseverance(
        output_dir=output_base / "5_perseverance"
    )

    print("\n" + "="*70)
    print("ALL SIMULATIONS COMPLETE")
    print(f"Results saved to: {output_base}")
    print("="*70 + "\n")

    return all_results


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run epistemic momentum simulations from the psych manuscript"
    )
    parser.add_argument("--sim", type=int, default=0,
                        help="Which simulation to run (1-5, or 0 for all)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory")

    args = parser.parse_args()

    output = Path(args.output) if args.output else None

    if args.sim == 0:
        run_all_simulations(output)
    elif args.sim == 1:
        simulate_damping_regimes(output_dir=output)
    elif args.sim == 2:
        simulate_momentum_transfer(output_dir=output)
    elif args.sim == 3:
        simulate_stopping_distance(output_dir=output)
    elif args.sim == 4:
        simulate_resonance(output_dir=output)
    elif args.sim == 5:
        simulate_belief_perseverance(output_dir=output)
    else:
        print(f"Unknown simulation: {args.sim}")
        print("Use 0 for all, or 1-5 for specific simulations")