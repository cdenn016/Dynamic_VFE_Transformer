"""
Hamiltonian Dynamics for Belief Evolution
==========================================

Building on the insight that the Fisher metric (∇²KL) provides natural kinetic
terms, this module implements full Hamiltonian dynamics for belief evolution.

Mathematical Foundation
-----------------------

From KL divergence expansion:
    KL(q||p) ≈ KL₀ + g_i δθ^i + (1/2) G_ij δθ^i δθ^j

where:
    - g_i = ∂KL/∂θ^i (gradient → "force")
    - G_ij = ∂²KL/∂θ^i∂θ^j = Fisher metric (→ "mass tensor")

This naturally suggests Hamiltonian:
    H(θ, p) = (1/2) p^T G^{-1} p + V(θ)

where:
    - θ: Belief parameters (μ, log σ, φ, ...)
    - p: Conjugate momenta (p = G θ̇)
    - G: Fisher metric tensor (mass/metric)
    - V: Free energy potential

Hamilton's Equations:
    dθ/dt = ∂H/∂p = G^{-1} p
    dp/dt = -∂H/∂θ = -∇V - (1/2) p^T ∂_θ(G^{-1}) p

Regimes:
    - Overdamped (current): θ̇ = -∇V (first-order, dissipative)
    - Underdamped (new): Second-order dynamics with momentum
    - Critical damping: Transition between regimes

Author: Chris
Date: November 2025
"""

import numpy as np
from typing import Tuple, Dict, Optional, Callable
from dataclasses import dataclass
# import torch  # Not currently used

from geometry.signature_analysis import analyze_metric_signature, MetricSignature


@dataclass
class HamiltonianState:
    """
    Complete phase space state for Hamiltonian dynamics.

    Attributes:
        q: Generalized coordinates (belief parameters θ)
        p: Conjugate momenta
        t: Time
        energy: Total energy H(q, p)
    """
    q: np.ndarray  # Position (belief parameters)
    p: np.ndarray  # Momentum
    t: float = 0.0
    energy: Optional[float] = None

    def __repr__(self):
        return f"HamiltonianState(t={self.t:.3f}, E={self.energy:.6f if self.energy else 'N/A'})"


class BeliefHamiltonian:
    """
    Hamiltonian for belief dynamics on statistical manifold.

    The Hamiltonian is:
        H(θ, p) = T(p) + V(θ)

    where:
        T(p) = (1/2) p^T G^{-1}(θ) p  (kinetic energy, Fisher metric)
        V(θ) = Free energy F(θ)        (potential energy)

    Hamilton's equations:
        dθ/dt = ∂H/∂p = G^{-1} p
        dp/dt = -∂H/∂θ = -∇V - (kinetic gradient correction)
    """

    def __init__(
        self,
        potential: Callable,  # V(θ) → float
        metric: Callable,     # G(θ) → (d, d) Fisher metric
        metric_gradient: Optional[Callable] = None,  # ∂G/∂θ (for kinetic correction)
        potential_gradient: Optional[Callable] = None,  # ∇V(θ) (for stable integration)
    ):
        """
        Initialize Hamiltonian.

        Args:
            potential: V(θ) returns potential energy (free energy)
            metric: G(θ) returns Fisher metric tensor
            metric_gradient: ∂G/∂θ for kinetic term correction (optional)
            potential_gradient: ∇V(θ) analytical gradient (optional, RECOMMENDED!)
                               If not provided, uses numerical finite differences.
                               Analytical gradients are much more stable!
        """
        self.potential = potential
        self.metric = metric
        self.metric_gradient = metric_gradient
        self.potential_gradient = potential_gradient

    def kinetic_energy(self, q: np.ndarray, p: np.ndarray) -> float:
        """
        Kinetic energy T = (1/2) p^T G^{-1} p.

        Args:
            q: Position (belief parameters)
            p: Momentum

        Returns:
            T: Kinetic energy
        """
        G = self.metric(q)
        try:
            G_inv = np.linalg.inv(G)
        except np.linalg.LinAlgError:
            # Regularize singular metric
            G_inv = np.linalg.inv(G + 1e-6 * np.eye(len(G)))

        return 0.5 * p @ G_inv @ p

    def potential_energy(self, q: np.ndarray) -> float:
        """
        Potential energy V = Free energy F(θ).

        Args:
            q: Position (belief parameters)

        Returns:
            V: Potential energy
        """
        return self.potential(q)

    def total_energy(self, q: np.ndarray, p: np.ndarray) -> float:
        """
        Total Hamiltonian H = T + V.

        Args:
            q: Position
            p: Momentum

        Returns:
            H: Total energy (conserved quantity)
        """
        return self.kinetic_energy(q, p) + self.potential_energy(q)

    def equations_of_motion(
        self,
        q: np.ndarray,
        p: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Hamilton's equations of motion.

        Returns:
            dq_dt: dθ/dt = ∂H/∂p = G^{-1} p
            dp_dt: dp/dt = -∂H/∂θ = -∇V - (kinetic correction)
        """
        # Get metric
        G = self.metric(q)
        try:
            G_inv = np.linalg.inv(G)
        except np.linalg.LinAlgError:
            G_inv = np.linalg.inv(G + 1e-6 * np.eye(len(G)))

        # dq/dt = ∂H/∂p = G^{-1} p
        dq_dt = G_inv @ p

        # dp/dt = -∂H/∂θ
        # Need: -∇V - (1/2) p^T ∂(G^{-1})/∂θ p

        # Potential gradient (force)
        if self.potential_gradient is not None:
            # Use analytical gradient (MUCH more stable!)
            grad_V = self.potential_gradient(q)
        else:
            # Fall back to numerical gradient (can be unstable)
            eps = 1e-5
            grad_V = np.zeros_like(q)
            for i in range(len(q)):
                q_plus = q.copy()
                q_plus[i] += eps
                q_minus = q.copy()
                q_minus[i] -= eps
                grad_V[i] = (self.potential(q_plus) - self.potential(q_minus)) / (2 * eps)

        # Kinetic gradient correction (from metric dependence on q)
        kinetic_correction = np.zeros_like(q)
        if self.metric_gradient is not None:
            # Use provided metric gradient
            dG_dq = self.metric_gradient(q)
            for i in range(len(q)):
                dG_inv_dq_i = -G_inv @ dG_dq[i] @ G_inv  # Matrix derivative chain rule
                kinetic_correction[i] = 0.5 * p @ dG_inv_dq_i @ p
        else:
            # Finite difference approximation for metric gradient
            eps_metric = 1e-5
            for i in range(len(q)):
                q_plus = q.copy()
                q_plus[i] += eps_metric
                q_minus = q.copy()
                q_minus[i] -= eps_metric

                G_plus = self.metric(q_plus)
                G_minus = self.metric(q_minus)

                try:
                    G_inv_plus = np.linalg.inv(G_plus + 1e-6 * np.eye(len(G_plus)))
                    G_inv_minus = np.linalg.inv(G_minus + 1e-6 * np.eye(len(G_minus)))
                except np.linalg.LinAlgError:
                    continue

                dG_inv_dq_i = (G_inv_plus - G_inv_minus) / (2 * eps_metric)
                kinetic_correction[i] = 0.5 * p @ dG_inv_dq_i @ p

        dp_dt = -grad_V - kinetic_correction

        return dq_dt, dp_dt

    def check_metric_signature(self, q: np.ndarray) -> MetricSignature:
        """
        Check if metric is Riemannian or Lorentzian at position q.

        Args:
            q: Position in parameter space

        Returns:
            signature: Metric signature type
        """
        G = self.metric(q)
        sig = analyze_metric_signature(G)
        return sig.signature


def create_belief_hamiltonian_from_agent(
    agent,
    point_idx: Optional[int] = None,
    include_dark: bool = True
) -> BeliefHamiltonian:
    """
    Create Hamiltonian for agent's belief dynamics.

    This constructs H(θ, p) where:
        - θ = belief parameters (μ, Σ, φ)
        - Kinetic term uses Fisher metric
        - Potential term uses free energy

    Args:
        agent: Agent with beliefs and free energy
        point_idx: Which point in base manifold (for spatially varying beliefs)
        include_dark: Include gauge field contribution

    Returns:
        hamiltonian: BeliefHamiltonian instance
    """
    # Define potential energy (free energy)
    def potential(theta):
        # theta encodes (μ, log σ, φ)
        # For now, simplified: use agent's current free energy
        # Full implementation would vary agent parameters and recompute

        # Placeholder: quadratic potential around current state
        theta_current = flatten_agent_params(agent, point_idx)
        return 0.5 * np.sum((theta - theta_current)**2)

    # Define metric (Fisher metric)
    def metric(theta):
        # Fisher metric for Gaussian beliefs
        # For now, use agent's current covariance

        if point_idx is not None and agent.Sigma_q.ndim > 2:
            # Spatial grid case
            if agent.Sigma_q.ndim == 3:
                Sigma = agent.Sigma_q[point_idx]
            else:
                # 2D grid
                H, W = agent.Sigma_q.shape[:2]
                i = point_idx // W
                j = point_idx % W
                Sigma = agent.Sigma_q[i, j]
        else:
            Sigma = agent.Sigma_q

        # Fisher metric: G = Σ^{-1}
        try:
            G = np.linalg.inv(Sigma)
        except np.linalg.LinAlgError:
            G = np.linalg.inv(Sigma + 1e-6 * np.eye(len(Sigma)))

        return G

    return BeliefHamiltonian(potential, metric)


def flatten_agent_params(agent, point_idx: Optional[int] = None) -> np.ndarray:
    """
    Flatten agent's belief parameters into vector θ.

    Args:
        agent: Agent with μ_q, Sigma_q, phi
        point_idx: Which spatial point (if grid)

    Returns:
        theta: Flattened parameter vector
    """
    if point_idx is not None and agent.mu_q.ndim > 1:
        # Spatial grid case
        if agent.mu_q.ndim == 2:
            mu = agent.mu_q[point_idx]
        else:
            # 2D grid
            H, W = agent.mu_q.shape[:2]
            i = point_idx // W
            j = point_idx % W
            mu = agent.mu_q[i, j]
    else:
        mu = agent.mu_q

    # For now, just use mu
    # Full version would include log(diag(Σ)), φ
    return mu


class SymplecticStructure:
    """
    Canonical symplectic structure on phase space (q, p).

    The symplectic 2-form is:
        ω = dq ∧ dp

    In matrix form:
        ω = [[0,  I],
             [-I, 0]]

    This defines the Poisson bracket and ensures energy conservation.
    """

    def __init__(self, dim: int):
        """
        Args:
            dim: Dimension of configuration space (phase space is 2*dim)
        """
        self.dim = dim
        self.phase_dim = 2 * dim

        # Symplectic matrix
        I = np.eye(dim)
        zero = np.zeros((dim, dim))
        self.omega = np.block([[zero, I], [-I, zero]])

    def poisson_bracket(
        self,
        f: Callable,
        g: Callable,
        q: np.ndarray,
        p: np.ndarray
    ) -> float:
        """
        Poisson bracket {f, g} = ∂f/∂q ∂g/∂p - ∂f/∂p ∂g/∂q.

        Args:
            f, g: Scalar functions on phase space
            q, p: Phase space coordinates

        Returns:
            {f, g}: Poisson bracket value
        """
        eps = 1e-5

        # Compute gradients
        df_dq = np.zeros(self.dim)
        df_dp = np.zeros(self.dim)
        dg_dq = np.zeros(self.dim)
        dg_dp = np.zeros(self.dim)

        for i in range(self.dim):
            # f gradients
            q_plus = q.copy(); q_plus[i] += eps
            q_minus = q.copy(); q_minus[i] -= eps
            df_dq[i] = (f(q_plus, p) - f(q_minus, p)) / (2 * eps)

            p_plus = p.copy(); p_plus[i] += eps
            p_minus = p.copy(); p_minus[i] -= eps
            df_dp[i] = (f(q, p_plus) - f(q, p_minus)) / (2 * eps)

            # g gradients
            dg_dq[i] = (g(q_plus, p) - g(q_minus, p)) / (2 * eps)
            dg_dp[i] = (g(q, p_plus) - g(q, p_minus)) / (2 * eps)

        return np.dot(df_dq, dg_dp) - np.dot(df_dp, dg_dq)


# Convenience functions

def overdamped_limit(hamiltonian: BeliefHamiltonian, q: np.ndarray, friction: float = 1.0) -> np.ndarray:
    """
    Overdamped (high friction) limit: γ θ̇ = -∇V.

    This recovers the current first-order dynamics.

    Args:
        hamiltonian: Hamiltonian system
        q: Current position
        friction: Friction coefficient γ

    Returns:
        dq_dt: Overdamped velocity
    """
    eps = 1e-5
    grad_V = np.zeros_like(q)
    for i in range(len(q)):
        q_plus = q.copy(); q_plus[i] += eps
        q_minus = q.copy(); q_minus[i] -= eps
        grad_V[i] = (hamiltonian.potential(q_plus) - hamiltonian.potential(q_minus)) / (2 * eps)

    return -grad_V / friction


def underdamped_dynamics(
    hamiltonian: BeliefHamiltonian,
    q: np.ndarray,
    p: np.ndarray,
    friction: float = 0.1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Underdamped dynamics with friction: q̈ + γq̇ = -∇V.

    In phase space:
        dq/dt = p/m
        dp/dt = -∇V - γp

    Args:
        hamiltonian: Hamiltonian system
        q, p: Phase space coordinates
        friction: Damping coefficient γ

    Returns:
        dq_dt, dp_dt: Time derivatives with damping
    """
    dq_dt, dp_dt = hamiltonian.equations_of_motion(q, p)

    # Add friction to momentum equation
    dp_dt = dp_dt - friction * p

    return dq_dt, dp_dt