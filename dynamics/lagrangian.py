# -*- coding: utf-8 -*-
"""
Created on Sat Nov 22 12:55:17 2025

@author: chris and christine
"""

"""
Lagrangian Dynamics for Belief Evolution
=========================================

Lagrangian formulation as an alternative to Hamiltonian mechanics.

The Lagrangian L = T - V encodes the same dynamics but in configuration
space (q, q̇) rather than phase space (q, p).

Mathematical Foundation
-----------------------

Kinetic energy from Fisher metric:
    T = (1/2) q̇^T G(q) q̇

where G(q) is the Fisher information metric (plays role of mass tensor).

Potential energy from free energy:
    V = F(q)

Lagrangian:
    L(q, q̇) = T - V = (1/2) q̇^T G(q) q̇ - F(q)

Euler-Lagrange equations:
    d/dt(∂L/∂q̇) - ∂L/∂q = 0

This gives:
    G(q) q̈ + dG/dq q̇ q̇ - (1/2) ∂G/∂q q̇^T q̇ = -∇F

Relation to Hamiltonian:
    p = ∂L/∂q̇ = G(q) q̇    (conjugate momentum)
    H(q, p) = p^T q̇ - L    (Legendre transform)

Advantages:
    - More intuitive (configuration space)
    - Natural for constrained systems
    - Direct connection to action principle

Author: Chris
Date: November 2025
"""

import numpy as np
from typing import Callable, Tuple, Optional
from dataclasses import dataclass


@dataclass
class LagrangianState:
    """
    State in configuration space.

    Attributes:
        q: Generalized coordinates
        q_dot: Generalized velocities
        t: Time
        energy: Total energy E = T + V
    """
    q: np.ndarray
    q_dot: np.ndarray
    t: float = 0.0
    energy: Optional[float] = None


class BeliefLagrangian:
    """
    Lagrangian for belief dynamics: L = T - V.

    Where:
        T = (1/2) q̇^T G(q) q̇  (kinetic, Fisher metric)
        V = F(q)                (potential, free energy)
    """

    def __init__(
        self,
        potential: Callable,      # V(q) → float
        metric: Callable,         # G(q) → (d, d)
        metric_gradient: Optional[Callable] = None  # ∂G/∂q
    ):
        """
        Args:
            potential: Potential energy V(q)
            metric: Riemannian metric G(q) (Fisher metric)
            metric_gradient: Gradient ∂G/∂q for Christoffel symbols
        """
        self.potential = potential
        self.metric = metric
        self.metric_gradient = metric_gradient

    def kinetic_energy(self, q: np.ndarray, q_dot: np.ndarray) -> float:
        """
        Kinetic energy T = (1/2) q̇^T G(q) q̇.

        Args:
            q: Position
            q_dot: Velocity

        Returns:
            T: Kinetic energy
        """
        G = self.metric(q)
        return 0.5 * q_dot @ G @ q_dot

    def potential_energy(self, q: np.ndarray) -> float:
        """
        Potential energy V = F(q).

        Args:
            q: Position

        Returns:
            V: Potential energy (free energy)
        """
        return self.potential(q)

    def lagrangian(self, q: np.ndarray, q_dot: np.ndarray) -> float:
        """
        Lagrangian L = T - V.

        Args:
            q: Position
            q_dot: Velocity

        Returns:
            L: Lagrangian value
        """
        return self.kinetic_energy(q, q_dot) - self.potential_energy(q)

    def euler_lagrange(
        self,
        q: np.ndarray,
        q_dot: np.ndarray
    ) -> np.ndarray:
        """
        Euler-Lagrange equations: d/dt(∂L/∂q̇) - ∂L/∂q = 0.

        Returns acceleration q̈.

        The equation is:
            G(q) q̈ + [Christoffel terms] = -∇V

        where Christoffel symbols come from d/dt(G q̇).

        Args:
            q: Position
            q_dot: Velocity

        Returns:
            q_ddot: Acceleration
        """
        G = self.metric(q)

        # Compute force: F = -∇V
        eps = 1e-5
        force = np.zeros_like(q)
        for i in range(len(q)):
            q_plus = q.copy(); q_plus[i] += eps
            q_minus = q.copy(); q_minus[i] -= eps
            force[i] = -(self.potential(q_plus) - self.potential(q_minus)) / (2 * eps)

        # Christoffel symbol contribution: Γ^k_ij q̇^i q̇^j
        # From: d/dt(∂L/∂q̇) = d/dt(G q̇) = Ġ q̇ + G q̈
        # Where: Ġ = dG/dq · q̇ (chain rule)

        christoffel_force = np.zeros_like(q)

        if self.metric_gradient is not None:
            # Use provided gradient
            dG_dq = self.metric_gradient(q)
            for k in range(len(q)):
                G_dot = np.zeros_like(G)
                for i in range(len(q)):
                    G_dot += dG_dq[i] * q_dot[i]  # dG/dq^i * q̇^i
                christoffel_force[k] = -0.5 * q_dot @ (G_dot[k, :] + G_dot[:, k]) @ q_dot
        else:
            # Finite difference for metric gradient
            for k in range(len(q)):
                q_plus = q.copy(); q_plus[k] += eps
                q_minus = q.copy(); q_minus[k] -= eps

                G_plus = self.metric(q_plus)
                G_minus = self.metric(q_minus)

                dG_dk = (G_plus - G_minus) / (2 * eps)

                # Christoffel: -1/2 * q̇^T (∂_k G + ∂_k G^T) q̇
                # For symmetric G: -q̇^T ∂_k G q̇
                christoffel_force[k] = -q_dot @ dG_dk @ q_dot

        # Solve: G q̈ = F + Christoffel
        try:
            G_inv = np.linalg.inv(G)
        except np.linalg.LinAlgError:
            G_inv = np.linalg.inv(G + 1e-6 * np.eye(len(G)))

        q_ddot = G_inv @ (force + christoffel_force)

        return q_ddot

    def to_hamiltonian(self, q: np.ndarray, q_dot: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Legendre transform to Hamiltonian variables.

        p = ∂L/∂q̇ = G(q) q̇

        Args:
            q: Position
            q_dot: Velocity

        Returns:
            q, p: Phase space coordinates
        """
        G = self.metric(q)
        p = G @ q_dot
        return q, p

    def from_hamiltonian(self, q: np.ndarray, p: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Inverse Legendre transform from Hamiltonian.

        q̇ = G^{-1}(q) p

        Args:
            q: Position
            p: Momentum

        Returns:
            q, q_dot: Configuration space coordinates
        """
        G = self.metric(q)
        try:
            G_inv = np.linalg.inv(G)
        except np.linalg.LinAlgError:
            G_inv = np.linalg.inv(G + 1e-6 * np.eye(len(G)))

        q_dot = G_inv @ p
        return q, q_dot


def action_integral(
    lagrangian: BeliefLagrangian,
    q_trajectory: np.ndarray,
    t_array: np.ndarray
) -> float:
    """
    Compute action S = ∫ L(q, q̇) dt.

    The principle of least action: true trajectory minimizes S.

    Args:
        lagrangian: Lagrangian system
        q_trajectory: (T, d) trajectory
        t_array: (T,) time points

    Returns:
        S: Action
    """
    T = len(t_array)
    S = 0.0

    for i in range(T - 1):
        q = q_trajectory[i]
        dt = t_array[i+1] - t_array[i]

        # Velocity (finite difference)
        q_dot = (q_trajectory[i+1] - q) / dt

        # Lagrangian
        L = lagrangian.lagrangian(q, q_dot)

        # Integrate (trapezoidal rule)
        q_next = q_trajectory[i+1]
        q_dot_next = (q_trajectory[i+1] - q_trajectory[i]) / dt if i < T-2 else q_dot
        L_next = lagrangian.lagrangian(q_next, q_dot_next)

        S += 0.5 * (L + L_next) * dt

    return S


def variational_principle(
    lagrangian: BeliefLagrangian,
    q_trajectory: np.ndarray,
    t_array: np.ndarray,
    perturbation: Optional[np.ndarray] = None
) -> float:
    """
    Test variational principle: δS = 0 for true trajectory.

    Args:
        lagrangian: System
        q_trajectory: Candidate trajectory
        t_array: Time points
        perturbation: δq (if None, use random)

    Returns:
        δS: Variation of action (should be ≈ 0 for true trajectory)
    """
    if perturbation is None:
        # Random perturbation (vanishing at endpoints)
        perturbation = np.random.randn(*q_trajectory.shape) * 0.01
        perturbation[0] = 0
        perturbation[-1] = 0

    # Original action
    S0 = action_integral(lagrangian, q_trajectory, t_array)

    # Perturbed action
    q_perturbed = q_trajectory + perturbation
    S1 = action_integral(lagrangian, q_perturbed, t_array)

    # First variation
    delta_S = S1 - S0

    return delta_S


class LagrangianIntegrator:
    """
    Integrate Euler-Lagrange equations (second-order ODEs).

    This is less efficient than symplectic Hamiltonian integrators,
    but sometimes more intuitive.
    """

    def __init__(self, lagrangian: BeliefLagrangian):
        self.lagrangian = lagrangian

    def rk4_step(
        self,
        q: np.ndarray,
        q_dot: np.ndarray,
        dt: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        RK4 step for second-order ODE.

        State: y = [q, q̇]
        Derivative: dy/dt = [q̇, q̈(q, q̇)]

        Args:
            q, q_dot: Current state
            dt: Time step

        Returns:
            q_new, q_dot_new: Updated state
        """
        # RK4 for system: q̇ = v, v̇ = a(q, v)

        def derivative(q_curr, v_curr):
            a_curr = self.lagrangian.euler_lagrange(q_curr, v_curr)
            return v_curr, a_curr

        # k1
        dq1, dv1 = derivative(q, q_dot)

        # k2
        q2 = q + 0.5 * dt * dq1
        v2 = q_dot + 0.5 * dt * dv1
        dq2, dv2 = derivative(q2, v2)

        # k3
        q3 = q + 0.5 * dt * dq2
        v3 = q_dot + 0.5 * dt * dv2
        dq3, dv3 = derivative(q3, v3)

        # k4
        q4 = q + dt * dq3
        v4 = q_dot + dt * dv3
        dq4, dv4 = derivative(q4, v4)

        # Combine
        q_new = q + (dt/6) * (dq1 + 2*dq2 + 2*dq3 + dq4)
        v_new = q_dot + (dt/6) * (dv1 + 2*dv2 + 2*dv3 + dv4)

        return q_new, v_new

    def integrate(
        self,
        q0: np.ndarray,
        q_dot0: np.ndarray,
        t_span: Tuple[float, float],
        dt: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Integrate Euler-Lagrange equations.

        Args:
            q0, q_dot0: Initial conditions
            t_span: (t_start, t_end)
            dt: Time step

        Returns:
            t, q_history, q_dot_history: Trajectory
        """
        t_start, t_end = t_span
        n_steps = int((t_end - t_start) / dt)

        t = np.zeros(n_steps + 1)
        q_history = np.zeros((n_steps + 1, len(q0)))
        q_dot_history = np.zeros((n_steps + 1, len(q0)))

        t[0] = t_start
        q_history[0] = q0
        q_dot_history[0] = q_dot0

        q, q_dot = q0.copy(), q_dot0.copy()

        for i in range(n_steps):
            q, q_dot = self.rk4_step(q, q_dot, dt)
            t[i+1] = t_start + (i+1) * dt
            q_history[i+1] = q
            q_dot_history[i+1] = q_dot

        return t, q_history, q_dot_history