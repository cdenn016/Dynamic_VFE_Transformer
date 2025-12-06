# -*- coding: utf-8 -*-
"""
Created on Sat Nov 22 12:54:48 2025

@author: chris and christine
"""

"""
Symplectic Integrators for Hamiltonian Dynamics
================================================

Numerical integration schemes that preserve the symplectic structure
and approximately conserve energy for Hamiltonian systems.

Why Symplectic Integrators?
----------------------------
Standard methods (Euler, RK4) don't preserve:
    1. Symplectic structure ω = dq ∧ dp
    2. Phase space volume (Liouville's theorem)
    3. Energy (for conservative systems)

Symplectic integrators maintain these geometric properties, giving:
    - Bounded energy error (no drift)
    - Correct long-term dynamics
    - Preservation of invariants

Methods Implemented:
-------------------
1. **Symplectic Euler** (1st order)
   - Simple, fast, O(h) error
   - Good for quick tests

2. **Verlet / Leapfrog** (2nd order)
   - Industry standard, O(h²) error
   - Time-reversible
   - Used in molecular dynamics

3. **Störmer-Verlet** (2nd order)
   - Symmetric, good energy conservation
   - Handles arbitrary Hamiltonians

4. **Ruth-3** (3rd order)
   - Higher accuracy, O(h³) error
   - More expensive per step

5. **Forest-Ruth / PEFRL** (4th order)
   - Very accurate, O(h⁴) error
   - Best for high-precision work

Author: Chris
Date: November 2025
"""

import numpy as np
from typing import Callable, Tuple, Optional
from dataclasses import dataclass

from dynamics.hamiltonian import BeliefHamiltonian, HamiltonianState


@dataclass
class IntegratorStats:
    """Statistics from integration."""
    steps: int
    energy_drift: float
    energy_std: float
    final_time: float


class SymplecticIntegrator:
    """
    Base class for symplectic integrators.

    All symplectic integrators preserve the canonical Poisson bracket
    and approximately conserve energy.
    """

    def __init__(self, hamiltonian: BeliefHamiltonian):
        """
        Args:
            hamiltonian: Hamiltonian system to integrate
        """
        self.hamiltonian = hamiltonian

    def step(
        self,
        q: np.ndarray,
        p: np.ndarray,
        dt: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Take one integration step.

        Args:
            q: Position at time t
            p: Momentum at time t
            dt: Time step

        Returns:
            q_new, p_new: State at time t + dt
        """
        raise NotImplementedError

    def integrate(
        self,
        q0: np.ndarray,
        p0: np.ndarray,
        t_span: Tuple[float, float],
        dt: float,
        save_interval: int = 1
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, IntegratorStats]:
        """
        Integrate Hamilton's equations.

        Args:
            q0, p0: Initial conditions
            t_span: (t_start, t_end)
            dt: Time step
            save_interval: Save every N steps

        Returns:
            t: Time array
            q_history: Position trajectory
            p_history: Momentum trajectory
            stats: Integration statistics
        """
        t_start, t_end = t_span
        n_steps = int((t_end - t_start) / dt)

        # Storage
        n_save = n_steps // save_interval + 1
        t_history = np.zeros(n_save)
        q_history = np.zeros((n_save, len(q0)))
        p_history = np.zeros((n_save, len(p0)))
        energies = np.zeros(n_save)

        # Initial state
        q, p = q0.copy(), p0.copy()
        t = t_start

        t_history[0] = t
        q_history[0] = q
        p_history[0] = p
        energies[0] = self.hamiltonian.total_energy(q, p)

        # Integrate
        save_idx = 1
        for step in range(1, n_steps + 1):
            q, p = self.step(q, p, dt)
            t += dt

            if step % save_interval == 0:
                t_history[save_idx] = t
                q_history[save_idx] = q
                p_history[save_idx] = p
                energies[save_idx] = self.hamiltonian.total_energy(q, p)
                save_idx += 1

        # Compute statistics
        energy_drift = energies[-1] - energies[0]
        energy_std = np.std(energies)

        stats = IntegratorStats(
            steps=n_steps,
            energy_drift=energy_drift,
            energy_std=energy_std,
            final_time=t
        )

        return t_history, q_history, p_history, stats


class SymplecticEuler(SymplecticIntegrator):
    """
    Symplectic Euler (1st order).

    Update scheme:
        p_{n+1} = p_n - dt * ∇V(q_n)
        q_{n+1} = q_n + dt * M^{-1} p_{n+1}

    Properties:
        - O(h) local error, O(h) global error
        - Symplectic (preserves ω)
        - Not time-reversible
        - Energy drift bounded but nonzero
    """

    def step(self, q: np.ndarray, p: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        dq_dt, dp_dt = self.hamiltonian.equations_of_motion(q, p)

        # Update momentum first (using old position)
        p_new = p + dt * dp_dt

        # Update position (using new momentum)
        dq_dt_new, _ = self.hamiltonian.equations_of_motion(q, p_new)
        q_new = q + dt * dq_dt_new

        return q_new, p_new


class Verlet(SymplecticIntegrator):
    """
    Verlet / Leapfrog integrator (2nd order).

    Update scheme (velocity-Verlet form):
        p_{n+1/2} = p_n - (dt/2) * ∇V(q_n)
        q_{n+1} = q_n + dt * M^{-1} p_{n+1/2}
        p_{n+1} = p_{n+1/2} - (dt/2) * ∇V(q_{n+1})

    Properties:
        - O(h²) local error, O(h²) global error
        - Symplectic
        - Time-reversible
        - Excellent energy conservation
        - Gold standard for molecular dynamics
    """

    def step(self, q: np.ndarray, p: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        # Half-step momentum update
        _, dp_dt = self.hamiltonian.equations_of_motion(q, p)
        p_half = p + 0.5 * dt * dp_dt

        # Full-step position update
        dq_dt, _ = self.hamiltonian.equations_of_motion(q, p_half)
        q_new = q + dt * dq_dt

        # Half-step momentum update
        _, dp_dt_new = self.hamiltonian.equations_of_motion(q_new, p_half)
        p_new = p_half + 0.5 * dt * dp_dt_new

        return q_new, p_new


class StormerVerlet(SymplecticIntegrator):
    """
    Störmer-Verlet (2nd order, position-form).

    Update scheme:
        q_{n+1} = 2q_n - q_{n-1} - dt² * M^{-1} ∇V(q_n)

    This is equivalent to Verlet but uses positions only (no velocities).

    Properties:
        - O(h²) error
        - Symplectic
        - Time-reversible
        - Requires storing q_{n-1}
    """

    def __init__(self, hamiltonian: BeliefHamiltonian):
        super().__init__(hamiltonian)
        self.q_prev = None  # Store previous position

    def step(self, q: np.ndarray, p: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        if self.q_prev is None:
            # First step: use Verlet
            verlet = Verlet(self.hamiltonian)
            q_new, p_new = verlet.step(q, p, dt)
            self.q_prev = q
            return q_new, p_new

        # Störmer-Verlet step
        _, dp_dt = self.hamiltonian.equations_of_motion(q, p)

        # Get metric for acceleration
        G = self.hamiltonian.metric(q)
        try:
            G_inv = np.linalg.inv(G)
        except np.linalg.LinAlgError:
            G_inv = np.linalg.inv(G + 1e-6 * np.eye(len(G)))

        # Acceleration: a = -G^{-1} ∇V
        eps = 1e-5
        grad_V = np.zeros_like(q)
        for i in range(len(q)):
            q_plus = q.copy(); q_plus[i] += eps
            q_minus = q.copy(); q_minus[i] -= eps
            grad_V[i] = (self.hamiltonian.potential(q_plus) -
                        self.hamiltonian.potential(q_minus)) / (2 * eps)

        accel = -G_inv @ grad_V

        # Position update
        q_new = 2 * q - self.q_prev + dt**2 * accel

        # Velocity (for momentum): v ≈ (q_new - q_prev) / (2*dt)
        velocity = (q_new - self.q_prev) / (2 * dt)
        p_new = G @ velocity  # p = G * v (Fisher metric as mass)

        # Update stored position
        self.q_prev = q

        return q_new, p_new


class Ruth3(SymplecticIntegrator):
    """
    Ruth's 3rd-order symplectic integrator.

    Uses composition of 3 symplectic Euler steps with special coefficients.

    Coefficients:
        c1 = 2/3, c2 = -2/3, c3 = 1
        d1 = 7/24, d2 = 3/4, d3 = -1/24

    Properties:
        - O(h³) local error
        - Symplectic
        - More accurate than Verlet for same step size
    """

    def __init__(self, hamiltonian: BeliefHamiltonian):
        super().__init__(hamiltonian)

        # Ruth's coefficients
        self.c = [2/3, -2/3, 1.0]
        self.d = [7/24, 3/4, -1/24]

    def step(self, q: np.ndarray, p: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        q_curr, p_curr = q.copy(), p.copy()

        for i in range(3):
            # Position update
            dq_dt, _ = self.hamiltonian.equations_of_motion(q_curr, p_curr)
            q_curr = q_curr + self.c[i] * dt * dq_dt

            # Momentum update
            _, dp_dt = self.hamiltonian.equations_of_motion(q_curr, p_curr)
            p_curr = p_curr + self.d[i] * dt * dp_dt

        return q_curr, p_curr


class PEFRL(SymplecticIntegrator):
    """
    Position-Extended Forest-Ruth Like (PEFRL) integrator (4th order).

    Optimized 4th-order symplectic integrator with excellent
    energy conservation properties.

    Properties:
        - O(h⁴) local error
        - Symplectic
        - Very stable for long-time integration
        - Optimized coefficients minimize error

    Reference: Omelyan, Mryglod, Folk (2002)
    """

    def __init__(self, hamiltonian: BeliefHamiltonian):
        super().__init__(hamiltonian)

        # PEFRL coefficients (optimized)
        self.xi = 0.1786178958448091
        self.lambda_ = -0.2123418310626054
        self.chi = -0.06626458266981849

    def step(self, q: np.ndarray, p: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        # PEFRL composition
        xi, lam, chi = self.xi, self.lambda_, self.chi

        # Step 1: q_1 = q_0 + xi * dt * v_0
        dq_dt, _ = self.hamiltonian.equations_of_motion(q, p)
        q1 = q + xi * dt * dq_dt

        # Step 2: p_1 = p_0 + (1-2*lam)/2 * dt * F(q_1)
        _, dp_dt = self.hamiltonian.equations_of_motion(q1, p)
        p1 = p + (1 - 2*lam) / 2 * dt * dp_dt

        # Step 3: q_2 = q_1 + chi * dt * v_1
        dq_dt, _ = self.hamiltonian.equations_of_motion(q1, p1)
        q2 = q1 + chi * dt * dq_dt

        # Step 4: p_2 = p_1 + lam * dt * F(q_2)
        _, dp_dt = self.hamiltonian.equations_of_motion(q2, p1)
        p2 = p1 + lam * dt * dp_dt

        # Step 5: q_3 = q_2 + (1 - 2*(chi+xi)) * dt * v_2
        dq_dt, _ = self.hamiltonian.equations_of_motion(q2, p2)
        q3 = q2 + (1 - 2*(chi + xi)) * dt * dq_dt

        # Step 6: p_3 = p_2 + lam * dt * F(q_3)
        _, dp_dt = self.hamiltonian.equations_of_motion(q3, p2)
        p3 = p2 + lam * dt * dp_dt

        # Step 7: q_4 = q_3 + chi * dt * v_3
        dq_dt, _ = self.hamiltonian.equations_of_motion(q3, p3)
        q4 = q3 + chi * dt * dq_dt

        # Step 8: p_4 = p_3 + (1-2*lam)/2 * dt * F(q_4)
        _, dp_dt = self.hamiltonian.equations_of_motion(q4, p3)
        p4 = p3 + (1 - 2*lam) / 2 * dt * dp_dt

        # Step 9: q_5 = q_4 + xi * dt * v_4
        dq_dt, _ = self.hamiltonian.equations_of_motion(q4, p4)
        q5 = q4 + xi * dt * dq_dt

        return q5, p4


# Utility functions

def compare_integrators(
    hamiltonian: BeliefHamiltonian,
    q0: np.ndarray,
    p0: np.ndarray,
    dt: float = 0.01,
    t_end: float = 10.0
) -> dict:
    """
    Compare different integrators on the same Hamiltonian.

    Args:
        hamiltonian: System to integrate
        q0, p0: Initial conditions
        dt: Time step
        t_end: Final time

    Returns:
        results: Dict with stats for each integrator
    """
    integrators = {
        'Symplectic Euler': SymplecticEuler(hamiltonian),
        'Verlet': Verlet(hamiltonian),
        'Ruth-3': Ruth3(hamiltonian),
        'PEFRL': PEFRL(hamiltonian)
    }

    results = {}

    print(f"\n{'='*70}")
    print(f"INTEGRATOR COMPARISON")
    print(f"{'='*70}")
    print(f"Initial energy: {hamiltonian.total_energy(q0, p0):.6f}")
    print(f"Time step: dt = {dt}")
    print(f"Final time: T = {t_end}")
    print()

    for name, integrator in integrators.items():
        _, _, _, stats = integrator.integrate(
            q0, p0,
            t_span=(0.0, t_end),
            dt=dt,
            save_interval=10
        )

        results[name] = stats

        print(f"{name:20s} | Energy drift: {stats.energy_drift:+.6e} | "
              f"Energy std: {stats.energy_std:.6e}")

    print(f"{'='*70}\n")

    return results