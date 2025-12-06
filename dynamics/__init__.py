# -*- coding: utf-8 -*-
"""
Created on Sat Nov 22 12:54:11 2025

@author: chris and christine
"""

"""
Dynamics Module
===============

Full Hamiltonian and Lagrangian dynamics for belief evolution.

Key Insight:
-----------
The Fisher metric (∇²KL) naturally provides kinetic terms:

    KL(q||p) ≈ const + g·δθ + (1/2)δθ^T G δθ
                              \_____________/
                              kinetic term!

This gives Hamiltonian: H = (1/2)p^T G^{-1} p + V(θ)
Or Lagrangian: L = (1/2)θ̇^T G θ̇ - V(θ)

Regimes:
--------
- Overdamped (current): θ̇ = -∇V (first-order, dissipative)
- Underdamped (new): Full second-order dynamics with momentum
- Critical: Transition regime

Modules:
--------
- hamiltonian: Hamiltonian formulation (phase space)
- lagrangian: Lagrangian formulation (configuration space)
- integrators: Symplectic integration schemes

Usage:
------
    from dynamics import BeliefHamiltonian, Verlet

    # Create Hamiltonian
    H = BeliefHamiltonian(potential=free_energy, metric=fisher_metric)

    # Integrate with Verlet
    integrator = Verlet(H)
    t, q, p, stats = integrator.integrate(q0, p0, (0, 10), dt=0.01)

Author: Chris
Date: November 2025
"""

from dynamics.hamiltonian import (
    BeliefHamiltonian,
    HamiltonianState,
    SymplecticStructure,
    create_belief_hamiltonian_from_agent,
    overdamped_limit,
    underdamped_dynamics
)

from dynamics.lagrangian import (
    BeliefLagrangian,
    LagrangianState,
    LagrangianIntegrator,
    action_integral,
    variational_principle
)

from dynamics.integrators import (
    SymplecticIntegrator,
    SymplecticEuler,
    Verlet,
    StormerVerlet,
    Ruth3,
    PEFRL,
    IntegratorStats,
    compare_integrators
)

__all__ = [
    # Hamiltonian
    'BeliefHamiltonian',
    'HamiltonianState',
    'SymplecticStructure',
    'create_belief_hamiltonian_from_agent',
    'overdamped_limit',
    'underdamped_dynamics',

    # Lagrangian
    'BeliefLagrangian',
    'LagrangianState',
    'LagrangianIntegrator',
    'action_integral',
    'variational_principle',

    # Integrators
    'SymplecticIntegrator',
    'SymplecticEuler',
    'Verlet',
    'StormerVerlet',
    'Ruth3',
    'PEFRL',
    'IntegratorStats',
    'compare_integrators'
]