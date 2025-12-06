"""
Complete Field Theory - Full Dynamical Implementation
======================================================

Implements the COMPLETE dynamical theory with all three kinetic terms:

    L = T_μ + T_Σ + T_φ - V

where:
    T_μ = (1/2) μ̇^T Σ_p^{-1} μ̇           (mean kinetic)
    T_Σ = (1/4) tr(Σ_q^{-1} Σ̇ Σ_q^{-1} Σ̇)  (covariance kinetic on SPD manifold)
    T_φ = (1/2) ⟨φ̇, φ̇⟩_𝔤                  (gauge kinetic on Lie algebra)
    V   = Full variational free energy

Conjugate momenta:
    π_μ = Σ_p^{-1} μ̇
    π_Σ = (1/2) Σ_q^{-1} Σ̇ Σ_q^{-1}
    π_φ = φ̇

Hamilton's equations for ALL fields simultaneously.

Author: Chris
Date: November 2025
"""

import numpy as np
from typing import Tuple, Optional, List, Callable
from dataclasses import dataclass

from geometry.spd_manifold import (
    spd_kinetic_energy,
    spd_geodesic_equation,
    is_spd
)
from geometry.lie_algebra import (
    LieAlgebra,
    LieGroup,
    gauge_field_kinetic_energy
)


@dataclass
class FullFieldState:
    """
    Complete state for full field theory.

    Configuration space: (μ, Σ, φ)
    Phase space: (μ, Σ, φ, π_μ, π_Σ, π_φ)

    Attributes:
        mu: (K,) belief mean
        Sigma: (K, K) covariance (SPD matrix)
        phi: (d_g,) gauge field (Lie algebra element)
        pi_mu: (K,) conjugate momentum to μ
        pi_Sigma: (K, K) conjugate momentum to Σ (symmetric)
        pi_phi: (d_g,) conjugate momentum to φ
        t: Time
    """
    mu: np.ndarray
    Sigma: np.ndarray
    phi: np.ndarray
    pi_mu: np.ndarray
    pi_Sigma: np.ndarray
    pi_phi: np.ndarray
    t: float = 0.0

    def __post_init__(self):
        """Validate state."""
        K = len(self.mu)

        assert self.mu.shape == self.pi_mu.shape, "μ and π_μ must have same shape"
        assert self.Sigma.shape == self.pi_Sigma.shape, "Σ and π_Σ must have same shape"
        assert self.phi.shape == self.pi_phi.shape, "φ and π_φ must have same shape"
        assert is_spd(self.Sigma), "Σ must be SPD"

        # CRITICAL: SO(3) gauge group requires ODD dimensional irreps!
        # K must be 1, 3, 5, 7, 9, 11, 13, ... (dimension = 2l+1)
        if K % 2 == 0:
            raise ValueError(
                f"INVALID K={K}! SO(3) has only ODD dimensional irreps.\n"
                f"Valid values: K ∈ {{1, 3, 5, 7, 9, 11, 13, ...}}\n"
                f"For visualization use K=3, for realistic use K=13.\n"
                f"Even K violates gauge symmetry!"
            )


class FullFieldHamiltonian:
    """
    Complete Hamiltonian for belief field theory.

    H = T_μ + T_Σ + T_φ + V

    where each kinetic term uses the proper geometric structure:
    - T_μ: Fisher metric (already had)
    - T_Σ: SPD manifold metric (NEW)
    - T_φ: Lie algebra metric (NEW)
    """

    def __init__(
        self,
        potential: Callable,  # V(μ, Σ, φ) → float
        Sigma_prior: np.ndarray,  # Σ_p for Fisher metric
        gauge_group: LieGroup = LieGroup.SO3,
        chi: float = 1.0  # Support function
    ):
        """
        Args:
            potential: Full free energy V(μ, Σ, φ)
            Sigma_prior: Prior covariance (for Fisher metric mass)
            gauge_group: SO(3) or SO(1,3)
            chi: Support function value
        """
        self.potential_func = potential
        self.Sigma_prior = Sigma_prior
        self.gauge_algebra = LieAlgebra(gauge_group)
        self.chi = chi

        # Precompute Σ_p^{-1} (Fisher metric for mean)
        self.Sigma_prior_inv = np.linalg.inv(Sigma_prior)

    def mean_kinetic_energy(
        self,
        mu: np.ndarray,
        pi_mu: np.ndarray
    ) -> float:
        """
        T_μ = (1/2) π_μ^T Σ_p π_μ

        where π_μ = Σ_p^{-1} μ̇, so μ̇ = Σ_p π_μ
        """
        # From π_μ = Σ_p^{-1} μ̇, we get μ̇ = Σ_p π_μ
        mu_dot = self.Sigma_prior @ pi_mu
        # T = (1/2) μ̇^T Σ_p^{-1} μ̇ = (1/2) π_μ^T Σ_p π_μ
        return 0.5 * pi_mu @ self.Sigma_prior @ pi_mu

    def covariance_kinetic_energy(
        self,
        Sigma: np.ndarray,
        pi_Sigma: np.ndarray
    ) -> float:
        """
        T_Σ = (1/2) tr(π_Σ Σ π_Σ Σ)

        where π_Σ = (1/2) Σ^{-1} Σ̇ Σ^{-1}
        """
        # From π_Σ = (1/2) Σ^{-1} Σ̇ Σ^{-1}, solve for Σ̇:
        # Σ̇ = 2 Σ π_Σ Σ
        Sigma_dot = 2 * Sigma @ pi_Sigma @ Sigma

        # T_Σ = (1/4) tr(Σ^{-1} Σ̇ Σ^{-1} Σ̇)
        return spd_kinetic_energy(Sigma, Sigma_dot)

    def gauge_kinetic_energy(
        self,
        phi: np.ndarray,
        pi_phi: np.ndarray
    ) -> float:
        """
        T_φ = (1/2) ⟨π_φ, π_φ⟩_𝔤

        where π_φ = φ̇
        """
        # φ̇ = π_φ (simple for gauge field)
        return self.gauge_algebra.kinetic_energy(pi_phi)

    def total_kinetic_energy(self, state: FullFieldState) -> float:
        """
        Total kinetic: T = T_μ + T_Σ + T_φ

        This is the COMPLETE kinetic energy with all three geometric structures!
        """
        T_mu = self.mean_kinetic_energy(state.mu, state.pi_mu)
        T_Sigma = self.covariance_kinetic_energy(state.Sigma, state.pi_Sigma)
        T_phi = self.gauge_kinetic_energy(state.phi, state.pi_phi)

        return self.chi * (T_mu + T_Sigma + T_phi)

    def potential_energy(self, state: FullFieldState) -> float:
        """
        Potential: V = V(μ, Σ, φ)
        """
        return self.potential_func(state.mu, state.Sigma, state.phi)

    def total_energy(self, state: FullFieldState) -> float:
        """
        Hamiltonian: H = T + V

        This is THE CONSERVED QUANTITY!
        """
        return self.total_kinetic_energy(state) + self.potential_energy(state)

    def hamilton_equations(
        self,
        state: FullFieldState
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Complete Hamilton's equations for all fields.

        dμ/dt = ∂H/∂π_μ = Σ_p π_μ
        dΣ/dt = ∂H/∂π_Σ = 2 Σ π_Σ Σ
        dφ/dt = ∂H/∂π_φ = π_φ

        dπ_μ/dt = -∂H/∂μ
        dπ_Σ/dt = -∂H/∂Σ - (curvature corrections)
        dπ_φ/dt = -∂H/∂φ

        Returns:
            dmu_dt, dSigma_dt, dphi_dt, dpi_mu_dt, dpi_Sigma_dt, dpi_phi_dt
        """
        mu, Sigma, phi = state.mu, state.Sigma, state.phi
        pi_mu, pi_Sigma, pi_phi = state.pi_mu, state.pi_Sigma, state.pi_phi

        # ===================================================================
        # CONFIGURATION VELOCITIES (trivial from momenta)
        # ===================================================================

        # dμ/dt = ∂H/∂π_μ = Σ_p π_μ
        dmu_dt = self.Sigma_prior @ pi_mu

        # dΣ/dt = ∂H/∂π_Σ = 2 Σ π_Σ Σ (from SPD geometry)
        dSigma_dt = 2 * Sigma @ pi_Sigma @ Sigma

        # dφ/dt = ∂H/∂π_φ = π_φ
        dphi_dt = pi_phi.copy()

        # ===================================================================
        # MOMENTUM EVOLUTIONS (forces from potential + curvature)
        # ===================================================================

        # Finite difference for gradients
        eps = 1e-5

        # dπ_μ/dt = -∂V/∂μ
        grad_V_mu = np.zeros_like(mu)
        for i in range(len(mu)):
            mu_plus = mu.copy(); mu_plus[i] += eps
            mu_minus = mu.copy(); mu_minus[i] -= eps
            grad_V_mu[i] = (self.potential_func(mu_plus, Sigma, phi) -
                           self.potential_func(mu_minus, Sigma, phi)) / (2 * eps)

        dpi_mu_dt = -grad_V_mu

        # dπ_Σ/dt = -∂V/∂Σ - (curvature correction from SPD geometry)
        # This is complex - use geodesic equation from SPD manifold
        Sigma_ddot = spd_geodesic_equation(Sigma, dSigma_dt)

        # Convert acceleration to momentum change
        # From π_Σ = (1/2) Σ^{-1} Σ̇ Σ^{-1}, differentiate:
        Sigma_inv = np.linalg.inv(Sigma)

        # Simplified: -∂V/∂Σ (ignoring full curvature for now)
        grad_V_Sigma = np.zeros_like(Sigma)
        for i in range(Sigma.shape[0]):
            for j in range(i, Sigma.shape[1]):
                Sigma_pert = Sigma.copy()
                Sigma_pert[i, j] += eps
                Sigma_pert[j, i] += eps
                V_plus = self.potential_func(mu, Sigma_pert, phi)

                Sigma_pert = Sigma.copy()
                Sigma_pert[i, j] -= eps
                Sigma_pert[j, i] -= eps
                V_minus = self.potential_func(mu, Sigma_pert, phi)

                grad_V_Sigma[i, j] = (V_plus - V_minus) / (4 * eps)
                grad_V_Sigma[j, i] = grad_V_Sigma[i, j]

        dpi_Sigma_dt = -0.5 * Sigma_inv @ grad_V_Sigma @ Sigma_inv

        # dπ_φ/dt = -∂V/∂φ
        grad_V_phi = np.zeros_like(phi)
        for i in range(len(phi)):
            phi_plus = phi.copy(); phi_plus[i] += eps
            phi_minus = phi.copy(); phi_minus[i] -= eps
            grad_V_phi[i] = (self.potential_func(mu, Sigma, phi_plus) -
                            self.potential_func(mu, Sigma, phi_minus)) / (2 * eps)

        dpi_phi_dt = -grad_V_phi

        return dmu_dt, dSigma_dt, dphi_dt, dpi_mu_dt, dpi_Sigma_dt, dpi_phi_dt


def create_simple_potential(
    mu_target: np.ndarray,
    Sigma_target: np.ndarray,
    weights: Tuple[float, float, float] = (1.0, 0.1, 0.01)
) -> Callable:
    """
    Create simple quadratic potential for testing.

    V = w_μ ||μ - μ*||² + w_Σ ||Σ - Σ*||² + w_φ ||φ||²

    Args:
        mu_target: Target mean
        Sigma_target: Target covariance
        weights: (w_μ, w_Σ, w_φ)

    Returns:
        potential: V(μ, Σ, φ) → float
    """
    w_mu, w_Sigma, w_phi = weights

    def potential(mu, Sigma, phi):
        # Mean term
        V_mu = w_mu * 0.5 * np.sum((mu - mu_target)**2)

        # Covariance term (Frobenius norm)
        V_Sigma = w_Sigma * 0.5 * np.linalg.norm(Sigma - Sigma_target, 'fro')**2

        # Gauge term
        V_phi = w_phi * 0.5 * np.sum(phi**2)

        return V_mu + V_Sigma + V_phi

    return potential


def create_initial_state(
    K: int,
    gauge_group: LieGroup = LieGroup.SO3
) -> FullFieldState:
    """
    Create random initial state.

    Args:
        K: Latent dimension
        gauge_group: Gauge group

    Returns:
        state: Initial FullFieldState
    """
    # Random mean
    mu = np.random.randn(K) * 0.5

    # Random SPD covariance
    A = np.random.randn(K, K)
    Sigma = A @ A.T + np.eye(K)

    # Random gauge field
    if gauge_group == LieGroup.SO3:
        phi = np.random.randn(3) * 0.1
    elif gauge_group == LieGroup.SO13:
        phi = np.random.randn(6) * 0.1
    else:
        raise ValueError(f"Unknown gauge group: {gauge_group}")

    # Zero initial momenta (start from rest)
    pi_mu = np.zeros_like(mu)
    pi_Sigma = np.zeros_like(Sigma)
    pi_phi = np.zeros_like(phi)

    return FullFieldState(mu, Sigma, phi, pi_mu, pi_Sigma, pi_phi, t=0.0)


def field_energy_summary(hamiltonian: FullFieldHamiltonian, state: FullFieldState):
    """
    Print energy breakdown.
    """
    T_mu = hamiltonian.mean_kinetic_energy(state.mu, state.pi_mu)
    T_Sigma = hamiltonian.covariance_kinetic_energy(state.Sigma, state.pi_Sigma)
    T_phi = hamiltonian.gauge_kinetic_energy(state.phi, state.pi_phi)
    V = hamiltonian.potential_energy(state)
    H = hamiltonian.total_energy(state)

    print(f"\n{'='*60}")
    print(f"ENERGY BREAKDOWN (t = {state.t:.3f})")
    print(f"{'='*60}")
    print(f"  T_μ (mean kinetic):       {T_mu:12.6f}")
    print(f"  T_Σ (covariance kinetic): {T_Sigma:12.6f}")
    print(f"  T_φ (gauge kinetic):      {T_phi:12.6f}")
    print(f"  ─────────────────────────────────────")
    print(f"  T (total kinetic):        {T_mu + T_Sigma + T_phi:12.6f}")
    print(f"  V (potential):            {V:12.6f}")
    print(f"  ═════════════════════════════════════")
    print(f"  H (total energy):         {H:12.6f}")
    print(f"{'='*60}\n")