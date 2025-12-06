"""
Thermostats for Hamiltonian Dynamics
=====================================

Temperature control mechanisms for Hamiltonian belief dynamics.

The pure Hamiltonian H = T + V conserves energy, which means the system
oscillates at fixed temperature. For training, we want controlled
annealing: exploration at high T, exploitation at low T.

NOSÉ-HOOVER THERMOSTAT
----------------------
The Nosé-Hoover extended Hamiltonian couples the system to a heat bath:

    H_NH = H(q, p) + π_ξ²/(2Q) + gkT·ξ

where:
    ξ = thermostat position (dimensionless)
    π_ξ = thermostat momentum
    Q = thermostat mass (controls coupling strength)
    g = number of degrees of freedom
    kT = target temperature (thermal energy)

Modified equations of motion:
    dq/dt = ∂H/∂p
    dp/dt = -∂H/∂q - ξ·p        ← friction term!
    dξ/dt = π_ξ / Q
    dπ_ξ/dt = (p²/m - gkT)      ← thermostat force

The thermostat friction ξ·p extracts/injects energy to maintain T.

ANNEALING SCHEDULES
-------------------
For training, we anneal temperature according to:

    T(t) = T_init × schedule(t)

Schedules:
    - Exponential: exp(-t/τ)
    - Polynomial: 1/(1 + t/τ)
    - Cosine: (1 + cos(πt/τ_max))/2
    - Step: T_init if t < t_switch else T_final

PHYSICAL INTERPRETATION
-----------------------
High temperature → Momenta have large variance → Exploration
Low temperature → Momenta have small variance → Exploitation

At T=0, we recover gradient descent (system falls to minimum).
At T>0, we have thermal fluctuations that help escape local minima.

Author: Chris & Claude
Date: December 2025
"""

import torch
import torch.nn as nn
from typing import Optional, Literal, Callable
from dataclasses import dataclass
import math


# =============================================================================
# Nosé-Hoover Thermostat State
# =============================================================================

@dataclass
class ThermostatState:
    """
    State of the Nosé-Hoover thermostat.

    The thermostat is an extended degree of freedom that couples
    the physical system to a heat bath at temperature T.
    """
    xi: torch.Tensor          # (B,) thermostat position (friction coefficient)
    pi_xi: torch.Tensor       # (B,) thermostat momentum
    Q: float                  # Thermostat mass (coupling strength)
    target_temperature: float  # Target kT

    def clone(self) -> 'ThermostatState':
        return ThermostatState(
            xi=self.xi.clone(),
            pi_xi=self.pi_xi.clone(),
            Q=self.Q,
            target_temperature=self.target_temperature,
        )


# =============================================================================
# Annealing Schedules
# =============================================================================

class AnnealingSchedule(nn.Module):
    """Base class for temperature annealing schedules."""

    def __init__(self, T_init: float = 1.0, T_final: float = 0.01):
        super().__init__()
        self.T_init = T_init
        self.T_final = T_final

    def forward(self, step: int, total_steps: int) -> float:
        """Return temperature at given step."""
        raise NotImplementedError


class ExponentialAnnealing(AnnealingSchedule):
    """
    Exponential annealing: T(t) = T_init × exp(-t/τ)

    Fast initial cooling, slow final approach to T_final.
    """

    def __init__(self, T_init: float = 1.0, T_final: float = 0.01, tau: Optional[float] = None):
        super().__init__(T_init, T_final)
        self.tau = tau  # If None, computed from total_steps

    def forward(self, step: int, total_steps: int) -> float:
        if self.tau is None:
            # Compute τ such that T(total_steps) ≈ T_final
            tau = -total_steps / math.log(self.T_final / self.T_init + 1e-10)
        else:
            tau = self.tau

        return self.T_init * math.exp(-step / tau)


class PolynomialAnnealing(AnnealingSchedule):
    """
    Polynomial annealing: T(t) = T_init / (1 + t/τ)^α

    Slower cooling than exponential. α controls decay rate.
    """

    def __init__(self, T_init: float = 1.0, T_final: float = 0.01, alpha: float = 1.0):
        super().__init__(T_init, T_final)
        self.alpha = alpha

    def forward(self, step: int, total_steps: int) -> float:
        # Compute τ such that T(total_steps) = T_final
        tau = total_steps / ((self.T_init / self.T_final) ** (1 / self.alpha) - 1)
        return self.T_init / (1 + step / tau) ** self.alpha


class CosineAnnealing(AnnealingSchedule):
    """
    Cosine annealing: T(t) = T_final + (T_init - T_final) × (1 + cos(πt/T))/2

    Smooth S-curve transition. Popular in deep learning.
    """

    def forward(self, step: int, total_steps: int) -> float:
        cos_factor = (1 + math.cos(math.pi * step / total_steps)) / 2
        return self.T_final + (self.T_init - self.T_final) * cos_factor


class CyclicAnnealing(AnnealingSchedule):
    """
    Cyclic annealing with warm restarts.

    Periodically reheats to escape local minima, then cools again.
    """

    def __init__(
        self,
        T_init: float = 1.0,
        T_final: float = 0.01,
        n_cycles: int = 4,
        cycle_mult: float = 1.0,  # Each cycle is cycle_mult × longer
    ):
        super().__init__(T_init, T_final)
        self.n_cycles = n_cycles
        self.cycle_mult = cycle_mult

    def forward(self, step: int, total_steps: int) -> float:
        # Compute cycle boundaries
        if self.cycle_mult == 1.0:
            cycle_length = total_steps / self.n_cycles
            cycle_idx = int(step / cycle_length)
            cycle_step = step - cycle_idx * cycle_length
        else:
            # Geometric series for cycle lengths
            r = self.cycle_mult
            total_length = (1 - r ** self.n_cycles) / (1 - r)
            base_length = total_steps / total_length

            # Find which cycle we're in
            cumulative = 0
            for i in range(self.n_cycles):
                cycle_length = base_length * (r ** i)
                if cumulative + cycle_length > step:
                    cycle_step = step - cumulative
                    break
                cumulative += cycle_length
            else:
                cycle_step = 0
                cycle_length = base_length * (r ** (self.n_cycles - 1))

        # Cosine within cycle
        cos_factor = (1 + math.cos(math.pi * cycle_step / cycle_length)) / 2
        return self.T_final + (self.T_init - self.T_final) * cos_factor


# =============================================================================
# Nosé-Hoover Integrator
# =============================================================================

class NoseHooverIntegrator(nn.Module):
    """
    Nosé-Hoover chain thermostat integrator.

    Extends the physical system with thermostat degrees of freedom
    to sample from the canonical (constant-T) ensemble.

    The extended Hamiltonian:
        H_NH = H_phys + π_ξ²/(2Q) + g·kT·ξ

    Modified equations:
        dp/dt = -∂V/∂q - ξ·p     (friction!)
        dξ/dt = π_ξ / Q
        dπ_ξ/dt = (⟨p²/m⟩ - g·kT)  (thermostat force)

    where g = number of degrees of freedom.
    """

    def __init__(
        self,
        Q: float = 1.0,           # Thermostat mass
        n_chain: int = 1,         # Chain length (1 = simple NH)
        target_temperature: float = 1.0,
    ):
        """
        Args:
            Q: Thermostat mass. Larger Q = weaker coupling to bath.
            n_chain: Number of thermostats in chain (reduces oscillations).
            target_temperature: Target thermal energy kT.
        """
        super().__init__()
        self.Q = Q
        self.n_chain = n_chain
        self.target_temperature = target_temperature

    def init_thermostat(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype
    ) -> ThermostatState:
        """Initialize thermostat state."""
        return ThermostatState(
            xi=torch.zeros(batch_size, device=device, dtype=dtype),
            pi_xi=torch.zeros(batch_size, device=device, dtype=dtype),
            Q=self.Q,
            target_temperature=self.target_temperature,
        )

    def thermostat_step(
        self,
        pi: torch.Tensor,           # (B, ...) momenta
        thermostat: ThermostatState,
        n_dof: int,                 # Number of degrees of freedom
        dt: float,
    ) -> tuple[torch.Tensor, ThermostatState]:
        """
        Single Nosé-Hoover thermostat step.

        1. Update thermostat momentum based on kinetic energy mismatch
        2. Update thermostat position
        3. Apply friction to physical momenta

        Args:
            pi: Physical momenta (any shape, will be summed for KE)
            thermostat: Current thermostat state
            n_dof: Number of degrees of freedom
            dt: Time step

        Returns:
            pi_new: Friction-modified momenta
            thermostat_new: Updated thermostat state
        """
        # Compute current kinetic energy (sum over all dimensions except batch)
        # Assuming unit mass: KE = ½ Σ p²
        KE = 0.5 * (pi ** 2).sum(dim=tuple(range(1, pi.dim())))  # (B,)

        # Target kinetic energy: ⟨KE⟩ = ½ g kT
        target_KE = 0.5 * n_dof * thermostat.target_temperature

        # Thermostat force: dπ_ξ/dt = 2(KE - target_KE)
        # Factor of 2 because ∂KE/∂p = p, and we want ⟨KE⟩ = target
        thermostat_force = 2.0 * (KE - target_KE)

        # Update thermostat momentum (half step)
        pi_xi_new = thermostat.pi_xi + 0.5 * dt * thermostat_force

        # Update thermostat position
        xi_new = thermostat.xi + dt * pi_xi_new / thermostat.Q

        # Apply friction to physical momenta
        # dp/dt = -ξ·p → p(t+dt) ≈ p(t)·exp(-ξ·dt)
        friction = torch.exp(-xi_new.view(-1, *([1] * (pi.dim() - 1))) * dt)
        pi_new = pi * friction

        # Update thermostat momentum (half step with new KE)
        KE_new = 0.5 * (pi_new ** 2).sum(dim=tuple(range(1, pi_new.dim())))
        thermostat_force_new = 2.0 * (KE_new - target_KE)
        pi_xi_new = pi_xi_new + 0.5 * dt * thermostat_force_new

        thermostat_new = ThermostatState(
            xi=xi_new,
            pi_xi=pi_xi_new,
            Q=thermostat.Q,
            target_temperature=thermostat.target_temperature,
        )

        return pi_new, thermostat_new

    def update_temperature(
        self,
        thermostat: ThermostatState,
        new_temperature: float
    ) -> ThermostatState:
        """Update target temperature (for annealing)."""
        return ThermostatState(
            xi=thermostat.xi,
            pi_xi=thermostat.pi_xi,
            Q=thermostat.Q,
            target_temperature=new_temperature,
        )


# =============================================================================
# Langevin Thermostat (Alternative)
# =============================================================================

class LangevinThermostat(nn.Module):
    """
    Langevin thermostat: stochastic dynamics with friction + noise.

    dp/dt = -∂V/∂q - γp + √(2γkT) η(t)

    where η(t) is white noise satisfying ⟨η(t)η(t')⟩ = δ(t-t').

    Properties:
        - Samples from canonical ensemble at temperature T
        - γ controls coupling strength (friction coefficient)
        - Simpler than Nosé-Hoover but breaks time-reversibility

    The fluctuation-dissipation relation ensures correct equilibrium:
        friction (γp) extracts energy
        noise (√(2γkT) η) injects energy
        at equilibrium: ⟨KE⟩ = ½ g kT
    """

    def __init__(
        self,
        gamma: float = 0.1,       # Friction coefficient
        temperature: float = 1.0,  # kT
    ):
        super().__init__()
        self.gamma = gamma
        self.temperature = temperature

    def apply(
        self,
        pi: torch.Tensor,
        dt: float,
        temperature: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Apply Langevin thermostat to momenta.

        Uses BAOAB splitting for stability:
            B: p ← p - (γ/2)·dt·p + √(γ·kT·dt)·η

        Args:
            pi: Momenta tensor
            dt: Time step
            temperature: Override default temperature

        Returns:
            pi_new: Thermostatted momenta
        """
        T = temperature if temperature is not None else self.temperature

        # Friction factor
        c1 = math.exp(-self.gamma * dt)

        # Noise scale (fluctuation-dissipation)
        c2 = math.sqrt((1 - c1 ** 2) * T) if T > 0 else 0.0

        # Apply: p_new = c1·p + c2·η
        noise = torch.randn_like(pi)
        pi_new = c1 * pi + c2 * noise

        return pi_new


# =============================================================================
# Unified Thermostat Interface
# =============================================================================

class Thermostat(nn.Module):
    """
    Unified thermostat interface supporting multiple thermostat types.

    Supports:
        - 'none': Pure Hamiltonian (energy conserved)
        - 'langevin': Stochastic friction + noise
        - 'nose_hoover': Deterministic extended dynamics
        - 'velocity_rescale': Simple velocity rescaling
    """

    def __init__(
        self,
        thermostat_type: Literal['none', 'langevin', 'nose_hoover', 'velocity_rescale'] = 'langevin',
        temperature: float = 1.0,
        gamma: float = 0.1,        # For Langevin
        Q: float = 1.0,            # For Nosé-Hoover
        annealing: Optional[AnnealingSchedule] = None,
    ):
        super().__init__()
        self.thermostat_type = thermostat_type
        self.temperature = temperature
        self.annealing = annealing

        if thermostat_type == 'langevin':
            self.impl = LangevinThermostat(gamma, temperature)
        elif thermostat_type == 'nose_hoover':
            self.impl = NoseHooverIntegrator(Q, target_temperature=temperature)
        else:
            self.impl = None

        self.current_step = 0
        self.total_steps = 1

    def set_training_length(self, total_steps: int):
        """Set total training steps for annealing schedule."""
        self.total_steps = total_steps

    def get_temperature(self) -> float:
        """Get current temperature (with annealing if enabled)."""
        if self.annealing is not None:
            return self.annealing(self.current_step, self.total_steps)
        return self.temperature

    def step(self):
        """Advance annealing schedule by one step."""
        self.current_step += 1

    def apply(
        self,
        pi: torch.Tensor,
        dt: float,
        n_dof: Optional[int] = None,
        thermostat_state: Optional[ThermostatState] = None,
    ) -> tuple[torch.Tensor, Optional[ThermostatState]]:
        """
        Apply thermostat to momenta.

        Args:
            pi: Momenta tensor
            dt: Time step
            n_dof: Degrees of freedom (for Nosé-Hoover)
            thermostat_state: Current NH state (for Nosé-Hoover)

        Returns:
            pi_new: Modified momenta
            thermostat_state_new: Updated NH state (or None)
        """
        T = self.get_temperature()

        if self.thermostat_type == 'none':
            return pi, thermostat_state

        elif self.thermostat_type == 'langevin':
            pi_new = self.impl.apply(pi, dt, T)
            return pi_new, thermostat_state

        elif self.thermostat_type == 'nose_hoover':
            if thermostat_state is None:
                thermostat_state = self.impl.init_thermostat(
                    pi.shape[0], pi.device, pi.dtype
                )
            thermostat_state = self.impl.update_temperature(thermostat_state, T)
            pi_new, thermostat_state_new = self.impl.thermostat_step(
                pi, thermostat_state, n_dof or pi.numel() // pi.shape[0], dt
            )
            return pi_new, thermostat_state_new

        elif self.thermostat_type == 'velocity_rescale':
            # Simple velocity rescaling to target temperature
            current_KE = 0.5 * (pi ** 2).sum()
            n_dof = n_dof or pi.numel()
            target_KE = 0.5 * n_dof * T
            scale = torch.sqrt(target_KE / (current_KE + 1e-10))
            return pi * scale, thermostat_state

        return pi, thermostat_state


# =============================================================================
# Testing
# =============================================================================

if __name__ == '__main__':
    # Fix OpenMP issue on Windows/Anaconda
    import os
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

    print("=" * 70)
    print("THERMOSTAT TEST")
    print("=" * 70)

    # Test annealing schedules
    print("\n[1] Testing annealing schedules...")
    schedules = [
        ("Exponential", ExponentialAnnealing(T_init=1.0, T_final=0.01)),
        ("Polynomial", PolynomialAnnealing(T_init=1.0, T_final=0.01)),
        ("Cosine", CosineAnnealing(T_init=1.0, T_final=0.01)),
        ("Cyclic", CyclicAnnealing(T_init=1.0, T_final=0.01, n_cycles=4)),
    ]

    total_steps = 100
    test_steps = [0, 25, 50, 75, 100]

    for name, schedule in schedules:
        temps = [schedule(s, total_steps) for s in test_steps]
        print(f"    {name}: {[f'{t:.3f}' for t in temps]}")

    # Test Langevin thermostat
    print("\n[2] Testing Langevin thermostat...")
    langevin = LangevinThermostat(gamma=0.5, temperature=1.0)
    pi = torch.randn(10, 100)  # 10 samples, 100 DOF

    print(f"    Initial KE: {0.5 * (pi ** 2).mean():.4f}")
    for _ in range(100):
        pi = langevin.apply(pi, dt=0.01)
    print(f"    Final KE: {0.5 * (pi ** 2).mean():.4f}")
    print(f"    Target KE: {0.5 * 1.0:.4f}")  # ½ kT per DOF

    # Test Nosé-Hoover thermostat
    print("\n[3] Testing Nosé-Hoover thermostat...")
    nh = NoseHooverIntegrator(Q=1.0, target_temperature=1.0)
    pi = torch.randn(5, 50)
    thermostat = nh.init_thermostat(5, pi.device, pi.dtype)

    print(f"    Initial KE: {0.5 * (pi ** 2).mean():.4f}")
    for _ in range(200):
        pi, thermostat = nh.thermostat_step(pi, thermostat, n_dof=50, dt=0.01)
    print(f"    Final KE: {0.5 * (pi ** 2).mean():.4f}")
    print(f"    Thermostat ξ: {thermostat.xi.mean():.4f}")

    # Test unified interface
    print("\n[4] Testing unified thermostat with annealing...")
    thermostat = Thermostat(
        thermostat_type='langevin',
        temperature=1.0,
        gamma=0.1,
        annealing=CosineAnnealing(T_init=1.0, T_final=0.01),
    )
    thermostat.set_training_length(100)

    for step in [0, 25, 50, 75, 99]:
        thermostat.current_step = step
        T = thermostat.get_temperature()
        print(f"    Step {step}: T = {T:.4f}")

    print("\n" + "=" * 70)
    print("✓ Thermostat tests complete!")
    print("=" * 70)
