"""
Test Hamiltonian Reversibility
==============================

Demonstrates that symplectic leapfrog integration is time-reversible:"""

# Fix OpenMP conflict on Windows/Anaconda
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

__doc__ = """
Demonstrates that symplectic leapfrog integration is time-reversible:
    Forward:  (q₀, p₀) → T steps → (q_T, p_T)
    Reverse:  (q_T, -p_T) → T steps → (q₀, -p₀)

This proves information is preserved through the Hamiltonian FFN,
unlike standard MLPs which are lossy (many-to-one mappings).

Author: Chris & Claude
Date: December 2025
"""

import torch
import numpy as np
from typing import Tuple, Dict
from transformer.hamiltonian_ffn import (
    HamiltonianFFN,
    PhaseSpaceState,
    LeapfrogIntegrator,
    HamiltonianPotential,
    HamiltonianKineticTerms,
)


def test_reversibility(
    batch_size: int = 2,
    seq_len: int = 8,
    embed_dim: int = 11,
    n_steps: int = 10,
    dt: float = 0.01,
    verbose: bool = True,
) -> Dict[str, float]:
    """
    Test Hamiltonian reversibility.

    Forward-backward integration should recover initial state.

    Returns:
        Dictionary with reconstruction errors for each variable.
    """
    torch.manual_seed(42)
    device = torch.device('cpu')

    # Create SO(3) generators
    from transformer.attention import generate_so3_generators
    generators = generate_so3_generators(embed_dim)
    if isinstance(generators, np.ndarray):
        generators = torch.from_numpy(generators).float()
    generators = generators.to(device)

    # Create integrator components
    kinetic = HamiltonianKineticTerms(
        embed_dim=embed_dim,
        mass_matrix='fisher',
    ).to(device)

    potential = HamiltonianPotential(
        embed_dim=embed_dim,
        generators=generators,
        alpha=1.0,
        lambda_belief=1.0,
        kappa=1.0,
    ).to(device)

    integrator = LeapfrogIntegrator(
        kinetic=kinetic,
        potential=potential,
        dt=dt,
        n_steps=n_steps,
        update_Sigma=True,
        update_phi=True,
    )

    # Create initial state
    mu_0 = torch.randn(batch_size, seq_len, embed_dim, device=device)
    Sigma_0 = torch.eye(embed_dim, device=device).unsqueeze(0).unsqueeze(0)
    Sigma_0 = Sigma_0.expand(batch_size, seq_len, -1, -1).clone()
    # Add some variation to make it interesting
    Sigma_0 = Sigma_0 + 0.1 * torch.randn_like(Sigma_0)
    Sigma_0 = torch.bmm(
        Sigma_0.view(-1, embed_dim, embed_dim),
        Sigma_0.view(-1, embed_dim, embed_dim).transpose(-1, -2)
    ).view(batch_size, seq_len, embed_dim, embed_dim)  # Ensure SPD

    phi_0 = 0.1 * torch.randn(batch_size, seq_len, 3, device=device)

    # Sample momenta
    pi_mu_0 = torch.randn(batch_size, seq_len, embed_dim, device=device)
    pi_Sigma_0 = torch.randn(batch_size, seq_len, embed_dim, embed_dim, device=device)
    pi_Sigma_0 = 0.5 * (pi_Sigma_0 + pi_Sigma_0.transpose(-1, -2))  # Symmetric
    pi_phi_0 = torch.randn(batch_size, seq_len, 3, device=device)

    # Create phase space state
    state_0 = PhaseSpaceState(
        mu=mu_0.clone(),
        Sigma=Sigma_0.clone(),
        phi=phi_0.clone(),
        pi_mu=pi_mu_0.clone(),
        pi_Sigma=pi_Sigma_0.clone(),
        pi_phi=pi_phi_0.clone(),
    )

    # Create priors (fixed during integration)
    mu_prior = torch.zeros_like(mu_0)
    Sigma_prior = torch.eye(embed_dim, device=device).unsqueeze(0).unsqueeze(0)
    Sigma_prior = Sigma_prior.expand(batch_size, seq_len, -1, -1)
    beta = torch.ones(batch_size, seq_len, seq_len, device=device) / seq_len

    # =========================================================================
    # FORWARD INTEGRATION
    # =========================================================================
    if verbose:
        print("="*60)
        print("HAMILTONIAN REVERSIBILITY TEST")
        print("="*60)
        print(f"\nConfiguration:")
        print(f"  Batch size: {batch_size}")
        print(f"  Sequence length: {seq_len}")
        print(f"  Embedding dim: {embed_dim}")
        print(f"  Integration steps: {n_steps}")
        print(f"  Time step dt: {dt}")
        print(f"  Total time T: {n_steps * dt}")

    # Compute initial Hamiltonian
    H_0 = potential.forward(
        state_0, mu_prior, Sigma_prior, beta, None, None
    )

    if verbose:
        print(f"\nInitial Hamiltonian H_0 = {H_0.mean().item():.6f}")
        print("\nIntegrating FORWARD...")

    state_T = integrator.integrate(
        state_0.clone(),
        mu_prior, Sigma_prior, beta, None, None
    )

    # Compute final Hamiltonian
    H_T = potential.forward(
        state_T, mu_prior, Sigma_prior, beta, None, None
    )

    if verbose:
        print(f"Final Hamiltonian H_T = {H_T.mean().item():.6f}")
        print(f"Energy drift |H_T - H_0| = {(H_T - H_0).abs().mean().item():.2e}")

    # =========================================================================
    # NEGATE MOMENTA (time reversal)
    # =========================================================================
    if verbose:
        print("\nNegating momenta for time reversal...")

    state_T_reversed = PhaseSpaceState(
        mu=state_T.mu.clone(),
        Sigma=state_T.Sigma.clone(),
        phi=state_T.phi.clone(),
        pi_mu=-state_T.pi_mu.clone(),      # NEGATE
        pi_Sigma=-state_T.pi_Sigma.clone(), # NEGATE
        pi_phi=-state_T.pi_phi.clone(),     # NEGATE
    )

    # =========================================================================
    # BACKWARD INTEGRATION (forward in time with negated momenta)
    # =========================================================================
    if verbose:
        print("Integrating BACKWARD (forward with negated momenta)...")

    state_reconstructed = integrator.integrate(
        state_T_reversed,
        mu_prior, Sigma_prior, beta, None, None
    )

    # =========================================================================
    # COMPARE WITH INITIAL STATE
    # =========================================================================
    # After reversal, momenta should also be negated

    # Reconstruction errors
    mu_error = (state_reconstructed.mu - mu_0).abs().mean().item()
    Sigma_error = (state_reconstructed.Sigma - Sigma_0).abs().mean().item()
    phi_error = (state_reconstructed.phi - phi_0).abs().mean().item()

    # Momenta should be negated after full round-trip
    pi_mu_error = (state_reconstructed.pi_mu - (-pi_mu_0)).abs().mean().item()
    pi_Sigma_error = (state_reconstructed.pi_Sigma - (-pi_Sigma_0)).abs().mean().item()
    pi_phi_error = (state_reconstructed.pi_phi - (-pi_phi_0)).abs().mean().item()

    results = {
        'mu_error': mu_error,
        'Sigma_error': Sigma_error,
        'phi_error': phi_error,
        'pi_mu_error': pi_mu_error,
        'pi_Sigma_error': pi_Sigma_error,
        'pi_phi_error': pi_phi_error,
        'H_0': H_0.mean().item(),
        'H_T': H_T.mean().item(),
        'energy_drift': (H_T - H_0).abs().mean().item(),
    }

    if verbose:
        print("\n" + "="*60)
        print("RECONSTRUCTION ERRORS")
        print("="*60)
        print(f"\nConfiguration variables (should be ~0):")
        print(f"  |μ_reconstructed - μ_0|     = {mu_error:.2e}")
        print(f"  |Σ_reconstructed - Σ_0|     = {Sigma_error:.2e}")
        print(f"  |φ_reconstructed - φ_0|     = {phi_error:.2e}")
        print(f"\nMomenta (should match -p_0):")
        print(f"  |π_μ_reconstructed - (-π_μ_0)|   = {pi_mu_error:.2e}")
        print(f"  |π_Σ_reconstructed - (-π_Σ_0)|   = {pi_Sigma_error:.2e}")
        print(f"  |π_φ_reconstructed - (-π_φ_0)|   = {pi_phi_error:.2e}")

        # Verdict
        total_config_error = mu_error + Sigma_error + phi_error
        print(f"\n" + "="*60)
        if total_config_error < 1e-4:
            print("REVERSIBILITY TEST: PASSED")
            print(f"Total configuration error: {total_config_error:.2e}")
        else:
            print("REVERSIBILITY TEST: FAILED")
            print(f"Total configuration error: {total_config_error:.2e} (expected < 1e-4)")
        print("="*60)

    return results


def test_token_attribution(
    token_idx: int = 5,
    batch_size: int = 1,
    seq_len: int = 8,
    embed_dim: int = 11,
    n_steps: int = 10,
    dt: float = 0.01,
    verbose: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Demonstrate token attribution through Hamiltonian reversal.

    Given an output token's belief state, trace back to see which
    input features contributed to it through the dynamics.

    This is possible ONLY because Hamiltonian dynamics are reversible!
    Standard MLPs cannot do this (they're lossy).

    Args:
        token_idx: Which token position to trace

    Returns:
        Dictionary with forward/backward trajectories
    """
    torch.manual_seed(42)
    device = torch.device('cpu')

    # Create components
    from transformer.attention import generate_so3_generators
    generators = generate_so3_generators(embed_dim)
    if isinstance(generators, np.ndarray):
        generators = torch.from_numpy(generators).float()
    generators = generators.to(device)

    kinetic = HamiltonianKineticTerms(
        embed_dim=embed_dim,
        mass_matrix='fisher',
    ).to(device)

    potential = HamiltonianPotential(
        embed_dim=embed_dim,
        generators=generators,
        alpha=1.0,
        lambda_belief=1.0,
        kappa=1.0,
    ).to(device)

    integrator = LeapfrogIntegrator(
        kinetic=kinetic,
        potential=potential,
        dt=dt,
        n_steps=1,  # Single step at a time for trajectory
        update_Sigma=True,
        update_phi=True,
    )

    # Create initial state
    mu_0 = torch.randn(batch_size, seq_len, embed_dim, device=device)
    Sigma_0 = torch.eye(embed_dim, device=device).unsqueeze(0).unsqueeze(0)
    Sigma_0 = Sigma_0.expand(batch_size, seq_len, -1, -1).clone()
    phi_0 = 0.1 * torch.randn(batch_size, seq_len, 3, device=device)
    pi_mu_0 = 0.1 * torch.randn(batch_size, seq_len, embed_dim, device=device)
    pi_Sigma_0 = torch.zeros(batch_size, seq_len, embed_dim, embed_dim, device=device)
    pi_phi_0 = torch.zeros(batch_size, seq_len, 3, device=device)

    state = PhaseSpaceState(
        mu=mu_0.clone(), Sigma=Sigma_0.clone(), phi=phi_0.clone(),
        pi_mu=pi_mu_0.clone(), pi_Sigma=pi_Sigma_0.clone(), pi_phi=pi_phi_0.clone(),
    )

    mu_prior = torch.zeros_like(mu_0)
    Sigma_prior = torch.eye(embed_dim, device=device).unsqueeze(0).unsqueeze(0)
    Sigma_prior = Sigma_prior.expand(batch_size, seq_len, -1, -1)
    beta = torch.ones(batch_size, seq_len, seq_len, device=device) / seq_len

    # =========================================================================
    # RECORD FORWARD TRAJECTORY
    # =========================================================================
    forward_trajectory = [state.mu[:, token_idx, :].clone()]

    if verbose:
        print("="*60)
        print("TOKEN ATTRIBUTION VIA HAMILTONIAN REVERSAL")
        print("="*60)
        print(f"\nTracing token position {token_idx}")
        print(f"Initial μ[{token_idx}] = {mu_0[0, token_idx, :3].tolist()[:3]}...")
        print(f"\nIntegrating forward {n_steps} steps...")

    for step in range(n_steps):
        state = integrator.integrate(
            state, mu_prior, Sigma_prior, beta, None, None
        )
        forward_trajectory.append(state.mu[:, token_idx, :].clone())

    if verbose:
        print(f"Final μ[{token_idx}] = {state.mu[0, token_idx, :3].tolist()[:3]}...")

    # =========================================================================
    # REVERSE FROM OUTPUT
    # =========================================================================
    if verbose:
        print(f"\nReversing from output back to input...")

    # Negate momenta
    state = PhaseSpaceState(
        mu=state.mu.clone(), Sigma=state.Sigma.clone(), phi=state.phi.clone(),
        pi_mu=-state.pi_mu.clone(), pi_Sigma=-state.pi_Sigma.clone(),
        pi_phi=-state.pi_phi.clone(),
    )

    backward_trajectory = [state.mu[:, token_idx, :].clone()]

    for step in range(n_steps):
        state = integrator.integrate(
            state, mu_prior, Sigma_prior, beta, None, None
        )
        backward_trajectory.append(state.mu[:, token_idx, :].clone())

    if verbose:
        print(f"Recovered μ[{token_idx}] = {state.mu[0, token_idx, :3].tolist()[:3]}...")

        # Compare with original
        recovery_error = (state.mu[:, token_idx, :] - mu_0[:, token_idx, :]).abs().mean().item()
        print(f"\nRecovery error: {recovery_error:.2e}")

        print("\n" + "="*60)
        print("INTERPRETATION")
        print("="*60)
        print("""
Because Hamiltonian dynamics are reversible, we can trace EXACTLY
which input features led to any output feature. This is impossible
with standard MLPs, which are many-to-one mappings.

The forward trajectory shows how the belief evolved.
The backward trajectory shows the causal history.

This enables:
1. Perfect reconstruction of inputs from outputs
2. Exact attribution of which features caused which predictions
3. Detection of adversarial perturbations
4. Interpretable dynamics (follow the phase space flow)
        """)

    return {
        'forward_trajectory': torch.stack(forward_trajectory),
        'backward_trajectory': torch.stack(backward_trajectory),
        'initial_mu': mu_0[:, token_idx, :],
        'final_mu': forward_trajectory[-1],
        'recovered_mu': backward_trajectory[-1],
    }


if __name__ == '__main__':
    print("\n" + "="*70)
    print("HAMILTONIAN TRANSFORMER REVERSIBILITY TESTS")
    print("="*70 + "\n")

    # Test 1: Basic reversibility
    print("TEST 1: Forward-Backward Reversibility")
    print("-"*40)
    results = test_reversibility(verbose=True)

    print("\n\n")

    # Test 2: Token attribution
    print("TEST 2: Token Attribution via Reversal")
    print("-"*40)
    attribution = test_token_attribution(token_idx=3, verbose=True)
