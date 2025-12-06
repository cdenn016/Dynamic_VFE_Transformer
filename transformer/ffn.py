"""
Feed-Forward Networks for Gauge Transformer
===========================================

Supports FIVE modes:
1. LEARNED: Standard FFN with learned weights (default)
2. VARIATIONAL_APPROX: Approximate variational descent (legacy, μ only, no ∂β/∂μ)
3. VARIATIONAL_FULL: Full variational descent (legacy, μ only, with ∂β/∂μ)
4. VARIATIONAL_GRADIENT_ENGINE: Full active inference via validated gradient_engine.py
   - Updates both μ AND Σ
   - Natural gradients via Fisher-Rao metric
   - All energy terms (self-coupling, alignment, observations, softmax coupling)
   - Theoretically principled and validated!
5. HAMILTONIAN: Symplectic Hamiltonian dynamics on belief space (NEW!)
   - Energy-conserving dynamics via leapfrog integration
   - Full faithful SPD geometry with curvature corrections
   - Phase space: (μ, Σ, φ, π_μ, π_Σ, π_φ)
   - H = T + V where V is the free energy functional

Author: Extended architecture with gradient_engine and Hamiltonian integration
Date: December 2025
"""

import torch
import torch.nn as nn
from typing import Optional, Literal, Tuple, Union

from transformer.variational_ffn import (
    VariationalFFNApproximate,
    VariationalFFNFull,
    VariationalFFNGradientEngine
)
from transformer.hamiltonian_ffn import HamiltonianFFN


class GaugeFFN(nn.Module):
    """
    Unified FFN module supporting learned, variational, and Hamiltonian modes.

    Modes:
        'learned': Standard MLP (default)
        'variational_approx': Approximate variational descent (legacy)
        'variational_full': Full variational descent (legacy)
        'variational_gradient_engine': Full active inference
        'hamiltonian': Symplectic Hamiltonian dynamics (NEW!)

    Switch via mode parameter or at runtime.
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        generators: Optional[torch.Tensor] = None,  # (3, K, K)
        dropout: float = 0.1,
        mode: Literal['learned', 'variational_approx', 'variational_full', 'variational_gradient_engine', 'hamiltonian'] = 'learned',
        # Variational parameters
        alpha: float = 0.001,
        tau_eff: float = 1.0,
        kappa: float = 1.0,
        n_iterations: int = 1,
        learnable_lr: bool = True,
        # Gradient engine specific
        lambda_belief: float = 1.0,
        lambda_prior: float = 0.0,
        lambda_phi: float = 0.0,
        update_sigma: bool = True,
        # Hamiltonian specific
        hamiltonian_dt: float = 0.01,
        hamiltonian_n_steps: int = 10,
        hamiltonian_update_phi: bool = False,
        hamiltonian_momentum_scale: float = 1.0,
        hamiltonian_gamma: float = 0.0,  # Damping (0 = pure Hamiltonian)
    ):
        """
        Initialize unified FFN.

        Args:
            embed_dim: K
            hidden_dim: Hidden layer size (for learned mode)
            generators: SO(3) generators (required for variational/hamiltonian modes)
            dropout: Dropout rate (for learned mode)
            mode: 'learned', 'variational_approx', 'variational_full', 'variational_gradient_engine', 'hamiltonian'
            alpha: Prior weight (variational/hamiltonian)
            tau_eff: Temperature (variational approx/full)
            kappa: Softmax temperature (variational_full/hamiltonian)
            n_iterations: Inference steps (variational)
            learnable_lr: Learn step size? (variational)
            lambda_belief: Belief alignment weight (gradient_engine/hamiltonian)
            lambda_prior: Prior alignment weight (gradient_engine)
            lambda_phi: Gauge field weight (gradient_engine)
            update_sigma: Update covariances? (gradient_engine/hamiltonian)
            hamiltonian_dt: Time step for leapfrog integration
            hamiltonian_n_steps: Number of leapfrog steps per forward pass
            hamiltonian_update_phi: Evolve gauge field in Hamiltonian dynamics?
            hamiltonian_momentum_scale: Scale for initial momentum sampling
            hamiltonian_gamma: Damping coefficient (0 = pure Hamiltonian, >0 = Langevin-like)
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.mode = mode

        # =================================================================
        # Learned FFN (standard transformer)
        # =================================================================
        self.learned_ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

        # =================================================================
        # Variational FFNs (active inference)
        # =================================================================
        if mode in ['variational_approx', 'variational_full', 'variational_gradient_engine']:
            if generators is None:
                raise ValueError("generators required for variational modes")

            if mode == 'variational_approx':
                self.variational_ffn = VariationalFFNApproximate(
                    embed_dim=embed_dim,
                    generators=generators,
                    alpha=alpha,
                    tau_eff=tau_eff,
                    n_iterations=n_iterations,
                    learnable_lr=learnable_lr,
                )
            elif mode == 'variational_full':
                self.variational_ffn = VariationalFFNFull(
                    embed_dim=embed_dim,
                    generators=generators,
                    alpha=alpha,
                    tau_eff=tau_eff,
                    kappa=kappa,
                    n_iterations=n_iterations,
                    learnable_lr=learnable_lr,
                )
            else:  # variational_gradient_engine
                self.variational_ffn = VariationalFFNGradientEngine(
                    embed_dim=embed_dim,
                    generators=generators,
                    alpha=alpha,
                    lambda_belief=lambda_belief,
                    lambda_prior=lambda_prior,
                    lambda_phi=lambda_phi,
                    kappa_beta=kappa,
                    n_iterations=n_iterations,
                    learnable_lr=learnable_lr,
                    update_sigma=update_sigma,
                )

        # =================================================================
        # Hamiltonian FFN (symplectic dynamics)
        # =================================================================
        if mode == 'hamiltonian':
            if generators is None:
                raise ValueError("generators required for hamiltonian mode")

            self.hamiltonian_ffn = HamiltonianFFN(
                embed_dim=embed_dim,
                generators=generators,
                n_leapfrog_steps=hamiltonian_n_steps,
                dt=hamiltonian_dt,
                alpha=alpha,
                lambda_belief=lambda_belief,
                kappa=kappa,
                update_Sigma=update_sigma,
                update_phi=hamiltonian_update_phi,
                momentum_scale=hamiltonian_momentum_scale,
                gamma=hamiltonian_gamma,
            )

    def forward(
        self,
        mu: torch.Tensor,          # (B, N, K) - always required
        # Variational/Hamiltonian inputs (optional for learned mode)
        beta: Optional[torch.Tensor] = None,      # (B, n_heads, N, N) or (B, N, N)
        mu_prior: Optional[torch.Tensor] = None,  # (B, N, K)
        phi: Optional[torch.Tensor] = None,       # (B, N, 3)
        sigma: Optional[torch.Tensor] = None,     # (B, N, K, K)
        sigma_prior: Optional[torch.Tensor] = None,  # (B, N, K, K) - for Hamiltonian
        mask: Optional[torch.Tensor] = None,      # (B, N, N)
        # Observation inputs (for gradient_engine/hamiltonian E-step)
        targets: Optional[torch.Tensor] = None,   # (B, N) - target tokens
        W_out: Optional[torch.Tensor] = None,     # (V, K) - output projection
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]]:
        """
        Forward pass - dispatches to appropriate FFN.

        Args:
            mu: Current beliefs (always required)
            beta: Attention weights (for variational/hamiltonian)
            mu_prior: Embedding priors (for variational/hamiltonian)
            phi: Gauge frames (for variational/hamiltonian)
            sigma: Covariances (for variational_full/hamiltonian)
            sigma_prior: Prior covariances (for hamiltonian - mass matrix)
            mask: Causal mask (for variational)
            targets: Target token IDs (for gradient_engine/hamiltonian E-step)
            W_out: Output projection matrix (for computing CE gradient)

        Returns:
            - 'learned': mu_out (B, N, K)
            - 'variational_*': mu_out (B, N, K)
            - 'variational_gradient_engine': (mu_out, sigma_out)
            - 'hamiltonian': (mu_out, sigma_out, phi_out, diagnostics)
        """
        if self.mode == 'learned':
            # Standard learned FFN
            return self.learned_ffn(mu)

        elif self.mode == 'variational_approx':
            # Check required inputs
            if beta is None or mu_prior is None or phi is None:
                raise ValueError("variational_approx requires beta, mu_prior, phi")

            return self.variational_ffn(
                mu=mu,
                beta=beta,
                mu_prior=mu_prior,
                phi=phi,
                mask=mask,
            )

        elif self.mode == 'variational_full':
            # Check required inputs
            if beta is None or mu_prior is None or phi is None:
                raise ValueError("variational_full requires beta, mu_prior, phi")

            return self.variational_ffn(
                mu=mu,
                beta=beta,
                mu_prior=mu_prior,
                phi=phi,
                sigma=sigma,
                mask=mask,
            )

        elif self.mode == 'variational_gradient_engine':
            # Check required inputs
            if beta is None or mu_prior is None or phi is None:
                raise ValueError("variational_gradient_engine requires beta, mu_prior, phi")

            # Gradient engine returns (mu, sigma) tuple
            # E-STEP: Minimize full F including DISCRETE observations (cross-entropy)
            mu_out, sigma_out = self.variational_ffn(
                mu=mu,
                beta=beta,
                mu_prior=mu_prior,
                phi=phi,
                sigma=sigma,
                mask=mask,
                targets=targets,  # Target tokens as DISCRETE observations!
                W_out=W_out,      # Output projection for computing ∂CE/∂μ
            )
            # Return BOTH mu and sigma (full Gaussian updates!)
            return (mu_out, sigma_out)

        elif self.mode == 'hamiltonian':
            # Check required inputs
            if mu_prior is None or phi is None or sigma is None:
                raise ValueError("hamiltonian mode requires mu_prior, phi, sigma")

            # Use prior covariance as mass matrix if not provided
            if sigma_prior is None:
                # Default: identity prior (unit mass)
                B, N, K = mu.shape
                sigma_prior = torch.eye(K, device=mu.device, dtype=mu.dtype).unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1)

            # Hamiltonian dynamics returns (mu, sigma, phi, diagnostics)
            # Symplectic integration preserving energy!
            mu_out, sigma_out, phi_out, diagnostics = self.hamiltonian_ffn(
                mu=mu,
                Sigma=sigma,
                phi=phi,
                mu_prior=mu_prior,
                Sigma_prior=sigma_prior,
                beta=beta,          # Attention weights (optional)
                targets=targets,    # Target tokens for CE term
                W_out=W_out,        # Output projection
            )
            # Return full phase space update with diagnostics
            return (mu_out, sigma_out, phi_out, diagnostics)

        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def set_mode(self, mode: str):
        """Switch FFN mode at runtime."""
        valid_modes = ['learned', 'variational_approx', 'variational_full', 'variational_gradient_engine', 'hamiltonian']
        if mode not in valid_modes:
            raise ValueError(f"Invalid mode: {mode}. Valid modes: {valid_modes}")

        if mode in ['variational_approx', 'variational_full', 'variational_gradient_engine']:
            if not hasattr(self, 'variational_ffn'):
                raise ValueError(f"Mode {mode} not initialized")

        if mode == 'hamiltonian':
            if not hasattr(self, 'hamiltonian_ffn'):
                raise ValueError("Hamiltonian mode not initialized")

        self.mode = mode

    def get_mode(self) -> str:
        """Get current FFN mode."""
        return self.mode

    def get_hamiltonian_diagnostics(self) -> Optional[dict]:
        """Get diagnostics from last Hamiltonian forward pass."""
        if hasattr(self, 'hamiltonian_ffn') and hasattr(self.hamiltonian_ffn, 'last_diagnostics'):
            return self.hamiltonian_ffn.last_diagnostics
        return None


# =============================================================================
# Convenience functions
# =============================================================================

def create_ffn(
    embed_dim: int,
    hidden_dim: int,
    generators: Optional[torch.Tensor] = None,
    mode: str = 'learned',
    **kwargs
) -> GaugeFFN:
    """
    Factory function for creating FFN with correct mode.

    Example:
        >>> # Learned FFN (standard)
        >>> ffn = create_ffn(embed_dim=11, hidden_dim=44, mode='learned')

        >>> # Variational FFN (approximate)
        >>> ffn = create_ffn(
        ...     embed_dim=11, hidden_dim=44, mode='variational_approx',
        ...     generators=generators, alpha=0.001
        ... )
    """
    return GaugeFFN(
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        generators=generators,
        mode=mode,
        **kwargs
    )


def convert_to_variational(
    ffn_module: GaugeFFN,
    mode: Literal['variational_approx', 'variational_full'],
    generators: torch.Tensor,
    **kwargs
) -> GaugeFFN:
    """
    Convert existing learned FFN to variational mode.

    Useful for:
    - Ablation studies
    - Progressive training (learned → variational)
    - Comparison experiments

    Args:
        ffn_module: Existing GaugeFFN module
        mode: Target variational mode
        generators: SO(3) generators
        **kwargs: Variational parameters

    Returns:
        Same module, now with variational mode initialized and active
    """
    # Initialize variational FFN
    if mode == 'variational_approx':
        ffn_module.variational_ffn = VariationalFFNApproximate(
            embed_dim=ffn_module.embed_dim,
            generators=generators,
            **kwargs
        )
    elif mode == 'variational_full':
        ffn_module.variational_ffn = VariationalFFNFull(
            embed_dim=ffn_module.embed_dim,
            generators=generators,
            **kwargs
        )
    else:
        raise ValueError(f"Invalid mode: {mode}")

    # Switch to variational mode
    ffn_module.set_mode(mode)

    return ffn_module