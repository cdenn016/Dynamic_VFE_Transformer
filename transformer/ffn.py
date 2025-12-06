"""
Feed-Forward Networks for Gauge Transformer
===========================================

Supports FOUR modes:
1. LEARNED: Standard FFN with learned weights (default)
2. VARIATIONAL_APPROX: Approximate variational descent (legacy, μ only, no ∂β/∂μ)
3. VARIATIONAL_FULL: Full variational descent (legacy, μ only, with ∂β/∂μ)
4. VARIATIONAL_GRADIENT_ENGINE: Full active inference via validated gradient_engine.py (RECOMMENDED!)
   - Updates both μ AND Σ
   - Natural gradients via Fisher-Rao metric
   - All energy terms (self-coupling, alignment, observations, softmax coupling)
   - Theoretically principled and validated!

Author: Extended architecture with gradient_engine integration
Date: November 2025
"""

import torch
import torch.nn as nn
from typing import Optional, Literal

from transformer.variational_ffn import (
    VariationalFFNApproximate,
    VariationalFFNFull,
    VariationalFFNGradientEngine
)


class GaugeFFN(nn.Module):
    """
    Unified FFN module supporting learned and variational modes.

    Modes:
        'learned': Standard MLP (default)
        'variational_approx': Approximate variational descent (legacy)
        'variational_full': Full variational descent (legacy)
        'variational_gradient_engine': Full active inference (RECOMMENDED!)

    Switch via mode parameter or at runtime.
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        generators: Optional[torch.Tensor] = None,  # (3, K, K)
        dropout: float = 0.1,
        mode: Literal['learned', 'variational_approx', 'variational_full', 'variational_gradient_engine'] = 'learned',
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
    ):
        """
        Initialize unified FFN.

        Args:
            embed_dim: K
            hidden_dim: Hidden layer size (for learned mode)
            generators: SO(3) generators (required for variational modes)
            dropout: Dropout rate (for learned mode)
            mode: 'learned', 'variational_approx', 'variational_full', 'variational_gradient_engine'
            alpha: Prior weight (variational)
            tau_eff: Temperature (variational approx/full)
            kappa: Softmax temperature (variational_full)
            n_iterations: Inference steps (variational)
            learnable_lr: Learn step size? (variational)
            lambda_belief: Belief alignment weight (gradient_engine)
            lambda_prior: Prior alignment weight (gradient_engine)
            lambda_phi: Gauge field weight (gradient_engine)
            update_sigma: Update covariances? (gradient_engine)
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

    def forward(
        self,
        mu: torch.Tensor,          # (B, N, K) - always required
        # Variational-specific inputs (optional for learned mode)
        beta: Optional[torch.Tensor] = None,      # (B, n_heads, N, N)
        mu_prior: Optional[torch.Tensor] = None,  # (B, N, K)
        phi: Optional[torch.Tensor] = None,       # (B, N, 3)
        sigma: Optional[torch.Tensor] = None,     # (B, N, K, K)
        mask: Optional[torch.Tensor] = None,      # (B, N, N)
        # Observation inputs (for gradient_engine E-step)
        targets: Optional[torch.Tensor] = None,   # (B, N) - target tokens
        W_out: Optional[torch.Tensor] = None,     # (V, K) - output projection
    ) -> torch.Tensor:
        """
        Forward pass - dispatches to appropriate FFN.

        Args:
            mu: Current beliefs (always required)
            beta: Attention weights (for variational)
            mu_prior: Embedding priors (for variational)
            phi: Gauge frames (for variational)
            sigma: Covariances (for variational_full)
            mask: Causal mask (for variational)
            targets: Target token IDs (for gradient_engine E-step - discrete observations)
            W_out: Output projection matrix (for computing CE gradient in E-step)

        Returns:
            mu_out: Transformed beliefs (B, N, K)
                    OR (mu_out, sigma_out) for gradient_engine
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

        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def set_mode(self, mode: str):
        """Switch FFN mode at runtime."""
        if mode not in ['learned', 'variational_approx', 'variational_full', 'variational_gradient_engine']:
            raise ValueError(f"Invalid mode: {mode}")

        if mode in ['variational_approx', 'variational_full', 'variational_gradient_engine']:
            if not hasattr(self, 'variational_ffn'):
                raise ValueError(f"Mode {mode} not initialized")

        self.mode = mode

    def get_mode(self) -> str:
        """Get current FFN mode."""
        return self.mode


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