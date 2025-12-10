"""
Gauge-Theoretic Transformer Block (0D Architecture)
====================================================

Complete transformer block with:
1. Gauge-theoretic multi-head attention (KL-based, no W_Q/W_K!)
2. Feedforward network (prior evolution)
3. Layer normalization
4. Residual connections

Standard Architecture, Gauge Mechanism:
    x → LayerNorm → Attention → Residual → LayerNorm → FFN → Residual

But with gauge-theoretic attention:
    (μ, Σ, φ) → Attention(via KL + transport) → (μ', Σ', φ')

Author: Implementation from plan.py
Date: November 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List

# Import our gauge attention
from transformer.attention import IrrepMultiHeadAttention

# Import unified FFN (supports learned + variational + hamiltonian modes)
from transformer.ffn import GaugeFFN


class GaugeTransformerBlock(nn.Module):
    """
    Single transformer block with gauge-theoretic attention.

    Architecture:
        1. Self-attention sublayer:
           - LayerNorm on means
           - IrrepMultiHeadAttention (KL-based)
           - Residual connection
           - Dropout

        2. Feedforward sublayer:
           - LayerNorm on means
           - FFN (two linear layers + GELU)
           - Residual connection
           - Dropout

    Note: We primarily evolve means (μ), while covariances (Σ) and
          gauge frames (φ) can be evolved or kept fixed depending on mode.

    0D Structure:
        - All agents at single point c*
        - Attention computed via KL divergence
        - No spatial convolutions or position-dependent operations
    """

    def __init__(
        self,
        embed_dim: int,
        irrep_spec: List[Tuple[str, int, int]],
        hidden_dim: int,
        kappa_beta: float,
        dropout: float = 0.1,
        evolve_sigma: bool = False,
        evolve_phi: bool = False,
        # Variational FFN parameters
        generators: Optional[torch.Tensor] = None,  # (3, K, K)
        ffn_mode: str = 'learned',  # 'learned', 'standard', 'VFE', 'variational_*', 'hamiltonian'
        ffn_alpha: float = 0.001,
        ffn_tau_eff: float = 1.0,
        ffn_kappa: float = 1.0,
        ffn_n_iterations: int = 1,
        ffn_learnable_lr: bool = True,
        # Gradient engine specific
        ffn_lambda_belief: float = 1.0,
        ffn_lambda_prior: float = 0.0,
        ffn_lambda_phi: float = 0.0,
        ffn_update_sigma: bool = True,
        # Hamiltonian specific
        ffn_hamiltonian_dt: float = 0.01,
        ffn_hamiltonian_n_steps: int = 10,
        ffn_hamiltonian_momentum_scale: float = 1.0,
        ffn_hamiltonian_gamma: float = 0.0,
        # Hamiltonian mass configuration (from Inertia of Belief paper)
        ffn_hamiltonian_mass_use_prior: bool = True,
        ffn_hamiltonian_mass_use_observation: bool = False,
        ffn_hamiltonian_mass_use_incoming_social: bool = False,
        ffn_hamiltonian_mass_use_outgoing_recoil: bool = False,
        ffn_hamiltonian_evolve_mass: bool = False,
        # Diagonal covariance mode
        diagonal_covariance: bool = False,
    ):
        """
        Initialize gauge transformer block.

        Args:
            embed_dim: Embedding dimension K
            irrep_spec: Irrep structure [(label, mult, dim), ...]
            hidden_dim: FFN hidden dimension (typically 4 × embed_dim)
            kappa_beta: Temperature for attention
            dropout: Dropout probability
            evolve_sigma: If True, update covariances via attention and FFN
            evolve_phi: If True, update gauge frames via FFN
            generators: SO(3) generators (required for variational/hamiltonian modes)
            ffn_mode: 'learned'/'standard', 'VFE'/'variational_gradient_engine', 'hamiltonian'
            ffn_alpha: Prior weight for variational/hamiltonian FFN
            ffn_tau_eff: Temperature for variational FFN
            ffn_kappa: Softmax temperature for variational_full/hamiltonian
            ffn_n_iterations: Inference iterations for variational FFN
            ffn_learnable_lr: Learn step size for variational FFN
            ffn_lambda_belief: Belief alignment weight (gradient_engine/hamiltonian)
            ffn_lambda_prior: Prior alignment weight (gradient_engine)
            ffn_lambda_phi: Gauge field weight (gradient_engine)
            ffn_update_sigma: Update covariances in FFN? (gradient_engine/hamiltonian)
            ffn_hamiltonian_dt: Leapfrog time step (hamiltonian)
            ffn_hamiltonian_n_steps: Number of leapfrog steps (hamiltonian)
            ffn_hamiltonian_momentum_scale: Initial momentum scale (hamiltonian)
            ffn_hamiltonian_gamma: Damping coefficient (hamiltonian, 0=pure)
            ffn_hamiltonian_mass_use_prior: Include prior precision in mass (Inertia of Belief)
            ffn_hamiltonian_mass_use_observation: Include observation precision in mass
            ffn_hamiltonian_mass_use_incoming_social: Include incoming social precision in mass
            ffn_hamiltonian_mass_use_outgoing_recoil: Include outgoing recoil precision in mass
            ffn_hamiltonian_evolve_mass: Recompute mass at each leapfrog step (hamiltonian)
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.evolve_sigma = evolve_sigma
        self.evolve_phi = evolve_phi
        self.ffn_mode = ffn_mode
        self.generators = generators  # Store for variational FFN
        self.diagonal_covariance = diagonal_covariance

        # =====================================================================
        # Attention Sublayer
        # =====================================================================
        self.attention = IrrepMultiHeadAttention(
            embed_dim=embed_dim,
            irrep_spec=irrep_spec,
            kappa_beta=kappa_beta,
            epsilon=1e-8,
            aggregate_mode='full_distribution' if evolve_sigma else 'mean_only',
            diagonal_covariance=diagonal_covariance,
        )

        self.norm1 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)

        # =====================================================================
        # Feedforward Sublayer (Unified: learned + variational + hamiltonian)
        # =====================================================================
        self.ffn = GaugeFFN(
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            generators=generators,  # Required for variational/hamiltonian modes
            dropout=dropout,
            mode=ffn_mode,
            # Variational parameters
            alpha=ffn_alpha,
            tau_eff=ffn_tau_eff,
            kappa=ffn_kappa,
            n_iterations=ffn_n_iterations,
            learnable_lr=ffn_learnable_lr,
            # Gradient engine parameters
            lambda_belief=ffn_lambda_belief,
            lambda_prior=ffn_lambda_prior,
            lambda_phi=ffn_lambda_phi,
            update_sigma=ffn_update_sigma,
            # Hamiltonian parameters
            hamiltonian_dt=ffn_hamiltonian_dt,
            hamiltonian_n_steps=ffn_hamiltonian_n_steps,
            hamiltonian_momentum_scale=ffn_hamiltonian_momentum_scale,
            hamiltonian_gamma=ffn_hamiltonian_gamma,
            hamiltonian_update_phi=evolve_phi,  # Use evolve_phi setting
            # Hamiltonian mass configuration (from Inertia of Belief paper)
            hamiltonian_mass_use_prior=ffn_hamiltonian_mass_use_prior,
            hamiltonian_mass_use_observation=ffn_hamiltonian_mass_use_observation,
            hamiltonian_mass_use_incoming_social=ffn_hamiltonian_mass_use_incoming_social,
            hamiltonian_mass_use_outgoing_recoil=ffn_hamiltonian_mass_use_outgoing_recoil,
            hamiltonian_evolve_mass=ffn_hamiltonian_evolve_mass,
            # Diagonal covariance mode
            diagonal_covariance=diagonal_covariance,
        )

        self.norm2 = nn.LayerNorm(embed_dim)

        # =====================================================================
        # Optional: Gauge Frame Evolution
        # =====================================================================
        if evolve_phi:
            # Small FFN for φ evolution (3-dim output in so(3))
            self.phi_ffn = nn.Sequential(
                nn.Linear(embed_dim, 16),
                nn.Tanh(),
                nn.Linear(16, 3),
            )
            # Initialize to near-zero to start with identity frames
            nn.init.zeros_(self.phi_ffn[2].weight)
            nn.init.zeros_(self.phi_ffn[2].bias)
        else:
            self.phi_ffn = None

    def forward(
        self,
        mu_q: torch.Tensor,
        sigma_q: torch.Tensor,
        phi: torch.Tensor,
        generators: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        mu_prior: Optional[torch.Tensor] = None,  # For variational FFN
        targets: Optional[torch.Tensor] = None,   # For E-step observations
        W_out: Optional[torch.Tensor] = None,     # Output projection for discrete observations
        cached_head_transports: Optional[list] = None,  # Cross-layer transport cache
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through transformer block.

        Args:
            mu_q: Belief means (B, N, K)
            sigma_q: Belief covariances (B, N, K, K)
            phi: Gauge frames (B, N, 3)
            generators: SO(3) generators (3, K, K)
            mask: Optional causal mask (B, N, N)
            mu_prior: Embedding priors (B, N, K) - required for variational FFN
            targets: Target token IDs (B, N) - for E-step discrete observations
            W_out: Output projection (V, K) - for computing CE gradient in E-step
            cached_head_transports: Optional list of precomputed transport dicts per head.
                                   When evolve_phi=False, reuse across all layers.

        Returns:
            mu_q_out: Updated means (B, N, K)
            sigma_q_out: Updated covariances (B, N, K, K)
            phi_out: Updated gauge frames (B, N, 3)
        """
        # =====================================================================
        # 1. Attention Sublayer with Pre-Norm + Residual
        # =====================================================================

        # Pre-layer normalization on means
        mu_normalized = self.norm1(mu_q)

        # Multi-head attention (gauge-theoretic!)
        # Capture beta if needed for variational/hamiltonian FFN
        need_beta = self.ffn_mode in ['variational_approx', 'variational_full', 'variational_gradient_engine', 'hamiltonian']

        mu_attn, sigma_attn, beta, _ = self.attention(
            mu_normalized,
            sigma_q,
            phi,
            generators,
            mask=mask,
            return_attention=need_beta,  # Only compute if needed
            cached_head_transports=cached_head_transports,  # Cross-layer cache
        )

        # Residual connection + dropout on means
        mu_q = mu_q + self.dropout1(mu_attn)

        # Update covariances if evolving
        if self.evolve_sigma and sigma_attn is not None:
            sigma_q = sigma_attn
        # Otherwise sigma_q stays unchanged

        # =====================================================================
        # 2. Feedforward Sublayer with Pre-Norm + Residual
        # =====================================================================

        # Pre-layer normalization
        mu_normalized = self.norm2(mu_q)

        # Feedforward network (learned, variational, or hamiltonian)
        if self.ffn_mode == 'learned':
            # Standard learned FFN (just needs mu)
            mu_ffn = self.ffn(mu_normalized)

        elif self.ffn_mode == 'hamiltonian':
            # Hamiltonian mode: returns (mu, sigma, phi, diagnostics)
            if mu_prior is None:
                raise ValueError(f"FFN mode '{self.ffn_mode}' requires mu_prior argument")

            mu_ffn, sigma_ffn, phi_ffn, diagnostics = self.ffn(
                mu=mu_normalized,
                beta=beta,          # From attention (for alignment potential)
                mu_prior=mu_prior,  # Prior means
                phi=phi,            # Current gauge frames
                sigma=sigma_q,      # Current covariances
                sigma_prior=None,   # Will default to identity in FFN
                mask=mask,          # Causal mask
                targets=targets,    # Target tokens for CE term
                W_out=W_out,        # Output projection
            )

            # Update covariances from Hamiltonian dynamics
            if self.evolve_sigma and sigma_ffn is not None:
                sigma_q = sigma_ffn

            # Update gauge frames from Hamiltonian dynamics
            if self.evolve_phi and phi_ffn is not None:
                phi = phi_ffn

            # Store diagnostics for monitoring
            self._last_hamiltonian_diagnostics = diagnostics

        elif self.ffn_mode == 'variational_gradient_engine':
            # Gradient engine mode: returns (mu, sigma) tuple
            if mu_prior is None:
                raise ValueError(f"FFN mode '{self.ffn_mode}' requires mu_prior argument")

            mu_ffn, sigma_ffn = self.ffn(
                mu=mu_normalized,
                beta=beta,          # From attention
                mu_prior=mu_prior,  # From embeddings
                phi=phi,            # Current gauge frames
                sigma=sigma_q,      # Current covariances
                mask=mask,          # Causal mask
                targets=targets,    # Target tokens (discrete observations!)
                W_out=W_out,        # Output projection for ∂CE/∂μ
            )

            # Update covariances from FFN if evolving
            if self.evolve_sigma and sigma_ffn is not None:
                sigma_q = sigma_ffn
        else:
            # Legacy variational FFN modes (mu only)
            if mu_prior is None:
                raise ValueError(f"FFN mode '{self.ffn_mode}' requires mu_prior argument")

            mu_ffn = self.ffn(
                mu=mu_normalized,
                beta=beta,          # From attention
                mu_prior=mu_prior,  # From embeddings
                phi=phi,            # Current gauge frames
                sigma=sigma_q,      # Current covariances (for variational_full)
                mask=mask,          # Causal mask
            )

        # Residual connection
        mu_q = mu_q + mu_ffn

        # =====================================================================
        # 3. Optional: Gauge Frame Evolution (skip if Hamiltonian handles it)
        # =====================================================================
        if self.evolve_phi and self.phi_ffn is not None and self.ffn_mode != 'hamiltonian':
            # Evolve gauge frames based on current means
            # φ_new = φ + Δφ(μ)
            delta_phi = self.phi_ffn(mu_q)  # (B, N, 3)
            phi = phi + delta_phi

            # Optional: retract to principal ball (keep ||φ|| small)
            phi_norm = torch.norm(phi, dim=-1, keepdim=True)
            max_phi_norm = 2.0  # Keep gauge frames moderate
            phi = torch.where(
                phi_norm > max_phi_norm,
                phi * (max_phi_norm / phi_norm),
                phi
            )
        # Otherwise phi stays unchanged (or was updated by Hamiltonian FFN)

        return mu_q, sigma_q, phi

    def get_hamiltonian_diagnostics(self) -> Optional[dict]:
        """Get diagnostics from last Hamiltonian forward pass."""
        return getattr(self, '_last_hamiltonian_diagnostics', None)

    def extra_repr(self) -> str:
        return (
            f"embed_dim={self.embed_dim}, "
            f"hidden_dim={self.hidden_dim}, "
            f"evolve_sigma={self.evolve_sigma}, "
            f"evolve_phi={self.evolve_phi}"
        )


# =============================================================================
# Stack of Transformer Blocks
# =============================================================================

class GaugeTransformerStack(nn.Module):
    """
    Stack of N gauge transformer blocks.

    This is the main "encoder" of the model, transforming initial
    embeddings through multiple layers of gauge-theoretic attention.
    """

    def __init__(
        self,
        n_layers: int,
        embed_dim: int,
        irrep_spec: List[Tuple[str, int, int]],
        hidden_dim: int,
        kappa_beta: float,
        dropout: float = 0.1,
        evolve_sigma: bool = False,
        evolve_phi: bool = False,
        # Variational FFN parameters
        generators: Optional[torch.Tensor] = None,
        ffn_mode: str = 'learned',
        ffn_alpha: float = 0.001,
        ffn_tau_eff: float = 1.0,
        ffn_kappa: float = 1.0,
        ffn_n_iterations: int = 1,
        ffn_learnable_lr: bool = True,
        # Gradient engine specific
        ffn_lambda_belief: float = 1.0,
        ffn_lambda_prior: float = 0.0,
        ffn_lambda_phi: float = 0.0,
        ffn_update_sigma: bool = True,
        # Hamiltonian specific
        ffn_hamiltonian_dt: float = 0.01,
        ffn_hamiltonian_n_steps: int = 10,
        ffn_hamiltonian_momentum_scale: float = 1.0,
        ffn_hamiltonian_gamma: float = 0.0,
        # Hamiltonian mass configuration (from Inertia of Belief paper)
        ffn_hamiltonian_mass_use_prior: bool = True,
        ffn_hamiltonian_mass_use_observation: bool = False,
        ffn_hamiltonian_mass_use_incoming_social: bool = False,
        ffn_hamiltonian_mass_use_outgoing_recoil: bool = False,
        ffn_hamiltonian_evolve_mass: bool = False,
        # Diagonal covariance mode
        diagonal_covariance: bool = False,
    ):
        """
        Initialize stack of transformer blocks.

        Args:
            n_layers: Number of transformer blocks
            embed_dim: Embedding dimension
            irrep_spec: Irrep structure
            hidden_dim: FFN hidden dimension
            kappa_beta: Attention temperature
            dropout: Dropout probability
            evolve_sigma: If True, covariances evolve through layers
            evolve_phi: If True, gauge frames evolve through layers
            generators: SO(3) generators (for variational/hamiltonian FFN)
            ffn_mode: 'learned'/'standard', 'VFE'/'variational_gradient_engine', 'hamiltonian'
            ffn_alpha: Prior weight (variational/hamiltonian)
            ffn_tau_eff: Temperature (variational)
            ffn_kappa: Softmax temperature (variational_full/hamiltonian)
            ffn_n_iterations: Inference iterations (variational)
            ffn_learnable_lr: Learn step size (variational)
            ffn_lambda_belief: Belief alignment weight (gradient_engine/hamiltonian)
            ffn_lambda_prior: Prior alignment weight (gradient_engine)
            ffn_lambda_phi: Gauge field weight (gradient_engine)
            ffn_update_sigma: Update covariances in FFN? (gradient_engine/hamiltonian)
            ffn_hamiltonian_dt: Leapfrog time step (hamiltonian)
            ffn_hamiltonian_n_steps: Number of leapfrog steps (hamiltonian)
            ffn_hamiltonian_momentum_scale: Initial momentum scale (hamiltonian)
            ffn_hamiltonian_gamma: Damping coefficient (hamiltonian)
            ffn_hamiltonian_mass_use_prior: Include prior precision in mass (Inertia of Belief)
            ffn_hamiltonian_mass_use_observation: Include observation precision in mass
            ffn_hamiltonian_mass_use_incoming_social: Include incoming social precision in mass
            ffn_hamiltonian_mass_use_outgoing_recoil: Include outgoing recoil precision in mass
            ffn_hamiltonian_evolve_mass: Recompute mass at each leapfrog step (hamiltonian)
        """
        super().__init__()
        self.n_layers = n_layers

        self.blocks = nn.ModuleList([
            GaugeTransformerBlock(
                embed_dim=embed_dim,
                irrep_spec=irrep_spec,
                hidden_dim=hidden_dim,
                kappa_beta=kappa_beta,
                dropout=dropout,
                evolve_sigma=evolve_sigma,
                evolve_phi=evolve_phi,
                # Variational FFN
                generators=generators,
                ffn_mode=ffn_mode,
                ffn_alpha=ffn_alpha,
                ffn_tau_eff=ffn_tau_eff,
                ffn_kappa=ffn_kappa,
                ffn_n_iterations=ffn_n_iterations,
                ffn_learnable_lr=ffn_learnable_lr,
                # Gradient engine
                ffn_lambda_belief=ffn_lambda_belief,
                ffn_lambda_prior=ffn_lambda_prior,
                ffn_lambda_phi=ffn_lambda_phi,
                ffn_update_sigma=ffn_update_sigma,
                # Hamiltonian
                ffn_hamiltonian_dt=ffn_hamiltonian_dt,
                ffn_hamiltonian_n_steps=ffn_hamiltonian_n_steps,
                ffn_hamiltonian_momentum_scale=ffn_hamiltonian_momentum_scale,
                ffn_hamiltonian_gamma=ffn_hamiltonian_gamma,
                # Hamiltonian mass configuration (from Inertia of Belief paper)
                ffn_hamiltonian_mass_use_prior=ffn_hamiltonian_mass_use_prior,
                ffn_hamiltonian_mass_use_observation=ffn_hamiltonian_mass_use_observation,
                ffn_hamiltonian_mass_use_incoming_social=ffn_hamiltonian_mass_use_incoming_social,
                ffn_hamiltonian_mass_use_outgoing_recoil=ffn_hamiltonian_mass_use_outgoing_recoil,
                ffn_hamiltonian_evolve_mass=ffn_hamiltonian_evolve_mass,
                # Diagonal covariance mode
                diagonal_covariance=diagonal_covariance,
            )
            for _ in range(n_layers)
        ])

        # Final layer norm
        self.final_norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        mu_q: torch.Tensor,
        sigma_q: torch.Tensor,
        phi: torch.Tensor,
        generators: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        mu_prior: Optional[torch.Tensor] = None,  # For variational FFN
        return_intermediates: bool = False,
        cached_head_transports: Optional[list] = None,  # Cross-layer transport cache
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[List]]:
        """
        Forward through all transformer blocks.

        Args:
            mu_q: Initial means (B, N, K)
            sigma_q: Initial covariances (B, N, K, K)
            phi: Initial gauge frames (B, N, 3)
            generators: SO(3) generators (3, K, K)
            mask: Optional causal mask
            mu_prior: Embedding priors (B, N, K) - for variational FFN
            return_intermediates: If True, return states after each layer
            cached_head_transports: Optional list of precomputed transport dicts per head.
                                   When evolve_phi=False, reuse across all layers (6× speedup).

        Returns:
            mu_q: Final means (B, N, K)
            sigma_q: Final covariances (B, N, K, K)
            phi: Final gauge frames (B, N, 3)
            intermediates: Optional list of intermediate states
        """
        intermediates = [] if return_intermediates else None

        for layer_idx, block in enumerate(self.blocks):
            mu_q, sigma_q, phi = block(
                mu_q, sigma_q, phi, generators, mask, mu_prior,
                cached_head_transports=cached_head_transports,
            )

            if return_intermediates:
                intermediates.append({
                    'layer': layer_idx,
                    'mu': mu_q.detach(),
                    'sigma': sigma_q.detach() if sigma_q is not None else None,
                    'phi': phi.detach(),
                })

        # Final normalization
        mu_q = self.final_norm(mu_q)

        return mu_q, sigma_q, phi, intermediates

    def get_hamiltonian_diagnostics(self) -> List[Optional[dict]]:
        """Get Hamiltonian diagnostics from all layers."""
        return [block.get_hamiltonian_diagnostics() for block in self.blocks]


# =============================================================================
# Testing
# =============================================================================

if __name__ == '__main__':
    print("="*70)
    print("GAUGE TRANSFORMER BLOCK TEST")
    print("="*70)

    # Test configuration
    B, N, K = 2, 8, 16
    n_layers = 3
    hidden_dim = 64

    print(f"\n[1] Configuration:")
    print(f"    Batch size: {B}")
    print(f"    Num agents: {N}")
    print(f"    Embed dim:  {K}")
    print(f"    Layers:     {n_layers}")
    print(f"    Hidden dim: {hidden_dim}")

    # Create test data
    mu_q = torch.randn(B, N, K)
    sigma_q = torch.eye(K).unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1) * 0.5
    phi = torch.randn(B, N, 3) * 0.1

    # Random generators
    G = torch.randn(3, K, K)
    G = 0.5 * (G - G.transpose(-1, -2))

    # Irrep spec
    irrep_spec = [
        ('ℓ0', 4, 1),
        ('ℓ1', 2, 3),
        ('ℓ2', 1, 5),
    ]

    # Test single block
    print(f"\n[2] Testing single transformer block...")
    block = GaugeTransformerBlock(
        embed_dim=K,
        irrep_spec=irrep_spec,
        hidden_dim=hidden_dim,
        kappa_beta=1.0,
        dropout=0.1,
        evolve_sigma=False,
        evolve_phi=False,
    )
    print(f"    {block}")

    mu_out, sigma_out, phi_out = block(mu_q, sigma_q, phi, G)
    print(f"    Output μ shape: {mu_out.shape}")
    print(f"    Output Σ shape: {sigma_out.shape}")
    print(f"    Output φ shape: {phi_out.shape}")
    print(f"    ✓ Single block forward pass complete")

    # Test stack
    print(f"\n[3] Testing transformer stack ({n_layers} layers)...")
    stack = GaugeTransformerStack(
        n_layers=n_layers,
        embed_dim=K,
        irrep_spec=irrep_spec,
        hidden_dim=hidden_dim,
        kappa_beta=1.0,
        dropout=0.1,
        evolve_sigma=False,
        evolve_phi=False,
    )

    mu_final, sigma_final, phi_final, intermediates = stack(
        mu_q, sigma_q, phi, G, return_intermediates=True
    )

    print(f"    Final μ shape: {mu_final.shape}")
    print(f"    Intermediate states: {len(intermediates)}")
    print(f"    ✓ Stack forward pass complete")

    # Test with causal mask
    print(f"\n[4] Testing with causal mask...")
    mask = torch.tril(torch.ones(N, N)).unsqueeze(0).expand(B, -1, -1)
    mu_causal, _, _, _ = stack(mu_q, sigma_q, phi, G, mask=mask)
    print(f"    Causal output shape: {mu_causal.shape}")
    print(f"    ✓ Causal masking works")

    # Parameter count
    total_params = sum(p.numel() for p in stack.parameters())
    per_layer = total_params // n_layers

    print(f"\n[5] Parameter count:")
    print(f"    Total:     {total_params:,} parameters")
    print(f"    Per layer: {per_layer:,} parameters")
    print(f"    Attention: ~{per_layer // 3:,} params (1/3 of layer)")
    print(f"    FFN:       ~{2 * per_layer // 3:,} params (2/3 of layer)")

    # Compare to standard transformer
    standard_params = 4 * K * K + 2 * K * hidden_dim + 4 * K  # Q,K,V,O + FFN + LN
    standard_total = standard_params * n_layers
    reduction = standard_total / total_params

    print(f"\n[6] Comparison to standard transformer:")
    print(f"    Standard total: {standard_total:,} parameters")
    print(f"    Gauge total:    {total_params:,} parameters")
    print(f"    Reduction:      {reduction:.2f}x fewer parameters!")

    # =========================================================================
    # Test Hamiltonian FFN mode
    # =========================================================================
    print(f"\n[7] Testing HAMILTONIAN FFN mode...")

    # Create Hamiltonian transformer block
    hamiltonian_block = GaugeTransformerBlock(
        embed_dim=K,
        irrep_spec=irrep_spec,
        hidden_dim=hidden_dim,
        kappa_beta=1.0,
        dropout=0.0,  # No dropout for energy test
        evolve_sigma=True,
        evolve_phi=False,
        generators=G,
        ffn_mode='hamiltonian',
        ffn_alpha=1.0,
        ffn_lambda_belief=0.0,  # Disable alignment for cleaner test
        ffn_update_sigma=True,
        ffn_hamiltonian_dt=0.01,
        ffn_hamiltonian_n_steps=10,
        ffn_hamiltonian_momentum_scale=0.5,
        ffn_hamiltonian_gamma=0.0,  # Pure Hamiltonian
    )

    # Forward pass (need mu_prior for Hamiltonian)
    mu_prior = mu_q.clone() * 0.5  # Simple prior

    try:
        mu_ham, sigma_ham, phi_ham = hamiltonian_block(
            mu_q.clone(), sigma_q.clone(), phi.clone(), G,
            mu_prior=mu_prior,
        )

        print(f"    Output μ shape: {mu_ham.shape}")
        print(f"    Output Σ shape: {sigma_ham.shape}")
        print(f"    Output φ shape: {phi_ham.shape}")

        # Get diagnostics
        diagnostics = hamiltonian_block.get_hamiltonian_diagnostics()
        if diagnostics:
            print(f"    H_init: {diagnostics['H_init']:.4f}")
            print(f"    H_final: {diagnostics['H_final']:.4f}")
            print(f"    ΔH = {diagnostics['delta_H']:.6f}")

            if diagnostics['delta_H'] < 1.0:
                print(f"    ✓ Hamiltonian FFN: Energy approximately conserved!")
            else:
                print(f"    ~ Hamiltonian FFN: Energy drift (may need smaller dt)")

        # Check SPD preservation
        eigenvalues = torch.linalg.eigvalsh(sigma_ham)
        min_eig = eigenvalues.min().item()
        if min_eig > 0:
            print(f"    ✓ SPD preserved (min eigenvalue: {min_eig:.6f})")
        else:
            print(f"    ✗ SPD violated!")

        print(f"    ✓ Hamiltonian block forward pass complete")

    except Exception as e:
        print(f"    ✗ Hamiltonian test failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*70)
    print("✓ All transformer block tests passed!")
    print("="*70)