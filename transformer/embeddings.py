# -*- coding: utf-8 -*-
"""
Created on Thu Nov 13 12:34:45 2025

@author: chris and christine
"""

"""
Gauge-Theoretic Token Embeddings (0D Transformer)
==================================================

Maps discrete tokens → agent beliefs (μ_i, Σ_i, φ_i) at single base manifold point c*.

Key Insight from plan.py:
    "0D Transformer: All N tokens → N agents at the SAME base point c*
     Each token i → (μ_i, Σ_i, φ_i) where:
     - μ_i ∈ ℝ^K: mean belief vector (NO spatial dependence)
     - Σ_i ∈ SPD(K): covariance (scalar matrix per agent)
     - φ_i ∈ so(3): gauge frame (3D vector, not a field)"

Author: Implementation from plan.py
Date: November 2025
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, Optional


class GaugeTokenEmbedding(nn.Module):
    """
    Map discrete tokens to gauge-equivariant agent beliefs at single point.

    0D Transformer: All N tokens → N agents at the SAME base point c*
    Each token i → (μ_i, Σ_i, φ_i) where:
    - μ_i ∈ ℝ^K: mean belief vector (NO spatial dependence)
    - Σ_i ∈ SPD(K): covariance (scalar matrix per agent)
    - φ_i ∈ so(3): gauge frame (3D vector, not a field)

    Architecture:
        token_id → [Embedding Layer] → (μ, Σ, φ)

        where:
        - μ: Learnable embedding (standard)
        - Σ: Initialized to small isotropic (σ²I), optionally learnable
        - φ: Initialized to zero (identity gauge frame)
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        irrep_spec: list = None,
        init_std: float = 0.02,
        init_sigma_scale: float = 0.1,
        learnable_sigma: bool = False,
        learnable_phi: bool = False,
        gauge_fixed_priors: bool = False,
        generators: Optional[torch.Tensor] = None,
        diagonal_covariance: bool = False,
    ):
        """
        Initialize gauge token embedding.

        Args:
            vocab_size: Number of tokens in vocabulary
            embed_dim: Embedding dimension K (fiber dimension)
            irrep_spec: List of (label, multiplicity, dim) for SO(3) irreps
            init_std: Std dev for initializing mean embeddings
            init_sigma_scale: Initial scale for covariance (σ in σ²I)
            learnable_sigma: If True, Σ evolves during training
            learnable_phi: If True, φ evolves during training
            gauge_fixed_priors: If True, priors are defined as SO(3) rotations of a
                               single base prior: p_i = R_i ▷ p_0. This guarantees
                               gauge covariance: p_i = Ω_ij[p_j] where Ω_ij = R_i R_j^{-1}.
                               Requires generators for computing rotations.
            generators: SO(3) generators (3, K, K), required if gauge_fixed_priors=True
            diagonal_covariance: If True, output sigma as (B,N,K) diagonal variances
                                instead of (B,N,K,K) full matrices. Saves O(K) memory.
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.irrep_spec = irrep_spec
        self.learnable_sigma = learnable_sigma
        self.learnable_phi = learnable_phi
        self.gauge_fixed_priors = gauge_fixed_priors

        # CRITICAL: diagonal_covariance is incompatible with gauge_fixed_priors!
        # When gauge_fixed_priors=True, Σ_i = R_i Σ_0 R_i^T produces a FULL matrix
        # even if Σ_0 is diagonal. Extracting only diagonal loses correlations
        # induced by rotation, breaking gauge covariance: Σ_i ≠ Ω_ij Σ_j Ω_ij^T
        if gauge_fixed_priors and diagonal_covariance:
            import warnings
            warnings.warn(
                "gauge_fixed_priors=True is incompatible with diagonal_covariance=True. "
                "Rotation R_i Σ_0 R_i^T produces full matrices. "
                "Forcing diagonal_covariance=False to preserve gauge covariance.",
                UserWarning
            )
            diagonal_covariance = False

        self.diagonal_covariance = diagonal_covariance

        if gauge_fixed_priors and generators is None:
            raise ValueError("gauge_fixed_priors=True requires generators to be provided")

        if generators is not None:
            self.register_buffer('generators', generators)

        # =================================================================
        # Mean Embeddings μ_i (or base prior μ_0 if gauge_fixed_priors)
        # =================================================================
        if gauge_fixed_priors:
            # Single base prior mean μ_0 - all token priors are rotations of this
            self.base_mu = nn.Parameter(torch.randn(embed_dim) * init_std)
        else:
            # Standard learnable embedding: vocab_size × embed_dim
            self.mu_embed = nn.Embedding(vocab_size, embed_dim)
            nn.init.normal_(self.mu_embed.weight, mean=0.0, std=init_std)

        # =================================================================
        # Covariance Embeddings Σ_i (or base prior Σ_0 if gauge_fixed_priors)
        # =================================================================
        # Parameterize via log-diagonal (ensures positivity):
        #   Σ = diag(exp(log_σ_diag))
        #
        # This is a simplified SPD parametrization. Future: full Cholesky.

        if gauge_fixed_priors:
            # Single base prior covariance Σ_0 - all token priors are rotations of this
            self.base_log_sigma_diag = nn.Parameter(
                torch.full((embed_dim,), np.log(init_sigma_scale))
            )
        elif learnable_sigma:
            # Per-token covariance
            self.log_sigma_diag = nn.Parameter(
                torch.full((vocab_size, embed_dim), np.log(init_sigma_scale))
            )
        else:
            # Shared isotropic covariance across all tokens
            self.register_buffer(
                'log_sigma_diag',
                torch.full((embed_dim,), np.log(init_sigma_scale))
            )

        # =================================================================
        # Gauge Frame Embeddings φ_i ∈ so(3)
        # =================================================================
        # Initialize at zero (identity frame exp(0) = I)
        # When gauge_fixed_priors=True, these define both the gauge frame
        # AND the rotation R_i for computing p_i = R_i ▷ p_0

        if learnable_phi or gauge_fixed_priors:
            # Per-token gauge frame (required for gauge_fixed_priors)
            self.phi_embed = nn.Embedding(vocab_size, 3)  # so(3) is 3D
            nn.init.zeros_(self.phi_embed.weight)
        else:
            # All tokens start at identity frame
            self.register_buffer('phi_base', torch.zeros(3))

    def forward(
        self,
        token_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Embed tokens as agent beliefs at single base manifold point c*.

        Args:
            token_ids: (batch, seq_len) integer token indices

        Returns:
            mu: (batch, num_agents, K) mean beliefs (one per agent, NOT per spatial point)
            sigma: (batch, num_agents, K, K) covariances if diagonal_covariance=False
                   (batch, num_agents, K) diagonal variances if diagonal_covariance=True
            phi: (batch, num_agents, 3) gauge frames (one per agent)

        NOTE: seq_len = number of agents at the single point c*
              This is NOT a spatial dimension!

        When gauge_fixed_priors=True:
            Priors are computed as p_i = R_i ▷ p_0 where R_i = exp(φ_i · generators).
            This guarantees p_i = Ω_ij[p_j] for all i,j, restoring gauge covariance.
        """
        batch_size, num_agents = token_ids.shape

        # =================================================================
        # Gauge Frame Embeddings (computed first for gauge_fixed_priors)
        # =================================================================
        if self.learnable_phi or self.gauge_fixed_priors:
            # Per-token gauge frame
            phi = self.phi_embed(token_ids)  # (B, N, 3)
        else:
            # All agents at identity frame
            phi = self.phi_base.unsqueeze(0).unsqueeze(0)  # (1, 1, 3)
            phi = phi.expand(batch_size, num_agents, -1)  # (B, N, 3)

        # =================================================================
        # Mean and Covariance Embeddings
        # =================================================================
        if self.gauge_fixed_priors:
            # Compute rotation matrices R_i = exp(φ_i · generators)
            # phi: (B, N, 3), generators: (3, K, K)
            phi_matrix = torch.einsum('bnc,ckl->bnkl', phi, self.generators)  # (B, N, K, K)
            R = torch.linalg.matrix_exp(phi_matrix)  # (B, N, K, K)

            # Rotate base prior mean: μ_i = R_i @ μ_0
            # base_mu: (K,), R: (B, N, K, K)
            mu = torch.einsum('bnkl,l->bnk', R, self.base_mu)  # (B, N, K)

            # Build base covariance Σ_0 = diag(exp(log_σ_0))
            sigma_diag_base = torch.exp(self.base_log_sigma_diag)  # (K,)
            Sigma_0 = torch.diag(sigma_diag_base)  # (K, K)

            # Rotate base prior covariance: Σ_i = R_i @ Σ_0 @ R_i^T
            # R: (B, N, K, K), Sigma_0: (K, K)
            # NOTE: diagonal_covariance is forced False when gauge_fixed_priors=True
            # (see __init__), so we always output full matrices here.
            sigma = torch.einsum('bnij,jk,bnlk->bnil', R, Sigma_0, R)  # (B, N, K, K)
        else:
            # Standard per-token embeddings
            # μ(token_i) for each agent i at c*
            mu = self.mu_embed(token_ids)  # (B, N, K) where N = num_agents

            # Build diagonal covariances: Σ = diag(exp(log_σ))
            if self.learnable_sigma:
                # Per-token covariance
                log_sigma = self.log_sigma_diag[token_ids]  # (B, N, K)
                sigma_diag = torch.exp(log_sigma)  # (B, N, K)
            else:
                # Shared covariance
                sigma_diag = torch.exp(self.log_sigma_diag)  # (K,)
                sigma_diag = sigma_diag.unsqueeze(0).unsqueeze(0)  # (1, 1, K)
                sigma_diag = sigma_diag.expand(batch_size, num_agents, -1)  # (B, N, K)

            if self.diagonal_covariance:
                # Keep as diagonal variances (B, N, K)
                sigma = sigma_diag
            else:
                # Convert to full covariance matrices (diagonal)
                sigma = torch.diag_embed(sigma_diag)  # (B, N, K, K)

        return mu, sigma, phi

    def extra_repr(self) -> str:
        """Pretty print for model summary."""
        return (
            f"vocab_size={self.vocab_size}, "
            f"embed_dim={self.embed_dim}, "
            f"learnable_sigma={self.learnable_sigma}, "
            f"learnable_phi={self.learnable_phi}, "
            f"gauge_fixed_priors={self.gauge_fixed_priors}"
        )


def so3_log_torch(R: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Logarithm map from SO(3) → so(3) (PyTorch version).

    Given R ∈ SO(3), find φ ∈ ℝ³ such that exp([φ]_×) = R.

    Formula:
        θ = arccos((tr(R) - 1) / 2)
        φ = (θ / (2 sin θ)) * vex(R - Rᵀ)

    Args:
        R: Rotation matrices, shape (..., 3, 3)
        eps: Threshold for small angle approximation

    Returns:
        phi: Lie algebra elements, shape (..., 3)
    """
    # Compute rotation angle from trace
    trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]  # (...)
    cos_theta = (trace - 1.0) / 2.0
    cos_theta = torch.clamp(cos_theta, -1.0 + eps, 1.0 - eps)
    theta = torch.acos(cos_theta)  # (...)

    # Extract skew-symmetric part: vex(R - R^T) / 2
    # vex extracts [v_x, v_y, v_z] from skew-symmetric matrix
    skew = R - R.transpose(-1, -2)  # (..., 3, 3)
    v_x = (skew[..., 2, 1] - skew[..., 1, 2]) / 2.0
    v_y = (skew[..., 0, 2] - skew[..., 2, 0]) / 2.0
    v_z = (skew[..., 1, 0] - skew[..., 0, 1]) / 2.0
    vex_skew = torch.stack([v_x, v_y, v_z], dim=-1)  # (..., 3)

    # Coefficient: θ / (2 sin θ), handle small angles
    sin_theta = torch.sin(theta)
    # For small θ: θ/(2sinθ) ≈ 1/2 + θ²/12
    small_angle = theta < eps
    coeff = torch.where(
        small_angle,
        0.5 + theta**2 / 12.0,
        theta / (2.0 * sin_theta + eps)
    )

    phi = coeff.unsqueeze(-1) * vex_skew
    return phi


def so3_compose_bch(
    phi1: torch.Tensor,
    phi2: torch.Tensor,
    order: int = 1,
) -> torch.Tensor:
    """
    Compose two so(3) elements using Baker-Campbell-Hausdorff formula.

    log(exp(φ₁)·exp(φ₂)) = φ₁ + φ₂ + ½[φ₁,φ₂] + (1/12)[φ₁,[φ₁,φ₂]] - ...

    For so(3), the Lie bracket is: [X, Y] = X × Y (cross product)

    Args:
        phi1: First so(3) element, shape (..., 3)
        phi2: Second so(3) element, shape (..., 3)
        order: BCH expansion order (0=addition, 1=first correction, 2=second)

    Returns:
        phi_composed: Composed element in so(3), shape (..., 3)
    """
    if order == 0:
        # Simple addition (valid for small angles only)
        return phi1 + phi2

    # First-order BCH: φ₁ + φ₂ + ½[φ₁,φ₂]
    # In so(3): [φ₁,φ₂] = φ₁ × φ₂ (cross product)
    bracket_12 = torch.cross(phi1, phi2, dim=-1)
    result = phi1 + phi2 + 0.5 * bracket_12

    if order >= 2:
        # Second-order: + (1/12)[φ₁,[φ₁,φ₂]] - (1/12)[φ₂,[φ₁,φ₂]]
        bracket_1_12 = torch.cross(phi1, bracket_12, dim=-1)
        bracket_2_12 = torch.cross(phi2, bracket_12, dim=-1)
        result = result + (1.0/12.0) * bracket_1_12 - (1.0/12.0) * bracket_2_12

    return result


class GaugePositionalEncoding(nn.Module):
    """
    Agent-index-dependent gauge frame modulation (0D positional encoding).

    In 0D: Position encodes AGENT INDEX, not spatial location.
    All agents are at the same point c*, but need to distinguish
    their roles in the sequence via gauge frame modulation.

    Composition modes:
        - 'add': φ_combined = φ_base + φ_pos (valid for small angles)
        - 'bch1': φ_combined = φ_base + φ_pos + ½[φ_base, φ_pos] (BCH order 1)
        - 'bch2': Higher-order BCH correction
        - 'exact': Full SO(3) composition via exp → multiply → log

    This is analogous to standard positional encoding, but in so(3) instead of ℝ^K.
    """

    def __init__(
        self,
        max_seq_len: int,
        mode: str = 'learned',
        scale: float = 0.1,
        composition: str = 'bch1',
    ):
        """
        Initialize positional encoding in gauge space.

        Args:
            max_seq_len: Maximum sequence length (max number of agents at c*)
            mode: 'learned' or 'sinusoidal'
            scale: Scaling factor for positional encodings
            composition: How to combine token φ with positional φ:
                - 'add': Simple addition (φ_base + φ_pos) - fast but only valid for small angles
                - 'bch1': BCH order 1 correction (φ_base + φ_pos + ½[φ_base, φ_pos])
                - 'bch2': BCH order 2 correction (more accurate for larger angles)
                - 'exact': Full SO(3) composition via exp → multiply → log
        """
        super().__init__()
        self.max_seq_len = max_seq_len
        self.mode = mode
        self.scale = scale
        self.composition = composition

        if mode == 'learned':
            # Learnable agent-index-specific gauge biases
            # Each agent index i gets a unique φ_pos(i) ∈ so(3)
            self.pos_phi = nn.Parameter(torch.randn(max_seq_len, 3) * scale)

        elif mode == 'sinusoidal':
            # Sinusoidal encoding projected to so(3)
            # Fixed (not learnable)
            self.register_buffer('pos_phi', self._make_sinusoidal(max_seq_len, scale))

        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'learned' or 'sinusoidal'.")

    def _make_sinusoidal(self, max_len: int, scale: float) -> torch.Tensor:
        """
        Create sinusoidal positional encoding in so(3).

        This encodes agent index i, not spatial position!

        Formula (adapted from Transformer):
            φ_pos[i, 0] = scale * sin(i / 10000^(0/3))
            φ_pos[i, 1] = scale * sin(i / 10000^(1/3))
            φ_pos[i, 2] = scale * cos(i / 10000^(2/3))

        Args:
            max_len: Maximum sequence length
            scale: Scaling factor

        Returns:
            pos_phi: (max_len, 3) positional gauge frames
        """
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)  # (L, 1)
        div_term = torch.exp(torch.arange(0, 3, 1, dtype=torch.float32) * -(np.log(10000.0) / 3))

        phi = torch.zeros(max_len, 3)
        phi[:, 0] = torch.sin(position.squeeze() * div_term[0])
        phi[:, 1] = torch.sin(position.squeeze() * div_term[1])
        phi[:, 2] = torch.cos(position.squeeze() * div_term[2])

        return phi * scale

    def forward(self, num_agents: int, device: Optional[torch.device] = None) -> torch.Tensor:
        """
        Get positional gauge frames for given number of agents.

        Args:
            num_agents: Number of agents (sequence length)
            device: Device to place output on

        Returns:
            pos_phi: (num_agents, 3) agent-index-dependent gauge frames

        NOTE: This is NOT a spatial field! Just one φ per agent index.
        """
        if num_agents > self.max_seq_len:
            raise ValueError(
                f"Sequence length {num_agents} exceeds max {self.max_seq_len}. "
                f"Increase max_seq_len in config."
            )

        pos_phi = self.pos_phi[:num_agents]  # (N, 3)

        if device is not None:
            pos_phi = pos_phi.to(device)

        return pos_phi

    def compose(
        self,
        phi: torch.Tensor,
        num_agents: int,
        device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """
        Compose token gauge frames with positional gauge frames using proper SO(3) composition.

        Instead of φ_combined = φ + φ_pos (which violates group structure),
        this method uses the BCH formula or exact SO(3) composition.

        Args:
            phi: Token gauge frames, shape (B, N, 3)
            num_agents: Number of agents (sequence length)
            device: Device to place output on

        Returns:
            phi_combined: Composed gauge frames, shape (B, N, 3)

        Mathematical background:
            In SO(3), the correct composition is R_combined = exp(φ) · exp(φ_pos)
            In so(3), this is NOT simply φ + φ_pos. The BCH formula gives:
            log(exp(X)·exp(Y)) = X + Y + ½[X,Y] + (1/12)[X,[X,Y]] - (1/12)[Y,[X,Y]] + ...
            For so(3), the Lie bracket is the cross product: [X,Y] = X × Y
        """
        pos_phi = self.forward(num_agents, device)  # (N, 3)
        pos_phi = pos_phi.unsqueeze(0).expand(phi.shape[0], -1, -1)  # (B, N, 3)

        if self.composition == 'add':
            # Simple addition (original behavior, valid for small angles only)
            return phi + pos_phi

        elif self.composition == 'bch1':
            # First-order BCH correction
            return so3_compose_bch(phi, pos_phi, order=1)

        elif self.composition == 'bch2':
            # Second-order BCH correction
            return so3_compose_bch(phi, pos_phi, order=2)

        elif self.composition == 'exact':
            # Full SO(3) composition: log(exp(φ) · exp(φ_pos))
            # Build skew-symmetric matrices and exponentiate
            # [φ]_× for so(3) → SO(3)
            def skew_symmetric_batch(v):
                """v: (..., 3) -> (..., 3, 3) skew-symmetric"""
                zeros = torch.zeros_like(v[..., 0])
                return torch.stack([
                    torch.stack([zeros, -v[..., 2], v[..., 1]], dim=-1),
                    torch.stack([v[..., 2], zeros, -v[..., 0]], dim=-1),
                    torch.stack([-v[..., 1], v[..., 0], zeros], dim=-1),
                ], dim=-2)

            phi_skew = skew_symmetric_batch(phi)  # (B, N, 3, 3)
            pos_phi_skew = skew_symmetric_batch(pos_phi)  # (B, N, 3, 3)

            R_phi = torch.matrix_exp(phi_skew)  # (B, N, 3, 3)
            R_pos = torch.matrix_exp(pos_phi_skew)  # (B, N, 3, 3)

            R_combined = R_phi @ R_pos  # (B, N, 3, 3)

            # Map back to so(3) via logarithm
            phi_combined = so3_log_torch(R_combined)  # (B, N, 3)
            return phi_combined

        else:
            raise ValueError(f"Unknown composition mode: {self.composition}")

    def extra_repr(self) -> str:
        return f"max_seq_len={self.max_seq_len}, mode={self.mode}, scale={self.scale}, composition={self.composition}"


# =============================================================================
# Testing & Visualization
# =============================================================================

if __name__ == '__main__':
    print("="*70)
    print("GAUGE TOKEN EMBEDDING TEST (0D Transformer)")
    print("="*70)

    # Test configuration
    vocab_size = 100
    embed_dim = 32
    batch_size = 4
    seq_len = 10

    # Create embedding layer
    print("\n[1] Creating GaugeTokenEmbedding...")
    embedder = GaugeTokenEmbedding(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        init_std=0.02,
        init_sigma_scale=0.1,
        learnable_sigma=False,  # Start simple
        learnable_phi=False,
    )
    print(embedder)

    # Create random tokens
    print(f"\n[2] Embedding random tokens...")
    token_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    print(f"    Token IDs shape: {token_ids.shape}")

    # Forward pass
    mu, sigma, phi = embedder(token_ids)

    print(f"\n[3] Output shapes:")
    print(f"    μ (means):      {mu.shape}     # (B, N, K) where N=num_agents at c*")
    print(f"    Σ (covariances): {sigma.shape}   # (B, N, K, K)")
    print(f"    φ (gauge frames): {phi.shape}      # (B, N, 3) in so(3)")

    # Validate covariance is SPD
    print(f"\n[4] Validating covariances...")
    eigenvalues = torch.linalg.eigvalsh(sigma[0, 0])  # Check first agent
    print(f"    Eigenvalues of Σ[0,0]: {eigenvalues.numpy()}")
    assert torch.all(eigenvalues > 0), "Covariance not positive definite!"
    print("    ✓ All eigenvalues positive (SPD verified)")

    # Test positional encoding
    print(f"\n{'='*70}")
    print("GAUGE POSITIONAL ENCODING TEST")
    print('='*70)

    max_seq_len = 64

    # Test learned encoding
    print("\n[5] Testing learned positional encoding...")
    pos_enc_learned = GaugePositionalEncoding(max_seq_len, mode='learned', scale=0.1)
    pos_phi_learned = pos_enc_learned(seq_len)
    print(f"    Learned φ_pos shape: {pos_phi_learned.shape}  # (N, 3)")
    print(f"    φ_pos[0]: {pos_phi_learned[0].detach().numpy()}")
    print(f"    φ_pos[9]: {pos_phi_learned[9].detach().numpy()}")

    # Test sinusoidal encoding
    print("\n[6] Testing sinusoidal positional encoding...")
    pos_enc_sin = GaugePositionalEncoding(max_seq_len, mode='sinusoidal', scale=0.1)
    pos_phi_sin = pos_enc_sin(seq_len)
    print(f"    Sinusoidal φ_pos shape: {pos_phi_sin.shape}")
    print(f"    φ_pos[0]: {pos_phi_sin[0].numpy()}")
    print(f"    φ_pos[9]: {pos_phi_sin[9].numpy()}")

    # Combined: Embedding + Position
    print(f"\n[7] Combined embedding with positional encoding...")
    phi_combined = phi + pos_phi_learned.unsqueeze(0)  # (B, N, 3)
    print(f"    φ_total = φ_base + φ_pos: {phi_combined.shape}")

    # Parameter count
    total_params = sum(p.numel() for p in embedder.parameters())
    print(f"\n[8] Parameter count:")
    print(f"    Token embedder: {total_params:,} parameters")
    pos_params = sum(p.numel() for p in pos_enc_learned.parameters())
    print(f"    Position encoder (learned): {pos_params:,} parameters")

    print("\n" + "="*70)
    print("✓ All tests passed!")
    print("="*70)