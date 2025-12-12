"""
Training Loop for Gauge-Theoretic Transformer
==============================================

Implements COMPLETE free energy minimization with all gauge-theoretic terms.

Full Free Energy:
    F = (1) Σ_i KL(q_i || p_i)                    [Belief prior - alpha]
      + (3) Σ_{i,j} β_{ij} · KL(q_i || Ω_{ij} q_j) [Belief alignment - beta]
      + (4) Σ_{i,j} γ_{ij} · KL(p_i || Ω_{ij} p_j) [Model alignment - gamma]
      - (5) E[log p(o|x)]                         [Observation likelihood]

where:
    - q_i = N(μ_i, Σ_i): Agent beliefs (evolved through transformer)
    - p_i = N(μ_embed[i], Σ_embed[i]): Embedding priors (initial)
    - β_{ij} = softmax_j(-KL(q_i||Ω_{ij}q_j)/κ): Belief coupling weights
    - γ_{ij} = softmax_j(-KL(p_i||Ω_{ij}p_j)/κ'): Prior coupling weights
    - Ω_{ij}: Parallel transport operator

Note: (2) Model prior Σ_i KL(s_i || r_i) = 0 since s_i = r_i

Author: Implementation from validated suite + complete gamma term
Date: November 2025
"""

# Suppress Triton and CuPy warnings BEFORE torch import (torch may trigger imports)
import warnings
warnings.filterwarnings("ignore", message="Failed to find cuobjdump", module="triton")
warnings.filterwarnings("ignore", message="Failed to find nvdisasm", module="triton")
warnings.filterwarnings("ignore", message="CUDA path could not be detected", module="cupy")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import time
import json
import numpy as np

# Import attention computation for gamma term
from transformer.attention import compute_attention_weights


# =============================================================================
# Gaussian KL Divergence (Proper Implementation)
# =============================================================================

def gaussian_kl_divergence(
    mu_q: torch.Tensor,      # (B, N, K)
    sigma_q: torch.Tensor,   # (B, N, K, K) full, (B, N, K) diagonal, or None (uses identity)
    mu_p: torch.Tensor,      # (B, N, K)
    sigma_p: torch.Tensor,   # (B, N, K, K) full, (B, N, K) diagonal, or None (uses identity)
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Compute KL(N(μ_q, Σ_q) || N(μ_p, Σ_p)) for Gaussian distributions.

    Handles both full covariance matrices (B, N, K, K) and diagonal covariances (B, N, K).
    Diagonal covariances are detected automatically based on tensor dimensions.

    For full covariances:
        KL = 0.5 * [tr(Σ_p⁻¹ Σ_q) + (μ_p - μ_q)ᵀ Σ_p⁻¹ (μ_p - μ_q) - K + log(|Σ_p|/|Σ_q|)]

    For diagonal covariances (O(K) instead of O(K³)):
        KL = 0.5 * [Σ_k(σ_q[k]/σ_p[k]) + Σ_k((μ_p[k]-μ_q[k])²/σ_p[k]) - K + Σ_k(log(σ_p[k])-log(σ_q[k]))]

    Args:
        mu_q: Posterior means (B, N, K)
        sigma_q: Posterior covariances - (B, N, K, K) full or (B, N, K) diagonal or None
        mu_p: Prior means (B, N, K)
        sigma_p: Prior covariances - (B, N, K, K) full or (B, N, K) diagonal or None
        eps: Numerical stability constant

    Returns:
        kl: KL divergence per agent, shape (B, N)
    """
    K = mu_q.shape[-1]
    device = mu_q.device
    dtype = mu_q.dtype

    # Detect if covariances are diagonal (3D) or full (4D)
    sigma_q_is_diagonal = sigma_q is not None and sigma_q.dim() == 3
    sigma_p_is_diagonal = sigma_p is not None and sigma_p.dim() == 3

    # If either is diagonal, use diagonal path for both (more efficient)
    use_diagonal = sigma_q_is_diagonal or sigma_p_is_diagonal

    if use_diagonal:
        # =================================================================
        # DIAGONAL PATH: O(K) per agent instead of O(K³)
        # =================================================================
        # Convert to diagonal variances if needed
        if sigma_q is None:
            sigma_q_diag = torch.ones(*mu_q.shape, device=device, dtype=dtype)
        elif sigma_q.dim() == 3:
            sigma_q_diag = sigma_q  # Already diagonal (B, N, K)
        else:
            # Extract diagonal from full matrix
            sigma_q_diag = torch.diagonal(sigma_q, dim1=-2, dim2=-1)  # (B, N, K)

        if sigma_p is None:
            sigma_p_diag = torch.ones(*mu_p.shape, device=device, dtype=dtype)
        elif sigma_p.dim() == 3:
            sigma_p_diag = sigma_p  # Already diagonal (B, N, K)
        else:
            # Extract diagonal from full matrix
            sigma_p_diag = torch.diagonal(sigma_p, dim1=-2, dim2=-1)  # (B, N, K)

        # Ensure positivity
        sigma_q_diag = sigma_q_diag.clamp(min=eps)
        sigma_p_diag = sigma_p_diag.clamp(min=eps)

        # Trace term: Σ_k (σ_q[k] / σ_p[k])
        trace_term = (sigma_q_diag / sigma_p_diag).sum(dim=-1)  # (B, N)

        # Mahalanobis term: Σ_k ((μ_p[k] - μ_q[k])² / σ_p[k])
        delta_mu = mu_p - mu_q  # (B, N, K)
        mahal_term = ((delta_mu ** 2) / sigma_p_diag).sum(dim=-1)  # (B, N)

        # Log determinant term: Σ_k (log(σ_p[k]) - log(σ_q[k]))
        logdet_term = (torch.log(sigma_p_diag) - torch.log(sigma_q_diag)).sum(dim=-1)  # (B, N)

        # KL divergence
        kl = 0.5 * (trace_term + mahal_term - K + logdet_term)

    else:
        # =================================================================
        # FULL COVARIANCE PATH: O(K³) via Cholesky
        # =================================================================
        # Default to identity covariances if not provided
        if sigma_q is None:
            sigma_q = torch.eye(K, device=device, dtype=dtype).expand(*mu_q.shape[:-1], K, K)
        if sigma_p is None:
            sigma_p = torch.eye(K, device=device, dtype=dtype).expand(*mu_p.shape[:-1], K, K)

        # Regularize for numerical stability
        sigma_q_reg = sigma_q + eps * torch.eye(K, device=device, dtype=dtype)
        sigma_p_reg = sigma_p + eps * torch.eye(K, device=device, dtype=dtype)

        # Compute Σ_p⁻¹ via Cholesky
        L_p = torch.linalg.cholesky(sigma_p_reg)

        # Trace term: tr(Σ_p⁻¹ Σ_q)
        # Solve L_p @ Y = Σ_q for Y, then tr(Σ_p⁻¹ Σ_q) = tr(L_p⁻ᵀ Y)
        Y = torch.linalg.solve_triangular(L_p, sigma_q_reg, upper=False)
        Z = torch.linalg.solve_triangular(L_p.transpose(-1, -2), Y, upper=True)
        trace_term = torch.diagonal(Z, dim1=-2, dim2=-1).sum(dim=-1)  # (B, N)

        # Mahalanobis term: (μ_p - μ_q)ᵀ Σ_p⁻¹ (μ_p - μ_q)
        delta_mu = mu_p - mu_q  # (B, N, K)
        # Solve L_p @ v = delta_mu
        v = torch.linalg.solve_triangular(L_p, delta_mu.unsqueeze(-1), upper=False).squeeze(-1)
        mahal_term = torch.sum(v ** 2, dim=-1)  # (B, N)

        # Log determinant terms
        logdet_p = 2.0 * torch.sum(torch.log(torch.diagonal(L_p, dim1=-2, dim2=-1) + eps), dim=-1)
        L_q = torch.linalg.cholesky(sigma_q_reg)
        logdet_q = 2.0 * torch.sum(torch.log(torch.diagonal(L_q, dim1=-2, dim2=-1) + eps), dim=-1)
        logdet_term = logdet_p - logdet_q  # (B, N)

        # KL divergence
        kl = 0.5 * (trace_term + mahal_term - K + logdet_term)

    # Clamp to non-negative (numerical safety)
    return torch.clamp(kl, min=0.0)

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


# =============================================================================
# Free Energy Loss Computation (ATTENTION-WEIGHTED)
# =============================================================================

def compute_free_energy_loss(
    model,
    token_ids: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.0,           # Self-consistency weight
    lambda_beta: float = 1.0,     # Belief alignment weight
    lambda_gamma: float = 0.0,    # Model alignment weight
    kappa_gamma: float = 1.0,     # Temperature for γ_ij coupling weights
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute COMPLETE free energy loss with all gauge-theoretic terms.

    Full Free Energy:
        F = (1) α · Σ_i KL(q_i || p_i)                    [Belief prior]
          + (3) λ_β · Σ_{i,j} β_ij · KL(q_i || Ω_{ij}q_j) [Belief alignment]
          + (4) λ_γ · Σ_{i,j} γ_ij · KL(p_i || Ω_{ij}p_j) [Model alignment]
          - (5) E[log p(o|x)]                             [Observation likelihood]

    where:
        - q_i = N(μ_i, Σ_i): Agent beliefs (evolved through transformer)
        - p_i = N(μ_embed, Σ_embed): Embedding priors (initial, from token_embed)
        - β_ij = softmax_j(-KL(q_i||Ω_{ij}q_j)/κ_β): Belief coupling weights
        - γ_ij = softmax_j(-KL(p_i||Ω_{ij}p_j)/κ_γ): Prior coupling weights
        - Ω_{ij}: Parallel transport operator (gauge connection)
        - PyTorch autodiff handles ∂β_ij/∂μ_i and ∂γ_ij/∂μ_embed automatically!

    This is the CORRECT formulation from active inference + gauge theory:
        - Belief alignment (β): Encourages consistency between evolved beliefs
        - Model alignment (γ): Encourages consistency between embedding priors

    Args:
        model: GaugeTransformerLM with forward_with_attention() method
        token_ids: (B, N) input token IDs
        targets: (B, N) target token IDs
        alpha: Weight for belief prior KL(q||p) (default: 0.0)
        lambda_beta: Weight for belief alignment term (default: 1.0)
        lambda_gamma: Weight for model alignment term (default: 0.0)
        kappa_gamma: Temperature for γ_ij coupling weights (default: 1.0)

    Returns:
        total_loss: Scalar loss for backprop
        metrics: Dict with loss components

    Example:
        >>> # Standard training (gamma disabled)
        >>> loss, metrics = compute_free_energy_loss(
        ...     model, inputs, targets,
        ...     alpha=0.001, lambda_beta=0.1, lambda_gamma=0.0
        ... )

        >>> # With model alignment (regularize embedding space)
        >>> loss, metrics = compute_free_energy_loss(
        ...     model, inputs, targets,
        ...     alpha=0.001, lambda_beta=0.1, lambda_gamma=0.01
        ... )
    """
    # =================================================================
    # Forward pass with attention weights and KL matrices
    # =================================================================
    # NOTE: Do NOT pass targets here! Passing targets allows VFE FFN to
    # "cheat" by using targets to adjust beliefs before CE is computed,
    # causing CE to collapse to 0. Targets should only be used for loss
    # computation AFTER the forward pass.
    logits, attn_info = model.forward_with_attention(token_ids, targets=None)

    beta = attn_info['beta']    # (B, n_heads, N, N)
    kl = attn_info['kl']        # (B, n_heads, N, N)
    mu_q = attn_info['mu']      # (B, N, K) - evolved beliefs
    sigma_q = attn_info['sigma']  # (B, N, K, K) or None

    # Extract priors for gamma term
    mu_p = attn_info['mu_prior']      # (B, N, K) - embedding priors
    sigma_p = attn_info['sigma_prior']  # (B, N, K, K)
    phi_p = attn_info['phi_prior']      # (B, N, 3)
    generators = model.generators      # (3, K, K)

    # =================================================================
    # 1. Observation Likelihood: -E[log p(o|x)] = Cross-Entropy
    # =================================================================
    ce_loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),  # (B*N, V)
        targets.reshape(-1),                   # (B*N,)
        reduction='mean',
        ignore_index=-1,
    )

    # =================================================================
    # 2. Attention-Weighted Free Energy: Σ_ij β_ij · KL(q_i||Ω_ij[q_j])
    # =================================================================
    # This is the CRITICAL term from the validated suite!
    # Line 189 from gradients.free_energy_clean.py: weighted_field = beta_ij * kl_field

    if lambda_beta > 0.0:
        # Pointwise multiplication: β_ij * KL_ij for each head
        weighted_kl = beta * kl  # (B, n_heads, N, N)

        # Sum over all pairs (i,j) and average over heads and batch
        # Note: Averaging over batch and heads, summing over agent pairs
        belief_align_loss = weighted_kl.sum(dim=(-2, -1)).mean()  # Mean over (batch, heads)

        belief_align_loss = lambda_beta * belief_align_loss
    else:
        belief_align_loss = torch.tensor(0.0, device=ce_loss.device)

    # =================================================================
    # 3. Self-Consistency: α·KL(q||p) - Beliefs should stay close to priors
    # =================================================================
    # This is the key VFE term that pulls evolved beliefs back toward
    # their embedding priors. Without this, beliefs can drift arbitrarily.
    #
    # KL(q||p) = KL(N(μ_q, Σ_q) || N(μ_p, Σ_p))
    #
    # This provides gradients to embeddings even when belief evolution is detached!
    # =================================================================
    if alpha > 0.0:
        # Proper KL divergence between evolved beliefs and embedding priors
        kl_per_agent = gaussian_kl_divergence(
            mu_q=mu_q,        # Evolved beliefs
            sigma_q=sigma_q,  # Evolved covariances (or None)
            mu_p=mu_p,        # Embedding priors
            sigma_p=sigma_p,  # Embedding covariances
        )  # (B, N)

        # CRITICAL: Normalize by K to keep gradients O(1) regardless of dimension
        K = mu_q.shape[-1]
        kl_per_agent = kl_per_agent / K

        # Average over batch and agents
        self_consistency_loss = alpha * kl_per_agent.mean()
    else:
        self_consistency_loss = torch.tensor(0.0, device=ce_loss.device)

    # =================================================================
    # 4. Model Alignment: λ_γ·Σ_{i,j} γ_ij · KL(p_i || Ω_{ij} p_j)
    # =================================================================
    # This term encourages consistency between embedding priors p_i across agents.
    #
    # Formula:
    #   L_model = λ_γ · Σ_{i,j} γ_{ij} · KL(p_i || Ω_{ij} p_j)
    #
    # where:
    #   - p_i = N(μ_embed[i], Σ_embed[i]): Initial embedding prior
    #   - γ_{ij} = softmax_j(-KL(p_i || Ω_{ij} p_j) / κ_γ): Prior coupling weights
    #   - Ω_{ij} = exp(φ_i) · exp(-φ_j): Parallel transport operator
    #
    # This is symmetric to belief alignment (β term), but operates on
    # the embedding space rather than the evolved belief space.
    #
    # Use cases:
    #   - Regularize embedding space to be gauge-consistent
    #   - Prevent embeddings from having arbitrary gauge choices
    #   - Encourage smooth gauge structure in token space
    # =================================================================
    if lambda_gamma > 0.0:
        batch_size, num_agents, K = mu_p.shape
        device = mu_p.device

        # Causal mask (same as for beliefs)
        mask = torch.tril(torch.ones(num_agents, num_agents, device=device))
        mask = mask.unsqueeze(0).expand(batch_size, -1, -1)

        # Compute γ_{ij} coupling weights and KL(p_i || Ω_{ij} p_j)
        # Using same attention mechanism as β_{ij}, but on priors
        # Detect if sigma_p is diagonal (3D) or full (4D)
        diagonal_cov = sigma_p is not None and sigma_p.dim() == 3
        gamma, kl_prior = compute_attention_weights(
            mu_p,
            sigma_p,
            phi_p,
            generators,
            kappa_gamma,
            epsilon=1e-8,
            mask=mask,
            use_numba=False,  # Use PyTorch for gradient tracking
            return_kl=True,
            diagonal_covariance=diagonal_cov,
        )
        # gamma: (B, N, N)
        # kl_prior: (B, N, N)

        # Weighted model alignment: Σ_{i,j} γ_{ij} · KL(p_i || Ω_{ij} p_j)
        weighted_kl_prior = gamma * kl_prior  # (B, N, N)

        # Sum over all agent pairs and average over batch
        model_align_loss = lambda_gamma * weighted_kl_prior.sum(dim=(-2, -1)).mean()
    else:
        model_align_loss = torch.tensor(0.0, device=ce_loss.device)

    # =================================================================
    # Total Free Energy (ALL FOUR TERMS)
    # =================================================================
    total_loss = ce_loss + belief_align_loss + self_consistency_loss + model_align_loss

    # Compute attention entropy: -Σ β_ij log(β_ij) per query position
    # Average over heads first
    beta_avg = beta.mean(dim=1)  # (B, N, N) - average over heads
    beta_safe = beta_avg.clamp(min=1e-10)  # Numerical stability
    attn_entropy = -(beta_safe * beta_safe.log()).sum(dim=-1).mean()  # Mean over (B, N)

    # Attention concentration (max attention weight)
    attn_concentration = beta_avg.max(dim=-1)[0].mean()

    # Metrics
    metrics = {
        'loss/total': total_loss.item(),
        'loss/ce': ce_loss.item(),
        'loss/belief_align': belief_align_loss.item(),
        'loss/self_consistency': self_consistency_loss.item() if alpha > 0 else 0.0,
        'loss/model_align': model_align_loss.item() if lambda_gamma > 0 else 0.0,
        'attention/beta_mean': beta.mean().item(),
        'attention/kl_mean': kl.mean().item(),
        'attention/entropy': attn_entropy.item(),  # How diffuse is attention?
        'attention/concentration': attn_concentration.item(),  # How peaked is attention?
    }

    if lambda_gamma > 0.0:
        metrics['attention/gamma_mean'] = gamma.mean().item()
        metrics['attention/kl_prior_mean'] = kl_prior.mean().item()

    # =================================================================
    # Retrieve Hamiltonian diagnostics (if available)
    # =================================================================
    # For Hamiltonian mode, diagnostics are stored in the transformer blocks
    # after the forward pass. Retrieve them for publication metrics.
    hamiltonian_diagnostics = None
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'get_hamiltonian_diagnostics'):
        all_diagnostics = model.transformer.get_hamiltonian_diagnostics()
        # Get the last layer's diagnostics (most relevant for output)
        for diag in reversed(all_diagnostics):
            if diag is not None:
                hamiltonian_diagnostics = diag
                break

    if hamiltonian_diagnostics is not None:
        metrics['hamiltonian_diagnostics'] = hamiltonian_diagnostics

    return total_loss, metrics


# =============================================================================
# Training Configuration
# =============================================================================

@dataclass
class TrainingConfig:
    """
    Unified training configuration supporting both simple and multi-group parameter optimization.

    Modes:
    - Simple (use_param_groups=False): Single learning rate for all parameters
    - Multi-group (use_param_groups=True): Separate learning rates for mu, sigma, phi, attention, ffn, output
    """

    # Parameter grouping strategy
    use_param_groups: bool = False  # If True, use multi-group learning rates (natural gradients!)

    # Simple mode: Single learning rate (used when use_param_groups=False)
    learning_rate: float = 3e-4

    # Multi-group mode: Per-parameter group learning rates (used when use_param_groups=True)
    mu_lr: float = 0.1           # Mean embeddings (natural gradient scale)
    sigma_lr: float = 0.005      # Covariance embeddings (smaller for stability)
    phi_lr: float = 0.01         # Gauge frames
    attention_lr: float = 0.01   # Attention parameters
    ffn_lr: float = 0.001        # FFN parameters (standard)
    output_lr: float = 0.001     # Output projection

    # Optimizer hyperparameters
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    grad_clip: float = 1.0

    # Learning rate schedule
    warmup_steps: int = 1000
    max_steps: int = 50000
    lr_decay: str = 'cosine'  # 'cosine', 'linear', 'constant'
    min_lr: float = 3e-5

    # Free energy weights
    # NOTE: alpha > 0 is CRITICAL for gradient flow to embeddings!
    # KL(q||p) pulls evolved beliefs back to embedding priors and provides
    # gradients to mu_embed even when FFN outputs are detached.
    alpha: float = 0.1           # Self-consistency: KL(q||p) to embedding priors
    lambda_beta: float = 1.0     # Belief alignment: Σβ_ij·KL (CRUCIAL!)
    lambda_gamma: float = 0.0    # Model alignment (disabled by default)
    kappa_gamma: float = 1.0     # Temperature for γ_ij coupling weights

    # Training
    batch_size: int = 16
    num_epochs: int = None
    accumulation_steps: int = 1

    # Logging
    log_every: int = 100
    eval_every: int = 1000
    save_every: int = 5000
    log_interval: int = 10       # Alias for log_every (for compatibility)
    eval_interval: int = 100     # Alias for eval_every (for compatibility)
    checkpoint_interval: int = 200

    # Early stopping
    patience: int = 0  # If > 0, stop if no improvement for this many evals

    # Checkpointing
    checkpoint_dir: Optional[Path] = None
    save_optimizer: bool = True
    save_total_limit: int = 3

    # Weights & Biases
    use_wandb: bool = False
    wandb_project: str = 'gauge-transformer'
    wandb_run_name: Optional[str] = None

    # Device
    device: str = 'cpu'
    use_amp: bool = False


# Backward compatibility: FastTrainingConfig is now an alias with use_param_groups=True
@dataclass
class FastTrainingConfig(TrainingConfig):
    """
    DEPRECATED: Use TrainingConfig with use_param_groups=True instead.

    This class exists for backward compatibility with existing code.
    """
    use_param_groups: bool = True  # Enable multi-group optimization by default
    max_steps: int = 1000
    warmup_steps: int = 50
    alpha: float = 1.0             # Different default than base TrainingConfig


# =============================================================================
# Trainer Class
# =============================================================================

class Trainer:
    """Training orchestration for gauge transformer."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        config: Optional[TrainingConfig] = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config or TrainingConfig()

        # Move model to device
        self.device = torch.device(self.config.device)
        self.model = self.model.to(self.device)

        # Optimizer
        self.optimizer = self._create_optimizer()

        # Learning rate scheduler
        self.scheduler = self._create_scheduler()

        # Mixed precision scaler (using modern AMP API for PyTorch 2.x / CUDA 12+)
        if self.config.use_amp and self.config.device == 'cuda':
            self.scaler = torch.amp.GradScaler('cuda')
        else:
            self.scaler = None

        # Training state
        self.step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')

        # Create checkpoint directory
        if self.config.checkpoint_dir is not None:
            self.config.checkpoint_dir = Path(self.config.checkpoint_dir)
            self.config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Initialize W&B
        if self.config.use_wandb and WANDB_AVAILABLE:
            wandb.init(
                project=self.config.wandb_project,
                name=self.config.wandb_run_name,
                config=vars(self.config),
            )
            wandb.watch(self.model, log='all', log_freq=1000)

        print("Trainer initialized")
        print(f"  Device: {self.device}")
        print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"  Optimizer: AdamW (lr={self.config.learning_rate})")
        print(f"  λ_β (attention-weighted KL): {self.config.lambda_beta}")
        print(f"  Max steps: {self.config.max_steps:,}")

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """
        Create AdamW optimizer with configurable parameter grouping.

        Modes:
        - Simple (use_param_groups=False): 2 groups (decay vs no-decay) with single LR
        - Multi-group (use_param_groups=True): 6 groups (mu, sigma, phi, attention, ffn, output)
        """
        if self.config.use_param_groups:
            # Multi-group mode: Natural gradients with per-parameter-type learning rates
            return self._create_multigroup_optimizer()
        else:
            # Simple mode: Traditional 2-group optimizer (decay vs no-decay)
            return self._create_simple_optimizer()

    def _create_simple_optimizer(self) -> torch.optim.Optimizer:
        """Create simple 2-group optimizer (decay vs no-decay)."""
        decay_params = []
        no_decay_params = []

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if 'bias' in name or 'norm' in name:
                    no_decay_params.append(param)
                else:
                    decay_params.append(param)

        optimizer = torch.optim.AdamW([
            {'params': decay_params, 'weight_decay': self.config.weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0},
        ], lr=self.config.learning_rate, betas=(self.config.beta1, self.config.beta2))

        return optimizer

    def _create_multigroup_optimizer(self) -> torch.optim.Optimizer:
        """
        Create optimizer with per-parameter group learning rates.

        Parameter Groups:
            1. mu_embed: Mean embeddings
            2. sigma_embed: Covariance embeddings
            3. phi_embed: Gauge frame embeddings
            4. attention: Attention mechanism
            5. ffn: Feed-forward networks
            6. output: Output projection

        This exploits natural gradient structure on statistical manifolds!
        """
        # Collect parameters by type
        mu_params = []
        sigma_params = []
        phi_params = []
        attention_params = []
        ffn_params = []
        output_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            # Mean embeddings
            if 'mu_embed' in name:
                mu_params.append(param)
            # Covariance embeddings
            elif 'sigma_embed' in name:
                sigma_params.append(param)
            # Gauge frame embeddings
            elif 'phi_embed' in name:
                phi_params.append(param)
            # Positional encoding (treat as gauge frames)
            elif 'pos_encoding' in name:
                phi_params.append(param)
            # Attention mechanism
            elif 'attention' in name or 'attn' in name:
                attention_params.append(param)
            # Output projection
            elif 'out_proj' in name:
                output_params.append(param)
            # FFN (default for everything else)
            else:
                ffn_params.append(param)

        # Create parameter groups
        param_groups = []

        if mu_params:
            param_groups.append({
                'params': mu_params,
                'lr': self.config.mu_lr,
                'weight_decay': 0.0,  # No decay for embeddings
                'name': 'mu_embed',
            })
            print(f"  Parameter group 'mu_embed': {len(mu_params)} tensors @ lr={self.config.mu_lr}")

        if sigma_params:
            param_groups.append({
                'params': sigma_params,
                'lr': self.config.sigma_lr,
                'weight_decay': 0.0,
                'name': 'sigma_embed',
            })
            print(f"  Parameter group 'sigma_embed': {len(sigma_params)} tensors @ lr={self.config.sigma_lr}")

        if phi_params:
            param_groups.append({
                'params': phi_params,
                'lr': self.config.phi_lr,
                'weight_decay': 0.0,
                'name': 'phi_embed',
            })
            print(f"  Parameter group 'phi_embed': {len(phi_params)} tensors @ lr={self.config.phi_lr}")

        if attention_params:
            param_groups.append({
                'params': attention_params,
                'lr': self.config.attention_lr,
                'weight_decay': self.config.weight_decay,
                'name': 'attention',
            })
            print(f"  Parameter group 'attention': {len(attention_params)} tensors @ lr={self.config.attention_lr}")

        if ffn_params:
            param_groups.append({
                'params': ffn_params,
                'lr': self.config.ffn_lr,
                'weight_decay': self.config.weight_decay,
                'name': 'ffn',
            })
            print(f"  Parameter group 'ffn': {len(ffn_params)} tensors @ lr={self.config.ffn_lr}")

        if output_params:
            param_groups.append({
                'params': output_params,
                'lr': self.config.output_lr,
                'weight_decay': 0.0,  # Often tied to embeddings
                'name': 'output',
            })
            print(f"  Parameter group 'output': {len(output_params)} tensors @ lr={self.config.output_lr}")

        optimizer = torch.optim.AdamW(
            param_groups,
            betas=(self.config.beta1, self.config.beta2),
            eps=self.config.eps,
        )

        return optimizer

    def _create_scheduler(self):
        """Create learning rate scheduler."""
        if self.config.lr_decay == 'constant':
            return None

        def lr_lambda(step):
            if step < self.config.warmup_steps:
                return step / max(1, self.config.warmup_steps)

            if self.config.lr_decay == 'cosine':
                progress = (step - self.config.warmup_steps) / max(1, self.config.max_steps - self.config.warmup_steps)
                return self.config.min_lr / self.config.learning_rate + \
                       0.5 * (1 - self.config.min_lr / self.config.learning_rate) * \
                       (1 + torch.cos(torch.tensor(progress * 3.14159265)))
            elif self.config.lr_decay == 'linear':
                return max(0.0, (self.config.max_steps - step) / (self.config.max_steps - self.config.warmup_steps))
            else:
                return 1.0

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def train_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Dict[str, float]:
        """Single training step."""
        self.model.train()

        token_ids, targets = batch
        token_ids = token_ids.to(self.device)
        targets = targets.to(self.device)

        # Forward + loss
        with torch.amp.autocast('cuda', enabled=(self.scaler is not None)):
            loss, metrics = compute_free_energy_loss(
                self.model,
                token_ids,
                targets,
                alpha=self.config.alpha,
                lambda_beta=self.config.lambda_beta,
                lambda_gamma=self.config.lambda_gamma,
                kappa_gamma=self.config.kappa_gamma,
            )

            # Scale loss for gradient accumulation
            loss = loss / self.config.accumulation_steps

        # Backward
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        # =================================================================
        # Gradient Monitoring (verify gradients flow to embeddings)
        # =================================================================
        grad_norms = {}
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                # Track key parameter groups
                if 'mu_embed' in name:
                    grad_norms['grad/mu_embed'] = grad_norm
                elif 'sigma_embed' in name or 'log_sigma' in name:
                    grad_norms['grad/sigma_embed'] = grad_norm
                elif 'phi_embed' in name:
                    grad_norms['grad/phi_embed'] = grad_norm
                elif 'out_proj' in name:
                    grad_norms['grad/out_proj'] = grad_norm

        # Add to metrics
        metrics.update(grad_norms)

        # Warn if embedding gradients are zero (broken gradient flow!)
        if 'grad/mu_embed' in grad_norms and grad_norms['grad/mu_embed'] == 0.0:
            print("[WARNING] mu_embed gradient is ZERO - gradient flow may be broken!")

        # Optimizer step (if accumulation complete)
        if (self.step + 1) % self.config.accumulation_steps == 0:
            # Gradient clipping
            if self.scaler is not None:
                self.scaler.unscale_(self.optimizer)

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.grad_clip
            )

            # Optimizer step
            if self.scaler is not None:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

            # Update learning rate
            if self.scheduler is not None:
                self.scheduler.step()

            # Zero gradients
            self.optimizer.zero_grad()

        # Add learning rate to metrics
        metrics['lr'] = self.optimizer.param_groups[0]['lr']

        return metrics

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validation pass."""
        if self.val_loader is None:
            return {}

        self.model.eval()
        total_loss = 0.0
        total_ce_loss = 0.0
        n_batches = 0

        for batch in self.val_loader:
            token_ids, targets = batch
            token_ids = token_ids.to(self.device)
            targets = targets.to(self.device)

            loss, metrics = compute_free_energy_loss(
                self.model, token_ids, targets,
                alpha=self.config.alpha,
                lambda_beta=self.config.lambda_beta,
            )

            total_loss += loss.item()
            total_ce_loss += metrics['loss/ce']
            n_batches += 1

        avg_loss = total_loss / n_batches
        avg_ce_loss = total_ce_loss / n_batches
        perplexity = torch.exp(torch.tensor(avg_ce_loss)).item()

        return {
            'val/loss': avg_loss,
            'val/ce_loss': avg_ce_loss,
            'val/perplexity': perplexity,
        }

    def train(self):
        """Main training loop."""
        print("\n" + "="*70)
        print("STARTING TRAINING (Attention-Weighted Free Energy)")
        print("="*70)

        if TQDM_AVAILABLE:
            pbar = tqdm(total=self.config.max_steps, desc="Training")
        else:
            pbar = None

        start_time = time.time()

        try:
            while self.step < self.config.max_steps:
                for batch in self.train_loader:
                    # Training step
                    metrics = self.train_step(batch)

                    # Logging
                    if self.step % self.config.log_every == 0:
                        elapsed = time.time() - start_time
                        tokens_per_sec = (self.step * self.config.batch_size * batch[0].shape[1]) / elapsed

                        # Basic metrics
                        print(f"\nStep {self.step:6d} | Loss: {metrics['loss/total']:.4f} | "
                              f"CE: {metrics['loss/ce']:.4f} | Align: {metrics['loss/belief_align']:.4f} | "
                              f"LR: {metrics['lr']:.2e}")

                        # Gradient norms (verify gradients are flowing!)
                        grad_mu = metrics.get('grad/mu_embed', 0.0)
                        grad_out = metrics.get('grad/out_proj', 0.0)
                        grad_phi = metrics.get('grad/phi_embed', 0.0)
                        print(f"         Grads | μ_embed: {grad_mu:.4f} | out_proj: {grad_out:.4f} | φ_embed: {grad_phi:.4f}")

                        if self.config.use_wandb and WANDB_AVAILABLE:
                            wandb.log(metrics, step=self.step)

                    # Validation
                    if self.step % self.config.eval_every == 0 and self.step > 0:
                        val_metrics = self.validate()
                        if val_metrics:
                            print(f"\nValidation | Loss: {val_metrics['val/loss']:.4f} | "
                                  f"PPL: {val_metrics['val/perplexity']:.2f}")

                            if self.config.use_wandb and WANDB_AVAILABLE:
                                wandb.log(val_metrics, step=self.step)

                            # Save best model
                            if val_metrics['val/loss'] < self.best_val_loss:
                                self.best_val_loss = val_metrics['val/loss']
                                self.save_checkpoint('best_model.pt')

                    # Checkpointing
                    if self.step % self.config.save_every == 0 and self.step > 0:
                        self.save_checkpoint(f'checkpoint_step_{self.step}.pt')

                    # Update progress
                    if pbar is not None:
                        pbar.update(1)
                        pbar.set_postfix({'loss': f"{metrics['loss/total']:.4f}"})

                    self.step += 1

                    if self.step >= self.config.max_steps:
                        break

                self.epoch += 1

        except KeyboardInterrupt:
            print("\n⚠ Training interrupted by user")

        finally:
            if pbar is not None:
                pbar.close()

        # Final validation
        print("\n" + "="*70)
        print("TRAINING COMPLETE")
        print("="*70)

        final_metrics = self.validate()
        if final_metrics:
            print(f"Final Validation Loss: {final_metrics['val/loss']:.4f}")
            print(f"Final Perplexity: {final_metrics['val/perplexity']:.2f}")

        # Save final model
        self.save_checkpoint('final_model.pt')

        print("="*70)

    def save_checkpoint(self, filename: str):
        """Save training checkpoint."""
        if self.config.checkpoint_dir is None:
            return

        checkpoint_path = self.config.checkpoint_dir / filename

        checkpoint = {
            'step': self.step,
            'epoch': self.epoch,
            'model_state': self.model.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.model.config,
        }

        if self.config.save_optimizer:
            checkpoint['optimizer_state'] = self.optimizer.state_dict()
            if self.scheduler is not None:
                checkpoint['scheduler_state'] = self.scheduler.state_dict()

        torch.save(checkpoint, checkpoint_path)
        print(f"  💾 Saved checkpoint: {checkpoint_path.name}")

    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state'])

        if 'optimizer_state' in checkpoint and self.config.save_optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])

        if 'scheduler_state' in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state'])

        self.step = checkpoint.get('step', 0)
        self.epoch = checkpoint.get('epoch', 0)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))

        print(f"✓ Loaded checkpoint from step {self.step}")


# =============================================================================
# FastTrainer Alias (Backward Compatibility)
# =============================================================================

# DEPRECATED: FastTrainer is now an alias to the unified Trainer class.
# Use Trainer with config.use_param_groups=True instead.
#
# This alias exists for backward compatibility with existing code.
# The unified Trainer class supports both simple and multi-group optimization
# modes via the use_param_groups configuration flag.
FastTrainer = Trainer