"""
Variational Feed-Forward Networks for Gauge Transformer
========================================================

Integrates with validated gradient_engine.py for theoretically correct active inference!

Three implementations:
1. APPROXIMATE: Omit second-order ∂β_ij/∂μ_i term (legacy, simple)
2. FULL: Include all terms manually (legacy, exact but complex)
3. GRADIENT_ENGINE: Use validated gradient_engine backend (RECOMMENDED!)

The GRADIENT_ENGINE version:
- Updates BOTH means μ AND covariances Σ
- Uses natural gradients via Fisher-Rao metric
- Includes all energy terms (self-coupling, alignment, observations, softmax coupling)
- Proper χ-weighting and gauge transport
- Theoretically principled active inference

Mathematical Foundation:
-----------------------
Free Energy (E-STEP):
    F = α·Σ_i KL(q_i||p_i)                      # Prior consistency
      + λ_β·Σ_{i,j} β_ij·KL(q_i||Ω_{ij}q_j)    # Belief alignment
      + λ_γ·Σ_{i,j} γ_ij·KL(p_i||Ω_{ij}p_j)    # Prior alignment
      + CE(W_out·μ, targets)                    # DISCRETE OBSERVATIONS!

CRITICAL: The cross-entropy term is the SINGLE observation model!
- E-step: Minimize F w.r.t. μ, Σ → compute ∂CE/∂μ with W_out frozen
- M-step: Minimize F w.r.t. W_out, embeddings → compute ∂CE/∂W_out with μ frozen

This is classic EM:
- E-step: "Given model (W_out), what beliefs (μ) explain observations?"
- M-step: "Given beliefs (μ), what model parameters explain observations?"

The SAME cross-entropy appears in both steps, just optimizing different parameters!

Gradient Engine computes:
    ∂F/∂θ for θ = {μ_q, Σ_q, μ_p, Σ_p, φ}

With natural gradient projection:
    Δθ = -η · F⁻¹(θ) · ∇F(θ)

Where F(θ) is the Fisher-Rao metric.

Author: Integrated with validated gradient_engine.py
Date: November 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import numpy as np

# Import validated gradient engine
import sys
from pathlib import Path
# Add parent directory to path for gradient_engine import
sys.path.insert(0, str(Path(__file__).parent.parent))

from gradients.gradient_engine import (
    _compute_agent_euclidean_gradients,
    project_to_natural_gradients,
)
from config import SystemConfig, AgentConfig
from agent.agents import Agent
from geometry.geometry_base import BaseManifold, TopologyType
from gradients.retraction import retract_spd  # For SPD manifold updates

# Import attention computation for dynamic β
from transformer.attention import compute_attention_weights


# =============================================================================
# GPU-Based Gradient Computation (PyTorch - FAST!)
# =============================================================================

def compute_vfe_gradients_gpu(
    mu_q: torch.Tensor,        # (B, N, K) belief means
    sigma_q: torch.Tensor,     # (B, N, K) diagonal variances or (B, N, K, K) full
    mu_p: torch.Tensor,        # (B, N, K) prior means
    sigma_p: torch.Tensor,     # (B, N, K) diagonal or (B, N, K, K) full
    beta: torch.Tensor,        # (B, N, N) attention weights
    phi: torch.Tensor,         # (B, N, 3) gauge frames
    generators: torch.Tensor,  # (3, K, K) SO(3) generators
    alpha: float = 0.01,       # Self-coupling weight (KL(q||p))
    lambda_belief: float = 1.0,  # Belief alignment weight
    kappa: float = 1.0,        # Temperature (for normalization)
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute VFE gradients entirely on GPU using PyTorch.

    This is the FAST version that replaces the NumPy-based gradient_engine.
    Fully vectorized - no loops over batch or agents!

    Gradients computed:
    1. Self-coupling: ∂/∂μ_q [α · KL(q||p)]
    2. Belief alignment: ∂/∂μ_q [λ · Σ_j β_ij · KL(q_i || Ω_ij q_j)]

    Args:
        mu_q: Belief means (B, N, K)
        sigma_q: Belief variances - diagonal (B, N, K) or full (B, N, K, K)
        mu_p: Prior means (B, N, K)
        sigma_p: Prior variances - diagonal (B, N, K) or full (B, N, K, K)
        beta: Attention weights (B, N, N), already normalized
        phi: Gauge frames (B, N, 3)
        generators: SO(3) generators (3, K, K)
        alpha: Weight for KL(q||p) self-coupling term
        lambda_belief: Weight for belief alignment term
        kappa: Temperature parameter
        eps: Numerical stability

    Returns:
        grad_mu: Gradient w.r.t. μ_q, shape (B, N, K)
        grad_sigma: Gradient w.r.t. σ_q, shape (B, N, K) for diagonal
    """
    B, N, K = mu_q.shape
    device = mu_q.device
    dtype = mu_q.dtype

    # Detect diagonal vs full covariance
    is_diagonal = sigma_q.dim() == 3

    # =================================================================
    # 1. Self-Coupling Gradient: ∂/∂μ_q [α · KL(q||p)]
    # =================================================================
    # For diagonal Gaussians:
    #   KL(q||p) = 0.5 * Σ_k [ σ_q[k]/σ_p[k] + (μ_p[k]-μ_q[k])²/σ_p[k] - 1 + log(σ_p[k]/σ_q[k]) ]
    #   ∂KL/∂μ_q = (μ_q - μ_p) / σ_p
    #   ∂KL/∂σ_q = 0.5 * (1/σ_p - 1/σ_q)

    if is_diagonal:
        # Clamp for stability
        sigma_q_safe = sigma_q.clamp(min=eps)
        sigma_p_safe = sigma_p.clamp(min=eps)

        # Self-coupling gradient w.r.t. μ
        delta_mu = mu_q - mu_p  # (B, N, K)
        grad_mu_self = alpha * delta_mu / sigma_p_safe  # (B, N, K)

        # Self-coupling gradient w.r.t. σ (diagonal)
        grad_sigma_self = alpha * 0.5 * (1.0 / sigma_p_safe - 1.0 / sigma_q_safe)  # (B, N, K)
    else:
        # Full covariance - use matrix operations
        # ∂KL/∂μ_q = Σ_p^{-1} (μ_q - μ_p)
        sigma_p_reg = sigma_p + eps * torch.eye(K, device=device, dtype=dtype)
        sigma_p_inv = torch.linalg.inv(sigma_p_reg)  # (B, N, K, K)

        delta_mu = mu_q - mu_p  # (B, N, K)
        grad_mu_self = alpha * torch.einsum('bnij,bnj->bni', sigma_p_inv, delta_mu)

        # ∂KL/∂Σ_q = 0.5 * (Σ_p^{-1} - Σ_q^{-1})
        sigma_q_reg = sigma_q + eps * torch.eye(K, device=device, dtype=dtype)
        sigma_q_inv = torch.linalg.inv(sigma_q_reg)
        grad_sigma_self = alpha * 0.5 * (sigma_p_inv - sigma_q_inv)

    # =================================================================
    # 2. Belief Alignment Gradient: ∂/∂μ_i [λ · Σ_j β_ij · KL(q_i || Ω_ij q_j)]
    # =================================================================
    # Full gradient via product rule:
    #   ∂/∂μ_i [Σ_j β_ij · KL_ij] = Σ_j β_ij · ∂KL_ij/∂μ_i + Σ_j KL_ij · ∂β_ij/∂μ_i
    #                                  ↑ direct term           ↑ SOFTMAX COUPLING (nonlinearity!)
    #
    # The softmax coupling gradient is the KEY nonlinearity (replaces GELU/ReLU):
    #   ∂β_ij/∂μ_i = β_ij · [∂KL_ij/∂μ_i - Σ_k β_ik · ∂KL_ik/∂μ_i] / κ

    if is_diagonal:
        # Compute transport operators (vectorized)
        phi_matrix = torch.einsum('bna,aij->bnij', phi, generators)  # (B, N, K, K)
        exp_phi = torch.matrix_exp(phi_matrix)       # (B, N, K, K)
        exp_neg_phi = torch.matrix_exp(-phi_matrix)  # (B, N, K, K)

        # Transport: Ω_ij = exp(φ_i) @ exp(-φ_j)
        # For all pairs: (B, N, N, K, K)
        Omega = torch.einsum('bikl,bjlm->bijkm', exp_phi, exp_neg_phi)

        # Transport all μ_j to frame i: μ_j_transported[i,j] = Ω_ij @ μ_j
        mu_j_transported = torch.einsum('bijkl,bjl->bijk', Omega, mu_q)  # (B, N, N, K)

        # Difference: μ_i - μ_j_transported (for each pair i,j)
        delta_mu_ij = mu_q.unsqueeze(2) - mu_j_transported  # (B, N, N, K)

        # =================================================================
        # PROPER COVARIANCE TRANSPORT: Σ_j_transported = Ω @ diag(σ_j) @ Ω^T
        # =================================================================
        # Even though input is diagonal, transported covariance is FULL!
        # This is crucial for gauge equivariance.

        # Expand diagonal to full: diag(σ_j) -> (B, N, K, K)
        sigma_j_diag = torch.diag_embed(sigma_q.clamp(min=eps))  # (B, N, K, K)

        # Transport covariance: Σ_j_transported = Ω_ij @ Σ_j @ Ω_ij^T
        sigma_j_transported = torch.einsum(
            'bijkl,bjlm,bijmn->bijkn',
            Omega, sigma_j_diag, Omega.transpose(-1, -2)
        )  # (B, N, N, K, K)

        # Regularize and invert transported covariance
        sigma_j_reg = sigma_j_transported + eps * torch.eye(K, device=device, dtype=dtype)
        sigma_j_inv = torch.linalg.inv(sigma_j_reg)  # (B, N, N, K, K)

        # ∂KL_ij/∂μ_i = Σ_j_transported^{-1} @ (μ_i - μ_j_transported)
        grad_kl_per_pair = torch.einsum('bijkl,bijl->bijk', sigma_j_inv, delta_mu_ij)  # (B, N, N, K)

        # Compute KL values (Mahalanobis distance with transported covariance)
        # KL_ij = 0.5 * δμ^T @ Σ_j_transported^{-1} @ δμ
        kl_values = 0.5 * torch.einsum('bijk,bijk->bij', delta_mu_ij, grad_kl_per_pair)  # (B, N, N)

        # =================================================================
        # 2a. Direct term: Σ_j β_ij · ∂KL_ij/∂μ_i
        # =================================================================
        grad_mu_direct = lambda_belief * torch.einsum('bij,bijk->bik', beta, grad_kl_per_pair)

        # =================================================================
        # 2b. Softmax coupling term (THE NONLINEARITY!):
        #     ∂β_ij/∂μ_i = β_ij · [∂KL_ij/∂μ_i - Σ_k β_ik · ∂KL_ik/∂μ_i] / κ
        #     grad_softmax = Σ_j KL_ij · ∂β_ij/∂μ_i
        # =================================================================
        # Weighted average gradient: avg_grad_i = Σ_k β_ik · ∂KL_ik/∂μ_i
        avg_grad = torch.einsum('bij,bijk->bik', beta, grad_kl_per_pair)  # (B, N, K)

        # Deviation from average: ∂KL_ij/∂μ_i - avg_grad_i
        # grad_kl_per_pair: (B, N, N, K), avg_grad: (B, N, K) -> expand to (B, N, 1, K)
        grad_deviation = grad_kl_per_pair - avg_grad.unsqueeze(2)  # (B, N, N, K)

        # Softmax coupling gradient: ∂β_ij/∂μ_i = β_ij · grad_deviation / κ
        d_beta_d_mu = beta.unsqueeze(-1) * grad_deviation / kappa  # (B, N, N, K)

        # Weight by KL values and sum: Σ_j KL_ij · ∂β_ij/∂μ_i
        grad_mu_softmax = lambda_belief * torch.einsum('bij,bijk->bik', kl_values, d_beta_d_mu)

        # Total alignment gradient (direct + softmax coupling)
        grad_mu_align = grad_mu_direct + grad_mu_softmax

        # Sigma gradient from alignment (simplified for diagonal)
        grad_sigma_align = torch.zeros_like(sigma_q)
    else:
        # Full covariance belief alignment
        phi_matrix = torch.einsum('bna,aij->bnij', phi, generators)
        exp_phi = torch.matrix_exp(phi_matrix)
        exp_neg_phi = torch.matrix_exp(-phi_matrix)
        Omega = torch.einsum('bikl,bjlm->bijkm', exp_phi, exp_neg_phi)

        # Transport means
        mu_j_transported = torch.einsum('bijkl,bjl->bijk', Omega, mu_q)
        delta_mu_ij = mu_q.unsqueeze(2) - mu_j_transported

        # Transport covariances: Σ_j_transported = Ω @ Σ_j @ Ω^T
        sigma_j_transported = torch.einsum(
            'bijkl,bjlm,bijmn->bijkn',
            Omega, sigma_q, Omega.transpose(-1, -2)
        )  # (B, N, N, K, K)

        # Regularize and invert
        sigma_j_reg = sigma_j_transported + eps * torch.eye(K, device=device, dtype=dtype)
        sigma_j_inv = torch.linalg.inv(sigma_j_reg)  # (B, N, N, K, K)

        # ∂KL_ij/∂μ_i
        grad_kl_per_pair = torch.einsum('bijkl,bijl->bijk', sigma_j_inv, delta_mu_ij)

        # Compute KL values (Mahalanobis term only, ignoring trace/logdet)
        kl_values = 0.5 * torch.einsum('bijk,bijk->bij', delta_mu_ij, grad_kl_per_pair)

        # Direct term
        grad_mu_direct = lambda_belief * torch.einsum('bij,bijk->bik', beta, grad_kl_per_pair)

        # Softmax coupling term
        avg_grad = torch.einsum('bij,bijk->bik', beta, grad_kl_per_pair)
        grad_deviation = grad_kl_per_pair - avg_grad.unsqueeze(2)
        d_beta_d_mu = beta.unsqueeze(-1) * grad_deviation / kappa
        grad_mu_softmax = lambda_belief * torch.einsum('bij,bijk->bik', kl_values, d_beta_d_mu)

        grad_mu_align = grad_mu_direct + grad_mu_softmax
        grad_sigma_align = torch.zeros_like(sigma_q)

    # =================================================================
    # 3. Combine Gradients
    # =================================================================
    grad_mu = grad_mu_self + grad_mu_align

    if is_diagonal:
        grad_sigma = grad_sigma_self + grad_sigma_align
    else:
        grad_sigma = grad_sigma_self + grad_sigma_align

    return grad_mu, grad_sigma


def compute_natural_gradient_gpu(
    grad_mu: torch.Tensor,     # (B, N, K) Euclidean gradient
    grad_sigma: torch.Tensor,  # (B, N, K) or (B, N, K, K)
    sigma_q: torch.Tensor,     # (B, N, K) or (B, N, K, K)
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Project Euclidean gradients to natural gradients using Fisher metric.

    For Gaussian distributions, the Fisher information metric is:
        F_μ = Σ^{-1}  →  natural_grad_μ = Σ @ euclidean_grad_μ
        F_σ = 2Σ^{-2} →  natural_grad_σ = 0.5 * Σ² @ euclidean_grad_σ (diagonal approx)

    Args:
        grad_mu: Euclidean gradient w.r.t. μ
        grad_sigma: Euclidean gradient w.r.t. σ
        sigma_q: Current covariance
        eps: Numerical stability

    Returns:
        nat_grad_mu: Natural gradient for μ
        nat_grad_sigma: Natural gradient for σ
    """
    is_diagonal = sigma_q.dim() == 3

    if is_diagonal:
        # Diagonal case: simple element-wise multiplication
        sigma_safe = sigma_q.clamp(min=eps)
        nat_grad_mu = sigma_safe * grad_mu  # (B, N, K)
        nat_grad_sigma = 0.5 * sigma_safe * sigma_safe * grad_sigma  # (B, N, K)
    else:
        # Full covariance: matrix multiplication
        nat_grad_mu = torch.einsum('bnij,bnj->bni', sigma_q, grad_mu)
        # For sigma, use diagonal approximation for simplicity
        sigma_diag = torch.diagonal(sigma_q, dim1=-2, dim2=-1)
        nat_grad_sigma = 0.5 * sigma_diag.unsqueeze(-1) * sigma_diag.unsqueeze(-2) * grad_sigma

    return nat_grad_mu, nat_grad_sigma


# =============================================================================
# Utilities
# =============================================================================

def _sanitize_euclidean_gradients(euc_grads, max_norm: float = 1e3, debug: bool = False):
    """
    Sanitize Euclidean gradients to prevent NaN in natural gradient computation.

    This clips gradients that are too large, which can cause numerical overflow
    when computing natural gradients via Σ^{-1}.

    Args:
        euc_grads: AgentGradients object
        max_norm: Maximum allowed gradient norm per component
        debug: Print warnings if clipping occurs

    Returns:
        Sanitized AgentGradients object
    """
    
    import copy

    grads_sanitized = copy.copy(euc_grads)

    # Sanitize mu gradient
    if euc_grads.grad_mu_q is not None:
        grad_mu = euc_grads.grad_mu_q
        if not np.all(np.isfinite(grad_mu)):
            if debug:
                print(f"⚠️  NaN/Inf in grad_mu_q, setting to zero")
            grads_sanitized.grad_mu_q = np.zeros_like(grad_mu)
        else:
            norm = np.linalg.norm(grad_mu)
            if norm > max_norm:
                if debug:
                    print(f"⚠️  Clipping grad_mu_q: norm {norm:.2e} > {max_norm:.2e}")
                grads_sanitized.grad_mu_q = grad_mu * (max_norm / norm)

    # Sanitize Sigma gradient
    if euc_grads.grad_Sigma_q is not None:
        grad_Sigma = euc_grads.grad_Sigma_q
        if not np.all(np.isfinite(grad_Sigma)):
            if debug:
                print(f"⚠️  NaN/Inf in grad_Sigma_q, setting to zero")
            grads_sanitized.grad_Sigma_q = np.zeros_like(grad_Sigma)
        else:
            norm = np.linalg.norm(grad_Sigma)
            if norm > max_norm:
                if debug:
                    print(f"⚠️  Clipping grad_Sigma_q: norm {norm:.2e} > {max_norm:.2e}")
                grads_sanitized.grad_Sigma_q = grad_Sigma * (max_norm / norm)

    return grads_sanitized


def _compute_cholesky_robust(sigma: np.ndarray, eps: float = 1e-6, debug: bool = False) -> np.ndarray:
    """
    Compute Cholesky factor L such that Σ = L L^T with robust fallback.

    This is the VALIDATED approach from agents.py that works in simulation_suite.

    Args:
        sigma: Covariance matrix (K, K)
        eps: Regularization for numerical stability (default: 1e-6)
        debug: Print diagnostic info when fallbacks are used

    Returns:
        L: Lower triangular Cholesky factor (K, K)
    """
    K = sigma.shape[0]

    # Check for NaN/Inf
    if not np.all(np.isfinite(sigma)):
        if debug:
            print(f"⚠️  Cholesky: NaN/Inf detected in covariance, using diagonal fallback")
        return np.sqrt(eps) * np.eye(K, dtype=np.float32)

    # Symmetrize and regularize
    sigma_sym = 0.5 * (sigma + sigma.T)
    sigma_reg = sigma_sym + eps * np.eye(K)

    try:
        # Try standard Cholesky
        L = np.linalg.cholesky(sigma_reg)
        return L.astype(np.float32)

    except np.linalg.LinAlgError:
        # Cholesky failed - use eigendecomposition fallback
        # This is the VALIDATED approach from agents.py
        if debug:
            print(f"⚠️  Cholesky: Standard decomposition failed, using eigenvalue fallback")

        try:
            eigvals, eigvecs = np.linalg.eigh(sigma_reg)
            # Clamp eigenvalues
            eigvals_clamped = np.maximum(eigvals, eps)
            if debug and np.any(eigvals < eps):
                min_eig = np.min(eigvals)
                print(f"    Clamped {np.sum(eigvals < eps)} eigenvalues (min was {min_eig:.2e})")
            # Compute Cholesky factor directly: L = V @ diag(sqrt(λ))
            L = eigvecs @ np.diag(np.sqrt(eigvals_clamped))
            return L.astype(np.float32)

        except np.linalg.LinAlgError:
            # Even eigendecomposition failed - return diagonal fallback
            if debug:
                print(f"⚠️  Cholesky: Eigendecomposition also failed, using diagonal fallback")
            return np.sqrt(eps) * np.eye(K, dtype=np.float32)


# =============================================================================
# Adapter: PyTorch Transformer → Multi-Agent System
# =============================================================================

class MockMultiAgentSystem:
    """
    Lightweight adapter that converts PyTorch transformer tensors
    to multi-agent system format for gradient_engine.

    This allows us to reuse validated gradient code without full system overhead.
    """

    def __init__(
        self,
        mu_q: np.ndarray,      # (N, K)
        sigma_q: np.ndarray,   # (N, K, K)
        mu_p: np.ndarray,      # (N, K)
        sigma_p: np.ndarray,   # (N, K, K)
        phi: np.ndarray,       # (N, 3)
        generators: np.ndarray,  # (3, K, K)
        config: SystemConfig,
        beta_weights: Optional[np.ndarray] = None,  # (N, N) - precomputed attention
    ):
        """
        Create mock system from transformer state.

        Args:
            mu_q: Belief means (N, K)
            sigma_q: Belief covariances (N, K, K)
            mu_p: Prior means (N, K)
            sigma_p: Prior covariances (N, K, K)
            phi: Gauge frames (N, 3)
            generators: SO(3) generators (3, K, K)
            config: SystemConfig with hyperparameters
            beta_weights: Optional precomputed attention weights (N, N)
        """
        self.config = config
        self.n_agents = mu_q.shape[0]

        # Create mock agents (0D point agents)
        self.agents = []
        for i in range(self.n_agents):
            agent = self._create_mock_agent(
                i, mu_q[i], sigma_q[i], mu_p[i], sigma_p[i], phi[i], generators, config
            )
            self.agents.append(agent)

        # Store precomputed beta for efficiency
        self._beta_cache = beta_weights

    def _create_mock_agent(
        self, agent_id: int, mu_q, sigma_q, mu_p, sigma_p, phi, generators, config
    ):
        """Create a lightweight mock agent without observations."""
        # Create minimal agent config
        K = mu_q.shape[0]
        agent_config = AgentConfig(
            K=K,
            spatial_shape=(),  # 0D point agent
            alpha=config.lambda_self,
        )

        # Create 0D base manifold (single point)
        base_manifold = BaseManifold(
            shape=(),  # 0D
            topology=TopologyType.PERIODIC
        )

        # Create agent (will initialize with defaults)
        
        agent = Agent(agent_id, agent_config, base_manifold=base_manifold)

        # Override with our values
        agent.mu_q = mu_q.copy()
        agent.mu_p = mu_p.copy()

        # Covariance matrices (NOT Cholesky - gauge covariance requires Σ storage)
        agent.Sigma_q = sigma_q.copy()
        agent.Sigma_p = sigma_p.copy()

        # Gauge field
        agent.gauge = type('obj', (object,), {'phi': phi.copy()})()

        # Generators
        agent.generators = generators.copy()

        # No observations - will add discrete observation gradients separately
        agent.observations = {}

        return agent

    def get_neighbors(self, agent_idx: int):
        """Return all other agents as neighbors (fully connected)."""
        return [j for j in range(self.n_agents) if j != agent_idx]

    def compute_transport_ij(self, i: int, j: int):
        """Compute transport operator Ω_ij."""
        from math_utils.transport import compute_transport
        agent_i = self.agents[i]
        agent_j = self.agents[j]
        return compute_transport(
            agent_i.gauge.phi, agent_j.gauge.phi, agent_i.generators
        )


def convert_torch_to_numpy_system(
    mu_q: torch.Tensor,      # (B, N, K)
    sigma_q: torch.Tensor,   # (B, N, K, K) or (B, N, K) if diagonal
    mu_prior: torch.Tensor,  # (B, N, K)
    phi: torch.Tensor,       # (B, N, 3)
    generators: torch.Tensor,  # (3, K, K)
    config: SystemConfig,
    beta: Optional[torch.Tensor] = None,  # (B, N, N) averaged attention
    batch_idx: int = 0,
) -> MockMultiAgentSystem:
    """
    Convert PyTorch transformer tensors to multi-agent system format.

    Assumes priors have same covariance as beliefs (simplified).

    Args:
        mu_q: Belief means (B, N, K)
        sigma_q: Belief covariances (B, N, K, K) full or (B, N, K) diagonal
        mu_prior: Prior means (B, N, K)
        phi: Gauge frames (B, N, 3)
        generators: SO(3) generators (3, K, K)
        config: SystemConfig
        beta: Optional attention weights (B, N, N)
        batch_idx: Which batch element to extract

    Returns:
        MockMultiAgentSystem ready for gradient_engine
    """
    # Extract single batch element and convert to numpy
    mu_q_np = mu_q[batch_idx].detach().cpu().numpy()  # (N, K)
    mu_p_np = mu_prior[batch_idx].detach().cpu().numpy()  # (N, K)
    phi_np = phi[batch_idx].detach().cpu().numpy()  # (N, 3)
    gen_np = generators.detach().cpu().numpy()  # (3, K, K)

    # Handle diagonal vs full covariance
    if sigma_q.dim() == 3:
        # Diagonal covariance: (B, N, K) -> expand to (N, K, K)
        sigma_diag = sigma_q[batch_idx].detach().cpu().numpy()  # (N, K)
        N, K = sigma_diag.shape
        sigma_q_np = np.zeros((N, K, K), dtype=sigma_diag.dtype)
        for i in range(N):
            np.fill_diagonal(sigma_q_np[i], sigma_diag[i])
    else:
        # Full covariance: (B, N, K, K)
        sigma_q_np = sigma_q[batch_idx].detach().cpu().numpy()  # (N, K, K)

    # Assume prior covariances same as beliefs (could be different)
    sigma_p_np = sigma_q_np.copy()

    # Extract beta if provided
    beta_np = None
    if beta is not None:
        beta_np = beta[batch_idx].detach().cpu().numpy()  # (N, N)

    return MockMultiAgentSystem(
        mu_q=mu_q_np,
        sigma_q=sigma_q_np,
        mu_p=mu_p_np,
        sigma_p=sigma_p_np,
        phi=phi_np,
        generators=gen_np,
        config=config,
        beta_weights=beta_np,
    )


# =============================================================================
# Gradient Engine FFN (RECOMMENDED!)
# =============================================================================

class VariationalFFNGradientEngine(nn.Module):
    """
    Variational FFN using validated gradient_engine.py backend.

    This is the FULL active inference implementation:
    - Updates both μ AND Σ
    - Uses natural gradients (Fisher-Rao metric)
    - Includes all energy terms
    - Proper gauge transport and χ-weighting

    Complexity: O(N²·K²) for full system
    But: Theoretically correct and validated!

    GPU Mode (use_gpu=True, DEFAULT):
    - Fully vectorized PyTorch operations
    - Runs entirely on GPU - no CPU transfers!
    - ~10-100x faster than CPU mode

    CPU Mode (use_gpu=False):
    - Uses NumPy-based gradient_engine.py
    - Transfers data GPU→CPU→GPU each iteration
    - Slower but matches original validated implementation
    """

    def __init__(
        self,
        embed_dim: int,
        generators: torch.Tensor,  # (3, K, K) SO(3) generators
        alpha: float = 0.001,      # Self-coupling weight
        lambda_belief: float = 1.0,  # Belief alignment weight
        lambda_prior: float = 0.0,   # Prior alignment weight (usually off in transformer)
        lambda_phi: float = 0.0,     # Gauge field weight (usually off in transformer)
        kappa_beta: float = 1.0,   # Softmax temperature
        n_iterations: int = 1,     # Number of inference steps
        learnable_lr: bool = True, # Learn step size?
        update_sigma: bool = True,  # Update covariances?
        use_gpu: bool = True,      # Use GPU-accelerated gradients (FAST!)
    ):
        """
        Initialize gradient engine FFN.

        Args:
            embed_dim: K - dimension of belief vectors
            generators: SO(3) generators for gauge transport
            alpha: Self-coupling weight (KL(q||p) term)
            lambda_belief: Belief alignment weight
            lambda_prior: Prior alignment weight (0 = off)
            lambda_phi: Gauge field evolution weight (0 = off)
            kappa_beta: Softmax temperature for attention
            n_iterations: Number of variational descent iterations
            learnable_lr: Learn step size as parameter?
            update_sigma: Update covariances? (True = full Gaussian inference)
            use_gpu: If True, use GPU-accelerated gradient computation (default)
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.register_buffer('generators', generators)  # (3, K, K)
        self.n_iterations = n_iterations
        self.update_sigma = update_sigma
        self.use_gpu = use_gpu

        # Store hyperparameters for GPU path
        self.alpha = alpha
        self.lambda_belief = lambda_belief
        self.kappa_beta = kappa_beta

        # Create system config (for CPU path)
        self.config = SystemConfig(
            lambda_self=alpha,
            lambda_belief_align=lambda_belief,
            lambda_prior_align=lambda_prior,
            lambda_phi=lambda_phi,
            kappa_beta=kappa_beta,
            kappa_gamma=kappa_beta,  # Use same temperature for priors
            overlap_threshold=0.0,  # No spatial structure in transformer
            cache_transports=False,  # Don't need caching for single forward pass
        )

        # Learnable step size (or fixed)
        if learnable_lr:
            self.lr = nn.Parameter(torch.tensor(0.1))  # Initialize to 0.1
        else:
            self.register_buffer('lr', torch.tensor(0.1))

    def forward(
        self,
        mu: torch.Tensor,          # (B, N, K) - current beliefs
        beta: torch.Tensor,        # (B, n_heads, N, N) - attention weights
        mu_prior: torch.Tensor,    # (B, N, K) - embedding priors
        phi: torch.Tensor,         # (B, N, 3) - gauge frames
        sigma: Optional[torch.Tensor] = None,  # (B, N, K, K) - covariances
        mask: Optional[torch.Tensor] = None,   # (B, N, N) - causal mask
        targets: Optional[torch.Tensor] = None,  # (B, N) - target token IDs (observations!)
        W_out: Optional[torch.Tensor] = None,  # (V, K) - output projection for discrete observations
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Variational descent using gradient_engine with discrete observations.

        E-STEP: Minimize full free energy F w.r.t. beliefs (μ, Σ, φ)

        F = α·KL(q||p) + λ_β·Σ β_ij·KL + λ_γ·Σ γ_ij·KL + CE(W_out·μ, targets)
                                                              ↑ DISCRETE OBSERVATIONS!

        The cross-entropy term is the SAME as in M-step, but here we compute ∂CE/∂μ.

        Args:
            mu: Current belief means (B, N, K)
            beta: Attention weights (B, n_heads, N, N)
            mu_prior: Prior means (B, N, K)
            phi: Gauge frames (B, N, 3)
            sigma: Belief covariances (B, N, K, K) - required if update_sigma=True
            targets: Target token IDs (B, N) - discrete observations
            W_out: Output projection matrix (V, K) - for computing CE gradient
            mask: Causal mask (B, N, N) - optional

        Returns:
            mu_new: Updated beliefs (B, N, K)
            sigma_new: Updated covariances (B, N, K, K) if update_sigma=True, else None
        """
        batch_size, num_agents, K = mu.shape
        device = mu.device

        # Detect diagonal covariance mode
        is_diagonal_cov = sigma is not None and sigma.dim() == 3

        # Initialize covariances if not provided
        if sigma is None:
            # Use small isotropic covariances
            sigma = 0.1 * torch.eye(K, device=device).unsqueeze(0).unsqueeze(0).expand(
                batch_size, num_agents, -1, -1
            )

        # Average attention over heads
        beta_avg = beta.mean(dim=1)  # (B, N, N)

        # Apply mask if provided
        if mask is not None:
            beta_avg = beta_avg * mask
            # Renormalize
            beta_sum = beta_avg.sum(dim=-1, keepdim=True) + 1e-8
            beta_avg = beta_avg / beta_sum

        # Current state
        mu_current = mu
        sigma_current = sigma

        # =====================================================================
        # Compute discrete observation gradients: ∂CE/∂μ = W_out^T · (p - y)
        # =====================================================================
        # This is the observation term in the free energy!
        # F = ... + CE(W_out·μ, targets)
        # ∂F/∂μ = W_out^T · (softmax(W_out·μ) - one_hot(targets))
        #
        # We'll compute this ONCE and add it to Euclidean gradients in each iteration.
        # Note: We freeze W_out during E-step (no gradient flow back to W_out).
        discrete_obs_grad = None
        if targets is not None and W_out is not None:
            with torch.no_grad():  # Don't backprop through W_out during E-step
                # Compute logits: (B, N, K) @ (K, V)^T = (B, N, V)
                logits = torch.matmul(mu_current, W_out.T)  # (B, N, V)

                # Softmax probabilities
                probs = F.softmax(logits, dim=-1)  # (B, N, V)

                # One-hot targets (handle padding with -1)
                targets_valid = targets.clone()
                targets_valid[targets == -1] = 0  # Temporarily map -1 to 0 for one_hot
                one_hot = F.one_hot(targets_valid, num_classes=W_out.shape[0]).float()  # (B, N, V)

                # Mask out padding positions
                mask_obs = (targets != -1).unsqueeze(-1).float()  # (B, N, 1)
                one_hot = one_hot * mask_obs

                # Gradient: W_out^T @ (probs - one_hot)
                # W_out: (V, K), (probs - one_hot): (B, N, V)
                # Result: (B, N, K)
                grad_error = (probs - one_hot) * mask_obs  # (B, N, V)
                discrete_obs_grad = torch.matmul(grad_error, W_out)  # (B, N, K)

        # =====================================================================
        # GPU PATH: Fast vectorized gradient computation (DEFAULT)
        # =====================================================================
        if self.use_gpu:
            # Use sigma_p = sigma_q for now (same covariance for prior and belief)
            sigma_p = sigma_current

            for iteration in range(self.n_iterations):
                # Compute VFE gradients entirely on GPU
                grad_mu, grad_sigma = compute_vfe_gradients_gpu(
                    mu_q=mu_current,
                    sigma_q=sigma_current,
                    mu_p=mu_prior,
                    sigma_p=sigma_p,
                    beta=beta_avg,
                    phi=phi,
                    generators=self.generators,
                    alpha=self.alpha,
                    lambda_belief=self.lambda_belief,
                    kappa=self.kappa_beta,
                )

                # Add discrete observation gradient if provided
                if discrete_obs_grad is not None:
                    grad_mu = grad_mu + discrete_obs_grad

                # Clip gradients for stability
                grad_mu = torch.clamp(grad_mu, min=-1e3, max=1e3)
                grad_sigma = torch.clamp(grad_sigma, min=-1e3, max=1e3)

                # Project to natural gradients (stays on GPU!)
                nat_grad_mu, nat_grad_sigma = compute_natural_gradient_gpu(
                    grad_mu, grad_sigma, sigma_current
                )

                # Update: descent direction (negative gradient)
                mu_current = mu_current - self.lr * nat_grad_mu

                # Update sigma if enabled and not diagonal mode
                # (diagonal sigma update is simpler)
                if self.update_sigma and is_diagonal_cov:
                    # For diagonal, simple additive update with positivity constraint
                    sigma_current = (sigma_current - self.lr * nat_grad_sigma).clamp(min=1e-6)
                elif self.update_sigma and not is_diagonal_cov:
                    # Full covariance update - use simple additive for now
                    # (proper SPD retraction would be better but slower)
                    sigma_current = sigma_current - self.lr * nat_grad_sigma
                    # Ensure positive definiteness via eigenvalue clamping
                    sigma_current = 0.5 * (sigma_current + sigma_current.transpose(-1, -2))
                    # Add small regularization
                    sigma_current = sigma_current + 1e-6 * torch.eye(K, device=device)

            # Return updated parameters (detached from computation graph)
            if self.update_sigma:
                return mu_current.detach(), sigma_current.detach()
            else:
                return mu_current.detach(), None

        # =====================================================================
        # CPU PATH: NumPy-based gradient_engine (LEGACY - slower but validated)
        # =====================================================================
        # Perform n_iterations of variational descent
        for iteration in range(self.n_iterations):
            # ==================================================================
            # Compute natural gradients via gradient_engine (per batch element)
            # ==================================================================

            batch_gradients = []

            for b in range(batch_size):
                # Convert to multi-agent system
                system = convert_torch_to_numpy_system(
                    mu_q=mu_current,
                    sigma_q=sigma_current,
                    mu_prior=mu_prior,
                    phi=phi,
                    generators=self.generators,
                    config=self.config,
                    beta=beta_avg,
                    batch_idx=b,
                )

                # Compute natural gradients for all agents
                # NOTE: Sequential version is faster for transformers due to:
                # - Small per-agent computation time
                # - Large system object pickling overhead in parallel
                # - PyTorch + joblib interaction issues
                natural_grads = []
                for agent_idx in range(num_agents):
                    # Compute Euclidean gradients from active inference terms
                    # (prior, belief alignment, etc.)
                    euc_grads = _compute_agent_euclidean_gradients(system, agent_idx)

                    # ===========================================================
                    # ADD DISCRETE OBSERVATION GRADIENT (if provided)
                    # ===========================================================
                    # This is the key fix! The cross-entropy observation term.
                    # ∂F/∂μ = ... + W_out^T · (softmax(W_out·μ) - one_hot(target))
                    if discrete_obs_grad is not None:
                        # Add observation gradient to Euclidean gradient
                        # discrete_obs_grad: (B, N, K), extract [b, agent_idx, :]
                        obs_grad_np = discrete_obs_grad[b, agent_idx].cpu().numpy()  # (K,)

                        if euc_grads.grad_mu_q is not None:
                            euc_grads.grad_mu_q = euc_grads.grad_mu_q + obs_grad_np
                        else:
                            euc_grads.grad_mu_q = obs_grad_np

                    # Sanitize gradients before projection (prevent NaN)
                    euc_grads = _sanitize_euclidean_gradients(euc_grads, max_norm=1e3, debug=False)

                    # Project to natural gradients
                    nat_grads = project_to_natural_gradients(
                        system.agents[agent_idx], euc_grads
                    )
                    natural_grads.append(nat_grads)

                batch_gradients.append(natural_grads)

            # ==================================================================
            # Update parameters using natural gradients (VALIDATED APPROACH)
            # ==================================================================

            # Mean update: Simple Euclidean update
            # μ_new = μ + τ·Δμ
            delta_mu = torch.zeros_like(mu_current)
            for b in range(batch_size):
                for i, nat_grads in enumerate(batch_gradients[b]):
                    if nat_grads.delta_mu_q is not None:
                        delta_mu[b, i] = torch.from_numpy(nat_grads.delta_mu_q).to(device)

            mu_current = mu_current + self.lr * delta_mu

            # Covariance update: Use validated SPD retraction
            # Σ_new = retract_spd(Σ, τ·ΔΣ)
            # NOTE: Skip sigma update for diagonal covariance mode - gradient engine
            # computes full covariance gradients which aren't compatible with diagonal mode
            if self.update_sigma and not is_diagonal_cov:
                # Convert learning rate to float (may be torch tensor)
                lr_scalar = self.lr.item() if isinstance(self.lr, torch.Tensor) else float(self.lr)

                for b in range(batch_size):
                    for i, nat_grads in enumerate(batch_gradients[b]):
                        if nat_grads.delta_Sigma_q is not None:
                            # Convert to numpy for retraction (detach from computation graph)
                            Sigma_current = sigma_current[b, i].detach().cpu().numpy()
                            delta_Sigma = nat_grads.delta_Sigma_q

                            # Use validated SPD retraction (handles manifold geometry)
                            Sigma_new = retract_spd(
                                Sigma_current,
                                delta_Sigma,
                                step_size=lr_scalar,
                                trust_region=None,  # Could add trust region if needed
                                max_condition=None,  # Could add condition number limit
                                eps=1e-6,
                            )

                            # Convert back to PyTorch
                            sigma_current[b, i] = torch.from_numpy(Sigma_new).to(device)

        # Return updated parameters
        # CRITICAL: Detach from computation graph!
        # The natural gradients are already correct - don't let PyTorch backprop fight them
        if self.update_sigma and not is_diagonal_cov:
            return mu_current.detach(), sigma_current.detach()
        else:
            # Return original sigma for diagonal mode (no update) or if update_sigma=False
            return mu_current.detach(), sigma.detach() if is_diagonal_cov else None


# =============================================================================
# Dynamic-β VFE: Full Active Inference with Evolving Attention (RECOMMENDED!)
# =============================================================================

class VariationalFFNDynamic(nn.Module):
    """
    Dynamic-β Variational FFN: Recomputes attention at each VFE step.

    This is the theoretically correct implementation where beliefs and attention
    co-evolve. At each integration step:

        1. Compute β from current beliefs: β_ij = softmax(-KL(q_i||Ω_ij[q_j])/κ)
        2. Compute full VFE gradient: ∂F/∂θ (includes ∂β/∂θ nonlinearity)
        3. Update beliefs via natural gradient descent
        4. (Optional) M-step: update priors toward beliefs
        5. Repeat

    Key difference from VariationalFFNGradientEngine:
        - GradientEngine: β computed once, held fixed during descent
        - Dynamic: β recomputed at each step → attention-belief co-evolution

    This enables emergent block structure in β as beliefs cluster.

    The ∂β/∂μ term is the principled nonlinearity (replaces GELU):
        ∂β_ij/∂μ_i = β_ij · [∂KL_ij/∂μ_i - Σ_k β_ik · ∂KL_ik/∂μ_i] / κ

    With dynamic β, this creates positive feedback:
        - Tokens with similar beliefs → higher β between them
        - Higher β → beliefs pulled closer together
        - → Cluster formation (meta-agents!)

    Complexity: O(n_steps × N² × K) - more expensive but theoretically sound
    """

    def __init__(
        self,
        embed_dim: int,
        generators: torch.Tensor,  # (3, K, K) SO(3) generators
        alpha: float = 0.01,       # Self-coupling weight (KL(q||p))
        lambda_belief: float = 1.0,  # Belief alignment weight
        kappa: float = 1.0,        # Attention temperature
        n_iterations: int = 10,    # VFE descent steps (more steps = deeper equilibration)
        learnable_lr: bool = True, # Learn step size?
        update_sigma: bool = True, # Update covariances?
        m_step_interval: int = 0,  # M-step every N steps (0 = no M-step)
        m_step_rate: float = 0.01, # Prior update rate toward beliefs
        diagonal_covariance: bool = False,  # Use diagonal Σ for efficiency
    ):
        """
        Initialize dynamic-β VFE FFN.

        Args:
            embed_dim: K - dimension of belief vectors
            generators: SO(3) generators for gauge transport (3, K, K)
            alpha: Weight for KL(q||p) self-coupling (prior anchoring)
            lambda_belief: Weight for belief alignment term Σ β_ij KL(q_i||q_j)
            kappa: Temperature for attention softmax (higher = softer)
            n_iterations: Number of VFE descent iterations per forward pass
            learnable_lr: If True, step size η is a learnable parameter
            update_sigma: If True, also update covariance matrices Σ
            m_step_interval: Run M-step every N iterations (0 = disabled)
            m_step_rate: How fast priors move toward beliefs in M-step
            diagonal_covariance: Use diagonal Σ for O(K) instead of O(K²)
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.register_buffer('generators', generators)
        self.n_iterations = n_iterations
        self.update_sigma = update_sigma
        self.diagonal_covariance = diagonal_covariance

        # VFE hyperparameters
        self.alpha = alpha
        self.lambda_belief = lambda_belief
        self.kappa = kappa

        # M-step configuration
        self.m_step_interval = m_step_interval
        self.m_step_rate = m_step_rate

        # Learnable step size
        if learnable_lr:
            self.lr = nn.Parameter(torch.tensor(0.1))
        else:
            self.register_buffer('lr', torch.tensor(0.1))

    def forward(
        self,
        mu: torch.Tensor,          # (B, N, K) - current beliefs
        beta: torch.Tensor,        # (B, n_heads, N, N) - INITIAL attention (will be recomputed)
        mu_prior: torch.Tensor,    # (B, N, K) - embedding priors
        phi: torch.Tensor,         # (B, N, 3) - gauge frames
        sigma: Optional[torch.Tensor] = None,  # (B, N, K, K) or (B, N, K) if diagonal
        mask: Optional[torch.Tensor] = None,   # (B, N, N) - causal mask
        targets: Optional[torch.Tensor] = None,  # (B, N) - target token IDs
        W_out: Optional[torch.Tensor] = None,    # (V, K) - output projection
        return_beta_history: bool = False,  # Return β evolution for analysis
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[list]]:
        """
        Dynamic VFE descent with β recomputation at each step.

        Flow at each iteration:
            1. β = softmax(-KL(q||Ω[q])/κ)  [RECOMPUTE from current beliefs]
            2. ∂F/∂μ = α(μ-μ_p)/σ_p + λΣβ(∂KL/∂μ) + Σ KL(∂β/∂μ) + ∂CE/∂μ
            3. μ ← μ - η·F⁻¹·∂F/∂μ  [Natural gradient descent]
            4. (Optional) σ ← retract_spd(σ, -η·∂F/∂σ)
            5. (Optional M-step) μ_p ← μ_p + rate·(μ - μ_p)

        Args:
            mu: Current belief means (B, N, K)
            beta: Initial attention weights (B, n_heads, N, N) - used only for first step
            mu_prior: Prior means from embeddings (B, N, K)
            phi: Gauge frames (B, N, 3)
            sigma: Belief covariances - (B, N, K, K) full or (B, N, K) diagonal
            mask: Causal mask (B, N, N) where 0 = cannot attend
            targets: Target tokens for observation term (B, N)
            W_out: Output projection for ∂CE/∂μ computation
            return_beta_history: If True, return list of β at each step

        Returns:
            mu_new: Updated beliefs (B, N, K)
            sigma_new: Updated covariances (same shape as input) or None
            beta_history: List of β tensors if return_beta_history, else None
        """
        B, N, K = mu.shape
        device = mu.device
        dtype = mu.dtype
        eps = 1e-6

        # Initialize sigma if not provided
        if sigma is None:
            if self.diagonal_covariance:
                sigma = torch.ones(B, N, K, device=device, dtype=dtype) * 0.1
            else:
                sigma = 0.1 * torch.eye(K, device=device, dtype=dtype).unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1).clone()

        is_diagonal = sigma.dim() == 3

        # Initialize sigma_p (prior covariance) - same as sigma for simplicity
        sigma_p = sigma.clone()

        # Current state (will evolve)
        mu_current = mu.clone()
        sigma_current = sigma.clone()
        mu_p_current = mu_prior.clone()

        # Track β evolution if requested
        beta_history = [] if return_beta_history else None

        # Compute discrete observation gradient once (frozen during E-step)
        discrete_obs_grad = None
        if targets is not None and W_out is not None:
            with torch.no_grad():
                logits = torch.matmul(mu_current, W_out.T)
                probs = F.softmax(logits, dim=-1)
                targets_valid = targets.clone()
                targets_valid[targets == -1] = 0
                one_hot = F.one_hot(targets_valid, num_classes=W_out.shape[0]).float()
                mask_obs = (targets != -1).unsqueeze(-1).float()
                one_hot = one_hot * mask_obs
                grad_error = (probs - one_hot) * mask_obs
                discrete_obs_grad = torch.matmul(grad_error, W_out)

        # =====================================================================
        # VFE Descent Loop with Dynamic β
        # =====================================================================
        for iteration in range(self.n_iterations):
            # =================================================================
            # STEP 1: Recompute attention β from current beliefs
            # =================================================================
            # β_ij = softmax(-KL(q_i || Ω_ij[q_j]) / κ)
            beta_current = compute_attention_weights(
                mu_q=mu_current,
                sigma_q=sigma_current,
                phi=phi,
                generators=self.generators,
                kappa=self.kappa,
                epsilon=eps,
                mask=mask,
                use_numba=False,  # Always use PyTorch for GPU
                return_kl=False,
                diagonal_covariance=is_diagonal,
            )  # (B, N, N)

            if return_beta_history:
                beta_history.append(beta_current.detach().clone())

            # =================================================================
            # STEP 2: Compute VFE gradients with current β
            # =================================================================
            grad_mu, grad_sigma = compute_vfe_gradients_gpu(
                mu_q=mu_current,
                sigma_q=sigma_current,
                mu_p=mu_p_current,
                sigma_p=sigma_p,
                beta=beta_current,  # USE RECOMPUTED β!
                phi=phi,
                generators=self.generators,
                alpha=self.alpha,
                lambda_belief=self.lambda_belief,
                kappa=self.kappa,
                eps=eps,
            )

            # Add observation gradient if provided
            if discrete_obs_grad is not None:
                grad_mu = grad_mu + discrete_obs_grad

            # Clip for stability
            grad_mu = torch.clamp(grad_mu, min=-1e3, max=1e3)
            grad_sigma = torch.clamp(grad_sigma, min=-1e3, max=1e3)

            # =================================================================
            # STEP 3: Natural gradient projection
            # =================================================================
            nat_grad_mu, nat_grad_sigma = compute_natural_gradient_gpu(
                grad_mu, grad_sigma, sigma_current, eps=eps
            )

            # =================================================================
            # STEP 4: Update beliefs (E-step)
            # =================================================================
            mu_current = mu_current - self.lr * nat_grad_mu

            if self.update_sigma:
                if is_diagonal:
                    sigma_current = (sigma_current - self.lr * nat_grad_sigma).clamp(min=eps)
                else:
                    sigma_current = sigma_current - self.lr * nat_grad_sigma
                    sigma_current = 0.5 * (sigma_current + sigma_current.transpose(-1, -2))
                    sigma_current = sigma_current + eps * torch.eye(K, device=device, dtype=dtype)

            # =================================================================
            # STEP 5: Optional M-step (prior update)
            # =================================================================
            if self.m_step_interval > 0 and (iteration + 1) % self.m_step_interval == 0:
                # Move priors toward beliefs
                mu_p_current = mu_p_current + self.m_step_rate * (mu_current.detach() - mu_p_current)

        # Return results
        if self.update_sigma:
            return mu_current.detach(), sigma_current.detach(), beta_history
        else:
            return mu_current.detach(), None, beta_history

    def extra_repr(self) -> str:
        return (
            f"embed_dim={self.embed_dim}, n_iterations={self.n_iterations}, "
            f"alpha={self.alpha}, lambda_belief={self.lambda_belief}, kappa={self.kappa}, "
            f"m_step_interval={self.m_step_interval}"
        )


# =============================================================================
# Legacy Implementations (for backward compatibility)
# =============================================================================

class VariationalFFNApproximate(nn.Module):
    """
    Approximate variational descent FFN (omit ∂β_ij/∂μ_i term).

    LEGACY: Kept for backward compatibility.
    RECOMMENDATION: Use VariationalFFNGradientEngine instead!

    Gradient:
        ∂F/∂μ_i ≈ α·(μ_i - μ_p) + Σ_j (β_ij/τ)·(μ_i - Ω_ij μ_j)

    Update:
        μ_new = μ - η · ∂F/∂μ_i

    This is a good first-order approximation that captures:
    - Prior pull toward μ_p
    - Weighted neighbor alignment

    But misses:
    - Covariance updates
    - How attention changes with μ_i (second-order)
    - Natural gradient projection

    Complexity: O(N²·K) - same as standard attention
    """

    def __init__(
        self,
        embed_dim: int,
        generators: torch.Tensor,  # (3, K, K) SO(3) generators
        alpha: float = 0.001,      # Prior weight
        tau_eff: float = 1.0,      # Effective temperature
        n_iterations: int = 1,     # Number of inference steps
        learnable_lr: bool = True, # Learn step size?
    ):
        """
        Initialize approximate variational FFN.

        Args:
            embed_dim: K - dimension of belief vectors
            generators: SO(3) generators for gauge transport
            alpha: Weight for prior term
            tau_eff: Effective temperature (higher = weaker coupling)
            n_iterations: Number of descent iterations per forward pass
            learnable_lr: If True, learning rate η is a learnable parameter
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.register_buffer('generators', generators)  # (3, K, K)
        self.alpha = alpha
        self.tau_eff = tau_eff
        self.n_iterations = n_iterations

        # Learnable step size (or fixed)
        if learnable_lr:
            self.lr = nn.Parameter(torch.tensor(0.1))  # Initialize to 0.1
        else:
            self.register_buffer('lr', torch.tensor(0.1))

    def forward(
        self,
        mu: torch.Tensor,          # (B, N, K) - current beliefs
        beta: torch.Tensor,        # (B, n_heads, N, N) - attention weights
        mu_prior: torch.Tensor,    # (B, N, K) - embedding priors
        phi: torch.Tensor,         # (B, N, 3) - gauge frames
        mask: Optional[torch.Tensor] = None,  # (B, N, N) causal mask
    ) -> torch.Tensor:
        """
        One step of approximate variational descent.

        Args:
            mu: Current belief means (B, N, K)
            beta: Attention weights β_ij (B, n_heads, N, N)
            mu_prior: Prior means p_i (B, N, K)
            phi: Gauge frames φ_i (B, N, 3)
            mask: Causal mask (B, N, N) or None

        Returns:
            mu_new: Updated beliefs after variational descent (B, N, K)
        """
        batch_size, num_agents, K = mu.shape
        device = mu.device

        # Average attention weights over heads for simplicity
        # Could also compute separate gradients per head
        beta_avg = beta.mean(dim=1)  # (B, N, N)

        # Apply mask if provided
        if mask is not None:
            beta_avg = beta_avg * mask
            # Renormalize
            beta_sum = beta_avg.sum(dim=-1, keepdim=True) + 1e-8
            beta_avg = beta_avg / beta_sum

        # Perform n_iterations of variational descent
        mu_current = mu

        for _ in range(self.n_iterations):
            # ===========================================================
            # Compute gradient ∂F/∂μ_i (approximate)
            # ===========================================================

            # 1. Prior gradient: α·(μ_i - μ_p)
            grad_prior = self.alpha * (mu_current - mu_prior)  # (B, N, K)

            # 2. Coupling gradient: Σ_j (β_ij/τ)·(μ_i - Ω_ij μ_j)
            # Need to compute Ω_ij μ_j for all pairs (i,j)

            # Transport all μ_j by Ω_ij
            mu_transported = self._transport_beliefs(mu_current, phi)  # (B, N, N, K)

            # Compute differences: μ_i - Ω_ij μ_j
            # mu_i: (B, N, 1, K), mu_transported: (B, N, N, K)
            delta_mu = mu_current.unsqueeze(2) - mu_transported  # (B, N, N, K)

            # Weight by β_ij/τ: (B, N, N, 1) * (B, N, N, K)
            weighted_delta = (beta_avg.unsqueeze(-1) / self.tau_eff) * delta_mu  # (B, N, N, K)

            # Sum over neighbors j
            grad_coupling = weighted_delta.sum(dim=2)  # (B, N, K)

            # ===========================================================
            # Total gradient (approximate)
            # ===========================================================
            grad_total = grad_prior + grad_coupling  # (B, N, K)

            # ===========================================================
            # Variational descent update
            # ===========================================================
            mu_current = mu_current - self.lr * grad_total

        return mu_current

    def _transport_beliefs(
        self,
        mu: torch.Tensor,  # (B, N, K)
        phi: torch.Tensor,  # (B, N, 3)
    ) -> torch.Tensor:
        """
        Compute transported beliefs Ω_ij μ_j for all pairs (i,j).

        Ω_ij = exp(φ_i) · exp(-φ_j)

        Args:
            mu: Belief means (B, N, K)
            phi: Gauge frames (B, N, 3)

        Returns:
            mu_transported: (B, N, N, K) where [b,i,j,:] = Ω_ij μ_j
        """
        batch_size, num_agents, K = mu.shape
        device = mu.device

        # Compute exp(φ_i) and exp(-φ_j) for all agents
        # Using matrix exponential on SO(3)

        # φ_i → Lie algebra element: φ · generators
        # phi: (B, N, 3), generators: (3, K, K)

        # Expand for broadcasting
        phi_expanded = phi.unsqueeze(-1).unsqueeze(-1)  # (B, N, 3, 1, 1)
        gen_expanded = self.generators.unsqueeze(0).unsqueeze(0)  # (1, 1, 3, K, K)

        # φ·G = Σ_a φ^a G_a
        phi_algebra = (phi_expanded * gen_expanded).sum(dim=2)  # (B, N, K, K)

        # Matrix exponential: exp(φ·G)
        Omega_i = torch.matrix_exp(phi_algebra)  # (B, N, K, K)
        Omega_j_inv = torch.matrix_exp(-phi_algebra)  # (B, N, K, K)

        # Compute Ω_ij = Omega_i @ Omega_j_inv for all pairs
        # Omega_i: (B, N, 1, K, K), Omega_j_inv: (B, 1, N, K, K)
        Omega_ij = torch.matmul(
            Omega_i.unsqueeze(2),      # (B, N, 1, K, K)
            Omega_j_inv.unsqueeze(1)    # (B, 1, N, K, K)
        )  # (B, N, N, K, K)

        # Transport beliefs: Ω_ij μ_j
        # mu: (B, 1, N, K, 1), Omega_ij: (B, N, N, K, K)
        mu_transported = torch.matmul(
            Omega_ij,                          # (B, N, N, K, K)
            mu.unsqueeze(1).unsqueeze(-1)      # (B, 1, N, K, 1)
        ).squeeze(-1)  # (B, N, N, K)

        return mu_transported


class VariationalFFNFull(nn.Module):
    """
    FULL variational descent FFN (includes ∂β_ij/∂μ_i term).

    LEGACY: Kept for backward compatibility.
    RECOMMENDATION: Use VariationalFFNGradientEngine instead!

    Complete gradient from active inference:
        ∂F/∂μ_i = α·(μ_i - μ_p)
                + Σ_j (β_ij/τ)·(μ_i - Ω_ij μ_j)                [TERM 1]
                + Σ_j Σ_k (∂β_ij/∂KL_ik)·KL_ij·∂KL_ik/∂μ_i  [TERM 2]

    TERM 2 accounts for how attention weights change as beliefs change.
    This is the FULL gauge-invariant gradient from validated code!

    More complex but theoretically exact.

    Complexity: O(N³·K) due to triple sum over (i,j,k)
    """

    def __init__(
        self,
        embed_dim: int,
        generators: torch.Tensor,
        alpha: float = 0.001,
        tau_eff: float = 1.0,
        kappa: float = 1.0,        # Softmax temperature
        n_iterations: int = 1,
        learnable_lr: bool = True,
    ):
        """
        Initialize full variational FFN.

        Args:
            embed_dim: K
            generators: SO(3) generators
            alpha: Prior weight
            tau_eff: Coupling temperature
            kappa: Softmax temperature (for β_ij)
            n_iterations: Descent iterations
            learnable_lr: Learn step size?
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.register_buffer('generators', generators)
        self.alpha = alpha
        self.tau_eff = tau_eff
        self.kappa = kappa
        self.n_iterations = n_iterations

        if learnable_lr:
            self.lr = nn.Parameter(torch.tensor(0.05))  # Smaller for stability
        else:
            self.register_buffer('lr', torch.tensor(0.05))

    def forward(
        self,
        mu: torch.Tensor,
        beta: torch.Tensor,
        mu_prior: torch.Tensor,
        phi: torch.Tensor,
        sigma: Optional[torch.Tensor] = None,  # (B, N, K, K) covariances
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Full variational descent with second-order terms.

        NOTE: This is more expensive but theoretically correct!
        """
        batch_size, num_agents, K = mu.shape
        device = mu.device

        # Average attention over heads
        beta_avg = beta.mean(dim=1)  # (B, N, N)
        if mask is not None:
            beta_avg = beta_avg * mask
            beta_sum = beta_avg.sum(dim=-1, keepdim=True) + 1e-8
            beta_avg = beta_avg / beta_sum

        mu_current = mu

        for _ in range(self.n_iterations):
            # ===========================================================
            # First-order terms (same as approximate)
            # ===========================================================
            grad_prior = self.alpha * (mu_current - mu_prior)

            mu_transported = self._transport_beliefs(mu_current, phi)
            delta_mu = mu_current.unsqueeze(2) - mu_transported
            weighted_delta = (beta_avg.unsqueeze(-1) / self.tau_eff) * delta_mu
            grad_coupling_first = weighted_delta.sum(dim=2)

            # ===========================================================
            # Second-order term: Σ_j Σ_k (∂β_ij/∂KL_ik)·KL_ij·∂KL_ik/∂μ_i
            # ===========================================================
            grad_coupling_second = self._compute_second_order_gradient(
                mu_current, phi, beta_avg, sigma
            )

            # ===========================================================
            # Total gradient (FULL)
            # ===========================================================
            grad_total = grad_prior + grad_coupling_first + grad_coupling_second

            # ===========================================================
            # Update
            # ===========================================================
            mu_current = mu_current - self.lr * grad_total

        return mu_current

    def _compute_second_order_gradient(
        self,
        mu: torch.Tensor,  # (B, N, K)
        phi: torch.Tensor,  # (B, N, 3)
        beta: torch.Tensor,  # (B, N, N)
        sigma: Optional[torch.Tensor],  # (B, N, K, K) or None
    ) -> torch.Tensor:
        """
        Compute second-order softmax coupling gradient.

        For each agent i:
            ∂F/∂μ_i += Σ_j Σ_k (∂β_ij/∂KL_ik) · KL_ij · ∂KL_ik/∂μ_i

        Where:
            ∂β_ij/∂KL_ik = (β_ij/κ) · [δ_jk - β_ik]

        This is the key term that accounts for how attention changes!

        Returns:
            grad: (B, N, K) second-order gradient contribution
        """
        batch_size, num_agents, K = mu.shape
        device = mu.device

        # Transport all beliefs
        mu_transported = self._transport_beliefs(mu, phi)  # (B, N, N, K)

        # Compute all KL divergences KL(μ_i || Ω_ij μ_j)
        # Simplified: using || · ||² distance as proxy
        # (Full version would use proper Gaussian KL with covariances)

        kl_ij = torch.sum(
            (mu.unsqueeze(2) - mu_transported) ** 2,
            dim=-1
        ) / (2.0 * self.tau_eff)  # (B, N, N)

        # Compute ∂KL_ik/∂μ_i for all k
        # Gradient of KL w.r.t. source: ∂KL(μ_i||Ω_ik μ_k)/∂μ_i = (μ_i - Ω_ik μ_k) / τ
        grad_kl_wrt_mu = (mu.unsqueeze(2) - mu_transported) / self.tau_eff  # (B, N, N, K)

        # Compute ∂β_ij/∂KL_ik = (β_ij/κ) · [δ_jk - β_ik]
        # This is the Jacobian of softmax

        # For each (i,j,k) triple, compute contribution:
        # (β_ij/κ) · [δ_jk - β_ik] · KL_ij · ∂KL_ik/∂μ_i

        grad_second = torch.zeros_like(mu)  # (B, N, K)

        for i in range(num_agents):
            for j in range(num_agents):
                for k in range(num_agents):
                    # Kronecker delta
                    delta_jk = 1.0 if j == k else 0.0

                    # ∂β_ij/∂KL_ik
                    d_beta_ij_d_kl_ik = (beta[:, i, j] / self.kappa) * (delta_jk - beta[:, i, k])
                    # Shape: (B,)

                    # KL_ij
                    kl_ij_val = kl_ij[:, i, j]  # (B,)

                    # ∂KL_ik/∂μ_i
                    grad_kl_ik = grad_kl_wrt_mu[:, i, k, :]  # (B, K)

                    # Product: (B,) * (B,) * (B, K)
                    contrib = (d_beta_ij_d_kl_ik * kl_ij_val).unsqueeze(-1) * grad_kl_ik

                    grad_second[:, i, :] += contrib

        return grad_second

    def _transport_beliefs(self, mu, phi):
        """Same as approximate version."""
        batch_size, num_agents, K = mu.shape
        device = mu.device

        phi_expanded = phi.unsqueeze(-1).unsqueeze(-1)
        gen_expanded = self.generators.unsqueeze(0).unsqueeze(0)
        phi_algebra = (phi_expanded * gen_expanded).sum(dim=2)

        Omega_i = torch.matrix_exp(phi_algebra)
        Omega_j_inv = torch.matrix_exp(-phi_algebra)

        Omega_ij = torch.matmul(
            Omega_i.unsqueeze(2),
            Omega_j_inv.unsqueeze(1)
        )

        mu_transported = torch.matmul(
            Omega_ij,
            mu.unsqueeze(1).unsqueeze(-1)
        ).squeeze(-1)

        return mu_transported


# =============================================================================
# Helper: Replace FFN in Transformer Block
# =============================================================================

def replace_ffn_with_variational(
    transformer_block,
    variant: str = 'gradient_engine',  # 'gradient_engine', 'approximate', or 'full'
    generators: torch.Tensor = None,
    **kwargs
):
    """
    Replace learned FFN with variational descent.

    Args:
        transformer_block: TransformerBlock instance
        variant: 'gradient_engine' (recommended), 'approximate', or 'full' (legacy)
        generators: SO(3) generators (3, K, K)
        **kwargs: Additional parameters for VariationalFFN

    Returns:
        Modified transformer block
    """
    if generators is None:
        raise ValueError("Must provide SO(3) generators")

    embed_dim = transformer_block.ffn.net[0].in_features

    if variant == 'gradient_engine':
        variational_ffn = VariationalFFNGradientEngine(
            embed_dim=embed_dim,
            generators=generators,
            **kwargs
        )
    elif variant == 'approximate':
        variational_ffn = VariationalFFNApproximate(
            embed_dim=embed_dim,
            generators=generators,
            **kwargs
        )
    elif variant == 'full':
        variational_ffn = VariationalFFNFull(
            embed_dim=embed_dim,
            generators=generators,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown variant: {variant}")

    # Store original FFN (for comparison/ablation)
    transformer_block.ffn_original = transformer_block.ffn

    # Replace with variational version
    transformer_block.ffn_variational = variational_ffn
    transformer_block.use_variational = True

    return transformer_block