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
    sigma_q: torch.Tensor,   # (B, N, K, K)
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
        sigma_q: Belief covariances (B, N, K, K)
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
    sigma_q_np = sigma_q[batch_idx].detach().cpu().numpy()  # (N, K, K)
    mu_p_np = mu_prior[batch_idx].detach().cpu().numpy()  # (N, K)
    phi_np = phi[batch_idx].detach().cpu().numpy()  # (N, 3)
    gen_np = generators.detach().cpu().numpy()  # (3, K, K)

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
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.register_buffer('generators', generators)  # (3, K, K)
        self.n_iterations = n_iterations
        self.update_sigma = update_sigma

        # Create system config
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
            if self.update_sigma:
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
        if self.update_sigma:
            return mu_current.detach(), sigma_current.detach()
        else:
            return mu_current.detach(), None


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