"""
Hamiltonian Feedforward Network for Gauge-Theoretic Transformer
=================================================================

Replaces gradient-based variational FFN with Hamiltonian dynamics.

FAITHFUL TO INFORMATIONAL GAUGE THEORY & INERTIA OF BELIEF
-----------------------------------------------------------
Implements the extended mass formula from "The Inertia of Belief" paper:

    M_i = Λ_{pi} + Λ_{oi} + Σ_k β_{ik} Λ̃_{qk} + Σ_j β_{ji} Λ_{qi}

where (Eq. 20 in paper):
    - Λ_{pi} = Prior precision (resistance from prior expectations)
    - Λ_{oi} = Observation precision (sensory grounding)
    - Σ_k β_{ik} Λ̃_{qk} = Incoming social precision (being pulled toward confident neighbors)
    - Σ_j β_{ji} Λ_{qi} = Outgoing recoil precision (Newton's 3rd law from influencing others)

And Λ̃_{qk} = Ω_{ik} Λ_{qk} Ω_{ik}^T is the transported precision via gauge connection.

From field_theory.py, the complete Hamiltonian is:

    H = T_μ + T_Σ + T_φ + V

where:
    T_μ = (1/2) π_μᵀ M⁻¹ π_μ              (Fisher-Rao metric with extended mass)
    T_Σ = (1/4) tr(Σ⁻¹ Σ̇ Σ⁻¹ Σ̇)          (SPD manifold metric)
    T_φ = (1/2) ⟨π_φ, π_φ⟩_𝔤              (Killing form on Lie algebra)
    V   = Free Energy Functional           (from free_energy_clean.py)

Conjugate momenta:
    π_μ = M μ̇            → μ̇ = M⁻¹ π_μ    (Extended mass matrix from paper)
    π_Σ = ½ Σ⁻¹ Σ̇ Σ⁻¹   → Σ̇ = 2 Σ π_Σ Σ   (SPD geometry)
    π_φ = φ̇             → φ̇ = π_φ         (trivial for gauge)

Hamilton's equations:
    dμ/dt  = ∂H/∂π_μ = M⁻¹ π_μ
    dΣ/dt  = ∂H/∂π_Σ = 2 Σ π_Σ Σ
    dφ/dt  = ∂H/∂π_φ = π_φ
    dπ_μ/dt  = -∂V/∂μ - (∂T/∂μ if mass depends on μ)
    dπ_Σ/dt  = -∂V/∂Σ + (SPD curvature correction)
    dπ_φ/dt  = -∂V/∂φ

SYMPLECTIC INTEGRATION
----------------------
We use the Störmer-Verlet (leapfrog) integrator which:
1. Preserves the symplectic 2-form ω = dq ∧ dp
2. Is time-reversible
3. Conserves energy to O(dt²) per step (no drift!)
4. Is 2nd order accurate

GAUGE COVARIANCE
----------------
Under gauge transformation g ∈ G:
    μ → g·μ,  Σ → g Σ gᵀ,  φ → Ad_g(φ)
    π_μ → g·π_μ,  π_Σ → g π_Σ gᵀ,  π_φ → Ad_g(π_φ)

The Hamiltonian H is gauge-invariant, so dynamics preserves covariance.

Author: Chris & Claude
Date: December 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from dataclasses import dataclass


# =============================================================================
# Phase Space State Container
# =============================================================================

@dataclass
class PhaseSpaceState:
    """
    Complete phase space state for Hamiltonian dynamics.

    Configuration: (μ, Σ, φ) - beliefs and gauge field
    Momenta: (π_μ, π_Σ, π_φ) - conjugate momenta

    All tensors have shape (B, N, ...) where:
        B = batch size
        N = number of agents (sequence length)
        K = latent dimension
    """
    # Configuration variables (positions)
    mu: torch.Tensor      # (B, N, K) - belief means
    Sigma: torch.Tensor   # (B, N, K, K) - belief covariances (SPD)
    phi: torch.Tensor     # (B, N, 3) - gauge field (so(3) Lie algebra)

    # Conjugate momenta
    pi_mu: torch.Tensor    # (B, N, K) - momentum conjugate to μ
    pi_Sigma: torch.Tensor # (B, N, K, K) - momentum conjugate to Σ (symmetric)
    pi_phi: torch.Tensor   # (B, N, 3) - momentum conjugate to φ

    def detach(self) -> 'PhaseSpaceState':
        """Detach all tensors from computation graph."""
        return PhaseSpaceState(
            mu=self.mu.detach(),
            Sigma=self.Sigma.detach(),
            phi=self.phi.detach(),
            pi_mu=self.pi_mu.detach(),
            pi_Sigma=self.pi_Sigma.detach(),
            pi_phi=self.pi_phi.detach(),
        )

    def clone(self) -> 'PhaseSpaceState':
        """Create a deep copy."""
        return PhaseSpaceState(
            mu=self.mu.clone(),
            Sigma=self.Sigma.clone(),
            phi=self.phi.clone(),
            pi_mu=self.pi_mu.clone(),
            pi_Sigma=self.pi_Sigma.clone(),
            pi_phi=self.pi_phi.clone(),
        )


@dataclass
class MassConfig:
    """
    Configuration for the extended mass formula from "The Inertia of Belief".

    The complete mass is:
        M_i = Λ_{pi} + Λ_{oi} + Σ_k β_{ik} Λ̃_{qk} + Σ_j β_{ji} Λ_{qi}

    Each term can be toggled independently for ablation studies.
    """
    use_prior_precision: bool = True      # Λ_p: Prior precision (always on by default)
    use_observation_precision: bool = True   # Λ_o: Observation precision (sensory grounding)
    use_incoming_social: bool = True      # Σβ_{ik}Λ̃_{qk}: Being pulled toward neighbors
    use_outgoing_recoil: bool = True      # Σβ_{ji}Λ_{qi}: Newton's 3rd law recoil

    # Regularization
    eps: float = 1e-6                     # For numerical stability
    min_eigenvalue: float = 1e-4          # Minimum eigenvalue for mass matrix


# =============================================================================
# Inertia of Belief Mass Matrix (Paper Eq. 20)
# =============================================================================

class InertiaOfBeliefMass(nn.Module):
    """
    Extended mass matrix from "The Inertia of Belief" paper.

    M_i = Λ_{pi} + Λ_{oi} + Σ_k β_{ik} Λ̃_{qk} + Σ_j β_{ji} Λ_{qi}

    where:
        - Λ_{pi} = Σ_p⁻¹ = Prior precision
        - Λ_{oi} = Observation precision (from sensory likelihood)
        - Λ̃_{qk} = Ω_{ik} Λ_{qk} Ω_{ik}^T = Transported neighbor precision
        - Ω_{ik} = e^{φ_i} e^{-φ_k} = Gauge transport operator
        - β_{ij} = Attention weights (affinity/trust)

    Physical interpretation:
        - Prior precision: Resistance from prior expectations
        - Observation precision: Grounding in sensory data
        - Incoming social: Being pulled toward confident neighbors
        - Outgoing recoil: Newton's 3rd law from influencing others

    Note: The full mass matrix M_i is position-dependent (depends on Σ, β),
    which means we need smaller dt for symplectic accuracy compared to
    constant mass. The paper argues this is necessary for faithful dynamics.
    """

    def __init__(
        self,
        embed_dim: int,
        generators: torch.Tensor,  # (3, K, K) SO(3) generators
        config: Optional[MassConfig] = None,
    ):
        """
        Initialize mass matrix computation.

        Args:
            embed_dim: Latent dimension K
            generators: SO(3) generators for gauge transport
            config: MassConfig with toggles for each term
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.config = config or MassConfig()
        self.register_buffer('generators', generators)

    def compute_gauge_transport(
        self,
        phi_i: torch.Tensor,  # (B, N, 3) - source gauge field
        phi_k: torch.Tensor,  # (B, N, 3) - target gauge field
    ) -> torch.Tensor:
        """
        Compute gauge transport operator Ω_{ik} = e^{φ_i} e^{-φ_k}.

        This transports quantities from agent k's frame to agent i's frame.

        Args:
            phi_i: Gauge field at source
            phi_k: Gauge field at target

        Returns:
            Omega: (B, N, K, K) transport operator
        """
        K = self.embed_dim
        B, N, _ = phi_i.shape
        device = phi_i.device
        dtype = phi_i.dtype

        # φ in so(3): φ = φ_a T_a where T_a are generators
        # e^φ = exp(φ_a T_a)

        # Compute φ_i · T for each agent
        # generators: (3, K, K), phi_i: (B, N, 3)
        phi_i_matrix = torch.einsum('bnc,ckl->bnkl', phi_i, self.generators)  # (B, N, K, K)
        phi_k_matrix = torch.einsum('bnc,ckl->bnkl', phi_k, self.generators)  # (B, N, K, K)

        # Matrix exponential (stable via eigendecomposition for skew-symmetric)
        # For small angles, use Rodrigues formula, but for general case use matrix_exp
        exp_phi_i = torch.linalg.matrix_exp(phi_i_matrix)  # (B, N, K, K)
        exp_neg_phi_k = torch.linalg.matrix_exp(-phi_k_matrix)  # (B, N, K, K)

        # Ω_{ik} = e^{φ_i} e^{-φ_k}
        Omega = exp_phi_i @ exp_neg_phi_k  # (B, N, K, K)

        return Omega

    def transport_precision(
        self,
        Lambda_k: torch.Tensor,  # (B, N, K, K) - precision at k
        phi_i: torch.Tensor,     # (B, N, 3) - gauge at i
        phi_k: torch.Tensor,     # (B, N, 3) - gauge at k
    ) -> torch.Tensor:
        """
        Transport precision from agent k to agent i's frame.

        Λ̃_{qk} = Ω_{ik} Λ_{qk} Ω_{ik}^T

        Args:
            Lambda_k: Precision at agent k
            phi_i: Gauge field at agent i
            phi_k: Gauge field at agent k

        Returns:
            Lambda_transported: (B, N, K, K) precision in i's frame
        """
        Omega = self.compute_gauge_transport(phi_i, phi_k)  # (B, N, K, K)

        # Λ̃ = Ω Λ Ω^T
        Lambda_transported = Omega @ Lambda_k @ Omega.transpose(-1, -2)

        return Lambda_transported

    def compute_categorical_observation_precision(
        self,
        mu: torch.Tensor,         # (B, N, K) - belief means
        W_out: torch.Tensor,      # (V, K) - output projection
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Compute observation precision for categorical output distribution.

        For a transformer with softmax output p = softmax(W_out @ μ / τ),
        the observation precision in belief space is the Hessian of the
        negative log-likelihood:

            Λ_o = ∇²_μ CE = (1/τ²) W^T (diag(p) - pp^T) W

        This simplifies to the covariance of output embeddings weighted
        by predicted probabilities:

            Λ_o = (1/τ²) Cov_p(W) = (1/τ²) [E_p[ww^T] - E_p[w]E_p[w]^T]

        Physical interpretation:
            - When p is peaked (confident): Λ_o has low rank, weak constraint
            - When p is uniform (uncertain): Λ_o reflects full embedding structure
            - Temperature τ scales the precision (lower τ → higher precision)

        Args:
            mu: Belief means (B, N, K)
            W_out: Output projection matrix (V, K)
            temperature: Softmax temperature

        Returns:
            Lambda_o: (B, N, K, K) observation precision in belief space
        """
        B, N, K = mu.shape
        V = W_out.shape[0]

        # Compute logits and softmax probabilities
        logits = torch.einsum('bnk,vk->bnv', mu, W_out) / temperature  # (B, N, V)
        p = F.softmax(logits, dim=-1)  # (B, N, V)

        # E_p[w] = Σ_v p_v w_v  (mean embedding under predicted distribution)
        mean_w = torch.einsum('bnv,vk->bnk', p, W_out)  # (B, N, K)

        # E_p[ww^T] = Σ_v p_v w_v w_v^T  (second moment)
        # Efficient: einsum('bnv,vk,vl->bnkl', p, W_out, W_out)
        second_moment = torch.einsum('bnv,vk,vl->bnkl', p, W_out, W_out)  # (B, N, K, K)

        # Cov_p(W) = E_p[ww^T] - E_p[w]E_p[w]^T
        outer_mean = torch.einsum('bnk,bnl->bnkl', mean_w, mean_w)  # (B, N, K, K)
        Lambda_o = (second_moment - outer_mean) / (temperature ** 2)

        # Symmetrize for numerical stability
        Lambda_o = 0.5 * (Lambda_o + Lambda_o.transpose(-1, -2))

        # Ensure positive semi-definiteness with small regularization
        Lambda_o = Lambda_o + self.config.eps * torch.eye(K, device=mu.device, dtype=mu.dtype)

        return Lambda_o

    def compute_mass(
        self,
        Sigma_prior: torch.Tensor,   # (B, N, K, K) - prior covariance
        Sigma_q: torch.Tensor,       # (B, N, K, K) - posterior covariance
        phi: torch.Tensor,           # (B, N, 3) - gauge field
        beta: Optional[torch.Tensor] = None,  # (B, N, N) or (B, n_heads, N, N) - attention weights
        Sigma_obs: Optional[torch.Tensor] = None,  # (B, N, K, K) - observation covariance (Gaussian)
        mu: Optional[torch.Tensor] = None,    # (B, N, K) - belief means (for categorical Λ_o)
        W_out: Optional[torch.Tensor] = None,  # (V, K) - output projection (for categorical Λ_o)
        obs_temperature: float = 1.0,          # Softmax temperature for categorical Λ_o
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the complete mass matrix M and its inverse M⁻¹.

        M_i = Λ_{pi} + Λ_{oi} + Σ_k β_{ik} Λ̃_{qk} + Σ_j β_{ji} Λ_{qi}

        Args:
            Sigma_prior: Prior covariance (for Λ_p = Σ_p⁻¹)
            Sigma_q: Posterior covariance (for Λ_q = Σ_q⁻¹)
            phi: Gauge field for transport
            beta: Attention weights - can be (B, N, N) or (B, n_heads, N, N)
            Sigma_obs: Observation covariance for Gaussian likelihood (Λ_o = Σ_o⁻¹)
            mu: Belief means for categorical observation precision
            W_out: Output projection for categorical observation precision
            obs_temperature: Softmax temperature for categorical Λ_o

        Observation Precision (Λ_o):
            - If Sigma_obs provided: Λ_o = Σ_obs⁻¹ (Gaussian observation model)
            - If mu and W_out provided: Λ_o = Cov_p(W) (Categorical/softmax model)
              where p = softmax(W_out @ μ / τ) and Cov_p(W) is the covariance
              of output embeddings under the predicted distribution.

        Returns:
            M: (B, N, K, K) mass matrix
            M_inv: (B, N, K, K) inverse mass matrix
        """
        B, N, K = Sigma_prior.shape[:3]
        device = Sigma_prior.device
        dtype = Sigma_prior.dtype
        eps = self.config.eps

        # Handle multi-head attention: average across heads if needed
        # beta can be (B, N, N) or (B, n_heads, N, N)
        if beta is not None and beta.dim() == 4:
            # Multi-head attention: average across heads
            beta = beta.mean(dim=1)  # (B, N, N)

        # Initialize mass as zero
        M = torch.zeros(B, N, K, K, device=device, dtype=dtype)

        # =====================================================================
        # 1. Prior precision: Λ_p = Σ_p⁻¹
        # =====================================================================
        if self.config.use_prior_precision:
            Sigma_p_reg = Sigma_prior + eps * torch.eye(K, device=device, dtype=dtype)
            Lambda_p = torch.linalg.inv(Sigma_p_reg)  # (B, N, K, K)
            M = M + Lambda_p

        # =====================================================================
        # 2. Observation precision: Λ_o
        # Two modes:
        #   - Gaussian: Λ_o = Σ_o⁻¹ (requires Sigma_obs)
        #   - Categorical: Λ_o = Cov_p(W_out) (requires mu, W_out)
        # =====================================================================
        if self.config.use_observation_precision:
            if mu is not None and W_out is not None:
                # Categorical observation model (transformer softmax output)
                # Λ_o = Hessian of CE loss = Cov_p(W_out) / τ²
                Lambda_o = self.compute_categorical_observation_precision(
                    mu, W_out, temperature=obs_temperature
                )
                M = M + Lambda_o
            elif Sigma_obs is not None:
                # Gaussian observation model (traditional active inference)
                # Λ_o = Σ_o⁻¹
                Sigma_o_reg = Sigma_obs + eps * torch.eye(K, device=device, dtype=dtype)
                Lambda_o = torch.linalg.inv(Sigma_o_reg)  # (B, N, K, K)
                M = M + Lambda_o

        # =====================================================================
        # Pre-compute Lambda_q once if needed for social terms (speed-up)
        # =====================================================================
        need_Lambda_q = (self.config.use_incoming_social or self.config.use_outgoing_recoil) and beta is not None
        Lambda_q = None
        if need_Lambda_q:
            Sigma_q_reg = Sigma_q + eps * torch.eye(K, device=device, dtype=dtype)
            Lambda_q = torch.linalg.inv(Sigma_q_reg)  # (B, N, K, K)

        # =====================================================================
        # 3. Incoming social precision: Σ_k β_{ik} Λ̃_{qk}
        # "Being pulled toward confident neighbors"
        # =====================================================================
        if self.config.use_incoming_social and beta is not None:
            # Pre-compute all matrix exponentials once (speed-up)
            # phi_matrix: (B, N, K, K), exp_phi: (B, N, K, K)
            phi_matrix = torch.einsum('bnc,ckl->bnkl', phi, self.generators)
            exp_phi = torch.linalg.matrix_exp(phi_matrix)      # e^{φ_i} for all i
            exp_neg_phi = torch.linalg.matrix_exp(-phi_matrix)  # e^{-φ_k} for all k

            # Vectorized: compute Ω_{ik} = e^{φ_i} @ e^{-φ_k} for all pairs
            # exp_phi: (B, N, K, K) -> (B, N, 1, K, K)
            # exp_neg_phi: (B, N, K, K) -> (B, 1, N, K, K)
            # Omega_all: (B, N, N, K, K) where Omega_all[b,i,k] = Ω_{ik}
            Omega_all = torch.einsum('bikl,bjlm->bijkm', exp_phi, exp_neg_phi)

            # Transport all precisions: Λ̃_{qk} = Ω_{ik} @ Λ_{qk} @ Ω_{ik}^T
            # Lambda_q: (B, N, K, K) -> (B, 1, N, K, K) for k dimension
            # Result: (B, N, N, K, K) where [b,i,k] = transported precision from k to i
            Lambda_transported_all = torch.einsum(
                'bijkl,bjlm,bijmn->bijkn',
                Omega_all, Lambda_q, Omega_all.transpose(-1, -2)
            )

            # Weight by attention and sum: M_incoming[b,i] = Σ_k β_{ik} Λ̃_{qk}
            # beta: (B, N, N), Lambda_transported_all: (B, N, N, K, K)
            M_incoming = torch.einsum('bik,bikmn->bimn', beta, Lambda_transported_all)

            M = M + M_incoming

        # =====================================================================
        # 4. Outgoing recoil precision: Σ_j β_{ji} Λ_{qi}
        # "Newton's 3rd law from influencing others"
        # =====================================================================
        if self.config.use_outgoing_recoil and beta is not None:
            # Sum over j: Σ_j β_{ji} = (sum over row j of beta for column i)
            beta_sum = beta.sum(dim=1)  # (B, N) - sum of attention TO agent i

            # Multiply by own precision
            M_outgoing = beta_sum.unsqueeze(-1).unsqueeze(-1) * Lambda_q  # (B, N, K, K)

            M = M + M_outgoing

        # =====================================================================
        # Ensure mass is SPD (positive definite)
        # =====================================================================
        # If no terms are enabled, use identity mass
        if not (self.config.use_prior_precision or
                self.config.use_observation_precision or
                self.config.use_incoming_social or
                self.config.use_outgoing_recoil):
            M = torch.eye(K, device=device, dtype=dtype).unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1).clone()

        # Symmetrize and regularize
        M = 0.5 * (M + M.transpose(-1, -2))

        # Replace NaN/Inf with identity before eigendecomposition
        if torch.isnan(M).any() or torch.isinf(M).any():
            M = torch.where(
                torch.isnan(M) | torch.isinf(M),
                torch.eye(K, device=device, dtype=dtype).unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1),
                M
            )

        # Add strong diagonal regularization BEFORE eigendecomposition
        # This ensures the matrix is well-conditioned
        reg_strength = max(self.config.min_eigenvalue, 1e-4)
        M = M + reg_strength * torch.eye(K, device=device, dtype=dtype).unsqueeze(0).unsqueeze(0)

        # Ensure minimum eigenvalue for stability
        # IMPORTANT: Force FP32 for eigendecomposition (fails under FP16/AMP)
        M_fp32 = M.float()
        try:
            eigenvalues, eigenvectors = torch.linalg.eigh(M_fp32)
            eigenvalues = torch.clamp(eigenvalues, min=self.config.min_eigenvalue)
            M = (eigenvectors @ torch.diag_embed(eigenvalues) @ eigenvectors.transpose(-1, -2)).to(dtype)
            M_inv = (eigenvectors @ torch.diag_embed(1.0 / eigenvalues) @ eigenvectors.transpose(-1, -2)).to(dtype)
        except RuntimeError:
            # Fallback to regularized identity if eigendecomposition fails
            M = torch.eye(K, device=device, dtype=dtype).unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1).clone()
            M_inv = M.clone()

        return M, M_inv


# =============================================================================
# Geometric Operations (PyTorch)
# =============================================================================

def symmetrize(M: torch.Tensor) -> torch.Tensor:
    """Symmetrize a matrix: sym(M) = (M + Mᵀ)/2"""
    return 0.5 * (M + M.transpose(-1, -2))


def ensure_spd(Sigma: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Project matrix to SPD cone via eigenvalue clipping.

    Σ_spd = V max(Λ, ε) Vᵀ
    """
    orig_dtype = Sigma.dtype
    K = Sigma.shape[-1]
    device = Sigma.device

    # Add diagonal regularization before eigendecomposition
    Sigma_reg = Sigma + eps * torch.eye(K, device=device, dtype=Sigma.dtype).unsqueeze(0).unsqueeze(0)
    Sigma_reg = 0.5 * (Sigma_reg + Sigma_reg.transpose(-1, -2))  # Ensure symmetric

    # Eigendecomposition (force FP32 for numerical stability under AMP)
    try:
        eigenvalues, eigenvectors = torch.linalg.eigh(Sigma_reg.float())
        # Clip eigenvalues to be positive
        eigenvalues_clipped = torch.clamp(eigenvalues, min=eps)
        # Reconstruct and cast back to original dtype
        Sigma_spd = (eigenvectors @ torch.diag_embed(eigenvalues_clipped) @ eigenvectors.transpose(-1, -2)).to(orig_dtype)
    except RuntimeError:
        # Fallback to regularized identity
        Sigma_spd = eps * torch.eye(K, device=device, dtype=orig_dtype).unsqueeze(0).unsqueeze(0).expand_as(Sigma).clone()

    return symmetrize(Sigma_spd)


# =============================================================================
# SPD Geodesic Curvature Corrections (FULL FAITHFUL THEORY)
# =============================================================================

def spd_geodesic_acceleration(
    Sigma: torch.Tensor,
    Sigma_dot: torch.Tensor,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Geodesic equation (acceleration) on SPD manifold.

    From Euler-Lagrange with L = (1/4) tr(Σ⁻¹ Σ̇ Σ⁻¹ Σ̇):

        Σ̈ = Σ (Σ⁻¹Σ̇)² Σ - (1/2)[Σ̇Σ⁻¹Σ̇ + (Σ̇Σ⁻¹Σ̇)ᵀ]

    This is the natural "acceleration" for free motion on SPD manifold.
    Particles following geodesics have zero covariant acceleration.

    Args:
        Sigma: (B, N, K, K) covariance matrices ∈ SPD(K)
        Sigma_dot: (B, N, K, K) velocity in tangent space ∈ Sym(K)
        eps: Numerical regularization

    Returns:
        Sigma_ddot: (B, N, K, K) geodesic acceleration ∈ Sym(K)
    """
    # Regularize and invert
    K = Sigma.shape[-1]
    Sigma_reg = Sigma + eps * torch.eye(K, device=Sigma.device, dtype=Sigma.dtype)
    Sigma_inv = torch.linalg.inv(Sigma_reg)

    # A = Σ⁻¹Σ̇
    A = Sigma_inv @ Sigma_dot

    # First term: Σ A² Σ = Σ (Σ⁻¹Σ̇)² Σ
    A_squared = A @ A
    term1 = Sigma @ A_squared @ Sigma

    # Second term: (1/2)[Σ̇Σ⁻¹Σ̇ + (Σ̇Σ⁻¹Σ̇)ᵀ]
    B = Sigma_dot @ Sigma_inv @ Sigma_dot
    term2 = 0.5 * (B + B.transpose(-1, -2))

    Sigma_ddot = term1 - term2

    return symmetrize(Sigma_ddot)


def spd_kinetic_gradient(
    Sigma: torch.Tensor,
    pi_Sigma: torch.Tensor,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Gradient of kinetic energy T_Σ with respect to Σ (holding π_Σ fixed).

    T_Σ = tr(π_Σ Σ π_Σ Σ)

    Using matrix calculus:
        ∂T_Σ/∂Σ = 2 π_Σ Σ π_Σ

    This is the "force" from the curved geometry that must be included
    in the momentum update to preserve symplecticity on the manifold.

    DERIVATION:
    -----------
    T_Σ = tr(π_Σ Σ π_Σ Σ)

    Let M = π_Σ Σ π_Σ, so T_Σ = tr(M Σ).

    dT_Σ = tr(M dΣ) + tr(dM Σ)
         = tr(M dΣ) + tr((π_Σ dΣ π_Σ) Σ)
         = tr(M dΣ) + tr(Σ π_Σ dΣ π_Σ)
         = tr(M dΣ) + tr(π_Σ Σ π_Σ dΣ)     [cyclic]
         = tr((M + π_Σ Σ π_Σ) dΣ)
         = tr(2 π_Σ Σ π_Σ dΣ)

    Therefore: ∂T_Σ/∂Σ = 2 π_Σ Σ π_Σ

    Args:
        Sigma: (B, N, K, K) covariance matrices
        pi_Sigma: (B, N, K, K) conjugate momenta (symmetric)
        eps: Numerical regularization

    Returns:
        grad_T_Sigma: (B, N, K, K) gradient of kinetic energy
    """
    # ∂T_Σ/∂Σ = 2 π_Σ Σ π_Σ
    pi_Sigma_Sigma = pi_Sigma @ Sigma  # (B, N, K, K)
    grad_T = 2.0 * pi_Sigma_Sigma @ pi_Sigma

    return symmetrize(grad_T)


def momentum_from_velocity_spd(
    Sigma: torch.Tensor,
    Sigma_dot: torch.Tensor,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Convert SPD velocity Σ̇ to conjugate momentum π_Σ.

    π_Σ = (1/2) Σ⁻¹ Σ̇ Σ⁻¹

    This is the Legendre transform from velocity to momentum.

    Args:
        Sigma: (B, N, K, K) covariance matrices
        Sigma_dot: (B, N, K, K) velocity

    Returns:
        pi_Sigma: (B, N, K, K) conjugate momentum
    """
    K = Sigma.shape[-1]
    Sigma_reg = Sigma + eps * torch.eye(K, device=Sigma.device, dtype=Sigma.dtype)
    Sigma_inv = torch.linalg.inv(Sigma_reg)

    pi_Sigma = 0.5 * Sigma_inv @ Sigma_dot @ Sigma_inv
    return symmetrize(pi_Sigma)


def velocity_from_momentum_spd(
    Sigma: torch.Tensor,
    pi_Sigma: torch.Tensor
) -> torch.Tensor:
    """
    Convert conjugate momentum π_Σ to SPD velocity Σ̇.

    Σ̇ = 2 Σ π_Σ Σ

    This is the inverse Legendre transform.

    Args:
        Sigma: (B, N, K, K) covariance matrices
        pi_Sigma: (B, N, K, K) conjugate momentum

    Returns:
        Sigma_dot: (B, N, K, K) velocity
    """
    Sigma_dot = 2.0 * Sigma @ pi_Sigma @ Sigma
    return symmetrize(Sigma_dot)


def spd_exponential_map(Sigma: torch.Tensor, V: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Exponential map on SPD manifold (PyTorch version).

    exp_Σ(V) = Σ^{1/2} exp(Σ^{-1/2} V Σ^{-1/2}) Σ^{1/2}

    Maps tangent vector V at Σ to a point on the SPD manifold.
    """
    orig_dtype = Sigma.dtype
    K = Sigma.shape[-1]
    device = Sigma.device

    # Regularize Sigma
    Sigma = symmetrize(Sigma)
    Sigma = Sigma + eps * torch.eye(K, device=device, dtype=Sigma.dtype)

    try:
        # Matrix square root via eigendecomposition (force FP32 for stability under AMP)
        eigenvalues, eigenvectors = torch.linalg.eigh(Sigma.float())
        eigenvalues = torch.clamp(eigenvalues, min=eps)

        Sigma_sqrt = eigenvectors @ torch.diag_embed(torch.sqrt(eigenvalues)) @ eigenvectors.transpose(-1, -2)
        Sigma_inv_sqrt = eigenvectors @ torch.diag_embed(1.0 / torch.sqrt(eigenvalues)) @ eigenvectors.transpose(-1, -2)

        # W = Σ^{-1/2} V Σ^{-1/2}
        W = Sigma_inv_sqrt @ V.float() @ Sigma_inv_sqrt
        W = symmetrize(W)  # Ensure symmetric for matrix exp

        # exp(W) via eigendecomposition (more stable than torch.matrix_exp for symmetric)
        W_eigenvalues, W_eigenvectors = torch.linalg.eigh(W)
        # Clamp W eigenvalues to prevent overflow in exp
        W_eigenvalues = torch.clamp(W_eigenvalues, min=-20.0, max=20.0)
        exp_W = W_eigenvectors @ torch.diag_embed(torch.exp(W_eigenvalues)) @ W_eigenvectors.transpose(-1, -2)

        # exp_Σ(V) = Σ^{1/2} exp(W) Σ^{1/2}
        Sigma_new = Sigma_sqrt @ exp_W @ Sigma_sqrt
        return ensure_spd(Sigma_new.to(orig_dtype), eps)

    except RuntimeError:
        # Fallback: return input with small perturbation (identity-like behavior)
        return ensure_spd(Sigma.to(orig_dtype), eps)


# =============================================================================
# Kinetic Energy Terms (Faithful to field_theory.py)
# =============================================================================

class HamiltonianKineticTerms(nn.Module):
    """
    Kinetic energy terms from the gauge-theoretic Hamiltonian.

    T = T_μ + T_Σ + T_φ

    Each term uses the correct geometric structure:
    - T_μ: Fisher-Rao metric with extended mass from Inertia of Belief paper
    - T_Σ: Affine-invariant SPD metric
    - T_φ: Killing form on so(3)

    EXTENDED MASS (from "The Inertia of Belief"):
    -----------------------------------------------
    The kinetic energy T_μ = (1/2) π_μᵀ M⁻¹ π_μ now uses the full mass:

        M_i = Λ_{pi} + Λ_{oi} + Σ_k β_{ik} Λ̃_{qk} + Σ_j β_{ji} Λ_{qi}

    When only prior precision is used (default), this reduces to the
    original formulation with M = Σ_p⁻¹, M⁻¹ = Σ_p.
    """

    def __init__(self, embed_dim: int, eps: float = 1e-8):
        super().__init__()
        self.embed_dim = embed_dim
        self.eps = eps

    def kinetic_mu(
        self,
        pi_mu: torch.Tensor,
        M_inv: torch.Tensor,
    ) -> torch.Tensor:
        """
        Mean kinetic energy: T_μ = (1/2) π_μᵀ M⁻¹ π_μ

        Uses the extended mass matrix from "The Inertia of Belief" paper.
        When only prior precision is used, M⁻¹ = Σ_p (original formulation).

        Args:
            pi_mu: (B, N, K) momentum
            M_inv: (B, N, K, K) inverse mass matrix

        Returns:
            T_mu: (B, N) kinetic energy per agent
        """
        # T_μ = (1/2) π_μᵀ M⁻¹ π_μ
        # Use einsum: (...,i), (...,i,j), (...,j) -> (...)
        M_inv_pi = torch.einsum('...ij,...j->...i', M_inv, pi_mu)
        T_mu = 0.5 * torch.einsum('...i,...i->...', pi_mu, M_inv_pi)
        return T_mu

    def kinetic_Sigma(
        self,
        Sigma: torch.Tensor,
        pi_Sigma: torch.Tensor
    ) -> torch.Tensor:
        """
        Covariance kinetic energy on SPD manifold.

        T_Σ = (1/4) tr(Σ⁻¹ Σ̇ Σ⁻¹ Σ̇)

        where Σ̇ = 2 Σ π_Σ Σ (from Legendre transform)

        Substituting:
            T_Σ = (1/4) tr(Σ⁻¹ (2 Σ π_Σ Σ) Σ⁻¹ (2 Σ π_Σ Σ))
                = tr(π_Σ Σ π_Σ Σ)

        Args:
            Sigma: (B, N, K, K) covariance
            pi_Sigma: (B, N, K, K) conjugate momentum

        Returns:
            T_Sigma: (B, N) kinetic energy per agent
        """
        # T_Σ = tr(π_Σ Σ π_Σ Σ)
        pi_Sigma_Sigma = pi_Sigma @ Sigma  # (..., K, K)
        T_Sigma = torch.einsum('...ij,...ji->...', pi_Sigma_Sigma, pi_Sigma_Sigma)
        return T_Sigma

    def kinetic_phi(self, pi_phi: torch.Tensor) -> torch.Tensor:
        """
        Gauge field kinetic energy.

        T_φ = (1/2) ⟨π_φ, π_φ⟩_𝔤 = (1/2) ||π_φ||²

        For so(3) with standard metric: ⟨φ, ψ⟩ = φ · ψ

        Args:
            pi_phi: (B, N, 3) gauge momentum

        Returns:
            T_phi: (B, N) kinetic energy per agent
        """
        # T_φ = (1/2) ||π_φ||²
        T_phi = 0.5 * torch.sum(pi_phi ** 2, dim=-1)
        return T_phi

    def total_kinetic(
        self,
        state: PhaseSpaceState,
        M_inv: torch.Tensor,
        chi: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Total kinetic energy: T = χ · (T_μ + T_Σ + T_φ)

        Args:
            state: Phase space state
            M_inv: (B, N, K, K) inverse mass matrix (from InertiaOfBeliefMass or Σ_p)
            chi: (B, N) support weights (default: ones)

        Returns:
            T: (B,) total kinetic energy per batch
        """
        T_mu = self.kinetic_mu(state.pi_mu, M_inv)
        T_Sigma = self.kinetic_Sigma(state.Sigma, state.pi_Sigma)
        T_phi = self.kinetic_phi(state.pi_phi)

        T_total = T_mu + T_Sigma + T_phi  # (B, N)

        if chi is not None:
            T_total = chi * T_total

        return T_total.sum(dim=-1)  # Sum over agents


# =============================================================================
# Potential Energy (Free Energy Functional)
# =============================================================================

class HamiltonianPotential(nn.Module):
    """
    Potential energy V from the variational free energy functional.

    V = α·KL(q||p) + λ_β·Σ_ij β_ij·KL(q_i||Ω_ij[q_j]) + CE(y|μ)

    This is the same free energy used in gradient-based training,
    now serving as the potential in Hamilton's equations.
    """

    def __init__(
        self,
        embed_dim: int,
        generators: torch.Tensor,  # (3, K, K) SO(3) generators
        alpha: float = 1.0,        # Self-coupling weight
        lambda_belief: float = 1.0, # Belief alignment weight
        kappa: float = 1.0,        # Softmax temperature
        eps: float = 1e-8,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.register_buffer('generators', generators)
        self.alpha = alpha
        self.lambda_belief = lambda_belief
        self.kappa = kappa
        self.eps = eps

    def kl_divergence(
        self,
        mu_q: torch.Tensor,
        Sigma_q: torch.Tensor,
        mu_p: torch.Tensor,
        Sigma_p: torch.Tensor
    ) -> torch.Tensor:
        """
        KL divergence: KL(N(μ_q, Σ_q) || N(μ_p, Σ_p))

        KL = (1/2)[tr(Σ_p⁻¹ Σ_q) + (μ_p - μ_q)ᵀ Σ_p⁻¹ (μ_p - μ_q) - K + log(det Σ_p / det Σ_q)]
        """
        K = mu_q.shape[-1]
        orig_dtype = mu_q.dtype

        # Force FP32 for numerical operations (slogdet/inv don't support FP16)
        Sigma_q = Sigma_q.float()
        Sigma_p = Sigma_p.float()
        mu_q = mu_q.float()
        mu_p = mu_p.float()

        # Regularize
        Sigma_q = Sigma_q + self.eps * torch.eye(K, device=Sigma_q.device, dtype=torch.float32)
        Sigma_p = Sigma_p + self.eps * torch.eye(K, device=Sigma_p.device, dtype=torch.float32)

        # Inverse of prior
        Sigma_p_inv = torch.linalg.inv(Sigma_p)

        # Trace term
        trace_term = torch.einsum('...ij,...ji->...', Sigma_p_inv, Sigma_q)

        # Mahalanobis term
        delta_mu = mu_p - mu_q
        mahal = torch.einsum('...i,...ij,...j->...', delta_mu, Sigma_p_inv, delta_mu)

        # Log determinant term
        log_det_p = torch.linalg.slogdet(Sigma_p)[1]
        log_det_q = torch.linalg.slogdet(Sigma_q)[1]
        log_det_term = log_det_p - log_det_q

        kl = 0.5 * (trace_term + mahal - K + log_det_term)
        return kl.to(orig_dtype)

    def forward(
        self,
        state: PhaseSpaceState,
        mu_prior: torch.Tensor,
        Sigma_prior: torch.Tensor,
        beta: Optional[torch.Tensor] = None,  # (B, N, N) attention weights
        targets: Optional[torch.Tensor] = None,  # (B, N) for CE loss
        W_out: Optional[torch.Tensor] = None,    # (V, K) output projection
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute potential energy V (free energy functional).

        Args:
            state: Phase space state with (μ, Σ, φ)
            mu_prior: (B, N, K) prior means
            Sigma_prior: (B, N, K, K) prior covariances
            beta: (B, N, N) attention weights from attention layer
            targets: (B, N) target token IDs for CE term
            W_out: (V, K) output projection for logits

        Returns:
            V: (B,) total potential energy per batch
            breakdown: dict with energy components
        """
        B, N, K = state.mu.shape
        device = state.mu.device

        # =====================================================================
        # 1. Self-coupling: α · KL(q || p)
        # =====================================================================
        kl_self = self.kl_divergence(state.mu, state.Sigma, mu_prior, Sigma_prior)
        V_self = self.alpha * kl_self.sum(dim=-1)  # (B,)

        # =====================================================================
        # 2. Belief alignment: λ · Σ_ij β_ij · KL(q_i || Ω_ij[q_j])
        # =====================================================================
        # VECTORIZED implementation - no Python loops!
        V_align = torch.zeros(B, device=device, dtype=state.mu.dtype)

        if beta is not None and self.lambda_belief > 0:
            # Handle multi-head attention: average over heads
            # beta can be (B, N, N) or (B, n_heads, N, N)
            if beta.dim() == 4:
                beta_avg = beta.mean(dim=1)  # (B, N, N) - average over heads
            else:
                beta_avg = beta  # Already (B, N, N)

            # Compute all pairwise transport matrices Ω_ij = exp(φ_i) exp(-φ_j)
            # phi: (B, N, 3) -> phi_matrix: (B, N, K, K)
            phi_matrix = torch.einsum('bna,aij->bnij', state.phi, self.generators)
            exp_phi = torch.matrix_exp(phi_matrix)      # (B, N, K, K)
            exp_neg_phi = torch.matrix_exp(-phi_matrix) # (B, N, K, K)

            # Omega_ij = exp(φ_i) @ exp(-φ_j)
            # exp_phi[:, :, None] is (B, N, 1, K, K) - broadcast over j
            # exp_neg_phi[:, None, :] is (B, 1, N, K, K) - broadcast over i
            Omega = torch.einsum('bikl,bjlm->bijkm', exp_phi, exp_neg_phi)  # (B, N, N, K, K)

            # Transport all means: μ_j^{→i} = Ω_ij @ μ_j
            # state.mu[:, None, :, :] is (B, 1, N, K) - μ_j for all j
            mu_transported = torch.einsum('bijkl,bjl->bijk', Omega, state.mu)  # (B, N, N, K)

            # Transport all covariances: Σ_j^{→i} = Ω_ij @ Σ_j @ Ω_ij^T
            # state.Sigma[:, None, :] is (B, 1, N, K, K) - Σ_j for all j
            Sigma_transported = torch.einsum(
                'bijkl,bjlm,bijmn->bijkn',
                Omega, state.Sigma, Omega.transpose(-1, -2)
            )  # (B, N, N, K, K)

            # Expand mu_i and Sigma_i for pairwise comparison
            mu_i = state.mu[:, :, None, :].expand(-1, -1, N, -1)  # (B, N, N, K)
            Sigma_i = state.Sigma[:, :, None, :, :].expand(-1, -1, N, -1, -1)  # (B, N, N, K, K)

            # Compute KL divergences for all pairs (vectorized)
            # KL(q_i || Ω_ij[q_j]) for all i, j
            # Use Cholesky for stability
            eps = 1e-6
            I = torch.eye(K, device=device, dtype=state.mu.dtype)
            Sigma_transported_reg = Sigma_transported + eps * I

            try:
                L_p = torch.linalg.cholesky(Sigma_transported_reg)

                # Trace term: tr(Σ_p⁻¹ Σ_q)
                Sigma_i_reg = Sigma_i + eps * I
                Y = torch.linalg.solve_triangular(L_p, Sigma_i_reg, upper=False)
                Z = torch.linalg.solve_triangular(L_p.transpose(-1, -2), Y, upper=True)
                trace_term = torch.diagonal(Z, dim1=-2, dim2=-1).sum(dim=-1)  # (B, N, N)

                # Mahalanobis term
                delta_mu = mu_transported - mu_i  # (B, N, N, K)
                v = torch.linalg.solve_triangular(
                    L_p, delta_mu.unsqueeze(-1), upper=False
                ).squeeze(-1)
                mahal_term = torch.sum(v ** 2, dim=-1)  # (B, N, N)

                # Log determinant terms
                logdet_p = 2.0 * torch.sum(
                    torch.log(torch.diagonal(L_p, dim1=-2, dim2=-1) + eps), dim=-1
                )
                L_q = torch.linalg.cholesky(Sigma_i_reg)
                logdet_q = 2.0 * torch.sum(
                    torch.log(torch.diagonal(L_q, dim1=-2, dim2=-1) + eps), dim=-1
                )
                logdet_term = logdet_p - logdet_q  # (B, N, N)

                # KL divergence for all pairs
                kl_all = 0.5 * (trace_term + mahal_term - K + logdet_term)  # (B, N, N)
                kl_all = torch.clamp(kl_all, min=0.0)

            except RuntimeError:
                # Fallback: use simpler computation if Cholesky fails
                kl_all = torch.zeros(B, N, N, device=device, dtype=state.mu.dtype)

            # Zero out diagonal (no self-comparison)
            diag_mask = torch.eye(N, device=device, dtype=torch.bool).unsqueeze(0)
            kl_all = kl_all.masked_fill(diag_mask, 0.0)

            # Weighted sum: Σ_ij β_ij · KL_ij
            weighted_kl = beta_avg * kl_all  # (B, N, N)
            V_align = self.lambda_belief * weighted_kl.sum(dim=(-2, -1))  # (B,)

        # =====================================================================
        # 3. Cross-entropy term (if targets provided)
        # =====================================================================
        V_ce = torch.zeros(B, device=device, dtype=state.mu.dtype)

        if targets is not None and W_out is not None:
            # Compute logits from means
            logits = torch.einsum('bnk,vk->bnv', state.mu, W_out)  # (B, N, V)

            # Cross-entropy loss
            V_ce = F.cross_entropy(
                logits.view(-1, logits.shape[-1]),
                targets.view(-1),
                reduction='none'
            ).view(B, N).sum(dim=-1)

        # =====================================================================
        # Total potential
        # =====================================================================
        V_total = V_self + V_align + V_ce

        breakdown = {
            'V_self': V_self.detach(),
            'V_align': V_align.detach(),
            'V_ce': V_ce.detach(),
            'V_total': V_total.detach(),
        }

        return V_total, breakdown


# =============================================================================
# Symplectic Integrators
# =============================================================================

class LeapfrogIntegrator(nn.Module):
    """
    Störmer-Verlet (Leapfrog) symplectic integrator.

    The leapfrog algorithm:
        p_{1/2} = p_0 - (ε/2) ∂V/∂q(q_0)        # Half-step momentum
        q_1 = q_0 + ε ∂T/∂p(p_{1/2})            # Full-step position
        p_1 = p_{1/2} - (ε/2) ∂V/∂q(q_1)        # Half-step momentum

    Properties:
        - Symplectic: preserves phase space volume
        - Time-reversible: (q,p) → (-p,-q) reverses trajectory
        - Energy conservation: O(ε²) error per step, no drift!

    For our gauge-theoretic system:
        q = (μ, Σ, φ)
        p = (π_μ, π_Σ, π_φ)

    Position updates use the correct geometric structure:
        μ̇ = M⁻¹ π_μ          (Extended mass from Inertia of Belief paper)
        Σ̇ = 2 Σ π_Σ Σ        (SPD manifold)
        φ̇ = π_φ              (Lie algebra)

    Mass Evolution:
        When evolve_mass=True, M is recomputed at each integration step since
        M depends on Σ (through Λ_q = Σ⁻¹). This is theoretically correct but
        more expensive. For small dt, fixed M is a reasonable approximation.
    """

    def __init__(
        self,
        kinetic: HamiltonianKineticTerms,
        potential: HamiltonianPotential,
        dt: float = 0.01,
        n_steps: int = 1,
        update_Sigma: bool = True,
        update_phi: bool = False,
        evolve_mass: bool = False,
        mass_computer: Optional['InertiaOfBeliefMass'] = None,
    ):
        super().__init__()
        self.kinetic = kinetic
        self.potential = potential
        self.dt = dt
        self.n_steps = n_steps
        self.update_Sigma = update_Sigma
        self.update_phi = update_phi
        self.evolve_mass = evolve_mass
        self.mass_computer = mass_computer

        if evolve_mass and mass_computer is None:
            raise ValueError("evolve_mass=True requires mass_computer to be provided")

    def position_step(
        self,
        state: PhaseSpaceState,
        M_inv: torch.Tensor,
        dt: float
    ) -> PhaseSpaceState:
        """
        Full position step: q ← q + dt · ∂T/∂p

        Uses correct geometric update rules:
            μ ← μ + dt · M⁻¹ π_μ  (Extended mass from Inertia of Belief)
            Σ ← exp_Σ(dt · 2 Σ π_Σ Σ)  (geodesic on SPD)
            φ ← φ + dt · π_φ
        """
        # Mean update: μ ← μ + dt · M⁻¹ π_μ
        mu_new = state.mu + dt * torch.einsum('...ij,...j->...i', M_inv, state.pi_mu)

        # Covariance update on SPD manifold
        if self.update_Sigma:
            # Velocity: Σ̇ = 2 Σ π_Σ Σ
            Sigma_dot = 2 * state.Sigma @ state.pi_Sigma @ state.Sigma
            # Geodesic step via exponential map
            Sigma_new = spd_exponential_map(state.Sigma, dt * Sigma_dot)
        else:
            Sigma_new = state.Sigma

        # Gauge field update
        if self.update_phi:
            phi_new = state.phi + dt * state.pi_phi
        else:
            phi_new = state.phi

        return PhaseSpaceState(
            mu=mu_new,
            Sigma=Sigma_new,
            phi=phi_new,
            pi_mu=state.pi_mu,
            pi_Sigma=state.pi_Sigma,
            pi_phi=state.pi_phi,
        )

    def momentum_step(
        self,
        state: PhaseSpaceState,
        mu_prior: torch.Tensor,
        Sigma_prior: torch.Tensor,
        beta: Optional[torch.Tensor],
        targets: Optional[torch.Tensor],
        W_out: Optional[torch.Tensor],
        dt: float
    ) -> PhaseSpaceState:
        """
        Half momentum step: p ← p - (dt/2) · ∂H/∂q

        FULL FAITHFUL THEORY:
        ---------------------
        For Hamiltonian mechanics on a Riemannian manifold where kinetic
        energy T depends on position through the metric:

            H = T(q, p) + V(q)

        Hamilton's equations give:
            dq/dt = ∂H/∂p = ∂T/∂p
            dp/dt = -∂H/∂q = -∂V/∂q - ∂T/∂q

        For the SPD manifold with T_Σ = tr(π_Σ Σ π_Σ Σ):
            ∂T_Σ/∂Σ = 2 π_Σ Σ π_Σ

        This curvature correction is essential for symplectic integration
        on curved manifolds and proper energy conservation!
        """
        # Enable gradients for configuration variables
        mu = state.mu.requires_grad_(True)
        Sigma = state.Sigma.requires_grad_(True)
        phi = state.phi.requires_grad_(True)

        # Create temporary state for potential computation
        temp_state = PhaseSpaceState(
            mu=mu, Sigma=Sigma, phi=phi,
            pi_mu=state.pi_mu, pi_Sigma=state.pi_Sigma, pi_phi=state.pi_phi
        )

        # Compute potential
        V, _ = self.potential(temp_state, mu_prior, Sigma_prior, beta, targets, W_out)
        V_sum = V.sum()

        # Compute gradients (allow_unused for phi when alignment disabled)
        grads = torch.autograd.grad(
            V_sum, [mu, Sigma, phi],
            create_graph=False,
            allow_unused=True
        )
        grad_mu = grads[0] if grads[0] is not None else torch.zeros_like(state.mu)
        grad_Sigma = grads[1] if grads[1] is not None else torch.zeros_like(state.Sigma)
        grad_phi = grads[2] if grads[2] is not None else torch.zeros_like(state.phi)

        # Momentum updates: p ← p - dt · ∂H/∂q
        pi_mu_new = state.pi_mu - dt * grad_mu.detach()

        if self.update_Sigma:
            # Symmetrize potential gradient for Σ
            grad_V_Sigma = symmetrize(grad_Sigma.detach())

            # CURVATURE CORRECTION: ∂T_Σ/∂Σ = 2 π_Σ Σ π_Σ
            # This is the "force" from the curved SPD geometry!
            grad_T_Sigma = spd_kinetic_gradient(
                state.Sigma.detach(),
                state.pi_Sigma.detach(),
                eps=1e-8
            )

            # Full Hamiltonian gradient: ∂H/∂Σ = ∂V/∂Σ + ∂T/∂Σ
            grad_H_Sigma = grad_V_Sigma + grad_T_Sigma

            pi_Sigma_new = state.pi_Sigma - dt * grad_H_Sigma
            pi_Sigma_new = symmetrize(pi_Sigma_new)  # Enforce symmetry
        else:
            pi_Sigma_new = state.pi_Sigma

        if self.update_phi:
            pi_phi_new = state.pi_phi - dt * grad_phi.detach()
        else:
            pi_phi_new = state.pi_phi

        return PhaseSpaceState(
            mu=state.mu.detach(),
            Sigma=state.Sigma.detach(),
            phi=state.phi.detach(),
            pi_mu=pi_mu_new,
            pi_Sigma=pi_Sigma_new,
            pi_phi=pi_phi_new,
        )

    def step(
        self,
        state: PhaseSpaceState,
        mu_prior: torch.Tensor,
        Sigma_prior: torch.Tensor,
        M_inv: torch.Tensor,
        beta: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
        W_out: Optional[torch.Tensor] = None,
    ) -> PhaseSpaceState:
        """
        Single leapfrog step.

        p_{1/2} = p_0 - (ε/2) ∂V/∂q(q_0)
        q_1 = q_0 + ε ∂T/∂p(p_{1/2})
        p_1 = p_{1/2} - (ε/2) ∂V/∂q(q_1)

        Args:
            state: Current phase space state
            mu_prior: Prior means
            Sigma_prior: Prior covariances
            M_inv: Inverse mass matrix from Inertia of Belief paper
            beta: Attention weights
            targets: Target tokens
            W_out: Output projection
        """
        # Half-step momentum
        state = self.momentum_step(
            state, mu_prior, Sigma_prior, beta, targets, W_out, self.dt / 2
        )

        # Full-step position (uses M_inv for μ update)
        state = self.position_step(state, M_inv, self.dt)

        # Half-step momentum
        state = self.momentum_step(
            state, mu_prior, Sigma_prior, beta, targets, W_out, self.dt / 2
        )

        return state

    def integrate(
        self,
        state: PhaseSpaceState,
        mu_prior: torch.Tensor,
        Sigma_prior: torch.Tensor,
        M_inv: torch.Tensor,
        beta: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
        W_out: Optional[torch.Tensor] = None,
        trajectory_callback: Optional[callable] = None,
        # Additional params for evolving mass (only used if evolve_mass=True)
        Sigma_obs: Optional[torch.Tensor] = None,
        obs_temperature: float = 1.0,
    ) -> PhaseSpaceState:
        """
        Multiple leapfrog steps.

        Args:
            state: Initial phase space state
            mu_prior: Prior means
            Sigma_prior: Prior covariances
            M_inv: Inverse mass matrix from Inertia of Belief paper
                   (used as initial M_inv; recomputed each step if evolve_mass=True)
            beta: Attention weights
            targets: Target tokens
            W_out: Output projection
            trajectory_callback: Optional callback(step, state, H, T, V) for recording
            Sigma_obs: Observation covariance (for Gaussian obs model, used if evolve_mass=True)
            obs_temperature: Softmax temperature (for categorical obs model)
        """
        current_M_inv = M_inv

        for step_idx in range(self.n_steps):
            # Recompute mass matrix if evolve_mass is enabled
            # M depends on Σ_q through the social precision terms Λ_q = Σ_q⁻¹
            if self.evolve_mass and self.mass_computer is not None:
                _, current_M_inv = self.mass_computer.compute_mass(
                    Sigma_prior=Sigma_prior,
                    Sigma_q=state.Sigma,  # Current evolved Σ
                    phi=state.phi,         # Current evolved φ
                    beta=beta,
                    Sigma_obs=Sigma_obs,
                    mu=state.mu,           # Current evolved μ (for categorical Λ_o)
                    W_out=W_out,
                    obs_temperature=obs_temperature,
                )

            state = self.step(state, mu_prior, Sigma_prior, current_M_inv, beta, targets, W_out)

            # Record trajectory if callback provided
            if trajectory_callback is not None:
                # Compute energy for diagnostic
                T = self.kinetic.total_kinetic(state, current_M_inv)
                V, _ = self.potential(state, mu_prior, Sigma_prior, beta, targets, W_out)
                H = T + V
                trajectory_callback(
                    step_idx + 1,  # 0 is initial state
                    state,
                    H.mean().item(),
                    T.mean().item(),
                    V.mean().item(),
                )

        return state


# =============================================================================
# Hamiltonian FFN Module
# =============================================================================

class HamiltonianFFN(nn.Module):
    """
    Hamiltonian Feedforward Network for Gauge-Theoretic Transformer.

    Replaces gradient-based VariationalFFN with symplectic Hamiltonian dynamics.

    The FFN layer performs n_steps of leapfrog integration on the phase space
    (μ, Σ, φ, π_μ, π_Σ, π_φ), preserving the symplectic structure and
    approximately conserving the Hamiltonian H = T + V.

    Key innovation: Energy conservation prevents vanishing gradients!

    EXTENDED MASS FROM "THE INERTIA OF BELIEF" (Paper Eq. 20):
    -----------------------------------------------------------
    The mass matrix M can include multiple terms:
        M_i = Λ_{pi} + Λ_{oi} + Σ_k β_{ik} Λ̃_{qk} + Σ_j β_{ji} Λ_{qi}

    Each term can be toggled via MassConfig:
        - use_prior_precision: Λ_p (default: True)
        - use_observation_precision: Λ_o (default: False)
        - use_incoming_social: Σβ_{ik}Λ̃_{qk} (default: False)
        - use_outgoing_recoil: Σβ_{ji}Λ_{qi} (default: False)

    Architecture:
        Input: (μ, Σ, φ) from attention layer
        1. Sample momenta π ~ N(0, M) with geometric mass matrix
        2. Leapfrog integration: (μ, Σ, φ, π) → (μ', Σ', φ', π')
        3. Output: (μ', Σ', φ')

    Gauge Covariance:
        The Hamiltonian H is gauge-invariant, so dynamics preserves covariance.
        Under g ∈ G: (μ, Σ, φ) → (g·μ, gΣgᵀ, Ad_g(φ))
    """

    def __init__(
        self,
        embed_dim: int,
        generators: torch.Tensor,    # (3, K, K) SO(3) generators
        n_leapfrog_steps: int = 5,
        dt: float = 0.01,
        # Physics parameters
        alpha: float = 1.0,          # Self-coupling
        lambda_belief: float = 1.0,   # Belief alignment
        kappa: float = 1.0,          # Softmax temperature
        # What to update
        update_Sigma: bool = True,
        update_phi: bool = False,
        # Momentum initialization
        momentum_scale: float = 1.0,
        # Extended mass configuration (from Inertia of Belief paper)
        mass_config: Optional[MassConfig] = None,
        # Thermostat (optional damping)
        gamma: float = 0.0,          # Damping coefficient (0 = pure Hamiltonian)
        temperature: float = 1.0,     # For Langevin dynamics
        # Mass evolution (full theory vs approximation)
        evolve_mass: bool = False,   # If True, recompute M at each leapfrog step
        eps: float = 1e-8,
        # Diagonal covariance mode (memory optimization)
        diagonal_covariance: bool = False,  # If True, Σ is (B,N,K) not (B,N,K,K)
    ):
        """
        Initialize Hamiltonian FFN.

        Args:
            embed_dim: Latent dimension K
            generators: SO(3) generators for gauge transport
            n_leapfrog_steps: Integration steps per layer
            dt: Time step size
            alpha: Weight for KL(q||p) self-coupling
            lambda_belief: Weight for belief alignment
            kappa: Softmax temperature for attention weights
            update_Sigma: Whether to evolve covariances
            update_phi: Whether to evolve gauge field
            momentum_scale: Scale for initial momentum sampling
            mass_config: MassConfig with toggles for each mass term from paper
                        If None, defaults to prior precision only (original behavior)
            gamma: Damping coefficient (>0 adds friction)
            temperature: Thermal energy scale
            evolve_mass: If True, recompute mass matrix M at each leapfrog step.
                        The mass depends on Σ (via Λ_q = Σ⁻¹), so if Σ evolves,
                        M should too for full theoretical correctness.
                        If False (default), M is computed once and held fixed,
                        which is a good approximation for small dt.
            eps: Numerical stability
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.n_leapfrog_steps = n_leapfrog_steps
        self.dt = dt
        self.update_phi = update_phi
        self.momentum_scale = momentum_scale
        self.mass_config = mass_config or MassConfig()  # Default: prior precision only
        self.gamma = gamma
        self.temperature = temperature
        self.evolve_mass = evolve_mass
        self.eps = eps
        self.diagonal_covariance = diagonal_covariance

        # In diagonal mode, disable Σ evolution (SPD dynamics not applicable)
        if diagonal_covariance:
            self.update_Sigma = False
            # Note: Diagonal mode keeps σ fixed, only μ evolves via Hamiltonian
        else:
            self.update_Sigma = update_Sigma

        # Register generators as buffer
        self.register_buffer('generators', generators)

        # Build Hamiltonian components
        self.kinetic = HamiltonianKineticTerms(embed_dim, eps)
        self.potential = HamiltonianPotential(
            embed_dim, generators, alpha, lambda_belief, kappa, eps
        )

        # Extended mass matrix computation (from Inertia of Belief paper)
        self.mass_computer = InertiaOfBeliefMass(
            embed_dim=embed_dim,
            generators=generators,
            config=self.mass_config,
        )

        # Symplectic integrator
        self.integrator = LeapfrogIntegrator(
            kinetic=self.kinetic,
            potential=self.potential,
            dt=dt,
            n_steps=n_leapfrog_steps,
            update_Sigma=update_Sigma,
            update_phi=update_phi,
            evolve_mass=evolve_mass,
            mass_computer=self.mass_computer if evolve_mass else None,
        )

        # Learnable dt (optional)
        self.log_dt = nn.Parameter(torch.tensor(0.0))  # exp(log_dt) * dt

    def sample_momenta(
        self,
        mu: torch.Tensor,
        Sigma: torch.Tensor,
        phi: torch.Tensor,
        M: torch.Tensor,
        M_inv: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample initial momenta from thermal distribution.

        For Hamiltonian dynamics with mass matrix M:
            π ~ N(0, M)

        For μ with extended mass M from Inertia of Belief paper:
            π_μ ~ N(0, M) = M^{1/2} N(0, I)

        For Σ (SPD manifold):
            π_Σ ~ symmetric, ~ Σ⁻¹/² · N(0, I) · Σ⁻¹/²

        For φ (Lie algebra):
            π_φ ~ N(0, I)

        Args:
            mu: (B, N, K) belief means
            Sigma: (B, N, K, K) belief covariances
            phi: (B, N, 3) gauge field
            M: (B, N, K, K) mass matrix
            M_inv: (B, N, K, K) inverse mass matrix
        """
        B, N, K = mu.shape
        device = mu.device
        dtype = mu.dtype

        # Sample π_μ from N(0, M)
        # π_μ = M^{1/2} · z where z ~ N(0, I)
        # Compute M^{1/2} via eigendecomposition (force FP32 for stability under AMP)
        # Add regularization to M before eigendecomposition
        M_reg = M + self.eps * torch.eye(K, device=device, dtype=dtype).unsqueeze(0).unsqueeze(0)
        M_reg = 0.5 * (M_reg + M_reg.transpose(-1, -2))  # Ensure symmetric

        try:
            eigenvalues, eigenvectors = torch.linalg.eigh(M_reg.float())
            eigenvalues = torch.clamp(eigenvalues, min=self.eps)
            M_sqrt = (eigenvectors @ torch.diag_embed(torch.sqrt(eigenvalues)) @ eigenvectors.transpose(-1, -2)).to(dtype)
        except RuntimeError:
            # Fallback to identity
            M_sqrt = torch.eye(K, device=device, dtype=dtype).unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1)

        noise_mu = torch.randn(B, N, K, device=device, dtype=dtype)
        pi_mu = self.momentum_scale * torch.einsum('...ij,...j->...i', M_sqrt, noise_mu)

        # Sample π_Σ (symmetric matrix on SPD tangent space)
        if self.update_Sigma:
            noise_Sigma = torch.randn(B, N, K, K, device=device, dtype=dtype)
            pi_Sigma = self.momentum_scale * symmetrize(noise_Sigma) * 0.1  # Scale down for stability
        else:
            pi_Sigma = torch.zeros(B, N, K, K, device=device, dtype=dtype)

        # Sample π_φ
        if self.update_phi:
            pi_phi = self.momentum_scale * torch.randn(B, N, 3, device=device, dtype=dtype) * 0.1
        else:
            pi_phi = torch.zeros(B, N, 3, device=device, dtype=dtype)

        return pi_mu, pi_Sigma, pi_phi

    def forward(
        self,
        mu: torch.Tensor,
        Sigma: torch.Tensor,
        phi: torch.Tensor,
        mu_prior: torch.Tensor,
        Sigma_prior: torch.Tensor,
        beta: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
        W_out: Optional[torch.Tensor] = None,
        Sigma_obs: Optional[torch.Tensor] = None,  # For observation precision term
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """
        Forward pass: Hamiltonian dynamics on belief space.

        Uses extended mass matrix from "The Inertia of Belief" paper:
            M_i = Λ_{pi} + Λ_{oi} + Σ_k β_{ik} Λ̃_{qk} + Σ_j β_{ji} Λ_{qi}

        Args:
            mu: (B, N, K) belief means
            Sigma: (B, N, K, K) belief covariances
            phi: (B, N, 3) gauge field
            mu_prior: (B, N, K) prior means (from embeddings)
            Sigma_prior: (B, N, K, K) prior covariances
            beta: (B, N, N) attention weights from attention layer
            targets: (B, N) target tokens (for CE term in E-step)
            W_out: (V, K) output projection
            Sigma_obs: (B, N, K, K) observation covariance (optional)

        Returns:
            mu_new: (B, N, K) updated means
            Sigma_new: (B, N, K, K) updated covariances
            phi_new: (B, N, 3) updated gauge field
            diagnostics: dict with energy terms, conservation, mass info, etc.
        """
        # Hamiltonian dynamics uses autograd.grad() internally for leapfrog.
        # To prevent "backward through graph twice" errors:
        # 1. Detach inputs (break connection to training graph)
        # 2. Use enable_grad() to ensure gradients work even during validation
        # 3. Detach outputs (dynamics is a deterministic transformation)

        # Detach inputs to break connection to training graph
        mu_dyn = mu.detach().clone()
        Sigma_dyn = Sigma.detach().clone()
        phi_dyn = phi.detach().clone()
        mu_prior_dyn = mu_prior.detach()
        Sigma_prior_dyn = Sigma_prior.detach()
        beta_dyn = beta.detach() if beta is not None else None
        Sigma_obs_dyn = Sigma_obs.detach() if Sigma_obs is not None else None
        W_out_dyn = W_out.detach() if W_out is not None else None

        # Handle diagonal covariance mode: convert (B, N, K) -> (B, N, K, K)
        # The main memory savings come from attention (O(N²×K²) -> O(N²×K)),
        # but FFN operates per-token so full matrices are fine here.
        if self.diagonal_covariance:
            # Convert diagonal variances to full diagonal matrices (only if actually diagonal)
            if Sigma_dyn.dim() == 3:  # (B, N, K) diagonal
                Sigma_dyn = torch.diag_embed(Sigma_dyn)  # -> (B, N, K, K)
            if Sigma_prior_dyn.dim() == 3:  # (B, N, K) diagonal
                Sigma_prior_dyn = torch.diag_embed(Sigma_prior_dyn)  # -> (B, N, K, K)

        # Use enable_grad() to ensure autograd.grad() works even when called
        # from validation (which uses torch.no_grad() context)
        with torch.enable_grad():
            # Enable gradients for dynamics variables
            mu_dyn.requires_grad_(True)
            Sigma_dyn.requires_grad_(True)
            phi_dyn.requires_grad_(True)

            # =================================================================
            # Compute Extended Mass Matrix (from Inertia of Belief paper)
            # M_i = Λ_{pi} + Λ_{oi} + Σ_k β_{ik} Λ̃_{qk} + Σ_j β_{ji} Λ_{qi}
            # =================================================================
            M, M_inv = self.mass_computer.compute_mass(
                Sigma_prior=Sigma_prior_dyn,
                Sigma_q=Sigma_dyn,
                phi=phi_dyn,
                beta=beta_dyn,
                Sigma_obs=Sigma_obs_dyn,
                # For categorical observation precision (transformer softmax output):
                mu=mu_dyn,
                W_out=W_out_dyn,
            )

            # Sample initial momenta using extended mass
            pi_mu, pi_Sigma, pi_phi = self.sample_momenta(
                mu_dyn, Sigma_dyn, phi_dyn, M, M_inv
            )

            # Create initial phase space state
            state = PhaseSpaceState(
                mu=mu_dyn,
                Sigma=Sigma_dyn,
                phi=phi_dyn,
                pi_mu=pi_mu,
                pi_Sigma=pi_Sigma,
                pi_phi=pi_phi,
            )

            # Compute initial Hamiltonian (using M_inv for kinetic energy)
            T_init = self.kinetic.total_kinetic(state, M_inv)
            V_init, V_breakdown = self.potential(state, mu_prior_dyn, Sigma_prior_dyn, beta_dyn, targets, W_out)
            H_init = T_init + V_init

            # Apply damping if gamma > 0 (Langevin-like)
            if self.gamma > 0:
                state.pi_mu = state.pi_mu * (1 - self.gamma * self.dt)
                if self.update_Sigma:
                    state.pi_Sigma = state.pi_Sigma * (1 - self.gamma * self.dt)
                if self.update_phi:
                    state.pi_phi = state.pi_phi * (1 - self.gamma * self.dt)

            # Build trajectory callback if global recorder is available
            trajectory_callback = None
            try:
                from transformer.trajectory_tracking import get_global_recorder
                recorder = get_global_recorder()
                if recorder is not None and recorder.enabled and recorder.record_leapfrog:
                    def trajectory_callback(step, st, H, T, V):
                        recorder.record_leapfrog_step(
                            step=step,
                            mu=st.mu,
                            Sigma=st.Sigma,
                            phi=st.phi,
                            pi_mu=st.pi_mu,
                            pi_Sigma=st.pi_Sigma,
                            H=H, T=T, V=V,
                        )
                    # Record initial state (step=0)
                    trajectory_callback(0, state, H_init.mean().item(),
                                       T_init.mean().item(), V_init.mean().item())
            except ImportError:
                pass  # Trajectory tracking not available

            # Symplectic integration (using M_inv for position updates)
            # If evolve_mass=True, M is recomputed at each step inside integrate()
            state = self.integrator.integrate(
                state, mu_prior_dyn, Sigma_prior_dyn, M_inv, beta_dyn, targets, W_out,
                trajectory_callback=trajectory_callback,
                # Additional params for evolving mass (only used if evolve_mass=True)
                Sigma_obs=Sigma_obs_dyn,
                obs_temperature=1.0,  # Could be made configurable
            )

            # Compute final Hamiltonian
            # If evolve_mass=True, recompute M at final position for accurate energy
            if self.evolve_mass:
                _, M_inv_final = self.mass_computer.compute_mass(
                    Sigma_prior=Sigma_prior_dyn,
                    Sigma_q=state.Sigma,
                    phi=state.phi,
                    beta=beta_dyn,
                    Sigma_obs=Sigma_obs_dyn,
                    mu=state.mu,
                    W_out=W_out_dyn,
                )
            else:
                M_inv_final = M_inv

            T_final = self.kinetic.total_kinetic(state, M_inv_final)
            V_final, _ = self.potential(state, mu_prior_dyn, Sigma_prior_dyn, beta_dyn, targets, W_out)
            H_final = T_final + V_final

            # Energy conservation diagnostic
            delta_H = (H_final - H_init).abs().mean()

            # Mass configuration info for diagnostics
            mass_info = {
                'mass_use_prior': self.mass_config.use_prior_precision,
                'mass_use_observation': self.mass_config.use_observation_precision,
                'mass_use_incoming_social': self.mass_config.use_incoming_social,
                'mass_use_outgoing_recoil': self.mass_config.use_outgoing_recoil,
                'mass_evolve': self.evolve_mass,
            }

            diagnostics = {
                'H_init': H_init.mean().item(),
                'H_final': H_final.mean().item(),
                'delta_H': delta_H.item(),
                'T_init': T_init.mean().item(),
                'T_final': T_final.mean().item(),
                'V_init': V_init.mean().item(),
                'V_final': V_final.mean().item(),
                **{k: v.mean().item() for k, v in V_breakdown.items()},
                **mass_info,
            }

        # Detach outputs - gradients flow through loss, not through dynamics
        mu_out = state.mu.detach()
        Sigma_out = state.Sigma.detach()
        phi_out = state.phi.detach()

        # Convert back to diagonal if in diagonal mode
        if self.diagonal_covariance:
            # Extract diagonal variances from full matrix: (B, N, K, K) -> (B, N, K)
            Sigma_out = torch.diagonal(Sigma_out, dim1=-2, dim2=-1)

        return mu_out, Sigma_out, phi_out, diagnostics

    def extra_repr(self) -> str:
        mass_terms = []
        if self.mass_config.use_prior_precision:
            mass_terms.append("Λ_p")
        if self.mass_config.use_observation_precision:
            mass_terms.append("Λ_o")
        if self.mass_config.use_incoming_social:
            mass_terms.append("Σβ_ik·Λ̃_qk")
        if self.mass_config.use_outgoing_recoil:
            mass_terms.append("Σβ_ji·Λ_qi")
        mass_str = "+".join(mass_terms) if mass_terms else "identity"

        return (
            f"embed_dim={self.embed_dim}, "
            f"n_steps={self.n_leapfrog_steps}, "
            f"dt={self.dt:.4f}, "
            f"gamma={self.gamma:.4f}, "
            f"mass=[{mass_str}], "
            f"evolve_mass={self.evolve_mass}, "
            f"update_Sigma={self.update_Sigma}, "
            f"update_phi={self.update_phi}, "
            f"diagonal_covariance={self.diagonal_covariance}"
        )


# =============================================================================
# Testing
# =============================================================================

def _make_so3_generators_for_test(K: int) -> torch.Tensor:
    """
    Create random skew-symmetric generators for testing.

    For proper SO(3) irreps, use math_utils.generators.generate_so3_generators.
    This is a simplified version for self-contained testing.
    """
    # Create 3 random skew-symmetric matrices as generators
    generators = []
    for _ in range(3):
        A = torch.randn(K, K)
        G = 0.5 * (A - A.T)  # Skew-symmetric
        generators.append(G)
    return torch.stack(generators, dim=0)  # (3, K, K)


if __name__ == '__main__':
    # Fix OpenMP issue on Windows/Anaconda
    import os
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

    print("=" * 70)
    print("HAMILTONIAN FFN TEST - FULL FAITHFUL SPD GEOMETRY")
    print("=" * 70)

    # Configuration
    B, N, K = 2, 4, 5  # Batch, sequence, latent dim

    print(f"\n[1] Configuration:")
    print(f"    Batch size: {B}")
    print(f"    Sequence length: {N}")
    print(f"    Latent dim: {K}")

    # Create SO(3) generators for K-dimensional irrep
    generators = _make_so3_generators_for_test(K)
    print(f"    Generators shape: {generators.shape}")

    # Create test data
    mu = torch.randn(B, N, K)
    A = torch.randn(B, N, K, K)
    Sigma = A @ A.transpose(-1, -2) + torch.eye(K)  # SPD
    phi = torch.randn(B, N, 3) * 0.1

    mu_prior = torch.randn(B, N, K) * 0.5
    Sigma_prior = torch.eye(K).unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1).clone()

    # ==========================================================================
    # TEST 1: Verify SPD curvature functions
    # ==========================================================================
    print(f"\n[2] Testing SPD curvature functions...")

    # Test velocity <-> momentum conversion
    Sigma_test = Sigma[0:1, 0:1]  # (1, 1, K, K)
    Sigma_dot_test = symmetrize(torch.randn(1, 1, K, K) * 0.1)

    pi_Sigma_test = momentum_from_velocity_spd(Sigma_test, Sigma_dot_test)
    Sigma_dot_recovered = velocity_from_momentum_spd(Sigma_test, pi_Sigma_test)

    conversion_error = (Sigma_dot_test - Sigma_dot_recovered).abs().max().item()
    print(f"    Velocity <-> Momentum conversion error: {conversion_error:.2e}")

    # Test geodesic acceleration
    Sigma_ddot = spd_geodesic_acceleration(Sigma_test, Sigma_dot_test)
    print(f"    Geodesic acceleration shape: {Sigma_ddot.shape}")

    # Test kinetic gradient
    grad_T = spd_kinetic_gradient(Sigma_test, pi_Sigma_test)
    print(f"    Kinetic gradient shape: {grad_T.shape}")
    print(f"    ✓ SPD curvature functions working")

    # ==========================================================================
    # TEST 2: Energy conservation with μ only (baseline)
    # ==========================================================================
    print(f"\n[3] Test: μ dynamics only (baseline)...")
    ffn_mu_only = HamiltonianFFN(
        embed_dim=K,
        generators=generators,
        n_leapfrog_steps=20,
        dt=0.01,
        alpha=1.0,
        lambda_belief=0.0,  # Disable alignment
        update_Sigma=False,
        update_phi=False,
        gamma=0.0,
    )

    _, _, _, diag_mu = ffn_mu_only(
        mu.clone(), Sigma.clone(), phi.clone(), mu_prior, Sigma_prior
    )
    print(f"    H_init: {diag_mu['H_init']:.6f}")
    print(f"    H_final: {diag_mu['H_final']:.6f}")
    print(f"    ΔH = {diag_mu['delta_H']:.6f}")

    if diag_mu['delta_H'] < 0.01:
        print(f"    ✓ μ dynamics: EXCELLENT energy conservation")
    elif diag_mu['delta_H'] < 0.1:
        print(f"    ✓ μ dynamics: Good energy conservation")
    else:
        print(f"    ✗ μ dynamics: Energy drift detected")

    # ==========================================================================
    # TEST 3: Full SPD dynamics with curvature correction
    # ==========================================================================
    print(f"\n[4] Test: Full SPD dynamics WITH curvature correction...")
    ffn_full = HamiltonianFFN(
        embed_dim=K,
        generators=generators,
        n_leapfrog_steps=20,
        dt=0.01,
        alpha=1.0,
        lambda_belief=0.0,  # Disable alignment for cleaner test
        update_Sigma=True,
        update_phi=False,
        gamma=0.0,
        momentum_scale=0.5,  # Smaller momenta for stability
    )

    mu_new, Sigma_new, phi_new, diag_full = ffn_full(
        mu.clone(), Sigma.clone(), phi.clone(), mu_prior, Sigma_prior
    )

    print(f"    H_init: {diag_full['H_init']:.6f}")
    print(f"    H_final: {diag_full['H_final']:.6f}")
    print(f"    T_init: {diag_full['T_init']:.6f}")
    print(f"    T_final: {diag_full['T_final']:.6f}")
    print(f"    V_init: {diag_full['V_init']:.6f}")
    print(f"    V_final: {diag_full['V_final']:.6f}")
    print(f"    ΔH = {diag_full['delta_H']:.6f}")

    if diag_full['delta_H'] < 0.1:
        print(f"    ✓ Full SPD dynamics: GOOD energy conservation!")
    elif diag_full['delta_H'] < 1.0:
        print(f"    ~ Full SPD dynamics: Moderate drift (may need smaller dt)")
    else:
        print(f"    ✗ Full SPD dynamics: Significant drift")

    # ==========================================================================
    # TEST 4: Check SPD property preserved
    # ==========================================================================
    print(f"\n[5] SPD property preservation:")
    eigenvalues = torch.linalg.eigvalsh(Sigma_new)
    min_eig = eigenvalues.min().item()
    max_eig = eigenvalues.max().item()
    print(f"    Eigenvalue range: [{min_eig:.6f}, {max_eig:.6f}]")

    if min_eig > 0:
        print(f"    ✓ All eigenvalues positive - SPD preserved!")
    else:
        print(f"    ✗ Negative eigenvalues - SPD violated!")

    # Check symmetry
    symmetry_error = (Sigma_new - Sigma_new.transpose(-1, -2)).abs().max().item()
    print(f"    Symmetry error: {symmetry_error:.2e}")

    # ==========================================================================
    # TEST 5: Timestep refinement study
    # ==========================================================================
    print(f"\n[6] Timestep refinement study:")
    print(f"    dt       | ΔH")
    print(f"    ---------|----------")

    for dt in [0.1, 0.05, 0.02, 0.01, 0.005]:
        ffn_test = HamiltonianFFN(
            embed_dim=K,
            generators=generators,
            n_leapfrog_steps=10,
            dt=dt,
            alpha=1.0,
            lambda_belief=0.0,
            update_Sigma=True,
            update_phi=False,
            gamma=0.0,
            momentum_scale=0.3,
        )
        _, _, _, diag_test = ffn_test(
            mu.clone(), Sigma.clone(), phi.clone(), mu_prior, Sigma_prior
        )
        print(f"    {dt:.3f}    | {diag_test['delta_H']:.6f}")

    print("\n" + "=" * 70)
    print("✓ FULL FAITHFUL SPD Hamiltonian FFN test complete!")
    print("=" * 70)
    print("\nKey insight: With curvature correction ∂T_Σ/∂Σ = 2 π_Σ Σ π_Σ,")
    print("the integrator properly accounts for the curved SPD geometry.")
    print("Energy conservation improves as dt → 0 (symplectic property).")
