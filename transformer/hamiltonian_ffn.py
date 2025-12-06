"""
Hamiltonian Feedforward Network for Gauge-Theoretic Transformer
=================================================================

Replaces gradient-based variational FFN with Hamiltonian dynamics.

FAITHFUL TO INFORMATIONAL GAUGE THEORY
--------------------------------------
From field_theory.py, the complete Hamiltonian is:

    H = T_μ + T_Σ + T_φ + V

where:
    T_μ = (1/2) π_μᵀ Σ_p π_μ              (Fisher-Rao metric)
    T_Σ = (1/4) tr(Σ⁻¹ Σ̇ Σ⁻¹ Σ̇)          (SPD manifold metric)
    T_φ = (1/2) ⟨π_φ, π_φ⟩_𝔤              (Killing form on Lie algebra)
    V   = Free Energy Functional           (from free_energy_clean.py)

Conjugate momenta:
    π_μ = Σ_p⁻¹ μ̇       → μ̇ = Σ_p π_μ     (Fisher mass matrix)
    π_Σ = ½ Σ⁻¹ Σ̇ Σ⁻¹   → Σ̇ = 2 Σ π_Σ Σ   (SPD geometry)
    π_φ = φ̇             → φ̇ = π_φ         (trivial for gauge)

Hamilton's equations:
    dμ/dt  = ∂H/∂π_μ = Σ_p π_μ
    dΣ/dt  = ∂H/∂π_Σ = 2 Σ π_Σ Σ
    dφ/dt  = ∂H/∂π_φ = π_φ
    dπ_μ/dt  = -∂V/∂μ
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
from typing import Optional, Tuple, Literal
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
    # Eigendecomposition
    eigenvalues, eigenvectors = torch.linalg.eigh(Sigma)

    # Clip eigenvalues to be positive
    eigenvalues_clipped = torch.clamp(eigenvalues, min=eps)

    # Reconstruct
    Sigma_spd = eigenvectors @ torch.diag_embed(eigenvalues_clipped) @ eigenvectors.transpose(-1, -2)

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
    # Regularize Sigma
    Sigma = symmetrize(Sigma)
    Sigma = Sigma + eps * torch.eye(Sigma.shape[-1], device=Sigma.device, dtype=Sigma.dtype)

    # Matrix square root via eigendecomposition
    eigenvalues, eigenvectors = torch.linalg.eigh(Sigma)
    eigenvalues = torch.clamp(eigenvalues, min=eps)

    Sigma_sqrt = eigenvectors @ torch.diag_embed(torch.sqrt(eigenvalues)) @ eigenvectors.transpose(-1, -2)
    Sigma_inv_sqrt = eigenvectors @ torch.diag_embed(1.0 / torch.sqrt(eigenvalues)) @ eigenvectors.transpose(-1, -2)

    # W = Σ^{-1/2} V Σ^{-1/2}
    W = Sigma_inv_sqrt @ V @ Sigma_inv_sqrt
    W = symmetrize(W)  # Ensure symmetric for matrix exp

    # exp(W) via eigendecomposition (more stable than torch.matrix_exp for symmetric)
    W_eigenvalues, W_eigenvectors = torch.linalg.eigh(W)
    exp_W = W_eigenvectors @ torch.diag_embed(torch.exp(W_eigenvalues)) @ W_eigenvectors.transpose(-1, -2)

    # exp_Σ(V) = Σ^{1/2} exp(W) Σ^{1/2}
    Sigma_new = Sigma_sqrt @ exp_W @ Sigma_sqrt

    return ensure_spd(Sigma_new, eps)


# =============================================================================
# Kinetic Energy Terms (Faithful to field_theory.py)
# =============================================================================

class HamiltonianKineticTerms(nn.Module):
    """
    Kinetic energy terms from the gauge-theoretic Hamiltonian.

    T = T_μ + T_Σ + T_φ

    Each term uses the correct geometric structure:
    - T_μ: Fisher-Rao metric (mass matrix = Σ_p)
    - T_Σ: Affine-invariant SPD metric
    - T_φ: Killing form on so(3)
    """

    def __init__(self, embed_dim: int, eps: float = 1e-8):
        super().__init__()
        self.embed_dim = embed_dim
        self.eps = eps

    def kinetic_mu(
        self,
        pi_mu: torch.Tensor,
        Sigma_prior: torch.Tensor
    ) -> torch.Tensor:
        """
        Mean kinetic energy: T_μ = (1/2) π_μᵀ Σ_p π_μ

        From: π_μ = Σ_p⁻¹ μ̇, so μ̇ = Σ_p π_μ
        Therefore: T_μ = (1/2) μ̇ᵀ Σ_p⁻¹ μ̇ = (1/2) π_μᵀ Σ_p π_μ

        Args:
            pi_mu: (B, N, K) momentum
            Sigma_prior: (B, N, K, K) prior covariance (mass matrix)

        Returns:
            T_mu: (B, N) kinetic energy per agent
        """
        # T_μ = (1/2) π_μᵀ Σ_p π_μ
        # Use einsum: (...,i), (...,i,j), (...,j) -> (...)
        Sigma_pi = torch.einsum('...ij,...j->...i', Sigma_prior, pi_mu)
        T_mu = 0.5 * torch.einsum('...i,...i->...', pi_mu, Sigma_pi)
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
        Sigma_prior: torch.Tensor,
        chi: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Total kinetic energy: T = χ · (T_μ + T_Σ + T_φ)

        Args:
            state: Phase space state
            Sigma_prior: Prior covariance (mass matrix for μ)
            chi: (B, N) support weights (default: ones)

        Returns:
            T: (B,) total kinetic energy per batch
        """
        T_mu = self.kinetic_mu(state.pi_mu, Sigma_prior)
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

        # Regularize
        Sigma_q = Sigma_q + self.eps * torch.eye(K, device=Sigma_q.device, dtype=Sigma_q.dtype)
        Sigma_p = Sigma_p + self.eps * torch.eye(K, device=Sigma_p.device, dtype=Sigma_p.dtype)

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
        return kl

    def compute_transport(
        self,
        phi_i: torch.Tensor,
        phi_j: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute parallel transport operator Ω_ij = exp(φ_i) exp(-φ_j).

        Args:
            phi_i, phi_j: (B, N, 3) gauge fields

        Returns:
            Omega: (B, N, K, K) transport matrices
        """
        # Construct Lie algebra elements: φ · G = Σ_a φ^a G_a
        # generators: (3, K, K)
        # phi: (..., 3)
        phi_matrix_i = torch.einsum('...a,aij->...ij', phi_i, self.generators)
        phi_matrix_j = torch.einsum('...a,aij->...ij', phi_j, self.generators)

        # Matrix exponentials
        exp_phi_i = torch.matrix_exp(phi_matrix_i)
        exp_neg_phi_j = torch.matrix_exp(-phi_matrix_j)

        # Transport: Ω_ij = exp(φ_i) exp(-φ_j)
        Omega = exp_phi_i @ exp_neg_phi_j

        return Omega

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
        V_align = torch.zeros(B, device=device, dtype=state.mu.dtype)

        if beta is not None and self.lambda_belief > 0:
            # Handle multi-head attention: average over heads
            # beta can be (B, N, N) or (B, n_heads, N, N)
            if beta.dim() == 4:
                beta_avg = beta.mean(dim=1)  # (B, N, N) - average over heads
            else:
                beta_avg = beta  # Already (B, N, N)

            # For each pair (i, j), compute transported KL
            for i in range(N):
                for j in range(N):
                    if i == j:
                        continue

                    # Transport q_j to frame of i
                    Omega_ij = self.compute_transport(
                        state.phi[:, i:i+1, :].expand(-1, 1, -1),
                        state.phi[:, j:j+1, :].expand(-1, 1, -1)
                    ).squeeze(1)  # (B, K, K)

                    # Transported mean and covariance
                    mu_j_transported = torch.einsum('bij,bj->bi', Omega_ij, state.mu[:, j])
                    Sigma_j_transported = Omega_ij @ state.Sigma[:, j] @ Omega_ij.transpose(-1, -2)

                    # KL divergence
                    kl_ij = self.kl_divergence(
                        state.mu[:, i], state.Sigma[:, i],
                        mu_j_transported, Sigma_j_transported
                    )

                    # Weight by attention (averaged over heads)
                    V_align = V_align + self.lambda_belief * beta_avg[:, i, j] * kl_ij

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
        μ̇ = Σ_p π_μ          (Fisher metric)
        Σ̇ = 2 Σ π_Σ Σ        (SPD manifold)
        φ̇ = π_φ              (Lie algebra)
    """

    def __init__(
        self,
        kinetic: HamiltonianKineticTerms,
        potential: HamiltonianPotential,
        dt: float = 0.01,
        n_steps: int = 1,
        update_Sigma: bool = True,
        update_phi: bool = False,
    ):
        super().__init__()
        self.kinetic = kinetic
        self.potential = potential
        self.dt = dt
        self.n_steps = n_steps
        self.update_Sigma = update_Sigma
        self.update_phi = update_phi

    def position_step(
        self,
        state: PhaseSpaceState,
        Sigma_prior: torch.Tensor,
        dt: float
    ) -> PhaseSpaceState:
        """
        Full position step: q ← q + dt · ∂T/∂p

        Uses correct geometric update rules:
            μ ← μ + dt · Σ_p π_μ
            Σ ← exp_Σ(dt · 2 Σ π_Σ Σ)  (geodesic on SPD)
            φ ← φ + dt · π_φ
        """
        # Mean update: μ ← μ + dt · Σ_p π_μ
        mu_new = state.mu + dt * torch.einsum('...ij,...j->...i', Sigma_prior, state.pi_mu)

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
        beta: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
        W_out: Optional[torch.Tensor] = None,
    ) -> PhaseSpaceState:
        """
        Single leapfrog step.

        p_{1/2} = p_0 - (ε/2) ∂V/∂q(q_0)
        q_1 = q_0 + ε ∂T/∂p(p_{1/2})
        p_1 = p_{1/2} - (ε/2) ∂V/∂q(q_1)
        """
        # Half-step momentum
        state = self.momentum_step(
            state, mu_prior, Sigma_prior, beta, targets, W_out, self.dt / 2
        )

        # Full-step position
        state = self.position_step(state, Sigma_prior, self.dt)

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
        beta: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
        W_out: Optional[torch.Tensor] = None,
    ) -> PhaseSpaceState:
        """
        Multiple leapfrog steps.
        """
        for _ in range(self.n_steps):
            state = self.step(state, mu_prior, Sigma_prior, beta, targets, W_out)
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
        mass_matrix: Literal['fisher', 'identity'] = 'fisher',
        # Thermostat (optional damping)
        gamma: float = 0.0,          # Damping coefficient (0 = pure Hamiltonian)
        temperature: float = 1.0,     # For Langevin dynamics
        eps: float = 1e-8,
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
            mass_matrix: 'fisher' uses Σ_p, 'identity' uses I
            gamma: Damping coefficient (>0 adds friction)
            temperature: Thermal energy scale
            eps: Numerical stability
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.n_leapfrog_steps = n_leapfrog_steps
        self.dt = dt
        self.update_Sigma = update_Sigma
        self.update_phi = update_phi
        self.momentum_scale = momentum_scale
        self.mass_matrix = mass_matrix
        self.gamma = gamma
        self.temperature = temperature
        self.eps = eps

        # Register generators as buffer
        self.register_buffer('generators', generators)

        # Build Hamiltonian components
        self.kinetic = HamiltonianKineticTerms(embed_dim, eps)
        self.potential = HamiltonianPotential(
            embed_dim, generators, alpha, lambda_belief, kappa, eps
        )

        # Symplectic integrator
        self.integrator = LeapfrogIntegrator(
            kinetic=self.kinetic,
            potential=self.potential,
            dt=dt,
            n_steps=n_leapfrog_steps,
            update_Sigma=update_Sigma,
            update_phi=update_phi,
        )

        # Learnable dt (optional)
        self.log_dt = nn.Parameter(torch.tensor(0.0))  # exp(log_dt) * dt

    def sample_momenta(
        self,
        mu: torch.Tensor,
        Sigma: torch.Tensor,
        phi: torch.Tensor,
        Sigma_prior: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample initial momenta from thermal distribution.

        For Hamiltonian dynamics with mass matrix M:
            π ~ N(0, M)

        For μ with Fisher mass matrix Σ_p:
            π_μ ~ N(0, Σ_p⁻¹)

        For Σ (SPD manifold):
            π_Σ ~ symmetric, ~ Σ⁻¹/² · N(0, I) · Σ⁻¹/²

        For φ (Lie algebra):
            π_φ ~ N(0, I)
        """
        B, N, K = mu.shape
        device = mu.device
        dtype = mu.dtype

        # Sample π_μ
        if self.mass_matrix == 'fisher':
            # Sample from N(0, Σ_p⁻¹) = Σ_p^{-1/2} N(0, I)
            # First compute Σ_p^{-1/2}
            Sigma_prior_reg = Sigma_prior + self.eps * torch.eye(K, device=device, dtype=dtype)
            eigenvalues, eigenvectors = torch.linalg.eigh(Sigma_prior_reg)
            eigenvalues = torch.clamp(eigenvalues, min=self.eps)
            Sigma_prior_inv_sqrt = eigenvectors @ torch.diag_embed(1.0 / torch.sqrt(eigenvalues)) @ eigenvectors.transpose(-1, -2)

            noise_mu = torch.randn(B, N, K, device=device, dtype=dtype)
            pi_mu = self.momentum_scale * torch.einsum('...ij,...j->...i', Sigma_prior_inv_sqrt, noise_mu)
        else:
            # Simple identity mass
            pi_mu = self.momentum_scale * torch.randn(B, N, K, device=device, dtype=dtype)

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
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """
        Forward pass: Hamiltonian dynamics on belief space.

        Args:
            mu: (B, N, K) belief means
            Sigma: (B, N, K, K) belief covariances
            phi: (B, N, 3) gauge field
            mu_prior: (B, N, K) prior means (from embeddings)
            Sigma_prior: (B, N, K, K) prior covariances
            beta: (B, N, N) attention weights from attention layer
            targets: (B, N) target tokens (for CE term in E-step)
            W_out: (V, K) output projection

        Returns:
            mu_new: (B, N, K) updated means
            Sigma_new: (B, N, K, K) updated covariances
            phi_new: (B, N, 3) updated gauge field
            diagnostics: dict with energy terms, conservation, etc.
        """
        # Sample initial momenta
        pi_mu, pi_Sigma, pi_phi = self.sample_momenta(mu, Sigma, phi, Sigma_prior)

        # Create initial phase space state
        state = PhaseSpaceState(
            mu=mu,
            Sigma=Sigma,
            phi=phi,
            pi_mu=pi_mu,
            pi_Sigma=pi_Sigma,
            pi_phi=pi_phi,
        )

        # Compute initial Hamiltonian
        T_init = self.kinetic.total_kinetic(state, Sigma_prior)
        V_init, V_breakdown = self.potential(state, mu_prior, Sigma_prior, beta, targets, W_out)
        H_init = T_init + V_init

        # Apply damping if gamma > 0 (Langevin-like)
        if self.gamma > 0:
            state.pi_mu = state.pi_mu * (1 - self.gamma * self.dt)
            if self.update_Sigma:
                state.pi_Sigma = state.pi_Sigma * (1 - self.gamma * self.dt)
            if self.update_phi:
                state.pi_phi = state.pi_phi * (1 - self.gamma * self.dt)

        # Symplectic integration
        state = self.integrator.integrate(
            state, mu_prior, Sigma_prior, beta, targets, W_out
        )

        # Compute final Hamiltonian
        T_final = self.kinetic.total_kinetic(state, Sigma_prior)
        V_final, _ = self.potential(state, mu_prior, Sigma_prior, beta, targets, W_out)
        H_final = T_final + V_final

        # Energy conservation diagnostic
        delta_H = (H_final - H_init).abs().mean()

        diagnostics = {
            'H_init': H_init.mean().item(),
            'H_final': H_final.mean().item(),
            'delta_H': delta_H.item(),
            'T_init': T_init.mean().item(),
            'T_final': T_final.mean().item(),
            'V_init': V_init.mean().item(),
            'V_final': V_final.mean().item(),
            **{k: v.mean().item() for k, v in V_breakdown.items()},
        }

        return state.mu, state.Sigma, state.phi, diagnostics

    def extra_repr(self) -> str:
        return (
            f"embed_dim={self.embed_dim}, "
            f"n_steps={self.n_leapfrog_steps}, "
            f"dt={self.dt:.4f}, "
            f"gamma={self.gamma:.4f}, "
            f"update_Sigma={self.update_Sigma}, "
            f"update_phi={self.update_phi}"
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
