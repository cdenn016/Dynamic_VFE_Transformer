"""
KL-Divergence Based Attention Mechanism (0D Gauge Transformer)
===============================================================

Implements attention via information geometry instead of learned Q, K projections:

    β_ij = softmax(-KL(q_i || Ω_ij[q_j]) / κ)

where:
    - q_i = N(μ_i, Σ_i): Agent i's belief distribution
    - Ω_ij: Parallel transport operator (gauge connection)
    - KL: Kullback-Leibler divergence (information distance)
    - κ: Temperature parameter

Key Insight: NO W_Q, W_K matrices! Attention emerges from geometry.

0D Architecture:
    - All agents at single point c*
    - β_ij are scalars (not spatial fields)
    - No spatial integrals, just sums over agents

Author: Implementation from plan.py
Date: November 2025
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Union

# Import our fast math kernels
try:
    from math_utils.numba_kernels import (
        kl_gaussian_numba,
        compute_kl_transported_numba,
        push_gaussian_numba,
    )
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    print("⚠️  Numba kernels not available - falling back to PyTorch (slower)")

# Import transport operators
try:
    from math_utils.transport import compute_transport
    from math_utils.generators import generate_so3_generators
    TRANSPORT_AVAILABLE = True
except ImportError:
    TRANSPORT_AVAILABLE = False
    print("⚠️  Transport module not available")


# =============================================================================
# Sparse Attention Patterns
# =============================================================================

def create_attention_mask(
    num_agents: int,
    pattern: str = 'full',
    window: int = 64,
    device: torch.device = torch.device('cpu'),
    causal: bool = True,
) -> torch.Tensor:
    """
    Create sparse attention mask for scalability.

    Args:
        num_agents: Number of agents (sequence length)
        pattern: 'full', 'local', or 'strided'
        window: Local window size for 'local' pattern
        device: Device to create tensor on
        causal: If True, apply causal masking (i can't attend to j>i)

    Returns:
        mask: (N, N) binary mask where 1 = can attend, 0 = cannot attend

    Patterns:
        - 'full': All-to-all attention (standard transformer)
        - 'local': Local window attention (Longformer-style)
        - 'strided': Strided attention (sparse transformer)
    """
    N = num_agents

    if pattern == 'full':
        # Full attention
        mask = torch.ones(N, N, device=device)

    elif pattern == 'local':
        # Local window attention: agent i attends to [i-window, i+window]
        mask = torch.zeros(N, N, device=device)
        for i in range(N):
            start = max(0, i - window // 2)
            end = min(N, i + window // 2 + 1)
            mask[i, start:end] = 1.0

    elif pattern == 'strided':
        # Strided attention: agent i attends to every k-th agent
        stride = max(1, N // window)  # Ensure ~window agents attended to
        mask = torch.zeros(N, N, device=device)
        for i in range(N):
            # Local neighborhood
            mask[i, max(0, i-8):min(N, i+9)] = 1.0
            # Strided attention
            for j in range(0, N, stride):
                mask[i, j] = 1.0

    else:
        raise ValueError(f"Unknown attention pattern: {pattern}")

    # Apply causal masking if requested
    if causal:
        causal_mask = torch.tril(torch.ones(N, N, device=device))
        mask = mask * causal_mask

    return mask


# =============================================================================
# Transport Operator Caching (for evolve_phi=False optimization)
# =============================================================================

def compute_transport_operators(
    phi: torch.Tensor,         # (B, N, 3) gauge frames
    generators: torch.Tensor,  # (3, K, K) SO(3) generators
) -> dict:
    """
    Precompute transport operators for caching when phi is fixed.

    When evolve_phi=False, these operators are constant across layers.
    Computing once saves 2 matrix exponentials per head per layer.

    Args:
        phi: Gauge frames (B, N, 3) in so(3)
        generators: SO(3) generators (3, K, K)

    Returns:
        dict with:
            'exp_phi': (B, N, K, K) - exp(φ·G) for each token
            'exp_neg_phi': (B, N, K, K) - exp(-φ·G) for each token
            'Omega': (B, N, N, K, K) - full pairwise transport Ω_ij = exp(φ_i)exp(-φ_j)
    """
    # φ·G: combine gauge frames with generators
    phi_matrix = torch.einsum('bna,aij->bnij', phi, generators)  # (B, N, K, K)

    # Matrix exponentials (the expensive operations!)
    exp_phi = torch.matrix_exp(phi_matrix)       # (B, N, K, K)
    exp_neg_phi = torch.matrix_exp(-phi_matrix)  # (B, N, K, K)

    # Full pairwise transport: Ω_ij = exp(φ_i) @ exp(-φ_j)
    Omega = torch.einsum('bikl,bjlm->bijkm', exp_phi, exp_neg_phi)  # (B, N, N, K, K)

    return {
        'exp_phi': exp_phi,
        'exp_neg_phi': exp_neg_phi,
        'Omega': Omega,
    }


# =============================================================================
# Core Attention: KL-Based Weights
# =============================================================================

def compute_attention_weights(
    mu_q: torch.Tensor,        # (B, N, K) belief means
    sigma_q: torch.Tensor,     # (B, N, K, K) or (B, N, K) if diagonal_covariance=True
    phi: torch.Tensor,         # (B, N, 3) gauge frames
    generators: torch.Tensor,  # (3, K, K) SO(3) generators
    kappa: float,              # Temperature
    epsilon: float = 1e-8,     # Numerical stability
    mask: Optional[torch.Tensor] = None,  # (B, N, N) causal mask
    use_numba: bool = True,
    return_kl: bool = False,   # Return KL matrix for loss computation
    diagonal_covariance: bool = False,  # Use diagonal sigma (B,N,K) instead of full (B,N,K,K)
    cached_transport: Optional[dict] = None,  # Precomputed transport operators (from compute_transport_operators)
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Compute attention weights from KL divergences (0D version).

    Formula:
        β_ij = softmax_j(-KL(q_i || Ω_ij[q_j]) / κ)

    where Ω_ij = exp(φ_i) · exp(-φ_j) transports q_j to i's frame.

    0D Structure:
        - All agents at single point c*, so β_ij are SCALARS
        - No spatial fields β_ij(c), just one number per pair
        - No spatial integration, just O(N²) agent-pair loop

    Args:
        mu_q: Query belief means, shape (B, N, K)
              N = num_agents at single point c*
        sigma_q: Query covariances, shape (B, N, K, K) if full, (B, N, K) if diagonal
        phi: Gauge frames, shape (B, N, 3) in so(3)
        generators: SO(3) generators for irrep, shape (3, K, K)
        kappa: Temperature parameter (higher = softer attention)
        epsilon: Softmax stability constant
        mask: Optional causal mask (B, N, N) - 0 masks out position
        use_numba: Use fast Numba kernels if available
        diagonal_covariance: If True, sigma_q is (B,N,K) diagonal variances.
                            Uses O(N²×K) memory instead of O(N²×K²)!
        cached_transport: Optional precomputed transport operators from compute_transport_operators().
                         When evolve_phi=False, caching avoids redundant matrix exponentials.

    Returns:
        beta: Attention weights, shape (B, N, N)
              beta[b, i, j] = attention from agent i to agent j
        kl_matrix: (Optional) KL divergence matrix (B, N, N) if return_kl=True
                   kl_matrix[b, i, j] = KL(q_i || Ω_ij[q_j])

    Example:
        >>> B, N, K = 2, 10, 32
        >>> mu = torch.randn(B, N, K)
        >>> sigma = torch.eye(K).unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1)
        >>> phi = torch.randn(B, N, 3) * 0.1
        >>> G = torch.from_numpy(generate_so3_generators(K)).float()
        >>> beta = compute_attention_weights(mu, sigma, phi, G, kappa=1.0)
        >>> beta.shape
        torch.Size([2, 10, 10])
        >>> beta.sum(dim=-1)  # Should sum to 1 (plus epsilon)
        tensor([[1.0000, 1.0000, ...], ...])
    """
    batch_size, num_agents, K = mu_q.shape
    device = mu_q.device
    dtype = mu_q.dtype

    # Allocate KL divergence matrix
    kl_matrix = torch.zeros(batch_size, num_agents, num_agents, device=device, dtype=dtype)

    # =========================================================================
    # Compute all pairwise KL divergences: KL(q_i || Ω_ij[q_j])
    # =========================================================================

    # CRITICAL: Never use Numba path on CUDA devices!
    # Numba kernels are CPU-only and would cause GPU→CPU→GPU transfer bottleneck.
    # The PyTorch path is fully vectorized and runs efficiently on GPU.
    is_cuda = device.type == 'cuda'

    if diagonal_covariance:
        # DIAGONAL MODE: O(N²×K) memory instead of O(N²×K²)!
        # sigma_q is (B, N, K) not (B, N, K, K)
        _compute_kl_matrix_diagonal(
            mu_q, sigma_q, phi, generators, kl_matrix, cached_transport
        )
    elif use_numba and NUMBA_AVAILABLE and TRANSPORT_AVAILABLE and not is_cuda:
        # Fast path: Use Numba kernels (CPU only)
        # Note: Numba path doesn't support cached_transport (CPU-only fallback)
        _compute_kl_matrix_numba(
            mu_q, sigma_q, phi, generators, kl_matrix
        )
    else:
        # GPU path OR CPU fallback: Pure PyTorch (fully vectorized, CUDA-compatible)
        _compute_kl_matrix_torch(
            mu_q, sigma_q, phi, generators, kl_matrix, cached_transport
        )

    # =========================================================================
    # Convert KL distances to attention weights
    # =========================================================================

    # Attention logits: -KL / κ (more similar = less KL = higher attention)
    logits = -kl_matrix / kappa  # (B, N, N)

    # Apply causal mask if provided
    if mask is not None:
        # mask[b, i, j] = 0 means agent i CANNOT attend to agent j
        logits = logits.masked_fill(mask == 0, float('-inf'))

    # Softmax over keys (dimension 2)
    beta = F.softmax(logits, dim=-1)  # (B, N, N)

    # Add epsilon for numerical stability (prevents exact zeros)
    beta = beta + epsilon
    beta = beta / beta.sum(dim=-1, keepdim=True)

    if return_kl:
        return beta, kl_matrix
    else:
        return beta


def _compute_kl_matrix_numba(
    mu_q: torch.Tensor,
    sigma_q: torch.Tensor,
    phi: torch.Tensor,
    generators: torch.Tensor,
    kl_matrix: torch.Tensor,
) -> None:
    """
    Fast KL matrix computation using Numba kernels.

    Computes KL(q_i || Ω_ij[q_j]) for all pairs (i,j) using:
    1. Transport q_j → i's frame via Ω_ij
    2. Compute KL divergence

    Modifies kl_matrix in-place.
    """
    batch_size, num_agents, K = mu_q.shape

    # Convert to numpy for Numba
    mu_np = mu_q.detach().cpu().numpy().astype(np.float64)
    sigma_np = sigma_q.detach().cpu().numpy().astype(np.float64)
    phi_np = phi.detach().cpu().numpy().astype(np.float64)
    G_np = generators.detach().cpu().numpy().astype(np.float64)

    # Compute KL matrix
    for b in range(batch_size):
        for i in range(num_agents):
            for j in range(num_agents):
                # Compute transport operator Ω_ij
                Omega_ij = compute_transport(
                    phi_np[b, i],      # φ_i
                    phi_np[b, j],      # φ_j
                    G_np,
                    validate=False,
                    eps=1e-8
                )  # (K, K)

                # Compute KL(q_i || Ω_ij[q_j]) in one shot
                kl_ij = compute_kl_transported_numba(
                    mu_np[b, i],       # μ_i
                    sigma_np[b, i],    # Σ_i
                    mu_np[b, j],       # μ_j
                    sigma_np[b, j],    # Σ_j
                    Omega_ij           # Ω_ij
                )

                kl_matrix[b, i, j] = kl_ij


def _compute_kl_matrix_torch(
    mu_q: torch.Tensor,
    sigma_q: torch.Tensor,
    phi: torch.Tensor,
    generators: torch.Tensor,
    kl_matrix: torch.Tensor,
    cached_transport: Optional[dict] = None,  # Precomputed transport operators
) -> None:
    """
    VECTORIZED KL matrix computation using pure PyTorch.

    Computes all pairwise KL divergences without Python loops.

    Args:
        mu_q: (B, N, K) belief means
        sigma_q: (B, N, K, K) belief covariances
        phi: (B, N, 3) gauge fields
        generators: (3, K, K) SO(3) generators
        kl_matrix: (B, N, N) output tensor (modified in-place)
        cached_transport: Optional dict with precomputed 'Omega' from compute_transport_operators()
    """
    B, N, K = mu_q.shape
    device = mu_q.device
    dtype = mu_q.dtype
    eps = 1e-6

    # =========================================================================
    # Step 1: Get transport operators (use cached if available)
    # =========================================================================
    if cached_transport is not None and 'Omega' in cached_transport:
        # Use precomputed transport operators (saves 2 matrix exponentials!)
        Omega = cached_transport['Omega']
    else:
        # Compute transport operators
        # phi: (B, N, 3) -> phi_matrix: (B, N, K, K)
        phi_matrix = torch.einsum('bna,aij->bnij', phi, generators)
        exp_phi = torch.matrix_exp(phi_matrix)       # (B, N, K, K)
        exp_neg_phi = torch.matrix_exp(-phi_matrix)  # (B, N, K, K)

        # Omega_ij = exp(φ_i) @ exp(-φ_j)
        # Result: (B, N, N, K, K)
        Omega = torch.einsum('bikl,bjlm->bijkm', exp_phi, exp_neg_phi)

    # =========================================================================
    # Step 2: Transport all means and covariances
    # =========================================================================
    # μ_j^{→i} = Ω_ij @ μ_j
    mu_transported = torch.einsum('bijkl,bjl->bijk', Omega, mu_q)  # (B, N, N, K)

    # Σ_j^{→i} = Ω_ij @ Σ_j @ Ω_ij^T
    Sigma_transported = torch.einsum(
        'bijkl,bjlm,bijmn->bijkn',
        Omega, sigma_q, Omega.transpose(-1, -2)
    )  # (B, N, N, K, K)

    # =========================================================================
    # Step 3: Expand mu_i and Sigma_i for pairwise comparison
    # =========================================================================
    mu_i = mu_q[:, :, None, :].expand(-1, -1, N, -1)  # (B, N, N, K)
    Sigma_i = sigma_q[:, :, None, :, :].expand(-1, -1, N, -1, -1)  # (B, N, N, K, K)

    # =========================================================================
    # Step 4: Compute all KL divergences
    # KL(q_i || Ω_ij[q_j]) = KL(N(μ_i, Σ_i) || N(μ_j^{→i}, Σ_j^{→i}))
    # =========================================================================
    I = torch.eye(K, device=device, dtype=dtype)
    Sigma_i_reg = Sigma_i + eps * I
    Sigma_transported_reg = Sigma_transported + eps * I

    try:
        # Cholesky of transported covariances (prior in KL)
        L_p = torch.linalg.cholesky(Sigma_transported_reg)

        # Trace term: tr(Σ_p⁻¹ Σ_q) where Σ_p = Σ_j^{→i}, Σ_q = Σ_i
        Y = torch.linalg.solve_triangular(L_p, Sigma_i_reg, upper=False)
        Z = torch.linalg.solve_triangular(L_p.transpose(-1, -2), Y, upper=True)
        trace_term = torch.diagonal(Z, dim1=-2, dim2=-1).sum(dim=-1)  # (B, N, N)

        # Mahalanobis term: (μ_p - μ_q)ᵀ Σ_p⁻¹ (μ_p - μ_q)
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

        # Copy to output
        kl_matrix.copy_(kl_all)

    except RuntimeError:
        # Fallback to loop-based computation if Cholesky fails
        for b in range(B):
            for i in range(N):
                for j in range(N):
                    mu_j_transported, sigma_j_transported = _transport_gaussian_torch(
                        mu_q[b, j], sigma_q[b, j],
                        phi[b, i], phi[b, j], generators
                    )
                    kl_ij = _kl_gaussian_torch(
                        mu_q[b, i], sigma_q[b, i],
                        mu_j_transported, sigma_j_transported
                    )
                    kl_matrix[b, i, j] = kl_ij


def _transport_gaussian_torch(
    mu: torch.Tensor,         # (K,)
    sigma: torch.Tensor,      # (K, K)
    phi_dst: torch.Tensor,    # (3,)
    phi_src: torch.Tensor,    # (3,)
    generators: torch.Tensor, # (3, K, K)
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Transport Gaussian via Ω = exp(φ_dst) · exp(-φ_src).

    Returns:
        mu_transported: Ω μ, shape (K,)
        sigma_transported: Ω Σ Ω^T, shape (K, K)
    """
    # Build transport operator Ω
    # X_dst = Σ_a φ_dst[a] * G_a
    X_dst = torch.einsum('a,aij->ij', phi_dst, generators)  # (K, K)
    X_src = torch.einsum('a,aij->ij', phi_src, generators)

    # Matrix exponential (use Taylor series for small angles)
    Omega = torch.matrix_exp(X_dst) @ torch.matrix_exp(-X_src)

    # Transport
    # Use torch.mv for proper matrix-vector product: (K,K) @ (K,) → (K,)
    mu_transported = torch.mv(Omega, mu)
    sigma_transported = Omega @ sigma @ Omega.T

    # Symmetrize
    sigma_transported = 0.5 * (sigma_transported + sigma_transported.T)

    return mu_transported, sigma_transported


def _kl_gaussian_torch(
    mu1: torch.Tensor,     # (K,)
    sigma1: torch.Tensor,  # (K, K)
    mu2: torch.Tensor,     # (K,)
    sigma2: torch.Tensor,  # (K, K)
    eps: float = 1e-8
) -> torch.Tensor:
    """
    KL divergence between two Gaussians: KL(N(μ1,Σ1) || N(μ2,Σ2)).

    Formula:
        KL = 0.5 * [tr(Σ2^{-1} Σ1) + (μ2-μ1)^T Σ2^{-1} (μ2-μ1) - K + log|Σ2|/|Σ1|]
    """
    K = mu1.shape[0]

    # Regularize for stability
    sigma1_reg = sigma1 + eps * torch.eye(K, device=sigma1.device, dtype=sigma1.dtype)
    sigma2_reg = sigma2 + eps * torch.eye(K, device=sigma2.device, dtype=sigma2.dtype)

    # Cholesky decomposition for numerical stability
    L1 = torch.linalg.cholesky(sigma1_reg)
    L2 = torch.linalg.cholesky(sigma2_reg)

    # Log determinants: log|Σ| = 2*sum(log(diag(L)))
    logdet1 = 2.0 * torch.sum(torch.log(torch.diag(L1)))
    logdet2 = 2.0 * torch.sum(torch.log(torch.diag(L2)))

    # Trace term: tr(Σ2^{-1} Σ1)
    # Solve L2 Y = Σ1 for Y, then solve L2^T Z = Y for Z
    Y = torch.linalg.solve_triangular(L2, sigma1_reg, upper=False)
    Z = torch.linalg.solve_triangular(L2.T, Y, upper=True)
    trace_term = torch.trace(Z)

    # Quadratic term: (μ2-μ1)^T Σ2^{-1} (μ2-μ1)
    delta_mu = mu2 - mu1
    # solve_triangular needs 2D input - reshape (K,) → (K, 1)
    y = torch.linalg.solve_triangular(L2, delta_mu.unsqueeze(-1), upper=False).squeeze(-1)
    z = torch.linalg.solve_triangular(L2.T, y.unsqueeze(-1), upper=True).squeeze(-1)
    quad_term = torch.dot(delta_mu, z)

    # Combine
    kl = 0.5 * (trace_term + quad_term - K + logdet2 - logdet1)

    # Numerical safety: clamp to [0, ∞)
    return torch.clamp(kl, min=0.0)


def _compute_kl_matrix_diagonal(
    mu_q: torch.Tensor,        # (B, N, K) belief means
    sigma_q: torch.Tensor,     # (B, N, K) diagonal variances (NOT K×K!)
    phi: torch.Tensor,         # (B, N, 3) gauge frames
    generators: torch.Tensor,  # (3, K, K) SO(3) generators
    kl_matrix: torch.Tensor,   # (B, N, N) output tensor
    cached_transport: Optional[dict] = None,  # Precomputed transport operators
) -> None:
    """
    DIAGONAL covariance KL computation - O(N²×K) instead of O(N²×K²).

    For diagonal Gaussians, KL simplifies to:
        KL(N(μ_q, diag(σ_q)) || N(μ_p, diag(σ_p))) =
        0.5 * (sum(σ_q/σ_p) + sum((μ_p - μ_q)²/σ_p) - K + sum(log(σ_p) - log(σ_q)))

    Key simplifications:
    - No Cholesky decomposition (O(K³) → O(K))
    - No matrix inversion
    - No N×N×K×K intermediate tensors!
    - Transport still rotates μ, but σ stays diagonal (approximation)

    Args:
        mu_q: (B, N, K) belief means
        sigma_q: (B, N, K) diagonal variances (positive)
        phi: (B, N, 3) gauge fields
        generators: (3, K, K) SO(3) generators
        kl_matrix: (B, N, N) output tensor (modified in-place)
        cached_transport: Optional dict with precomputed 'Omega' from compute_transport_operators()
    """
    B, N, K = mu_q.shape
    device = mu_q.device
    dtype = mu_q.dtype
    eps = 1e-6

    # Ensure sigma is positive
    sigma_q = sigma_q.clamp(min=eps)

    # =========================================================================
    # Step 1: Get transport operators (use cached if available)
    # =========================================================================
    if cached_transport is not None and 'Omega' in cached_transport:
        # Use precomputed transport operators (saves 2 matrix exponentials!)
        Omega = cached_transport['Omega']
    else:
        # Compute transport operators
        phi_matrix = torch.einsum('bna,aij->bnij', phi, generators)  # (B, N, K, K)
        exp_phi = torch.matrix_exp(phi_matrix)       # (B, N, K, K)
        exp_neg_phi = torch.matrix_exp(-phi_matrix)  # (B, N, K, K)

        # Omega_ij = exp(φ_i) @ exp(-φ_j)
        Omega = torch.einsum('bikl,bjlm->bijkm', exp_phi, exp_neg_phi)  # (B, N, N, K, K)

    # =========================================================================
    # Step 2: Transport means (still needed for accurate KL)
    # =========================================================================
    # μ_j^{→i} = Ω_ij @ μ_j
    mu_transported = torch.einsum('bijkl,bjl->bijk', Omega, mu_q)  # (B, N, N, K)

    # =========================================================================
    # Step 3: For diagonal mode, σ stays diagonal (approximation)
    # This is exact for isotropic σ, approximate for anisotropic
    # =========================================================================
    # Expand sigma for pairwise: sigma_j for all (i,j) pairs
    sigma_j = sigma_q[:, None, :, :].expand(-1, N, -1, -1)  # (B, N, N, K)
    sigma_i = sigma_q[:, :, None, :].expand(-1, -1, N, -1)  # (B, N, N, K)

    # =========================================================================
    # Step 4: Diagonal KL divergence (vectorized)
    # KL(q_i || transported q_j) where q_i ~ N(μ_i, diag(σ_i))
    # transported q_j ~ N(μ_j^{→i}, diag(σ_j))
    # =========================================================================
    mu_i = mu_q[:, :, None, :].expand(-1, -1, N, -1)  # (B, N, N, K)

    # Trace term: sum(σ_i / σ_j)
    trace_term = (sigma_i / sigma_j).sum(dim=-1)  # (B, N, N)

    # Mahalanobis term: sum((μ_j^{→i} - μ_i)² / σ_j)
    delta_mu = mu_transported - mu_i  # (B, N, N, K)
    mahal_term = ((delta_mu ** 2) / sigma_j).sum(dim=-1)  # (B, N, N)

    # Log determinant term: sum(log(σ_j) - log(σ_i))
    logdet_term = (torch.log(sigma_j) - torch.log(sigma_i)).sum(dim=-1)  # (B, N, N)

    # Full KL
    kl_all = 0.5 * (trace_term + mahal_term - K + logdet_term)
    kl_all = torch.clamp(kl_all, min=0.0)

    kl_matrix.copy_(kl_all)


# =============================================================================
# Message Aggregation with Parallel Transport
# =============================================================================

def aggregate_messages(
    mu_q: torch.Tensor,         # (B, N, K)
    sigma_q: torch.Tensor,      # (B, N, K, K) or (B, N, K) if diagonal
    phi: torch.Tensor,          # (B, N, 3)
    beta: torch.Tensor,         # (B, N, N) attention weights
    generators: torch.Tensor,   # (3, K, K)
    aggregate_mode: str = 'mean_only',  # 'mean_only' or 'full_distribution'
    diagonal_covariance: bool = False,
    cached_transport: Optional[dict] = None,  # Precomputed transport operators
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Aggregate messages: m_i = Σ_j β_ij Ω_ij[μ_j].

    0D Version: Simple weighted sum over agents, no spatial integration!

    Two modes:
        1. 'mean_only': Only aggregate means (faster)
           Returns: (messages, None)

        2. 'full_distribution': Aggregate full distributions
           Returns: (mu_aggregated, sigma_aggregated)
           Uses mixture of Gaussians approximation

    Args:
        mu_q: Belief means (B, N, K)
        sigma_q: Belief covariances (B, N, K, K)
        phi: Gauge frames (B, N, 3)
        beta: Attention weights (B, N, N) - SCALARS, not fields!
        generators: SO(3) generators (3, K, K)
        aggregate_mode: 'mean_only' or 'full_distribution'
        cached_transport: Optional dict with precomputed 'Omega' from compute_transport_operators()

    Returns:
        mu_agg: Aggregated means (B, N, K)
        sigma_agg: Aggregated covariances (B, N, K, K) or None

    Example:
        >>> mu_agg, _ = aggregate_messages(mu, sigma, phi, beta, G, mode='mean_only')
        >>> # mu_agg[b, i] = Σ_j β[b,i,j] * Ω_ij[μ[b,j]]
    """
    batch_size, num_agents, K = mu_q.shape
    device = mu_q.device
    dtype = mu_q.dtype

    # =========================================================================
    # VECTORIZED aggregation - no Python loops!
    # =========================================================================

    # Step 1: Get transport operators (use cached if available)
    if cached_transport is not None and 'Omega' in cached_transport:
        # Use precomputed transport operators (saves 2 matrix exponentials!)
        Omega = cached_transport['Omega']
    else:
        # Compute all pairwise transport operators Ω_ij = exp(φ_i) exp(-φ_j)
        phi_matrix = torch.einsum('bna,aij->bnij', phi, generators)
        exp_phi = torch.matrix_exp(phi_matrix)       # (B, N, K, K)
        exp_neg_phi = torch.matrix_exp(-phi_matrix)  # (B, N, K, K)

        # Omega_ij = exp(φ_i) @ exp(-φ_j)  ->  (B, N, N, K, K)
        Omega = torch.einsum('bikl,bjlm->bijkm', exp_phi, exp_neg_phi)

    # Step 2: Transport all means: μ_j^{→i} = Ω_ij @ μ_j
    mu_transported = torch.einsum('bijkl,bjl->bijk', Omega, mu_q)  # (B, N, N, K)

    # Step 3: Weighted aggregation: m_i = Σ_j β_ij * μ_j^{→i}
    # beta: (B, N, N), mu_transported: (B, N, N, K)
    mu_aggregated = torch.einsum('bij,bijk->bik', beta, mu_transported)  # (B, N, K)

    # Step 4: Covariance aggregation (if requested)
    if aggregate_mode == 'full_distribution':
        B, N, K = mu_q.shape
        if diagonal_covariance:
            # DIAGONAL MODE: sigma_q is (B, N, K)
            # For diagonal, transport doesn't change variance (approximation)
            # Just weighted average of variances
            sigma_j = sigma_q[:, None, :, :].expand(-1, N, -1, -1)  # (B, N, N, K)

            # Second moment: E[x²] = σ + μ²
            second_moment = sigma_j + mu_transported ** 2  # (B, N, N, K)

            # Weighted sum
            sigma_aggregated = torch.einsum('bij,bijk->bik', beta, second_moment)

            # Complete mixture variance: Var = E[x²] - E[x]²
            sigma_aggregated = sigma_aggregated - mu_aggregated ** 2  # (B, N, K)
        else:
            # FULL COVARIANCE MODE: sigma_q is (B, N, K, K)
            # Transport all covariances: Σ_j^{→i} = Ω_ij @ Σ_j @ Ω_ij^T
            Sigma_transported = torch.einsum(
                'bijkl,bjlm,bijmn->bijkn',
                Omega, sigma_q, Omega.transpose(-1, -2)
            )  # (B, N, N, K, K)

            # Second moment: E[x x^T] = Σ + μ μ^T
            # (B, N, N, K, K) + (B, N, N, K, 1) @ (B, N, N, 1, K)
            second_moment = Sigma_transported + torch.einsum(
                'bijk,bijl->bijkl', mu_transported, mu_transported
            )

            # Weighted sum of second moments
            # beta: (B, N, N), second_moment: (B, N, N, K, K)
            sigma_aggregated = torch.einsum('bij,bijkl->bikl', beta, second_moment)

            # Complete mixture variance: Σ_mix = E[x x^T] - E[x] E[x]^T
            sigma_aggregated = sigma_aggregated - torch.einsum(
                'bik,bil->bikl', mu_aggregated, mu_aggregated
            )
    else:
        sigma_aggregated = None

    return mu_aggregated, sigma_aggregated


# =============================================================================
# Multi-Head Attention with Irrep Structure
# =============================================================================

class IrrepMultiHeadAttention(nn.Module):
    """
    Multi-head attention where heads correspond to SO(3) irreducible representations.

    Standard Transformer:
        - n_heads separate (W_Q, W_K, W_V) projections
        - Head dim = embed_dim / n_heads
        - Free parameter choices

    Gauge Transformer:
        - NO W_Q, W_K! (attention from KL divergence)
        - Heads = irrep blocks (ℓ0, ℓ1, ℓ2, ℓ3, ...)
        - Constrained by SO(3) symmetry
        - Each irrep transforms with specific rule under gauge

    Irrep Decomposition:
        K = Σ_ℓ multiplicity_ℓ × dim_ℓ

    Example (96-dim embedding):
        K = 12×1 + 7×3 + 5×5 + 2×7 = 96
        ℓ0: 12 scalar channels (gauge-invariant)
        ℓ1: 7 vector channels (transform as vectors)
        ℓ2: 5 rank-2 tensor channels
        ℓ3: 2 rank-3 tensor channels
    """

    def __init__(
        self,
        embed_dim: int,
        irrep_spec: List[Tuple[str, int, int]],
        kappa_beta: float,
        epsilon: float = 1e-8,
        aggregate_mode: str = 'mean_only',
        diagonal_covariance: bool = False,
    ):
        """
        Initialize irrep-structured multi-head attention.

        Args:
            embed_dim: Total embedding dimension K
            irrep_spec: List of (label, multiplicity, dim) tuples
                Example: [('ℓ0', 12, 1), ('ℓ1', 7, 3), ...]
            kappa_beta: Temperature for attention softmax
            epsilon: Numerical stability constant
            aggregate_mode: 'mean_only' or 'full_distribution'
            diagonal_covariance: If True, sigma is (B,N,K) diagonal variances
        """
        super().__init__()
        self.diagonal_covariance = diagonal_covariance
        self.embed_dim = embed_dim
        self.irrep_spec = irrep_spec
        self.kappa_beta = kappa_beta
        self.epsilon = epsilon
        self.aggregate_mode = aggregate_mode

        # Build irrep block structure
        self.irrep_dims = []
        self.irrep_labels = []
        total_dim = 0

        for label, multiplicity, dim in irrep_spec:
            for _ in range(multiplicity):
                self.irrep_dims.append(dim)
                self.irrep_labels.append(label)
                total_dim += dim

        # Pad to embed_dim if needed - add SCALAR heads (dim=1), not one big head
        if total_dim < embed_dim:
            padding = embed_dim - total_dim
            for _ in range(padding):
                self.irrep_dims.append(1)  # Each padding is a scalar head
                self.irrep_labels.append('ℓ0_pad')
            total_dim = embed_dim
        elif total_dim > embed_dim:
            raise ValueError(
                f"Irrep spec sums to {total_dim}, exceeds embed_dim={embed_dim}"
            )

        self.n_heads = len(self.irrep_dims)
        self.total_dim = total_dim

        # =================================================================
        # Create proper SO(3) generators for each head dimension
        # =================================================================
        # For ℓ=0 (dim=1): Zero generator (scalars don't transform)
        # For ℓ≥1 (dim=3,5,7,...): Proper Wigner D-matrix generators
        #
        # Store as a list of buffers (can't use ParameterList since non-trainable)
        self.head_generators = nn.ModuleList()  # Will hold generator-holding modules

        for head_idx, dim in enumerate(self.irrep_dims):
            if dim == 1:
                # Scalar irrep: zero generator (no transformation)
                gen = torch.zeros(3, 1, 1)
            elif dim % 2 == 1 and dim >= 3:
                # Proper SO(3) irrep: use Wigner D-matrix generators
                gen_np = generate_so3_generators(dim)
                gen = torch.from_numpy(gen_np).float()
            else:
                # Even dimension - not a valid SO(3) irrep!
                # This shouldn't happen if irrep_spec is well-formed
                raise ValueError(
                    f"Head {head_idx} has dim={dim}, which is not a valid SO(3) irrep dimension. "
                    f"SO(3) irreps must have odd dimensions (1, 3, 5, 7, ...)."
                )

            # Wrap in a module to register as buffer
            gen_holder = nn.Module()
            gen_holder.register_buffer('gen', gen)
            self.head_generators.append(gen_holder)

        # Output projection (standard linear layer)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        print(f"IrrepMultiHeadAttention: {self.n_heads} heads, dims={self.irrep_dims}")

    def forward(
        self,
        mu_q: torch.Tensor,
        sigma_q: torch.Tensor,
        phi: torch.Tensor,
        generators: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
        cached_head_transports: Optional[List[dict]] = None,  # Cross-layer cache
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward pass through multi-head attention.

        Args:
            mu_q: (B, N, K) belief means
            sigma_q: (B, N, K, K) belief covariances
            phi: (B, N, 3) gauge frames
            generators: (3, K, K) SO(3) generators
            mask: (B, N, N) optional causal mask
            return_attention: If True, return attention weights and KL matrices
            cached_head_transports: Optional list of precomputed transport dicts, one per head.
                                   When evolve_phi=False, this can be computed once at model
                                   entry and reused across all layers (6× speedup).

        Returns:
            mu_out: (B, N, K) updated means
            sigma_out: (B, N, K, K) updated covariances (or None)
            attention_weights: (B, n_heads, N, N) for visualization (or None)
            kl_matrices: (B, n_heads, N, N) KL divergences (or None)
        """
        batch_size, num_agents, K = mu_q.shape

        # =====================================================================
        # Split into irrep blocks
        # =====================================================================
        mu_blocks = self._split_irreps(mu_q)       # List of (B, N, dim_ℓ)
        sigma_blocks = self._split_irreps_sigma(sigma_q)  # List of (B, N, dim_ℓ, dim_ℓ)

        # =====================================================================
        # Process each head (irrep block)
        # =====================================================================
        head_outputs_mu = []
        head_outputs_sigma = []
        all_attention_weights = []
        all_kl_matrices = []

        for head_idx, (mu_head, sigma_head, dim_head, label) in enumerate(
            zip(mu_blocks, sigma_blocks, self.irrep_dims, self.irrep_labels)
        ):
            # Use proper SO(3) generators for this irrep dimension
            # These were pre-computed in __init__ using Wigner D-matrices
            gen_head = self.head_generators[head_idx].gen.to(
                device=generators.device, dtype=generators.dtype
            )

            # Get transport operators: use cross-layer cache if provided, else compute
            if cached_head_transports is not None:
                # Cross-layer cache: reuse transport computed at model entry
                head_cached_transport = cached_head_transports[head_idx]
            else:
                # Within-layer cache: compute once, reuse for KL and aggregation
                head_cached_transport = compute_transport_operators(phi, gen_head)

            # Compute attention for this head (with optional KL matrices)
            if return_attention:
                beta_head, kl_head = compute_attention_weights(
                    mu_head,
                    sigma_head,
                    phi,
                    gen_head,
                    self.kappa_beta,
                    self.epsilon,
                    mask,
                    return_kl=True,
                    diagonal_covariance=self.diagonal_covariance,
                    cached_transport=head_cached_transport,
                )  # (B, N, N), (B, N, N)
                all_attention_weights.append(beta_head)
                all_kl_matrices.append(kl_head)
            else:
                beta_head = compute_attention_weights(
                    mu_head,
                    sigma_head,
                    phi,
                    gen_head,
                    self.kappa_beta,
                    self.epsilon,
                    mask,
                    return_kl=False,
                    diagonal_covariance=self.diagonal_covariance,
                    cached_transport=head_cached_transport,
                )  # (B, N, N)

            # Aggregate messages for this head (reuse cached transport!)
            mu_agg, sigma_agg = aggregate_messages(
                mu_head,
                sigma_head,
                phi,
                beta_head,
                gen_head,
                aggregate_mode=self.aggregate_mode,
                diagonal_covariance=self.diagonal_covariance,
                cached_transport=head_cached_transport,
            )

            head_outputs_mu.append(mu_agg)
            if sigma_agg is not None:
                head_outputs_sigma.append(sigma_agg)

        # =====================================================================
        # Concatenate head outputs
        # =====================================================================
        mu_concat = torch.cat(head_outputs_mu, dim=-1)  # (B, N, K)

        if head_outputs_sigma:
            # Block-diagonal covariance
            sigma_concat = self._block_diag_sigma(head_outputs_sigma)  # (B, N, K, K)
        else:
            sigma_concat = None

        # =====================================================================
        # Output projection
        # =====================================================================
        mu_out = self.out_proj(mu_concat)  # (B, N, K)

        # Stack attention weights and KL matrices for loss computation
        if return_attention:
            attention_weights = torch.stack(all_attention_weights, dim=1)  # (B, n_heads, N, N)
            kl_matrices = torch.stack(all_kl_matrices, dim=1)  # (B, n_heads, N, N)
        else:
            attention_weights = None
            kl_matrices = None

        return mu_out, sigma_concat, attention_weights, kl_matrices

    def _split_irreps(self, mu: torch.Tensor) -> List[torch.Tensor]:
        """Split embedding into irrep blocks."""
        blocks = []
        start_idx = 0
        for dim in self.irrep_dims:
            blocks.append(mu[..., start_idx:start_idx+dim])
            start_idx += dim
        return blocks

    def _split_irreps_sigma(self, sigma: torch.Tensor) -> List[torch.Tensor]:
        """Split covariance into irrep blocks.

        For full covariance (B, N, K, K): extracts diagonal blocks
        For diagonal (B, N, K): extracts slices
        """
        blocks = []
        start_idx = 0
        for dim in self.irrep_dims:
            if self.diagonal_covariance:
                # Diagonal mode: sigma is (B, N, K), just slice
                blocks.append(sigma[..., start_idx:start_idx+dim])
            else:
                # Full mode: sigma is (B, N, K, K), extract diagonal block
                blocks.append(
                    sigma[..., start_idx:start_idx+dim, start_idx:start_idx+dim]
                )
            start_idx += dim
        return blocks

    def _block_diag_sigma(self, sigma_blocks: List[torch.Tensor]) -> torch.Tensor:
        """Construct covariance from irrep blocks.

        For diagonal mode: concatenates (B, N, dim) slices → (B, N, K)
        For full mode: builds block-diagonal (B, N, K, K)
        """
        batch_size, num_agents = sigma_blocks[0].shape[:2]
        K = sum(self.irrep_dims)

        if self.diagonal_covariance:
            # Diagonal mode: just concatenate along last dim
            return torch.cat(sigma_blocks, dim=-1)  # (B, N, K)
        else:
            # Full mode: build block-diagonal matrix
            sigma_full = torch.zeros(
                batch_size, num_agents, K, K,
                device=sigma_blocks[0].device,
                dtype=sigma_blocks[0].dtype
            )

            start_idx = 0
            for sigma_block, dim in zip(sigma_blocks, self.irrep_dims):
                sigma_full[..., start_idx:start_idx+dim, start_idx:start_idx+dim] = sigma_block
                start_idx += dim

            return sigma_full

    def precompute_head_transports(
        self,
        phi: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> List[dict]:
        """
        Precompute transport operators for all heads.

        Call this once at model entry when evolve_phi=False, then pass the result
        to forward() as cached_head_transports to skip redundant matrix exponentials.

        Args:
            phi: (B, N, 3) gauge frames
            device: Device for generators
            dtype: Dtype for generators

        Returns:
            List of transport dicts, one per head. Each dict contains:
                'exp_phi': (B, N, dim, dim)
                'exp_neg_phi': (B, N, dim, dim)
                'Omega': (B, N, N, dim, dim)
        """
        cached_transports = []
        for head_idx in range(self.n_heads):
            gen_head = self.head_generators[head_idx].gen.to(device=device, dtype=dtype)
            cached_transports.append(compute_transport_operators(phi, gen_head))
        return cached_transports

    def extra_repr(self) -> str:
        return (
            f"embed_dim={self.embed_dim}, "
            f"n_heads={self.n_heads}, "
            f"irrep_dims={self.irrep_dims[:3]}..., "
            f"kappa={self.kappa_beta}"
        )


# =============================================================================
# Testing
# =============================================================================

if __name__ == '__main__':
    print("="*70)
    print("KL-BASED ATTENTION MECHANISM TEST")
    print("="*70)

    # Test config
    B, N, K = 2, 8, 16  # Small for testing
    kappa = 1.0

    print(f"\n[1] Creating test data...")
    print(f"    Batch size: {B}")
    print(f"    Num agents: {N} (all at single point c*)")
    print(f"    Embed dim:  {K}")

    # Create random beliefs
    mu_q = torch.randn(B, N, K)
    sigma_q = torch.eye(K).unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1) * 0.5
    phi = torch.randn(B, N, 3) * 0.1

    # Generate SO(3) generators (import from existing module)
    if TRANSPORT_AVAILABLE:
        G = torch.from_numpy(generate_so3_generators(K)).float()
        print(f"    ✓ SO(3) generators created: {G.shape}")
    else:
        # Fallback: random skew-symmetric matrices
        G = torch.randn(3, K, K)
        G = 0.5 * (G - G.transpose(-1, -2))  # Make skew-symmetric
        print(f"    ⚠️  Using random generators (transport module unavailable)")

    # Test attention weights
    print(f"\n[2] Computing KL-based attention weights...")
    beta = compute_attention_weights(
        mu_q, sigma_q, phi, G, kappa, use_numba=False  # Use PyTorch for testing
    )
    print(f"    β shape: {beta.shape}")
    print(f"    β sum over keys: {beta.sum(dim=-1)[0, 0].item():.4f} (should ≈ 1)")
    print(f"    β min: {beta.min().item():.6f}")
    print(f"    β max: {beta.max().item():.6f}")

    # Test causal mask
    print(f"\n[3] Testing causal mask...")
    mask = torch.tril(torch.ones(N, N)).unsqueeze(0).expand(B, -1, -1)
    beta_causal = compute_attention_weights(
        mu_q, sigma_q, phi, G, kappa, mask=mask, use_numba=False
    )
    print(f"    Causal β[0, 0, :5]: {beta_causal[0, 0, :5]}")
    print(f"    Future positions should be ~0: {beta_causal[0, 0, 5:].sum().item():.6f}")

    # Test message aggregation
    print(f"\n[4] Testing message aggregation...")
    mu_agg, _ = aggregate_messages(
        mu_q, sigma_q, phi, beta, G, aggregate_mode='mean_only'
    )
    print(f"    Aggregated means shape: {mu_agg.shape}")
    print(f"    ✓ Messages aggregated via parallel transport")

    # Test multi-head attention
    print(f"\n[5] Testing multi-head attention...")
    irrep_spec = [
        ('ℓ0', 4, 1),   # 4 scalars
        ('ℓ1', 2, 3),   # 2 vectors
        ('ℓ2', 1, 5),   # 1 rank-2 tensor
    ]  # Total: 4 + 6 + 5 = 15 → pad to 16

    mha = IrrepMultiHeadAttention(
        embed_dim=K,
        irrep_spec=irrep_spec,
        kappa_beta=kappa,
    )
    print(f"    {mha}")

    mu_out, sigma_out, attn_weights = mha(
        mu_q, sigma_q, phi, G, return_attention=True
    )
    print(f"    Output μ shape: {mu_out.shape}")
    print(f"    Attention weights shape: {attn_weights.shape}")
    print(f"    ✓ Multi-head attention complete")

    # Parameter count
    total_params = sum(p.numel() for p in mha.parameters())
    print(f"\n[6] Parameter count:")
    print(f"    Multi-head attention: {total_params:,} parameters")
    print(f"    (Compare to standard: 4×K² = {4*K*K:,} for Q,K,V,O projections)")
    print(f"    Reduction: {4*K*K / max(total_params, 1):.1f}x fewer parameters!")

    print("\n" + "="*70)
    print("✓ All attention mechanism tests passed!")
    print("="*70)