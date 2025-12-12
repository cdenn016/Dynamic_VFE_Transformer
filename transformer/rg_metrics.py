# -*- coding: utf-8 -*-
"""
Created on Thu Dec 11 19:04:20 2025

@author: chris and christine
"""

"""
Renormalization Group Metrics for VFE Transformer
==================================================

This module provides tools for analyzing emergent renormalization group (RG)
structure in the VFE transformer's attention-belief dynamics.

The Core Insight
----------------
VFE theory has a natural self-similar structure: meta-agents satisfy the same
definition as agents. This is the defining property of RG.

Scale hierarchy:
    Scale ζ=0:  Tokens q_i = N(μ_i, Σ_i)     interact via β_{ij}
                        ↓ clustering (KL → 0 within groups)
    Scale ζ=1:  Meta-agents q_A = N(μ_A, Σ_A)  interact via β'_{AB}
                        ↓ further clustering
    Scale ζ=2:  Super-meta-agents...

Key Metrics
-----------
- Modularity Q(β): Measures block structure in attention matrix
- Effective rank: Number of effective degrees of freedom
- KL within clusters: Coherence within meta-agent groups
- KL between clusters: Distinctness between groups

Author: VFE Transformer Team
Date: December 2025
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, List, Dict, NamedTuple
from dataclasses import dataclass, field


# =============================================================================
# Data Structures for RG Analysis
# =============================================================================

@dataclass
class RGDiagnostics:
    """
    Container for renormalization group diagnostics at each VFE step.

    Attributes:
        step: VFE iteration number
        modularity: Newman-Girvan modularity Q(β)
        effective_rank: Effective dimensionality of attention matrix
        n_clusters: Number of detected clusters (meta-agents)
        cluster_labels: Token-to-cluster assignments (B, N)
        kl_within_mean: Mean KL divergence within clusters
        kl_within_std: Std of KL divergence within clusters
        kl_between_mean: Mean KL divergence between clusters
        kl_between_std: Std of KL divergence between clusters
        beta_entropy: Entropy of attention distribution
        meta_agent_sizes: Size of each detected meta-agent
    """
    step: int
    modularity: float
    effective_rank: float
    n_clusters: int
    cluster_labels: Optional[torch.Tensor] = None
    kl_within_mean: float = 0.0
    kl_within_std: float = 0.0
    kl_between_mean: float = 0.0
    kl_between_std: float = 0.0
    beta_entropy: float = 0.0
    meta_agent_sizes: Optional[List[int]] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for logging/serialization."""
        return {
            'step': self.step,
            'modularity': self.modularity,
            'effective_rank': self.effective_rank,
            'n_clusters': self.n_clusters,
            'kl_within_mean': self.kl_within_mean,
            'kl_within_std': self.kl_within_std,
            'kl_between_mean': self.kl_between_mean,
            'kl_between_std': self.kl_between_std,
            'beta_entropy': self.beta_entropy,
            'meta_agent_sizes': self.meta_agent_sizes,
        }


@dataclass
class RGFlowSummary:
    """
    Summary of RG flow across all VFE steps.

    Tracks evolution of key observables to verify RG predictions.
    """
    n_steps: int
    modularity_history: List[float] = field(default_factory=list)
    effective_rank_history: List[float] = field(default_factory=list)
    n_clusters_history: List[int] = field(default_factory=list)
    kl_within_history: List[float] = field(default_factory=list)
    kl_between_history: List[float] = field(default_factory=list)
    entropy_history: List[float] = field(default_factory=list)

    def add_step(self, diagnostics: RGDiagnostics):
        """Add diagnostics from a single step."""
        self.modularity_history.append(diagnostics.modularity)
        self.effective_rank_history.append(diagnostics.effective_rank)
        self.n_clusters_history.append(diagnostics.n_clusters)
        self.kl_within_history.append(diagnostics.kl_within_mean)
        self.kl_between_history.append(diagnostics.kl_between_mean)
        self.entropy_history.append(diagnostics.beta_entropy)

    def get_rg_trends(self) -> dict:
        """
        Compute trends in RG observables.

        Expected RG behavior:
            - modularity: increasing (block structure emerges)
            - effective_rank: decreasing (fewer DOF)
            - kl_within: decreasing (clusters tighten)
            - kl_between: stable (groups remain distinct)
        """
        def compute_trend(values):
            if len(values) < 2:
                return 0.0
            # Linear regression slope
            x = np.arange(len(values))
            y = np.array(values)
            if np.std(y) < 1e-10:
                return 0.0
            slope = np.corrcoef(x, y)[0, 1] * np.std(y) / np.std(x)
            return float(slope)

        return {
            'modularity_trend': compute_trend(self.modularity_history),
            'effective_rank_trend': compute_trend(self.effective_rank_history),
            'kl_within_trend': compute_trend(self.kl_within_history),
            'kl_between_trend': compute_trend(self.kl_between_history),
            'entropy_trend': compute_trend(self.entropy_history),
        }

    def is_rg_behavior(self, threshold: float = 0.01) -> dict:
        """
        Check if observed behavior matches RG predictions.

        Returns dict with boolean flags for each prediction.
        """
        trends = self.get_rg_trends()
        return {
            'modularity_increasing': trends['modularity_trend'] > threshold,
            'effective_rank_decreasing': trends['effective_rank_trend'] < -threshold,
            'kl_within_decreasing': trends['kl_within_trend'] < -threshold,
            'kl_between_stable': abs(trends['kl_between_trend']) < threshold * 10,
        }


# =============================================================================
# Core RG Metrics
# =============================================================================

def compute_modularity(
    beta: torch.Tensor,
    cluster_labels: Optional[torch.Tensor] = None,
    resolution: float = 1.0,
) -> float:
    """
    Compute Newman-Girvan modularity Q(β) measuring block structure.

    Modularity measures how well the attention matrix decomposes into
    communities/clusters. High modularity indicates emergent meta-agents.

    Q = (1/2m) Σ_{ij} [β_ij - γ·k_i·k_j/(2m)] δ(c_i, c_j)

    where:
        - β_ij: attention weight from i to j
        - k_i = Σ_j β_ij: degree of node i
        - m = (1/2) Σ_ij β_ij: total edge weight
        - c_i: cluster assignment of node i
        - γ: resolution parameter

    Args:
        beta: Attention matrix (B, N, N) or (N, N)
        cluster_labels: Optional cluster assignments (B, N) or (N,)
                       If None, uses spectral clustering to detect clusters
        resolution: Resolution parameter γ (higher = more clusters)

    Returns:
        modularity: Average modularity across batch
    """
    # Handle batch dimension
    if beta.dim() == 2:
        beta = beta.unsqueeze(0)

    B, N, _ = beta.shape
    device = beta.device

    # Auto-detect clusters if not provided
    if cluster_labels is None:
        cluster_labels = detect_clusters_spectral(beta, n_clusters=None)

    if cluster_labels.dim() == 1:
        cluster_labels = cluster_labels.unsqueeze(0).expand(B, -1)

    modularity_sum = 0.0

    for b in range(B):
        beta_b = beta[b]  # (N, N)
        labels_b = cluster_labels[b]  # (N,)

        # Make symmetric for undirected modularity
        beta_sym = 0.5 * (beta_b + beta_b.T)

        # Compute degree k_i = Σ_j β_ij
        k = beta_sym.sum(dim=1)  # (N,)

        # Total edge weight
        m = beta_sym.sum() / 2 + 1e-10

        # Null model: k_i * k_j / (2m)
        null_model = torch.outer(k, k) / (2 * m)  # (N, N)

        # Cluster indicator: δ(c_i, c_j)
        same_cluster = (labels_b.unsqueeze(0) == labels_b.unsqueeze(1)).float()  # (N, N)

        # Modularity matrix
        B_matrix = beta_sym - resolution * null_model  # (N, N)

        # Sum contributions from same-cluster pairs
        Q = (B_matrix * same_cluster).sum() / (2 * m)

        modularity_sum += Q.item()

    return modularity_sum / B


def compute_effective_rank(
    beta: torch.Tensor,
    eps: float = 1e-10,
) -> float:
    """
    Compute effective rank of attention matrix via spectral entropy.

    Effective rank = exp(H(p))

    where H(p) = -Σ p_i log(p_i) is the entropy of normalized singular values.

    This measures the "effective dimensionality" or number of degrees of freedom.
    Lower effective rank indicates attention is concentrating on fewer modes.

    Args:
        beta: Attention matrix (B, N, N) or (N, N)
        eps: Numerical stability

    Returns:
        effective_rank: Average effective rank across batch
    """
    if beta.dim() == 2:
        beta = beta.unsqueeze(0)

    B, N, _ = beta.shape
    eff_rank_sum = 0.0

    for b in range(B):
        # SVD of attention matrix
        try:
            U, S, Vh = torch.linalg.svd(beta[b], full_matrices=False)
        except RuntimeError:
            # Fallback for numerical issues
            eff_rank_sum += N
            continue

        # Normalize singular values to get probability distribution
        S_norm = S / (S.sum() + eps)
        S_norm = S_norm.clamp(min=eps)  # Avoid log(0)

        # Spectral entropy
        H = -(S_norm * torch.log(S_norm)).sum()

        # Effective rank = exp(H)
        eff_rank = torch.exp(H)
        eff_rank_sum += eff_rank.item()

    return eff_rank_sum / B


def compute_beta_entropy(
    beta: torch.Tensor,
    eps: float = 1e-10,
) -> float:
    """
    Compute average entropy of attention distributions.

    H_i = -Σ_j β_ij log(β_ij)

    Lower entropy means attention is more focused (sparse).
    Higher entropy means attention is more diffuse (uniform).

    Args:
        beta: Attention matrix (B, N, N) or (N, N)
        eps: Numerical stability

    Returns:
        mean_entropy: Average entropy across all positions
    """
    if beta.dim() == 2:
        beta = beta.unsqueeze(0)

    beta_safe = beta.clamp(min=eps)

    # Entropy per position: H_i = -Σ_j β_ij log(β_ij)
    H = -(beta_safe * torch.log(beta_safe)).sum(dim=-1)  # (B, N)

    return H.mean().item()


# =============================================================================
# Cluster Detection
# =============================================================================

def detect_clusters_spectral(
    beta: torch.Tensor,
    n_clusters: Optional[int] = None,
    min_clusters: int = 2,
    max_clusters: int = None,
) -> torch.Tensor:
    """
    Detect clusters (meta-agents) using spectral clustering on attention matrix.

    Uses the Fiedler eigenvector method for graph partitioning.

    Args:
        beta: Attention matrix (B, N, N) or (N, N)
        n_clusters: Number of clusters (auto-detect if None)
        min_clusters: Minimum clusters to consider
        max_clusters: Maximum clusters (default: N//2)

    Returns:
        cluster_labels: Cluster assignments (B, N) or (N,)
    """
    squeeze_output = False
    if beta.dim() == 2:
        beta = beta.unsqueeze(0)
        squeeze_output = True

    B, N, _ = beta.shape
    device = beta.device

    if max_clusters is None:
        max_clusters = max(min_clusters + 1, N // 2)

    all_labels = []

    for b in range(B):
        beta_b = beta[b]

        # Symmetrize attention for undirected graph
        W = 0.5 * (beta_b + beta_b.T)

        # Degree matrix
        D = torch.diag(W.sum(dim=1))

        # Laplacian: L = D - W
        L = D - W

        # Normalized Laplacian: L_norm = D^{-1/2} L D^{-1/2}
        D_inv_sqrt = torch.diag(1.0 / (torch.sqrt(D.diag()) + 1e-10))
        L_norm = D_inv_sqrt @ L @ D_inv_sqrt

        # Eigendecomposition
        try:
            eigenvalues, eigenvectors = torch.linalg.eigh(L_norm)
        except RuntimeError:
            # Fallback: all same cluster
            all_labels.append(torch.zeros(N, dtype=torch.long, device=device))
            continue

        # Auto-detect number of clusters using eigengap
        if n_clusters is None:
            # Find largest gap in eigenvalues
            eigengaps = eigenvalues[1:max_clusters] - eigenvalues[:max_clusters-1]
            if len(eigengaps) > 0:
                n_clusters_detected = eigengaps.argmax().item() + 2  # +2 because gap at index i means (i+2) clusters
                n_clusters_detected = max(min_clusters, min(n_clusters_detected, max_clusters))
            else:
                n_clusters_detected = min_clusters
        else:
            n_clusters_detected = n_clusters

        # Use first k eigenvectors for k-means
        features = eigenvectors[:, :n_clusters_detected]  # (N, k)

        # Normalize rows
        features = F.normalize(features, p=2, dim=1)

        # K-means clustering (simple iterative)
        labels = _kmeans_simple(features, n_clusters_detected)

        all_labels.append(labels)

    result = torch.stack(all_labels, dim=0)  # (B, N)

    if squeeze_output:
        result = result.squeeze(0)

    return result


def _kmeans_simple(
    X: torch.Tensor,  # (N, K) features
    n_clusters: int,
    max_iters: int = 50,
) -> torch.Tensor:
    """
    Simple k-means clustering on GPU.

    Args:
        X: Feature matrix (N, K)
        n_clusters: Number of clusters
        max_iters: Maximum iterations

    Returns:
        labels: Cluster assignments (N,)
    """
    N, K = X.shape
    device = X.device

    # Initialize centroids using k-means++ style
    centroids = [X[torch.randint(N, (1,)).item()]]

    for _ in range(n_clusters - 1):
        # Distance to nearest centroid
        dists = torch.stack([
            ((X - c.unsqueeze(0)) ** 2).sum(dim=1)
            for c in centroids
        ], dim=1).min(dim=1)[0]

        # Sample proportional to squared distance
        probs = dists / (dists.sum() + 1e-10)
        idx = torch.multinomial(probs, 1).item()
        centroids.append(X[idx])

    centroids = torch.stack(centroids, dim=0)  # (n_clusters, K)

    # Iterate
    for _ in range(max_iters):
        # Assign to nearest centroid
        dists = ((X.unsqueeze(1) - centroids.unsqueeze(0)) ** 2).sum(dim=2)  # (N, n_clusters)
        labels = dists.argmin(dim=1)  # (N,)

        # Update centroids
        new_centroids = []
        for c in range(n_clusters):
            mask = (labels == c)
            if mask.sum() > 0:
                new_centroids.append(X[mask].mean(dim=0))
            else:
                new_centroids.append(centroids[c])

        new_centroids = torch.stack(new_centroids, dim=0)

        # Check convergence
        if torch.allclose(centroids, new_centroids, atol=1e-6):
            break

        centroids = new_centroids

    return labels


def detect_clusters_kl(
    mu: torch.Tensor,      # (B, N, K) belief means
    sigma: torch.Tensor,   # (B, N, K) diagonal or (B, N, K, K) full
    threshold: float = 1.0,
) -> torch.Tensor:
    """
    Detect clusters based on KL divergence between beliefs.

    Tokens with KL < threshold are assigned to same cluster.
    Uses agglomerative clustering with KL as distance metric.

    Args:
        mu: Belief means (B, N, K)
        sigma: Belief covariances
        threshold: KL threshold for same-cluster assignment

    Returns:
        cluster_labels: (B, N) cluster assignments
    """
    if mu.dim() == 2:
        mu = mu.unsqueeze(0)
        sigma = sigma.unsqueeze(0)

    B, N, K = mu.shape
    device = mu.device
    is_diagonal = sigma.dim() == 3

    all_labels = []

    for b in range(B):
        # Compute pairwise KL matrix
        kl_matrix = _compute_pairwise_kl(mu[b], sigma[b], is_diagonal)  # (N, N)

        # Agglomerative clustering
        labels = _agglomerative_kl(kl_matrix, threshold)
        all_labels.append(labels)

    return torch.stack(all_labels, dim=0)


def _compute_pairwise_kl(
    mu: torch.Tensor,     # (N, K)
    sigma: torch.Tensor,  # (N, K) or (N, K, K)
    is_diagonal: bool,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute pairwise KL divergence matrix."""
    N, K = mu.shape
    device = mu.device

    if is_diagonal:
        # Diagonal case: KL(p||q) = 0.5 * sum(σ_q/σ_p + (μ_p - μ_q)²/σ_p - 1 + log(σ_p/σ_q))
        sigma_safe = sigma.clamp(min=eps)  # (N, K)

        # Expand for pairwise computation
        mu_i = mu.unsqueeze(1)  # (N, 1, K)
        mu_j = mu.unsqueeze(0)  # (1, N, K)
        sigma_i = sigma_safe.unsqueeze(1)  # (N, 1, K)
        sigma_j = sigma_safe.unsqueeze(0)  # (1, N, K)

        # KL(q_i || q_j)
        kl = 0.5 * (
            (sigma_i / sigma_j).sum(dim=-1)
            + ((mu_j - mu_i) ** 2 / sigma_j).sum(dim=-1)
            - K
            + (torch.log(sigma_j) - torch.log(sigma_i)).sum(dim=-1)
        )  # (N, N)

        # Symmetrize
        kl = 0.5 * (kl + kl.T)

    else:
        # Full covariance case
        kl = torch.zeros(N, N, device=device)

        for i in range(N):
            for j in range(N):
                if i == j:
                    continue

                mu_i, Sigma_i = mu[i], sigma[i] + eps * torch.eye(K, device=device)
                mu_j, Sigma_j = mu[j], sigma[j] + eps * torch.eye(K, device=device)

                Sigma_j_inv = torch.linalg.inv(Sigma_j)

                kl_ij = 0.5 * (
                    torch.trace(Sigma_j_inv @ Sigma_i)
                    + (mu_j - mu_i) @ Sigma_j_inv @ (mu_j - mu_i)
                    - K
                    + torch.logdet(Sigma_j) - torch.logdet(Sigma_i)
                )

                kl[i, j] = kl_ij

        kl = 0.5 * (kl + kl.T)

    return kl


def _agglomerative_kl(
    kl_matrix: torch.Tensor,  # (N, N) symmetric KL distances
    threshold: float,
) -> torch.Tensor:
    """
    Agglomerative clustering using KL distance.

    Simple single-linkage: merge clusters if min KL < threshold.
    """
    N = kl_matrix.shape[0]
    device = kl_matrix.device

    # Initialize: each point in own cluster
    labels = torch.arange(N, device=device)

    # Create distance matrix (use max for self-loops)
    dist = kl_matrix.clone()
    dist.fill_diagonal_(float('inf'))

    while True:
        # Find minimum distance
        min_val = dist.min()
        if min_val > threshold:
            break

        # Find pair with minimum distance
        idx = (dist == min_val).nonzero(as_tuple=True)
        if len(idx[0]) == 0:
            break

        i, j = idx[0][0].item(), idx[1][0].item()

        # Merge: relabel j to i
        labels[labels == labels[j]] = labels[i]

        # Update distances (single linkage: min)
        dist[i] = torch.minimum(dist[i], dist[j])
        dist[:, i] = torch.minimum(dist[:, i], dist[:, j])
        dist[j] = float('inf')
        dist[:, j] = float('inf')
        dist[i, i] = float('inf')

    # Relabel to consecutive integers
    unique_labels = labels.unique()
    for new_idx, old_label in enumerate(unique_labels):
        labels[labels == old_label] = new_idx

    return labels


# =============================================================================
# KL Within/Between Clusters
# =============================================================================

def compute_kl_within_clusters(
    mu: torch.Tensor,           # (B, N, K) belief means
    sigma: torch.Tensor,        # (B, N, K) or (B, N, K, K)
    cluster_labels: torch.Tensor,  # (B, N)
    eps: float = 1e-6,
) -> Tuple[float, float]:
    """
    Compute mean and std of KL divergence within clusters.

    Lower values indicate tighter clusters (more coherent meta-agents).

    Args:
        mu: Belief means
        sigma: Belief covariances
        cluster_labels: Cluster assignments

    Returns:
        mean_kl: Mean KL divergence within clusters
        std_kl: Standard deviation
    """
    if mu.dim() == 2:
        mu = mu.unsqueeze(0)
        sigma = sigma.unsqueeze(0)
        cluster_labels = cluster_labels.unsqueeze(0)

    B, N, K = mu.shape
    is_diagonal = sigma.dim() == 3

    all_kl = []

    for b in range(B):
        # Get unique clusters
        unique_clusters = cluster_labels[b].unique()

        for c in unique_clusters:
            # Get indices in this cluster
            mask = cluster_labels[b] == c
            indices = mask.nonzero(as_tuple=True)[0]

            if len(indices) < 2:
                continue

            # Compute pairwise KL within cluster
            mu_cluster = mu[b, indices]  # (n_c, K)
            sigma_cluster = sigma[b, indices]  # (n_c, K) or (n_c, K, K)

            kl_matrix = _compute_pairwise_kl(mu_cluster, sigma_cluster, is_diagonal, eps)

            # Extract upper triangle (avoid double counting and diagonal)
            n_c = len(indices)
            triu_indices = torch.triu_indices(n_c, n_c, offset=1)
            kl_values = kl_matrix[triu_indices[0], triu_indices[1]]

            all_kl.extend(kl_values.tolist())

    if len(all_kl) == 0:
        return 0.0, 0.0

    all_kl = np.array(all_kl)
    return float(all_kl.mean()), float(all_kl.std())


def compute_kl_between_clusters(
    mu: torch.Tensor,           # (B, N, K)
    sigma: torch.Tensor,        # (B, N, K) or (B, N, K, K)
    cluster_labels: torch.Tensor,  # (B, N)
    eps: float = 1e-6,
) -> Tuple[float, float]:
    """
    Compute mean and std of KL divergence between cluster centroids.

    Stable values indicate clusters remain distinct (good RG behavior).

    Args:
        mu: Belief means
        sigma: Belief covariances
        cluster_labels: Cluster assignments

    Returns:
        mean_kl: Mean KL between cluster centroids
        std_kl: Standard deviation
    """
    if mu.dim() == 2:
        mu = mu.unsqueeze(0)
        sigma = sigma.unsqueeze(0)
        cluster_labels = cluster_labels.unsqueeze(0)

    B, N, K = mu.shape
    is_diagonal = sigma.dim() == 3
    device = mu.device

    all_kl = []

    for b in range(B):
        unique_clusters = cluster_labels[b].unique()
        n_clusters = len(unique_clusters)

        if n_clusters < 2:
            continue

        # Compute cluster centroids (meta-agent beliefs)
        centroids_mu = []
        centroids_sigma = []

        for c in unique_clusters:
            mask = cluster_labels[b] == c

            # Centroid mean = average of member means
            mu_c = mu[b, mask].mean(dim=0)  # (K,)
            centroids_mu.append(mu_c)

            # Centroid variance = average of member variances
            # (simple approximation; full version would account for variance of means)
            if is_diagonal:
                sigma_c = sigma[b, mask].mean(dim=0)  # (K,)
            else:
                sigma_c = sigma[b, mask].mean(dim=0)  # (K, K)
            centroids_sigma.append(sigma_c)

        centroids_mu = torch.stack(centroids_mu, dim=0)  # (n_clusters, K)
        centroids_sigma = torch.stack(centroids_sigma, dim=0)  # (n_clusters, K) or (n_clusters, K, K)

        # Compute pairwise KL between centroids
        kl_matrix = _compute_pairwise_kl(centroids_mu, centroids_sigma, is_diagonal, eps)

        # Extract upper triangle
        triu_indices = torch.triu_indices(n_clusters, n_clusters, offset=1)
        kl_values = kl_matrix[triu_indices[0], triu_indices[1]]

        all_kl.extend(kl_values.tolist())

    if len(all_kl) == 0:
        return 0.0, 0.0

    all_kl = np.array(all_kl)
    return float(all_kl.mean()), float(all_kl.std())


# =============================================================================
# Meta-Agent Construction
# =============================================================================

def construct_meta_agents(
    mu: torch.Tensor,           # (B, N, K) token beliefs
    sigma: torch.Tensor,        # (B, N, K) or (B, N, K, K)
    beta: torch.Tensor,         # (B, N, N) attention matrix
    cluster_labels: torch.Tensor,  # (B, N) cluster assignments
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Construct meta-agent beliefs from token beliefs using coherence-weighted average.

    Meta-agent coarse-graining map (emergent, not imposed):
        q_A = Σ_{i∈A} w_i · q_i

    where w_i is proportional to attention coherence within cluster.

    This implements the RG transformation at the belief level.

    Args:
        mu: Token belief means (B, N, K)
        sigma: Token belief covariances
        beta: Attention matrix
        cluster_labels: Cluster assignments

    Returns:
        mu_meta: Meta-agent means (B, n_clusters, K)
        sigma_meta: Meta-agent covariances
        beta_meta: Meta-agent attention matrix (B, n_clusters, n_clusters)
    """
    if mu.dim() == 2:
        mu = mu.unsqueeze(0)
        sigma = sigma.unsqueeze(0)
        beta = beta.unsqueeze(0)
        cluster_labels = cluster_labels.unsqueeze(0)

    B, N, K = mu.shape
    is_diagonal = sigma.dim() == 3
    device = mu.device

    # Get number of clusters
    max_clusters = cluster_labels.max().item() + 1

    # Initialize meta-agent tensors
    mu_meta = torch.zeros(B, max_clusters, K, device=device)
    if is_diagonal:
        sigma_meta = torch.zeros(B, max_clusters, K, device=device)
    else:
        sigma_meta = torch.zeros(B, max_clusters, K, K, device=device)
    beta_meta = torch.zeros(B, max_clusters, max_clusters, device=device)

    for b in range(B):
        for c in range(max_clusters):
            mask = cluster_labels[b] == c
            if mask.sum() == 0:
                continue

            indices = mask.nonzero(as_tuple=True)[0]
            n_c = len(indices)

            # Compute coherence weights: w_i ∝ Σ_j∈A β_ij (total attention to cluster)
            beta_within = beta[b, indices][:, indices]  # (n_c, n_c)
            w = beta_within.sum(dim=1)  # (n_c,)
            w = w / (w.sum() + 1e-10)  # Normalize

            # Weighted mean
            mu_meta[b, c] = (w.unsqueeze(-1) * mu[b, indices]).sum(dim=0)

            # Weighted covariance (simple average for now)
            if is_diagonal:
                sigma_meta[b, c] = (w.unsqueeze(-1) * sigma[b, indices]).sum(dim=0)
            else:
                sigma_meta[b, c] = (w.unsqueeze(-1).unsqueeze(-1) * sigma[b, indices]).sum(dim=0)

        # Construct meta-attention: β'_AB = Σ_{i∈A, j∈B} β_ij / |A||B|
        for c1 in range(max_clusters):
            for c2 in range(max_clusters):
                mask1 = cluster_labels[b] == c1
                mask2 = cluster_labels[b] == c2

                if mask1.sum() == 0 or mask2.sum() == 0:
                    continue

                beta_block = beta[b, mask1][:, mask2]
                beta_meta[b, c1, c2] = beta_block.mean()

        # Normalize meta-attention
        beta_meta[b] = beta_meta[b] / (beta_meta[b].sum(dim=-1, keepdim=True) + 1e-10)

    return mu_meta, sigma_meta, beta_meta


# =============================================================================
# Full RG Analysis
# =============================================================================

def compute_rg_diagnostics(
    mu: torch.Tensor,           # (B, N, K)
    sigma: torch.Tensor,        # (B, N, K) or (B, N, K, K)
    beta: torch.Tensor,         # (B, N, N)
    step: int,
    auto_cluster: bool = True,
    n_clusters: Optional[int] = None,
) -> RGDiagnostics:
    """
    Compute full RG diagnostics for a single VFE step.

    Args:
        mu: Belief means
        sigma: Belief covariances
        beta: Attention matrix
        step: VFE iteration number
        auto_cluster: Auto-detect clusters if True
        n_clusters: Fixed number of clusters (if not auto)

    Returns:
        RGDiagnostics with all metrics
    """
    # Detect clusters
    if auto_cluster:
        cluster_labels = detect_clusters_spectral(beta, n_clusters=n_clusters)
    else:
        cluster_labels = detect_clusters_kl(mu, sigma)

    # Compute metrics
    modularity = compute_modularity(beta, cluster_labels)
    effective_rank = compute_effective_rank(beta)
    beta_entropy = compute_beta_entropy(beta)

    # KL statistics
    kl_within_mean, kl_within_std = compute_kl_within_clusters(
        mu, sigma, cluster_labels
    )
    kl_between_mean, kl_between_std = compute_kl_between_clusters(
        mu, sigma, cluster_labels
    )

    # Cluster sizes
    unique_clusters = cluster_labels.unique()
    n_clusters_detected = len(unique_clusters)

    if cluster_labels.dim() > 1:
        cluster_labels_flat = cluster_labels[0]  # Use first batch element
    else:
        cluster_labels_flat = cluster_labels

    meta_agent_sizes = [
        (cluster_labels_flat == c).sum().item()
        for c in unique_clusters
    ]

    return RGDiagnostics(
        step=step,
        modularity=modularity,
        effective_rank=effective_rank,
        n_clusters=n_clusters_detected,
        cluster_labels=cluster_labels,
        kl_within_mean=kl_within_mean,
        kl_within_std=kl_within_std,
        kl_between_mean=kl_between_mean,
        kl_between_std=kl_between_std,
        beta_entropy=beta_entropy,
        meta_agent_sizes=meta_agent_sizes,
    )


def analyze_rg_flow(
    beta_history: List[torch.Tensor],
    mu_history: Optional[List[torch.Tensor]] = None,
    sigma_history: Optional[List[torch.Tensor]] = None,
    auto_cluster: bool = True,
) -> RGFlowSummary:
    """
    Analyze RG flow across all VFE steps.

    Args:
        beta_history: List of attention matrices from each VFE step
        mu_history: Optional list of belief means
        sigma_history: Optional list of belief covariances
        auto_cluster: Auto-detect clusters at each step

    Returns:
        RGFlowSummary with full evolution data
    """
    n_steps = len(beta_history)
    summary = RGFlowSummary(n_steps=n_steps)

    for step, beta in enumerate(beta_history):
        # Use dummy beliefs if not provided
        if mu_history is None:
            B, N, _ = beta.shape if beta.dim() == 3 else (1, beta.shape[0], beta.shape[1])
            K = 64  # Dummy dimension
            mu = torch.zeros(B, N, K, device=beta.device)
            sigma = torch.ones(B, N, K, device=beta.device)
        else:
            mu = mu_history[step]
            sigma = sigma_history[step] if sigma_history else torch.ones_like(mu)

        diagnostics = compute_rg_diagnostics(
            mu=mu,
            sigma=sigma,
            beta=beta,
            step=step,
            auto_cluster=auto_cluster,
        )

        summary.add_step(diagnostics)

    return summary


# =============================================================================
# Utilities for Visualization
# =============================================================================

def get_cluster_block_order(
    beta: torch.Tensor,
    cluster_labels: torch.Tensor,
) -> torch.Tensor:
    """
    Get permutation indices to reorder beta into block-diagonal form.

    Useful for visualizing cluster structure.

    Args:
        beta: Attention matrix (N, N) or (B, N, N)
        cluster_labels: Cluster assignments (N,) or (B, N)

    Returns:
        permutation: Indices to reorder tokens by cluster
    """
    if cluster_labels.dim() > 1:
        cluster_labels = cluster_labels[0]  # Use first batch

    # Sort by cluster label
    _, permutation = cluster_labels.sort()

    return permutation


def reorder_beta_by_clusters(
    beta: torch.Tensor,
    cluster_labels: torch.Tensor,
) -> torch.Tensor:
    """
    Reorder attention matrix to show block structure.

    Args:
        beta: Attention matrix (N, N) or (B, N, N)
        cluster_labels: Cluster assignments

    Returns:
        beta_reordered: Reordered attention matrix
    """
    perm = get_cluster_block_order(beta, cluster_labels)

    if beta.dim() == 2:
        return beta[perm][:, perm]
    else:
        return beta[:, perm][:, :, perm]


# =============================================================================
# Testing
# =============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("RG METRICS MODULE TEST")
    print("=" * 70)

    # Create synthetic attention matrix with block structure
    torch.manual_seed(42)

    N = 20  # 20 tokens
    B = 2   # batch size
    K = 32  # embedding dim

    # Create block-structured attention (4 clusters of 5 tokens each)
    beta = torch.zeros(B, N, N)
    for b in range(B):
        for i in range(4):
            start = i * 5
            end = (i + 1) * 5
            beta[b, start:end, start:end] = torch.rand(5, 5) * 0.8 + 0.2

        # Add some cross-cluster attention
        beta[b] = beta[b] + torch.rand(N, N) * 0.1

        # Normalize
        beta[b] = beta[b] / beta[b].sum(dim=-1, keepdim=True)

    print(f"\n[1] Testing modularity computation...")
    Q = compute_modularity(beta)
    print(f"    Modularity Q = {Q:.4f}")
    print(f"    (High Q indicates block structure)")

    print(f"\n[2] Testing effective rank...")
    eff_rank = compute_effective_rank(beta)
    print(f"    Effective rank = {eff_rank:.2f}")
    print(f"    (Lower = more concentrated attention)")

    print(f"\n[3] Testing cluster detection...")
    clusters = detect_clusters_spectral(beta)
    print(f"    Detected clusters: {clusters[0].unique().tolist()}")
    print(f"    Cluster sizes: {[(clusters[0] == c).sum().item() for c in clusters[0].unique()]}")

    print(f"\n[4] Testing KL within/between clusters...")
    mu = torch.randn(B, N, K)
    sigma = torch.ones(B, N, K) * 0.5

    kl_within, kl_within_std = compute_kl_within_clusters(mu, sigma, clusters)
    kl_between, kl_between_std = compute_kl_between_clusters(mu, sigma, clusters)
    print(f"    KL within clusters: {kl_within:.4f} +/- {kl_within_std:.4f}")
    print(f"    KL between clusters: {kl_between:.4f} +/- {kl_between_std:.4f}")

    print(f"\n[5] Testing meta-agent construction...")
    mu_meta, sigma_meta, beta_meta = construct_meta_agents(mu, sigma, beta, clusters)
    print(f"    Meta-agent μ shape: {mu_meta.shape}")
    print(f"    Meta-agent β shape: {beta_meta.shape}")

    print(f"\n[6] Testing full RG diagnostics...")
    diagnostics = compute_rg_diagnostics(mu, sigma, beta, step=0)
    print(f"    Diagnostics: {diagnostics.to_dict()}")

    print(f"\n[7] Testing RG flow analysis...")
    # Simulate evolution: modularity should increase
    beta_history = [beta.clone()]
    for step in range(5):
        # Make blocks more pronounced
        beta_evolved = beta_history[-1].clone()
        for b in range(B):
            for i in range(4):
                start = i * 5
                end = (i + 1) * 5
                beta_evolved[b, start:end, start:end] *= 1.1
            beta_evolved[b] = beta_evolved[b] / beta_evolved[b].sum(dim=-1, keepdim=True)
        beta_history.append(beta_evolved)

    summary = analyze_rg_flow(beta_history)
    trends = summary.get_rg_trends()
    rg_behavior = summary.is_rg_behavior()

    print(f"    Modularity history: {[f'{m:.3f}' for m in summary.modularity_history]}")
    print(f"    Modularity trend: {trends['modularity_trend']:.4f}")
    print(f"    RG behavior detected: {rg_behavior}")

    print("\n" + "=" * 70)
    print("All RG metrics tests passed!")
    print("=" * 70)