
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
from transformer.rg_metrics import (
    compute_rg_diagnostics,
    RGDiagnostics,
    RGFlowSummary,
)

# Import attention computation for gamma term
from transformer.attention import compute_attention_weights



def compute_rg_metrics_from_attention(
    attn_info: Dict,
    step: int,
    auto_cluster: bool = True,
    n_clusters: Optional[int] = None,
) -> Dict[str, float]:
    """
    Compute RG metrics from attention info returned by forward_with_attention.

    This analyzes the emergent renormalization group structure in the
    attention-belief dynamics, detecting meta-agent emergence.

    Args:
        attn_info: Dict with 'beta', 'mu', 'sigma' from forward_with_attention
        step: Current training step
        auto_cluster: Auto-detect clusters via spectral clustering
        n_clusters: Fixed number of clusters (None = auto)

    Returns:
        Dict with RG metrics for logging:
            - rg/modularity: Block structure in attention (higher = more meta-agents)
            - rg/effective_rank: Effective dimensionality (lower = concentrated)
            - rg/n_clusters: Number of detected meta-agents
            - rg/kl_within_mean: KL divergence within clusters (lower = tighter)
            - rg/kl_between_mean: KL divergence between clusters (stable = distinct)
            - rg/beta_entropy: Attention distribution entropy
    """
    beta = attn_info.get('beta')  # (B, n_heads, N, N) or (B, N, N)
    mu = attn_info.get('mu')      # (B, N, K)
    sigma = attn_info.get('sigma')  # (B, N, K) or (B, N, K, K)

    if beta is None or mu is None:
        return {}

    # Average over heads if multi-head attention
    if beta.dim() == 4:
        beta_avg = beta.mean(dim=1)  # (B, N, N)
    else:
        beta_avg = beta

    # Handle sigma - default to ones if None
    if sigma is None:
        sigma = torch.ones_like(mu)

    # Compute RG diagnostics
    try:
        diagnostics = compute_rg_diagnostics(
            mu=mu,
            sigma=sigma,
            beta=beta_avg,
            step=step,
            auto_cluster=auto_cluster,
            n_clusters=n_clusters,
        )

        # Convert to metrics dict
        rg_metrics = {
            'rg/modularity': diagnostics.modularity,
            'rg/effective_rank': diagnostics.effective_rank,
            'rg/n_clusters': diagnostics.n_clusters,
            'rg/kl_within_mean': diagnostics.kl_within_mean,
            'rg/kl_within_std': diagnostics.kl_within_std,
            'rg/kl_between_mean': diagnostics.kl_between_mean,
            'rg/kl_between_std': diagnostics.kl_between_std,
            'rg/beta_entropy': diagnostics.beta_entropy,
        }

        # Add meta-agent sizes if available
        if diagnostics.meta_agent_sizes:
            rg_metrics['rg/meta_agent_sizes'] = diagnostics.meta_agent_sizes

        return rg_metrics

    except Exception as e:
        # Return empty metrics on error (don't crash training)
        print(f"[WARNING] RG metrics computation failed: {e}")
        return {}
