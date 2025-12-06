# -*- coding: utf-8 -*-
"""
Created on Fri Nov 21 22:30:22 2025

@author: chris and christine
"""

"""
Softmax Weight Visualization
=============================

Plotting functions for visualizing spatial softmax weights:
- β_ij(c): Belief alignment softmax weights
- γ_ij(c): Prior alignment softmax weights

These weights show how agent i weights information from neighbor j
across the spatial manifold.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from analysis.plots import get_spatial_shape_from_system


def _pick_reference_agent(system):
    """Pick a reference agent i that actually has neighbors."""
    for i in range(system.n_agents):
        neighbors = system.get_neighbors(i)
        if neighbors:
            return i
    return 0


# =============================================================================
# Softmax Weight Visualization
# =============================================================================

def plot_softmax_weights(system, out_dir: Path, agent_idx: int = None, mode: str = "belief"):
    """
    Visualize spatial softmax weights β_ij(c) or γ_ij(c).

    For visualization, we mask by agent i's support χ_i so that weights
    are zero outside the agent's domain. This does NOT affect the physics
    or training, only the plots.

    Args:
        system: MultiAgentSystem with compute_softmax_weights method
        out_dir: Directory to save plots
        agent_idx: Which agent to visualize (default: auto-select one with neighbors)
        mode: Either "belief" (β) or "prior" (γ)
    """
    if system is None:
        return

    if agent_idx is None:
        agent_idx = _pick_reference_agent(system)

    shape = get_spatial_shape_from_system(system)
    ndim = len(shape)

    neighbors = system.get_neighbors(agent_idx)
    if not neighbors:
        return

    # Compute raw β/γ fields (defined on full base grid)
    weights = system.compute_softmax_weights(agent_idx, mode=mode)
    if not weights:
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    # Only keep neighbors for which we have weights
    js = sorted([j for j in neighbors if j in weights])
    if not js:
        return

    # Agent i's continuous support mask χ_i(c), shape (*S,)
    agent_i = system.agents[agent_idx]
    chi_i = np.asarray(agent_i.support.mask_continuous, dtype=float)

    if ndim == 1:
        # 1D base manifold
        x = np.arange(shape[0])

        plt.figure(figsize=(8, 4))
        for j in js:
            wj = np.asarray(weights[j], dtype=float)  # shape (S,)
            # Mask for visualization: zero outside agent i
            wj_vis = wj * chi_i
            plt.plot(x, wj_vis, label=f"j={j}", alpha=0.9)

        plt.xlabel("Position")
        plt.ylabel(f"{'β' if mode == 'belief' else 'γ'}_ij(x)")
        plt.title(f"{mode.capitalize()} softmax weights for agent i={agent_idx}")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()

        fname = f"{'beta' if mode == 'belief' else 'gamma'}_agent{agent_idx}_{mode}_1d.png"
        path = out_dir / fname
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"✓ Saved {path}")

    elif ndim == 2:
        # 2D base manifold
        H, W = shape

        # Reshape χ_i to 2D
        chi_i_2d = chi_i.reshape(H, W)

        n_nb = len(js)
        cols = min(3, n_nb)
        rows = int(np.ceil(n_nb / cols))
        plt.figure(figsize=(4 * cols, 4 * rows))

        for k, j in enumerate(js):
            ax = plt.subplot(rows, cols, k + 1)

            field = np.asarray(weights[j], dtype=float).reshape(H, W)
            # Mask for visualization: zero outside agent i
            field_vis = field * chi_i_2d

            im = ax.imshow(field_vis, origin="lower")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_title(f"{'β' if mode == 'belief' else 'γ'}_{{{agent_idx},{j}}}(c)")
            ax.set_xticks([])
            ax.set_yticks([])

        plt.tight_layout()
        fname = f"{'beta' if mode == 'belief' else 'gamma'}_agent{agent_idx}_{mode}_2d_agents.png"
        path = out_dir / fname
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"✓ Saved {path}")

    else:
        print(f"⚠️ plot_softmax_weights: ndim={ndim} not supported yet.")