# -*- coding: utf-8 -*-
"""
Created on Fri Nov 21 22:30:43 2025

@author: chris and christine
"""

"""
Support and Overlap Visualization
==================================

Plotting functions for visualizing agent supports and their overlaps:
- Support masks: Spatial regions where agents are active
- Overlap matrix: Pairwise overlap fractions between agents
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from analysis.plots import get_spatial_shape_from_system


# =============================================================================
# Support Mask Visualization
# =============================================================================

def plot_supports(system, out_dir: Path):
    """
    Visualize agent supports in 1D or 2D.

    Shows which regions of the base manifold each agent is active in.

    Args:
        system: MultiAgentSystem with agents having support attributes
        out_dir: Directory to save plots
    """
    if system is None:
        return

    shape = get_spatial_shape_from_system(system)
    ndim = len(shape)

    if ndim == 0:
        print("Base manifold is 0D — supports are trivial. Skipping.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    masks = []
    for agent in system.agents:
        if hasattr(agent, "support") and agent.support is not None:
            masks.append(agent.support.mask.astype(float))
        else:
            masks.append(np.ones(shape, dtype=float))

    masks = np.stack(masks, axis=0)

    if ndim == 1:
        plt.figure(figsize=(8, 4))
        plt.imshow(masks, aspect="auto", origin="lower")
        plt.colorbar(label="support (1 = active)")
        plt.xlabel("position")
        plt.ylabel("agent")
        plt.yticks(range(system.n_agents))
        plt.title("Agent supports (1D)")
        plt.tight_layout()
        path = out_dir / "supports_1d.png"
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"✓ Saved {path}")

    elif ndim == 2:
        H, W = shape

        active_count = masks.sum(axis=0)
        plt.figure(figsize=(5, 4))
        im = plt.imshow(active_count, origin="lower")
        plt.colorbar(im, label="# active agents")
        plt.title("Number of active agents per location")
        plt.tight_layout()
        path = out_dir / "supports_2d_counts.png"
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"✓ Saved {path}")

        n_agents = system.n_agents
        cols = min(3, n_agents)
        rows = int(np.ceil(n_agents / cols))
        plt.figure(figsize=(4 * cols, 4 * rows))
        for i in range(n_agents):
            ax = plt.subplot(rows, cols, i + 1)
            im = ax.imshow(masks[i], origin="lower")
            ax.set_title(f"Agent {i} support")
            ax.set_xticks([])
            ax.set_yticks([])
        plt.tight_layout()
        path = out_dir / "supports_2d_agents.png"
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"✓ Saved {path}")


# =============================================================================
# Overlap Matrix Computation and Visualization
# =============================================================================

def compute_overlap_matrix(system):
    """
    Compute overlap fractions between all pairs of agents.

    Returns:
        O: (n_agents, n_agents) matrix where O[i,j] = |C_i ∩ C_j| / |C_i|
    """
    n = system.n_agents
    O = np.zeros((n, n), dtype=float)

    if not hasattr(system, "get_overlap_fraction"):
        return None

    for i in range(n):
        for j in range(n):
            if i == j:
                O[i, j] = 1.0
            else:
                try:
                    O[i, j] = system.get_overlap_fraction(i, j)
                except Exception:
                    O[i, j] = 0.0
    return O


def plot_overlap_matrix(system, out_dir: Path):
    """
    Plot heatmap of pairwise overlap fractions.

    Shows how much each agent's support overlaps with others.

    Args:
        system: MultiAgentSystem with get_overlap_fraction method
        out_dir: Directory to save plots
    """
    if system is None:
        return

    O = compute_overlap_matrix(system)
    if O is None:
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(5, 4))
    im = plt.imshow(O, vmin=0.0, vmax=1.0, origin="lower")
    plt.colorbar(im, label="|C_i ∩ C_j| / |C_i|")
    plt.xlabel("j")
    plt.ylabel("i")
    plt.title("Agent overlap matrix")
    plt.xticks(range(system.n_agents))
    plt.yticks(range(system.n_agents))
    plt.tight_layout()
    path = out_dir / "overlap_matrix.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"✓ Saved {path}")