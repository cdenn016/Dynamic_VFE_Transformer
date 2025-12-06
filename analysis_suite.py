# -*- coding: utf-8 -*-
"""
ENHANCED Analysis / Visualization Suite for Multi-Agent Runs
============================================================

Comprehensive analysis toolkit with all requested features:
- Separate energy component plots
- Gauge field phi(c) visualization
- Covariance field Sigma(c) analysis
- Phi norm and gradient analysis
- KL divergence spatial fields
- Complete validation reporting

Usage
-----
    python analysis_suite.py --run-dir results/playground

Author: Chris
Date: November 2025
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# Import modular analysis functions
from analysis.plots.fields import plot_phi_fields, plot_sigma_fields, plot_mu_fields
from analysis.plots.support import plot_supports, plot_overlap_matrix, compute_overlap_matrix
from analysis.plots.softmax import plot_softmax_weights

# Import mu tracking plots
from analysis.plots.mu_tracking import (
    plot_mu_norm_trajectories,
    plot_mu_component_trajectories,
    plot_norm_variance_evolution,
    plot_mu_phase_space,
    plot_mu_summary_report,
    plot_mu_gauge_orbit,
    plot_mu_gauge_orbit_projections
)

# Import data loading utilities
from analysis.core.loaders import (
    load_history,
    get_mu_tracker,
    filter_history_steps,
    filter_mu_tracker,
    normalize_history,
    load_system,
    DEFAULT_SKIP_STEPS
)


# =============================================================================
# NEW: SEPARATE Energy Component Plots
# =============================================================================

def plot_energy_components_separate(history, out_dir: Path):
    """
    Plot each energy component in its own subplot for clarity.
    
    Creates a grid showing:
    - Total energy
    - Self energy  
    - Belief alignment
    - Prior alignment
    - Observations (if present)
    - Gauge smoothness (if present)
    """
    if history is None or "step" not in history or "total" not in history:
        print("⚠ History missing 'step' or 'total' — skipping energy plots.")
        return

    steps = history["step"]
    out_dir.mkdir(parents=True, exist_ok=True)

    # Determine grid layout
    components = ['total', 'self', 'belief_align', 'prior_align']
    if 'observations' in history and np.any(np.array(history['observations']) != 0):
        components.append('observations')
    if 'gauge_smooth' in history and np.any(np.array(history['gauge_smooth']) != 0):
        components.append('gauge_smooth')
    
    n_plots = len(components)
    ncols = 2
    nrows = (n_plots + 1) // 2
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4*nrows))
    axes = axes.flatten() if n_plots > 1 else [axes]
    
    # Labels for pretty printing
    labels = {
        'total': 'Total Energy',
        'self': 'Self Energy (KL(q||p))',
        'belief_align': 'Belief Alignment',
        'prior_align': 'Prior Alignment',
        'observations': 'Observation Likelihood',
        'gauge_smooth': 'Gauge Smoothness'
    }
    
    for idx, key in enumerate(components):
        ax = axes[idx]
        if key in history and len(history[key]) == len(steps):
            data = history[key]
            ax.plot(steps, data, linewidth=2, color='C0' if key=='total' else 'C1')
            ax.set_xlabel("Step")
            ax.set_ylabel("Energy")
            ax.set_title(labels.get(key, key))
            ax.grid(alpha=0.3)
    
    # Hide unused subplots
    for idx in range(n_plots, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    path = out_dir / "energy_components_separate.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"✓ Saved {path}")
    
    # Also save combined plot for comparison
    plt.figure(figsize=(10, 6))
    plt.plot(steps, history["total"], label="Total", linewidth=2.5, color='black')
    
    for key, label in [
        ("self", "Self"),
        ("belief_align", "Belief"),
        ("prior_align", "Prior"),
        ("observations", "Obs"),
        ("gauge_smooth", "Gauge")
    ]:
        if key in history and len(history[key]) == len(steps):
            plt.plot(steps, history[key], "--", label=label, alpha=0.7)
    
    plt.xlabel("Step")
    plt.ylabel("Energy")
    plt.title("Energy Evolution (Combined)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    path = out_dir / "energy_components_combined.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"✓ Saved {path}")


from mpl_toolkits.mplot3d.art3d import Line3DCollection
# =============================================================================
# Mu Tracking and Gauge Orbit Plots
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Enhanced analysis / visualization for multi-agent runs."
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default="_results/_playground",
        help="Path to run directory (contains history.* and final_state.pkl).",
    )
    parser.add_argument(
        "--skip-initial-steps",
        type=int,
        default=DEFAULT_SKIP_STEPS,
        help=f"Skip first N steps when plotting (useful to ignore initial transients). Default: {DEFAULT_SKIP_STEPS}",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        print(f"✗ Run directory does not exist: {run_dir}")
        return

    print("\n" + "=" * 70)
    print(f"ENHANCED ANALYSIS SUITE – {run_dir}")
    print("=" * 70)
    if args.skip_initial_steps > 0:
        print(f"⏩ Skipping initial {args.skip_initial_steps} steps")

    out_dir = run_dir / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)


    # Load data
    history = load_history(run_dir)

    # Filter to skip initial transient steps if requested
    if args.skip_initial_steps > 0:
        history = filter_history_steps(history, args.skip_initial_steps)

    history_dict = normalize_history(history)   # TrainingHistory -> dict
    system = load_system(run_dir)
    # 🔥 MIGRATE TO CHOLESKY if needed
    
    
    # Generate all plots
    print("\nGenerating visualizations...")

    # ------------------------------------------------------------------
    # Energy analysis: use dict-style history
    # ------------------------------------------------------------------
    if history_dict is not None:
        plot_energy_components_separate(history_dict, out_dir)

    # Geometric structure
    #plot_overlap_matrix(system, out_dir)
    #plot_supports(system, out_dir)

    # Field analysis
    #plot_mu_fields(system, out_dir)
    #plot_phi_fields(system, out_dir)
    #plot_sigma_fields(system, out_dir)

    # ------------------------------------------------------------------
    # ✨ Mu tracking plots: use original TrainingHistory (needs mu_tracker)
    # ------------------------------------------------------------------
    if history is not None:
        mu_dir = out_dir / "mu_tracking"
        mu_dir.mkdir(exist_ok=True)

        print("\n[Mu Center Tracking]")
        # All of these expect history.mu_tracker, so pass `history`, not `history_dict`
        plot_mu_norm_trajectories(history, mu_dir)
        plot_norm_variance_evolution(history, mu_dir)
        plot_mu_summary_report(history, mu_dir)
        plot_mu_gauge_orbit(history, mu_dir)
        plot_mu_gauge_orbit_projections(history,mu_dir)
        # Components for first few agents
        for i in range(min(3, system.n_agents if system else 1)):
            plot_mu_component_trajectories(history, mu_dir, agent_idx=i)

        # Phase space projections
        plot_mu_phase_space(history, mu_dir, dims=(0, 1))
        if system and system.agents[0].config.K >= 3:
            plot_mu_phase_space(history, mu_dir, dims=(0, 2))

    # Softmax weights
    if system is not None:
        beta_dir = out_dir / "softmax"
        plot_softmax_weights(system, beta_dir, agent_idx=None, mode="belief")
        plot_softmax_weights(system, beta_dir, agent_idx=None, mode="prior")

    print("\n" + "=" * 70)
    print("✓ ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"Figures saved in: {out_dir}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()