
# Additional imports for advanced plotting
from mpl_toolkits.mplot3d.art3d import Line3DCollection
"""
Mu Center Tracking Plots
=========================

Plotting functions for visualizing mu center evolution over time.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Any


def plot_mu_summary(mu_tracker: Any, save_path: Path):
    """
    Plot summary of mu center tracking.

    Args:
        mu_tracker: MuCenterTracking object
        save_path: Path to save figure
    """
    if not hasattr(mu_tracker, 'steps') or not mu_tracker.steps:
        print("No mu tracking data available")
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Mu Center Tracking', fontsize=16, fontweight='bold')

    steps = mu_tracker.steps
    n_agents = len(mu_tracker.mu_norms) if hasattr(mu_tracker, 'mu_norms') else 0

    # Mu norms over time
    ax = axes[0, 0]
    if hasattr(mu_tracker, 'mu_norms'):
        for i, norms in enumerate(mu_tracker.mu_norms):
            if norms:
                ax.plot(steps[:len(norms)], norms, alpha=0.7, label=f'Agent {i}')
        ax.set_title('Mu Norm Evolution')
        ax.set_ylabel('||μ(center)||')
        ax.set_xlabel('Step')
        ax.grid(True, alpha=0.3)
        if n_agents < 10:
            ax.legend(fontsize=8)

    # Mean and variance of norms
    ax = axes[0, 1]
    if hasattr(mu_tracker, 'mu_norms'):
        all_norms = []
        for step_idx in range(len(steps)):
            step_norms = [norms[step_idx] if step_idx < len(norms) else 0
                         for norms in mu_tracker.mu_norms]
            all_norms.append(step_norms)

        mean_norms = [np.mean(n) for n in all_norms]
        std_norms = [np.std(n) for n in all_norms]

        ax.plot(steps, mean_norms, 'b-', linewidth=2, label='Mean')
        ax.fill_between(steps,
                        np.array(mean_norms) - np.array(std_norms),
                        np.array(mean_norms) + np.array(std_norms),
                        alpha=0.3, label='±1σ')
        ax.set_title('Mean Mu Norm (±1σ)')
        ax.set_ylabel('||μ||')
        ax.set_xlabel('Step')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Distribution at final step
    ax = axes[1, 0]
    if hasattr(mu_tracker, 'mu_norms'):
        final_norms = [norms[-1] if norms else 0 for norms in mu_tracker.mu_norms]
        ax.hist(final_norms, bins=20, alpha=0.7, edgecolor='black')
        ax.axvline(np.mean(final_norms), color='r', linestyle='--',
                  linewidth=2, label=f'Mean: {np.mean(final_norms):.3f}')
        ax.set_title('Final Mu Norm Distribution')
        ax.set_xlabel('||μ(center)||')
        ax.set_ylabel('Count')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Info text
    ax = axes[1, 1]
    ax.axis('off')
    info_text = f"""
    Mu Center Tracking Summary

    Number of agents: {n_agents}
    Training steps: {len(steps)}

    Final statistics:
    """
    if hasattr(mu_tracker, 'mu_norms'):
        final_norms = [norms[-1] if norms else 0 for norms in mu_tracker.mu_norms]
        info_text += f"""  Mean norm: {np.mean(final_norms):.4f}
      Std norm:  {np.std(final_norms):.4f}
      Min norm:  {np.min(final_norms):.4f}
      Max norm:  {np.max(final_norms):.4f}
    """
    ax.text(0.1, 0.5, info_text, fontsize=11, family='monospace',
            verticalalignment='center')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"  ✓ Saved mu tracking plot: {save_path}")



# =============================================================================
# Advanced Mu Tracking Plots (Extracted from analysis_suite.py)
# =============================================================================

def plot_mu_norm_trajectories(history, out_dir: Path):
    """
    Plot ||μ(center)|| over time for all agents.
    
    Key diagnostic:
    - Vacuum theory: All agents should have same norm (gauge orbit)
    - With observations: Norms diverge (symmetry breaking)
    """
    tracker = get_mu_tracker(history)
    if tracker is None:
        print("⚠️  No mu tracking data available")
        return
    
    out_dir.mkdir(parents=True, exist_ok=True)
    
    steps = np.array(tracker.steps)
    norms = tracker.get_all_norms()  # (n_agents, n_steps)
    
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Individual trajectories
    ax = axes[0]
    for i in range(norms.shape[0]):
        ax.plot(steps, norms[i], label=f'Agent {i}', alpha=0.7, linewidth=2)
    ax.set_xlabel('Step')
    ax.set_ylabel('||μ(center)||')
    ax.set_title('Belief Mean Norms at Agent Centers')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(alpha=0.3)
    
    # Right: Statistics
    ax = axes[1]
    mean_norm = np.mean(norms, axis=0)
    std_norm = np.std(norms, axis=0)
    
    ax.plot(steps, mean_norm, 'k-', linewidth=2.5, label='Mean')
    ax.fill_between(steps, mean_norm - std_norm, mean_norm + std_norm, 
                     alpha=0.3, color='gray', label='±1 std')
    ax.set_xlabel('Step')
    ax.set_ylabel('||μ||')
    ax.set_title('Norm Statistics Across Agents')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    path = out_dir / "mu_norm_trajectories.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved {path}")


def plot_mu_component_trajectories(history, out_dir: Path, agent_idx: int = 0):
    """
    Plot individual μ components over time for one agent.
    
    Shows how the agent moves through latent space.
    """
    tracker = get_mu_tracker(history)
    if tracker is None:
        return
    
    out_dir.mkdir(parents=True, exist_ok=True)
    
    steps = np.array(tracker.steps)
    components = tracker.get_component_array(agent_idx)  # (n_steps, K)
    K = components.shape[1]
    
    # Plot each component
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for k in range(K):
        ax.plot(steps, components[:, k], label=f'μ[{k}]', linewidth=2, alpha=0.7)
    
    ax.set_xlabel('Step')
    ax.set_ylabel('μ component value')
    ax.set_title(f'Belief Mean Components at Center - Agent {agent_idx}')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.axhline(0, color='black', linewidth=0.5, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    path = out_dir / f"mu_components_agent{agent_idx}.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"✓ Saved {path}")


def plot_norm_variance_evolution(history, out_dir: Path):
    """
    Plot variance of ||μ|| across agents over time.
    
    KEY SYMMETRY DIAGNOSTIC:
    - Var(||μ||) ≈ 0: Gauge symmetry preserved (vacuum theory)
    - Var(||μ||) > 0: Symmetry broken (observations active)
    """
    tracker = get_mu_tracker(history)
    if tracker is None:
        return
    
    out_dir.mkdir(parents=True, exist_ok=True)
    
    steps = np.array(tracker.steps)
    var_norm = tracker.compute_norm_variance()
    mean_norm = tracker.compute_mean_norm()
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    # Top: Variance (symmetry measure)
    ax = axes[0]
    ax.semilogy(steps, var_norm, 'r-', linewidth=2.5)
    ax.set_xlabel('Step')
    ax.set_ylabel('Var(||μ||)')
    ax.set_title('Norm Variance Across Agents (Symmetry Measure)')
    ax.grid(alpha=0.3)
    
    # Add symmetry interpretation
    if len(var_norm) > 0:
        final_var = var_norm[-1]
        if final_var < 1e-3:
            sym_text = "Gauge symmetry preserved"
            color = 'green'
        elif final_var < 0.1:
            sym_text = "Weak symmetry breaking"
            color = 'orange'
        else:
            sym_text = "Strong symmetry breaking"
            color = 'red'
        
        ax.text(0.95, 0.95, sym_text, 
               transform=ax.transAxes,
               ha='right', va='top',
               bbox=dict(boxstyle='round', facecolor=color, alpha=0.3),
               fontsize=12, fontweight='bold')
    
    # Bottom: Mean norm
    ax = axes[1]
    ax.plot(steps, mean_norm, 'b-', linewidth=2.5)
    ax.set_xlabel('Step')
    ax.set_ylabel('Mean ||μ||')
    ax.set_title('Average Norm Across Agents')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    path = out_dir / "mu_norm_variance.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"✓ Saved {path}")


def plot_mu_phase_space(history, out_dir: Path, dims: tuple = (0, 1)):
    """
    Plot μ trajectories in 2D phase space projection.
    
    Shows how agents explore the latent space manifold.
    """
    tracker = get_mu_tracker(history)
    if tracker is None:
        return
    
    out_dir.mkdir(parents=True, exist_ok=True)
    
    dim_x, dim_y = dims
    n_agents = len(tracker.mu_components)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Plot trajectory for each agent
    for i in range(n_agents):
        components = tracker.get_component_array(i)  # (n_steps, K)
        
        if components.shape[1] <= max(dims):
            continue
        
        x = components[:, dim_x]
        y = components[:, dim_y]
        
        # Plot trajectory with color gradient
        ax.plot(x, y, alpha=0.6, linewidth=1.5, label=f'Agent {i}')
        
        # Mark start and end
        ax.scatter(x[0], y[0], marker='o', s=100, alpha=0.8, edgecolors='black')
        ax.scatter(x[-1], y[-1], marker='s', s=100, alpha=0.8, edgecolors='black')
    
    ax.set_xlabel(f'μ[{dim_x}]')
    ax.set_ylabel(f'μ[{dim_y}]')
    ax.set_title(f'Belief Mean Trajectories (dims {dim_x}, {dim_y})')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.axhline(0, color='black', linewidth=0.5, linestyle='--', alpha=0.3)
    ax.axvline(0, color='black', linewidth=0.5, linestyle='--', alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    path = out_dir / f"mu_phase_space_dims{dim_x}_{dim_y}.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"✓ Saved {path}")


def plot_mu_summary_report(history, out_dir: Path):
    """
    Generate comprehensive summary report of μ tracking.
    
    Combines all key diagnostics in one multi-panel figure.
    """
    tracker = get_mu_tracker(history)
    if tracker is None:
        return
    
    out_dir.mkdir(parents=True, exist_ok=True)
    
    steps = np.array(tracker.steps)
    norms = tracker.get_all_norms()
    var_norm = tracker.compute_norm_variance()
    mean_norm = tracker.compute_mean_norm()
    
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # (1) Norm trajectories
    ax = fig.add_subplot(gs[0, :])
    for i in range(norms.shape[0]):
        ax.plot(steps, norms[i], label=f'Agent {i}', alpha=0.7)
    ax.set_xlabel('Step')
    ax.set_ylabel('||μ||')
    ax.set_title('Norm Trajectories')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', ncol=2)
    ax.grid(alpha=0.3)
    
    # (2) Variance (symmetry)
    ax = fig.add_subplot(gs[1, 0])
    ax.semilogy(steps, var_norm, 'r-', linewidth=2.5)
    ax.set_xlabel('Step')
    ax.set_ylabel('Var(||μ||)')
    ax.set_title('Symmetry Measure')
    ax.grid(alpha=0.3)
    
    # (3) Mean norm
    ax = fig.add_subplot(gs[1, 1])
    ax.plot(steps, mean_norm, 'b-', linewidth=2.5)
    ax.set_xlabel('Step')
    ax.set_ylabel('Mean ||μ||')
    ax.set_title('Average Norm')
    ax.grid(alpha=0.3)
    
    # (4) Phase space (first 2 dims)
    ax = fig.add_subplot(gs[2, 0])
    for i in range(min(norms.shape[0], 8)):  # Max 8 agents
        components = tracker.get_component_array(i)
        if components.shape[1] >= 2:
            ax.plot(components[:, 0], components[:, 1], alpha=0.6, label=f'Agent {i}')
            ax.scatter(components[0, 0], components[0, 1], marker='o', s=50, alpha=0.8)
    ax.set_xlabel('μ[0]')
    ax.set_ylabel('μ[1]')
    ax.set_title('Phase Space (dims 0,1)')
    ax.grid(alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    
    # (5) Final statistics table
    ax = fig.add_subplot(gs[2, 1])
    ax.axis('off')
    
    # Compute statistics
    final_var = var_norm[-1] if len(var_norm) > 0 else 0
    final_mean = mean_norm[-1] if len(mean_norm) > 0 else 0
       
    stats_text = f"""
    FINAL STATISTICS
    ================
    
    Mean ||μ||:     {final_mean:.4f}
    Var(||μ||):     {final_var:.4e}
    
    Symmetry Status:
    {_get_symmetry_status(final_var)}
    
    Norm Range:
    Min: {norms[:, -1].min():.4f}
    Max: {norms[:, -1].max():.4f}
    """
    
    ax.text(0.1, 0.5, stats_text, transform=ax.transAxes,
           fontsize=11, verticalalignment='center',
           fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.suptitle('μ(center) Tracking Summary', fontsize=16, fontweight='bold')
    
    path = out_dir / "mu_tracking_summary.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved {path}")


def _get_symmetry_status(var_norm: float) -> str:
    """Classify symmetry breaking based on norm variance."""
    if var_norm < 1e-4:
        return "✓ Gauge symmetry preserved"
    elif var_norm < 1e-2:
        return "~ Weak symmetry breaking"
    else:
        return "✗ Strong symmetry breaking"



from pathlib import Path

from pathlib import Path

def plot_mu_gauge_orbit(history, out_dir: Path):
    r"""
    Visualize normalized $\mu$-trajectories as gauge orbits.

    This plot shows how each agent's latent mean vector $\mu_i(t)$ moves on
    the **unit circle** (for $K=2$) or **unit sphere** (for $K\geq 3$) after
    per-step normalization:

    \[
        \hat{\mu}_i(t)
        \;=\;
        \frac{\mu_i(t)}{\lVert \mu_i(t) \rVert_2}
        \quad\in\quad
        \mathbb{S}^{K-1}.
    \]
    """
    # -------------------------------------------------------------------------
    # Safety checks and early exits
    # -------------------------------------------------------------------------
    tracker = get_mu_tracker(history)
    if tracker is None:
        print("⚠️  No mu tracking data available — skipping gauge orbit plot.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    # Use first agent to infer latent dimension K
    components0 = tracker.get_component_array(0)  # (n_steps, K)
    if components0.size == 0:
        print("⚠️  Empty μ tracking data — skipping gauge orbit plot.")
        return

    K = components0.shape[1]

    import numpy as np
    import matplotlib.pyplot as plt

    if K < 2:
        print("⚠️  Latent dim K < 2 — no meaningful gauge orbit to plot.")
        return

    # Determine number of agents from tracker
    n_agents = (
        len(tracker.agent_indices)
        if hasattr(tracker, "agent_indices")
        else tracker.n_agents
    )

    # -------------------------------------------------------------------------
    # Case 1: 2D latent space → unit circle
    # -------------------------------------------------------------------------
    if K == 2:
        fig, ax = plt.subplots(figsize=(6, 6))

        # Reference unit circle
        theta = np.linspace(0.0, 2.0 * np.pi, 512)
        ax.plot(
            np.cos(theta),
            np.sin(theta),
            linestyle="--",
            linewidth=1.0,
            alpha=0.6,
            label="Unit circle"
        )

        # Agent trajectories
        for i in range(n_agents):
            comps = tracker.get_component_array(i)  # (n_steps, 2)

            # Normalize onto unit circle
            norms = np.linalg.norm(comps, axis=1, keepdims=True)
            norms = np.clip(norms, 1e-12, None)
            unit = comps / norms  # (n_steps, 2)

            # Continuous trajectory
            ax.plot(
                unit[:, 0],
                unit[:, 1],
                alpha=0.9,
                linewidth=1.5,
                label=f"Agent {i}"
            )

            # Start and end markers
            ax.scatter(
                unit[0, 0],
                unit[0, 1],
                s=40,
                marker="o",
                alpha=0.95
            )
            ax.scatter(
                unit[-1, 0],
                unit[-1, 1],
                s=40,
                marker="x",
                alpha=0.95
            )

        ax.set_xlabel(r"$\hat{\mu}_0$")
        ax.set_ylabel(r"$\hat{\mu}_1$")
        ax.set_title(r"Gauge Orbit on Unit Circle ($\mu$ normalized)")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right", fontsize=8, frameon=False)

        path = out_dir / "mu_gauge_orbit_circle.png"
        plt.tight_layout()
        plt.savefig(path, dpi=300)
        plt.close()
        print(f"✓ Saved {path}")



    # -------------------------------------------------------------------------
    # Case 2: K ≥ 3 → project to first 3 dims on unit sphere
    # -------------------------------------------------------------------------
    else:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection="3d")

        # ---------------------------------------------------------------------
        # 1. Camera / view direction
        # ---------------------------------------------------------------------
        elev = 25
        azim = -55
        ax.view_init(elev=elev, azim=azim)

        elev_rad = np.deg2rad(elev)
        azim_rad = np.deg2rad(azim)
        view_dir = np.array([
            np.cos(elev_rad) * np.cos(azim_rad),
            np.cos(elev_rad) * np.sin(azim_rad),
            np.sin(elev_rad),
        ], dtype=float)
        view_dir /= np.linalg.norm(view_dir)

        # ---------------------------------------------------------------------
        # 2. Opaque shaded unit sphere (no transparency → no color tint)
        # ---------------------------------------------------------------------
        u = np.linspace(0, 2 * np.pi, 80)
        v = np.linspace(0, np.pi, 60)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones_like(u), np.cos(v))

        # Simple shading via light source
        nx, ny, nz = x, y, z
        light_dir = np.array([1.0, -1.0, 1.5], dtype=float)
        light_dir /= np.linalg.norm(light_dir)

        shade = nx * light_dir[0] + ny * light_dir[1] + nz * light_dir[2]
        shade = np.clip(shade, 0.0, 1.0)
        shade = (shade - shade.min()) / (shade.max() - shade.min() + 1e-12)
        shade = 0.25 + 0.75 * shade  # ambient + directional

        base_color = np.array([0.80, 0.83, 0.95], dtype=float)
        sphere_colors = base_color[None, None, :] * shade[..., None]

        ax.plot_surface(
            x, y, z,
            facecolors=sphere_colors,
            rstride=1, cstride=1,
            linewidth=0.3,
            edgecolor=(0.7, 0.7, 0.8),
            antialiased=True,
            alpha=0.5,      # OPAQUE: no tinting of lines
            shade=False,
        )

        # ---------------------------------------------------------------------
        # 2. Agent trajectories: per-agent color, explicit front/back segments
        # ---------------------------------------------------------------------
        base_cmap = plt.get_cmap("tab10")
        agent_colors = [base_cmap(i % base_cmap.N) for i in range(n_agents)]

        path_radius = 1.015  # slightly outside the unit sphere

        for i in range(n_agents):
            comps = tracker.get_component_array(i)  # (n_steps, K)

            norms = np.linalg.norm(comps, axis=1, keepdims=True)
            norms = np.clip(norms, 1e-12, None)
            unit = comps / norms                     # on S^K-1

            # points we actually plot: just outside the sphere
            xyz = path_radius * unit[:, :3]          # (n_steps, 3)


            n_steps = xyz.shape[0]
            if n_steps < 2:
                continue

            segments = np.stack([xyz[:-1], xyz[1:]], axis=1)

            # Front vs back wrt camera
            front_point_mask = (xyz @ view_dir) > 0
            front_seg_mask = front_point_mask[:-1] & front_point_mask[1:]
            back_seg_mask = ~front_seg_mask

            color = agent_colors[i]

            # Back segments: faint, dashed
            if np.any(back_seg_mask):
                seg_back = segments[back_seg_mask]
                lc_back = Line3DCollection(
                    seg_back,
                    colors=[color],
                    linewidths=1.5,
                    alpha=0.65,
                    linestyles="solid",
                )
                ax.add_collection3d(lc_back)

            # Front segments: solid, fully on top of everything (no front wireframe)
            if np.any(front_seg_mask):
                seg_front = segments[front_seg_mask]
                lc_front = Line3DCollection(
                    seg_front,
                    colors=[color],
                    linewidths=1.5,
                    alpha=1.0,
                    linestyles="solid",
                )
                ax.add_collection3d(lc_front)

            # Markers
            ax.scatter(
                xyz[0, 0], xyz[0, 1], xyz[0, 2],
                s=75, marker="o", color=color, edgecolor="k",
                linewidths=0.8, depthshade=False,
                label=f"Agent {i}" if i == 0 else None,
            )
            ax.scatter(
                xyz[-1, 0], xyz[-1, 1], xyz[-1, 2],
                s=75, marker="X", color=color, edgecolor="k",
                linewidths=0.8, depthshade=False,
            )

        # ---------------------------------------------------------------------
        # 3. Axes, layout
        # ---------------------------------------------------------------------
        ax.set_xlabel(r"$\hat{\mu}_0$")
        ax.set_ylabel(r"$\hat{\mu}_1$")
        ax.set_zlabel(r"$\hat{\mu}_2$")
        ax.set_title(r"Gauge Orbits on Unit Sphere ($\mu$ normalized)")

        max_range = np.array([
            x.max() - x.min(),
            y.max() - y.min(),
            z.max() - z.min(),
        ]).max() / 2.0
        ax.set_xlim(-max_range, max_range)
        ax.set_ylim(-max_range, max_range)
        ax.set_zlim(-max_range, max_range)
        ax.set_box_aspect((1, 1, 1))

        ax.grid(alpha=0.1)
        ax.legend(loc="upper left", fontsize=8, frameon=False)

        path = out_dir / "mu_gauge_orbit_solid_sphere.png"
        plt.tight_layout()
        plt.savefig(path, dpi=300)
        plt.close()
        print(f"✓ Saved {path}")




def plot_mu_gauge_orbit_projections(history, out_dir: Path):
    """
    2D projections of normalized μ-trajectories: (μ0, μ1), (μ0, μ2), (μ1, μ2).

    This complements the 3D gauge-orbit visualization by showing unambiguous
    planar projections of the same normalized trajectories.
    """
    tracker = get_mu_tracker(history)
    if tracker is None:
        print("⚠️  No mu tracking data available — skipping orbit projections.")
        return

    components0 = tracker.get_component_array(0)
    if components0.size == 0:
        print("⚠️  Empty μ tracking data — skipping orbit projections.")
        return

    K = components0.shape[1]
    if K < 3:
        print("⚠️  Need K ≥ 3 for 3-panel projections — skipping.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    import numpy as np
    import matplotlib.pyplot as plt

    n_agents = (
        len(tracker.agent_indices)
        if hasattr(tracker, "agent_indices")
        else tracker.n_agents
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    pairs = [(0, 1), (0, 2), (1, 2)]
    titles = [
        r"Projection $(\hat{\mu}_0, \hat{\mu}_1)$",
        r"Projection $(\hat{\mu}_0, \hat{\mu}_2)$",
        r"Projection $(\hat{\mu}_1, \hat{\mu}_2)$",
    ]

    cmap = plt.get_cmap("viridis")

    for ax, (i0, i1), title in zip(axes, pairs, titles):
        for a in range(n_agents):
            comps = tracker.get_component_array(a)  # (n_steps, K)

            norms = np.linalg.norm(comps, axis=1, keepdims=True)
            norms = np.clip(norms, 1e-12, None)
            unit = comps / norms

            xy = unit[:, [i0, i1]]
            n_steps = xy.shape[0]
            t = np.linspace(0.0, 1.0, n_steps)

            ax.plot(
                xy[:, 0],
                xy[:, 1],
                linewidth=1.5,
                alpha=0.9,
                label=f"Agent {a}"
            )
            ax.scatter(
                xy[0, 0],
                xy[0, 1],
                s=30,
                marker="o",
                alpha=0.95
            )
            ax.scatter(
                xy[-1, 0],
                xy[-1, 1],
                s=30,
                marker="X",
                alpha=0.95
            )

        ax.set_xlabel(rf"$\hat{{\mu}}_{i0}$".replace("i0", str(i0)))
        ax.set_ylabel(rf"$\hat{{\mu}}_{i1}$".replace("i1", str(i1)))
        ax.set_title(title)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.3)

    axes[0].legend(loc="upper right", fontsize=8, frameon=False)

    path = out_dir / "mu_gauge_orbit_projections.png"
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"✓ ploted projections {path}")



# =============================================================================
# Main
# =============================================================================