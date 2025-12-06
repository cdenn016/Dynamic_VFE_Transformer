# -*- coding: utf-8 -*-
"""
Created on Fri Nov 21 22:29:53 2025

@author: chris and christine
"""

"""
Spatial Field Visualization
============================

Plotting functions for visualizing spatial fields across the base manifold:
- Gauge fields phi(c): SO(3) rotations per location
- Covariance fields Sigma(c): Uncertainty quantification
- Mean fields mu(c): Belief centers
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from analysis.plots import get_spatial_shape_from_system


# =============================================================================
# Gauge Field phi(c) Visualization
# =============================================================================

def plot_phi_fields(system, out_dir: Path):
    """
    Visualize gauge fields phi(c) for all agents.

    Shows:
    - Individual phi components (phi_x, phi_y, phi_z)
    - Phi norm ||phi(c)||
    - Comparison across agents

    Args:
        system: MultiAgentSystem with agents having phi attributes
        out_dir: Directory to save plots
    """
    if system is None:
        return

    shape = get_spatial_shape_from_system(system)
    ndim = len(shape)
    out_dir.mkdir(parents=True, exist_ok=True)

    if ndim == 0:
        # 0D: bar plot of phi magnitudes
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Individual components
        ax = axes[0]
        agents = range(len(system.agents))
        width = 0.25
        x = np.arange(len(agents))

        for comp_idx, comp_name in enumerate(['φ_x', 'φ_y', 'φ_z']):
            values = [system.agents[i].phi[comp_idx] for i in agents]
            ax.bar(x + comp_idx*width, values, width, label=comp_name, alpha=0.7)

        ax.set_xlabel('Agent')
        ax.set_ylabel('φ component')
        ax.set_title('Gauge Field Components (0D)')
        ax.set_xticks(x + width)
        ax.set_xticklabels(agents)
        ax.legend()
        ax.grid(alpha=0.3)

        # Norms
        ax = axes[1]
        norms = [np.linalg.norm(system.agents[i].phi) for i in agents]
        ax.bar(agents, norms, color='purple', alpha=0.6)
        ax.axhline(np.pi, color='red', linestyle='--', label='π (branch cut)')
        ax.set_xlabel('Agent')
        ax.set_ylabel('||φ||')
        ax.set_title('Gauge Field Norms (0D)')
        ax.legend()
        ax.grid(alpha=0.3)

        plt.tight_layout()
        path = out_dir / "phi_fields_0d.png"
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"✓ Saved {path}")
        return

    if ndim == 1:
        # 1D: line plots
        x = np.arange(shape[0])

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Component plots
        for comp_idx, (ax, comp_name) in enumerate(zip(axes.flat[:3], ['φ_x', 'φ_y', 'φ_z'])):
            for i, agent in enumerate(system.agents):
                phi_comp = agent.phi[..., comp_idx]
                mask = agent.support.mask_continuous if hasattr(agent, 'support') else None
                if mask is not None:
                    ax.plot(x, phi_comp, label=f'Agent {i}', alpha=0.7)
                    ax.fill_between(x, phi_comp.min(), phi_comp.max(),
                                   where=mask>0.5, alpha=0.1)
                else:
                    ax.plot(x, phi_comp, label=f'Agent {i}', alpha=0.7)
            ax.set_xlabel('Position')
            ax.set_ylabel(comp_name)
            ax.set_title(f'{comp_name}(c) Fields')
            ax.legend()
            ax.grid(alpha=0.3)

        # Norm plot
        ax = axes.flat[3]
        for i, agent in enumerate(system.agents):
            phi_norm = np.linalg.norm(agent.phi, axis=-1)
            ax.plot(x, phi_norm, label=f'Agent {i}', linewidth=2, alpha=0.7)
        ax.axhline(np.pi, color='red', linestyle='--', linewidth=2, label='π (branch cut)')
        ax.set_xlabel('Position')
        ax.set_ylabel('||φ(c)||')
        ax.set_title('Gauge Field Norms')
        ax.legend()
        ax.grid(alpha=0.3)

        plt.tight_layout()
        path = out_dir / "phi_fields_1d.png"
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"✓ Saved {path}")

    elif ndim == 2:
        # 2D: heatmaps
        H, W = shape

        # Per-agent component plots
        for i, agent in enumerate(system.agents):
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))

            # Three components
            for comp_idx, (ax, comp_name) in enumerate(zip(axes.flat[:3], ['φ_x', 'φ_y', 'φ_z'])):
                phi_comp = agent.phi[..., comp_idx].reshape(H, W)
                im = ax.imshow(phi_comp, origin='lower', cmap='RdBu_r')
                plt.colorbar(im, ax=ax, fraction=0.046)
                ax.set_title(f'{comp_name}(c) - Agent {i}')

            # Norm
            ax = axes.flat[3]
            phi_norm = np.linalg.norm(agent.phi, axis=-1).reshape(H, W)
            im = ax.imshow(phi_norm, origin='lower', cmap='viridis')
            plt.colorbar(im, ax=ax, fraction=0.046, label='||φ||')
            ax.set_title(f'||φ(c)|| - Agent {i}')

            plt.tight_layout()
            path = out_dir / f"phi_fields_2d_agent{i}.png"
            plt.savefig(path, dpi=150)
            plt.close()
            print(f"✓ Saved {path}")


# =============================================================================
# Covariance Field Sigma(c) Analysis
# =============================================================================

def plot_sigma_fields(system, out_dir: Path):
    """
    Visualize covariance fields Sigma(c).

    Shows:
    - Eigenvalue spectrum across space
    - Determinant (generalized variance)
    - Trace (total variance)
    - Condition number (numerical stability)

    Args:
        system: MultiAgentSystem with agents having Sigma_q and Sigma_p
        out_dir: Directory to save plots
    """
    if system is None:
        return

    shape = get_spatial_shape_from_system(system)
    ndim = len(shape)
    out_dir.mkdir(parents=True, exist_ok=True)

    if ndim == 0:
        # 0D: eigenvalue bars
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        for ax_idx, (ax, which) in enumerate(zip(axes, ['q', 'p'])):
            agents = range(len(system.agents))
            Sigma_key = 'Sigma_q' if which == 'q' else 'Sigma_p'

            eigs_all = []
            for i in agents:
                Sigma = getattr(system.agents[i], Sigma_key)
                eigs = np.linalg.eigvalsh(Sigma)
                eigs_all.append(eigs)

            eigs_all = np.array(eigs_all)  # (n_agents, K)
            K = eigs_all.shape[1]

            x = np.arange(len(agents))
            width = 0.8 / K

            for k in range(K):
                ax.bar(x + k*width, eigs_all[:, k], width, label=f'λ_{k}', alpha=0.7)

            ax.set_xlabel('Agent')
            ax.set_ylabel('Eigenvalue')
            ax.set_title(f'Σ_{which} Eigenvalues (0D)')
            ax.set_xticks(x + width*(K-1)/2)
            ax.set_xticklabels(agents)
            ax.legend()
            ax.grid(alpha=0.3)
            ax.set_yscale('log')

        plt.tight_layout()
        path = out_dir / "sigma_fields_0d.png"
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"✓ Saved {path}")
        return

    if ndim == 1:
        x = np.arange(shape[0])

        for which in ['q', 'p']:
            Sigma_key = 'Sigma_q' if which == 'q' else 'Sigma_p'

            fig, axes = plt.subplots(2, 2, figsize=(14, 10))

            # Min eigenvalue (stability)
            ax = axes[0, 0]
            for i, agent in enumerate(system.agents):
                Sigma = getattr(agent, Sigma_key)
                eigs = np.linalg.eigvalsh(Sigma)  # (*S, K)
                min_eig = eigs[..., 0]
                ax.semilogy(x, min_eig, label=f'Agent {i}', alpha=0.7)
            ax.set_xlabel('Position')
            ax.set_ylabel('Min eigenvalue')
            ax.set_title(f'Σ_{which}: Minimum Eigenvalue (Stability)')
            ax.legend()
            ax.grid(alpha=0.3)

            # Max eigenvalue (scale)
            ax = axes[0, 1]
            for i, agent in enumerate(system.agents):
                Sigma = getattr(agent, Sigma_key)
                eigs = np.linalg.eigvalsh(Sigma)
                max_eig = eigs[..., -1]
                ax.semilogy(x, max_eig, label=f'Agent {i}', alpha=0.7)
            ax.set_xlabel('Position')
            ax.set_ylabel('Max eigenvalue')
            ax.set_title(f'Σ_{which}: Maximum Eigenvalue (Scale)')
            ax.legend()
            ax.grid(alpha=0.3)

            # Determinant (volume)
            ax = axes[1, 0]
            for i, agent in enumerate(system.agents):
                Sigma = getattr(agent, Sigma_key)
                det_Sigma = np.linalg.det(Sigma)
                det_Sigma = np.maximum(det_Sigma, 1e-20)  # Avoid log(0)
                ax.semilogy(x, det_Sigma, label=f'Agent {i}', alpha=0.7)
            ax.set_xlabel('Position')
            ax.set_ylabel('det(Σ)')
            ax.set_title(f'Σ_{which}: Determinant (Volume)')
            ax.legend()
            ax.grid(alpha=0.3)

            # Condition number (stability)
            ax = axes[1, 1]
            for i, agent in enumerate(system.agents):
                Sigma = getattr(agent, Sigma_key)
                eigs = np.linalg.eigvalsh(Sigma)
                cond = eigs[..., -1] / np.maximum(eigs[..., 0], 1e-20)
                ax.semilogy(x, cond, label=f'Agent {i}', alpha=0.7)
            ax.set_xlabel('Position')
            ax.set_ylabel('κ(Σ)')
            ax.set_title(f'Σ_{which}: Condition Number')
            ax.legend()
            ax.grid(alpha=0.3)

            plt.tight_layout()
            path = out_dir / f"sigma_fields_1d_{which}.png"
            plt.savefig(path, dpi=150)
            plt.close()
            print(f"✓ Saved {path}")

    elif ndim == 2:
        H, W = shape

        for which in ['q', 'p']:
            Sigma_key = 'Sigma_q' if which == 'q' else 'Sigma_p'

            for i, agent in enumerate(system.agents):
                Sigma = getattr(agent, Sigma_key)

                fig, axes = plt.subplots(2, 2, figsize=(12, 10))

                # Min eigenvalue
                ax = axes[0, 0]
                eigs = np.linalg.eigvalsh(Sigma).reshape(H, W, -1)
                min_eig = eigs[..., 0]
                im = ax.imshow(np.log10(min_eig), origin='lower', cmap='viridis')
                plt.colorbar(im, ax=ax, label='log10(λ_min)')
                ax.set_title(f'Min Eigenvalue - Agent {i}')

                # Max eigenvalue
                ax = axes[0, 1]
                max_eig = eigs[..., -1]
                im = ax.imshow(np.log10(max_eig), origin='lower', cmap='plasma')
                plt.colorbar(im, ax=ax, label='log10(λ_max)')
                ax.set_title(f'Max Eigenvalue - Agent {i}')

                # Determinant
                ax = axes[1, 0]
                det_Sigma = np.linalg.det(Sigma).reshape(H, W)
                det_Sigma = np.maximum(det_Sigma, 1e-20)
                im = ax.imshow(np.log10(det_Sigma), origin='lower', cmap='coolwarm')
                plt.colorbar(im, ax=ax, label='log10(det)')
                ax.set_title(f'Determinant - Agent {i}')

                # Condition number
                ax = axes[1, 1]
                cond = max_eig / np.maximum(min_eig, 1e-20)
                im = ax.imshow(np.log10(cond), origin='lower', cmap='hot')
                plt.colorbar(im, ax=ax, label='log10(κ)')
                ax.set_title(f'Condition Number - Agent {i}')

                plt.tight_layout()
                path = out_dir / f"sigma_fields_2d_{which}_agent{i}.png"
                plt.savefig(path, dpi=150)
                plt.close()
                print(f"✓ Saved {path}")


# =============================================================================
# Mean Field mu(c) Visualization
# =============================================================================

def plot_mu_fields(system, out_dir: Path):
    """
    Visualize μ_q fields for agents.

    Shows spatial distribution of belief means across the base manifold.

    Args:
        system: MultiAgentSystem with agents having mu_q attributes
        out_dir: Directory to save plots
    """
    if system is None:
        return

    shape = get_spatial_shape_from_system(system)
    ndim = len(shape)
    out_dir.mkdir(parents=True, exist_ok=True)

    mu_fields = []
    for agent in system.agents:
        mu = getattr(agent, "mu_q", None)
        if mu is None:
            mu_fields.append(None)
        elif mu.ndim == 1:
            mu_fields.append(mu[0:1])
        else:
            mu_fields.append(mu[..., 0])

    if ndim == 0:
        values = [mf[0] if mf is not None and mf.size > 0 else np.nan
                  for mf in mu_fields]
        plt.figure(figsize=(6, 4))
        plt.bar(range(len(values)), values)
        plt.xlabel("Agent")
        plt.ylabel("μ_q (first dim)")
        plt.title("0D μ_q per agent")
        plt.tight_layout()
        path = out_dir / "mu_fields_0d.png"
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"✓ Saved {path}")
        return

    if ndim == 1:
        x = np.arange(shape[0])
        plt.figure(figsize=(8, 4))
        for i, mf in enumerate(mu_fields):
            if mf is None:
                continue
            plt.plot(x, mf, label=f"Agent {i}", alpha=0.8)
        plt.xlabel("Position")
        plt.ylabel("μ_q[..., 0]")
        plt.title("Belief mean μ_q (first latent dim, 1D)")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        path = out_dir / "mu_fields_1d.png"
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"✓ Saved {path}")
        return

    if ndim == 2:
        H, W = shape
        n_agents = system.n_agents
        cols = min(3, n_agents)
        rows = int(np.ceil(n_agents / cols))
        plt.figure(figsize=(4 * cols, 4 * rows))
        for i, mf in enumerate(mu_fields):
            if mf is None:
                continue
            ax = plt.subplot(rows, cols, i + 1)
            im = ax.imshow(mf.reshape(H, W), origin="lower")
            plt.colorbar(im, ax=ax, fraction=0.046)
            ax.set_title(f"μ_q[...,0] – Agent {i}")
            ax.set_xticks([])
            ax.set_yticks([])
        plt.tight_layout()
        path = out_dir / "mu_fields_2d_agents.png"
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"✓ Saved {path}")