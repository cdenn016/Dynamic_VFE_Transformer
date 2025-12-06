#!/usr/bin/env python3
"""
Hamiltonian vs Gradient VFE Comparison
=======================================

Systematic comparison of Hamiltonian dynamics vs gradient flow for
Variational Free Energy minimization.

Usage:
    python experiments/hamiltonian_vs_gradient.py
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import pickle
import time
from copy import deepcopy

# Local imports
from agent.agents import Agent, AgentConfig
from agent.system import MultiAgentSystem
from agent.trainer import Trainer
from agent.hamiltonian_trainer import HamiltonianTrainer
from config import TrainingConfig
from geometry.geometry_base import BaseManifold, TopologyType
from gradients.free_energy_clean import compute_total_free_energy


@dataclass
class ExperimentConfig:
    """Configuration for comparison experiments."""
    n_agents: int = 4
    K: int = 3
    spatial_size: int = 50
    n_steps: int = 500
    dt: float = 0.01
    friction_values: List[float] = field(default_factory=lambda: [0.0, 0.1, 0.5, 1.0, 2.0])
    mass_scale: float = 1.0
    lr_mu: float = 0.01
    lr_sigma: float = 0.001
    lr_phi: float = 0.01
    output_dir: Path = field(default_factory=lambda: Path("./comparison_results"))
    seed: int = 42
    n_trials: int = 5


@dataclass
class ExperimentResult:
    """Container for experiment results."""
    method: str
    trial: int
    energy_trajectory: List[float] = field(default_factory=list)
    time_trajectory: List[float] = field(default_factory=list)
    final_energy: float = 0.0
    convergence_step: Optional[int] = None
    energy_conservation: Optional[float] = None
    has_oscillations: bool = False
    oscillation_period: Optional[float] = None
    mu_trajectory: Optional[np.ndarray] = None
    momentum_trajectory: Optional[np.ndarray] = None


def create_test_system(config: ExperimentConfig, rng: np.random.Generator) -> MultiAgentSystem:
    """Create a multi-agent system for testing."""
    from agent.masking import MaskConfig, SupportRegionSmooth
    from math_utils.sigma import CovarianceFieldInitializer

    manifold = BaseManifold(
        shape=(config.spatial_size,),
        topology=TopologyType.PERIODIC,
    )

    agents = []
    for i in range(config.n_agents):
        # Create agent config with correct parameters
        agent_cfg = AgentConfig(
            spatial_shape=(config.spatial_size,),
            K=config.K,
            phi_scale=0.05,  # Stronger initial gauge field to see dynamics
        )

        # Create agent with correct constructor signature
        agent = Agent(
            agent_id=i,
            config=agent_cfg,
            rng=rng,
            base_manifold=manifold,
        )

        # Override mu_q with random initial values for interesting dynamics
        agent.mu_q = 0.5 * rng.standard_normal((config.spatial_size, config.K)).astype(np.float32)

        # Create custom overlapping support
        center = int((i + 0.5) * config.spatial_size / config.n_agents)
        mask_cfg = MaskConfig(min_mask_for_normal_cov=0.1)

        x = np.arange(config.spatial_size)
        distances = np.minimum(
            np.abs(x - center),
            config.spatial_size - np.abs(x - center)
        )
        width = config.spatial_size / (2 * config.n_agents)
        mask_continuous = np.exp(-0.5 * (distances / width) ** 2).astype(np.float32)
        mask_binary = mask_continuous > 0.1

        support = SupportRegionSmooth(
            mask_binary=mask_binary,
            base_shape=(config.spatial_size,),
            config=mask_cfg,
            mask_continuous=mask_continuous,
        )
        agent.support = support

        agents.append(agent)

    from config import SystemConfig
    sys_config = SystemConfig(
        lambda_phi=1.0,           # Enable gauge field dynamics
        lambda_belief_align=1.0,  # Inter-agent belief coupling
        lambda_prior_align=0.5,   # Prior alignment
    )
    system = MultiAgentSystem(agents, config=sys_config)

    # Debug: Show initial phi stats
    phi_values = [np.linalg.norm(a.gauge.phi) for a in agents]
    print(f"  Initial φ norms: {[f'{v:.3f}' for v in phi_values]}")

    return system


def clone_system(system: MultiAgentSystem) -> MultiAgentSystem:
    """Deep copy a system for fair comparison."""
    return deepcopy(system)


def run_gradient_experiment(system: MultiAgentSystem, config: ExperimentConfig, trial: int) -> ExperimentResult:
    """Run gradient-based VFE minimization."""
    result = ExperimentResult(method="gradient", trial=trial)

    train_cfg = TrainingConfig(
        n_steps=config.n_steps,
        lr_mu_q=config.lr_mu,
        lr_sigma_q=config.lr_sigma,
        lr_phi=config.lr_phi,
        log_every=config.n_steps + 1,
        save_history=True,
    )

    trainer = Trainer(system, config=train_cfg)

    initial_energy = compute_total_free_energy(system).total
    result.energy_trajectory.append(initial_energy)
    result.time_trajectory.append(0.0)

    start_time = time.perf_counter()
    for step in range(config.n_steps):
        energies = trainer.step()
        result.energy_trajectory.append(energies.total)
        result.time_trajectory.append(time.perf_counter() - start_time)

    result.final_energy = result.energy_trajectory[-1]
    result.convergence_step = _find_convergence_step(result.energy_trajectory)
    return result


def run_hamiltonian_experiment(system: MultiAgentSystem, config: ExperimentConfig, friction: float, trial: int) -> ExperimentResult:
    """Run Hamiltonian VFE dynamics."""
    result = ExperimentResult(method=f"hamiltonian_γ={friction}", trial=trial)

    train_cfg = TrainingConfig(
        n_steps=config.n_steps,
        log_every=config.n_steps + 1,
        save_history=True,
    )

    trainer = HamiltonianTrainer(
        system,
        config=train_cfg,
        friction=friction,
        mass_scale=config.mass_scale,
        track_phase_space=True,
    )

    initial_H = trainer.history.total_hamiltonian[0] if trainer.history.total_hamiltonian else 0
    result.energy_trajectory.append(initial_H)
    result.time_trajectory.append(0.0)

    start_time = time.perf_counter()
    for step in range(config.n_steps):
        trainer.step(dt=config.dt)
        if trainer.history.total_hamiltonian:
            result.energy_trajectory.append(trainer.history.total_hamiltonian[-1])
        result.time_trajectory.append(time.perf_counter() - start_time)

    result.final_energy = result.energy_trajectory[-1]
    result.convergence_step = _find_convergence_step(result.energy_trajectory)

    if len(trainer.history.total_hamiltonian) > 1:
        H0 = trainer.history.total_hamiltonian[0]
        H_final = trainer.history.total_hamiltonian[-1]
        result.energy_conservation = abs(H_final - H0) / (abs(H0) + 1e-8)

    result.has_oscillations, result.oscillation_period = _detect_oscillations(result.energy_trajectory)
    return result


def _find_convergence_step(energy_trajectory: List[float], threshold: float = 1e-4) -> Optional[int]:
    """Find step where energy change falls below threshold."""
    if len(energy_trajectory) < 10:
        return None
    energies = np.array(energy_trajectory)
    for i in range(10, len(energies)):
        recent_change = np.std(energies[i-10:i]) / (np.abs(np.mean(energies[i-10:i])) + 1e-8)
        if recent_change < threshold:
            return i
    return None


def _detect_oscillations(energy_trajectory: List[float], min_peaks: int = 3) -> Tuple[bool, Optional[float]]:
    """Detect if energy trajectory has oscillations."""
    if len(energy_trajectory) < 20:
        return False, None

    energies = np.array(energy_trajectory)
    try:
        from scipy.signal import detrend, find_peaks
        detrended = detrend(energies)
        peaks, _ = find_peaks(detrended)
        if len(peaks) >= min_peaks:
            peak_diffs = np.diff(peaks)
            period = np.mean(peak_diffs) if len(peak_diffs) > 0 else None
            return True, period
    except:
        pass
    return False, None


def run_comparison_experiment(config: ExperimentConfig) -> Dict[str, List[ExperimentResult]]:
    """Run full comparison: gradient vs Hamiltonian at various friction values."""
    config.output_dir.mkdir(parents=True, exist_ok=True)

    results = {"gradient": []}
    for gamma in config.friction_values:
        results[f"hamiltonian_γ={gamma}"] = []

    print("=" * 70)
    print("HAMILTONIAN vs GRADIENT VFE COMPARISON")
    print("=" * 70)
    print(f"Agents: {config.n_agents}, K: {config.K}, Steps: {config.n_steps}")
    print(f"Friction values: {config.friction_values}")
    print(f"Trials per condition: {config.n_trials}")
    print("=" * 70)

    for trial in range(config.n_trials):
        print(f"\n--- Trial {trial + 1}/{config.n_trials} ---")

        rng = np.random.default_rng(config.seed + trial)
        base_system = create_test_system(config, rng)

        print("  Running gradient VFE...")
        system_copy = clone_system(base_system)
        grad_result = run_gradient_experiment(system_copy, config, trial)
        results["gradient"].append(grad_result)
        print(f"    Final energy: {grad_result.final_energy:.4f}")

        for gamma in config.friction_values:
            print(f"  Running Hamiltonian (γ={gamma})...")
            system_copy = clone_system(base_system)
            ham_result = run_hamiltonian_experiment(system_copy, config, gamma, trial)
            results[f"hamiltonian_γ={gamma}"].append(ham_result)
            osc_str = "Yes" if ham_result.has_oscillations else "No"
            cons_str = f"{ham_result.energy_conservation:.2e}" if ham_result.energy_conservation else "N/A"
            print(f"    Final: {ham_result.final_energy:.4f}, Conservation: {cons_str}, Oscillations: {osc_str}")

    return results


def plot_comparison_results(results: Dict[str, List[ExperimentResult]], config: ExperimentConfig):
    """Generate comparison figures."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    colors = plt.cm.viridis(np.linspace(0, 1, len(results)))
    method_colors = {method: colors[i] for i, method in enumerate(results.keys())}
    method_colors["gradient"] = "red"

    # Plot 1: Energy Trajectories
    ax1 = axes[0, 0]
    for method, result_list in results.items():
        all_energies = [r.energy_trajectory for r in result_list]
        min_len = min(len(e) for e in all_energies)
        energies_array = np.array([e[:min_len] for e in all_energies])
        mean_energy = np.mean(energies_array, axis=0)
        std_energy = np.std(energies_array, axis=0)
        steps = np.arange(len(mean_energy))
        label = "Gradient" if method == "gradient" else method.replace("hamiltonian_", "H-VFE ")
        ax1.plot(steps, mean_energy, label=label, color=method_colors[method], linewidth=2)
        ax1.fill_between(steps, mean_energy - std_energy, mean_energy + std_energy,
                        alpha=0.2, color=method_colors[method])
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Energy")
    ax1.set_title("Energy Trajectories")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Final Energy Comparison
    ax2 = axes[0, 1]
    methods = list(results.keys())
    final_energies = [np.mean([r.final_energy for r in results[m]]) for m in methods]
    final_stds = [np.std([r.final_energy for r in results[m]]) for m in methods]
    x_pos = np.arange(len(methods))
    ax2.bar(x_pos, final_energies, yerr=final_stds, capsize=5,
            color=[method_colors[m] for m in methods])
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([m.replace("hamiltonian_", "H-").replace("gradient", "Grad")
                        for m in methods], rotation=45, ha='right', fontsize=8)
    ax2.set_ylabel("Final Energy")
    ax2.set_title("Final Energy Comparison")
    ax2.grid(True, alpha=0.3, axis='y')

    # Plot 3: Convergence Speed
    ax3 = axes[1, 0]
    conv_data = []
    conv_labels = []
    for method in methods:
        conv_steps = [r.convergence_step for r in results[method] if r.convergence_step]
        if conv_steps:
            conv_data.append(conv_steps)
            conv_labels.append(method.replace("hamiltonian_", "H-").replace("gradient", "Grad"))
    if conv_data:
        bp = ax3.boxplot(conv_data, labels=conv_labels, patch_artist=True)
        for i, patch in enumerate(bp['boxes']):
            patch.set_facecolor(method_colors[methods[i]])
            patch.set_alpha(0.7)
        ax3.set_ylabel("Convergence Step")
        ax3.set_title("Convergence Speed (lower = faster)")
        ax3.tick_params(axis='x', rotation=45)

    # Plot 4: Oscillation Detection
    ax4 = axes[1, 1]
    ham_methods = [m for m in methods if "hamiltonian" in m]
    if ham_methods:
        osc_rates = []
        osc_labels = []
        for method in ham_methods:
            osc_count = sum(1 for r in results[method] if r.has_oscillations)
            osc_rate = osc_count / len(results[method])
            osc_rates.append(osc_rate * 100)
            osc_labels.append(method.replace("hamiltonian_", ""))
        x_pos = np.arange(len(ham_methods))
        ax4.bar(x_pos, osc_rates, color=[method_colors[m] for m in ham_methods])
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(osc_labels, fontsize=9)
        ax4.set_ylabel("% Trials with Oscillations")
        ax4.set_title("Oscillation Prevalence (Memory Indicator)")
        ax4.set_ylim(0, 100)
        ax4.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    fig_path = config.output_dir / "comparison_results.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved comparison figure: {fig_path}")
    plt.show()


def generate_summary_table(results: Dict[str, List[ExperimentResult]]) -> str:
    """Generate markdown summary table."""
    lines = [
        "| Method | Final Energy | Convergence Step | Oscillations | Energy Conservation |",
        "|--------|-------------|------------------|--------------|---------------------|",
    ]
    for method, result_list in results.items():
        final_mean = np.mean([r.final_energy for r in result_list])
        final_std = np.std([r.final_energy for r in result_list])
        conv_steps = [r.convergence_step for r in result_list if r.convergence_step]
        conv_str = f"{np.mean(conv_steps):.0f}±{np.std(conv_steps):.0f}" if conv_steps else "N/A"
        if "hamiltonian" in method:
            osc_rate = sum(1 for r in result_list if r.has_oscillations) / len(result_list) * 100
            osc_str = f"{osc_rate:.0f}%"
            conservations = [r.energy_conservation for r in result_list if r.energy_conservation is not None]
            cons_str = f"{np.mean(conservations):.2e}" if conservations else "N/A"
        else:
            osc_str = "N/A"
            cons_str = "N/A"
        method_label = method.replace("hamiltonian_", "H-VFE ").replace("gradient", "Gradient")
        lines.append(f"| {method_label} | {final_mean:.3f}±{final_std:.3f} | {conv_str} | {osc_str} | {cons_str} |")
    return "\n".join(lines)


def main():
    """Run comparison experiments."""
    config = ExperimentConfig(
        n_agents=4,
        K=3,
        spatial_size=50,
        n_steps=300,
        dt=0.01,
        friction_values=[0.0, 0.05, 0.1, 0.5],
        n_trials=3,
        seed=42,
        output_dir=Path("./comparison_results"),
    )

    results = run_comparison_experiment(config)

    results_path = config.output_dir / "results.pkl"
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"\n✓ Saved raw results: {results_path}")

    plot_comparison_results(results, config)

    summary = generate_summary_table(results)
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(summary)

    summary_path = config.output_dir / "summary.md"
    with open(summary_path, 'w') as f:
        f.write("# Hamiltonian vs Gradient VFE Comparison\n\n")
        f.write(summary)
    print(f"\n✓ Saved summary: {summary_path}")


if __name__ == "__main__":
    main()