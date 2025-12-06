#!/usr/bin/env python3
"""
==============================================
HAMILTONIAN vs GRADIENT VFE COMPARISON
==============================================

Click to run. Results saved to ./comparison_results/

This compares:
- Gradient descent VFE (standard approach)
- Hamiltonian VFE at various friction levels

Output:
- comparison_results.png: Visual comparison
- summary.md: Statistical summary
- results.pkl: Raw data for further analysis
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Suppress verbose output during import
import warnings
warnings.filterwarnings('ignore')

def main():
    print("\n" + "="*60)
    print("  HAMILTONIAN vs GRADIENT VFE COMPARISON")
    print("="*60 + "\n")

    # Import after path setup
    from experiments.hamiltonian_vs_gradient import (
        ExperimentConfig,
        run_comparison_experiment,
        plot_comparison_results,
        generate_summary_table,
    )
    import pickle

    # Configure experiment
    # (Adjust these for faster/slower runs)
    config = ExperimentConfig(
        n_agents=4,         # Number of agents
        K=7,                # Latent dimension
        spatial_size=25,    # Spatial resolution
        n_steps=300,        # Training steps per run
        dt=0.01,            # Hamiltonian timestep
        friction_values=[0.75, 1, 1.5, 2],  # γ values to test
        n_trials=3,         # Repeat each condition
        seed=42,            # For reproducibility
        output_dir=Path("./comparison_results"),
    )

    print("Configuration:")
    print(f"  Agents: {config.n_agents}")
    print(f"  Latent dim K: {config.K}")
    print(f"  Steps: {config.n_steps}")
    print(f"  Friction values: {config.friction_values}")
    print(f"  Trials: {config.n_trials}")
    print()

    # Run experiments
    print("Running experiments...")
    results = run_comparison_experiment(config)

    # Save results
    config.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = config.output_dir / "results.pkl"
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"\n✓ Raw results: {results_path}")

    # Generate plots
    print("\nGenerating plots...")
    plot_comparison_results(results, config)

    # Summary table
    summary = generate_summary_table(results)
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(summary)

    summary_path = config.output_dir / "summary.md"
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("# Hamiltonian vs Gradient VFE Comparison\n\n")
        f.write(summary)
    print(f"\n✓ Summary saved: {summary_path}")

    print("\n" + "="*60)
    print("  DONE! Check ./comparison_results/")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()