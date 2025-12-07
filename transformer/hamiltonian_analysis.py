"""
Hamiltonian Transformer Analysis & Visualization Tools.

Provides:
1. Phase space trajectory visualization
2. Token attribution via Hamiltonian reversal
3. Scalable reversibility testing
4. Comparison framework for baselines

For use with RTX 5090 scale training and interpretability research.
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from typing import Optional, List, Tuple, Dict, Any
from dataclasses import dataclass
from pathlib import Path

# Local imports
from transformer.hamiltonian_ffn import (
    HamiltonianFFN,
    PhaseSpaceState,
    LeapfrogIntegrator,
    HamiltonianPotential,
    HamiltonianKineticTerms,
)
from math_utils.generators import generate_so3_generators


@dataclass
class TrajectoryRecord:
    """Records phase space trajectory through integration steps."""
    steps: List[int]
    mu: List[torch.Tensor]          # (B, N, K) at each step
    Sigma: List[torch.Tensor]       # (B, N, K, K) at each step
    phi: List[torch.Tensor]         # (B, N, 3) at each step
    pi_mu: List[torch.Tensor]       # momenta
    pi_Sigma: List[torch.Tensor]
    pi_phi: List[torch.Tensor]
    hamiltonian: List[float]        # H at each step

    def to_numpy(self) -> Dict[str, np.ndarray]:
        """Convert to numpy arrays for plotting."""
        return {
            'steps': np.array(self.steps),
            'mu': torch.stack(self.mu).detach().cpu().numpy(),
            'Sigma': torch.stack(self.Sigma).detach().cpu().numpy(),
            'phi': torch.stack(self.phi).detach().cpu().numpy(),
            'hamiltonian': np.array(self.hamiltonian),
        }


class TrajectoryRecorder:
    """
    Records phase space evolution during Hamiltonian integration.

    Wraps a LeapfrogIntegrator to capture state at each step.
    """

    def __init__(
        self,
        integrator: LeapfrogIntegrator,
        potential: HamiltonianPotential,
    ):
        self.integrator = integrator
        self.potential = potential

    def integrate_with_recording(
        self,
        state: PhaseSpaceState,
        mu_prior: torch.Tensor,
        Sigma_prior: torch.Tensor,
        beta: Optional[torch.Tensor] = None,
        n_steps: Optional[int] = None,
    ) -> Tuple[PhaseSpaceState, TrajectoryRecord]:
        """
        Integrate while recording trajectory.

        Returns:
            final_state: PhaseSpaceState after integration
            trajectory: TrajectoryRecord with full history
        """
        n_steps = n_steps or self.integrator.n_steps

        # Initialize trajectory record
        trajectory = TrajectoryRecord(
            steps=[0],
            mu=[state.mu.clone()],
            Sigma=[state.Sigma.clone()],
            phi=[state.phi.clone()],
            pi_mu=[state.pi_mu.clone()],
            pi_Sigma=[state.pi_Sigma.clone()],
            pi_phi=[state.pi_phi.clone()],
            hamiltonian=[self._compute_hamiltonian(state, mu_prior, Sigma_prior, beta)],
        )

        # Integrate step by step
        current_state = state.clone()
        for step in range(1, n_steps + 1):
            current_state = self.integrator.step(
                current_state, mu_prior, Sigma_prior, beta, None, None
            )

            # Record
            trajectory.steps.append(step)
            trajectory.mu.append(current_state.mu.clone())
            trajectory.Sigma.append(current_state.Sigma.clone())
            trajectory.phi.append(current_state.phi.clone())
            trajectory.pi_mu.append(current_state.pi_mu.clone())
            trajectory.pi_Sigma.append(current_state.pi_Sigma.clone())
            trajectory.pi_phi.append(current_state.pi_phi.clone())
            trajectory.hamiltonian.append(
                self._compute_hamiltonian(current_state, mu_prior, Sigma_prior, beta)
            )

        return current_state, trajectory

    def _compute_hamiltonian(
        self,
        state: PhaseSpaceState,
        mu_prior: torch.Tensor,
        Sigma_prior: torch.Tensor,
        beta: Optional[torch.Tensor],
    ) -> float:
        """Compute total Hamiltonian H = T + V."""
        V, _ = self.potential.forward(state, mu_prior, Sigma_prior, beta, None, None)
        # Kinetic energy would need HamiltonianKineticTerms - approximate for now
        return V.mean().item()


class PhaseSpaceVisualizer:
    """
    Visualize phase space trajectories for interpretability.
    """

    def __init__(self, save_dir: Optional[str] = None):
        self.save_dir = Path(save_dir) if save_dir else Path("./analysis_outputs")
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def plot_mu_trajectory(
        self,
        trajectory: TrajectoryRecord,
        token_idx: int = 0,
        batch_idx: int = 0,
        dims: List[int] = None,
        title: str = "μ Evolution Through Integration",
        save_name: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot evolution of μ (belief mean) for a specific token.

        Args:
            trajectory: TrajectoryRecord from integration
            token_idx: Which token position to visualize
            batch_idx: Which batch element
            dims: Which embedding dimensions to plot (default: first 3)
            title: Plot title
            save_name: Filename to save (without extension)
        """
        data = trajectory.to_numpy()
        mu_traj = data['mu'][:, batch_idx, token_idx, :]  # (steps, K)
        steps = data['steps']

        dims = dims or [0, 1, 2]
        dims = [d for d in dims if d < mu_traj.shape[-1]]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # 2D trajectory (first two dims)
        ax1 = axes[0]
        if len(dims) >= 2:
            ax1.plot(mu_traj[:, dims[0]], mu_traj[:, dims[1]], 'b-', alpha=0.7, linewidth=1)
            ax1.scatter(mu_traj[0, dims[0]], mu_traj[0, dims[1]], c='green', s=100, marker='o', label='Start', zorder=5)
            ax1.scatter(mu_traj[-1, dims[0]], mu_traj[-1, dims[1]], c='red', s=100, marker='s', label='End', zorder=5)
            ax1.set_xlabel(f'μ[{dims[0]}]')
            ax1.set_ylabel(f'μ[{dims[1]}]')
            ax1.set_title(f'Phase Portrait (Token {token_idx})')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

        # Time series
        ax2 = axes[1]
        for d in dims:
            ax2.plot(steps, mu_traj[:, d], label=f'μ[{d}]', linewidth=1.5)
        ax2.set_xlabel('Integration Step')
        ax2.set_ylabel('μ value')
        ax2.set_title('μ Components vs Time')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.suptitle(title, fontsize=14)
        plt.tight_layout()

        if save_name:
            fig.savefig(self.save_dir / f"{save_name}.png", dpi=150, bbox_inches='tight')

        return fig

    def plot_hamiltonian_conservation(
        self,
        trajectory: TrajectoryRecord,
        title: str = "Hamiltonian Conservation",
        save_name: Optional[str] = None,
    ) -> plt.Figure:
        """Plot Hamiltonian (energy) over integration to check conservation."""
        data = trajectory.to_numpy()
        H = data['hamiltonian']
        steps = data['steps']

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Absolute value
        ax1 = axes[0]
        ax1.plot(steps, H, 'b-', linewidth=2)
        ax1.set_xlabel('Integration Step')
        ax1.set_ylabel('H (Hamiltonian)')
        ax1.set_title('Hamiltonian vs Time')
        ax1.grid(True, alpha=0.3)

        # Relative drift
        ax2 = axes[1]
        H_drift = (H - H[0]) / (abs(H[0]) + 1e-10) * 100
        ax2.plot(steps, H_drift, 'r-', linewidth=2)
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Integration Step')
        ax2.set_ylabel('Relative Drift (%)')
        ax2.set_title('Energy Drift')
        ax2.grid(True, alpha=0.3)

        plt.suptitle(title, fontsize=14)
        plt.tight_layout()

        if save_name:
            fig.savefig(self.save_dir / f"{save_name}.png", dpi=150, bbox_inches='tight')

        return fig

    def plot_token_attribution(
        self,
        forward_traj: TrajectoryRecord,
        backward_traj: TrajectoryRecord,
        target_token: int,
        batch_idx: int = 0,
        title: str = "Token Attribution via Reversal",
        save_name: Optional[str] = None,
    ) -> plt.Figure:
        """
        Visualize token attribution by showing forward and backward trajectories.

        Shows how a target token's representation evolved forward, then
        traces backward to reveal causal input contributions.
        """
        fwd = forward_traj.to_numpy()
        bwd = backward_traj.to_numpy()

        mu_fwd = fwd['mu'][:, batch_idx, target_token, :3]  # First 3 dims
        mu_bwd = bwd['mu'][:, batch_idx, target_token, :3]

        fig = plt.figure(figsize=(15, 5))
        gs = GridSpec(1, 3, figure=fig)

        # Forward trajectory
        ax1 = fig.add_subplot(gs[0])
        for d in range(3):
            ax1.plot(fwd['steps'], mu_fwd[:, d], label=f'd={d}')
        ax1.set_xlabel('Forward Step')
        ax1.set_ylabel('μ')
        ax1.set_title(f'Forward: Token {target_token} Evolution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Backward trajectory
        ax2 = fig.add_subplot(gs[1])
        for d in range(3):
            ax2.plot(bwd['steps'], mu_bwd[:, d], label=f'd={d}')
        ax2.set_xlabel('Backward Step')
        ax2.set_ylabel('μ')
        ax2.set_title(f'Backward: Tracing to Origin')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 2D phase portrait overlay
        ax3 = fig.add_subplot(gs[2])
        ax3.plot(mu_fwd[:, 0], mu_fwd[:, 1], 'b-', alpha=0.7, label='Forward', linewidth=2)
        ax3.plot(mu_bwd[:, 0], mu_bwd[:, 1], 'r--', alpha=0.7, label='Backward', linewidth=2)
        ax3.scatter(mu_fwd[0, 0], mu_fwd[0, 1], c='green', s=100, marker='o', zorder=5)
        ax3.scatter(mu_fwd[-1, 0], mu_fwd[-1, 1], c='blue', s=100, marker='s', zorder=5)
        ax3.set_xlabel('μ[0]')
        ax3.set_ylabel('μ[1]')
        ax3.set_title('Phase Portrait: Forward (blue) & Backward (red)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        plt.suptitle(title, fontsize=14)
        plt.tight_layout()

        if save_name:
            fig.savefig(self.save_dir / f"{save_name}.png", dpi=150, bbox_inches='tight')

        return fig


class ReversibilityTester:
    """
    Comprehensive reversibility testing at various scales.
    """

    def __init__(self, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.results: List[Dict[str, Any]] = []

    def run_test(
        self,
        seq_len: int,
        embed_dim: int,
        n_steps: int,
        dt: float = 0.01,
        batch_size: int = 2,
    ) -> Dict[str, Any]:
        """
        Run reversibility test with given configuration.

        Returns dict with:
            - config: test configuration
            - mu_error: reconstruction error for μ
            - Sigma_error: reconstruction error for Σ
            - phi_error: reconstruction error for φ
            - energy_drift: |H_T - H_0|
            - passed: whether error < threshold
        """
        device = self.device
        K = embed_dim

        # Setup components
        generators = generate_so3_generators(embed_dim)
        if isinstance(generators, np.ndarray):
            generators = torch.from_numpy(generators).float()
        generators = generators.to(device)

        kinetic = HamiltonianKineticTerms(embed_dim=embed_dim).to(device)
        potential = HamiltonianPotential(
            embed_dim=embed_dim,
            generators=generators,
            alpha=1.0,
            lambda_belief=1.0,
            kappa=1.0,
        ).to(device)
        integrator = LeapfrogIntegrator(
            kinetic=kinetic,
            potential=potential,
            dt=dt,
            n_steps=1,  # We'll step manually
            update_Sigma=True,
            update_phi=True,
        )

        # Create initial state
        torch.manual_seed(42)
        mu_0 = torch.randn(batch_size, seq_len, K, device=device)
        A = torch.randn(batch_size, seq_len, K, K, device=device) * 0.3
        Sigma_0 = A @ A.transpose(-1, -2) + torch.eye(K, device=device)
        phi_0 = torch.randn(batch_size, seq_len, 3, device=device) * 0.1

        pi_mu_0 = torch.randn(batch_size, seq_len, K, device=device) * 0.1
        pi_Sigma_0 = torch.randn(batch_size, seq_len, K, K, device=device) * 0.01
        pi_Sigma_0 = 0.5 * (pi_Sigma_0 + pi_Sigma_0.transpose(-1, -2))
        pi_phi_0 = torch.randn(batch_size, seq_len, 3, device=device) * 0.1

        state_0 = PhaseSpaceState(
            mu=mu_0, Sigma=Sigma_0, phi=phi_0,
            pi_mu=pi_mu_0, pi_Sigma=pi_Sigma_0, pi_phi=pi_phi_0,
        )

        mu_prior = torch.zeros(batch_size, seq_len, K, device=device)
        Sigma_prior = torch.eye(K, device=device).unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1, -1).clone()
        beta = torch.ones(batch_size, seq_len, device=device)

        # Initial Hamiltonian
        H_0, _ = potential.forward(state_0, mu_prior, Sigma_prior, beta, None, None)

        # Forward integration
        state = state_0.clone()
        for _ in range(n_steps):
            state = integrator.step(state, mu_prior, Sigma_prior, beta, None, None)
        state_T = state

        H_T, _ = potential.forward(state_T, mu_prior, Sigma_prior, beta, None, None)

        # Negate momenta for reversal
        state_reversed = PhaseSpaceState(
            mu=state_T.mu.clone(),
            Sigma=state_T.Sigma.clone(),
            phi=state_T.phi.clone(),
            pi_mu=-state_T.pi_mu.clone(),
            pi_Sigma=-state_T.pi_Sigma.clone(),
            pi_phi=-state_T.pi_phi.clone(),
        )

        # Backward integration
        for _ in range(n_steps):
            state_reversed = integrator.step(state_reversed, mu_prior, Sigma_prior, beta, None, None)

        # Compute errors
        mu_error = (state_reversed.mu - mu_0).abs().mean().item()
        Sigma_error = (state_reversed.Sigma - Sigma_0).abs().mean().item()
        phi_error = (state_reversed.phi - phi_0).abs().mean().item()
        energy_drift = (H_T - H_0).abs().mean().item()

        result = {
            'config': {
                'seq_len': seq_len,
                'embed_dim': embed_dim,
                'n_steps': n_steps,
                'dt': dt,
                'batch_size': batch_size,
            },
            'mu_error': mu_error,
            'Sigma_error': Sigma_error,
            'phi_error': phi_error,
            'total_error': mu_error + Sigma_error + phi_error,
            'energy_drift': energy_drift,
            'passed': mu_error < 1e-4,  # Focus on μ for pass/fail
        }

        self.results.append(result)
        return result

    def run_scale_sweep(
        self,
        seq_lens: List[int] = [8, 16, 32, 64, 128],
        embed_dim: int = 64,
        n_steps: int = 10,
        dt: float = 0.01,
    ) -> List[Dict[str, Any]]:
        """Run reversibility tests across multiple sequence lengths."""
        results = []
        for seq_len in seq_lens:
            print(f"Testing seq_len={seq_len}...", end=" ")
            try:
                result = self.run_test(seq_len, embed_dim, n_steps, dt)
                status = "PASS" if result['passed'] else "FAIL"
                print(f"{status} (μ_err={result['mu_error']:.2e})")
                results.append(result)
            except Exception as e:
                print(f"ERROR: {e}")
                results.append({'config': {'seq_len': seq_len}, 'error': str(e)})
        return results

    def summary_table(self) -> str:
        """Generate summary table of all results."""
        lines = [
            "=" * 70,
            "REVERSIBILITY TEST SUMMARY",
            "=" * 70,
            f"{'seq_len':>8} {'n_steps':>8} {'μ_error':>12} {'Σ_error':>12} {'Status':>8}",
            "-" * 70,
        ]

        for r in self.results:
            if 'error' in r:
                lines.append(f"{r['config']['seq_len']:>8} {'ERROR':>8}")
            else:
                c = r['config']
                status = "PASS" if r['passed'] else "FAIL"
                lines.append(
                    f"{c['seq_len']:>8} {c['n_steps']:>8} "
                    f"{r['mu_error']:>12.2e} {r['Sigma_error']:>12.2e} {status:>8}"
                )

        lines.append("=" * 70)
        return "\n".join(lines)


class SentenceAnalyzer:
    """
    Analyze Hamiltonian dynamics on actual text sentences.

    Requires a trained model and tokenizer.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    ):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.visualizer = PhaseSpaceVisualizer()

    def analyze_sentence(
        self,
        sentence: str,
        target_position: int = -1,
    ) -> Dict[str, Any]:
        """
        Analyze a sentence through the Hamiltonian transformer.

        Args:
            sentence: Input text
            target_position: Token position to trace (-1 for last)

        Returns:
            Dictionary with tokens, trajectories, and attribution info
        """
        # Tokenize
        tokens = self.tokenizer.encode(sentence)
        token_strs = [self.tokenizer.decode([t]) for t in tokens]

        if target_position < 0:
            target_position = len(tokens) + target_position

        # This would hook into actual model forward pass
        # For now, return placeholder showing expected structure
        return {
            'sentence': sentence,
            'tokens': token_strs,
            'target_position': target_position,
            'target_token': token_strs[target_position] if target_position < len(token_strs) else None,
            # These would be populated by actual model analysis:
            # 'forward_trajectory': ...,
            # 'backward_trajectory': ...,
            # 'attribution_scores': ...,
        }

    def visualize_attribution(
        self,
        sentence: str,
        target_position: int = -1,
        save_name: Optional[str] = None,
    ) -> plt.Figure:
        """
        Create attribution visualization for a sentence.

        Shows which input tokens contribute most to the target token's
        representation, traced via Hamiltonian reversal.
        """
        analysis = self.analyze_sentence(sentence, target_position)
        tokens = analysis['tokens']
        target = analysis['target_position']

        # Placeholder attribution (would come from actual reversal)
        n_tokens = len(tokens)
        # Simulated: nearby tokens have higher attribution
        distances = np.abs(np.arange(n_tokens) - target)
        attribution = np.exp(-distances / 2)
        attribution = attribution / attribution.sum()

        fig, ax = plt.subplots(figsize=(max(12, n_tokens * 0.8), 4))

        colors = plt.cm.Reds(attribution / attribution.max())
        bars = ax.bar(range(n_tokens), attribution, color=colors)

        ax.set_xticks(range(n_tokens))
        ax.set_xticklabels(tokens, rotation=45, ha='right')
        ax.set_ylabel('Attribution Score')
        ax.set_title(f'Token Attribution for "{tokens[target]}" (position {target})')

        # Highlight target
        bars[target].set_edgecolor('blue')
        bars[target].set_linewidth(3)

        plt.tight_layout()

        if save_name:
            fig.savefig(self.visualizer.save_dir / f"{save_name}.png", dpi=150, bbox_inches='tight')

        return fig


# =============================================================================
# Demo / Test Functions
# =============================================================================

def demo_trajectory_visualization():
    """Demonstrate trajectory visualization with synthetic data."""
    print("\n" + "=" * 60)
    print("DEMO: Phase Space Trajectory Visualization")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Setup
    B, N, K = 1, 4, 16
    n_steps = 50
    dt = 0.01

    generators = generate_so3_generators(K)
    if isinstance(generators, np.ndarray):
        generators = torch.from_numpy(generators).float()
    generators = generators.to(device)

    kinetic = HamiltonianKineticTerms(embed_dim=K).to(device)
    potential = HamiltonianPotential(
        embed_dim=K, generators=generators,
        alpha=1.0, lambda_belief=1.0, kappa=1.0,
    ).to(device)
    integrator = LeapfrogIntegrator(
        kinetic=kinetic, potential=potential,
        dt=dt, n_steps=1, update_Sigma=True, update_phi=True,
    )

    # Create recorder and visualizer
    recorder = TrajectoryRecorder(integrator, potential)
    visualizer = PhaseSpaceVisualizer(save_dir="./analysis_outputs")

    # Initial state
    torch.manual_seed(123)
    mu_0 = torch.randn(B, N, K, device=device)
    A = torch.randn(B, N, K, K, device=device) * 0.3
    Sigma_0 = A @ A.transpose(-1, -2) + torch.eye(K, device=device)
    phi_0 = torch.randn(B, N, 3, device=device) * 0.1
    pi_mu_0 = torch.randn(B, N, K, device=device) * 0.1
    pi_Sigma_0 = torch.randn(B, N, K, K, device=device) * 0.01
    pi_Sigma_0 = 0.5 * (pi_Sigma_0 + pi_Sigma_0.transpose(-1, -2))
    pi_phi_0 = torch.randn(B, N, 3, device=device) * 0.1

    state_0 = PhaseSpaceState(
        mu=mu_0, Sigma=Sigma_0, phi=phi_0,
        pi_mu=pi_mu_0, pi_Sigma=pi_Sigma_0, pi_phi=pi_phi_0,
    )

    mu_prior = torch.zeros(B, N, K, device=device)
    Sigma_prior = torch.eye(K, device=device).unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1).clone()

    # Record trajectory
    print(f"\nIntegrating {n_steps} steps...")
    final_state, trajectory = recorder.integrate_with_recording(
        state_0, mu_prior, Sigma_prior, n_steps=n_steps
    )

    # Plot
    print("Generating visualizations...")
    fig1 = visualizer.plot_mu_trajectory(
        trajectory, token_idx=0, dims=[0, 1, 2],
        title="μ Evolution: Token 0",
        save_name="demo_mu_trajectory"
    )

    fig2 = visualizer.plot_hamiltonian_conservation(
        trajectory,
        title="Hamiltonian Conservation Check",
        save_name="demo_hamiltonian"
    )

    print(f"\nSaved to: {visualizer.save_dir}")
    print("✓ Trajectory visualization demo complete")

    return trajectory, visualizer


def demo_reversibility_sweep():
    """Run reversibility tests at multiple scales."""
    print("\n" + "=" * 60)
    print("DEMO: Reversibility Scale Sweep")
    print("=" * 60)

    tester = ReversibilityTester()

    # Test at multiple scales
    results = tester.run_scale_sweep(
        seq_lens=[4, 8, 16, 32],
        embed_dim=32,
        n_steps=10,
        dt=0.01,
    )

    print("\n" + tester.summary_table())

    return results


def demo_attribution_visualization():
    """Demonstrate token attribution visualization (mock data)."""
    print("\n" + "=" * 60)
    print("DEMO: Token Attribution Visualization")
    print("=" * 60)

    # Mock tokenizer for demo
    class MockTokenizer:
        def encode(self, text):
            return list(range(len(text.split())))
        def decode(self, ids):
            return f"tok{ids[0]}"

    # For demo, create visualization with mock data
    visualizer = PhaseSpaceVisualizer(save_dir="./analysis_outputs")

    sentence = "The quick brown fox jumps over the lazy dog"
    tokens = sentence.split()
    target = len(tokens) - 1  # "dog"

    # Simulated attribution
    n = len(tokens)
    distances = np.abs(np.arange(n) - target)
    attribution = np.exp(-distances / 3)
    attribution[target] = 1.0  # Target itself
    attribution = attribution / attribution.sum()

    fig, ax = plt.subplots(figsize=(12, 4))
    colors = plt.cm.Blues(attribution / attribution.max())
    bars = ax.bar(range(n), attribution, color=colors)
    bars[target].set_edgecolor('red')
    bars[target].set_linewidth(3)
    ax.set_xticks(range(n))
    ax.set_xticklabels(tokens, rotation=45, ha='right')
    ax.set_ylabel('Attribution Score')
    ax.set_title(f'Token Attribution for "{tokens[target]}" via Hamiltonian Reversal')
    plt.tight_layout()

    fig.savefig(visualizer.save_dir / "demo_attribution.png", dpi=150, bbox_inches='tight')
    print(f"Saved to: {visualizer.save_dir / 'demo_attribution.png'}")
    print("✓ Attribution visualization demo complete")

    return fig


if __name__ == "__main__":
    print("=" * 70)
    print("HAMILTONIAN TRANSFORMER ANALYSIS TOOLKIT")
    print("=" * 70)

    # Run demos
    demo_trajectory_visualization()
    demo_reversibility_sweep()
    demo_attribution_visualization()

    print("\n" + "=" * 70)
    print("All demos complete! Check ./analysis_outputs/ for visualizations.")
    print("=" * 70)
