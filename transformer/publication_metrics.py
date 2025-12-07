"""
Publication Metrics Framework for Gauge-Theoretic Hamiltonian Transformer
==========================================================================

Comprehensive analytics for publication-quality experiments.

Metrics Categories:
1. Physics Validation - Energy conservation, symplecticity, reversibility
2. Training Dynamics - Loss decomposition, gradients, mass matrix stats
3. Ablation Studies - Systematic comparison of configurations
4. Interpretability - Token attribution, belief trajectories
5. Performance - Perplexity, BPC, efficiency

Integrates with:
- hamiltonian_ffn.py (HamiltonianFFN diagnostics)
- hamiltonian_analysis.py (trajectory recording, visualization)
- train_publication.py (training loop)

Author: Chris & Claude
Date: December 2025
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import numpy as np
import json
import csv
import time
import math
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any, Union
from dataclasses import dataclass, field, asdict
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches


# =============================================================================
# Data Classes for Metrics
# =============================================================================

@dataclass
class PhysicsSnapshot:
    """Single snapshot of physics metrics."""
    step: int
    H_init: float
    H_final: float
    delta_H: float
    T_init: float
    T_final: float
    V_init: float
    V_final: float
    V_self: float
    V_align: float
    V_ce: float
    mass_eigenvalue_min: float = 0.0
    mass_eigenvalue_max: float = 0.0
    mass_condition_number: float = 0.0
    spd_eigenvalue_min: float = 0.0
    spd_preserved: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TrainingSnapshot:
    """Single snapshot of training dynamics."""
    step: int
    epoch: float
    train_loss: float
    train_ce: float
    train_ppl: float
    train_bpc: float
    val_loss: Optional[float] = None
    val_ppl: Optional[float] = None
    val_bpc: Optional[float] = None
    grad_norm_total: float = 0.0
    grad_norm_mu: float = 0.0
    grad_norm_sigma: float = 0.0
    grad_norm_phi: float = 0.0
    grad_norm_ffn: float = 0.0
    lr_mu: float = 0.0
    lr_sigma: float = 0.0
    lr_phi: float = 0.0
    lr_ffn: float = 0.0
    tokens_per_sec: float = 0.0
    step_time: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AblationConfig:
    """Configuration for a single ablation experiment."""
    name: str
    ffn_mode: str
    mass_use_prior: bool = True
    mass_use_observation: bool = False
    mass_use_incoming_social: bool = False
    mass_use_outgoing_recoil: bool = False
    n_leapfrog_steps: int = 10
    dt: float = 0.01
    update_sigma: bool = True
    update_phi: bool = False
    gamma: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AblationResult:
    """Results from a single ablation experiment."""
    config: AblationConfig
    final_val_loss: float
    final_val_ppl: float
    final_val_bpc: float
    best_val_loss: float
    best_val_ppl: float
    convergence_step: int
    avg_delta_H: float  # Energy conservation
    avg_step_time: float
    total_time: float

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['config'] = self.config.to_dict()
        return d


# =============================================================================
# Physics Metrics
# =============================================================================

class PhysicsMetrics:
    """
    Track and analyze physics-related metrics.

    Key metrics:
    - Energy conservation: ΔH per step, scaling with dt
    - Symplectic structure: Phase space volume preservation
    - SPD preservation: Eigenvalue positivity for Σ
    - Mass matrix: Condition number, eigenvalue distribution
    """

    def __init__(self, save_dir: Optional[Path] = None):
        self.save_dir = Path(save_dir) if save_dir else Path("./publication_outputs")
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.history: List[PhysicsSnapshot] = []

    def record(self, step: int, diagnostics: Dict[str, Any],
               Sigma: Optional[torch.Tensor] = None,
               M: Optional[torch.Tensor] = None):
        """
        Record physics metrics from HamiltonianFFN diagnostics.

        Args:
            step: Training step
            diagnostics: Dict from HamiltonianFFN.forward()
            Sigma: Covariance matrices for SPD check
            M: Mass matrix for condition number
        """
        snapshot = PhysicsSnapshot(
            step=step,
            H_init=diagnostics.get('H_init', 0.0),
            H_final=diagnostics.get('H_final', 0.0),
            delta_H=diagnostics.get('delta_H', 0.0),
            T_init=diagnostics.get('T_init', 0.0),
            T_final=diagnostics.get('T_final', 0.0),
            V_init=diagnostics.get('V_init', 0.0),
            V_final=diagnostics.get('V_final', 0.0),
            V_self=diagnostics.get('V_self', 0.0),
            V_align=diagnostics.get('V_align', 0.0),
            V_ce=diagnostics.get('V_ce', 0.0),
        )

        # SPD preservation check
        if Sigma is not None:
            with torch.no_grad():
                eigenvalues = torch.linalg.eigvalsh(Sigma)
                snapshot.spd_eigenvalue_min = eigenvalues.min().item()
                snapshot.spd_preserved = snapshot.spd_eigenvalue_min > 0

        # Mass matrix conditioning
        if M is not None:
            with torch.no_grad():
                M_eigenvalues = torch.linalg.eigvalsh(M)
                snapshot.mass_eigenvalue_min = M_eigenvalues.min().item()
                snapshot.mass_eigenvalue_max = M_eigenvalues.max().item()
                if snapshot.mass_eigenvalue_min > 1e-10:
                    snapshot.mass_condition_number = (
                        snapshot.mass_eigenvalue_max / snapshot.mass_eigenvalue_min
                    )

        self.history.append(snapshot)

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        if not self.history:
            return {}

        delta_H_values = [s.delta_H for s in self.history]
        spd_violations = sum(1 for s in self.history if not s.spd_preserved)

        return {
            'n_samples': len(self.history),
            'delta_H_mean': np.mean(delta_H_values),
            'delta_H_std': np.std(delta_H_values),
            'delta_H_max': np.max(delta_H_values),
            'delta_H_min': np.min(delta_H_values),
            'spd_violations': spd_violations,
            'spd_violation_rate': spd_violations / len(self.history),
            'avg_V_self': np.mean([s.V_self for s in self.history]),
            'avg_V_align': np.mean([s.V_align for s in self.history]),
            'avg_V_ce': np.mean([s.V_ce for s in self.history]),
        }

    def save(self, filename: str = "physics_metrics.json"):
        """Save to JSON."""
        data = {
            'summary': self.get_summary(),
            'history': [s.to_dict() for s in self.history],
        }
        with open(self.save_dir / filename, 'w') as f:
            json.dump(data, f, indent=2)

    def run_dt_scaling_study(
        self,
        ffn_factory,  # Callable that creates HamiltonianFFN with given dt
        test_state: Dict[str, torch.Tensor],
        dt_values: List[float] = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001],
        n_steps: int = 10,
    ) -> Dict[str, List[float]]:
        """
        Study energy conservation scaling with dt.

        For symplectic integrators, ΔH should scale as O(dt²).

        Args:
            ffn_factory: Function(dt) -> HamiltonianFFN
            test_state: Dict with mu, Sigma, phi, mu_prior, Sigma_prior, beta
            dt_values: Time steps to test
            n_steps: Integration steps per test

        Returns:
            Dict with dt values and corresponding delta_H
        """
        results = {'dt': [], 'delta_H': [], 'delta_H_relative': []}

        for dt in dt_values:
            ffn = ffn_factory(dt)

            # Run forward pass
            with torch.no_grad():
                mu_new, Sigma_new, phi_new, diag = ffn(
                    test_state['mu'],
                    test_state['Sigma'],
                    test_state['phi'],
                    test_state['mu_prior'],
                    test_state['Sigma_prior'],
                    test_state.get('beta'),
                )

            delta_H = diag['delta_H']
            H_init = abs(diag['H_init']) + 1e-10

            results['dt'].append(dt)
            results['delta_H'].append(delta_H)
            results['delta_H_relative'].append(delta_H / H_init)

        return results

    def plot_energy_conservation(
        self,
        save_name: str = "energy_conservation",
        figsize: Tuple[int, int] = (14, 5)
    ) -> plt.Figure:
        """Plot energy conservation over training."""
        if not self.history:
            return None

        steps = [s.step for s in self.history]
        delta_H = [s.delta_H for s in self.history]
        H_init = [s.H_init for s in self.history]
        H_final = [s.H_final for s in self.history]

        fig, axes = plt.subplots(1, 3, figsize=figsize)

        # Absolute energy drift
        ax1 = axes[0]
        ax1.semilogy(steps, delta_H, 'b-', linewidth=1, alpha=0.7)
        ax1.axhline(y=np.mean(delta_H), color='r', linestyle='--',
                    label=f'Mean: {np.mean(delta_H):.2e}')
        ax1.set_xlabel('Training Step')
        ax1.set_ylabel('|ΔH| (log scale)')
        ax1.set_title('Energy Conservation')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Hamiltonian values
        ax2 = axes[1]
        ax2.plot(steps, H_init, 'b-', label='H_init', alpha=0.7)
        ax2.plot(steps, H_final, 'r-', label='H_final', alpha=0.7)
        ax2.set_xlabel('Training Step')
        ax2.set_ylabel('Hamiltonian')
        ax2.set_title('H Evolution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Distribution of delta_H
        ax3 = axes[2]
        ax3.hist(delta_H, bins=50, edgecolor='black', alpha=0.7)
        ax3.axvline(x=np.mean(delta_H), color='r', linestyle='--',
                    label=f'Mean: {np.mean(delta_H):.2e}')
        ax3.set_xlabel('|ΔH|')
        ax3.set_ylabel('Count')
        ax3.set_title('ΔH Distribution')
        ax3.legend()

        plt.tight_layout()
        fig.savefig(self.save_dir / f"{save_name}.png", dpi=150, bbox_inches='tight')
        fig.savefig(self.save_dir / f"{save_name}.pdf", bbox_inches='tight')

        return fig

    def plot_dt_scaling(
        self,
        results: Dict[str, List[float]],
        save_name: str = "dt_scaling",
    ) -> plt.Figure:
        """
        Plot dt scaling study results.

        For symplectic integrators: ΔH ∝ dt²
        """
        dt = np.array(results['dt'])
        delta_H = np.array(results['delta_H'])

        fig, ax = plt.subplots(figsize=(8, 6))

        # Data points
        ax.loglog(dt, delta_H, 'bo-', markersize=8, linewidth=2, label='Measured ΔH')

        # Reference lines
        # O(dt²) scaling
        dt_ref = np.logspace(np.log10(dt.min()), np.log10(dt.max()), 100)
        scale_factor = delta_H[len(dt)//2] / (dt[len(dt)//2]**2)
        ax.loglog(dt_ref, scale_factor * dt_ref**2, 'g--',
                  linewidth=2, alpha=0.7, label='O(dt²) reference')

        # O(dt) scaling for comparison
        scale_factor_1 = delta_H[len(dt)//2] / dt[len(dt)//2]
        ax.loglog(dt_ref, scale_factor_1 * dt_ref, 'r:',
                  linewidth=2, alpha=0.5, label='O(dt) reference')

        ax.set_xlabel('Time step dt', fontsize=12)
        ax.set_ylabel('Energy error |ΔH|', fontsize=12)
        ax.set_title('Symplectic Integrator: Energy Conservation Scaling', fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, which='both')

        plt.tight_layout()
        fig.savefig(self.save_dir / f"{save_name}.png", dpi=150, bbox_inches='tight')
        fig.savefig(self.save_dir / f"{save_name}.pdf", bbox_inches='tight')

        return fig


# =============================================================================
# Training Dynamics Tracker
# =============================================================================

class TrainingDynamicsTracker:
    """
    Track training dynamics with fine-grained metrics.

    Extends basic loss tracking with:
    - Per-component gradient norms
    - Loss decomposition over time
    - Momentum statistics for Hamiltonian mode
    - Mass matrix evolution
    """

    def __init__(self, save_dir: Optional[Path] = None):
        self.save_dir = Path(save_dir) if save_dir else Path("./publication_outputs")
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.history: List[TrainingSnapshot] = []
        self.momentum_stats: List[Dict[str, float]] = []
        self.mass_stats: List[Dict[str, float]] = []

    def record_step(
        self,
        step: int,
        epoch: float,
        train_metrics: Dict[str, float],
        grad_norms: Optional[Dict[str, float]] = None,
        lrs: Optional[Dict[str, float]] = None,
        step_time: float = 0.0,
        batch_size: int = 1,
        seq_len: int = 1,
    ):
        """Record a training step."""
        tokens_per_sec = (batch_size * seq_len) / step_time if step_time > 0 else 0

        train_ce = train_metrics.get('ce_loss', train_metrics.get('train_loss_ce', 0))
        train_bpc = train_ce / math.log(2) if train_ce else 0
        train_ppl = math.exp(min(train_ce, 10)) if train_ce else float('inf')

        snapshot = TrainingSnapshot(
            step=step,
            epoch=epoch,
            train_loss=train_metrics.get('loss', train_metrics.get('train_loss_total', 0)),
            train_ce=train_ce,
            train_ppl=train_ppl,
            train_bpc=train_bpc,
            grad_norm_total=grad_norms.get('total', 0) if grad_norms else 0,
            grad_norm_mu=grad_norms.get('mu', 0) if grad_norms else 0,
            grad_norm_sigma=grad_norms.get('sigma', 0) if grad_norms else 0,
            grad_norm_phi=grad_norms.get('phi', 0) if grad_norms else 0,
            grad_norm_ffn=grad_norms.get('ffn', 0) if grad_norms else 0,
            lr_mu=lrs.get('mu_embed', 0) if lrs else 0,
            lr_sigma=lrs.get('sigma_embed', 0) if lrs else 0,
            lr_phi=lrs.get('phi_embed', 0) if lrs else 0,
            lr_ffn=lrs.get('ffn', 0) if lrs else 0,
            tokens_per_sec=tokens_per_sec,
            step_time=step_time,
        )

        self.history.append(snapshot)

    def record_validation(self, step: int, val_metrics: Dict[str, float]):
        """Record validation metrics."""
        for snapshot in reversed(self.history):
            if snapshot.step == step:
                val_ce = val_metrics.get('ce_loss', val_metrics.get('loss', 0))
                snapshot.val_loss = val_metrics.get('loss', val_ce)
                snapshot.val_ppl = val_metrics.get('perplexity', math.exp(min(val_ce, 10)))
                snapshot.val_bpc = val_ce / math.log(2) if val_ce else None
                break

    def record_momentum_stats(
        self,
        step: int,
        pi_mu: torch.Tensor,
        pi_Sigma: Optional[torch.Tensor] = None,
        pi_phi: Optional[torch.Tensor] = None,
    ):
        """Record momentum distribution statistics."""
        stats = {
            'step': step,
            'pi_mu_mean': pi_mu.mean().item(),
            'pi_mu_std': pi_mu.std().item(),
            'pi_mu_max': pi_mu.abs().max().item(),
        }

        if pi_Sigma is not None:
            stats['pi_Sigma_mean'] = pi_Sigma.mean().item()
            stats['pi_Sigma_std'] = pi_Sigma.std().item()
            stats['pi_Sigma_frobenius'] = torch.norm(pi_Sigma, p='fro').mean().item()

        if pi_phi is not None:
            stats['pi_phi_mean'] = pi_phi.mean().item()
            stats['pi_phi_std'] = pi_phi.std().item()

        self.momentum_stats.append(stats)

    def record_mass_stats(
        self,
        step: int,
        M: torch.Tensor,
        M_inv: torch.Tensor,
    ):
        """Record mass matrix statistics."""
        with torch.no_grad():
            eigenvalues = torch.linalg.eigvalsh(M)

            stats = {
                'step': step,
                'M_eigenvalue_min': eigenvalues.min().item(),
                'M_eigenvalue_max': eigenvalues.max().item(),
                'M_eigenvalue_mean': eigenvalues.mean().item(),
                'M_condition_number': (eigenvalues.max() / eigenvalues.min()).item(),
                'M_trace_mean': torch.diagonal(M, dim1=-2, dim2=-1).sum(-1).mean().item(),
            }

            self.mass_stats.append(stats)

    def get_summary(self) -> Dict[str, Any]:
        """Get training summary."""
        if not self.history:
            return {}

        val_losses = [s.val_loss for s in self.history if s.val_loss is not None]

        return {
            'total_steps': len(self.history),
            'final_train_loss': self.history[-1].train_loss,
            'final_train_ppl': self.history[-1].train_ppl,
            'final_train_bpc': self.history[-1].train_bpc,
            'best_val_loss': min(val_losses) if val_losses else None,
            'best_val_ppl': min([s.val_ppl for s in self.history if s.val_ppl]) if val_losses else None,
            'avg_tokens_per_sec': np.mean([s.tokens_per_sec for s in self.history]),
            'total_time': sum([s.step_time for s in self.history]),
        }

    def save(self, filename: str = "training_dynamics.json"):
        """Save to JSON."""
        data = {
            'summary': self.get_summary(),
            'history': [s.to_dict() for s in self.history],
            'momentum_stats': self.momentum_stats,
            'mass_stats': self.mass_stats,
        }
        with open(self.save_dir / filename, 'w') as f:
            json.dump(data, f, indent=2)

    def save_csv(self, filename: str = "training_history.csv"):
        """Save history to CSV."""
        if not self.history:
            return

        with open(self.save_dir / filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(self.history[0].to_dict().keys()))
            writer.writeheader()
            for snapshot in self.history:
                writer.writerow(snapshot.to_dict())

    def plot_training_curves(
        self,
        save_name: str = "training_curves",
        figsize: Tuple[int, int] = (16, 10),
    ) -> plt.Figure:
        """Generate comprehensive training curves figure."""
        if not self.history:
            return None

        fig = plt.figure(figsize=figsize)
        gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

        steps = [s.step for s in self.history]

        # Loss curves
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(steps, [s.train_loss for s in self.history], 'b-',
                 label='Train', alpha=0.7)
        val_steps = [s.step for s in self.history if s.val_loss is not None]
        val_losses = [s.val_loss for s in self.history if s.val_loss is not None]
        if val_losses:
            ax1.plot(val_steps, val_losses, 'r-', label='Val', linewidth=2)
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Loss')
        ax1.set_title('Loss Curves')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Perplexity
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.semilogy(steps, [s.train_ppl for s in self.history], 'b-',
                     label='Train', alpha=0.7)
        val_ppls = [s.val_ppl for s in self.history if s.val_ppl is not None]
        if val_ppls:
            ax2.semilogy(val_steps, val_ppls, 'r-', label='Val', linewidth=2)
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Perplexity (log)')
        ax2.set_title('Perplexity')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # BPC
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.plot(steps, [s.train_bpc for s in self.history], 'b-',
                 label='Train', alpha=0.7)
        val_bpcs = [s.val_bpc for s in self.history if s.val_bpc is not None]
        if val_bpcs:
            ax3.plot(val_steps, val_bpcs, 'r-', label='Val', linewidth=2)
        ax3.set_xlabel('Step')
        ax3.set_ylabel('Bits per Character')
        ax3.set_title('BPC')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Gradient norms
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.semilogy(steps, [s.grad_norm_total for s in self.history],
                     'k-', label='Total', alpha=0.7)
        ax4.semilogy(steps, [s.grad_norm_mu for s in self.history],
                     'b-', label='μ', alpha=0.5)
        ax4.semilogy(steps, [s.grad_norm_ffn for s in self.history],
                     'g-', label='FFN', alpha=0.5)
        ax4.set_xlabel('Step')
        ax4.set_ylabel('Gradient Norm (log)')
        ax4.set_title('Gradient Norms')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        # Learning rates
        ax5 = fig.add_subplot(gs[1, 1])
        ax5.plot(steps, [s.lr_mu for s in self.history], label='μ')
        ax5.plot(steps, [s.lr_sigma for s in self.history], label='σ')
        ax5.plot(steps, [s.lr_phi for s in self.history], label='φ')
        ax5.plot(steps, [s.lr_ffn for s in self.history], label='FFN')
        ax5.set_xlabel('Step')
        ax5.set_ylabel('Learning Rate')
        ax5.set_title('Learning Rates')
        ax5.legend()
        ax5.grid(True, alpha=0.3)

        # Throughput
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.plot(steps, [s.tokens_per_sec for s in self.history], 'g-', alpha=0.7)
        avg_throughput = np.mean([s.tokens_per_sec for s in self.history])
        ax6.axhline(y=avg_throughput, color='r', linestyle='--',
                    label=f'Avg: {avg_throughput:.0f}')
        ax6.set_xlabel('Step')
        ax6.set_ylabel('Tokens/sec')
        ax6.set_title('Throughput')
        ax6.legend()
        ax6.grid(True, alpha=0.3)

        plt.suptitle('Training Dynamics', fontsize=14, y=1.02)

        fig.savefig(self.save_dir / f"{save_name}.png", dpi=150, bbox_inches='tight')
        fig.savefig(self.save_dir / f"{save_name}.pdf", bbox_inches='tight')

        return fig


# =============================================================================
# Ablation Study Framework
# =============================================================================

class AblationStudy:
    """
    Framework for systematic ablation experiments.

    Supports:
    - Mode comparison (learned vs VFE vs hamiltonian)
    - Mass term ablation (4 terms from Inertia of Belief)
    - Integration parameter sweeps (n_steps, dt)
    - Dynamics ablation (update_Σ, update_φ)
    """

    def __init__(self, save_dir: Optional[Path] = None):
        self.save_dir = Path(save_dir) if save_dir else Path("./publication_outputs/ablation")
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.results: List[AblationResult] = []

    def add_result(self, result: AblationResult):
        """Add a completed experiment result."""
        self.results.append(result)

    def create_mode_comparison_configs(self) -> List[AblationConfig]:
        """Create configs for mode comparison ablation."""
        return [
            AblationConfig(name="baseline_learned", ffn_mode="learned"),
            AblationConfig(name="vfe_gradient_engine", ffn_mode="variational_gradient_engine"),
            AblationConfig(name="hamiltonian_prior_only", ffn_mode="hamiltonian",
                          mass_use_prior=True),
            AblationConfig(name="hamiltonian_full_mass", ffn_mode="hamiltonian",
                          mass_use_prior=True, mass_use_incoming_social=True,
                          mass_use_outgoing_recoil=True),
        ]

    def create_mass_term_configs(self) -> List[AblationConfig]:
        """Create configs for mass term ablation (Table 2 in paper)."""
        configs = []

        # Base: prior only
        configs.append(AblationConfig(
            name="M_prior", ffn_mode="hamiltonian",
            mass_use_prior=True,
        ))

        # Add observation
        configs.append(AblationConfig(
            name="M_prior+obs", ffn_mode="hamiltonian",
            mass_use_prior=True, mass_use_observation=True,
        ))

        # Add incoming social
        configs.append(AblationConfig(
            name="M_prior+incoming", ffn_mode="hamiltonian",
            mass_use_prior=True, mass_use_incoming_social=True,
        ))

        # Add outgoing recoil
        configs.append(AblationConfig(
            name="M_prior+outgoing", ffn_mode="hamiltonian",
            mass_use_prior=True, mass_use_outgoing_recoil=True,
        ))

        # Full mass
        configs.append(AblationConfig(
            name="M_full", ffn_mode="hamiltonian",
            mass_use_prior=True, mass_use_observation=True,
            mass_use_incoming_social=True, mass_use_outgoing_recoil=True,
        ))

        return configs

    def create_integration_sweep_configs(
        self,
        n_steps_values: List[int] = [1, 2, 4, 8, 16],
        dt_values: List[float] = [0.1, 0.05, 0.01],
    ) -> List[AblationConfig]:
        """Create configs for integration parameter sweep."""
        configs = []
        for n_steps in n_steps_values:
            for dt in dt_values:
                configs.append(AblationConfig(
                    name=f"nsteps{n_steps}_dt{dt}",
                    ffn_mode="hamiltonian",
                    n_leapfrog_steps=n_steps,
                    dt=dt,
                ))
        return configs

    def get_summary_table(self) -> str:
        """Generate summary table as string."""
        if not self.results:
            return "No results yet."

        lines = [
            "=" * 90,
            "ABLATION STUDY RESULTS",
            "=" * 90,
            f"{'Config':<30} {'Val Loss':>10} {'Val PPL':>10} {'Val BPC':>10} {'ΔH':>10} {'Time':>10}",
            "-" * 90,
        ]

        for r in sorted(self.results, key=lambda x: x.best_val_loss):
            lines.append(
                f"{r.config.name:<30} "
                f"{r.best_val_loss:>10.4f} "
                f"{r.best_val_ppl:>10.2f} "
                f"{r.final_val_bpc:>10.4f} "
                f"{r.avg_delta_H:>10.2e} "
                f"{r.total_time:>10.1f}s"
            )

        lines.append("=" * 90)
        return "\n".join(lines)

    def save(self, filename: str = "ablation_results.json"):
        """Save all results to JSON."""
        data = {
            'summary': self.get_summary_table(),
            'results': [r.to_dict() for r in self.results],
        }
        with open(self.save_dir / filename, 'w') as f:
            json.dump(data, f, indent=2)

    def plot_mode_comparison(
        self,
        save_name: str = "mode_comparison",
    ) -> plt.Figure:
        """Plot mode comparison bar chart."""
        if not self.results:
            return None

        # Filter mode comparison results
        mode_results = [r for r in self.results if 'learned' in r.config.name or
                       'vfe' in r.config.name or 'hamiltonian' in r.config.name]

        if not mode_results:
            return None

        names = [r.config.name for r in mode_results]
        val_bpc = [r.final_val_bpc for r in mode_results]
        delta_H = [r.avg_delta_H for r in mode_results]

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # BPC comparison
        ax1 = axes[0]
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(names)]
        bars = ax1.bar(names, val_bpc, color=colors, edgecolor='black')
        ax1.set_ylabel('Bits per Character')
        ax1.set_title('Performance Comparison')
        ax1.tick_params(axis='x', rotation=45)

        # Energy conservation (only for Hamiltonian modes)
        ax2 = axes[1]
        ham_results = [r for r in mode_results if 'hamiltonian' in r.config.name]
        if ham_results:
            ham_names = [r.config.name for r in ham_results]
            ham_delta_H = [r.avg_delta_H for r in ham_results]
            ax2.bar(ham_names, ham_delta_H, color='green', edgecolor='black')
            ax2.set_ylabel('Average |ΔH|')
            ax2.set_title('Energy Conservation (Hamiltonian modes)')
            ax2.tick_params(axis='x', rotation=45)
            ax2.set_yscale('log')

        plt.tight_layout()
        fig.savefig(self.save_dir / f"{save_name}.png", dpi=150, bbox_inches='tight')
        fig.savefig(self.save_dir / f"{save_name}.pdf", bbox_inches='tight')

        return fig

    def plot_mass_ablation_heatmap(
        self,
        save_name: str = "mass_ablation_heatmap",
    ) -> plt.Figure:
        """Plot mass term ablation as heatmap."""
        # Filter mass ablation results
        mass_results = [r for r in self.results if r.config.ffn_mode == 'hamiltonian']

        if len(mass_results) < 2:
            return None

        # Create matrix: rows = configurations, columns = metrics
        metrics = ['val_bpc', 'delta_H', 'time']
        configs = [r.config.name for r in mass_results]

        data = np.array([
            [r.final_val_bpc, np.log10(r.avg_delta_H + 1e-10), r.total_time / 60]
            for r in mass_results
        ])

        fig, ax = plt.subplots(figsize=(8, 6))

        # Normalize each column
        data_norm = (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0) + 1e-10)

        im = ax.imshow(data_norm, cmap='RdYlGn_r', aspect='auto')

        ax.set_xticks(range(len(metrics)))
        ax.set_xticklabels(['BPC', 'log₁₀|ΔH|', 'Time (min)'])
        ax.set_yticks(range(len(configs)))
        ax.set_yticklabels(configs)

        # Add values
        for i in range(len(configs)):
            for j in range(len(metrics)):
                text = f'{data[i, j]:.3f}' if j < 2 else f'{data[i, j]:.1f}'
                ax.text(j, i, text, ha='center', va='center', fontsize=9)

        ax.set_title('Mass Term Ablation (normalized, lower is better for BPC/ΔH)')
        plt.colorbar(im, ax=ax, label='Normalized Score')

        plt.tight_layout()
        fig.savefig(self.save_dir / f"{save_name}.png", dpi=150, bbox_inches='tight')
        fig.savefig(self.save_dir / f"{save_name}.pdf", bbox_inches='tight')

        return fig


# =============================================================================
# Interpretability Metrics
# =============================================================================

class InterpretabilityMetrics:
    """
    Compute interpretability metrics for Hamiltonian transformer.

    Key metrics:
    - Token attribution via reversal
    - Belief trajectory analysis
    - Attention flow patterns
    - Gauge field semantics
    """

    def __init__(self, save_dir: Optional[Path] = None):
        self.save_dir = Path(save_dir) if save_dir else Path("./publication_outputs/interpretability")
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def compute_reversal_attribution(
        self,
        forward_mu: torch.Tensor,  # (T, B, N, K) trajectory
        backward_mu: torch.Tensor,  # (T, B, N, K) reversed trajectory
        target_position: int = -1,
    ) -> torch.Tensor:
        """
        Compute token attribution scores via trajectory reversal.

        The key insight: reversibility means we can trace which input
        tokens most influenced a given output token.

        Args:
            forward_mu: Forward trajectory of belief means
            backward_mu: Backward trajectory after momentum flip
            target_position: Which output token to attribute

        Returns:
            attribution: (B, N) attribution scores for each input position
        """
        T, B, N, K = forward_mu.shape

        if target_position < 0:
            target_position = N + target_position

        # Get initial and final states
        mu_init = forward_mu[0]  # (B, N, K)
        mu_final = forward_mu[-1]  # (B, N, K)
        mu_reversed = backward_mu[-1]  # (B, N, K) - should ≈ mu_init

        # Reversal error per position
        reversal_error = (mu_reversed - mu_init).abs().mean(dim=-1)  # (B, N)

        # Attribution = how much each position's reversal error correlates with target
        target_error = reversal_error[:, target_position:target_position+1]  # (B, 1)

        # Simple attribution: inverse of reversal error (well-reversed = high attribution)
        attribution = 1.0 / (reversal_error + 1e-8)
        attribution = attribution / attribution.sum(dim=-1, keepdim=True)  # Normalize

        return attribution

    def compute_trajectory_curvature(
        self,
        mu_trajectory: torch.Tensor,  # (T, B, N, K)
    ) -> torch.Tensor:
        """
        Compute curvature of belief trajectory.

        High curvature indicates complex dynamics / phase transitions.

        Returns:
            curvature: (T-2, B, N) curvature at each step
        """
        T = mu_trajectory.shape[0]

        if T < 3:
            return torch.zeros(1)

        # Velocity
        velocity = mu_trajectory[1:] - mu_trajectory[:-1]  # (T-1, B, N, K)

        # Acceleration
        acceleration = velocity[1:] - velocity[:-1]  # (T-2, B, N, K)

        # Curvature = |acceleration| / |velocity|²
        speed = velocity[:-1].norm(dim=-1) + 1e-8  # (T-2, B, N)
        accel_norm = acceleration.norm(dim=-1)  # (T-2, B, N)

        curvature = accel_norm / (speed ** 2)

        return curvature

    def analyze_attention_flow(
        self,
        beta: torch.Tensor,  # (B, N, N) or (B, H, N, N)
    ) -> Dict[str, torch.Tensor]:
        """
        Analyze attention flow patterns.

        Returns:
            Dict with:
            - entropy: Attention entropy per position
            - concentration: How concentrated attention is
            - bidirectionality: Symmetric vs asymmetric attention
        """
        # Handle multi-head
        if beta.dim() == 4:
            beta = beta.mean(dim=1)  # Average over heads

        B, N, _ = beta.shape

        # Entropy
        beta_safe = beta + 1e-10
        entropy = -(beta_safe * beta_safe.log()).sum(dim=-1)  # (B, N)

        # Concentration (max attention weight)
        concentration = beta.max(dim=-1)[0]  # (B, N)

        # Bidirectionality: how symmetric is attention?
        beta_T = beta.transpose(-1, -2)
        symmetry = 1.0 - (beta - beta_T).abs().mean(dim=-1)  # (B, N)

        return {
            'entropy': entropy,
            'concentration': concentration,
            'bidirectionality': symmetry,
            'mean_entropy': entropy.mean().item(),
            'mean_concentration': concentration.mean().item(),
        }

    def plot_token_attribution(
        self,
        tokens: List[str],
        attribution: torch.Tensor,  # (N,) or (B, N)
        target_idx: int,
        save_name: str = "token_attribution",
        title: Optional[str] = None,
    ) -> plt.Figure:
        """Plot token attribution visualization."""
        if attribution.dim() > 1:
            attribution = attribution[0]  # Take first batch

        attribution = attribution.detach().cpu().numpy()
        n_tokens = len(tokens)

        fig, ax = plt.subplots(figsize=(max(12, n_tokens * 0.8), 4))

        # Color by attribution
        colors = plt.cm.Reds(attribution / attribution.max())
        bars = ax.bar(range(n_tokens), attribution, color=colors, edgecolor='black')

        # Highlight target
        bars[target_idx].set_edgecolor('blue')
        bars[target_idx].set_linewidth(3)

        ax.set_xticks(range(n_tokens))
        ax.set_xticklabels(tokens, rotation=45, ha='right')
        ax.set_ylabel('Attribution Score')

        if title is None:
            title = f'Token Attribution for "{tokens[target_idx]}" (via Hamiltonian reversal)'
        ax.set_title(title)

        plt.tight_layout()
        fig.savefig(self.save_dir / f"{save_name}.png", dpi=150, bbox_inches='tight')
        fig.savefig(self.save_dir / f"{save_name}.pdf", bbox_inches='tight')

        return fig

    def plot_trajectory_3d(
        self,
        mu_trajectory: torch.Tensor,  # (T, B, N, K)
        token_idx: int = 0,
        batch_idx: int = 0,
        dims: List[int] = [0, 1, 2],
        save_name: str = "trajectory_3d",
    ) -> plt.Figure:
        """Plot 3D trajectory of belief evolution."""
        from mpl_toolkits.mplot3d import Axes3D

        traj = mu_trajectory[:, batch_idx, token_idx, :].detach().cpu().numpy()

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Trajectory line
        ax.plot(traj[:, dims[0]], traj[:, dims[1]], traj[:, dims[2]],
                'b-', linewidth=1, alpha=0.7)

        # Start and end points
        ax.scatter(*traj[0, dims], c='green', s=100, marker='o', label='Start')
        ax.scatter(*traj[-1, dims], c='red', s=100, marker='s', label='End')

        # Color trajectory by time
        for i in range(len(traj) - 1):
            color = plt.cm.viridis(i / len(traj))
            ax.plot(traj[i:i+2, dims[0]], traj[i:i+2, dims[1]], traj[i:i+2, dims[2]],
                   color=color, linewidth=2)

        ax.set_xlabel(f'μ[{dims[0]}]')
        ax.set_ylabel(f'μ[{dims[1]}]')
        ax.set_zlabel(f'μ[{dims[2]}]')
        ax.set_title(f'Belief Trajectory (Token {token_idx})')
        ax.legend()

        plt.tight_layout()
        fig.savefig(self.save_dir / f"{save_name}.png", dpi=150, bbox_inches='tight')

        return fig


# =============================================================================
# Publication Figure Generator
# =============================================================================

class PublicationFigureGenerator:
    """
    Generate publication-ready figures.

    Handles:
    - Consistent styling
    - Multiple formats (PNG, PDF, SVG)
    - Subfigure composition
    - LaTeX-compatible labels
    """

    # Publication style settings
    STYLE = {
        'font.size': 12,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 16,
        'lines.linewidth': 2,
        'axes.grid': True,
        'grid.alpha': 0.3,
    }

    def __init__(self, save_dir: Optional[Path] = None):
        self.save_dir = Path(save_dir) if save_dir else Path("./publication_outputs/figures")
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Apply style
        plt.rcParams.update(self.STYLE)

    def generate_main_figure(
        self,
        physics_metrics: PhysicsMetrics,
        training_tracker: TrainingDynamicsTracker,
        ablation_study: AblationStudy,
        save_name: str = "figure_1_main",
    ) -> plt.Figure:
        """
        Generate main results figure (Figure 1 in paper).

        Layout:
        (a) Training curves (loss, PPL)
        (b) Mode comparison bar chart
        (c) Energy conservation
        (d) Mass term ablation heatmap
        """
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

        # (a) Training curves
        ax1 = fig.add_subplot(gs[0, 0])
        if training_tracker.history:
            steps = [s.step for s in training_tracker.history]
            ax1.plot(steps, [s.train_bpc for s in training_tracker.history],
                    'b-', label='Train BPC', alpha=0.7)
            val_steps = [s.step for s in training_tracker.history if s.val_bpc]
            val_bpcs = [s.val_bpc for s in training_tracker.history if s.val_bpc]
            if val_bpcs:
                ax1.plot(val_steps, val_bpcs, 'r-', label='Val BPC', linewidth=2)
        ax1.set_xlabel('Training Step')
        ax1.set_ylabel('Bits per Character')
        ax1.set_title('(a) Training Convergence')
        ax1.legend()
        ax1.text(-0.1, 1.05, 'a', transform=ax1.transAxes, fontsize=16, fontweight='bold')

        # (b) Mode comparison
        ax2 = fig.add_subplot(gs[0, 1])
        if ablation_study.results:
            mode_results = ablation_study.results[:4]  # First 4 = mode comparison
            names = [r.config.name.replace('_', '\n') for r in mode_results]
            bpcs = [r.final_val_bpc for r in mode_results]
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(names)]
            ax2.bar(range(len(names)), bpcs, color=colors, edgecolor='black')
            ax2.set_xticks(range(len(names)))
            ax2.set_xticklabels(names, fontsize=8)
        ax2.set_ylabel('Validation BPC')
        ax2.set_title('(b) Mode Comparison')
        ax2.text(-0.1, 1.05, 'b', transform=ax2.transAxes, fontsize=16, fontweight='bold')

        # (c) Energy conservation
        ax3 = fig.add_subplot(gs[1, 0])
        if physics_metrics.history:
            steps = [s.step for s in physics_metrics.history]
            delta_H = [s.delta_H for s in physics_metrics.history]
            ax3.semilogy(steps, delta_H, 'b-', alpha=0.7)
            ax3.axhline(y=np.mean(delta_H), color='r', linestyle='--',
                       label=f'Mean: {np.mean(delta_H):.2e}')
        ax3.set_xlabel('Training Step')
        ax3.set_ylabel('|ΔH| (log scale)')
        ax3.set_title('(c) Hamiltonian Conservation')
        ax3.legend()
        ax3.text(-0.1, 1.05, 'c', transform=ax3.transAxes, fontsize=16, fontweight='bold')

        # (d) Potential energy decomposition
        ax4 = fig.add_subplot(gs[1, 1])
        if physics_metrics.history:
            steps = [s.step for s in physics_metrics.history]
            ax4.plot(steps, [s.V_self for s in physics_metrics.history],
                    label='V_self (KL)', alpha=0.7)
            ax4.plot(steps, [s.V_align for s in physics_metrics.history],
                    label='V_align (belief)', alpha=0.7)
            ax4.plot(steps, [s.V_ce for s in physics_metrics.history],
                    label='V_ce (prediction)', alpha=0.7)
        ax4.set_xlabel('Training Step')
        ax4.set_ylabel('Potential Energy')
        ax4.set_title('(d) Free Energy Components')
        ax4.legend()
        ax4.text(-0.1, 1.05, 'd', transform=ax4.transAxes, fontsize=16, fontweight='bold')

        plt.suptitle('Gauge-Theoretic Hamiltonian Transformer: Main Results',
                    fontsize=16, y=1.02)

        # Save in multiple formats
        for ext in ['png', 'pdf']:
            fig.savefig(self.save_dir / f"{save_name}.{ext}",
                       dpi=300 if ext == 'png' else None,
                       bbox_inches='tight')

        return fig

    def generate_physics_figure(
        self,
        physics_metrics: PhysicsMetrics,
        dt_scaling_results: Optional[Dict[str, List[float]]] = None,
        save_name: str = "figure_2_physics",
    ) -> plt.Figure:
        """
        Generate physics validation figure (Figure 2 in paper).

        Layout:
        (a) Energy conservation over training
        (b) dt scaling (O(dt²) verification)
        (c) Hamiltonian distribution
        (d) SPD eigenvalue evolution
        """
        fig = plt.figure(figsize=(14, 10))
        gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

        # (a) Energy conservation
        ax1 = fig.add_subplot(gs[0, 0])
        if physics_metrics.history:
            steps = [s.step for s in physics_metrics.history]
            H_init = [s.H_init for s in physics_metrics.history]
            H_final = [s.H_final for s in physics_metrics.history]
            ax1.plot(steps, H_init, 'b-', label='H_init', alpha=0.7)
            ax1.plot(steps, H_final, 'r-', label='H_final', alpha=0.7)
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Hamiltonian')
        ax1.set_title('(a) Hamiltonian Values')
        ax1.legend()

        # (b) dt scaling
        ax2 = fig.add_subplot(gs[0, 1])
        if dt_scaling_results:
            dt = np.array(dt_scaling_results['dt'])
            delta_H = np.array(dt_scaling_results['delta_H'])
            ax2.loglog(dt, delta_H, 'bo-', markersize=8, label='Measured')
            # O(dt²) reference
            scale = delta_H[len(dt)//2] / (dt[len(dt)//2]**2)
            dt_ref = np.logspace(np.log10(dt.min()), np.log10(dt.max()), 50)
            ax2.loglog(dt_ref, scale * dt_ref**2, 'g--', label='O(dt²)')
        ax2.set_xlabel('Time step dt')
        ax2.set_ylabel('|ΔH|')
        ax2.set_title('(b) Symplectic Scaling')
        ax2.legend()

        # (c) ΔH distribution
        ax3 = fig.add_subplot(gs[1, 0])
        if physics_metrics.history:
            delta_H = [s.delta_H for s in physics_metrics.history]
            ax3.hist(delta_H, bins=50, edgecolor='black', alpha=0.7)
            ax3.axvline(x=np.mean(delta_H), color='r', linestyle='--',
                       label=f'Mean: {np.mean(delta_H):.2e}')
        ax3.set_xlabel('|ΔH|')
        ax3.set_ylabel('Count')
        ax3.set_title('(c) Energy Error Distribution')
        ax3.legend()

        # (d) SPD eigenvalues
        ax4 = fig.add_subplot(gs[1, 1])
        if physics_metrics.history:
            steps = [s.step for s in physics_metrics.history]
            spd_min = [s.spd_eigenvalue_min for s in physics_metrics.history]
            ax4.semilogy(steps, spd_min, 'g-', alpha=0.7)
            ax4.axhline(y=0, color='r', linestyle='--', label='Zero (violation)')
        ax4.set_xlabel('Step')
        ax4.set_ylabel('Min Eigenvalue (log)')
        ax4.set_title('(d) SPD Manifold Preservation')
        ax4.legend()

        plt.suptitle('Physics Validation: Symplectic Structure', fontsize=16, y=1.02)

        for ext in ['png', 'pdf']:
            fig.savefig(self.save_dir / f"{save_name}.{ext}",
                       dpi=300 if ext == 'png' else None,
                       bbox_inches='tight')

        return fig


# =============================================================================
# Main Publication Metrics Coordinator
# =============================================================================

class PublicationMetrics:
    """
    Main coordinator for all publication metrics.

    Combines:
    - PhysicsMetrics
    - TrainingDynamicsTracker
    - AblationStudy
    - InterpretabilityMetrics
    - PublicationFigureGenerator
    """

    def __init__(self, experiment_name: str, base_dir: Optional[Path] = None):
        self.experiment_name = experiment_name
        self.base_dir = Path(base_dir) if base_dir else Path("./publication_outputs")
        self.experiment_dir = self.base_dir / experiment_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        # Initialize all trackers
        self.physics = PhysicsMetrics(self.experiment_dir / "physics")
        self.training = TrainingDynamicsTracker(self.experiment_dir / "training")
        self.ablation = AblationStudy(self.experiment_dir / "ablation")
        self.interpretability = InterpretabilityMetrics(self.experiment_dir / "interpretability")
        self.figures = PublicationFigureGenerator(self.experiment_dir / "figures")

        # Metadata
        self.metadata = {
            'experiment_name': experiment_name,
            'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
            'base_dir': str(self.base_dir),
        }

        print(f"📊 Publication metrics initialized: {self.experiment_dir}")

    def record_training_step(
        self,
        step: int,
        epoch: float,
        train_metrics: Dict[str, float],
        diagnostics: Optional[Dict[str, Any]] = None,
        grad_norms: Optional[Dict[str, float]] = None,
        lrs: Optional[Dict[str, float]] = None,
        step_time: float = 0.0,
        batch_size: int = 1,
        seq_len: int = 1,
        Sigma: Optional[torch.Tensor] = None,
        M: Optional[torch.Tensor] = None,
    ):
        """Record all metrics for a training step."""
        # Training dynamics
        self.training.record_step(
            step, epoch, train_metrics, grad_norms, lrs,
            step_time, batch_size, seq_len
        )

        # Physics (if Hamiltonian mode)
        if diagnostics:
            self.physics.record(step, diagnostics, Sigma, M)

    def record_validation(self, step: int, val_metrics: Dict[str, float]):
        """Record validation metrics."""
        self.training.record_validation(step, val_metrics)

    def save_all(self):
        """Save all metrics."""
        self.physics.save()
        self.training.save()
        self.training.save_csv()
        self.ablation.save()

        # Save metadata
        self.metadata['saved_at'] = time.strftime('%Y-%m-%d %H:%M:%S')
        with open(self.experiment_dir / 'metadata.json', 'w') as f:
            json.dump(self.metadata, f, indent=2)

        print(f"💾 All metrics saved to {self.experiment_dir}")

    def generate_all_figures(self, dt_scaling_results: Optional[Dict] = None):
        """Generate all publication figures."""
        # Training curves
        self.training.plot_training_curves(
            save_name=f"{self.experiment_name}_training"
        )

        # Physics
        self.physics.plot_energy_conservation(
            save_name=f"{self.experiment_name}_energy"
        )

        if dt_scaling_results:
            self.physics.plot_dt_scaling(
                dt_scaling_results,
                save_name=f"{self.experiment_name}_dt_scaling"
            )

        # Main figure
        self.figures.generate_main_figure(
            self.physics, self.training, self.ablation,
            save_name=f"{self.experiment_name}_main"
        )

        # Physics figure
        self.figures.generate_physics_figure(
            self.physics, dt_scaling_results,
            save_name=f"{self.experiment_name}_physics"
        )

        print(f"📈 All figures saved to {self.experiment_dir}/figures/")

    def print_summary(self):
        """Print summary of all metrics."""
        print("\n" + "=" * 70)
        print(f"EXPERIMENT SUMMARY: {self.experiment_name}")
        print("=" * 70)

        # Training summary
        train_summary = self.training.get_summary()
        if train_summary:
            print("\n📈 Training:")
            print(f"   Steps: {train_summary.get('total_steps', 0)}")
            print(f"   Final Train BPC: {train_summary.get('final_train_bpc', 0):.4f}")
            print(f"   Best Val BPC: {train_summary.get('best_val_ppl', 'N/A')}")
            print(f"   Throughput: {train_summary.get('avg_tokens_per_sec', 0):.0f} tok/s")

        # Physics summary
        physics_summary = self.physics.get_summary()
        if physics_summary:
            print("\n⚛️ Physics:")
            print(f"   Mean |ΔH|: {physics_summary.get('delta_H_mean', 0):.2e}")
            print(f"   Max |ΔH|: {physics_summary.get('delta_H_max', 0):.2e}")
            print(f"   SPD violations: {physics_summary.get('spd_violations', 0)}")

        # Ablation summary
        if self.ablation.results:
            print("\n🔬 Ablation:")
            print(self.ablation.get_summary_table())

        print("=" * 70)


# =============================================================================
# Demo
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("PUBLICATION METRICS FRAMEWORK - DEMO")
    print("=" * 70)

    # Create metrics tracker
    metrics = PublicationMetrics("demo_experiment")

    # Simulate some training data
    for step in range(100):
        train_metrics = {
            'loss': 5.0 - step * 0.03 + np.random.randn() * 0.1,
            'ce_loss': 4.5 - step * 0.025 + np.random.randn() * 0.1,
        }

        diagnostics = {
            'H_init': 10.0 + np.random.randn() * 0.1,
            'H_final': 10.0 + np.random.randn() * 0.1,
            'delta_H': abs(np.random.randn() * 0.01),
            'T_init': 5.0,
            'T_final': 5.0,
            'V_init': 5.0,
            'V_final': 5.0,
            'V_self': 2.0 - step * 0.01,
            'V_align': 1.5 - step * 0.005,
            'V_ce': 1.5 - step * 0.01,
        }

        grad_norms = {'total': 1.0 / (step + 1), 'mu': 0.5 / (step + 1), 'ffn': 0.3 / (step + 1)}
        lrs = {'mu_embed': 0.01, 'sigma_embed': 0.005, 'phi_embed': 0.01, 'ffn': 0.01}

        metrics.record_training_step(
            step=step,
            epoch=step / 50,
            train_metrics=train_metrics,
            diagnostics=diagnostics,
            grad_norms=grad_norms,
            lrs=lrs,
            step_time=0.1,
            batch_size=8,
            seq_len=32,
        )

        if step % 10 == 0:
            val_metrics = {'loss': 4.8 - step * 0.02, 'ce_loss': 4.3 - step * 0.02}
            metrics.record_validation(step, val_metrics)

    # Add some ablation results
    for i, name in enumerate(['learned', 'vfe', 'hamiltonian_prior', 'hamiltonian_full']):
        config = AblationConfig(name=name, ffn_mode=name.split('_')[0])
        result = AblationResult(
            config=config,
            final_val_loss=2.5 - i * 0.1,
            final_val_ppl=12.0 - i * 0.5,
            final_val_bpc=3.6 - i * 0.1,
            best_val_loss=2.4 - i * 0.1,
            best_val_ppl=11.0 - i * 0.5,
            convergence_step=80 - i * 5,
            avg_delta_H=0.01 * (i + 1),
            avg_step_time=0.1 * (i + 1),
            total_time=100 * (i + 1),
        )
        metrics.ablation.add_result(result)

    # Save and generate figures
    metrics.save_all()
    metrics.generate_all_figures()
    metrics.print_summary()

    print("\n✓ Demo complete! Check ./publication_outputs/demo_experiment/")
