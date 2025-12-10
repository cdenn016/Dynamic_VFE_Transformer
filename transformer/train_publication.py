"""
Publication Proof-of-Principle Training Script
===============================================

Language modeling on WikiText-2 with byte-level encoding for minimal publishable claim.

Demonstrates:
1. Variational FFN works - inference comparable to learned MLP
2. Architecture is trainable - converges to reasonable performance
3. Theoretical framework is sound - gauge-invariant inference holds
4. Hamiltonian dynamics - energy-conserving symplectic integration (NEW!)

Five FFN Modes for Ablation Study:
    - learned: Standard MLP baseline (GELU activation)
    - variational_approx: First-order active inference (O(N²K), legacy)
    - variational_full: Complete gauge-invariant with second-order terms (O(N³K), legacy)
    - variational_gradient_engine: Full active inference via gradient_engine.py
    - hamiltonian: Symplectic Hamiltonian dynamics on belief space (NEW!)
      * Energy-conserving leapfrog integration
      * Full faithful SPD geometry with curvature corrections
      * NO learned weights - pure physics!

Comprehensive Metrics Tracking:
    - Free energy components (α, β, γ terms)
    - Gradient norms (total, μ, FFN)
    - All learning rates (μ, σ, φ, FFN)
    - Bits-per-character (BPC)
    - Attention statistics (β_mean, KL_mean)
    - Performance (step time, tokens/sec)
    - Hamiltonian diagnostics (H_init, H_final, ΔH) for hamiltonian mode

Output Files:
    - checkpoints_publication/ffn_{mode}/metrics.csv - comprehensive training metrics
    - checkpoints_publication/ffn_{mode}/best_model.pt - best model checkpoint
    - checkpoints_publication/result_{mode}.json - final summary (if single mode)
    - checkpoints_publication/ablation_results.json - comparison (if --run_ablation)

Usage:
    # Just click Run (edit defaults below)
    python transformer/train_publication.py

    # Or use command-line args:
    python transformer/train_publication.py --ffn_mode learned
    python transformer/train_publication.py --ffn_mode hamiltonian

Author: Designed for minimal publishable claim
Date: December 2025
"""

import torch
import argparse
import json
import csv
import time
import math
import subprocess
import platform
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any


from transformer.model import GaugeTransformerLM
from transformer.data import create_dataloaders
from transformer.train import compute_free_energy_loss
from transformer.train_fast import FastTrainer, FastTrainingConfig
from transformer.publication_metrics import PublicationMetrics, AblationConfig, AblationResult


def get_git_info() -> Dict[str, str]:
    """Get current git commit info."""
    try:
        commit = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            stderr=subprocess.DEVNULL
        ).decode('utf-8').strip()

        branch = subprocess.check_output(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
            stderr=subprocess.DEVNULL
        ).decode('utf-8').strip()

        # Check for uncommitted changes
        status = subprocess.check_output(
            ['git', 'status', '--porcelain'],
            stderr=subprocess.DEVNULL
        ).decode('utf-8').strip()
        dirty = len(status) > 0

        return {
            'commit': commit,
            'branch': branch,
            'dirty': dirty,
        }
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {'commit': 'unknown', 'branch': 'unknown', 'dirty': False}


def get_system_info() -> Dict[str, Any]:
    """Get system/hardware information."""
    info = {
        'platform': platform.platform(),
        'python_version': platform.python_version(),
        'torch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
    }

    if torch.cuda.is_available():
        info['cuda_version'] = torch.version.cuda
        info['gpu_name'] = torch.cuda.get_device_name(0)
        info['gpu_count'] = torch.cuda.device_count()
        info['gpu_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / 1e9

    return info


def save_experiment_config(
    config: Dict[str, Any],
    ffn_mode: str,
    checkpoint_dir: Path,
    args: argparse.Namespace = None,
) -> Path:
    """
    Save complete experiment configuration to JSON.

    Args:
        config: Model/training configuration dictionary
        ffn_mode: FFN mode being used
        checkpoint_dir: Directory to save config
        args: Command-line arguments (if available)

    Returns:
        Path to saved config file
    """
    experiment_config = {
        # Metadata
        'experiment_id': datetime.now().strftime('%Y%m%d_%H%M%S'),
        'timestamp': datetime.now().isoformat(),
        'ffn_mode': ffn_mode,

        # Full model/training config
        'config': config,

        # Command-line args (if available)
        'args': vars(args) if args else None,

        # Git info for reproducibility
        'git': get_git_info(),

        # System info
        'system': get_system_info(),
    }

    # Save to checkpoint directory
    config_path = checkpoint_dir / 'experiment_config.json'
    config_path.parent.mkdir(parents=True, exist_ok=True)

    with open(config_path, 'w') as f:
        json.dump(experiment_config, f, indent=2, default=str)

    print(f"📋 Saved experiment config: {config_path}")

    return config_path


# ============================================================================
# EDIT THESE DEFAULTS TO RUN WITHOUT COMMAND-LINE ARGS
# ============================================================================
DEFAULT_FFN_MODE = 'hamiltonian'  # 'learned', 'variational_gradient_engine', 'hamiltonian', or None
DEFAULT_RUN_ABLATION = False  # Set True to run all three modes
DEFAULT_ENABLE_SIGMA_PHI = True   # Set True to enable learning Σ and φ (required for hamiltonian!)
DEFAULT_USE_GPU_OPTIMIZED = True  # Set True for RTX 5090 / high-end GPU settings
# ============================================================================



# =============================================================================
# GPU-OPTIMIZED CONFIG (RTX 5090 / 32GB VRAM)
# =============================================================================
# MEMORY REALITY CHECK:
#   Gauge transformer has O(N² × K²) memory for attention KL matrices!
#   Standard transformer: O(N² × d) for attention
#   Ours: O(N² × K²) because KL divergence uses full covariance matrices
#
#   Memory for KL computation: B × N × N × K² × 4 bytes (FP32)
#   Example: B=32, N=256, K=127 → 32 × 256 × 256 × 127² × 4 = ~134GB (!)
#
#   Realistic for 32GB: B=16, N=64, K=63 → ~2GB for KL matrices
#
GPU_OPTIMIZED_CONFIG = {
    # Model architecture - WITH diagonal_covariance=True, can scale up!
    # Diagonal mode: O(N²×K) memory instead of O(N²×K²)
    'vocab_size': 5000,       # BPE subwords (top 5K from GPT-2 tokenizer)
    'embed_dim': 255,         # K=255 (ODD for SO(3)) - close to Vaswani d=512!
    'n_layers': 6,            # Match Vaswani base
    'hidden_dim': 1020,       # 4×embed_dim
    'max_seq_len': 128,       # N=128 - reasonable context

    # GPU Training - diagonal mode fits easily in 32GB
    'batch_size': 32,         # Can push this higher with diagonal
    'use_amp': False,         # Disabled - Hamiltonian dynamics needs FP32 precision
    'num_workers': 4,         # Parallel data loading

    # Gauge transformer parameters
    'kappa_beta': 1,
    'epsilon': 1e-8,
    'pos_encoding_mode': 'learned',
    'evolve_sigma': True,     # Full geometric learning
    'evolve_phi': True,       # Full geometric learning
    'tie_embeddings': True,

    # =========================================================================
    # DIAGONAL COVARIANCE MODE (memory optimization)
    # True:  Σ is (B,N,K) diagonal - O(N²×K) memory - can scale to Vaswani size!
    # False: Σ is (B,N,K,K) full   - O(N²×K²) memory - limited to small K,N
    # Diagonal loses off-diagonal correlations but keeps per-dim uncertainty.
    # =========================================================================
    'diagonal_covariance': True,

    # Attention pattern
    'attention_pattern': 'full',
    'attention_window': 128,
    'attention_global_tokens': 0,

    # Variational FFN parameters
    'ffn_mode': 'hamiltonian',
    'ffn_alpha': 0.2,
    'ffn_tau_eff': 1.0,
    'ffn_kappa': 1.0,
    'ffn_n_iterations': 1,
    'ffn_learnable_lr': True,
    'ffn_pattern': 'full',
    'ffn_window': 128,

    # Hamiltonian FFN parameters
    # =========================================================================
    # SPEED vs PHYSICS FIDELITY TRADEOFF:
    #   n_steps=2  → Fast (~0.1-0.3s/step), good for development
    #   n_steps=10 → Moderate (~1-2s/step), reasonable physics
    #   n_steps=25 → Slow (~3-5s/step), high physics fidelity
    # The leapfrog loop is sequential - GPU can't parallelize it!
    # For full GPU utilization: use ffn_mode='learned' instead
    # =========================================================================
    'ffn_hamiltonian_dt': 0.05,           # Larger dt works with fewer steps
    'ffn_hamiltonian_n_steps': 2,         # TOGGLE THIS for speed vs physics!
    'ffn_hamiltonian_momentum_scale': 0.5,
    'ffn_hamiltonian_gamma': 0.0,
    'ffn_hamiltonian_mass_use_prior': True,
    'ffn_hamiltonian_mass_use_observation': False,
    'ffn_hamiltonian_mass_use_incoming_social': False,
    'ffn_hamiltonian_mass_use_outgoing_recoil': False,
    'ffn_hamiltonian_evolve_mass': False,
    'gauge_fixed_priors': False,

    # Training (scaled for GPU)
    'max_steps': 500,         # More steps for convergence

    # Learning rates (same natural gradient rates)
    'mu_lr': 0.25,
    'sigma_lr': 0.05,
    'phi_lr': 0.1,
    'ffn_lr': 0.25,
    'warmup_steps': 20,

    # Free energy weights
    'alpha': 0.2,
    'beta': 1,
    'lambda_gamma': 1,
    'kappa_gamma': 1.0,

    # Regularization
    'weight_decay': 0.01,
    'dropout': 0.1,
    'grad_clip': 1.0,

    # Logging (less frequent for speed)
    'log_interval': 10,
    'eval_interval': 50,
    'checkpoint_interval': 200,
    'patience': 5,

    # Irrep structure (for K=255)
    # 75×1 + 30×3 + 18×5 = 75 + 90 + 90 = 255 ✓
    'irrep_spec': [
        ('ℓ0', 75, 1),   # 75 dimensions (scalars)
        ('ℓ1', 30, 3),   # 90 dimensions (vectors)
        ('ℓ2', 18, 5),   # 90 dimensions (rank-2 tensors)
    ],
}

# =============================================================================
# ORIGINAL PUBLICATION CONFIG (CPU/low-end GPU)
# =============================================================================
PUBLICATION_CONFIG = {
    # Model architecture (minimal but meaningful)
    'vocab_size': 200,        # Byte-level vocab (up to 256). Set 100-256 for experiments.
    'embed_dim': 21,          # K=21 (ODD - required for SO(3) irreps!)
    'n_layers': 3,            # Depth for non-trivial learning
    'hidden_dim': 84,         # 4×embed_dim
    'max_seq_len': 32,        # N=32 (key: enough for patterns!)

    # Gauge transformer parameters
    'kappa_beta': 1,
    'epsilon': 1e-8,
    'pos_encoding_mode': 'learned',
    'evolve_sigma': False,  # Auto-enabled for variational_gradient_engine mode
    'evolve_phi': False,    # Keep simple for publication
    'tie_embeddings': True,

    # Attention pattern (full for small N=32)
    'attention_pattern': 'full',
    'attention_window': 32,
    'attention_global_tokens': 0,

    # Variational FFN parameters (will be varied in ablation study)
    'ffn_mode': 'variational_gradient_engine',        # Default: will be overridden in ablation
    'ffn_alpha': 0.2,             # Prior weight (balanced)
    'ffn_tau_eff': 1.0,           # Temperature
    'ffn_kappa': 1.0,             # Softmax temperature
    'ffn_n_iterations': 1,        # Single inference step per forward pass
    'ffn_learnable_lr': True,     # Learn step size for variational descent

    # Sparse variational inference (full for N=32)
    'ffn_pattern': 'full',
    'ffn_window': 32,

    # Hamiltonian FFN parameters
    'ffn_hamiltonian_dt': 0.01,           # Leapfrog time step
    'ffn_hamiltonian_n_steps': 10,        # Integration steps per forward pass
    'ffn_hamiltonian_momentum_scale': 0.5, # Initial momentum scale
    'ffn_hamiltonian_gamma': 0.0,         # Damping (0 = pure Hamiltonian, >0 = Langevin-like)

    # Hamiltonian Mass Configuration (from "The Inertia of Belief" paper, Eq. 20)
    # M_i = Λ_{pi} + Λ_{oi} + Σ_k β_{ik} Λ̃_{qk} + Σ_j β_{ji} Λ_{qi}
    # Each term can be toggled independently for ablation studies
    'ffn_hamiltonian_mass_use_prior': True,           # Λ_p: Prior precision (default: True)
    'ffn_hamiltonian_mass_use_observation': False,    # Λ_o: Observation precision (sensory grounding)
    'ffn_hamiltonian_mass_use_incoming_social': False, # Σβ_{ik}Λ̃_{qk}: Being pulled toward neighbors
    'ffn_hamiltonian_mass_use_outgoing_recoil': False, # Σβ_{ji}Λ_{qi}: Newton's 3rd law recoil
    'ffn_hamiltonian_evolve_mass': False,             # Recompute M at each leapfrog step (full theory)

    # Gauge-Fixed Priors (for restoring gauge covariance)
    # When enabled, priors are p_i = R_i ▷ p_0 where R_i = exp(φ_i · T)
    # This guarantees p_i = Ω_ij[p_j], making prior-anchoring gauge covariant
    'gauge_fixed_priors': False,                      # Use SO(3)-rotated base prior

    # Training (optimized for convergence)
    'batch_size': 8,             # Larger batches for stability
    'max_steps': 50,              # Adjusted for ~2 hour runtime

    # Natural gradient learning rates (balanced for fast convergence)
    'mu_lr':    0.25,                # Belief means
    'sigma_lr': 0.05,            # Belief covariances
    'phi_lr':   0.1,               # Gauge transformations
    'ffn_lr':   0.25,              # FFN parameters (if learned mode)

    'warmup_steps': 4,          # Gradual warmup for stability

    # Free energy weights (balanced gauge-theoretic learning)
    'alpha': 0.2,                # Self-consistency regularization
    'beta': 1,                  # Belief alignment (key gauge term)
    'lambda_gamma': 1,          # Model alignment (disabled)
    'kappa_gamma': 1.0,         # Temperature for γ_ij coupling

    # Regularization (light for small model)
    'weight_decay': 0.01,
    'dropout': 0.1,
    'grad_clip': 0.0,

    # Logging (frequent for publication plots)
    'log_interval': 1,
    'eval_interval': 2,          # Eval every 50 steps
    'checkpoint_interval': 15,
    'patience': 3,               # Early stopping patience

    # Irrep structure (for K=11)
    'irrep_spec': [
        ('ℓ0', 5, 1),    # 5 dimensions (scalars)
        ('ℓ1', 2, 3),    # 6 dimensions (vectors)
        ('ℓ2', 1, 5),    # 11 dimensions (tensors)
        # Total: 5 + 6 + 11= 21 ✓
    ],
}


class PublicationMetricsTracker:
    """Track ALL metrics needed for publication."""

    def __init__(self, save_path: Path):
        self.save_path = save_path
        self.history = []

        # Create CSV with comprehensive headers
        self.save_path.parent.mkdir(parents=True, exist_ok=True)

        self.headers = [
            # Core
            'step', 'timestamp',

            # Losses
            'train_loss_total', 'train_loss_ce', 'train_loss_belief_align',
            'train_loss_self_consistency', 'train_loss_model_align',
            'val_loss', 'val_ce',

            # Metrics
            'train_ppl', 'train_bpc', 'val_ppl', 'val_bpc',

            # Attention stats
            'beta_mean', 'beta_std', 'kl_mean', 'kl_std',

            # Learning rates
            'mu_lr', 'sigma_lr', 'phi_lr', 'ffn_lr',

            # Gradient norms
            'grad_norm_total', 'grad_norm_mu', 'grad_norm_ffn',

            # Performance
            'step_time', 'tokens_per_sec',
        ]

        with open(self.save_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(self.headers)

    def log_step(self, step: int, metrics: Dict, lrs: Dict, grad_norms: Dict,
                 step_time: float, batch_size: int, seq_len: int):
        """Log training step with full metrics."""

        # Compute tokens/sec
        tokens_per_sec = (batch_size * seq_len) / step_time if step_time > 0 else 0

        # Bits per character (convert from nats)
        train_bpc = metrics.get('train_loss_ce', 0) / math.log(2)

        entry = {
            'step': step,
            'timestamp': time.time(),

            # Losses
            'train_loss_total': metrics.get('train_loss_total'),
            'train_loss_ce': metrics.get('train_loss_ce'),
            'train_loss_belief_align': metrics.get('train_loss_belief_align', 0),
            'train_loss_self_consistency': metrics.get('train_loss_self_consistency', 0),
            'train_loss_model_align': metrics.get('train_loss_model_align', 0),
            'val_loss': None,
            'val_ce': None,

            # Metrics
            'train_ppl': metrics.get('train_ppl'),
            'train_bpc': train_bpc,
            'val_ppl': None,
            'val_bpc': None,

            # Attention
            'beta_mean': metrics.get('beta_mean'),
            'beta_std': metrics.get('beta_std'),
            'kl_mean': metrics.get('kl_mean'),
            'kl_std': metrics.get('kl_std'),

            # Learning rates
            'mu_lr': lrs.get('mu_embed', 0),
            'sigma_lr': lrs.get('sigma_embed', 0),
            'phi_lr': lrs.get('phi_embed', 0),
            'ffn_lr': lrs.get('ffn', 0),

            # Gradients
            'grad_norm_total': grad_norms.get('total', 0),
            'grad_norm_mu': grad_norms.get('mu', 0),
            'grad_norm_ffn': grad_norms.get('ffn', 0),

            # Performance
            'step_time': step_time,
            'tokens_per_sec': tokens_per_sec,
        }

        self.history.append(entry)

    def log_val(self, step: int, val_metrics: Dict):
        """Update entry with validation metrics."""
        for entry in reversed(self.history):
            if entry['step'] == step:
                entry['val_loss'] = val_metrics.get('loss')
                entry['val_ce'] = val_metrics.get('ce_loss', val_metrics.get('loss'))
                entry['val_ppl'] = val_metrics.get('perplexity')
                entry['val_bpc'] = entry['val_ce'] / math.log(2) if entry['val_ce'] else None
                break

    def save(self):
        """Save to CSV."""
        if not self.history:
            return

        with open(self.save_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.headers)
            writer.writeheader()
            writer.writerows(self.history)


class PublicationTrainer(FastTrainer):
    """Enhanced trainer with publication-quality metrics."""

    def __init__(self, *args, publication_metrics: PublicationMetrics = None, **kwargs):
        super().__init__(*args, **kwargs)

        # Basic CSV metrics tracker
        metrics_path = self.config.checkpoint_dir / 'metrics.csv'
        self.metrics_tracker = PublicationMetricsTracker(metrics_path)
        print(f"📊 Logging publication metrics to: {metrics_path}")

        # Comprehensive publication metrics (optional)
        self.pub_metrics = publication_metrics
        if self.pub_metrics:
            print(f"📈 Comprehensive metrics enabled: {self.pub_metrics.experiment_dir}")

    def train_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Train step with comprehensive metrics and AMP support."""
        self.model.train()

        input_ids, target_ids = batch
        input_ids = input_ids.to(self.device)
        target_ids = target_ids.to(self.device)

        # Forward pass with full metrics (with optional AMP)
        if self.scaler is not None:
            # Mixed precision forward pass
            with torch.amp.autocast('cuda'):
                loss, full_metrics = compute_free_energy_loss(
                    self.model,
                    input_ids,
                    target_ids,
                    alpha=self.config.alpha,
                    lambda_beta=self.config.beta,
                    lambda_gamma=self.config.lambda_gamma,
                    kappa_gamma=self.config.kappa_gamma,
                )
            # Scaled backward
            self.scaler.scale(loss).backward()
        else:
            # Standard forward pass
            loss, full_metrics = compute_free_energy_loss(
                self.model,
                input_ids,
                target_ids,
                alpha=self.config.alpha,
                lambda_beta=self.config.beta,
                lambda_gamma=self.config.lambda_gamma,
                kappa_gamma=self.config.kappa_gamma,
            )
            loss.backward()

        # Compute gradient norms BEFORE clipping (only at log intervals for speed)
        # This is computed conditionally in train() loop now
        grad_norms = None

        # Clip and step (with scaler if AMP enabled)
        if self.scaler is not None:
            if self.config.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.grad_clip,
                )
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            if self.config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.grad_clip,
                )
            self.optimizer.step()

        if self.scheduler is not None:
            self.scheduler.step()
        self.optimizer.zero_grad()

        # Format comprehensive metrics
        metrics = {
            'train_loss_total': full_metrics['loss/total'],
            'train_loss_ce': full_metrics['loss/ce'],
            'train_loss_belief_align': full_metrics.get('loss/belief_align', 0),
            'train_loss_self_consistency': full_metrics.get('loss/self_consistency', 0),
            'train_loss_model_align': full_metrics.get('loss/model_align', 0),
            'train_ppl': math.exp(full_metrics['loss/ce']),
            'beta_mean': full_metrics.get('attention/beta_mean', 0),
            'beta_std': 0,  # Could compute if needed
            'kl_mean': full_metrics.get('attention/kl_mean', 0),
            'kl_std': 0,
        }

        # Carry over Hamiltonian diagnostics for physics metrics
        if 'hamiltonian_diagnostics' in full_metrics:
            metrics['hamiltonian_diagnostics'] = full_metrics['hamiltonian_diagnostics']

        return metrics, grad_norms

    def _compute_gradient_norms(self) -> Dict[str, float]:
        """Compute gradient norms for different parameter groups."""
        norms = {'total': 0, 'mu': 0, 'ffn': 0}

        total_norm = 0
        mu_norm = 0
        ffn_norm = 0

        for name, param in self.model.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2).item()
                total_norm += param_norm ** 2

                if 'mu_embed' in name:
                    mu_norm += param_norm ** 2
                elif 'ffn' in name:
                    ffn_norm += param_norm ** 2

        norms['total'] = math.sqrt(total_norm)
        norms['mu'] = math.sqrt(mu_norm)
        norms['ffn'] = math.sqrt(ffn_norm)

        return norms

    def train(self):
        """Training loop with publication metrics."""
        print(f"{'='*70}")
        print("PUBLICATION-QUALITY TRAINING")
        print(f"{'='*70}\n")

        start_time = time.time()
        train_iterator = iter(self.train_loader)

        try:
            from tqdm import tqdm
            pbar = tqdm(range(self.config.max_steps), desc="Training")
            use_tqdm = True
        except ImportError:
            pbar = range(self.config.max_steps)
            use_tqdm = False

        for step in pbar:
            self.global_step = step
            step_start = time.time()

            # Get batch
            try:
                batch = next(train_iterator)
            except StopIteration:
                train_iterator = iter(self.train_loader)
                batch = next(train_iterator)

            # Train step with full metrics
            metrics, _ = self.train_step(batch)

            step_time = time.time() - step_start

            # Only compute gradient norms at log intervals (expensive for large models)
            is_log_step = (step + 1) % self.config.log_interval == 0
            grad_norms = self._compute_gradient_norms() if is_log_step else None

            # Get learning rates
            lrs = {group['name']: group['lr'] for group in self.optimizer.param_groups}

            # Log to basic tracker (only at log intervals)
            if is_log_step:
                batch_size = batch[0].shape[0]
                seq_len = batch[0].shape[1]
                self.metrics_tracker.log_step(
                    step + 1, metrics, lrs, grad_norms, step_time, batch_size, seq_len
                )

                # Log to comprehensive publication metrics (if enabled)
                if self.pub_metrics:
                    diagnostics = metrics.get('hamiltonian_diagnostics', None)
                    self.pub_metrics.record_training_step(
                        step=step + 1,
                        epoch=(step + 1) / len(self.train_loader),
                        train_metrics={
                            'loss': metrics['train_loss_total'],
                            'ce_loss': metrics['train_loss_ce'],
                        },
                        diagnostics=diagnostics,
                        grad_norms=grad_norms,
                        lrs=lrs,
                        step_time=step_time,
                        batch_size=batch_size,
                        seq_len=seq_len,
                    )

            # Console logging
            if is_log_step:
                log_msg = (
                    f"Step {step+1}/{self.config.max_steps} | "
                    f"Loss: {metrics['train_loss_total']:.4f} | "
                    f"CE: {metrics['train_loss_ce']:.4f} | "
                    f"β: {metrics['train_loss_belief_align']:.4f} | "
                    f"PPL: {metrics['train_ppl']:.1f}"
                )

                if use_tqdm:
                    pbar.set_description(log_msg)
                else:
                    print(log_msg)

            # Validation
            if (step + 1) % self.config.eval_interval == 0:
                val_metrics = self.validate()
                self.metrics_tracker.log_val(step + 1, val_metrics)

                # Log to comprehensive metrics
                if self.pub_metrics:
                    self.pub_metrics.record_validation(step + 1, val_metrics)

                print(f"\n  Validation @ step {step+1}:")
                print(f"    Loss: {val_metrics['loss']:.4f}")
                print(f"    CE: {val_metrics['ce_loss']:.4f}")
                print(f"    PPL: {val_metrics['perplexity']:.2f}")
                print(f"    BPC: {val_metrics['ce_loss']/math.log(2):.3f}\n")

                # Save best
                if val_metrics['loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['loss']
                    self.patience_counter = 0
                    self.save_checkpoint(is_best=True)
                else:
                    self.patience_counter += 1
                    if self.config.patience > 0 and self.patience_counter >= self.config.patience:
                        print(f"\n⚠ Early stopping!")
                        break

            # Checkpointing
            if (step + 1) % self.config.checkpoint_interval == 0:
                self.save_checkpoint(is_best=False)
                self.metrics_tracker.save()

        # Save final metrics
        self.metrics_tracker.save()
        print(f"\n📊 Final metrics saved to: {self.metrics_tracker.save_path}")

        # Save comprehensive publication metrics
        if self.pub_metrics:
            self.pub_metrics.save_all()
            self.pub_metrics.generate_all_figures()

            # Generate interpretability outputs using a sample batch from validation
            try:
                sample_batch = next(iter(self.val_loader))
                self.pub_metrics.generate_interpretability_outputs(
                    model=self.model,
                    sample_batch=sample_batch,
                    tokenizer=None,  # Byte-level, no tokenizer needed
                    device=self.device,
                )
            except Exception as e:
                print(f"⚠ Could not generate interpretability outputs: {e}")

            self.pub_metrics.print_summary()

        # Summary
        elapsed = time.time() - start_time
        print(f"\n{'='*70}")
        print("TRAINING COMPLETE!")
        print(f"{'='*70}")
        print(f"Time: {elapsed/3600:.2f} hours")
        print(f"Best val loss: {self.best_val_loss:.4f}")
        # Note: best_val_loss includes free energy terms, not just CE
        # PPL calculation is done in final eval using CE loss only
        print(f"{'='*70}\n")


def run_single_experiment(
    config: dict,
    ffn_mode: str,
    device: torch.device,
    checkpoint_dir: Path,
    use_wandb: bool = False,
    args: argparse.Namespace = None,
    enable_publication_metrics: bool = True,
) -> Dict:
    """
    Run a single training experiment.

    Args:
        config: Configuration dictionary
        ffn_mode: FFN mode ('learned' or 'variational_gradient_engine')
        device: Device to train on
        checkpoint_dir: Directory to save checkpoints
        use_wandb: Whether to use Weights & Biases logging
        args: Command-line arguments for logging
        enable_publication_metrics: Whether to enable comprehensive publication metrics

    Returns:
        Dictionary with final metrics
    """
    print("\n" + "="*70)
    print(f"EXPERIMENT: FFN_MODE = {ffn_mode}")
    print("="*70)

    # Override FFN mode in config
    config = config.copy()
    config['ffn_mode'] = ffn_mode

    # Create experiment-specific checkpoint directory
    exp_checkpoint_dir = checkpoint_dir / f"ffn_{ffn_mode}"
    exp_checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Save experiment configuration at the START
    save_experiment_config(config, ffn_mode, exp_checkpoint_dir, args)

    # =================================================================
    # Data Loading (BPE tokenization using GPT-2 tokenizer)
    # =================================================================

    print("\n" + "="*70)
    print("LOADING WIKITEXT-2 WITH BPE TOKENIZATION")
    print("="*70)

    train_loader, val_loader, actual_vocab_size = create_dataloaders(
        max_seq_len=config['max_seq_len'],
        batch_size=config['batch_size'],
        vocab_size=config['vocab_size'],  # Top K BPE tokens
        num_workers=config.get('num_workers', 0),
    )

    config['vocab_size'] = actual_vocab_size

    # =================================================================
    # Model Creation
    # =================================================================

    print("\n" + "="*70)
    print("CREATING MODEL")
    print("="*70)
    print(f"  FFN mode: {ffn_mode}")
    print(f"  N (seq len): {config['max_seq_len']}")
    print(f"  K (embed): {config['embed_dim']}")
    print(f"  Layers: {config['n_layers']}")
    print(f"  Vocab: {actual_vocab_size} (BPE)")

    model = GaugeTransformerLM(config)
    model = model.to(device)

    total_params = model.get_num_params(non_embedding=False)
    non_embed_params = model.get_num_params(non_embedding=True)

    print(f"\nModel Parameters:")
    print(f"  Total:         {total_params:,}")
    print(f"  Non-embedding: {non_embed_params:,}")
    print(f"  Embedding:     {total_params - non_embed_params:,}")

    # =================================================================
    # Training Configuration
    # =================================================================

    train_config = FastTrainingConfig(
        max_steps=config['max_steps'],
        warmup_steps=config['warmup_steps'],

        # Natural gradient learning rates
        mu_lr=config['mu_lr'],
        sigma_lr=config['sigma_lr'],
        phi_lr=config['phi_lr'],
        attention_lr=config['phi_lr'],
        ffn_lr=config['ffn_lr'],
        output_lr=config['ffn_lr'],

        weight_decay=config['weight_decay'],
        grad_clip=config['grad_clip'],

        alpha=config['alpha'],
        beta=config['beta'],
        lambda_gamma=config['lambda_gamma'],

        log_interval=config['log_interval'],
        eval_interval=config['eval_interval'],
        checkpoint_interval=config['checkpoint_interval'],

        use_wandb=use_wandb,
        checkpoint_dir=exp_checkpoint_dir,

        # GPU optimizations
        use_amp=config.get('use_amp', False),
    )

    print("\n" + "="*70)
    print("TRAINING CONFIGURATION")
    print("="*70)
    print(f"  Max steps:      {train_config.max_steps}")
    print(f"  Warmup:         {train_config.warmup_steps}")
    print(f"  Batch size:     {config['batch_size']}")
    print(f"  Seq length:     {config['max_seq_len']}")
    print(f"  Use AMP:        {train_config.use_amp}")
    print(f"  Num workers:    {config.get('num_workers', 0)}")
    print(f"\nFree Energy Weights:")
    print(f"  α (self-consistency): {train_config.alpha}")
    print(f"  β (belief align):     {train_config.beta}")
    print(f"  γ (model align):      {train_config.lambda_gamma}")

    # =================================================================
    # Create Trainer
    # =================================================================

    print("\n" + "="*70)
    print("INITIALIZING TRAINER")
    print("="*70)

    # Create comprehensive publication metrics tracker
    pub_metrics = None
    if enable_publication_metrics:
        experiment_name = f"{ffn_mode}_{time.strftime('%Y%m%d_%H%M%S')}"
        pub_metrics = PublicationMetrics(
            experiment_name=experiment_name,
            base_dir=exp_checkpoint_dir / "publication_outputs"
        )

    trainer = PublicationTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=train_config,
        device=device,
        publication_metrics=pub_metrics,
    )

    # =================================================================
    # Training
    # =================================================================

    print("\n" + "="*70)
    print("STARTING TRAINING")
    print("="*70)
    print(f"Device: {device}")
    print(f"FFN mode: {ffn_mode}")
    print(f"Total steps: {train_config.max_steps:,}")
    print("\nNOTE: First few batches may be slow (JIT compilation)")
    print("="*70 + "\n")

    try:
        trainer.train()

        print("\n" + "="*70)
        print("✓ TRAINING COMPLETE!")
        print("="*70)

        # Final evaluation
        final_metrics = trainer.validate()

        print(f"\nFinal Validation Metrics:")
        print(f"  Loss:       {final_metrics['loss']:.4f}")
        print(f"  Perplexity: {final_metrics['perplexity']:.2f}")

        # vs random baseline
        random_ppl = actual_vocab_size
        improvement = random_ppl / final_metrics['perplexity']
        print(f"\nImprovement over random:")
        print(f"  Random:     {random_ppl:.0f}")
        print(f"  Model:      {final_metrics['perplexity']:.2f}")
        print(f"  Factor:     {improvement:.1f}x better!")

        # Save final checkpoint
        final_ckpt = trainer.save_checkpoint(is_best=True)
        print(f"\n✓ Saved: {final_ckpt}")

        # Return metrics
        return {
            'ffn_mode': ffn_mode,
            'final_loss': final_metrics['loss'],
            'final_ppl': final_metrics['perplexity'],
            'random_ppl': random_ppl,
            'improvement': improvement,
            'total_params': total_params,
            'vocab_size': actual_vocab_size,
            'checkpoint': str(final_ckpt),
        }

    except KeyboardInterrupt:
        print("\n\n" + "="*70)
        print("TRAINING INTERRUPTED")
        print("="*70)
        ckpt = trainer.save_checkpoint(is_best=False)
        print(f"✓ Saved: {ckpt}")
        return None

    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        raise


def run_ablation_study(
    device: torch.device,
    checkpoint_dir: Path,
    use_wandb: bool = False,
    enable_sigma_phi: bool = False,
    args: argparse.Namespace = None,
) -> List[Dict]:
    """
    Run complete ablation study across all three FFN modes.

    Args:
        device: Device to train on
        checkpoint_dir: Directory to save checkpoints
        use_wandb: Whether to use Weights & Biases logging
        enable_sigma_phi: Enable learning Σ (covariances) and φ (gauge frames)
        args: Command-line arguments for logging

    Returns:
        List of result dictionaries for each FFN mode
    """
    print("\n" + "="*70)
    print("ABLATION STUDY: THREE FFN MODES")
    print("="*70)

    if enable_sigma_phi:
        print("\n🔥 FULL GEOMETRIC LEARNING ENABLED!")
        print("   Learning: μ (means), Σ (covariances), φ (gauge frames)")

    print("\nWill run:")
    print("  1. learned                     (baseline - standard MLP with GELU)")
    print("  2. variational_gradient_engine (gradient-based active inference)")
    print("  3. hamiltonian                 (symplectic dynamics - NO learned weights!)")
    print("="*70)

    modes = ['learned', 'variational_gradient_engine', 'hamiltonian']
    results = []

    for i, mode in enumerate(modes):
        print(f"\n\n{'='*70}")
        print(f"EXPERIMENT {i+1}/{len(modes)}: {mode}")
        print("="*70)

        config = PUBLICATION_CONFIG.copy()
        config['ffn_mode'] = mode

        # Gradient engine requires additional parameters
        if mode == 'variational_gradient_engine':
            config.setdefault('ffn_lambda_belief', 1.0)
            config.setdefault('ffn_lambda_prior', 0.0)
            config.setdefault('ffn_lambda_phi', 0.0)
            config.setdefault('ffn_update_sigma', True)
            config['evolve_sigma'] = True  # Enable sigma evolution for full Gaussian inference

        # Hamiltonian mode requires sigma evolution
        if mode == 'hamiltonian':
            config.setdefault('ffn_lambda_belief', 0.5)  # Moderate alignment
            config.setdefault('ffn_update_sigma', True)
            config['evolve_sigma'] = True  # Required for Hamiltonian dynamics

        # Enable full geometric learning if requested
        if enable_sigma_phi:
            config['evolve_sigma'] = True
            config['evolve_phi'] = True

        result = run_single_experiment(
            config=config,
            ffn_mode=mode,
            device=device,
            checkpoint_dir=checkpoint_dir,
            use_wandb=use_wandb,
            args=args,
        )

        if result is not None:
            results.append(result)

    # Save combined results
    results_file = checkpoint_dir / "ablation_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "="*70)
    print("ABLATION STUDY COMPLETE")
    print("="*70)

    # Print comparison
    print("\nResults Comparison:")
    print("-"*70)
    print(f"{'Mode':<20} {'PPL':>10} {'vs Random':>12} {'vs Learned':>12}")
    print("-"*70)

    learned_ppl = None
    for result in results:
        mode = result['ffn_mode']
        ppl = result['final_ppl']
        improvement = result['improvement']

        if mode == 'learned':
            learned_ppl = ppl
            vs_learned = "baseline"
        elif learned_ppl is not None:
            diff_pct = ((ppl - learned_ppl) / learned_ppl) * 100
            vs_learned = f"+{diff_pct:.1f}%"
        else:
            vs_learned = "N/A"

        print(f"{mode:<20} {ppl:>10.2f} {improvement:>11.1f}x {vs_learned:>12}")

    print("-"*70)
    print(f"\nSaved: {results_file}")

    # Check publishable claim
    print("\n" + "="*70)
    print("PUBLISHABILITY CHECK")
    print("="*70)

    if learned_ppl is not None:
        for result in results:
            if result['ffn_mode'] == 'variational_gradient_engine':
                mode = result['ffn_mode']
                ppl = result['final_ppl']
                diff_pct = ((ppl - learned_ppl) / learned_ppl) * 100

                print(f"\n{mode}:")
                print(f"  Learned PPL:     {learned_ppl:.2f}")
                print(f"  Variational PPL: {ppl:.2f}")
                print(f"  Difference:      +{diff_pct:.1f}%")

                if diff_pct < 20:
                    print(f"  ✓ Within 20% threshold - PUBLISHABLE!")
                else:
                    print(f"  ⚠ Outside 20% threshold - may need tuning")

    return results


def main():
    parser = argparse.ArgumentParser(description='Publication Proof-of-Principle Training')

    # FFN mode (uses defaults from top of file)
    parser.add_argument('--ffn_mode', type=str, default=DEFAULT_FFN_MODE,
                        choices=['learned', 'variational_gradient_engine', 'hamiltonian'],
                        help='FFN mode (or use --run_ablation for all three modes)')

    # Ablation study (uses defaults from top of file)
    parser.add_argument('--run_ablation', action='store_true', default=DEFAULT_RUN_ABLATION,
                        help='Run all four FFN modes (ablation study)')

    # Enable full geometric learning (Σ and φ)
    parser.add_argument('--enable_sigma_phi', action='store_true', default=DEFAULT_ENABLE_SIGMA_PHI,
                        help='Enable learning covariances (Σ) and gauge frames (φ) - full geometric learning!')

    # GPU optimization
    parser.add_argument('--gpu_optimized', action='store_true', default=DEFAULT_USE_GPU_OPTIMIZED,
                        help='Use GPU-optimized config (larger batch, AMP, bigger model) for RTX 5090 / high-end GPUs')
    parser.add_argument('--no_gpu_optimized', action='store_true',
                        help='Force use of original small config even on GPU')

    # System
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints_publication')
    parser.add_argument('--use_wandb', action='store_true')

    args = parser.parse_args()

    # Device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    print("="*70)
    print("PUBLICATION PROOF-OF-PRINCIPLE TRAINING")
    print("="*70)
    print(f"\nDevice: {device}")

    # Select config based on GPU optimization flag
    use_gpu_config = args.gpu_optimized and not args.no_gpu_optimized and device.type == 'cuda'
    if use_gpu_config:
        print("\n" + "="*70)
        print("🚀 GPU-OPTIMIZED MODE (RTX 5090 / High-end GPU)")
        print("="*70)
        print("   batch_size=32, embed_dim=127, seq_len=256 (Vaswani-scale)")
        print("   This will fully utilize your GPU!")
        print("="*70 + "\n")
        base_config = GPU_OPTIMIZED_CONFIG.copy()
    else:
        base_config = PUBLICATION_CONFIG.copy()

    checkpoint_dir = Path(args.checkpoint_dir)

    # Run experiments
    if args.run_ablation:
        # Run all three modes
        results = run_ablation_study(
            device=device,
            checkpoint_dir=checkpoint_dir,
            use_wandb=args.use_wandb,
            enable_sigma_phi=args.enable_sigma_phi,
            args=args,
        )

    else:
        # Run single mode
        if args.ffn_mode is None:
            print("\nError: Must specify --ffn_mode or --run_ablation")
            print("Edit DEFAULT_FFN_MODE at top of train_publication.py or use command-line args")
            return

        config = base_config.copy()
        config['ffn_mode'] = args.ffn_mode

        # Gradient engine requires additional parameters
        if args.ffn_mode == 'variational_gradient_engine':
            config.setdefault('ffn_lambda_belief', 1.0)
            config.setdefault('ffn_lambda_prior', 0.0)
            config.setdefault('ffn_lambda_phi', 0.0)
            config.setdefault('ffn_update_sigma', True)
            config['evolve_sigma'] = True  # Enable sigma evolution for full Gaussian inference

        # Hamiltonian mode requires sigma evolution
        if args.ffn_mode == 'hamiltonian':
            config.setdefault('ffn_lambda_belief', 0.5)  # Moderate alignment
            config.setdefault('ffn_update_sigma', True)
            config['evolve_sigma'] = True  # Required for Hamiltonian dynamics
            # Force enable_sigma_phi for Hamiltonian
            args.enable_sigma_phi = True

        # Enable full geometric learning if requested
        if args.enable_sigma_phi:
            print("\n" + "="*70)
            print("🔥 FULL GEOMETRIC LEARNING ENABLED!")
            print("="*70)
            print("   Learning: μ (means), Σ (covariances), φ (gauge frames)")
            print("   This tests the FULL natural gradient framework!")
            print("="*70 + "\n")
            config['evolve_sigma'] = True
            config['evolve_phi'] = True

        result = run_single_experiment(
            config=config,
            ffn_mode=args.ffn_mode,
            device=device,
            checkpoint_dir=checkpoint_dir,
            use_wandb=args.use_wandb,
            args=args,
        )

        if result is not None:
            # Save result
            result_file = checkpoint_dir / f"result_{args.ffn_mode}.json"
            result_file.parent.mkdir(parents=True, exist_ok=True)
            with open(result_file, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"\n✓ Saved result: {result_file}")

    print("\n" + "="*70)
    print("SESSION COMPLETE")
    print("="*70)


if __name__ == '__main__':
    main()