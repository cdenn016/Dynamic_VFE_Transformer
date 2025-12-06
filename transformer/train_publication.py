"""
Publication Proof-of-Principle Training Script
===============================================

Character-level language modeling on WikiText-2 for minimal publishable claim.

Demonstrates:
1. Variational FFN works - inference comparable to learned MLP
2. Architecture is trainable - converges to reasonable performance
3. Theoretical framework is sound - gauge-invariant inference holds

Four FFN Modes for Ablation Study:
    - learned: Standard MLP baseline
    - variational_approx: First-order active inference (O(N²K), legacy)
    - variational_full: Complete gauge-invariant with second-order terms (O(N³K), legacy)
    - variational_gradient_engine: Full active inference via validated gradient_engine.py (RECOMMENDED!)

Comprehensive Metrics Tracking:
    - Free energy components (α, β, γ terms)
    - Gradient norms (total, μ, FFN)
    - All learning rates (μ, σ, φ, FFN)
    - Bits-per-character (BPC)
    - Attention statistics (β_mean, KL_mean)
    - Performance (step time, tokens/sec)

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

Author: Designed for minimal publishable claim
Date: November 2025
"""

import torch
import argparse
import json
import csv
import time
import math
from pathlib import Path
from typing import Dict, List, Tuple


from transformer.model import GaugeTransformerLM
from transformer.data import create_char_dataloaders
from transformer.train import compute_free_energy_loss
from transformer.train_fast import FastTrainer, FastTrainingConfig


# ============================================================================
# EDIT THESE DEFAULTS TO RUN WITHOUT COMMAND-LINE ARGS
# ============================================================================
DEFAULT_FFN_MODE = 'variational_gradient_engine'  # 'learned', 'variational_approx', 'variational_full', 'variational_gradient_engine', or None
DEFAULT_RUN_ABLATION = False  # Set True to run all four modes
DEFAULT_ENABLE_SIGMA_PHI = False   # Set True to enable learning Σ and φ (full geometric learning!)
# ============================================================================



PUBLICATION_CONFIG = {
    # Model architecture (minimal but meaningful)
    'vocab_size': 512,        # Will be set by actual char vocab (typically ~100 chars in WikiText-2)
    'embed_dim': 21,          # K=11 (ODD - required for SO(3) irreps!)
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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Publication metrics
        metrics_path = self.config.checkpoint_dir / 'metrics.csv'
        self.metrics_tracker = PublicationMetricsTracker(metrics_path)
        print(f"📊 Logging publication metrics to: {metrics_path}")

    def train_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Train step with comprehensive metrics."""
        self.model.train()

        input_ids, target_ids = batch
        input_ids = input_ids.to(self.device)
        target_ids = target_ids.to(self.device)

        # Forward pass with full metrics
        loss, full_metrics = compute_free_energy_loss(
            self.model,
            input_ids,
            target_ids,
            alpha=self.config.alpha,
            lambda_beta=self.config.beta,
            lambda_gamma=self.config.lambda_gamma,
            kappa_gamma=self.config.kappa_gamma,
        )

        # Backward
        loss.backward()

        # Compute gradient norms BEFORE clipping
        grad_norms = self._compute_gradient_norms()

        # Clip and step
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
            metrics, grad_norms = self.train_step(batch)

            step_time = time.time() - step_start

            # Get learning rates
            lrs = {group['name']: group['lr'] for group in self.optimizer.param_groups}

            # Log to tracker
            batch_size = batch[0].shape[0]
            seq_len = batch[0].shape[1]
            self.metrics_tracker.log_step(
                step + 1, metrics, lrs, grad_norms, step_time, batch_size, seq_len
            )

            # Console logging
            if (step + 1) % self.config.log_interval == 0:
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
) -> Dict:
    """
    Run a single training experiment.

    Args:
        config: Configuration dictionary
        ffn_mode: FFN mode ('learned' or 'variational_gradient_engine')
        device: Device to train on
        checkpoint_dir: Directory to save checkpoints
        use_wandb: Whether to use Weights & Biases logging

    Returns:
        Dictionary with final metrics
    """
    print("\n" + "="*70)
    print(f"EXPERIMENT: FFN_MODE = {ffn_mode}")
    print("="*70)

    # Override FFN mode in config
    config = config.copy()
    config['ffn_mode'] = ffn_mode

    # =================================================================
    # Data Loading
    # =================================================================

    print("\n" + "="*70)
    print("LOADING CHARACTER-LEVEL DATA")
    print("="*70)

    train_loader, val_loader, actual_vocab_size = create_char_dataloaders(
        max_seq_len=config['max_seq_len'],
        batch_size=config['batch_size'],
        vocab_size=256,  # ASCII extended
        num_workers=0,
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
    print(f"  Vocab: {actual_vocab_size} chars")

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

    # Create experiment-specific checkpoint directory
    exp_checkpoint_dir = checkpoint_dir / f"ffn_{ffn_mode}"
    exp_checkpoint_dir.mkdir(parents=True, exist_ok=True)

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
    )

    print("\n" + "="*70)
    print("TRAINING CONFIGURATION")
    print("="*70)
    print(f"  Max steps:      {train_config.max_steps}")
    print(f"  Warmup:         {train_config.warmup_steps}")
    print(f"  Batch size:     {config['batch_size']}")
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

    trainer = PublicationTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=train_config,
        device=device,
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
) -> List[Dict]:
    """
    Run complete ablation study across all three FFN modes.

    Args:
        base_config: 'debug', 'standard', or 'extended'
        device: Device to train on
        checkpoint_dir: Directory to save checkpoints
        use_wandb: Whether to use Weights & Biases logging
        enable_sigma_phi: Enable learning Σ (covariances) and φ (gauge frames)

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
    print("  1. learned                   (baseline)")
    print("  2. variational_gradient_engine (full active inference via gradient_engine.py) - RECOMMENDED!")
    print("="*70)

    modes = ['learned', 'variational_gradient_engine']
    results = []

    for i, mode in enumerate(modes):
        print(f"\n\n{'='*70}")
        print(f"EXPERIMENT {i+1}/2: {mode}")
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
                        choices=['learned', 'variational_gradient_engine'],
                        help='FFN mode (or use --run_ablation for all four modes)')

    # Ablation study (uses defaults from top of file)
    parser.add_argument('--run_ablation', action='store_true', default=DEFAULT_RUN_ABLATION,
                        help='Run all four FFN modes (ablation study)')

    # Enable full geometric learning (Σ and φ)
    parser.add_argument('--enable_sigma_phi', action='store_true', default=DEFAULT_ENABLE_SIGMA_PHI,
                        help='Enable learning covariances (Σ) and gauge frames (φ) - full geometric learning!')

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

    checkpoint_dir = Path(args.checkpoint_dir)

    # Run experiments
    if args.run_ablation:
        # Run all three modes
        results = run_ablation_study(
            device=device,
            checkpoint_dir=checkpoint_dir,
            use_wandb=args.use_wandb,
            enable_sigma_phi=args.enable_sigma_phi,
        )

    else:
        # Run single mode
        if args.ffn_mode is None:
            print("\nError: Must specify --ffn_mode or --run_ablation")
            print("Edit DEFAULT_FFN_MODE at top of train_publication.py or use command-line args")
            return

        config = PUBLICATION_CONFIG.copy()
        config['ffn_mode'] = args.ffn_mode

        # Gradient engine requires additional parameters
        if args.ffn_mode == 'variational_gradient_engine':
            config.setdefault('ffn_lambda_belief', 1.0)
            config.setdefault('ffn_lambda_prior', 0.0)
            config.setdefault('ffn_lambda_phi', 0.0)
            config.setdefault('ffn_update_sigma', True)
            config['evolve_sigma'] = True  # Enable sigma evolution for full Gaussian inference

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