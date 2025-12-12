"""
Fast Training Loop with Natural Gradient Learning Rates
========================================================

Uses separate learning rates for different parameter types,
based on empirical convergence from test suite:
    - mu_q_lr = 0.1 (means)
    - Sigma_q_lr = 0.005 (covariances)
    - phi_lr = 0.01 (gauge frames)
    - ffn_lr = 0.001 (standard FFN parameters)

This exploits the natural gradient speedup on statistical manifolds.

Author: Optimized from test suite convergence
Date: November 2025
"""

# Suppress noisy warnings BEFORE other imports
import warnings
warnings.filterwarnings("ignore", message="CUDA path could not be detected", module="cupy")
warnings.filterwarnings("ignore", message="Failed to find cuobjdump", module="triton")
warnings.filterwarnings("ignore", message="Failed to find nvdisasm", module="triton")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass, field
from pathlib import Path
import time
import json

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# Import standard loss computation
from transformer.train import compute_free_energy_loss


# =============================================================================
# Fast Training Configuration
# =============================================================================

@dataclass
class FastTrainingConfig:
    """Training configuration with per-parameter group learning rates."""

    # Training steps
    max_steps: int = 1000
    warmup_steps: int = 50

    # Per-parameter group learning rates (NATURAL GRADIENTS!)
    mu_lr: float = 0.1           # Mean embeddings (from test suite)
    sigma_lr: float = 0.005      # Covariance embeddings (from test suite)
    phi_lr: float = 0.01         # Gauge frames
    attention_lr: float = 0.01   # Attention parameters
    ffn_lr: float = 0.001        # FFN parameters (standard)
    output_lr: float = 0.001     # Output projection

    # Optimizer hyperparameters
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    weight_decay: float = 0.01

    # Gradient control
    grad_clip: float = 1.0
    grad_accumulation_steps: int = 1

    # Free energy coefficients
    alpha: float = 1.0            # Self-consistency regularization
    beta: float = 1.0             # Belief alignment
    lambda_gamma: float = 0.0     # Model alignment (disabled by default)
    kappa_gamma: float = 1.0      # Temperature for γ_ij coupling weights

    # Learning rate schedule
    lr_decay: str = 'cosine'  # 'cosine', 'linear', 'constant'

    # Logging
    log_interval: int = 10
    eval_interval: int = 100
    checkpoint_interval: int = 200

    # Early stopping
    patience: int = 0  # If > 0, stop if no improvement for this many evals

    # Checkpointing
    checkpoint_dir: Path = Path('checkpoints')
    save_total_limit: int = 3

    # Weights & Biases
    use_wandb: bool = False
    wandb_project: str = 'gauge-transformer-fast'
    wandb_run_name: Optional[str] = None

    # Mixed precision
    use_amp: bool = False


# =============================================================================
# Fast Trainer with Parameter Group Learning Rates
# =============================================================================

class FastTrainer:
    """
    Trainer with separate learning rates for each parameter type.

    Parameter Groups:
        1. mu_embed: Mean embeddings (lr=0.1)
        2. sigma_embed: Covariance embeddings (lr=0.005)
        3. phi_embed: Gauge frame embeddings (lr=0.01)
        4. attention: Attention mechanism (lr=0.01)
        5. ffn: Feed-forward networks (lr=0.001)
        6. output: Output projection (lr=0.001)

    This exploits natural gradient structure on statistical manifolds!
    """

    def __init__(
        self,
        model,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: FastTrainingConfig,
        device: torch.device = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device or torch.device('cpu')

        self.model.to(self.device)

        # Create optimizer with parameter groups
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()

        # Training state
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0  # Early stopping counter

        # Mixed precision (using modern AMP API for PyTorch 2.x / CUDA 12+)
        self.scaler = torch.amp.GradScaler('cuda') if config.use_amp else None

        # W&B logging
        if config.use_wandb and WANDB_AVAILABLE:
            wandb.init(
                project=config.wandb_project,
                name=config.wandb_run_name,
                config=vars(config),
            )

        # Create checkpoint directory
        config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*70}")
        print("FAST TRAINER INITIALIZED")
        print(f"{'='*70}")
        print(f"  Device: {self.device}")
        print(f"  Max steps: {self.config.max_steps:,}")
        print(f"\n  Learning Rates (Natural Gradients!):")
        print(f"    μ (means):        {config.mu_lr}")
        print(f"    Σ (covariances):  {config.sigma_lr}")
        print(f"    φ (gauge frames): {config.phi_lr}")
        print(f"    Attention:        {config.attention_lr}")
        print(f"    FFN:              {config.ffn_lr}")
        print(f"    Output:           {config.output_lr}")
        print(f"{'='*70}\n")

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """
        Create optimizer with per-parameter group learning rates.

        Returns:
            AdamW optimizer with 6 parameter groups
        """
        # Collect parameters by type
        mu_params = []
        sigma_params = []
        phi_params = []
        attention_params = []
        ffn_params = []
        output_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            # Mean embeddings
            if 'mu_embed' in name:
                mu_params.append(param)

            # Covariance embeddings
            elif 'sigma_embed' in name:
                sigma_params.append(param)

            # Gauge frame embeddings
            elif 'phi_embed' in name:
                phi_params.append(param)

            # Positional encoding (treat as gauge frames)
            elif 'pos_encoding' in name:
                phi_params.append(param)

            # Attention mechanism
            elif 'attention' in name or 'attn' in name:
                attention_params.append(param)

            # Output projection
            elif 'out_proj' in name:
                output_params.append(param)

            # FFN (default for everything else)
            else:
                ffn_params.append(param)

        # Create parameter groups
        param_groups = []

        if mu_params:
            param_groups.append({
                'params': mu_params,
                'lr': self.config.mu_lr,
                'weight_decay': 0.0,  # No decay for embeddings
                'name': 'mu_embed',
            })
            print(f"  Parameter group 'mu_embed': {len(mu_params)} tensors @ lr={self.config.mu_lr}")

        if sigma_params:
            param_groups.append({
                'params': sigma_params,
                'lr': self.config.sigma_lr,
                'weight_decay': 0.0,
                'name': 'sigma_embed',
            })
            print(f"  Parameter group 'sigma_embed': {len(sigma_params)} tensors @ lr={self.config.sigma_lr}")

        if phi_params:
            param_groups.append({
                'params': phi_params,
                'lr': self.config.phi_lr,
                'weight_decay': 0.0,
                'name': 'phi_embed',
            })
            print(f"  Parameter group 'phi_embed': {len(phi_params)} tensors @ lr={self.config.phi_lr}")

        if attention_params:
            param_groups.append({
                'params': attention_params,
                'lr': self.config.attention_lr,
                'weight_decay': self.config.weight_decay,
                'name': 'attention',
            })
            print(f"  Parameter group 'attention': {len(attention_params)} tensors @ lr={self.config.attention_lr}")

        if ffn_params:
            param_groups.append({
                'params': ffn_params,
                'lr': self.config.ffn_lr,
                'weight_decay': self.config.weight_decay,
                'name': 'ffn',
            })
            print(f"  Parameter group 'ffn': {len(ffn_params)} tensors @ lr={self.config.ffn_lr}")

        if output_params:
            param_groups.append({
                'params': output_params,
                'lr': self.config.output_lr,
                'weight_decay': 0.0,  # Often tied to embeddings
                'name': 'output',
            })
            print(f"  Parameter group 'output': {len(output_params)} tensors @ lr={self.config.output_lr}")

        optimizer = torch.optim.AdamW(
            param_groups,
            betas=(self.config.beta1, self.config.beta2),
            eps=self.config.eps,
        )

        return optimizer

    def _create_scheduler(self):
        """Create learning rate scheduler for all parameter groups."""
        if self.config.lr_decay == 'constant':
            return None

        def lr_lambda(step):
            # Warmup
            if step < self.config.warmup_steps:
                return step / max(1, self.config.warmup_steps)

            # Decay
            progress = (step - self.config.warmup_steps) / max(1, self.config.max_steps - self.config.warmup_steps)

            if self.config.lr_decay == 'cosine':
                return 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159)).item())
            elif self.config.lr_decay == 'linear':
                return max(0.0, 1.0 - progress)
            else:
                return 1.0

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=[lr_lambda] * len(self.optimizer.param_groups),
        )

        return scheduler

    def train_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Dict[str, float]:
        """Single training step."""
        self.model.train()

        input_ids, target_ids = batch
        input_ids = input_ids.to(self.device)
        target_ids = target_ids.to(self.device)

        # Forward pass (with optional AMP using modern API for PyTorch 2.x / CUDA 12+)
        if self.config.use_amp:
            with torch.amp.autocast('cuda'):
                loss, metrics = compute_free_energy_loss(
                    self.model,
                    input_ids,
                    target_ids,
                    alpha=self.config.alpha,
                    lambda_beta=self.config.beta,
                    lambda_gamma=self.config.lambda_gamma,
                    kappa_gamma=self.config.kappa_gamma,
                )
        else:
            loss, metrics = compute_free_energy_loss(
                self.model,
                input_ids,
                target_ids,
                alpha=self.config.alpha,
                lambda_beta=self.config.beta,
                lambda_gamma=self.config.lambda_gamma,
                kappa_gamma=self.config.kappa_gamma,
            )

        # Backward pass
        loss = loss / self.config.grad_accumulation_steps

        if self.config.use_amp:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        # Gradient accumulation
        if (self.global_step + 1) % self.config.grad_accumulation_steps == 0:
            # Gradient clipping
            if self.config.grad_clip > 0:
                if self.config.use_amp:
                    self.scaler.unscale_(self.optimizer)

                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.grad_clip,
                )

            # Optimizer step
            if self.config.use_amp:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

            # Scheduler step
            if self.scheduler is not None:
                self.scheduler.step()

            self.optimizer.zero_grad()

        # Reformat metrics for logging
        formatted_metrics = {
            'total_loss': metrics['loss/total'],
            'ce_loss': metrics['loss/ce'],
            'perplexity': torch.exp(torch.tensor(metrics['loss/ce'])).item(),
        }

        return formatted_metrics

    def validate(self, max_batches: int = 5) -> Dict[str, float]:
        """Validation loop.

        Args:
            max_batches: Maximum number of validation batches (default: 5 for speed)
        """
        self.model.eval()

        total_loss = 0.0
        total_ce = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in self.val_loader:
                input_ids, target_ids = batch
                input_ids = input_ids.to(self.device)
                target_ids = target_ids.to(self.device)

                loss, metrics = compute_free_energy_loss(
                    self.model,
                    input_ids,
                    target_ids,
                    alpha=self.config.alpha,
                    lambda_beta=self.config.beta,
                    lambda_gamma=self.config.lambda_gamma,
                    kappa_gamma=self.config.kappa_gamma,
                )

                total_loss += loss.item()
                total_ce += metrics['loss/ce']  # Fix: use correct key format
                num_batches += 1

                # Limit validation batches for speed
                if num_batches >= max_batches:
                    break

        avg_loss = total_loss / max(1, num_batches)
        avg_ce = total_ce / max(1, num_batches)
        perplexity = torch.exp(torch.tensor(avg_ce)).item()

        return {
            'loss': avg_loss,
            'ce_loss': avg_ce,
            'perplexity': perplexity,
        }

    def train(self):
        """Main training loop."""
        print(f"{'='*70}")
        print("STARTING FAST TRAINING")
        print(f"{'='*70}\n")

        start_time = time.time()
        step_times = []

        # Training loop
        train_iterator = iter(self.train_loader)

        if TQDM_AVAILABLE:
            pbar = tqdm(range(self.config.max_steps), desc="Training")
        else:
            pbar = range(self.config.max_steps)

        for step in pbar:
            self.global_step = step
            step_start = time.time()

            # Get batch
            try:
                batch = next(train_iterator)
            except StopIteration:
                train_iterator = iter(self.train_loader)
                batch = next(train_iterator)

            # Train step
            metrics = self.train_step(batch)

            step_time = time.time() - step_start
            step_times.append(step_time)

            # Logging
            if (step + 1) % self.config.log_interval == 0:
                # Get current learning rates
                lrs = {group['name']: group['lr'] for group in self.optimizer.param_groups}

                log_msg = (
                    f"Step {step+1}/{self.config.max_steps} | "
                    f"Loss: {metrics['total_loss']:.4f} | "
                    f"PPL: {metrics['perplexity']:.1f} | "
                    f"μ_lr: {lrs.get('mu_embed', 0):.2e} | "
                    f"Time: {step_time:.2f}s"
                )

                if TQDM_AVAILABLE:
                    pbar.set_description(log_msg)
                else:
                    print(log_msg)

                # W&B logging
                if self.config.use_wandb and WANDB_AVAILABLE:
                    wandb.log({
                        'train/loss': metrics['total_loss'],
                        'train/ce_loss': metrics['ce_loss'],
                        'train/perplexity': metrics['perplexity'],
                        'train/step_time': step_time,
                        **{f'lr/{k}': v for k, v in lrs.items()},
                    }, step=step)

            # Validation
            if (step + 1) % self.config.eval_interval == 0:
                val_metrics = self.validate()

                print(f"\n  Validation @ step {step+1}:")
                print(f"    Loss: {val_metrics['loss']:.4f}")
                print(f"    Perplexity: {val_metrics['perplexity']:.2f}\n")

                # W&B logging
                if self.config.use_wandb and WANDB_AVAILABLE:
                    wandb.log({
                        'val/loss': val_metrics['loss'],
                        'val/perplexity': val_metrics['perplexity'],
                    }, step=step)

                # Save best model
                if val_metrics['loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['loss']
                    self.patience_counter = 0  # Reset patience
                    self.save_checkpoint(is_best=True)
                else:
                    self.patience_counter += 1  # Increment patience
                    if self.config.patience > 0 and self.patience_counter >= self.config.patience:
                        print(f"\n⚠ Early stopping triggered! No improvement for {self.config.patience} evaluations.")
                        print(f"  Best validation loss: {self.best_val_loss:.4f}")
                        break  # Stop training

            # Checkpointing
            if (step + 1) % self.config.checkpoint_interval == 0:
                self.save_checkpoint(is_best=False)

        # Training complete
        elapsed = time.time() - start_time
        avg_step_time = sum(step_times) / len(step_times)

        print(f"\n{'='*70}")
        print("TRAINING COMPLETE!")
        print(f"{'='*70}")
        print(f"Total time: {elapsed/60:.1f} minutes ({elapsed/3600:.2f} hours)")
        print(f"Average step time: {avg_step_time:.2f} seconds")
        print(f"Steps per second: {1.0/avg_step_time:.2f}")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"{'='*70}\n")

    def save_checkpoint(self, is_best: bool = False):
        """Save checkpoint."""
        checkpoint = {
            'step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': vars(self.config),
        }

        if is_best:
            path = self.config.checkpoint_dir / 'best_model.pt'
            print(f"  💾 Saving best model: {path}")
        else:
            path = self.config.checkpoint_dir / f'checkpoint_step_{self.global_step}.pt'

        torch.save(checkpoint, path)

        # Cleanup old checkpoints
        if not is_best:
            checkpoints = sorted(
                self.config.checkpoint_dir.glob('checkpoint_step_*.pt'),
                key=lambda p: p.stat().st_mtime,
            )
            while len(checkpoints) > self.config.save_total_limit:
                oldest = checkpoints.pop(0)
                oldest.unlink()

        return path


# =============================================================================
# Testing
# =============================================================================

if __name__ == '__main__':
    print("="*70)
    print("FAST TRAINER TEST")
    print("="*70)

    # This file just defines the trainer
    # See train_example.py for full integration

    print("\n✓ FastTrainer class defined")
    print("✓ Supports per-parameter group learning rates")
    print("\nExample usage:")
    print("""
from transformer.config_publication import PUBLICATION_CONFIG
from transformer.model import GaugeTransformerLM
from transformer.data import create_dataloaders
from transformer.train_fast import FastTrainer, FastTrainingConfig

# Create model & data
model = GaugeTransformerLM(PUBLICATION_CONFIG)
train_loader, val_loader, vocab_size = create_dataloaders(...)

# Fast training config
config = FastTrainingConfig(
    max_steps=1000,
    mu_lr=0.1,
    sigma_lr=0.005,
    phi_lr=0.01,
    ffn_lr=0.001,
)

# Train!
trainer = FastTrainer(model, train_loader, val_loader, config)
trainer.train()
""")

    print("="*70)