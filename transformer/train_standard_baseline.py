"""
Train Standard Transformer Baseline
====================================

Fair comparison to gauge-theoretic transformer:
    - Same data (character-level WikiText-2)
    - Same parameter budget (~5,334 params)
    - Same training steps (20 for debug, 100 for standard)
    - Standard dot-product attention (NO gauge theory, NO KL divergence)
    - OUTPUTS COMPATIBLE METRICS for plotting with gauge model figures

This answers the critical question:
    "Does SO(3) gauge structure help or hurt performance?"

Expected outcomes:
    1. Standard > Gauge: SO(3) is wrong inductive bias
    2. Standard = Gauge: SO(3) is neutral
    3. Standard < Gauge: SO(3) helps! (validates framework)

Output Format:
    - checkpoints_publication/standard_baseline/metrics.csv (compatible with plot_pub_figs.py)
    - checkpoints_publication/standard_baseline/best_model.pt
    - checkpoints_publication/standard_baseline/training_log.json

Usage:
    # Just click Run
    python transformer/train_standard_baseline.py

    # Or with args
    python transformer/train_standard_baseline.py --config debug_matched_lr

    # Then plot alongside gauge model
    python plot_pub_figs.py  # Will find both models automatically

Author: Ablation study baseline
Date: November 2025
"""

import sys
import os
import time
import json
import csv
import math
import argparse
from pathlib import Path
from typing import Dict, List
from collections import defaultdict

import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np

# Import standard transformer
from transformer.standard_transformer import StandardTransformerLM

# Import data loaders (reuse from gauge model)
from transformer.data import create_char_dataloaders, create_dataloaders


class StandardMetricsTracker:
    """Track metrics in format compatible with gauge model plotting utilities."""

    def __init__(self, save_path: Path):
        self.save_path = save_path
        self.history = []

        # Create CSV with same headers as PublicationMetricsTracker
        self.save_path.parent.mkdir(parents=True, exist_ok=True)

        self.headers = [
            # Core
            'step', 'timestamp',

            # Losses (standard transformer doesn't have free energy components)
            'train_loss_total', 'train_loss_ce', 'train_loss_belief_align',
            'train_loss_self_consistency', 'train_loss_model_align',
            'val_loss', 'val_ce',

            # Metrics
            'train_ppl', 'train_bpc', 'val_ppl', 'val_bpc',

            # Attention stats (standard transformer doesn't track these)
            'beta_mean', 'beta_std', 'kl_mean', 'kl_std',

            # Learning rates (standard has single lr)
            'mu_lr', 'sigma_lr', 'phi_lr', 'ffn_lr',

            # Gradient norms
            'grad_norm_total', 'grad_norm_mu', 'grad_norm_ffn',

            # Performance
            'step_time', 'tokens_per_sec',
        ]

        with open(self.save_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(self.headers)

    def log_step(self, step: int, train_loss: float, train_ce: float,
                 lr: float, grad_norm: float, step_time: float,
                 batch_size: int, seq_len: int):
        """Log training step with metrics."""

        # Compute tokens/sec
        tokens_per_sec = (batch_size * seq_len) / step_time if step_time > 0 else 0

        # Bits per character
        train_bpc = train_ce / math.log(2)
        train_ppl = math.exp(train_ce)

        entry = {
            'step': step,
            'timestamp': time.time(),

            # Losses - for standard transformer, total = CE (no free energy terms)
            'train_loss_total': train_loss,
            'train_loss_ce': train_ce,
            'train_loss_belief_align': 0.0,  # N/A for standard
            'train_loss_self_consistency': 0.0,  # N/A for standard
            'train_loss_model_align': 0.0,  # N/A for standard
            'val_loss': None,
            'val_ce': None,

            # Metrics
            'train_ppl': train_ppl,
            'train_bpc': train_bpc,
            'val_ppl': None,
            'val_bpc': None,

            # Attention (N/A for standard softmax attention)
            'beta_mean': None,
            'beta_std': None,
            'kl_mean': None,
            'kl_std': None,

            # Learning rates (standard has single lr for all params)
            'mu_lr': lr,
            'sigma_lr': 0.0,  # N/A
            'phi_lr': 0.0,  # N/A
            'ffn_lr': lr,

            # Gradients
            'grad_norm_total': grad_norm,
            'grad_norm_mu': 0.0,  # Could compute if needed
            'grad_norm_ffn': 0.0,  # Could compute if needed

            # Performance
            'step_time': step_time,
            'tokens_per_sec': tokens_per_sec,
        }

        self.history.append(entry)

    def log_val(self, step: int, val_loss: float, val_ce: float):
        """Update entry with validation metrics."""
        for entry in reversed(self.history):
            if entry['step'] == step:
                entry['val_loss'] = val_loss
                entry['val_ce'] = val_ce
                entry['val_ppl'] = math.exp(val_ce)
                entry['val_bpc'] = val_ce / math.log(2)
                break

    def save(self):
        """Save to CSV."""
        if not self.history:
            return

        with open(self.save_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.headers)
            writer.writeheader()
            writer.writerows(self.history)


def train_standard_baseline(
    config: dict,
    train_loader,
    val_loader,
    device: str = 'cpu',
    checkpoint_dir: str = 'checkpoints_publication/standard_baseline',
):
    """
    Train standard transformer baseline with publication-quality metrics.

    Args:
        config: Model configuration
        train_loader: Training dataloader
        val_loader: Validation dataloader
        device: Device to train on
        checkpoint_dir: Where to save checkpoints
    """
    # Create checkpoint directory
    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    # Create model
    print("\n" + "="*70)
    print("CREATING STANDARD TRANSFORMER")
    print("="*70)

    model = StandardTransformerLM(config).to(device)

    # Count parameters
    param_counts = model.count_parameters()
    print("\nParameter Breakdown:")
    for name, count in param_counts.items():
        print(f"  {name:20s}: {count:6d}")

    # Optimizer - use same settings as gauge model for fair comparison
    lr = config.get('lr', 0.05)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=config.get('weight_decay', 0.01),
        betas=(0.9, 0.999),
    )

    # Learning rate scheduler with warmup + cosine decay
    max_steps = config['max_steps']
    warmup_steps = config.get('warmup_steps', max(10, max_steps // 10))  # 10% warmup
    min_lr = config.get('min_lr', lr * 0.1)  # Decay to 10% of initial lr

    def lr_lambda(step):
        """Warmup then cosine decay."""
        if step < warmup_steps:
            # Linear warmup
            return step / warmup_steps
        else:
            # Cosine decay from 1.0 to min_lr/lr
            progress = (step - warmup_steps) / (max_steps - warmup_steps)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            return (min_lr / lr) + (1 - min_lr / lr) * cosine_decay

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    print(f"\nOptimizer: AdamW")
    print(f"  Learning rate: {lr}")
    print(f"  Weight decay:  {config.get('weight_decay', 0.01)}")
    print(f"  Warmup steps:  {warmup_steps}")
    print(f"  Min LR:        {min_lr}")

    # Initialize metrics tracker
    metrics_path = checkpoint_path / 'metrics.csv'
    metrics_tracker = StandardMetricsTracker(metrics_path)
    print(f"\n📊 Logging metrics to: {metrics_path}")

    # Training loop
    max_steps = config['max_steps']
    log_interval = config.get('log_interval', 5)
    eval_interval = config.get('eval_interval', 20)

    print("\n" + "="*70)
    print("STARTING TRAINING")
    print("="*70)
    print(f"Max steps: {max_steps}")
    print(f"Device: {device}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Sequence length: {config['max_seq_len']}")

    model.train()
    step = 0
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    start_time = time.time()

    pbar = tqdm(total=max_steps, desc="Training")
    train_iter = iter(train_loader)

    while step < max_steps:
        step += 1
        step_start = time.time()

        # Get batch
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        # Unpack batch
        input_ids, _ = batch
        input_ids = input_ids.to(device)
        labels = input_ids.clone()

        batch_size, seq_len = input_ids.shape

        # Forward pass
        optimizer.zero_grad()
        output = model(input_ids, labels=labels)
        loss = output['loss']

        # Backward pass
        loss.backward()

        # Compute gradient norm BEFORE clipping
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        grad_norm = total_norm ** 0.5

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.get('grad_clip', 1.0))
        optimizer.step()
        scheduler.step()

        # Get current learning rate
        current_lr = scheduler.get_last_lr()[0]

        step_time = time.time() - step_start
        train_losses.append(loss.item())

        # Log to metrics tracker
        metrics_tracker.log_step(
            step=step,
            train_loss=loss.item(),
            train_ce=loss.item(),  # For standard transformer, total loss = CE loss
            lr=current_lr,
            grad_norm=grad_norm,
            step_time=step_time,
            batch_size=batch_size,
            seq_len=seq_len,
        )

        # Compute perplexity
        ppl = torch.exp(loss).item()

        # Update progress bar
        pbar.update(1)
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'ppl': f'{ppl:.1f}',
            'lr': f'{current_lr:.2e}',
            'time': f'{step_time:.2f}s'
        })

        # Logging
        if step % log_interval == 0 or step == max_steps:
            bpc = loss.item() / math.log(2)
            print(f"\nStep {step}/{max_steps} | "
                  f"Loss: {loss.item():.4f} | "
                  f"PPL: {ppl:.1f} | "
                  f"BPC: {bpc:.3f} | "
                  f"LR: {current_lr:.2e}")

        # Validation
        if step % eval_interval == 0 or step == max_steps:
            model.eval()
            val_loss = 0.0
            n_val_batches = min(100, len(val_loader))

            with torch.no_grad():
                for i, batch in enumerate(val_loader):
                    if i >= n_val_batches:
                        break

                    input_ids, _ = batch
                    input_ids = input_ids.to(device)
                    labels = input_ids.clone()

                    output = model(input_ids, labels=labels)
                    val_loss += output['loss'].item()

            val_loss /= n_val_batches
            val_ppl = np.exp(val_loss)
            val_bpc = val_loss / math.log(2)
            val_losses.append(val_loss)

            # Log validation to tracker
            metrics_tracker.log_val(step, val_loss, val_loss)

            print(f"\n  Validation @ step {step}:")
            print(f"    Loss: {val_loss:.4f}")
            print(f"    Perplexity: {val_ppl:.2f}")
            print(f"    BPC: {val_bpc:.3f}")

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_checkpoint_path = checkpoint_path / 'best_model.pt'
                torch.save({
                    'step': step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_ppl': val_ppl,
                    'config': config,
                }, best_checkpoint_path)
                print(f"    💾 Saved best model: {best_checkpoint_path}")

            model.train()

        # Save metrics periodically
        if step % 100 == 0:
            metrics_tracker.save()

    pbar.close()

    # Save final metrics
    metrics_tracker.save()
    print(f"\n📊 Final metrics saved to: {metrics_path}")

    # Training complete
    total_time = time.time() - start_time
    avg_step_time = total_time / max_steps

    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"Total time: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
    print(f"Average step time: {avg_step_time:.2f} seconds")
    print(f"Steps per second: {1/avg_step_time:.2f}")
    print(f"Best validation loss: {best_val_loss:.4f}")

    # Compute improvement over random
    random_ppl = config['vocab_size']
    final_ppl = np.exp(best_val_loss)
    final_bpc = best_val_loss / math.log(2)
    improvement = random_ppl / final_ppl

    print(f"\nFinal Metrics:")
    print(f"  Loss: {best_val_loss:.4f}")
    print(f"  Perplexity: {final_ppl:.2f}")
    print(f"  BPC: {final_bpc:.3f}")
    print(f"  Improvement over random: {improvement:.1f}x")
    print("="*70)

    # Save training log (backward compatibility)
    log_path = checkpoint_path / 'training_log.json'
    log_data = {
        'config': config,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss,
        'best_val_ppl': final_ppl,
        'best_val_bpc': final_bpc,
        'total_time_seconds': total_time,
        'improvement_over_random': improvement,
    }

    with open(log_path, 'w') as f:
        json.dump(log_data, f, indent=2)

    print(f"\n💾 Saved training log: {log_path}")

    return model, best_val_loss, final_ppl


def main():
    parser = argparse.ArgumentParser(description='Train standard transformer baseline')
    parser.add_argument('--config', type=str, default='debug',
                        choices=['debug', 'debug_matched_lr', 'debug_moderate_lr', 'convergence_test', 'standard', 'extended', 'stable'],
                        help='Configuration preset')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate (overrides config default)')
    parser.add_argument('--device', type=str, default='auto', choices=['cpu', 'cuda', 'auto'],
                        help='Device to train on')
    args = parser.parse_args()

    # Configuration presets
    configs = {
        'debug': {
            'vocab_size': 256,
            'embed_dim': 12,
            'n_layers': 3,
            'n_heads': 3,
            'hidden_dim': 44,
            'max_seq_len': 32,
            'dropout': 0.1,
            'tie_embeddings': True,
            'batch_size': 16,
            'max_steps': 1000,
            'lr': 0.0025,  # Match gauge model
            'weight_decay': 0.01,
            'grad_clip': 1.0,
            'log_interval': 5,
            'eval_interval': 20,
        },
        'debug_matched_lr': {
            'vocab_size': 256,
            'embed_dim': 11,
            'n_layers': 2,
            'n_heads': 1,
            'hidden_dim': 44,
            'max_seq_len': 32,
            'dropout': 0.1,
            'tie_embeddings': True,
            'batch_size': 16,
            'max_steps': 20,
            'lr': 0.05,
            'weight_decay': 0.01,
            'grad_clip': 1.0,
            'log_interval': 5,
            'eval_interval': 20,
        },
        'debug_moderate_lr': {
            'vocab_size': 256,
            'embed_dim': 11,
            'n_layers': 2,
            'n_heads': 1,
            'hidden_dim': 44,
            'max_seq_len': 32,
            'dropout': 0.1,
            'tie_embeddings': True,
            'batch_size': 16,
            'max_steps': 20,
            'lr': 0.01,
            'weight_decay': 0.01,
            'grad_clip': 1.0,
            'log_interval': 5,
            'eval_interval': 20,
        },
        'convergence_test': {
            'vocab_size': 256,
            'embed_dim': 11,
            'n_layers': 2,
            'n_heads': 1,
            'hidden_dim': 44,
            'max_seq_len': 32,
            'dropout': 0.1,
            'tie_embeddings': True,
            'batch_size': 16,
            'max_steps': 1000,
            'lr': 0.001,
            'weight_decay': 0.01,
            'grad_clip': 1.0,
            'log_interval': 50,
            'eval_interval': 200,
        },
        'standard': {
            'vocab_size': 256,
            'embed_dim': 12,
            'n_layers': 3,
            'n_heads': 4,
            'hidden_dim': 44,
            'max_seq_len': 32,
            'dropout': 0.1,
            'tie_embeddings': True,
            'batch_size': 16,
            'max_steps': 100,
            'lr': 0.05,
            'weight_decay': 0.01,
            'grad_clip': 1.0,
            'log_interval': 10,
            'eval_interval': 50,
        },
        'extended': {
            'vocab_size': 256,
            'embed_dim': 11,
            'n_layers': 3,
            'n_heads': 1,
            'hidden_dim': 44,
            'max_seq_len': 32,
            'dropout': 0.1,
            'tie_embeddings': True,
            'batch_size': 32,
            'max_steps': 200,
            'lr': 0.001,
            'weight_decay': 0.01,
            'grad_clip': 1.0,
            'log_interval': 10,
            'eval_interval': 50,
        },
        # Stable config for longer training with proper LR schedule
        'stable': {
            'vocab_size': 5000,       # BPE tokenizer
            'embed_dim': 128,         # 128 = 4 × 32 (clean division for heads)
            'n_layers': 6,
            'n_heads': 4,             # 128 / 4 = 32 head_dim
            'hidden_dim': 512,        # 4 × embed_dim
            'max_seq_len': 48,
            'dropout': 0.1,
            'tie_embeddings': True,
            'batch_size': 16,
            'max_steps': 500,
            'lr': 0.0003,             # Standard transformer LR (3e-4)
            'min_lr': 0.00003,        # 10% of peak
            'warmup_steps': 50,       # 10% warmup
            'weight_decay': 0.01,
            'grad_clip': 1.0,
            'log_interval': 10,
            'eval_interval': 25,
        },
    }

    config = configs[args.config]

    # Override lr if specified
    if args.lr is not None:
        config['lr'] = args.lr

    # Determine device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    print("="*70)
    print("STANDARD TRANSFORMER BASELINE EXPERIMENT")
    print("="*70)
    print(f"\nConfiguration: {args.config}")
    print(f"Device: {device}")
    print(f"\nModel:")
    print(f"  vocab_size:   {config['vocab_size']}")
    print(f"  embed_dim:    {config['embed_dim']}")
    print(f"  n_layers:     {config['n_layers']}")
    print(f"  n_heads:      {config['n_heads']}")
    print(f"  hidden_dim:   {config['hidden_dim']}")
    print(f"  max_seq_len:  {config['max_seq_len']}")
    print(f"\nTraining:")
    print(f"  batch_size:   {config['batch_size']}")
    print(f"  max_steps:    {config['max_steps']}")
    print(f"  lr:           {config['lr']}")
    print(f"  weight_decay: {config['weight_decay']}")

    # Load data
    print("\n" + "="*70)
    print("LOADING DATA")
    print("="*70)

    # Use BPE tokenizer for vocab_size > 256, otherwise character-level
    use_bpe = config.get('tokenizer', 'auto') == 'bpe' or config['vocab_size'] > 256

    if use_bpe:
        print(f"Using BPE tokenizer (vocab_size={config['vocab_size']})")
        train_loader, val_loader, actual_vocab_size = create_dataloaders(
            batch_size=config['batch_size'],
            max_seq_len=config['max_seq_len'],
            vocab_size=config['vocab_size'],
            num_workers=0,
        )
    else:
        print("Using character-level tokenizer")
        train_loader, val_loader, actual_vocab_size = create_char_dataloaders(
            batch_size=config['batch_size'],
            max_seq_len=config['max_seq_len'],
            num_workers=0,
        )

    config['vocab_size'] = actual_vocab_size
    print(f"\nActual vocabulary size: {config['vocab_size']}")

    # Train
    checkpoint_dir = f'checkpoints_publication/standard_baseline_{args.config}'
    
    model, best_val_loss, best_val_ppl = train_standard_baseline(
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        checkpoint_dir=checkpoint_dir,
    )

    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE!")
    print("="*70)
    print(f"\nStandard Transformer Results:")
    print(f"  Validation Loss: {best_val_loss:.4f}")
    print(f"  Validation PPL:  {best_val_ppl:.2f}")
    print(f"\nMetrics saved in format compatible with plot_pub_figs.py")
    print(f"  Directory: {checkpoint_dir}/")
    print(f"  - metrics.csv (compatible with plotting)")
    print(f"  - best_model.pt (checkpoint)")
    print(f"  - training_log.json (summary)")
    print("\nNext steps:")
    print(f"  1. Train gauge model (if not already done)")
    print(f"  2. Run: python plot_pub_figs.py")
    print(f"     (Will auto-detect both standard and gauge results)")
    print("="*70)


if __name__ == '__main__':
    main()