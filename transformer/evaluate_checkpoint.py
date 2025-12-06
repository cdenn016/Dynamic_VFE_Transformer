"""
Evaluate a trained checkpoint on validation data.

Usage:
    python evaluate_checkpoint.py --checkpoint checkpoints_realistic/best_model.pt
"""

import torch
import argparse
from pathlib import Path
import numpy as np

from transformer.model import GaugeTransformerLM
from transformer.data import create_dataloaders
from transformer.train import compute_free_energy_loss


def evaluate_checkpoint(checkpoint_path: str, max_batches: int = 50):
    """
    Load checkpoint and evaluate on validation set.

    Args:
        checkpoint_path: Path to checkpoint file
        max_batches: Number of validation batches to evaluate
    """
    print("="*70)
    print("CHECKPOINT EVALUATION")
    print("="*70)

    # Load checkpoint
    print(f"\nLoading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    config = checkpoint['config']
    step = checkpoint.get('step', 0)
    best_val_loss = checkpoint.get('best_val_loss', None)

    print(f"\nCheckpoint info:")
    print(f"  Step: {step}")
    if best_val_loss is not None:
        print(f"  Best val loss: {best_val_loss:.4f}")
        print(f"  Best val PPL: {np.exp(best_val_loss):.2f}")

    # Handle both dict and dataclass configs
    def get_config_val(cfg, key, default=None):
        if isinstance(cfg, dict):
            return cfg.get(key, default)
        else:
            return getattr(cfg, key, default)

    # Extract vocab_size from checkpoint weights (more reliable than config)
    model_state = checkpoint['model_state_dict']
    if 'token_embed.mu_embed.weight' in model_state:
        vocab_size = model_state['token_embed.mu_embed.weight'].shape[0]
        embed_dim = model_state['token_embed.mu_embed.weight'].shape[1]
    else:
        vocab_size = get_config_val(config, 'vocab_size', 2000)
        embed_dim = get_config_val(config, 'embed_dim', 11)

    max_seq_len = get_config_val(config, 'max_seq_len', 40)
    n_layers = get_config_val(config, 'n_layers', 2)
    batch_size = get_config_val(config, 'batch_size', 4)

    print(f"\nModel config:")
    print(f"  Agents (N): {max_seq_len}")
    print(f"  Fiber (K): {embed_dim}")
    print(f"  Layers: {n_layers}")
    print(f"  Vocab: {vocab_size:,}")

    # Create model
    print(f"\nCreating model...")

    # Always build complete config dict (regardless of config type)
    config_dict = {
        'vocab_size': vocab_size,
        'embed_dim': embed_dim,
        'n_layers': n_layers,
        'max_seq_len': max_seq_len,
        'hidden_dim': get_config_val(config, 'hidden_dim', embed_dim * 4),
        'kappa_beta': get_config_val(config, 'kappa_beta', 1.0),
        'epsilon': get_config_val(config, 'epsilon', 1e-8),
        'pos_encoding_mode': get_config_val(config, 'pos_encoding_mode', 'learned'),
        'evolve_sigma': get_config_val(config, 'evolve_sigma', False),
        'evolve_phi': get_config_val(config, 'evolve_phi', False),
        'tie_embeddings': get_config_val(config, 'tie_embeddings', True),
        'dropout': get_config_val(config, 'dropout', 0.1),
        'irrep_spec': get_config_val(config, 'irrep_spec', [('ℓ0', 5, 1), ('ℓ1', 2, 3)]),
    }

    model = GaugeTransformerLM(config_dict)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    total_params = model.get_num_params(non_embedding=False)
    print(f"  Total params: {total_params:,}")

    # Create data loaders
    print(f"\nLoading data...")
    train_loader, val_loader, actual_vocab_size = create_dataloaders(
        max_seq_len=max_seq_len,
        batch_size=batch_size,
        vocab_size=vocab_size,
        num_workers=0,
    )

    # Check vocab size mismatch
    if actual_vocab_size != vocab_size:
        print(f"\n⚠ WARNING: Vocab size mismatch!")
        print(f"  Model expects: {vocab_size}")
        print(f"  Data has:      {actual_vocab_size}")
        print(f"  Adjusting model vocab size to match data...")

        # Recreate model with correct vocab size
        config_dict['vocab_size'] = actual_vocab_size
        model = GaugeTransformerLM(config_dict)

        # Load weights with strict=False to handle size mismatch
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        model.eval()

        vocab_size = actual_vocab_size

    # Evaluate on validation set
    print(f"\n{'='*70}")
    print("VALIDATION EVALUATION")
    print(f"{'='*70}")
    print(f"Evaluating on {max_batches} batches...")

    total_loss = 0.0
    total_ce = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            if batch_idx >= max_batches:
                break

            input_ids, target_ids = batch

            # Compute loss (all gauge terms disabled for pure CE evaluation)
            loss, metrics = compute_free_energy_loss(
                model,
                input_ids,
                target_ids,
                alpha=0.0,
                lambda_beta=0.0,
                lambda_gamma=0.0,
                kappa_gamma=1.0,  # Unused when lambda_gamma=0
            )

            total_loss += loss.item()
            total_ce += metrics['loss/ce']
            num_batches += 1

            if (batch_idx + 1) % 10 == 0:
                print(f"  Batch {batch_idx + 1}/{max_batches}...")

    # Results
    avg_loss = total_loss / max(1, num_batches)
    avg_ce = total_ce / max(1, num_batches)
    perplexity = np.exp(avg_ce)

    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")
    print(f"Validation Loss: {avg_loss:.4f}")
    print(f"Validation PPL:  {perplexity:.2f}")
    print(f"\nComparison:")
    print(f"  Random baseline: ~2000 PPL")
    print(f"  Your model:      {perplexity:.2f} PPL")
    print(f"  Improvement:     {2000/perplexity:.1f}x better!")

    # Performance assessment
    print(f"\nAssessment:")
    if perplexity < 150:
        print("  ✨ EXCELLENT! Comparable to much larger models!")
    elif perplexity < 250:
        print("  ✓ GOOD! Model is learning well.")
    elif perplexity < 400:
        print("  ~ ACCEPTABLE. Room for improvement.")
    else:
        print("  ⚠ POOR. Model barely learning.")

    print(f"\n{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description='Evaluate checkpoint')
    parser.add_argument('--checkpoint', type=str,
                       default='checkpoints_realistic/best_model.pt',
                       help='Path to checkpoint file')
    parser.add_argument('--max_batches', type=int, default=50,
                       help='Number of validation batches to evaluate')

    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        print(f"\nAvailable checkpoints:")
        checkpoint_dir = checkpoint_path.parent
        if checkpoint_dir.exists():
            for f in sorted(checkpoint_dir.glob("*.pt")):
                print(f"  - {f}")
        return

    evaluate_checkpoint(str(checkpoint_path), args.max_batches)


if __name__ == '__main__':
    main()