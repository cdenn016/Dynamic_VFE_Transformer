"""
Baseline Comparison Framework for Hamiltonian Transformer.

Compares against:
1. Standard Vaswani Transformer (baseline)
2. VFE Transformer (gradient-based variant)

Metrics:
- Perplexity / BPC
- Training speed (tokens/sec)
- Memory usage
- Interpretability (reversibility for Hamiltonian)

For RTX 5090 scale experiments.
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import time
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from pathlib import Path
import json
import csv


@dataclass
class ModelMetrics:
    """Metrics collected during training/evaluation."""
    model_name: str
    config: Dict[str, Any]
    train_loss: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    train_bpc: List[float] = field(default_factory=list)
    val_bpc: List[float] = field(default_factory=list)
    tokens_per_sec: List[float] = field(default_factory=list)
    memory_mb: List[float] = field(default_factory=list)
    step_times: List[float] = field(default_factory=list)
    reversibility_error: List[float] = field(default_factory=list)  # Hamiltonian only

    def to_dict(self) -> Dict:
        return {
            'model_name': self.model_name,
            'config': self.config,
            'train_loss': self.train_loss,
            'val_loss': self.val_loss,
            'train_bpc': self.train_bpc,
            'val_bpc': self.val_bpc,
            'tokens_per_sec': self.tokens_per_sec,
            'memory_mb': self.memory_mb,
            'step_times': self.step_times,
            'reversibility_error': self.reversibility_error,
        }

    def save(self, path: Path):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> 'ModelMetrics':
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)


class BaselineTrainer:
    """
    Unified training framework for model comparisons.

    Provides consistent training loop for:
    - Vaswani Transformer
    - VFE Transformer
    - Hamiltonian Transformer
    """

    def __init__(
        self,
        model: nn.Module,
        model_name: str,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any] = None,
        device: str = 'cuda',
        log_interval: int = 100,
        eval_interval: int = 500,
    ):
        self.model = model.to(device)
        self.model_name = model_name
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.log_interval = log_interval
        self.eval_interval = eval_interval

        self.metrics = ModelMetrics(
            model_name=model_name,
            config=self._get_config(),
        )

    def _get_config(self) -> Dict:
        """Extract model configuration."""
        config = {
            'model_class': self.model.__class__.__name__,
            'device': self.device,
        }
        # Try to get model-specific config
        if hasattr(self.model, 'config'):
            config.update(self.model.config)
        if hasattr(self.model, 'embed_dim'):
            config['embed_dim'] = self.model.embed_dim
        if hasattr(self.model, 'n_layers'):
            config['n_layers'] = self.model.n_layers
        return config

    def train_step(self, batch: torch.Tensor) -> Tuple[float, int]:
        """Single training step. Returns (loss, n_tokens)."""
        self.model.train()

        x = batch[:, :-1].to(self.device)
        y = batch[:, 1:].to(self.device)

        self.optimizer.zero_grad()
        start = time.perf_counter()

        # Forward
        logits = self.model(x)
        loss = nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1),
        )

        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        if self.scheduler:
            self.scheduler.step()

        step_time = time.perf_counter() - start
        n_tokens = y.numel()

        return loss.item(), n_tokens, step_time

    @torch.no_grad()
    def evaluate(self, dataloader) -> Tuple[float, float]:
        """Evaluate on dataset. Returns (loss, bpc)."""
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0

        for batch in dataloader:
            x = batch[:, :-1].to(self.device)
            y = batch[:, 1:].to(self.device)

            logits = self.model(x)
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                y.reshape(-1),
                reduction='sum',
            )
            total_loss += loss.item()
            total_tokens += y.numel()

        avg_loss = total_loss / total_tokens
        bpc = avg_loss / np.log(2)

        return avg_loss, bpc

    def get_memory_usage(self) -> float:
        """Get current GPU memory usage in MB."""
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 1024 / 1024
        return 0.0

    def train_epoch(
        self,
        train_loader,
        val_loader=None,
        epoch: int = 0,
    ) -> Dict[str, float]:
        """Train for one epoch."""
        epoch_loss = 0.0
        epoch_tokens = 0
        epoch_time = 0.0

        for step, batch in enumerate(train_loader):
            loss, n_tokens, step_time = self.train_step(batch)

            epoch_loss += loss * n_tokens
            epoch_tokens += n_tokens
            epoch_time += step_time

            if step % self.log_interval == 0:
                tps = n_tokens / step_time
                print(f"  Step {step}: loss={loss:.4f}, {tps:.1f} tok/s")
                self.metrics.tokens_per_sec.append(tps)
                self.metrics.step_times.append(step_time)

        avg_loss = epoch_loss / epoch_tokens
        bpc = avg_loss / np.log(2)
        avg_tps = epoch_tokens / epoch_time

        self.metrics.train_loss.append(avg_loss)
        self.metrics.train_bpc.append(bpc)
        self.metrics.memory_mb.append(self.get_memory_usage())

        result = {
            'train_loss': avg_loss,
            'train_bpc': bpc,
            'tokens_per_sec': avg_tps,
            'memory_mb': self.get_memory_usage(),
        }

        if val_loader:
            val_loss, val_bpc = self.evaluate(val_loader)
            self.metrics.val_loss.append(val_loss)
            self.metrics.val_bpc.append(val_bpc)
            result['val_loss'] = val_loss
            result['val_bpc'] = val_bpc

        return result


class ComparisonReport:
    """
    Generate comparison reports between models.
    """

    def __init__(self, output_dir: str = "./comparison_outputs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results: Dict[str, ModelMetrics] = {}

    def add_result(self, metrics: ModelMetrics):
        """Add model results to comparison."""
        self.results[metrics.model_name] = metrics

    def generate_table(self) -> str:
        """Generate comparison table."""
        lines = [
            "=" * 80,
            "MODEL COMPARISON REPORT",
            "=" * 80,
            "",
            f"{'Model':<25} {'Val BPC':>10} {'Tok/s':>10} {'Memory':>10} {'Reversible':>12}",
            "-" * 80,
        ]

        for name, m in self.results.items():
            val_bpc = m.val_bpc[-1] if m.val_bpc else float('nan')
            tps = np.mean(m.tokens_per_sec) if m.tokens_per_sec else 0
            mem = m.memory_mb[-1] if m.memory_mb else 0
            rev = m.reversibility_error[-1] if m.reversibility_error else "N/A"
            if isinstance(rev, float):
                rev = f"{rev:.2e}"

            lines.append(f"{name:<25} {val_bpc:>10.3f} {tps:>10.1f} {mem:>10.1f} {rev:>12}")

        lines.extend(["-" * 80, ""])

        return "\n".join(lines)

    def save_csv(self, filename: str = "comparison.csv"):
        """Save results to CSV."""
        path = self.output_dir / filename

        rows = []
        for name, m in self.results.items():
            row = {
                'model': name,
                'final_val_bpc': m.val_bpc[-1] if m.val_bpc else None,
                'final_val_loss': m.val_loss[-1] if m.val_loss else None,
                'avg_tokens_per_sec': np.mean(m.tokens_per_sec) if m.tokens_per_sec else None,
                'max_memory_mb': max(m.memory_mb) if m.memory_mb else None,
                'reversibility': m.reversibility_error[-1] if m.reversibility_error else None,
            }
            rows.append(row)

        if rows:
            with open(path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)

        print(f"Saved comparison to {path}")

    def plot_learning_curves(
        self,
        save_name: str = "learning_curves.png",
    ):
        """Plot learning curves for all models."""
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Train loss
        ax = axes[0, 0]
        for name, m in self.results.items():
            if m.train_loss:
                ax.plot(m.train_loss, label=name)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Train Loss')
        ax.set_title('Training Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Val BPC
        ax = axes[0, 1]
        for name, m in self.results.items():
            if m.val_bpc:
                ax.plot(m.val_bpc, label=name)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Validation BPC')
        ax.set_title('Validation BPC (lower is better)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Throughput
        ax = axes[1, 0]
        names = list(self.results.keys())
        tps = [np.mean(m.tokens_per_sec) if m.tokens_per_sec else 0 for m in self.results.values()]
        bars = ax.bar(names, tps)
        ax.set_ylabel('Tokens/second')
        ax.set_title('Training Throughput')
        ax.tick_params(axis='x', rotation=45)

        # Memory
        ax = axes[1, 1]
        mem = [max(m.memory_mb) if m.memory_mb else 0 for m in self.results.values()]
        bars = ax.bar(names, mem)
        ax.set_ylabel('Peak Memory (MB)')
        ax.set_title('Memory Usage')
        ax.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        fig.savefig(self.output_dir / save_name, dpi=150, bbox_inches='tight')
        print(f"Saved learning curves to {self.output_dir / save_name}")

        return fig


# =============================================================================
# Vaswani Baseline Implementation (Minimal)
# =============================================================================

class VaswaniTransformerBlock(nn.Module):
    """Standard transformer block (Vaswani et al., 2017)."""

    def __init__(self, embed_dim: int, n_heads: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, n_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout),
        )
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention with residual
        attn_out, _ = self.attention(x, x, x, attn_mask=mask, need_weights=False)
        x = self.ln1(x + self.dropout(attn_out))

        # FFN with residual
        ff_out = self.ff(x)
        x = self.ln2(x + ff_out)

        return x


class VaswaniTransformer(nn.Module):
    """
    Standard Vaswani Transformer for baseline comparison.

    This is a minimal implementation for benchmarking against
    the Hamiltonian and VFE transformers.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        n_heads: int,
        n_layers: int,
        ff_dim: int,
        max_seq_len: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_layers = n_layers

        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Embedding(max_seq_len, embed_dim)

        self.blocks = nn.ModuleList([
            VaswaniTransformerBlock(embed_dim, n_heads, ff_dim, dropout)
            for _ in range(n_layers)
        ])

        self.ln_final = nn.LayerNorm(embed_dim)
        self.output_proj = nn.Linear(embed_dim, vocab_size)

        self.config = {
            'vocab_size': vocab_size,
            'embed_dim': embed_dim,
            'n_heads': n_heads,
            'n_layers': n_layers,
            'ff_dim': ff_dim,
            'max_seq_len': max_seq_len,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T = x.shape
        device = x.device

        # Embeddings
        tok_emb = self.token_embedding(x)
        pos = torch.arange(T, device=device).unsqueeze(0)
        pos_emb = self.pos_embedding(pos)
        x = tok_emb + pos_emb

        # Causal mask
        mask = torch.triu(torch.ones(T, T, device=device), diagonal=1).bool()

        # Transformer blocks
        for block in self.blocks:
            x = block(x, mask)

        x = self.ln_final(x)
        logits = self.output_proj(x)

        return logits


# =============================================================================
# Comparison Runner
# =============================================================================

def run_comparison(
    train_loader,
    val_loader,
    vocab_size: int,
    embed_dim: int = 128,
    n_heads: int = 4,
    n_layers: int = 4,
    n_epochs: int = 5,
    device: str = 'cuda',
) -> ComparisonReport:
    """
    Run full comparison between Vaswani and Hamiltonian transformers.

    Args:
        train_loader: Training data loader
        val_loader: Validation data loader
        vocab_size: Vocabulary size
        embed_dim: Model dimension
        n_heads: Number of attention heads
        n_layers: Number of transformer layers
        n_epochs: Training epochs
        device: Device to train on

    Returns:
        ComparisonReport with all results
    """
    report = ComparisonReport()

    # 1. Train Vaswani baseline
    print("\n" + "=" * 60)
    print("Training Vaswani Transformer Baseline")
    print("=" * 60)

    vaswani = VaswaniTransformer(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        n_heads=n_heads,
        n_layers=n_layers,
        ff_dim=embed_dim * 4,
    )
    optimizer = torch.optim.AdamW(vaswani.parameters(), lr=1e-3)

    trainer = BaselineTrainer(
        model=vaswani,
        model_name="Vaswani",
        optimizer=optimizer,
        device=device,
    )

    for epoch in range(n_epochs):
        print(f"\nEpoch {epoch + 1}/{n_epochs}")
        result = trainer.train_epoch(train_loader, val_loader, epoch)
        print(f"  Train BPC: {result['train_bpc']:.3f}, Val BPC: {result.get('val_bpc', 'N/A')}")

    report.add_result(trainer.metrics)

    # 2. Hamiltonian transformer would be trained here
    # (Requires the full HamiltonianTransformer implementation)
    print("\n[Note: Add Hamiltonian transformer training for full comparison]")

    # Generate report
    print("\n" + report.generate_table())
    report.save_csv()
    report.plot_learning_curves()

    return report


# =============================================================================
# Demo
# =============================================================================

def demo():
    """Demo comparison with synthetic data."""
    print("=" * 70)
    print("BASELINE COMPARISON FRAMEWORK DEMO")
    print("=" * 70)

    # Create synthetic data
    vocab_size = 256
    seq_len = 64
    batch_size = 8
    n_batches = 10

    # Mock data loaders
    def make_loader():
        for _ in range(n_batches):
            yield torch.randint(0, vocab_size, (batch_size, seq_len))

    print("\nCreating Vaswani baseline...")
    model = VaswaniTransformer(
        vocab_size=vocab_size,
        embed_dim=64,
        n_heads=4,
        n_layers=2,
        ff_dim=256,
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    trainer = BaselineTrainer(
        model=model,
        model_name="Vaswani-Demo",
        optimizer=optimizer,
        device=device,
        log_interval=2,
    )

    print("\nTraining for 1 epoch...")
    result = trainer.train_epoch(list(make_loader()))

    print(f"\nResults:")
    print(f"  Train Loss: {result['train_loss']:.4f}")
    print(f"  Train BPC: {result['train_bpc']:.3f}")
    print(f"  Throughput: {result['tokens_per_sec']:.1f} tok/s")
    print(f"  Memory: {result['memory_mb']:.1f} MB")

    # Generate report
    report = ComparisonReport()
    report.add_result(trainer.metrics)
    print("\n" + report.generate_table())

    print("\n✓ Baseline comparison framework demo complete")


if __name__ == "__main__":
    demo()
