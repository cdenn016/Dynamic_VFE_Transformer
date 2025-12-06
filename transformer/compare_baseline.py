# -*- coding: utf-8 -*-
"""
Created on Fri Nov 14 15:43:00 2025

@author: chris and christine
"""

"""
Compare Gauge vs Standard Transformer Results
==============================================

Analyzes results from ablation study to determine:
    1. Does SO(3) gauge structure help or hurt performance?
    2. How much parameter efficiency gain from removing W_Q, W_K?
    3. Is KL-divergence attention better than dot-product?

Loads results from:
    - checkpoints_publication/ffn_learned/
    - checkpoints_publication/ffn_variational_approx/
    - checkpoints_publication/ffn_variational_full/
    - checkpoints_publication/standard_baseline_debug/

Author: Ablation analysis
Date: November 2025
"""

import json
from pathlib import Path
import sys


def load_results(checkpoint_dir: str) -> dict:
    """Load training results from checkpoint directory."""
    log_path = Path(checkpoint_dir) / 'training_log.json'

    if not log_path.exists():
        return None

    with open(log_path, 'r') as f:
        data = json.load(f)

    return data


def compare_results():
    """Compare all experimental results."""

    print("="*70)
    print("GAUGE vs STANDARD TRANSFORMER COMPARISON")
    print("="*70)

    # Load gauge model results
    gauge_results = {
        'learned': load_results('checkpoints_publication/ffn_learned'),
        'variational_approx': load_results('checkpoints_publication/ffn_variational_approx'),
        'variational_full': load_results('checkpoints_publication/ffn_variational_full'),
    }

    # Load standard baseline results
    standard_results = load_results('checkpoints_publication/standard_baseline_debug')

    # Check what data we have
    print("\n" + "="*70)
    print("DATA AVAILABILITY")
    print("="*70)

    for name, data in gauge_results.items():
        status = "✓" if data is not None else "✗"
        print(f"  {status} Gauge ({name})")

    status = "✓" if standard_results is not None else "✗"
    print(f"  {status} Standard baseline")

    # Performance comparison
    print("\n" + "="*70)
    print("PERFORMANCE COMPARISON")
    print("="*70)
    print(f"\n{'Model':<30s} {'Val PPL':>10s} {'Val Loss':>10s} {'vs Random':>10s}")
    print("-"*70)

    results_table = []

    # Gauge models
    for name, data in gauge_results.items():
        if data is not None:
            ppl = data['best_val_ppl']
            loss = data['best_val_loss']
            improvement = data.get('improvement_over_random', 0)
            results_table.append((f"Gauge ({name})", ppl, loss, improvement))
            print(f"{'Gauge (' + name + ')':<30s} {ppl:>10.2f} {loss:>10.4f} {improvement:>9.1f}x")

    # Standard baseline
    if standard_results is not None:
        ppl = standard_results['best_val_ppl']
        loss = standard_results['best_val_loss']
        improvement = standard_results.get('improvement_over_random', 0)
        results_table.append(("Standard baseline", ppl, loss, improvement))
        print(f"{'Standard baseline':<30s} {ppl:>10.2f} {loss:>10.4f} {improvement:>9.1f}x")

    # Analysis
    if standard_results is not None and all(data is not None for data in gauge_results.values()):
        print("\n" + "="*70)
        print("ANALYSIS")
        print("="*70)

        standard_ppl = standard_results['best_val_ppl']

        # Compare each gauge variant to standard
        print("\nGauge vs Standard:")
        for name, data in gauge_results.items():
            gauge_ppl = data['best_val_ppl']
            diff_pct = ((gauge_ppl - standard_ppl) / standard_ppl) * 100

            if diff_pct < -5:
                verdict = "🟢 BETTER"
            elif diff_pct > 5:
                verdict = "🔴 WORSE"
            else:
                verdict = "🟡 COMPARABLE"

            print(f"  Gauge ({name:20s}): {diff_pct:+6.1f}% {verdict}")

        # Overall verdict
        print("\n" + "="*70)
        print("CONCLUSION")
        print("="*70)

        avg_gauge_ppl = sum(d['best_val_ppl'] for d in gauge_results.values()) / len(gauge_results)
        diff = ((avg_gauge_ppl - standard_ppl) / standard_ppl) * 100

        print(f"\nAverage Gauge PPL: {avg_gauge_ppl:.2f}")
        print(f"Standard PPL:      {standard_ppl:.2f}")
        print(f"Difference:        {diff:+.1f}%")

        if diff < -5:
            print("\n🎯 RESULT: Gauge model OUTPERFORMS standard transformer")
            print("   → SO(3) gauge structure + KL attention is BENEFICIAL")
            print("   → Free inductive bias hypothesis VALIDATED ✓")
        elif diff > 5:
            print("\n⚠️  RESULT: Standard transformer OUTPERFORMS gauge model")
            print("   → SO(3) gauge structure is CONSTRAINING performance")
            print("   → Free inductive bias hypothesis REJECTED ✗")
            print("   → Recommendation: Learn symmetry structure from data")
        else:
            print("\n➖ RESULT: Performance is COMPARABLE")
            print("   → SO(3) gauge structure is neutral")
            print("   → Benefits: Interpretability, parameter efficiency")
            print("   → Drawback: No clear performance advantage")

    # Parameter comparison
    print("\n" + "="*70)
    print("PARAMETER EFFICIENCY")
    print("="*70)

    if gauge_results['learned'] is not None:
        gauge_config = gauge_results['learned']['config']
        print(f"\nGauge Model:")
        print(f"  Embedding dim (K):  {gauge_config['embed_dim']}")
        print(f"  Layers:             {gauge_config['n_layers']}")
        print(f"  Total params:       ~5,334")
        print(f"  Attention params:   ~0 (no W_Q, W_K!)")

    if standard_results is not None:
        standard_config = standard_results['config']
        print(f"\nStandard Transformer:")
        print(f"  Embedding dim (K):  {standard_config['embed_dim']}")
        print(f"  Layers:             {standard_config['n_layers']}")
        print(f"  Heads:              {standard_config['n_heads']}")
        print(f"  Total params:       ~5,500")
        print(f"  Attention params:   ~484 (W_Q, W_K, W_V, W_O)")

    # Speed comparison
    print("\n" + "="*70)
    print("TRAINING EFFICIENCY")
    print("="*70)

    if gauge_results['learned'] is not None:
        gauge_time = gauge_results['learned'].get('total_time_seconds', 0) / 60
        print(f"\nGauge Model (learned FFN):")
        print(f"  Training time: {gauge_time:.1f} minutes")
        print(f"  Learning rate: {gauge_config.get('mu_lr', 0.05)} (natural gradients!)")

    if standard_results is not None:
        standard_time = standard_results.get('total_time_seconds', 0) / 60
        print(f"\nStandard Transformer:")
        print(f"  Training time: {standard_time:.1f} minutes")
        print(f"  Learning rate: {standard_config.get('lr', 0.001)} (standard)")

    print("\n" + "="*70)


if __name__ == '__main__':
    compare_results()