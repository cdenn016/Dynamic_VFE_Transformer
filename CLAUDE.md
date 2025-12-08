# Claude Code Guidelines for Gauge Transformer Project

## Domain Expertise

Apply these when working on this codebase:

- **Differential Geometry**: SPD manifolds, geodesics, affine-invariant metrics, Lie theory, fiber bundles
- **Variational Inference**: KL divergence, free energy, ELBO, information geometry
- **Gauge Theory**: Symmetries, equivariance, parallel transport, irreps
- **Hamiltonian Mechanics**: Symplectic structure, phase space, mass matrices
- **Matrix/Linear Algebra**: Eigendecomposition, Kronecker products, matrix exponentials

## Code Standards

- Write modular, testable functions with type hints
- Docstrings should include LaTeX math where relevant
- Variable names should match paper notation (e.g., `mu_q` for μ_q, `Sigma` for Σ)
- Check tensor shapes at each step when debugging
- Verify gradient flow with small-dim smoke tests

## Numerical Stability

- Use log-sum-exp for softmax operations
- Cholesky factorization for SPD matrices
- Epsilon padding for matrix inverses
- Eigenvalue clamping to maintain positive definiteness

## Testing

- Property-based tests for mathematical invariants (symmetry, PSD)
- NumPy reference implementations for comparison
- Edge case coverage (single agent, identity matrices, zero inputs)

## Known Limitations

### Σ Reversibility in Hamiltonian Dynamics

The leapfrog integrator achieves excellent reversibility for μ (~10⁻⁷ error) but has higher drift for Σ (~10⁻² error over many steps). This is a fundamental limitation, not a bug:

- **μ dynamics**: T_μ uses constant mass matrix (prior Σ_p) → exactly symplectic
- **Σ dynamics**: T_Σ = tr(π_Σ Σ π_Σ Σ) has position-dependent mass → explicit leapfrog is only 1st-order accurate on curved SPD manifold

The `spd_kinetic_gradient` correction improves but doesn't achieve exact symplecticity. Options for exact Σ reversibility (implicit midpoint, RATTLE) would add 10-20× computational cost.

**Practical impact**: Negligible for typical use (1-4 integration steps per layer). Token attribution via μ reversal works excellently. Only affects stress tests with 100+ steps.

**Future work** (RTX 5090 training):
- Robustly test reversibility at scale (seq_len 32, 64, 128+)
- Visualize phase space trajectories for actual language sentences
- Token attribution examples showing input→output causal chains
- Compare with Vaswani baseline to demonstrate interpretability advantage

## Communication Style

**Be direct:**
- State errors and concerns plainly without excessive hedging
- "This is wrong because X" not "This might potentially be slightly off"

**Push back:**
- Challenge gaps in derivations, ask for justification
- If a claim needs proof, ask for it

**Skip praise preambles:**
- No "Great question!" openers—just answer
- No "Excellent point!"—just engage with the substance

**Flag simpler alternatives:**
- Call out over-engineering
- Ask what complexity buys if something seems unnecessarily elaborate

**Maintain position under pushback:**
- Don't fold immediately when disagreeing
- Ask "What am I missing?" rather than capitulating

**Honest uncertainty:**
- "I'm not sure this is right" beats confident speculation
- Acknowledge when something needs verification

## Hamiltonian Transformer Architecture Summary

### Core Difference from Standard Transformer

| Component | Standard Transformer | Hamiltonian Transformer |
|-----------|---------------------|------------------------|
| Attention | Q, K, V projections | KL-divergence + parallel transport (no W_Q, W_K) |
| FFN | Learned MLP + GELU | Leapfrog integration on free energy landscape |
| Nonlinearity | GELU/ReLU (ad hoc) | ∂β/∂μ (emerges from attention gradients) |
| FFN params | ~2/3 of model | Zero learned MLP weights |

### Beliefs vs Priors

| Symbol | Name | Learned? | Evolves in forward pass? |
|--------|------|----------|-------------------------|
| μ_p, Σ_p | Prior (embedding) | Yes | No |
| μ_q, Σ_q | Belief (posterior) | No | Yes, via Hamiltonian |

Training shapes the prior landscape. The forward pass evolves beliefs through that landscape.

### The Mass Matrix (Inertia of Belief)

Full formula from the paper:

```
M_i = Λ_{pi} + Λ_{oi} + Σ_k β_{ik} Λ̃_{qk} + Σ_j β_{ji} Λ_{qi}
      ↑        ↑              ↑                   ↑
    prior    obs       incoming attention    outgoing recoil
```

Where:
- **Λ_p = Σ_p⁻¹**: Prior precision (resistance from prior expectations)
- **Λ_o**: Observation precision (sensory grounding) — see below
- **Σ_k β_{ik} Λ̃_{qk}**: Incoming social precision (pulled toward confident neighbors)
- **Σ_j β_{ji} Λ_{qi}**: Outgoing recoil precision (Newton's 3rd law)
- **Λ̃_{qk} = Ω_{ik} Λ_{qk} Ω_{ik}^T**: Transported precision via gauge connection

**Physical interpretation**: Precision = Mass. Confident beliefs are heavy, resist change.

### Categorical Observation Precision (Transformer-Specific)

For transformers with softmax output p = softmax(W_out @ μ / τ):

```
Λ_o = (1/τ²) W^T (diag(p) - pp^T) W = (1/τ²) Cov_p(W)
```

This is the **Hessian of cross-entropy** with respect to μ:
- When p is peaked (confident): Λ_o has low rank, weak constraint
- When p is uniform (uncertain): Λ_o reflects full embedding structure
- Temperature τ scales precision (lower τ → higher precision)

### The Nonlinearity

Standard transformer: GELU(x) — ad hoc, nobody knows why it works

Ours: ∂β_{ij}/∂θ — emerges from differentiating softmax attention:

```
β_{ij} = softmax(-KL_{ij} / κ)

∂β_{ij}/∂μ_i = β_{ij} · [∂KL_{ij}/∂μ_i - Σ_k β_{ik} · ∂KL_{ik}/∂μ_i] / κ
∂β_{ij}/∂Σ_i = β_{ij} · [∂KL_{ij}/∂Σ_i - Σ_k β_{ik} · ∂KL_{ik}/∂Σ_i] / κ
∂β_{ij}/∂φ_i = β_{ij} · [∂KL_{ij}/∂φ_i - Σ_k β_{ik} · ∂KL_{ik}/∂φ_i] / κ
```

**Implementation detail**: In Hamiltonian FFN, β is computed once in the attention layer and held **fixed** during leapfrog integration (like standard transformers). The ∂β/∂θ nonlinearity affects **training gradients** (backprop through attention), not forward dynamics. This separates "what to attend to" (attention layer) from "how beliefs evolve" (FFN dynamics).

### Reversibility Scope

| Component | Reversible? | Why? |
|-----------|-------------|------|
| Hamiltonian FFN | Yes, μ to 10⁻⁷, Σ to 10⁻² | Symplectic integrator, negate momentum |
| Attention | No | Many-to-one weighted average (information loss) |
| Full transformer | No | Attention breaks it |
| logits → token | No | argmax is discrete |

### The Central Tension (Resolved)

| Mode | Converges to minimum? | Reversible? |
|------|----------------------|-------------|
| Pure Hamiltonian (γ=0) | No, orbits | Yes |
| Damped (γ>0) | Yes | No |

**Resolution**: We're not optimizing during forward pass. The Hamiltonian FFN is a transformation, not a solver. The orbit after n_steps IS the output. Training adjusts embeddings so this arc produces good predictions.

### What We Claim

1. FFN replaced with zero-parameter Hamiltonian dynamics
2. FFN exactly reversible (μ to 10⁻⁷, Σ to 10⁻²)
3. Nonlinearity principled (emerges from ∂β/∂μ), not ad hoc
4. Mass matrix incorporates categorical observation likelihood
5. Comparable performance to learned MLP (preliminary small-sequence tests)

### What We Don't Claim (Yet)

1. Full transformer reversibility (attention is lossy)
2. Token → input attribution (would need cached attention states)
3. Large-scale benchmarks (WikiText, etc.) — pending compute

