# Gauge Transformer Architectures

## Overview

Both architectures replace learned Q, K projections with **geometry-based attention**:

```
β_ij = softmax(-KL(q_i || Ω_ij[q_j]) / κ)
```

where `Ω_ij = exp(φ_i) exp(-φ_j)` is the parallel transport operator.

---

## (a) Gauge-VFE Transformer

Beliefs evolve via **gradient descent on free energy**:

```
                    ┌─────────────────────┐
                    │   Token Embedding   │
                    │   (μ_p, Σ_p, φ_p)   │
                    └──────────┬──────────┘
                               │
              ┌────────────────┼────────────────┐
              │                ▼                │
              │    ┌───────────────────────┐    │
              │    │     KL-Attention      │    │
              │    │                       │    │
              │    │  β_ij = softmax(      │    │
              │    │    -KL(q_i||Ω[q_j])/κ │    │
              │    └───────────┬───────────┘    │
              │                │                │
              │                ▼                │
              │    ┌───────────────────────┐    │
    ×L layers │    │    VFE Descent FFN    │    │  + (residual)
              │    │                       │    │◄────────────┐
              │    │  μ ← μ - η ∇_μ F      │    │             │
              │    │  Σ ← Σ - η ∇_Σ F      │────┼─────────────┘
              │    └───────────┬───────────┘    │
              │                │                │
              └────────────────┼────────────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │    Output: W_out μ  │
                    └─────────────────────┘

Free Energy:
  F = α·KL(q||p) + λ_β·Σ_ij β_ij·KL(q_i||Ω_ij[q_j]) + CE(W_out·μ, y)
      ─────────   ───────────────────────────────────   ──────────────
      self-       belief alignment                      observation
      consistency (gauge-equivariant)                   term
```

**Key Properties:**
- Minimizes variational free energy F
- Gradient flow: beliefs move toward local minimum
- Detached FFN outputs → gradients via KL(q||p) term

---

## (b) Gauge-Hamiltonian Transformer

Beliefs evolve via **symplectic dynamics** on phase space:

```
                    ┌─────────────────────┐
                    │   Token Embedding   │
                    │   (μ_p, Σ_p, φ_p)   │
                    └──────────┬──────────┘
                               │
              ┌────────────────┼────────────────┐
              │                ▼                │
              │    ┌───────────────────────┐    │
              │    │     KL-Attention      │    │
              │    │                       │    │
              │    │  β_ij = softmax(      │    │
              │    │    -KL(q_i||Ω[q_j])/κ │    │
              │    └───────────┬───────────┘    │
              │                │                │
              │                ▼                │
              │    ┌───────────────────────┐    │
    ×L layers │    │   Leapfrog FFN        │    │  + (residual)
              │    │   (Symplectic)        │    │◄────────────┐
              │    │                       │    │             │
              │    │  q̇ = ∂H/∂p           │    │   ΔH ≈ 0
              │    │  ṗ = -∂H/∂q          │────┼─────────────┘
              │    └───────────┬───────────┘    │
              │                │                │
              └────────────────┼────────────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │    Output: W_out μ  │
                    └─────────────────────┘

Hamiltonian:
  H = T(q, p) + V(q)

Kinetic Energy (Riemannian metrics):
  T = T_μ + T_Σ + T_φ
    = ½ π_μᵀ Σ_p π_μ           (Fisher-Rao on means)
    + tr(π_Σ Σ π_Σ Σ)          (Affine-invariant on SPD)
    + ½ ||π_φ||²               (Killing form on so(3))

Potential Energy:
  V = F(μ, Σ, φ)               (Free energy functional)
```

**Key Properties:**
- Conserves Hamiltonian energy: ΔH ≈ 0
- Symplectic integration preserves phase space volume
- Curvature correction for SPD manifold: ∂T_Σ/∂Σ = 2π_Σ Σ π_Σ
- No vanishing gradients (energy conservation)

---

## (c) Belief Dynamics on Statistical Manifold

```
                    Statistical Manifold M
    ════════════════════════════════════════════════════════

                              ○ Posterior q (VFE)
                           ↗     gradient descent
                         /         on F
                       /
                     /
        ● Prior p ────────────────────△ Posterior q (Ham)
        (μ_p, Σ_p)        ~~~~~~~~~~~
                       Hamiltonian orbit
                       (energy-conserving)

    ════════════════════════════════════════════════════════

    ● Prior:     Token embedding (μ_p, Σ_p)
    ○ VFE:       Converges to local F minimum via ∇F
    △ Hamiltonian: Explores via energy-conserving dynamics
```

---

## Comparison Table

| Aspect | VFE | Hamiltonian |
|--------|-----|-------------|
| **Dynamics** | Gradient descent on F | Symplectic flow on (q, p) |
| **Conservation** | F decreases | H conserved |
| **Phase space** | Configuration only (q) | Full (q, p) |
| **SPD geometry** | Implicit in metric | Explicit curvature term |
| **Vanishing gradients** | Possible | Prevented by energy conservation |
| **Learned params** | Optional (η, step sizes) | None (pure physics!) |

---

## Gradient Flow Analysis

Both architectures provide gradients to embeddings via:

```
Embeddings ──→ Attention ──→ FFN (detached) ──→ Residual ──→ logits
    ↑             ↑              ×                 ↑
    │             │                                │
    └─────────────┴────────────────────────────────┘
    Gradients via:
    1. KL(q||p) self-consistency (α term)
    2. γ-term on priors (model alignment)
    3. Attention weights (pre-FFN)
    4. Residual connections
```

The self-consistency term KL(q||p) is **critical**: it pulls evolved beliefs back toward embedding priors and provides gradients even when FFN outputs are detached.

---

## SO(3) Irrep Structure (Multi-Head Attention)

Each head corresponds to an SO(3) irreducible representation:

```
embed_dim = K

┌────────┬────────┬────────┬────────┬─────┐
│  ℓ=0   │  ℓ=0   │  ℓ=1   │  ℓ=1   │ ... │
│ (dim 1)│ (dim 1)│ (dim 3)│ (dim 3)│     │
│ scalar │ scalar │ vector │ vector │     │
└────────┴────────┴────────┴────────┴─────┘

Each head uses proper Wigner D-matrix generators:
- ℓ=0: Zero generators (scalars don't transform)
- ℓ=1: Standard so(3) generators (3×3)
- ℓ=2: 5×5 Wigner D-matrices
- ...
```

---

## Mathematical Details

### Parallel Transport
```
Ω_ij = exp(φ_i · G) exp(-φ_j · G)

where G = (G_x, G_y, G_z) are SO(3) generators satisfying:
  [G_x, G_y] = G_z   (cyclic)
  G_aᵀ = -G_a        (skew-symmetric)
  C₂ = -Σ_a G_a² = ℓ(ℓ+1) I   (Casimir)
```

### SPD Exponential Map
```
exp_Σ(V) = Σ^{1/2} exp(Σ^{-1/2} V Σ^{-1/2}) Σ^{1/2}
```

### Leapfrog Integration
```
p_{1/2} = p_0 - (ε/2) ∂V/∂q(q_0)      # Half-step momentum
q_1     = q_0 + ε ∂T/∂p(p_{1/2})      # Full-step position
p_1     = p_{1/2} - (ε/2) ∂V/∂q(q_1)  # Half-step momentum

For SPD: includes curvature term ∂T_Σ/∂Σ = 2 π_Σ Σ π_Σ
```
