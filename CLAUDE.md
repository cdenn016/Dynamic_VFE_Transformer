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
