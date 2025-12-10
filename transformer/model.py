"""
Complete Gauge-Theoretic Language Model (0D Architecture)
==========================================================

Full transformer language model using gauge theory and active inference.

Architecture:
    Token Embedding → Position Encoding →
    N × Transformer Blocks → Output Projection

Key Innovation: Attention via KL divergence on statistical manifold,
                no learned W_Q, W_K matrices!

0D Structure: All agents at single point c* (standard transformer topology)

Author: Implementation from plan.py
Date: November 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List, Union
import numpy as np

# Import our components
from transformer.embeddings import GaugeTokenEmbedding, GaugePositionalEncoding
from transformer.transformer_block import GaugeTransformerStack
from transformer.attention import create_attention_mask

# Try to import generators (fallback to random if unavailable)
try:
    from math_utils.generators import generate_so3_generators
    GENERATORS_AVAILABLE = True
except ImportError:
    GENERATORS_AVAILABLE = False


class GaugeTransformerLM(nn.Module):
    """
    Complete gauge-theoretic language model.

    Architecture Flow:
        token_ids → (μ, Σ, φ) → Transformer Stack → μ_final → logits

    Components:
        1. GaugeTokenEmbedding: Maps tokens to beliefs
        2. GaugePositionalEncoding: Agent-index encoding in so(3)
        3. GaugeTransformerStack: N layers of gauge attention
        4. Output projection: μ → logits over vocabulary

    0D Structure:
        - All N tokens → N agents at single point c*
        - No spatial structure, sequence via agent index
        - Attention β_ij are scalars (not fields)
    """

    def __init__(self, config: Dict):
        """
        Initialize gauge transformer language model.

        Args:
            config: Dictionary with model hyperparameters:
                - vocab_size: Vocabulary size
                - embed_dim: Embedding dimension K
                - n_layers: Number of transformer blocks
                - irrep_spec: Irrep structure [(label, mult, dim), ...]
                - hidden_dim: FFN hidden dimension
                - max_seq_len: Maximum sequence length
                - kappa_beta: Attention temperature
                - dropout: Dropout probability
                - pos_encoding_mode: 'learned' or 'sinusoidal'
                - evolve_sigma: If True, evolve covariances
                - evolve_phi: If True, evolve gauge frames
                - tie_embeddings: If True, tie input/output embeddings
        """
        super().__init__()
        self.config = config

        # Extract config
        vocab_size = config['vocab_size']
        embed_dim = config['embed_dim']
        n_layers = config['n_layers']
        irrep_spec = config['irrep_spec']
        hidden_dim = config['hidden_dim']
        max_seq_len = config['max_seq_len']
        kappa_beta = config['kappa_beta']
        dropout = config.get('dropout', 0.1)
        pos_mode = config.get('pos_encoding_mode', 'learned')
        evolve_sigma = config.get('evolve_sigma', False)
        evolve_phi = config.get('evolve_phi', False)
        tie_embeddings = config.get('tie_embeddings', True)

        # Variational FFN config
        ffn_mode = config.get('ffn_mode', 'learned')
        ffn_alpha = config.get('ffn_alpha', 0.001)
        ffn_tau_eff = config.get('ffn_tau_eff', 1.0)
        ffn_kappa = config.get('ffn_kappa', 1.0)
        ffn_n_iterations = config.get('ffn_n_iterations', 1)
        ffn_learnable_lr = config.get('ffn_learnable_lr', True)

        # Gradient engine config (for variational_gradient_engine mode)
        ffn_lambda_belief = config.get('ffn_lambda_belief', 1.0)
        ffn_lambda_prior = config.get('ffn_lambda_prior', 0.0)
        ffn_lambda_phi = config.get('ffn_lambda_phi', 0.0)
        ffn_update_sigma = config.get('ffn_update_sigma', True)

        # Hamiltonian FFN config
        ffn_hamiltonian_dt = config.get('ffn_hamiltonian_dt', 0.01)
        ffn_hamiltonian_n_steps = config.get('ffn_hamiltonian_n_steps', 10)
        ffn_hamiltonian_momentum_scale = config.get('ffn_hamiltonian_momentum_scale', 1.0)
        ffn_hamiltonian_gamma = config.get('ffn_hamiltonian_gamma', 0.0)

        # Hamiltonian mass config (from Inertia of Belief paper)
        ffn_hamiltonian_mass_use_prior = config.get('ffn_hamiltonian_mass_use_prior', True)
        ffn_hamiltonian_mass_use_observation = config.get('ffn_hamiltonian_mass_use_observation', False)
        ffn_hamiltonian_mass_use_incoming_social = config.get('ffn_hamiltonian_mass_use_incoming_social', False)
        ffn_hamiltonian_mass_use_outgoing_recoil = config.get('ffn_hamiltonian_mass_use_outgoing_recoil', False)
        ffn_hamiltonian_evolve_mass = config.get('ffn_hamiltonian_evolve_mass', False)

        # Gauge-fixed priors (for gauge covariance)
        gauge_fixed_priors = config.get('gauge_fixed_priors', False)

        # Diagonal covariance mode (memory optimization)
        diagonal_covariance = config.get('diagonal_covariance', False)
        self.diagonal_covariance = diagonal_covariance

        # Store evolve_phi for cross-layer transport caching optimization
        self.evolve_phi = evolve_phi

        # Sparse attention/FFN config
        self.attention_pattern = config.get('attention_pattern', 'full')
        self.attention_window = config.get('attention_window', 64)
        self.ffn_pattern = config.get('ffn_pattern', 'full')
        self.ffn_window = config.get('ffn_window', 64)

        # =================================================================
        # SO(3) Generators
        # =================================================================
        if GENERATORS_AVAILABLE:
            generators = generate_so3_generators(embed_dim)
        else:
            # Fallback: random skew-symmetric matrices
            generators = np.random.randn(3, embed_dim, embed_dim)
            generators = 0.5 * (generators - generators.transpose(0, 2, 1))

        self.register_buffer(
            'generators',
            torch.from_numpy(generators).float()
        )

        # =================================================================
        # Embedding Layers
        # =================================================================
        self.token_embed = GaugeTokenEmbedding(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            irrep_spec=irrep_spec,
            init_std=0.02,
            init_sigma_scale=0.1,
            learnable_sigma=False,  # Keep simple for now
            learnable_phi=gauge_fixed_priors,  # Enable phi learning if gauge_fixed_priors
            gauge_fixed_priors=gauge_fixed_priors,
            generators=self.generators if gauge_fixed_priors else None,
            diagonal_covariance=diagonal_covariance,
        )

        self.pos_encoding = GaugePositionalEncoding(
            max_seq_len=max_seq_len,
            mode=pos_mode,
            scale=0.1,
        )

        # =================================================================
        # Transformer Stack
        # =================================================================
        self.transformer = GaugeTransformerStack(
            n_layers=n_layers,
            embed_dim=embed_dim,
            irrep_spec=irrep_spec,
            hidden_dim=hidden_dim,
            kappa_beta=kappa_beta,
            dropout=dropout,
            evolve_sigma=evolve_sigma,
            evolve_phi=evolve_phi,
            # Variational FFN parameters
            generators=self.generators,
            ffn_mode=ffn_mode,
            ffn_alpha=ffn_alpha,
            ffn_tau_eff=ffn_tau_eff,
            ffn_kappa=ffn_kappa,
            ffn_n_iterations=ffn_n_iterations,
            ffn_learnable_lr=ffn_learnable_lr,
            # Gradient engine parameters
            ffn_lambda_belief=ffn_lambda_belief,
            ffn_lambda_prior=ffn_lambda_prior,
            ffn_lambda_phi=ffn_lambda_phi,
            ffn_update_sigma=ffn_update_sigma,
            # Hamiltonian parameters
            ffn_hamiltonian_dt=ffn_hamiltonian_dt,
            ffn_hamiltonian_n_steps=ffn_hamiltonian_n_steps,
            ffn_hamiltonian_momentum_scale=ffn_hamiltonian_momentum_scale,
            ffn_hamiltonian_gamma=ffn_hamiltonian_gamma,
            # Hamiltonian mass config (from Inertia of Belief paper)
            ffn_hamiltonian_mass_use_prior=ffn_hamiltonian_mass_use_prior,
            ffn_hamiltonian_mass_use_observation=ffn_hamiltonian_mass_use_observation,
            ffn_hamiltonian_mass_use_incoming_social=ffn_hamiltonian_mass_use_incoming_social,
            ffn_hamiltonian_mass_use_outgoing_recoil=ffn_hamiltonian_mass_use_outgoing_recoil,
            ffn_hamiltonian_evolve_mass=ffn_hamiltonian_evolve_mass,
            diagonal_covariance=diagonal_covariance,
        )

        # =================================================================
        # Output Projection
        # =================================================================
        self.out_proj = nn.Linear(embed_dim, vocab_size, bias=False)

        # Tie input/output embeddings (standard practice)
        # Note: Can't tie weights when gauge_fixed_priors=True since there's
        # no per-token embedding - just a single base_mu rotated per token
        if tie_embeddings and not gauge_fixed_priors:
            self.out_proj.weight = self.token_embed.mu_embed.weight
        elif tie_embeddings and gauge_fixed_priors:
            print("Warning: tie_embeddings disabled because gauge_fixed_priors=True")

        # =================================================================
        # Initialize Weights
        # =================================================================
        self.apply(self._init_weights)

        # Count parameters
        n_params = sum(p.numel() for p in self.parameters())
        print(f"GaugeTransformerLM initialized: {n_params/1e6:.2f}M parameters")

    def _init_weights(self, module):
        """Initialize weights following best practices."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    def forward(
        self,
        token_ids: torch.Tensor,
        return_agents: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:
        """
        Forward pass through 0D gauge transformer.

        Args:
            token_ids: (batch, seq_len) token indices
                       seq_len = number of agents at the single point c*
            return_agents: If True, return intermediate agent states

        Returns:
            logits: (batch, num_agents, vocab_size) next-token predictions
            agents: Optional dict with mu, sigma, phi for each agent

        0D STRUCTURE:
            - All agents exist at single base manifold point c*
            - No spatial variation: mu[i], sigma[i], phi[i] are per-agent, not per-location
            - Attention β_ij are scalars, not spatial fields
        """
        batch_size, num_agents = token_ids.shape
        device = token_ids.device

        # =================================================================
        # 1. Token Embeddings (0D: one per agent at c*, not per spatial point)
        # =================================================================
        mu_q, sigma_q, phi = self.token_embed(token_ids)

        # Save initial embeddings as priors for variational FFN
        mu_prior = mu_q.clone()

        # =================================================================
        # 2. Add Agent-Index Encoding to Gauge Frames
        # =================================================================
        # This distinguishes agent roles, NOT spatial positions
        pos_phi = self.pos_encoding(num_agents, device=device)  # (N, 3)
        phi = phi + pos_phi.unsqueeze(0)  # (B, N, 3)

        # =================================================================
        # 3. Attention Mask (causal + optional sparsity)
        # =================================================================
        # Create attention mask based on pattern (full, local, strided)
        mask = create_attention_mask(
            num_agents=num_agents,
            pattern=self.attention_pattern,
            window=self.attention_window,
            device=device,
            causal=True,  # Always use causal for autoregressive LM
        )
        mask = mask.unsqueeze(0).expand(batch_size, -1, -1)  # (B, N, N)

        # =================================================================
        # 3.5. Precompute Transport Operators (when evolve_phi=False)
        # =================================================================
        # When phi doesn't evolve, we can compute transport operators once
        # and reuse across all layers, saving ~6× matrix exponential calls.
        if not self.evolve_phi:
            # Get the first block's attention layer to access head generators
            first_attention = self.transformer.blocks[0].attention
            cached_head_transports = first_attention.precompute_head_transports(
                phi, device, mu_q.dtype
            )
        else:
            cached_head_transports = None

        # =================================================================
        # 4. Forward Through Transformer Stack
        # =================================================================
        mu_q, sigma_q, phi, intermediates = self.transformer(
            mu_q,
            sigma_q,
            phi,
            self.generators,
            mask=mask,
            mu_prior=mu_prior,  # Pass priors for variational FFN
            return_intermediates=return_agents,
            cached_head_transports=cached_head_transports,
        )

        # =================================================================
        # 5. Project to Vocabulary (one prediction per agent)
        # =================================================================
        logits = self.out_proj(mu_q)  # (B, N, V)

        if return_agents:
            agent_states = {
                'mu': mu_q.detach(),
                'sigma': sigma_q.detach() if sigma_q is not None else None,
                'phi': phi.detach(),
                'intermediates': intermediates,
            }
            return logits, agent_states

        return logits

    def forward_with_attention(
        self,
        token_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass that returns attention weights and KL matrices for loss computation.

        This is used during training to compute the attention-weighted free energy:
            F = Σ_ij β_ij · KL(q_i || Ω_ij[q_j]) - E[log p(o|x)]
                                                     ↑ Observations!

        Args:
            token_ids: (batch, seq_len) token indices
            targets: (batch, seq_len) target tokens - used as observations in E-step

        Returns:
            logits: (batch, num_agents, vocab_size) predictions
            attention_info: Dict with:
                - 'beta': (B, n_heads, N, N) attention weights per head
                - 'kl': (B, n_heads, N, N) KL divergences per head
                - 'mu': (B, N, K) final belief means
                - 'sigma': (B, N, K, K) final covariances
                - 'phi': (B, N, 3) final gauge frames
        """
        batch_size, num_agents = token_ids.shape
        device = token_ids.device

        # Embeddings
        mu_q, sigma_q, phi = self.token_embed(token_ids)

        # Save initial embeddings as priors for variational FFN
        mu_prior = mu_q.clone()

        # Position encoding
        pos_phi = self.pos_encoding(num_agents, device=device)
        phi = phi + pos_phi.unsqueeze(0)

        # Attention mask (causal + optional sparsity)
        mask = create_attention_mask(
            num_agents=num_agents,
            pattern=self.attention_pattern,
            window=self.attention_window,
            device=device,
            causal=True,
        )
        mask = mask.unsqueeze(0).expand(batch_size, -1, -1)

        # Precompute transport operators when evolve_phi=False (saves ~6× matrix exps)
        if not self.evolve_phi:
            first_attention = self.transformer.blocks[0].attention
            cached_head_transports = first_attention.precompute_head_transports(
                phi, device, mu_q.dtype
            )
        else:
            cached_head_transports = None

        # Forward through transformer blocks (all but last without attention tracking)
        for block in self.transformer.blocks[:-1]:
            mu_q, sigma_q, phi = block(
                mu_q, sigma_q, phi, self.generators, mask, mu_prior,
                targets=targets,  # Pass targets for E-step
                W_out=self.out_proj.weight if hasattr(self.out_proj, 'weight') else None,
                cached_head_transports=cached_head_transports,
            )

        # Final block WITH attention tracking
        final_block = self.transformer.blocks[-1]

        # Pre-norm + attention with tracking
        mu_normalized = final_block.norm1(mu_q)
        mu_attn, sigma_attn, beta, kl = final_block.attention(
            mu_normalized,
            sigma_q,
            phi,
            self.generators,
            mask=mask,
            return_attention=True,  # Get β_ij and KL_ij
            cached_head_transports=cached_head_transports,
        )

        # Complete final block forward (residual + FFN)
        mu_q = mu_q + final_block.dropout1(mu_attn)
        if final_block.evolve_sigma and sigma_attn is not None:
            sigma_q = sigma_attn

        # FFN sublayer
        mu_normalized = final_block.norm2(mu_q)

        # Call FFN with appropriate parameters based on mode
        if final_block.ffn_mode == 'learned':
            mu_ffn = final_block.ffn(mu_normalized)
        elif final_block.ffn_mode == 'variational_gradient_engine':
            # Gradient engine returns (mu, sigma) tuple
            # E-STEP: Minimize full F w.r.t. beliefs with DISCRETE observations (cross-entropy)
            mu_ffn, sigma_ffn = final_block.ffn(
                mu=mu_normalized,
                beta=beta,
                mu_prior=mu_prior,
                phi=phi,
                sigma=sigma_q,
                mask=mask,
                targets=targets,  # DISCRETE observations (token IDs)!
                W_out=self.out_proj.weight if hasattr(self.out_proj, 'weight') else None,
            )
            # Update covariances if evolving
            if final_block.evolve_sigma and sigma_ffn is not None:
                sigma_q = sigma_ffn
        elif final_block.ffn_mode == 'hamiltonian':
            # Hamiltonian mode returns (mu, sigma, phi, diagnostics) tuple
            mu_ffn, sigma_ffn, phi_ffn, diagnostics = final_block.ffn(
                mu=mu_normalized,
                beta=beta,
                mu_prior=mu_prior,
                phi=phi,
                sigma=sigma_q,
                sigma_prior=None,  # Will default to identity
                mask=mask,
                targets=targets,
                W_out=self.out_proj.weight if hasattr(self.out_proj, 'weight') else None,
            )
            # Update covariances from Hamiltonian dynamics
            if final_block.evolve_sigma and sigma_ffn is not None:
                sigma_q = sigma_ffn
            # Update gauge frames from Hamiltonian dynamics
            if final_block.evolve_phi and phi_ffn is not None:
                phi = phi_ffn
            # Store diagnostics for monitoring
            final_block._last_hamiltonian_diagnostics = diagnostics
        else:  # Legacy variational modes (variational_approx, variational_full)
            mu_ffn = final_block.ffn(
                mu=mu_normalized,
                beta=beta,
                mu_prior=mu_prior,
                phi=phi,
                sigma=sigma_q,
                mask=mask,
            )

        mu_q = mu_q + mu_ffn

        # Final norm
        mu_q = self.transformer.final_norm(mu_q)

        # Project to vocabulary
        logits = self.out_proj(mu_q)

        # Get initial embedding priors for gamma term
        mu_p, sigma_p, phi_p = self.token_embed(token_ids)

        attention_info = {
            'beta': beta,      # (B, n_heads, N, N)
            'kl': kl,          # (B, n_heads, N, N)
            'mu': mu_q,        # (B, N, K) - evolved beliefs
            'sigma': sigma_q,  # (B, N, K, K) or None
            'phi': phi,        # (B, N, 3)
            # Priors for gamma term
            'mu_prior': mu_p,      # (B, N, K) - initial embedding means
            'sigma_prior': sigma_p,  # (B, N, K, K) - initial embedding covariances
            'phi_prior': phi_p,      # (B, N, 3) - initial gauge frames
        }

        return logits, attention_info

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Autoregressive generation.

        Args:
            prompt_ids: (1, prompt_len) initial tokens
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: Top-k sampling (optional)
            top_p: Nucleus sampling (optional)

        Returns:
            generated: (1, prompt_len + max_new_tokens) full sequence
        """
        self.eval()
        generated = prompt_ids.clone()

        for _ in range(max_new_tokens):
            # Truncate if exceeds max_seq_len
            if generated.shape[1] > self.config['max_seq_len']:
                generated = generated[:, -self.config['max_seq_len']:]

            # Forward pass
            logits = self.forward(generated)  # (1, T, V)

            # Get logits for last token
            logits_next = logits[:, -1, :] / temperature  # (1, V)

            # Apply top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits_next, min(top_k, logits_next.size(-1)))
                logits_next[logits_next < v[:, [-1]]] = -float('inf')

            # Apply top-p (nucleus) filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits_next, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift right to keep first token above threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                # Scatter back to original indexing
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits_next[indices_to_remove] = -float('inf')

            # Sample
            probs = F.softmax(logits_next, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (1, 1)

            # Append
            generated = torch.cat([generated, next_token], dim=1)

        return generated

    def get_num_params(self, non_embedding: bool = True) -> int:
        """
        Return number of parameters.

        Args:
            non_embedding: If True, exclude embedding parameters

        Returns:
            n_params: Total parameter count
        """
        n_params = sum(p.numel() for p in self.parameters())

        if non_embedding:
            # Exclude embedding parameters
            if hasattr(self.token_embed, 'mu_embed'):
                # Standard per-token embeddings
                n_params -= self.token_embed.mu_embed.weight.numel()
            elif hasattr(self.token_embed, 'base_mu'):
                # Gauge-fixed priors: base_mu + base_log_sigma_diag + phi_embed
                n_params -= self.token_embed.base_mu.numel()
                n_params -= self.token_embed.base_log_sigma_diag.numel()
                n_params -= self.token_embed.phi_embed.weight.numel()

        return n_params


# =============================================================================
# Testing
# =============================================================================

if __name__ == '__main__':
    print("="*70)
    print("GAUGE TRANSFORMER LANGUAGE MODEL TEST")
    print("="*70)

    # Test configuration (small for quick testing)
    config = {
        'vocab_size': 100,
        'embed_dim': 32,
        'n_layers': 2,
        'hidden_dim': 128,
        'max_seq_len': 16,
        'kappa_beta': 1.0,
        'dropout': 0.1,
        'pos_encoding_mode': 'learned',
        'evolve_sigma': False,
        'evolve_phi': False,
        'tie_embeddings': True,
        'irrep_spec': [
            ('ℓ0', 8, 1),
            ('ℓ1', 4, 3),
            ('ℓ2', 2, 5),
        ],  # Total: 8 + 12 + 10 = 30 → pad to 32
    }

    print(f"\n[1] Creating model...")
    print(f"    Config: vocab={config['vocab_size']}, K={config['embed_dim']}, "
          f"layers={config['n_layers']}")

    model = GaugeTransformerLM(config)
    print(f"    ✓ Model created")

    # Test forward pass
    print(f"\n[2] Testing forward pass...")
    batch_size = 2
    seq_len = 8
    token_ids = torch.randint(0, config['vocab_size'], (batch_size, seq_len))

    logits = model(token_ids)
    print(f"    Input shape:  {token_ids.shape}")
    print(f"    Output shape: {logits.shape}")
    print(f"    ✓ Forward pass complete")

    # Test with return_agents
    print(f"\n[3] Testing with agent state tracking...")
    logits, agents = model(token_ids, return_agents=True)
    print(f"    Agent states: μ={agents['mu'].shape}, "
          f"φ={agents['phi'].shape}")
    print(f"    Intermediates: {len(agents['intermediates'])} layers")
    print(f"    ✓ Agent tracking works")

    # Test generation
    print(f"\n[4] Testing autoregressive generation...")
    prompt = torch.randint(0, config['vocab_size'], (1, 4))
    generated = model.generate(
        prompt,
        max_new_tokens=8,
        temperature=1.0,
        top_k=10,
    )
    print(f"    Prompt length:    {prompt.shape[1]}")
    print(f"    Generated length: {generated.shape[1]}")
    print(f"    Tokens: {generated[0].tolist()}")
    print(f"    ✓ Generation works")

    # Parameter count
    total_params = model.get_num_params(non_embedding=False)
    non_embed_params = model.get_num_params(non_embedding=True)

    print(f"\n[5] Parameter count:")
    print(f"    Total:         {total_params:,} parameters")
    print(f"    Non-embedding: {non_embed_params:,} parameters")
    print(f"    Embedding:     {total_params - non_embed_params:,} parameters")

    # Compare to standard transformer
    standard_params = (
        config['vocab_size'] * config['embed_dim'] +  # Token embedding
        config['max_seq_len'] * config['embed_dim'] +  # Position embedding
        config['n_layers'] * (
            4 * config['embed_dim'] ** 2 +  # Q,K,V,O
            2 * config['embed_dim'] * config['hidden_dim'] +  # FFN
            4 * config['embed_dim']  # LayerNorm
        )
    )

    print(f"\n[6] Comparison to standard transformer:")
    print(f"    Standard (est): {standard_params:,} parameters")
    print(f"    Gauge:          {total_params:,} parameters")
    print(f"    Reduction:      {standard_params / total_params:.2f}x")

    print("\n" + "="*70)
    print("✓ All model tests passed!")
    print("="*70)
    print("\nNext steps:")
    print("  1. Implement training loop (free energy loss)")
    print("  2. Create data pipeline (WikiText-2)")
    print("  3. Train the model!")
    print("="*70)