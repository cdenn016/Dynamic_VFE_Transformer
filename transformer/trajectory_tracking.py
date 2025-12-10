"""
Phase Space Trajectory Tracking for Hamiltonian Transformer
============================================================

Records and visualizes (μ, Σ, φ) trajectories through:
1. Leapfrog steps within FFN layers
2. Transformer layers (embedding → output)
3. Training iterations

Key use cases:
- Verify energy conservation in Hamiltonian dynamics
- Visualize belief evolution through the network
- Token attribution via trajectory reversal
- Training diagnostics (convergence, stability)

Author: Chris & Claude
Date: December 2025
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
import json
from pathlib import Path


# =============================================================================
# Trajectory Data Structures
# =============================================================================

@dataclass
class LeapfrogSnapshot:
    """
    Single snapshot of phase space state during leapfrog integration.

    Stores lightweight copies (detached, on CPU) to avoid memory issues.
    """
    step: int                          # Leapfrog step index (0 = initial)
    mu: np.ndarray                     # (B, N, K) belief means
    Sigma_diag: np.ndarray             # (B, N, K) diagonal of covariances (memory efficient)
    phi: np.ndarray                    # (B, N, 3) gauge frames
    pi_mu_norm: float                  # ||π_μ|| momentum magnitude
    pi_Sigma_norm: float               # ||π_Σ|| momentum magnitude
    H: float                           # Hamiltonian (total energy)
    T: float                           # Kinetic energy
    V: float                           # Potential energy

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return {
            'step': self.step,
            'mu': self.mu.tolist() if isinstance(self.mu, np.ndarray) else self.mu,
            'Sigma_diag': self.Sigma_diag.tolist() if isinstance(self.Sigma_diag, np.ndarray) else self.Sigma_diag,
            'phi': self.phi.tolist() if isinstance(self.phi, np.ndarray) else self.phi,
            'pi_mu_norm': float(self.pi_mu_norm),
            'pi_Sigma_norm': float(self.pi_Sigma_norm),
            'H': float(self.H),
            'T': float(self.T),
            'V': float(self.V),
        }


@dataclass
class LayerTrajectory:
    """
    Trajectory through a single transformer layer.

    Records:
    - Input/output beliefs
    - Attention weights
    - Leapfrog trajectory (if Hamiltonian FFN)
    - Energy diagnostics
    """
    layer_idx: int

    # Input state
    mu_in: np.ndarray                  # (B, N, K)
    Sigma_diag_in: np.ndarray          # (B, N, K)
    phi_in: np.ndarray                 # (B, N, 3)

    # Output state
    mu_out: np.ndarray                 # (B, N, K)
    Sigma_diag_out: np.ndarray         # (B, N, K)
    phi_out: np.ndarray                # (B, N, 3)

    # Attention
    beta: Optional[np.ndarray] = None  # (B, n_heads, N, N) or (B, N, N)
    kl_matrix: Optional[np.ndarray] = None  # (B, N, N) KL divergences

    # Hamiltonian FFN leapfrog trajectory (if applicable)
    leapfrog_steps: List[LeapfrogSnapshot] = field(default_factory=list)

    # Energy diagnostics
    H_init: Optional[float] = None
    H_final: Optional[float] = None
    delta_H: Optional[float] = None

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return {
            'layer_idx': self.layer_idx,
            'mu_in': self.mu_in.tolist() if isinstance(self.mu_in, np.ndarray) else None,
            'mu_out': self.mu_out.tolist() if isinstance(self.mu_out, np.ndarray) else None,
            'phi_in': self.phi_in.tolist() if isinstance(self.phi_in, np.ndarray) else None,
            'phi_out': self.phi_out.tolist() if isinstance(self.phi_out, np.ndarray) else None,
            'H_init': self.H_init,
            'H_final': self.H_final,
            'delta_H': self.delta_H,
            'n_leapfrog_steps': len(self.leapfrog_steps),
        }


@dataclass
class ForwardTrajectory:
    """
    Complete trajectory through one forward pass.

    Records evolution from token embeddings through all layers to output.
    """
    # Token info
    batch_size: int
    seq_len: int

    # Initial embeddings (priors)
    mu_embed: np.ndarray               # (B, N, K) token embedding means
    Sigma_diag_embed: np.ndarray       # (B, N, K) embedding covariance diagonals
    phi_embed: np.ndarray              # (B, N, 3) initial gauge frames

    # Per-layer trajectories
    layer_trajectories: List[LayerTrajectory] = field(default_factory=list)

    # Final output
    mu_final: Optional[np.ndarray] = None
    logits_entropy: Optional[float] = None  # Avg entropy of output distribution

    # Metadata
    ffn_mode: str = 'learned'
    total_delta_H: float = 0.0         # Sum of |ΔH| across all Hamiltonian layers

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return {
            'batch_size': self.batch_size,
            'seq_len': self.seq_len,
            'ffn_mode': self.ffn_mode,
            'total_delta_H': self.total_delta_H,
            'n_layers': len(self.layer_trajectories),
            'logits_entropy': self.logits_entropy,
            'layer_summaries': [lt.to_dict() for lt in self.layer_trajectories],
        }


# =============================================================================
# Trajectory Recorder
# =============================================================================

class TrajectoryRecorder:
    """
    Records phase space trajectories during forward passes.

    Usage:
        recorder = TrajectoryRecorder(enabled=True, record_leapfrog=True)

        # In model forward:
        recorder.start_forward(batch_size, seq_len, ffn_mode)
        recorder.record_embeddings(mu, Sigma, phi)

        # In each layer:
        recorder.start_layer(layer_idx)
        recorder.record_layer_input(mu, Sigma, phi)
        recorder.record_attention(beta, kl_matrix)

        # In Hamiltonian FFN (if applicable):
        recorder.record_leapfrog_step(step, state, H, T, V)

        recorder.record_layer_output(mu, Sigma, phi, diagnostics)
        recorder.end_layer()

        # After forward:
        trajectory = recorder.end_forward(mu_final, logits)
    """

    def __init__(
        self,
        enabled: bool = False,
        record_leapfrog: bool = False,
        record_attention: bool = False,
        max_batch_elements: int = 4,  # Only record first N batch elements (memory)
        device_for_storage: str = 'cpu',
    ):
        """
        Initialize trajectory recorder.

        Args:
            enabled: If False, all recording is no-op (zero overhead)
            record_leapfrog: Record per-step leapfrog trajectory (expensive)
            record_attention: Record full attention matrices (expensive)
            max_batch_elements: Limit batch size for storage
            device_for_storage: Device to store tensors ('cpu' recommended)
        """
        self.enabled = enabled
        self.record_leapfrog = record_leapfrog
        self.record_attention = record_attention
        self.max_batch_elements = max_batch_elements
        self.device_for_storage = device_for_storage

        # Current forward pass state
        self._current_forward: Optional[ForwardTrajectory] = None
        self._current_layer: Optional[LayerTrajectory] = None

        # History of forward passes
        self.history: List[ForwardTrajectory] = []
        self.max_history: int = 100  # Keep last N forward passes

    def _to_numpy(self, tensor: torch.Tensor, max_batch: Optional[int] = None) -> np.ndarray:
        """Convert tensor to numpy, limiting batch size."""
        if max_batch is None:
            max_batch = self.max_batch_elements
        return tensor[:max_batch].detach().cpu().numpy()

    def _sigma_to_diag(self, Sigma: torch.Tensor) -> np.ndarray:
        """Extract diagonal of covariance (memory efficient)."""
        # Sigma: (B, N, K, K) -> diag: (B, N, K)
        diag = torch.diagonal(Sigma, dim1=-2, dim2=-1)
        return self._to_numpy(diag)

    # =========================================================================
    # Forward Pass Control
    # =========================================================================

    def start_forward(
        self,
        batch_size: int,
        seq_len: int,
        ffn_mode: str = 'learned',
    ) -> None:
        """Start recording a new forward pass."""
        if not self.enabled:
            return

        self._current_forward = ForwardTrajectory(
            batch_size=min(batch_size, self.max_batch_elements),
            seq_len=seq_len,
            ffn_mode=ffn_mode,
            mu_embed=np.array([]),
            Sigma_diag_embed=np.array([]),
            phi_embed=np.array([]),
        )

    def record_embeddings(
        self,
        mu: torch.Tensor,
        Sigma: torch.Tensor,
        phi: torch.Tensor,
    ) -> None:
        """Record initial token embeddings."""
        if not self.enabled or self._current_forward is None:
            return

        self._current_forward.mu_embed = self._to_numpy(mu)
        self._current_forward.Sigma_diag_embed = self._sigma_to_diag(Sigma)
        self._current_forward.phi_embed = self._to_numpy(phi)

    def end_forward(
        self,
        mu_final: torch.Tensor,
        logits: Optional[torch.Tensor] = None,
    ) -> Optional[ForwardTrajectory]:
        """End forward pass recording and return trajectory."""
        if not self.enabled or self._current_forward is None:
            return None

        self._current_forward.mu_final = self._to_numpy(mu_final)

        # Compute output entropy if logits provided
        if logits is not None:
            probs = torch.softmax(logits[:self.max_batch_elements], dim=-1)
            entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1).mean()
            self._current_forward.logits_entropy = entropy.item()

        # Sum delta_H from all Hamiltonian layers
        total_dH = sum(
            abs(lt.delta_H) for lt in self._current_forward.layer_trajectories
            if lt.delta_H is not None
        )
        self._current_forward.total_delta_H = total_dH

        # Store in history
        trajectory = self._current_forward
        self.history.append(trajectory)
        if len(self.history) > self.max_history:
            self.history.pop(0)

        self._current_forward = None
        return trajectory

    # =========================================================================
    # Layer-Level Recording
    # =========================================================================

    def start_layer(self, layer_idx: int) -> None:
        """Start recording a transformer layer."""
        if not self.enabled or self._current_forward is None:
            return

        self._current_layer = LayerTrajectory(
            layer_idx=layer_idx,
            mu_in=np.array([]),
            Sigma_diag_in=np.array([]),
            phi_in=np.array([]),
            mu_out=np.array([]),
            Sigma_diag_out=np.array([]),
            phi_out=np.array([]),
        )

    def record_layer_input(
        self,
        mu: torch.Tensor,
        Sigma: torch.Tensor,
        phi: torch.Tensor,
    ) -> None:
        """Record layer input state."""
        if not self.enabled or self._current_layer is None:
            return

        self._current_layer.mu_in = self._to_numpy(mu)
        self._current_layer.Sigma_diag_in = self._sigma_to_diag(Sigma)
        self._current_layer.phi_in = self._to_numpy(phi)

    def record_attention(
        self,
        beta: torch.Tensor,
        kl_matrix: Optional[torch.Tensor] = None,
    ) -> None:
        """Record attention weights and KL matrix."""
        if not self.enabled or self._current_layer is None:
            return

        if self.record_attention:
            self._current_layer.beta = self._to_numpy(beta)
            if kl_matrix is not None:
                self._current_layer.kl_matrix = self._to_numpy(kl_matrix)

    def record_layer_output(
        self,
        mu: torch.Tensor,
        Sigma: torch.Tensor,
        phi: torch.Tensor,
        diagnostics: Optional[Dict] = None,
    ) -> None:
        """Record layer output state and diagnostics."""
        if not self.enabled or self._current_layer is None:
            return

        self._current_layer.mu_out = self._to_numpy(mu)
        self._current_layer.Sigma_diag_out = self._sigma_to_diag(Sigma)
        self._current_layer.phi_out = self._to_numpy(phi)

        # Extract Hamiltonian diagnostics
        if diagnostics is not None:
            self._current_layer.H_init = diagnostics.get('H_init')
            self._current_layer.H_final = diagnostics.get('H_final')
            self._current_layer.delta_H = diagnostics.get('delta_H')

    def end_layer(self) -> None:
        """End layer recording and add to forward trajectory."""
        if not self.enabled or self._current_forward is None or self._current_layer is None:
            return

        self._current_forward.layer_trajectories.append(self._current_layer)
        self._current_layer = None

    # =========================================================================
    # Leapfrog Step Recording (for Hamiltonian FFN)
    # =========================================================================

    def record_leapfrog_step(
        self,
        step: int,
        mu: torch.Tensor,
        Sigma: torch.Tensor,
        phi: torch.Tensor,
        pi_mu: torch.Tensor,
        pi_Sigma: torch.Tensor,
        H: float,
        T: float,
        V: float,
    ) -> None:
        """Record a single leapfrog integration step."""
        if not self.enabled or not self.record_leapfrog or self._current_layer is None:
            return

        snapshot = LeapfrogSnapshot(
            step=step,
            mu=self._to_numpy(mu),
            Sigma_diag=self._sigma_to_diag(Sigma),
            phi=self._to_numpy(phi),
            pi_mu_norm=pi_mu.norm().item(),
            pi_Sigma_norm=pi_Sigma.norm().item(),
            H=H,
            T=T,
            V=V,
        )
        self._current_layer.leapfrog_steps.append(snapshot)

    # =========================================================================
    # Analysis and Export
    # =========================================================================

    def get_energy_trace(self, trajectory: Optional[ForwardTrajectory] = None) -> Dict[str, List[float]]:
        """
        Extract energy conservation trace from trajectory.

        Returns:
            Dict with 'H', 'T', 'V' lists per leapfrog step (aggregated across layers)
        """
        if trajectory is None:
            trajectory = self.history[-1] if self.history else None

        if trajectory is None:
            return {'H': [], 'T': [], 'V': [], 'delta_H': []}

        H_trace = []
        T_trace = []
        V_trace = []

        for lt in trajectory.layer_trajectories:
            for snap in lt.leapfrog_steps:
                H_trace.append(snap.H)
                T_trace.append(snap.T)
                V_trace.append(snap.V)

        delta_H = [abs(lt.delta_H) for lt in trajectory.layer_trajectories if lt.delta_H is not None]

        return {
            'H': H_trace,
            'T': T_trace,
            'V': V_trace,
            'delta_H': delta_H,
        }

    def get_mu_trajectory(
        self,
        trajectory: Optional[ForwardTrajectory] = None,
        batch_idx: int = 0,
        token_idx: int = -1,  # -1 = last token
    ) -> np.ndarray:
        """
        Extract μ trajectory for a specific token through all layers.

        Returns:
            (n_layers + 1, K) array: [embedding, layer0_out, layer1_out, ...]
        """
        if trajectory is None:
            trajectory = self.history[-1] if self.history else None

        if trajectory is None:
            return np.array([])

        # Start with embedding
        if token_idx == -1:
            token_idx = trajectory.seq_len - 1

        mu_trace = [trajectory.mu_embed[batch_idx, token_idx]]

        for lt in trajectory.layer_trajectories:
            mu_trace.append(lt.mu_out[batch_idx, token_idx])

        return np.array(mu_trace)

    def save_trajectory(
        self,
        filepath: Union[str, Path],
        trajectory: Optional[ForwardTrajectory] = None,
    ) -> None:
        """Save trajectory to JSON file."""
        if trajectory is None:
            trajectory = self.history[-1] if self.history else None

        if trajectory is None:
            return

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'w') as f:
            json.dump(trajectory.to_dict(), f, indent=2)

    def clear_history(self) -> None:
        """Clear trajectory history to free memory."""
        self.history.clear()


# =============================================================================
# Global Recorder Instance (for easy access from model code)
# =============================================================================

_global_recorder: Optional[TrajectoryRecorder] = None


def get_global_recorder() -> Optional[TrajectoryRecorder]:
    """Get the global trajectory recorder instance."""
    return _global_recorder


def set_global_recorder(recorder: TrajectoryRecorder) -> None:
    """Set the global trajectory recorder instance."""
    global _global_recorder
    _global_recorder = recorder


def enable_trajectory_tracking(
    record_leapfrog: bool = False,
    record_attention: bool = False,
    max_batch_elements: int = 4,
) -> TrajectoryRecorder:
    """
    Enable trajectory tracking with a new global recorder.

    Example:
        >>> recorder = enable_trajectory_tracking(record_leapfrog=True)
        >>> # Run model forward passes
        >>> trajectory = recorder.history[-1]
        >>> energy = recorder.get_energy_trace(trajectory)
    """
    recorder = TrajectoryRecorder(
        enabled=True,
        record_leapfrog=record_leapfrog,
        record_attention=record_attention,
        max_batch_elements=max_batch_elements,
    )
    set_global_recorder(recorder)
    return recorder


def disable_trajectory_tracking() -> None:
    """Disable trajectory tracking."""
    global _global_recorder
    if _global_recorder is not None:
        _global_recorder.enabled = False


# =============================================================================
# Testing
# =============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("TRAJECTORY TRACKING TEST")
    print("=" * 70)

    # Create test data
    B, N, K = 2, 8, 16

    print(f"\n[1] Creating test tensors...")
    mu = torch.randn(B, N, K)
    Sigma = torch.eye(K).unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1).clone()
    phi = torch.randn(B, N, 3) * 0.1
    pi_mu = torch.randn(B, N, K) * 0.1
    pi_Sigma = torch.randn(B, N, K, K) * 0.01

    print(f"    Shapes: mu={mu.shape}, Sigma={Sigma.shape}, phi={phi.shape}")

    # Create recorder
    print(f"\n[2] Creating trajectory recorder...")
    recorder = TrajectoryRecorder(
        enabled=True,
        record_leapfrog=True,
        record_attention=True,
        max_batch_elements=2,
    )

    # Simulate forward pass
    print(f"\n[3] Simulating forward pass...")
    recorder.start_forward(B, N, ffn_mode='hamiltonian')
    recorder.record_embeddings(mu, Sigma, phi)

    # Simulate 2 layers
    for layer_idx in range(2):
        print(f"    Layer {layer_idx}...")
        recorder.start_layer(layer_idx)
        recorder.record_layer_input(mu, Sigma, phi)

        # Simulate attention
        beta = torch.softmax(torch.randn(B, 4, N, N), dim=-1)
        recorder.record_attention(beta)

        # Simulate leapfrog steps
        for step in range(5):
            H = 10.0 - step * 0.1 + torch.randn(1).item() * 0.01
            T = 5.0 + step * 0.05
            V = H - T
            recorder.record_leapfrog_step(
                step=step,
                mu=mu + step * 0.1,
                Sigma=Sigma,
                phi=phi,
                pi_mu=pi_mu,
                pi_Sigma=pi_Sigma,
                H=H, T=T, V=V,
            )

        # Layer output
        mu_out = mu + torch.randn_like(mu) * 0.1
        recorder.record_layer_output(
            mu_out, Sigma, phi,
            diagnostics={'H_init': 10.0, 'H_final': 9.95, 'delta_H': 0.05}
        )
        recorder.end_layer()
        mu = mu_out  # Update for next layer

    # End forward
    logits = torch.randn(B, N, 100)
    trajectory = recorder.end_forward(mu, logits)

    print(f"\n[4] Trajectory recorded:")
    print(f"    Batch size: {trajectory.batch_size}")
    print(f"    Seq length: {trajectory.seq_len}")
    print(f"    FFN mode: {trajectory.ffn_mode}")
    print(f"    Layers: {len(trajectory.layer_trajectories)}")
    print(f"    Total delta_H: {trajectory.total_delta_H:.6f}")
    print(f"    Output entropy: {trajectory.logits_entropy:.4f}")

    # Get energy trace
    print(f"\n[5] Energy trace:")
    energy = recorder.get_energy_trace(trajectory)
    print(f"    H samples: {len(energy['H'])}")
    print(f"    H range: [{min(energy['H']):.4f}, {max(energy['H']):.4f}]")
    print(f"    Per-layer delta_H: {energy['delta_H']}")

    # Get mu trajectory for last token
    print(f"\n[6] Mu trajectory for last token:")
    mu_trace = recorder.get_mu_trajectory(trajectory, batch_idx=0, token_idx=-1)
    print(f"    Shape: {mu_trace.shape}")
    print(f"    ||μ|| at each layer: {[np.linalg.norm(m) for m in mu_trace]}")

    # Test global recorder
    print(f"\n[7] Testing global recorder...")
    global_rec = enable_trajectory_tracking(record_leapfrog=True)
    assert get_global_recorder() is global_rec
    disable_trajectory_tracking()
    assert not get_global_recorder().enabled
    print(f"    Global recorder works")

    print("\n" + "=" * 70)
    print("TRAJECTORY TRACKING TEST PASSED")
    print("=" * 70)
