# -*- coding: utf-8 -*-
"""
Created on Mon Nov 10 11:20:48 2025

@author: chris and christine
"""

#!/usr/bin/env python3
"""
Base Manifold Connection Field
==============================

Geometric infrastructure for gauge connections A(c) on principal SO(N) bundle.

Mathematical Framework:
----------------------
Connection 1-form: A: TC → 𝔰𝔬(N)
- Lives in the same space as agent gauge frames φᵢ(c)
- Defines geometric structure of the principal bundle
- Enables covariant derivatives, curvature, holonomy

Discrete Representation:
-----------------------
For discrete base manifold C (grid, graph, etc.):
- A is stored as field A(c) ∈ ℝ³ at each point c
- A(c) represents tangent-space connection coefficients
- Can be viewed as background gauge field

Physical Interpretation (TBD):
-----------------------------
- Reference frame for agent gauge fields?
- Emergent from agent consensus?
- Intrinsic geometric structure?
- Regularization mechanism?

The connection exists as pure mathematical infrastructure.
Physical meaning will emerge from how it couples to agent dynamics.

Author: Chris
Date: November 2025
Status: Phase 1 - Infrastructure Only
"""

import numpy as np
from typing import Optional, Tuple, Literal
from dataclasses import dataclass
from math_utils.generators import generate_so3_generators


# =============================================================================
# Connection Field Container
# =============================================================================

@dataclass
class ConnectionField:
    """
    Connection A(c) on principal SO(N) bundle over base manifold C.
    
    Attributes:
        A: Connection field, shape (*S, 3) for 𝔰𝔬(N)
        K: Representation dimension
        N: Lie group dimension (N=3 for SO(3))
        support_shape: Spatial dimensions of base manifold C
        generators: Lie algebra generators, shape (N, K, K)
        
    Notes:
        - A(c) and φᵢ(c) live in the same space (both 𝔰𝔬(N))
        - Connection enables covariant derivatives: D_A = d + [A, ·]
        - Curvature: F_A = dA + ½[A, A]
    """
    
    A: np.ndarray  # (*S, 3) - connection coefficients
    K: int  # Representation dimension
    N: int  # Lie algebra dimension (3 for SO(3))
    support_shape: Tuple[int, ...]
    generators: np.ndarray  # (N, K, K)
    
    def __post_init__(self):
        """Validate field structure."""
        if self.A.shape[-1] != self.N:
            raise ValueError(
                f"Connection must have shape (*S, {self.N}), got {self.A.shape}"
            )
        
        if self.A.shape[:-1] != self.support_shape:
            raise ValueError(
                f"Spatial shape {self.A.shape[:-1]} doesn't match "
                f"support {self.support_shape}"
            )
        
        if self.generators.shape != (self.N, self.K, self.K):
            raise ValueError(
                f"Generators must have shape ({self.N}, {self.K}, {self.K}), "
                f"got {self.generators.shape}"
            )
    
    @property
    def ndim(self) -> int:
        """Spatial dimensionality of base manifold."""
        return len(self.support_shape)
    
    @property
    def spatial_volume(self) -> int:
        """Total number of spatial points."""
        return int(np.prod(self.support_shape))
    
    def norm(self, ord: Optional[float] = None) -> np.ndarray:
        """
        Compute ||A(c)|| at each point.
        
        Args:
            ord: Norm order (default: Euclidean 2-norm)
            
        Returns:
            norm_field: Shape (*S,)
        """
        return np.linalg.norm(self.A, ord=ord, axis=-1)
    
    def energy(self) -> float:
        """
        Compute ∫ ||A(c)||² dc (Euclidean norm).
        
        This is a regularization term penalizing large connection fields.
        """
        return float(np.sum(self.A**2))
    
    def copy(self) -> 'ConnectionField':
        """Deep copy of connection field."""
        return ConnectionField(
            A=self.A.copy(),
            K=self.K,
            N=self.N,
            support_shape=self.support_shape,
            generators=self.generators.copy(),
        )


# =============================================================================
# Initialization Methods
# =============================================================================

def initialize_flat_connection(
    support_shape: Tuple[int, ...],
    K: int,
    N: int = 3,
) -> ConnectionField:
    """
    Initialize flat connection: A = 0 everywhere.
    
    This is the trivial connection with no curvature.
    Parallel transport is path-independent.
    
    Args:
        support_shape: Spatial dimensions of C (e.g., (100,) or (32, 32))
        K: Representation dimension
        N: Lie algebra dimension (3 for SO(3))
        
    Returns:
        connection: Flat ConnectionField
        
    Examples:
        >>> conn = initialize_flat_connection((64,), K=3, N=3)
        >>> conn.A.shape
        (64, 3)
        >>> np.allclose(conn.A, 0)
        True
    """
    A = np.zeros((*support_shape, 3), dtype=np.float32)
    generators = generate_so3_generators(K)
    
    return ConnectionField(
        A=A,
        K=K,
        N=N,
        support_shape=support_shape,
        generators=generators,
    )


def initialize_random_connection(
    support_shape: Tuple[int, ...],
    K: int,
    N: int = 3,
    *,
    scale: float = 0.1,
    seed: Optional[int] = None,
) -> ConnectionField:
    """
    Initialize random smooth connection field.
    
    Samples A(c) ~ N(0, scale²) then optionally smooths spatially.
    
    Args:
        support_shape: Spatial dimensions
        K: Representation dimension
        N: Lie algebra dimension
        scale: Standard deviation of random field
        seed: Random seed for reproducibility
        
    Returns:
        connection: Random ConnectionField
    """
    if seed is not None:
        np.random.seed(seed)
    
    A = scale * np.random.randn(*support_shape, 3).astype(np.float32)
    generators = generate_so3_generators(K)
    
    return ConnectionField(
        A=A,
        K=K,
        N=N,
        support_shape=support_shape,
        generators=generators,
    )


def initialize_constant_connection(
    support_shape: Tuple[int, ...],
    K: int,
    A_const: np.ndarray,
) -> ConnectionField:
    """
    Initialize connection constant in space: A(c) = A₀ ∀c.
    
    Useful for testing uniform background fields.
    
    Args:
        support_shape: Spatial dimensions
        K: Representation dimension
        A_const: Constant value, shape (N,)
        
    Returns:
        connection: Constant ConnectionField
    """
    A_const = np.asarray(A_const, dtype=np.float32)
    N = A_const.shape[0]
    
    A = np.broadcast_to(A_const, (*support_shape, 3)).copy()
    generators = generate_so3_generators(K)
    
    return ConnectionField(
        A=A,
        K=K,
        N=N,
        support_shape=support_shape,
        generators=generators,
    )


# =============================================================================
# Curvature (Future Use)
# =============================================================================

def compute_curvature_2form(
    connection: ConnectionField,
    *,
    method: Literal['finite_diff', 'lattice'] = 'finite_diff',
) -> np.ndarray:
    """
    Compute curvature 2-form F_A = dA + ½[A, A].
    
    WARNING: This requires defining derivatives on C, which depends on
    the base manifold structure (grid, graph, continuous, etc.).
    
    For now, returns placeholder. Will implement when:
    1. We decide on discrete derivative operator
    2. We understand what curvature means physically
    
    Args:
        connection: ConnectionField
        method: Discretization scheme
        
    Returns:
        F_A: Curvature field (structure TBD)
        
    Status: PLACEHOLDER - Not implemented yet
    """
    raise NotImplementedError(
        "Curvature computation requires defining discrete exterior derivative "
        "on base manifold C. This will be implemented when needed."
    )


# =============================================================================
# Connection-Agent Interaction (Placeholders)
# =============================================================================

def compute_agent_connection_deviation(
    agent_gauge_field: np.ndarray,  # (*S, 3)
    connection: ConnectionField,
    *,
    agent_support_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Compute deviation δφ = φ_agent - A.
    
    Measures how much agent's gauge frame differs from background connection.
    
    Args:
        agent_gauge_field: Agent's φ(c), shape (*S, 3)
        connection: Background connection A(c)
        agent_support_mask: Boolean mask for agent's support
        
    Returns:
        deviation: δφ(c), shape (*S, 3)
    """
    delta_phi = agent_gauge_field - connection.A
    
    if agent_support_mask is not None:
        delta_phi = delta_phi * agent_support_mask[..., None]
    
    return delta_phi


def compute_consensus_connection(
    agent_gauge_fields: list[np.ndarray],  # List of (*S, 3)
    support_masks: list[np.ndarray],  # List of (*S,) boolean
    K: int,
    N: int = 3,
) -> ConnectionField:
    """
    Compute consensus connection from agent gauge fields.
    
    A(c) = weighted average of φᵢ(c) where agents overlap.
    
    This is a potential way to make A emergent from agent dynamics.
    For now, just a simple average - could be more sophisticated.
    
    Args:
        agent_gauge_fields: List of agent φᵢ fields
        support_masks: List of agent support regions
        K: Representation dimension
        N: Lie algebra dimension
        
    Returns:
        connection: Consensus ConnectionField
        
    Notes:
        - Where no agents exist, A = 0
        - Where multiple agents overlap, A = average(φᵢ)
        - This is Euclidean average, not Lie group average (could improve)
    """
    if not agent_gauge_fields:
        raise ValueError("Need at least one agent to compute consensus")
    
    # Get spatial shape from first agent
    support_shape = agent_gauge_fields[0].shape[:-1]
    
    # Accumulate weighted sum
    A_sum = np.zeros((*support_shape, N), dtype=np.float32)
    weight_sum = np.zeros(support_shape, dtype=np.float32)
    
    for phi, mask in zip(agent_gauge_fields, support_masks):
        mask = mask.astype(np.float32)
        A_sum += phi * mask[..., None]
        weight_sum += mask
    
    # Average where weights > 0
    mask_nonzero = weight_sum > 1e-12
    A = np.zeros_like(A_sum)
    A[mask_nonzero] = A_sum[mask_nonzero] / weight_sum[mask_nonzero, None]
    
    generators = generate_so3_generators(K)
    
    return ConnectionField(
        A=A,
        K=K,
        N=N,
        support_shape=support_shape,
        generators=generators,
    )


# =============================================================================
# Utilities
# =============================================================================

def connection_field_norm(connection: ConnectionField, ord: str = 'fro') -> float:
    """
    Compute global norm of connection field.
    
    Args:
        connection: ConnectionField
        ord: 'fro' (Frobenius), 'max', or int
        
    Returns:
        norm: Scalar norm value
    """
    if ord == 'fro':
        return float(np.sqrt(np.sum(connection.A**2)))
    elif ord == 'max':
        return float(np.max(np.abs(connection.A)))
    else:
        return float(np.linalg.norm(connection.A.ravel(), ord=ord))


def visualize_connection_field(
    connection: ConnectionField,
    *,
    slice_idx: Optional[Tuple[int, ...]] = None,
) -> None:
    """
    Visualize connection field (requires matplotlib).
    
    For 1D: plot A(c) components
    For 2D: quiver plot or heatmap
    For 3D+: requires slicing
    
    Args:
        connection: ConnectionField to visualize
        slice_idx: For high-dim, which slice to plot
        
    Status: PLACEHOLDER - Implement when needed for analysis
    """
    raise NotImplementedError(
        "Connection field visualization will be added when we start "
        "analyzing connection structure."
    )


# =============================================================================
# Module-Level Tests
# =============================================================================

if __name__ == "__main__":
    print("Testing connection.py...")
    
    # Test 1: Flat connection
    print("\n1. Flat connection (1D)")
    conn_1d = initialize_flat_connection((100,), K=3, N=3)
    print(f"   Shape: {conn_1d.A.shape}")
    print(f"   Energy: {conn_1d.energy():.6f}")
    assert conn_1d.energy() < 1e-10, "Flat connection should have zero energy"
    
    # Test 2: Random connection
    print("\n2. Random connection (2D)")
    conn_2d = initialize_random_connection((32, 32), K=3, scale=0.1, seed=42)
    print(f"   Shape: {conn_2d.A.shape}")
    print(f"   Energy: {conn_2d.energy():.6f}")
    print(f"   Max norm: {np.max(conn_2d.norm()):.6f}")
    
    # Test 3: Constant connection
    print("\n3. Constant connection")
    A0 = np.array([0.1, -0.2, 0.15], dtype=np.float32)
    conn_const = initialize_constant_connection((50, 50), K=3, A_const=A0)
    assert np.allclose(conn_const.A, A0), "All points should equal A0"
    print(f"   All points equal to A0: {np.allclose(conn_const.A, A0)}")
    
    # Test 4: Deviation computation
    print("\n4. Agent-connection deviation")
    phi_agent = np.random.randn(100, 3).astype(np.float32) * 0.2
    deviation = compute_agent_connection_deviation(phi_agent, conn_1d)
    print(f"   Deviation shape: {deviation.shape}")
    print(f"   Mean deviation: {np.mean(np.linalg.norm(deviation, axis=-1)):.6f}")
    
    # Test 5: Consensus connection
    print("\n5. Consensus connection from agents")
    phi1 = np.random.randn(50, 3).astype(np.float32) * 0.1
    phi2 = np.random.randn(50, 3).astype(np.float32) * 0.1
    mask1 = np.ones(50, dtype=bool)
    mask1[30:] = False
    mask2 = np.ones(50, dtype=bool)
    mask2[:20] = False
    
    conn_consensus = compute_consensus_connection([phi1, phi2], [mask1, mask2], K=3)
    print(f"   Consensus shape: {conn_consensus.A.shape}")
    print(f"   Consensus energy: {conn_consensus.energy():.6f}")
    
    print("\n✓ All connection field tests passed!")
    print("\nConnection infrastructure ready for integration.")
    print("Next: Decide how to couple connection to agent dynamics.")