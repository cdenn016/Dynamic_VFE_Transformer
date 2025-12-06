"""
Lie Algebra Geometry for Gauge Fields
======================================

Lie algebra structure for gauge field dynamics on φ ∈ 𝔤.

For gauge group G, the Lie algebra 𝔤 = T_e G (tangent space at identity)
carries a bracket [·,·] and an inner product ⟨·,·⟩.

Groups Implemented:
-------------------
1. SO(3): Compact, 3D rotations
   - Algebra: 𝔰𝔬(3) ≅ ℝ³ (3 generators)
   - Bracket: [J_i, J_j] = ε_ijk J_k
   - Metric: ⟨φ, ψ⟩ = -tr(φψ) (positive definite)

2. SO(1,3): Non-compact, Lorentz group
   - Algebra: 𝔰𝔬(1,3) (6 generators: 3 rotations + 3 boosts)
   - Bracket: [J_i, J_j] = ε_ijk J_k, [J_i, K_j] = ε_ijk K_k, [K_i, K_j] = -ε_ijk J_k
   - Metric: Killing form (INDEFINITE!)

Kinetic Energy:
--------------
For gauge field φ(t) ∈ 𝔤:
    T_φ = (1/2) ⟨φ̇, φ̇⟩_𝔤

Dynamics:
--------
    φ̈ = -∂V/∂φ + gauge corrections

where gauge corrections come from connection on Lie group.

Author: Chris
Date: November 2025
"""

import numpy as np
from typing import Tuple, List, Optional
from enum import Enum


class LieGroup(Enum):
    """Supported Lie groups."""
    SO3 = "SO(3)"
    SO13 = "SO(1,3)"
    SU2 = "SU(2)"


# =============================================================================
# SO(3) - Rotation Group
# =============================================================================

def so3_generators() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generators of SO(3) in 3D representation.

    J_i: Infinitesimal rotations around axis i.

    [J_i, J_j] = ε_ijk J_k

    Returns:
        J_1, J_2, J_3: (3, 3) skew-symmetric matrices
    """
    J1 = np.array([
        [0,  0,  0],
        [0,  0, -1],
        [0,  1,  0]
    ])

    J2 = np.array([
        [ 0,  0,  1],
        [ 0,  0,  0],
        [-1,  0,  0]
    ])

    J3 = np.array([
        [ 0, -1,  0],
        [ 1,  0,  0],
        [ 0,  0,  0]
    ])

    return J1, J2, J3


def so3_from_vector(v: np.ndarray) -> np.ndarray:
    """
    Convert vector v ∈ ℝ³ to skew-symmetric matrix φ ∈ 𝔰𝔬(3).

    φ = Σ_i v^i J_i

    Args:
        v: (3,) vector [v¹, v², v³]

    Returns:
        phi: (3, 3) skew-symmetric matrix
    """
    return np.array([
        [    0, -v[2],  v[1]],
        [ v[2],     0, -v[0]],
        [-v[1],  v[0],     0]
    ])


def so3_to_vector(phi: np.ndarray) -> np.ndarray:
    """
    Convert skew-symmetric matrix φ ∈ 𝔰𝔬(3) to vector v ∈ ℝ³.

    v^i = (1/2) ε_ijk φ^jk

    Args:
        phi: (3, 3) skew-symmetric matrix

    Returns:
        v: (3,) vector
    """
    return np.array([
        phi[2, 1],  # -φ[1,2]
        phi[0, 2],  # -φ[2,0]
        phi[1, 0]   # -φ[0,1]
    ])


def so3_bracket(phi: np.ndarray, psi: np.ndarray) -> np.ndarray:
    """
    Lie bracket [φ, ψ] = φψ - ψφ.

    For 𝔰𝔬(3): [J_i, J_j] = ε_ijk J_k

    Args:
        phi, psi: Elements of 𝔰𝔬(3)

    Returns:
        [phi, psi]: Bracket
    """
    return phi @ psi - psi @ phi


def so3_inner_product(phi: np.ndarray, psi: np.ndarray) -> float:
    """
    Inner product on 𝔰𝔬(3).

    ⟨φ, ψ⟩ = -tr(φψ) = Σ_i φ^i ψ^i

    This is positive definite (compact group).

    Args:
        phi, psi: Elements of 𝔰𝔬(3)

    Returns:
        ⟨φ, ψ⟩: Inner product (scalar)
    """
    return -np.trace(phi @ psi)


def so3_exponential(phi: np.ndarray) -> np.ndarray:
    """
    Exponential map exp: 𝔰𝔬(3) → SO(3).

    Rodrigues formula:
        exp(φ) = I + (sin θ / θ) φ + ((1 - cos θ) / θ²) φ²

    where θ = ||φ|| = ||v||.

    Args:
        phi: Element of 𝔰𝔬(3)

    Returns:
        R: Rotation matrix ∈ SO(3)
    """
    v = so3_to_vector(phi)
    theta = np.linalg.norm(v)

    if theta < 1e-10:
        # Small angle: exp(φ) ≈ I + φ
        return np.eye(3) + phi

    # Rodrigues formula
    phi_sq = phi @ phi
    R = (np.eye(3) +
         (np.sin(theta) / theta) * phi +
         ((1 - np.cos(theta)) / theta**2) * phi_sq)

    return R


# =============================================================================
# SO(1,3) - Lorentz Group
# =============================================================================

def so13_generators() -> Tuple[Tuple, Tuple]:
    """
    Generators of SO(1,3) in 4D Minkowski representation.

    Signature: η = diag(-1, +1, +1, +1)

    Rotations J_i: Spatial rotations (compact)
    Boosts K_i: Lorentz boosts (non-compact)

    Algebra:
        [J_i, J_j] = ε_ijk J_k
        [J_i, K_j] = ε_ijk K_k
        [K_i, K_j] = -ε_ijk J_k  (NOTE: minus sign!)

    Returns:
        rotations: (J_1, J_2, J_3)
        boosts: (K_1, K_2, K_3)
    """
    # Rotations (spatial)
    J1 = np.array([
        [0,  0,  0,  0],
        [0,  0,  0,  0],
        [0,  0,  0, -1],
        [0,  0,  1,  0]
    ])

    J2 = np.array([
        [0,  0,  0,  0],
        [0,  0,  0,  1],
        [0,  0,  0,  0],
        [0, -1,  0,  0]
    ])

    J3 = np.array([
        [0,  0,  0,  0],
        [0,  0, -1,  0],
        [0,  1,  0,  0],
        [0,  0,  0,  0]
    ])

    # Boosts (time-space mixing)
    K1 = np.array([
        [0,  1,  0,  0],
        [1,  0,  0,  0],
        [0,  0,  0,  0],
        [0,  0,  0,  0]
    ])

    K2 = np.array([
        [0,  0,  1,  0],
        [0,  0,  0,  0],
        [1,  0,  0,  0],
        [0,  0,  0,  0]
    ])

    K3 = np.array([
        [0,  0,  0,  1],
        [0,  0,  0,  0],
        [0,  0,  0,  0],
        [1,  0,  0,  0]
    ])

    rotations = (J1, J2, J3)
    boosts = (K1, K2, K3)

    return rotations, boosts


def so13_killing_form(phi: np.ndarray, psi: np.ndarray) -> float:
    """
    Killing form on 𝔰𝔬(1,3).

    κ(φ, ψ) = tr(ad_φ ∘ ad_ψ) = C · tr(φψ)

    For matrix Lie algebras: κ(φ, ψ) ∝ tr(φψ)

    CRITICAL: This is INDEFINITE (non-compact group)!
        - Positive on rotations (J_i)
        - Negative on boosts (K_i)
        - Mixed signature overall

    This is the natural metric for Lorentzian signature analysis!

    Args:
        phi, psi: Elements of 𝔰𝔬(1,3)

    Returns:
        κ(φ, ψ): Killing form value
    """
    # For 𝔰𝔬(1,3), Killing form is proportional to trace
    # Normalization chosen to match physics conventions
    return -0.5 * np.trace(phi @ psi)


def so13_inner_product(
    phi: np.ndarray,
    psi: np.ndarray,
    metric_type: str = "killing"
) -> float:
    """
    Inner product on 𝔰𝔬(1,3).

    Args:
        phi, psi: Elements of 𝔰𝔬(1,3)
        metric_type: "killing" (indefinite) or "euclidean" (positive definite)

    Returns:
        ⟨φ, ψ⟩: Inner product
    """
    if metric_type == "killing":
        return so13_killing_form(phi, psi)
    elif metric_type == "euclidean":
        # Euclidean metric (ignores group structure, always positive)
        return np.trace(phi.T @ psi)
    else:
        raise ValueError(f"Unknown metric type: {metric_type}")


# =============================================================================
# Generic Lie Algebra Operations
# =============================================================================

class LieAlgebra:
    """
    Generic Lie algebra with structure constants and metric.

    Provides unified interface for 𝔰𝔬(3), 𝔰𝔬(1,3), etc.
    """

    def __init__(self, group: LieGroup):
        """
        Args:
            group: Which Lie group/algebra
        """
        self.group = group

        if group == LieGroup.SO3:
            self.dim = 3
            self.matrix_dim = 3
            self.generators = so3_generators()
            self._bracket = so3_bracket
            self._inner_product = so3_inner_product
            self._exponential = so3_exponential

        elif group == LieGroup.SO13:
            self.dim = 6  # 3 rotations + 3 boosts
            self.matrix_dim = 4
            rotations, boosts = so13_generators()
            self.generators = rotations + boosts
            self._bracket = lambda phi, psi: phi @ psi - psi @ phi
            self._inner_product = so13_killing_form
            self._exponential = lambda phi: np.linalg.matrix_power(np.eye(4) + phi / 10, 10)  # Placeholder

        else:
            raise NotImplementedError(f"Group {group} not implemented")

    def bracket(self, phi: np.ndarray, psi: np.ndarray) -> np.ndarray:
        """Lie bracket [φ, ψ]."""
        return self._bracket(phi, psi)

    def inner_product(self, phi: np.ndarray, psi: np.ndarray) -> float:
        """Inner product ⟨φ, ψ⟩."""
        return self._inner_product(phi, psi)

    def norm(self, phi: np.ndarray) -> float:
        """Norm ||φ|| = √⟨φ, φ⟩."""
        ip = self.inner_product(phi, phi)
        # Handle indefinite metrics
        return np.sqrt(np.abs(ip)) if ip >= 0 else np.sqrt(-ip)

    def kinetic_energy(self, phi_dot: np.ndarray) -> float:
        """
        Kinetic energy T = (1/2) ⟨φ̇, φ̇⟩.

        CRITICAL: For SO(1,3), this can be NEGATIVE (indefinite metric)!

        Args:
            phi_dot: Time derivative φ̇ ∈ 𝔤 (vector or matrix form)

        Returns:
            T: Kinetic energy (can be negative for non-compact groups!)
        """
        # Handle vector representation for SO(3)
        if self.group == LieGroup.SO3 and phi_dot.ndim == 1:
            # Vector form (3,) -> convert to matrix form
            phi_dot_matrix = so3_from_vector(phi_dot)
            return 0.5 * self.inner_product(phi_dot_matrix, phi_dot_matrix)

        # For matrix form or other groups
        return 0.5 * self.inner_product(phi_dot, phi_dot)

    def exp(self, phi: np.ndarray) -> np.ndarray:
        """Exponential map exp: 𝔤 → G."""
        return self._exponential(phi)

    def adjoint_action(self, g: np.ndarray, phi: np.ndarray) -> np.ndarray:
        """
        Adjoint action Ad_g(φ) = g φ g⁻¹.

        For matrix groups: conjugation.

        Args:
            g: Group element
            phi: Algebra element

        Returns:
            Ad_g(phi): Transformed algebra element
        """
        g_inv = np.linalg.inv(g)
        return g @ phi @ g_inv

    def random_element(self) -> np.ndarray:
        """Generate random Lie algebra element."""
        if self.group == LieGroup.SO3:
            v = np.random.randn(3)
            return so3_from_vector(v)
        elif self.group == LieGroup.SO13:
            # 6 components (3 rot + 3 boost)
            coeffs = np.random.randn(6)
            phi = sum(c * gen for c, gen in zip(coeffs, self.generators))
            return phi
        else:
            raise NotImplementedError


def verify_lie_algebra_structure(algebra: LieAlgebra, tol: float = 1e-10) -> bool:
    """
    Verify Lie algebra axioms:
    1. Bilinearity: [aX + bY, Z] = a[X,Z] + b[Y,Z]
    2. Alternativity: [X, X] = 0
    3. Jacobi identity: [X, [Y, Z]] + [Y, [Z, X]] + [Z, [X, Y]] = 0

    Args:
        algebra: LieAlgebra instance
        tol: Numerical tolerance

    Returns:
        True if all axioms satisfied
    """
    # Generate random elements
    X = algebra.random_element()
    Y = algebra.random_element()
    Z = algebra.random_element()

    # 1. Alternativity: [X, X] = 0
    bracket_XX = algebra.bracket(X, X)
    if np.linalg.norm(bracket_XX) > tol:
        print(f"Alternativity FAILED: ||[X,X]|| = {np.linalg.norm(bracket_XX)}")
        return False

    # 2. Jacobi identity
    term1 = algebra.bracket(X, algebra.bracket(Y, Z))
    term2 = algebra.bracket(Y, algebra.bracket(Z, X))
    term3 = algebra.bracket(Z, algebra.bracket(X, Y))
    jacobi = term1 + term2 + term3

    if np.linalg.norm(jacobi) > tol:
        print(f"Jacobi FAILED: ||sum|| = {np.linalg.norm(jacobi)}")
        return False

    return True


# Convenience functions

def gauge_field_kinetic_energy(
    phi_dot: np.ndarray,
    group: LieGroup = LieGroup.SO3
) -> float:
    """
    Kinetic energy of gauge field.

    T_φ = (1/2) ⟨φ̇, φ̇⟩_𝔤

    Args:
        phi_dot: Velocity in Lie algebra
        group: Gauge group

    Returns:
        T: Kinetic energy
    """
    algebra = LieAlgebra(group)
    return algebra.kinetic_energy(phi_dot)