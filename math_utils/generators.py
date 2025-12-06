# -*- coding: utf-8 -*-
"""
SO(3) Lie Algebra Generators
=============================

Construction and validation of SO(3) generators for gauge theory.

For SO(3), we use the spin-ℓ irreducible representations:
- Dimension: K = 2ℓ + 1 (always odd)
- Generators: Real skew-symmetric K×K matrices
- Commutation: [G_x, G_y] = G_z (cyclic)
- Casimir eigenvalue: ℓ(ℓ+1)

Uses real tesseral harmonics (not spherical) to avoid complex arithmetic.
"""

import numpy as np
from typing import Dict


# =============================================================================
# Main Interface - SO(3) Generators
# =============================================================================

def generate_so3_generators(
    K: int,
    *,
    cache: bool = True,
    validate: bool = True,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    Generate SO(3) Lie algebra generators for dimension K.

    This is the primary interface for obtaining generators. Internally uses
    irrep construction with automatic validation.

    Args:
        K: Latent dimension (must be odd: K = 2ℓ + 1)
        cache: If True, cache generators for reuse
        validate: If True, verify commutation relations
        eps: Tolerance for validation

    Returns:
        G: Generators array, shape (3, K, K), float32
           G[a] is the a-th generator (a ∈ {0,1,2} for x,y,z)

    Properties:
        - G[a] is real skew-symmetric: G[a]ᵀ = -G[a]
        - Commutation: [G_x, G_y] = G_z (cyclic)
        - Casimir: -Σ_a G_a² = ℓ(ℓ+1) I where ℓ = (K-1)/2

    Examples:
        >>> # Spin-1 (3D, ℓ=1)
        >>> G = generate_so3_generators(3)
        >>> G.shape
        (3, 3, 3)

        >>> # Verify commutation
        >>> np.allclose(G[0] @ G[1] - G[1] @ G[0], G[2])
        True

        >>> # Spin-2 (5D, ℓ=2)
        >>> G = generate_so3_generators(5)
        >>> ell = (5 - 1) // 2  # = 2
        >>> casimir = ell * (ell + 1)  # = 6
        >>> C2 = -sum(G[a] @ G[a] for a in range(3))
        >>> np.allclose(C2, casimir * np.eye(5))
        True

    Raises:
        ValueError: If K is even (SO(3) irreps must have odd dimension)
        RuntimeError: If validation fails

    Notes:
        - For K=3: Standard 3D rotation generators (spin-1)
        - For K=5,7,9,...: Higher spin representations
        - Internally constructs irrep via tesseral harmonics
        - Cached by default for performance
    """
    # Validate K is odd
    if K % 2 == 0:
        raise ValueError(
            f"K must be odd for SO(3) irreps (K = 2ℓ + 1). Got K={K}."
        )

    # Check cache
    if cache and K in _GENERATOR_CACHE:
        return _GENERATOR_CACHE[K].copy()

    # Compute spin quantum number
    ell = (K - 1) // 2

    # Build irrep generators
    G = _build_so3_irrep_generators(ell)

    # Validate if requested
    if validate:
        _validate_so3_generators(G, eps=1e-5)

    # Cache for reuse
    if cache:
        _GENERATOR_CACHE[K] = G.copy()

    return G


# =============================================================================
# Irrep Construction (Tesseral Basis)
# =============================================================================

def _build_so3_irrep_generators(ell: int) -> np.ndarray:
    """
    Build SO(3) generators for spin-ℓ irrep in real tesseral basis.

    Algorithm:
    ---------
    1. Construct complex spherical harmonic operators J_x, J_y, J_z
    2. Build unitary transformation S: spherical → tesseral
    3. Transform: G_a = Re(S J_a S†) and enforce skew-symmetry

    Args:
        ell: Spin quantum number (ℓ ≥ 0)

    Returns:
        G: (3, K, K) float32 generators where K = 2ℓ + 1
    """
    K = 2 * ell + 1

    # ========== Step 1: Complex spherical operators ==========
    # Build J_+, J_-, J_z in complex basis
    J_plus = np.zeros((K, K), dtype=np.complex128)
    J_minus = np.zeros((K, K), dtype=np.complex128)
    J_z = np.zeros((K, K), dtype=np.complex128)

    for m in range(-ell, ell + 1):
        i = m + ell  # Index: m ∈ [-ℓ, ℓ] → i ∈ [0, K-1]

        # J_z is diagonal
        J_z[i, i] = m

        # J_+ raises m by 1
        if m < ell:
            a = np.sqrt((ell - m) * (ell + m + 1))
            J_plus[i, i + 1] = a

        # J_- lowers m by 1
        if m > -ell:
            a = np.sqrt((ell + m) * (ell - m + 1))
            J_minus[i, i - 1] = a

    # Cartesian operators
    J_x = (J_plus + J_minus) / 2.0
    J_y = (J_plus - J_minus) / (2.0j)

    # ========== Step 2: Spherical → Tesseral transformation ==========
    # S is unitary, transforms |ℓ,m⟩ → tesseral basis
    S = _build_tesseral_transform(ell)
    S_inv = S.conj().T

    # ========== Step 3: Transform to real basis ==========
    def _to_real_skew(J_complex: np.ndarray) -> np.ndarray:
        """Transform complex operator to real skew-symmetric generator."""
        # G = Re(S (iJ) S†) where factor of i makes it skew-symmetric
        G_complex = S @ (1j * J_complex) @ S_inv
        G_real = G_complex.real

        # Enforce skew-symmetry (remove any numerical symmetric part)
        G_skew = 0.5 * (G_real - G_real.T)
        return G_skew

    G_x = _to_real_skew(J_x)
    G_y = _to_real_skew(J_y)
    G_z = _to_real_skew(J_z)

    # Stack as (3, K, K)
    G = np.stack([G_x, G_y, G_z], axis=0)

    return G.astype(np.float32, copy=False)


def _build_tesseral_transform(ell: int) -> np.ndarray:
    """
    Construct unitary transformation from spherical to tesseral basis.

    Tesseral harmonics are real linear combinations of spherical harmonics:
        Y^c_{ℓm} = (Y_{ℓm} + (-1)^m Y_{ℓ,-m}) / √2        (cosine-like, m > 0)
        Y^s_{ℓm} = (Y_{ℓm} - (-1)^m Y_{ℓ,-m}) / (i√2)     (sine-like, m > 0)
        Y^0_{ℓ0} = Y_{ℓ0}                                  (m = 0)

    Args:
        ell: Spin quantum number

    Returns:
        S: (K, K) unitary matrix, complex128
    """
    K = 2 * ell + 1
    S = np.zeros((K, K), dtype=np.complex128)

    # m = 0 component (center)
    S[0, ell] = 1.0

    # m > 0 components (cosine and sine pairs)
    row = 1
    for m in range(1, ell + 1):
        phase = (-1) ** m
        sqrt2_inv = 1.0 / np.sqrt(2.0)

        # Cosine-like: Y^c_m = (Y_m + phase Y_{-m}) / √2
        S[row, ell + m] = sqrt2_inv
        S[row, ell - m] = phase * sqrt2_inv
        row += 1

        # Sine-like: Y^s_m = (Y_m - phase Y_{-m}) / (i√2)
        S[row, ell + m] = -1j * sqrt2_inv
        S[row, ell - m] = 1j * phase * sqrt2_inv
        row += 1

    return S


# =============================================================================
# Validation
# =============================================================================

def _validate_so3_generators(
    G: np.ndarray,
    *,
    eps: float = 1e-6,
    verbose: bool = False,
) -> None:
    """
    Validate SO(3) commutation relations and properties.

    Checks:
    ------
    1. Skew-symmetry: G[a]ᵀ = -G[a]
    2. Commutation: [G_x, G_y] = G_z (cyclic)
    3. Casimir: C_2 = -Σ G_a² = ℓ(ℓ+1) I

    Args:
        G: (3, K, K) generators
        eps: Tolerance for checks
        verbose: If True, print validation details

    Raises:
        RuntimeError: If any check fails
    """
    if G.shape[0] != 3:
        raise ValueError(f"Expected 3 generators (x,y,z), got {G.shape[0]}")

    K = G.shape[1]
    if G.shape != (3, K, K):
        raise ValueError(f"Expected shape (3, K, K), got {G.shape}")

    G_x, G_y, G_z = G[0], G[1], G[2]

    # ========== Check 1: Skew-symmetry ==========
    for a, name in enumerate(['x', 'y', 'z']):
        G_a = G[a]
        skew_error = np.linalg.norm(G_a + G_a.T, ord='fro')
        if skew_error > eps:
            raise RuntimeError(
                f"Generator G_{name} not skew-symmetric: ||G + Gᵀ|| = {skew_error:.3e}"
            )

    # ========== Check 2: Commutation relations ==========
    # [G_x, G_y] = G_z
    comm_xy = G_x @ G_y - G_y @ G_x
    error_xy = np.linalg.norm(comm_xy - G_z, ord='fro')

    # [G_y, G_z] = G_x (cyclic)
    comm_yz = G_y @ G_z - G_z @ G_y
    error_yz = np.linalg.norm(comm_yz - G_x, ord='fro')

    # [G_z, G_x] = G_y
    comm_zx = G_z @ G_x - G_x @ G_z
    error_zx = np.linalg.norm(comm_zx - G_y, ord='fro')

    max_error = max(error_xy, error_yz, error_zx)

    # Scale tolerance by generator norm
    scale = max(np.linalg.norm(G[a], ord='fro') for a in range(3))
    threshold = eps * max(scale, 1.0)

    if max_error > threshold:
        raise RuntimeError(
            f"SO(3) commutation relations violated:\n"
            f"  [G_x, G_y] - G_z: {error_xy:.3e}\n"
            f"  [G_y, G_z] - G_x: {error_yz:.3e}\n"
            f"  [G_z, G_x] - G_y: {error_zx:.3e}\n"
            f"  threshold: {threshold:.3e}"
        )

    C_2 = -sum(G[a] @ G[a] for a in range(3))

    # Extract eigenvalues (should all be ℓ(ℓ+1))
    eigenvalues    = np.linalg.eigvalsh(C_2)
    casimir_value  = float(np.mean(eigenvalues))
    casimir_spread = float(np.std(eigenvalues))

    # Expected value
    ell = (K - 1) // 2
    casimir_expected = ell * (ell + 1)
    casimir_error = abs(casimir_value - casimir_expected)

    # Scale tolerance by the size of C₂
    base = max(abs(casimir_expected), 1.0)
    tol  = eps * base

    if casimir_error > tol or casimir_spread > tol:
        raise RuntimeError(
            "Casimir operator check failed:\n"
            f"  Expected: {casimir_expected}\n"
            f"  Got: {casimir_value:.6f} ± {casimir_spread:.3e}\n"
            f"  Error: {casimir_error:.3e}"
        )


    if verbose:
        print("✓ SO(3) generator validation passed:")
        print(f"  Dimension: K = {K} (ℓ = {ell})")
        print(f"  Skew-symmetry: max error = {max([np.linalg.norm(G[a] + G[a].T) for a in range(3)]):.3e}")
        print(f"  Commutation: max error = {max_error:.3e}")
        print(f"  Casimir: C₂ = {casimir_value:.6f} (expected {casimir_expected})")


# =============================================================================
# Cache
# =============================================================================

_GENERATOR_CACHE: Dict[int, np.ndarray] = {}