# -*- coding: utf-8 -*-
"""
Agent as Smooth Section of Associated Bundle (Hybrid BaseManifold Version)
===========================================================================

An agent is a smooth local section of the statistical associated bundle:
    - Base space: C (via BaseManifold)
    - Structure group: G = SO(3)
    - Statistical fiber: F = {N(μ, Σ) : μ ∈ ℝᴷ, Σ ∈ Sym⁺⁺(K)}

Hybrid Approach:
---------------
- Uses new BaseManifold for geometric structure
- Full support everywhere (C_i = C) in this phase
- Maintains backward compatibility
- Prepares for sparse support patterns later

Author: Chris & Christine
Date: November 2025
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass  

from geometry.geometry_base import (
    BaseManifold, 
    TopologyType,
    SupportRegion,
    create_full_support
)


from config import AgentConfig  

from math_utils.push_pull import GaussianDistribution

from math_utils.generators import generate_so3_generators
from math_utils.sigma import CovarianceFieldInitializer
from agent.masking import MaskConfig

# =============================================================================
# Agent Class (Updated for BaseManifold)
# =============================================================================

@dataclass
class AgentGeometry:
    """
    Lightweight geometry descriptor for an agent.

    This keeps all the geometric / support metadata in one place so that
    other modules can query shapes, dimensionality, sparsity, etc.,
    without poking directly into the Agent internals.
    """
    base_manifold: BaseManifold
    support: SupportRegion
    K: int                    # Latent dimension
    ndim: int                 # Spatial dimension of the base manifold
    spatial_shape: Tuple[int, ...]
    is_particle: bool         # True if ndim == 0
    total_points: int         # Total |C| on the base manifold grid
    n_active: int             # Number of points with non-negligible support
    is_sparse: bool = False   # Whether support is truly sparse




class Agent:
    """
    Agent as smooth section over base manifold C.
    
    Carries:
        - Belief distribution: q(c) = N(μ_q(c), Σ_q(c))
        - Prior distribution: p(c) = N(μ_p(c), Σ_p(c))
        - Gauge field: φ(c) ∈ so(3)
    Fields (GAUGE-COVARIANT STORAGE):
        Sigma_q(c): Belief covariance (SPD), shape (*S, K, K) - PRIMARY
        Sigma_p(c): Prior covariance (SPD), shape (*S, K, K) - PRIMARY
        L_q(c), L_p(c): Cholesky factors (computed on-demand for sampling)
        mu_q(c), mu_p(c): Means (unchanged)
        phi(c): Gauge field (unchanged)

    CRITICAL: We store Σ directly (not L) to preserve gauge covariance:
        Under g ∈ SO(3): Σ → g Σ g^T (covariant)
        But Cholesky(g Σ g^T) ≠ g · Cholesky(Σ) (NOT covariant)
    Hybrid approach: Currently full support (C_i = C), sparse support coming later.
    """
    
# agents.py

    def __init__(
        self,
        agent_id: int,
        config: AgentConfig,
        rng: Optional[np.random.Generator] = None,
        base_manifold: Optional[BaseManifold] = None,
    ):
        """Initialize agent as section over base manifold."""
        self.agent_id = agent_id
        self.config = config
        self.alpha = config.alpha
        self.rng = rng if rng is not None else np.random.default_rng()
        
        # === GEOMETRY SETUP ===
        if base_manifold is not None:
            self.base_manifold = base_manifold
        else:
            self.base_manifold = BaseManifold(
                shape=config.spatial_shape,
                topology=TopologyType.PERIODIC
            )
        
        # Create full support (will be replaced by specific support later)
        self.support = create_full_support(self.base_manifold)
        
        # Legacy attributes
        self.support_shape = self.base_manifold.shape
        
        # === LATENT AND OBSERVATION DIMENSIONS ===
        self.K = config.K
      
        
        # Create unified geometry descriptor
        self.geometry = AgentGeometry(
            base_manifold=self.base_manifold,
            support=self.support,
            K=config.K,
            ndim=self.base_manifold.ndim,
            spatial_shape=self.base_manifold.shape,
            is_particle=(self.base_manifold.ndim == 0),
            total_points=self.base_manifold.n_points,
            n_active=self.support.n_active,
            is_sparse=False
        )
        
        # Covariance initializer
        self.cov_initializer = CovarianceFieldInitializer(
            strategy="smooth",
            **(config.covariance_kwargs or {})
        )
        
        # === INITIALIZE GENERATORS ===
        self._initialize_generators()

        # === INITIALIZE BELIEF AND PRIOR FIELDS ===
        self._initialize_belief_covariance()
        self._initialize_prior_covariance()

        # === INITIALIZE GAUGE FIELD ===
        self._initialize_gauge()

        # Observations (set later)
        self.x_obs = None

        # Cache flags
        self._transport_cache_dirty = True
    
    # =========================================================================
    # Covariance Access (GAUGE-COVARIANT STORAGE)
    # =========================================================================
    # Sigma_q and Sigma_p are stored directly as fields (set in __init__)
    # L_q and L_p are computed on-demand for sampling

    @property
    def L_q(self) -> np.ndarray:
        """
        Belief Cholesky factor L_q where Σ_q = L_q L_q^T.

        Computed on-demand from stored Sigma_q. Used for sampling.
        CACHED for performance (invalidated when Sigma_q changes).
        """
        if not hasattr(self, '_L_q_cache') or self._L_q_cache is None:
            self._L_q_cache = self._cov_to_cholesky(self.Sigma_q)
        return self._L_q_cache

    @property
    def L_p(self) -> np.ndarray:
        """
        Prior Cholesky factor L_p where Σ_p = L_p L_p^T.

        Computed on-demand from stored Sigma_p. Used for sampling.
        CACHED for performance (invalidated when Sigma_p changes).
        """
        if not hasattr(self, '_L_p_cache') or self._L_p_cache is None:
            self._L_p_cache = self._cov_to_cholesky(self.Sigma_p)
        return self._L_p_cache
    
    # =========================================================================
    # Cholesky Utilities
    # =========================================================================
    
    @staticmethod
    def _cholesky_to_cov(L: np.ndarray) -> np.ndarray:
        """
        Compute Σ = L L^T from Cholesky factor.
        
        Args:
            L: Cholesky factor, shape (*S, K, K)
        
        Returns:
            Sigma: Covariance, shape (*S, K, K)
        """
        return np.einsum('...ij,...kj->...ik', L, L, optimize=True)
    
    @staticmethod
    def _cov_to_cholesky(Sigma: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        """
        Compute Cholesky factor L such that Σ = L L^T.
        
        Args:
            Sigma: Covariance, shape (*S, K, K)
            eps: Regularization for numerical stability
        
        Returns:
            L: Lower triangular Cholesky factor, shape (*S, K, K)
        """
        # Add small regularization
        if len(Sigma.shape) == 2:
            # Single matrix
            Sigma_reg = Sigma + eps * np.eye(Sigma.shape[0])
            try:
                L = np.linalg.cholesky(Sigma_reg)
            except np.linalg.LinAlgError:
                # Fallback: eigendecomposition
                eigs, V = np.linalg.eigh(Sigma_reg)
                eigs = np.maximum(eigs, eps)
                L = V @ np.diag(np.sqrt(eigs))
            return L.astype(np.float32)
        
        # Spatial field
        spatial_shape = Sigma.shape[:-2]
        K = Sigma.shape[-1]
        L = np.zeros_like(Sigma)
        
        for idx in np.ndindex(spatial_shape):
            Sigma_reg = Sigma[idx] + eps * np.eye(K)
            try:
                L[idx] = np.linalg.cholesky(Sigma_reg)
            except np.linalg.LinAlgError:
                eigs, V = np.linalg.eigh(Sigma_reg)
                eigs = np.maximum(eigs, eps)
                L[idx] = V @ np.diag(np.sqrt(eigs))
        
        return L.astype(np.float32)
    
    # =========================================================================
    # Initialization: Generate smooth L fields
    # =========================================================================
    

    
    
    def _initialize_belief_covariance(self):
        """
        Initialize belief covariance Σ_q with smooth spatial structure.

        GAUGE-COVARIANT: Stores Σ_q directly (not Cholesky factor).
        """
        from agent.masking import FieldEnforcer, SupportRegionSmooth

        spatial_shape = self.base_manifold.shape
        K = self.K

        # Generate mean field
        if self.config.is_particle:
            mu_q_raw = (0.1 * self.config.mu_scale *
                       self.rng.standard_normal(K)).astype(np.float32)
        else:
            mu_q_raw = self._generate_smooth_mean_field(
                spatial_shape, K,
                scale=1.5 * self.config.mu_scale,
                smoothness_scale=self.config.mean_smoothness_scale_effective
            )

        # Generate smooth Σ field
        if self.config.covariance_strategy == "constant":
            from math_utils.sigma import generate_constant_field_safe
            Sigma_q_raw = generate_constant_field_safe(
                spatial_shape, K,
                scale=self.config.sigma_scale,
                min_eigenvalue=0.1 * self.config.sigma_scale,
                rng=self.rng
            )
        else:
            Sigma_q_raw = self.cov_initializer.generate_for_agent(
                self, scale=self.config.sigma_scale, rng=self.rng
            )

        # Enforce support constraints on Sigma directly
        if hasattr(self, 'support') and self.support is not None:
            # Check if we need to upgrade to SupportRegionSmooth
            if not isinstance(self.support, SupportRegionSmooth):
                # Get mask from support, shape from base_manifold
                mask_binary = getattr(self.support, 'mask', None)
                if mask_binary is None:
                    mask_binary = getattr(self.support, 'mask_binary', None)
                if mask_binary is None:
                    # No mask - use full support
                    mask_binary = np.ones(spatial_shape, dtype=bool)

                self.support = SupportRegionSmooth(
                    mask_binary=mask_binary,
                    base_shape=spatial_shape,
                    config=self.config.mask_config
                )

            # Enforce mean
            self.mu_q = FieldEnforcer.enforce_mean_field(mu_q_raw, self.support)

            # Enforce covariance field directly (GAUGE-COVARIANT)
            self.Sigma_q = FieldEnforcer.enforce_covariance_field(
                Sigma_q_raw,
                self.support,
                inside_scale=self.config.sigma_scale,
                outside_scale=self.config.mask_config.outside_cov_scale,
                use_smooth_transition=self.config.mask_config.use_smooth_cov_transition
            )
        else:
            self.mu_q = mu_q_raw
            self.Sigma_q = Sigma_q_raw

        # Initialize L_q cache as None (computed on first access)
        self._L_q_cache = None

        # Validate: Σ_q should be SPD
        self._validate_covariance_field(self.Sigma_q, "Σ_q")
    
    
    def _initialize_prior_covariance(self):
        """
        Initialize prior covariance Σ_p with smooth spatial structure.

        GAUGE-COVARIANT: Stores Σ_p directly (not Cholesky factor).
        """
        from agent.masking import FieldEnforcer, SupportRegionSmooth

        spatial_shape = self.base_manifold.shape
        K = self.K

        # Generate smooth mean (flatter than beliefs)
        mu_p_raw = self._generate_smooth_mean_field(
            spatial_shape, K,
            scale=1.0 * self.config.mu_scale,
            smoothness_scale=self.config.mean_smoothness_scale_effective
        )

        # Generate smooth Σ field (DIFFERENT from Sigma_q for non-zero gradients!)
        if self.config.covariance_strategy == "constant":
            from math_utils.sigma import generate_constant_field_safe
            Sigma_p_raw = generate_constant_field_safe(
                spatial_shape, K,
                scale=1.5 * self.config.sigma_scale,  # Larger scale → Σ_p ≠ Σ_q
                min_eigenvalue=0.3 * self.config.sigma_scale,
                rng=self.rng
            )
        else:
            Sigma_p_raw = self.cov_initializer.generate_for_agent(
                self, scale=1.5 * self.config.sigma_scale, rng=self.rng  # Different scale
            )

        # Enforce support constraints on Sigma directly
        if hasattr(self, 'support') and self.support is not None:
            if not isinstance(self.support, SupportRegionSmooth):
                # Get mask from support, shape from base_manifold
                mask_binary = getattr(self.support, 'mask', None)
                if mask_binary is None:
                    mask_binary = getattr(self.support, 'mask_binary', None)
                if mask_binary is None:
                    mask_binary = np.ones(spatial_shape, dtype=bool)

                self.support = SupportRegionSmooth(
                    mask_binary=mask_binary,
                    base_shape=spatial_shape,
                    config=self.config.mask_config
                )

            self.mu_p = FieldEnforcer.enforce_mean_field(mu_p_raw, self.support)

            # Enforce covariance field directly (GAUGE-COVARIANT)
            self.Sigma_p = FieldEnforcer.enforce_covariance_field(
                Sigma_p_raw,
                self.support,
                inside_scale=1.0 * self.config.sigma_scale,
                outside_scale=self.config.mask_config.outside_cov_scale,
                use_smooth_transition=self.config.mask_config.use_smooth_cov_transition
            )
        else:
            self.mu_p = mu_p_raw
            self.Sigma_p = Sigma_p_raw

        # Initialize L_p cache as None (computed on first access)
        self._L_p_cache = None

        self._validate_covariance_field(self.Sigma_p, "Σ_p")

    # =============================================================================
    # SECTION 1: Backward Compatibility Aliases
    # =============================================================================
    
    def get_beliefs_batch(self) -> GaussianDistribution:
        """
        Get all beliefs as batch distribution.
        
        Backward compatibility alias for get_belief_distribution().
        
        Returns:
            q: Distribution with batch shape (*S, K) and (*S, K, K)
        """
        return GaussianDistribution(self.mu_q, self.Sigma_q)
    
    
    def get_priors_batch(self) -> GaussianDistribution:
        """
        Get all priors as batch distribution.
        
        Backward compatibility alias for get_prior_distribution().
        
        Returns:
            p: Distribution with batch shape (*S, K) and (*S, K, K)
        """
        return GaussianDistribution(self.mu_p, self.Sigma_p)
    


    # =============================================================================
    # SECTION 2: Point-wise Access Methods
    # =============================================================================
    
    def get_belief_at(self, index: Tuple[int, ...]) -> GaussianDistribution:
        """
        Get belief distribution q(c) at spatial index.
        
        Args:
            index: Spatial index tuple (can be empty () for 0D)
        
        Returns:
            q: Gaussian distribution at that point
        """
        mu = self.mu_q[index] if index else self.mu_q
        Sigma = self.Sigma_q[index] if index else self.Sigma_q
        return GaussianDistribution(mu, Sigma)
    
    
    def get_prior_at(self, index: Tuple[int, ...]) -> GaussianDistribution:
        """
        Get prior distribution p(c) at spatial index.
        
        Args:
            index: Spatial index tuple (can be empty () for 0D)
        
        Returns:
            p: Gaussian distribution at that point
        """
        mu = self.mu_p[index] if index else self.mu_p
        Sigma = self.Sigma_p[index] if index else self.Sigma_p
        return GaussianDistribution(mu, Sigma)
    
    


  
    # =============================================================================
    # SECTION 5: Constraint Checking
    # =============================================================================
    
    def check_constraints(self) -> dict:
        """
        Verify all manifold constraints are satisfied.
        
        Returns:
            status: Dictionary with constraint violations
        """
        status = {
            'agent_id': self.agent_id,
            'valid': True,
            'violations': [],
        }
        
        # Check Σ_q positive-definite
        try:
            min_eig_q = np.min(np.linalg.eigvalsh(self.Sigma_q))
            if min_eig_q <= 0:
                status['valid'] = False
                status['violations'].append(f"Σ_q not SPD: min eig = {min_eig_q:.3e}")
        except np.linalg.LinAlgError:
            status['valid'] = False
            status['violations'].append("Σ_q: LinAlg error (numerical instability)")
        
        # Check Σ_p positive-definite
        try:
            min_eig_p = np.min(np.linalg.eigvalsh(self.Sigma_p))
            if min_eig_p <= 0:
                status['valid'] = False
                status['violations'].append(f"Σ_p not SPD: min eig = {min_eig_p:.3e}")
        except np.linalg.LinAlgError:
            status['valid'] = False
            status['violations'].append("Σ_p: LinAlg error (numerical instability)")
        
        # Check φ in principal ball
        phi_norms = np.linalg.norm(self.gauge.phi, axis=-1)
        max_norm = float(np.max(phi_norms))
        if max_norm >= np.pi - 1e-2:
            status['valid'] = False
            status['violations'].append(f"φ violates principal ball: max ||φ|| = {max_norm:.3f}")
        
        # Check for NaNs
        if np.any(np.isnan(self.mu_q)) or np.any(np.isnan(self.Sigma_q)):
            status['valid'] = False
            status['violations'].append("NaN detected in beliefs")
        
        if np.any(np.isnan(self.mu_p)) or np.any(np.isnan(self.Sigma_p)):
            status['valid'] = False
            status['violations'].append("NaN detected in priors")
        
        if np.any(np.isnan(self.gauge.phi)):
            status['valid'] = False
            status['violations'].append("NaN detected in gauge field")
        
        return status
    
    
    # =============================================================================
    # SECTION 6: Additional Missing Properties
    # =============================================================================
    
    @property
    def phi(self) -> np.ndarray:
        """Gauge field φ(c), shape (*S, 3)."""
        return self.gauge.phi
    
    
    @property  
    def n_points(self) -> int:
        """Number of spatial points."""
        return self.support.n_active
    


    
    def _initialize_generators(self):
        """Initialize SO(3) generators for this agent."""
        self.generators = generate_so3_generators(self.K, cache=True, validate=True)
    

    
    # =============================================================================
    # FIX 2: Better Smooth Mean Field (in agents.py)
    # =============================================================================
    
    def _generate_smooth_mean_field(
        self,
        spatial_shape: Tuple[int, ...],
        K: int,
        scale: float,
        smoothness_scale: float,
    ) -> np.ndarray:
        """
        Generate smooth random mean field with controlled MAXIMUM magnitude.
        
        Key: scale parameter directly controls the max absolute value!
        """
        from scipy.ndimage import gaussian_filter
        
        mu_field = np.zeros((*spatial_shape, K), dtype=np.float32)
        
        for k in range(K):
            # White noise
            noise = self.rng.standard_normal(spatial_shape)
            
            # Smooth
            smooth = gaussian_filter(
                noise,
                sigma=smoothness_scale,
                mode='wrap'
            )
            
            # Normalize by MAXIMUM, not std
            max_val = np.max(np.abs(smooth))
            if max_val > 1e-8:
                smooth = smooth / max_val  # Now in [-1, 1]
            
            # Scale to desired magnitude
            # Now max absolute value is EXACTLY scale
            mu_field[..., k] = scale * smooth
        
        return mu_field

    # =============================================================================
    # FIX 3: Safer Initialization (in agents.py)
    # =============================================================================
    
    def _initialize_belief_general(self):
        """Initialize belief with smooth support enforcement."""
        from agent.masking import FieldEnforcer
        
        spatial_shape = self.base_manifold.shape
        K = self.K
        
        # ---------------------------
        # Step 1: Generate fields everywhere (ignoring support for now)
        # ---------------------------
        if self.config.is_particle:
            # 0D: Small random values
            mu_q_raw = (
                0.1 * self.config.mu_scale *
                self.rng.standard_normal(K)
            ).astype(np.float32)
        else:
            # Spatial: Smooth random fields
            mu_q_raw = self._generate_smooth_mean_field(
                spatial_shape,
                K,
                scale=1.5 * self.config.mu_scale,
                smoothness_scale=self.config.mean_smoothness_scale_effective,
            )
        
        # Covariance (everywhere)
        if self.config.covariance_strategy == "constant":
            from sigma import generate_constant_field_safe
            Sigma_q_raw = generate_constant_field_safe(
                spatial_shape,
                K,
                scale=self.config.sigma_scale,
                min_eigenvalue=0.1 * self.config.sigma_scale,
                rng=self.rng,
            )
        else:
            Sigma_q_raw = self.cov_initializer.generate_for_agent(
                self,
                scale=self.config.sigma_scale,
                rng=self.rng,
            )
            Sigma_q_raw = self._regularize_covariance_field(
                Sigma_q_raw,
                min_eigenvalue=0.1 * self.config.sigma_scale,
            )
        
        # ---------------------------
        # Step 2: Enforce support constraints
        # ---------------------------
        if hasattr(self, 'support') and self.support is not None:
            # Convert to SupportRegionSmooth if needed
            from agent.masking import SupportRegionSmooth
            
            if not isinstance(self.support, SupportRegionSmooth):
                # Upgrade basic SupportRegion to smooth version
                self.support = SupportRegionSmooth(
                    mask_binary=self.support.mask,
                    base_shape=self.support.base_shape,
                    config=self.config.mask_config
                )
            
            # Enforce mean field (zero outside support)
            self.mu_q = FieldEnforcer.enforce_mean_field(mu_q_raw, self.support)
            
            # Enforce covariance (large diagonal outside support)
            self.Sigma_q = FieldEnforcer.enforce_covariance_field(
                Sigma_q_raw,
                self.support,
                inside_scale=self.config.sigma_scale,
                outside_scale=self.config.mask_config.outside_cov_scale,
                use_smooth_transition=self.config.mask_config.use_smooth_cov_transition
            )
        else:
            # No support constraints (full manifold)
            self.mu_q = mu_q_raw
            self.Sigma_q = Sigma_q_raw
        
        # Validate
        self._validate_covariance_field(self.Sigma_q, "Σ_q")
        
        print(f"[Agent {self.agent_id}] ✓ μ_q and Σ_q initialized with support enforcement")
    
    
    def _initialize_prior_general(self):
        """Initialize prior with smooth support enforcement."""
        from agent.masking import FieldEnforcer
        
        spatial_shape = self.base_manifold.shape
        K = self.K
        
        # ---------------------------
        # Step 1: Generate fields everywhere
        # ---------------------------
        mu_shape = (*spatial_shape, K)
        mu_p_raw = np.zeros(mu_shape, dtype=np.float32)
        
        # Instead of strict zeros, use a smoother, lower-amplitude field
        mu_p_raw = self._generate_smooth_mean_field(
            spatial_shape,
            K,
            scale=0.5 * self.config.mu_scale,  # priors a bit “flatter”
            smoothness_scale=self.config.mean_smoothness_scale_effective,
        )
        
        if self.config.covariance_strategy == "constant":
            from sigma import generate_constant_field_safe
            Sigma_p_raw = generate_constant_field_safe(
                spatial_shape,
                K,
                scale=2.0 * self.config.sigma_scale,
                min_eigenvalue=0.2 * self.config.sigma_scale,
                rng=self.rng,
            )
        else:
            Sigma_p_raw = self.cov_initializer.generate_for_agent(
                self,
                scale=2.0 * self.config.sigma_scale,
                rng=self.rng,
            )
            Sigma_p_raw = self._regularize_covariance_field(
                Sigma_p_raw,
                min_eigenvalue=0.2 * self.config.sigma_scale,
            )
        
        # ---------------------------
        # Step 2: Enforce support constraints
        # ---------------------------
        if hasattr(self, 'support') and self.support is not None:
            from agent.masking import SupportRegionSmooth
            
            if not isinstance(self.support, SupportRegionSmooth):
                self.support = SupportRegionSmooth(
                    mask_binary=self.support.mask,
                    base_shape=self.support.base_shape,
                    config=self.config.mask_config
                )
            
            self.mu_p = FieldEnforcer.enforce_mean_field(mu_p_raw, self.support)
            
            self.Sigma_p = FieldEnforcer.enforce_covariance_field(
                Sigma_p_raw,
                self.support,
                inside_scale=2.0 * self.config.sigma_scale,
                outside_scale=self.config.mask_config.outside_cov_scale,
                use_smooth_transition=self.config.mask_config.use_smooth_cov_transition
            )
        else:
            self.mu_p = mu_p_raw
            self.Sigma_p = Sigma_p_raw
        
        self._validate_covariance_field(self.Sigma_p, "Σ_p")
        
        print(f"[Agent {self.agent_id}] ✓ μ_p and Σ_p initialized with support enforcement")
    
    # =============================================================================
    # FIX 3: Safer Initialization (in agents.py)
    # =============================================================================

        
    def enforce_support_constraints(self):
        """
        Re-apply support constraints to current fields.

        GAUGE-COVARIANT: Operates directly on Σ fields (not Cholesky).
        """
        from agent.masking import FieldEnforcer, SupportRegionSmooth

        if not hasattr(self, 'support') or self.support is None:
            return

        # Ensure smooth support
        if not isinstance(self.support, SupportRegionSmooth):
            # Get mask and shape
            mask_binary = getattr(self.support, 'mask', None)
            if mask_binary is None:
                mask_binary = getattr(self.support, 'mask_binary', None)
            if mask_binary is None:
                mask_binary = np.ones(self.base_manifold.shape, dtype=bool)

            self.support = SupportRegionSmooth(
                mask_binary=mask_binary,
                base_shape=self.base_manifold.shape,
                config=self.config.mask_config
            )

            # --- μ fields ---
            self.mu_q = FieldEnforcer.enforce_mean_field(self.mu_q, self.support)
            self.mu_p = FieldEnforcer.enforce_mean_field(self.mu_p, self.support)

            # --- Σ fields: enforce directly (GAUGE-COVARIANT) ---
            self.Sigma_q = FieldEnforcer.enforce_covariance_field(
                self.Sigma_q,
                self.support,
                inside_scale=self.config.sigma_scale,
                outside_scale=self.config.mask_config.outside_cov_scale,
                use_smooth_transition=self.config.mask_config.use_smooth_cov_transition,
            )

            self.Sigma_p = FieldEnforcer.enforce_covariance_field(
                self.Sigma_p,
                self.support,
                inside_scale=2.0 * self.config.sigma_scale,
                outside_scale=self.config.mask_config.outside_cov_scale,
                use_smooth_transition=self.config.mask_config.use_smooth_cov_transition,
            )

            # Invalidate Cholesky caches
            self._L_q_cache = None
            self._L_p_cache = None
            
          
        self.gauge.phi = FieldEnforcer.enforce_gauge_field(
            self.gauge.phi,
            self.support,
            fill_value=0.0
        )

    @staticmethod
    def enforce_covariance_field(Sigma_field,
                                 support,
                                 inside_scale,
                                 outside_scale,
                                 use_smooth_transition):
        """
        GI-SAFE covariance enforcement.
        Operates directly on Σ instead of L.
    
        Sigma_field: (..., K, K) SPD matrix field.
        """
        mask = support.mask   # shape (...,)
    
        # Outside mask → large covariance
        Sigma_out = outside_scale * np.eye(Sigma_field.shape[-1], dtype=Sigma_field.dtype)
    
        # Inside mask → base scale
        Sigma_in = inside_scale * np.eye(Sigma_field.shape[-1], dtype=Sigma_field.dtype)
    
        if use_smooth_transition:
            # smooth mask χ ∈ [0,1]
            chi = support.smooth_mask  # (...,)
    
            # convex interpolation, but we must stay SPD → use log-euclidean blend
            # However: simple linear mixture is OK because both are SPD and diagonal.
            return (chi[..., None, None] * Sigma_field +
                   (1 - chi[..., None, None]) * Sigma_out)
    
        else:
            # hard mask
            Sigma_new = np.where(mask[..., None, None],
                                 Sigma_field,
                                 Sigma_out)
            return Sigma_new
  

    
   
    
    def _validate_covariance_field(self, Sigma_field: np.ndarray, name: str):
        """
        Validate that covariance field is SPD everywhere.
        
        Raises AssertionError if any matrix is not positive-definite.
        """
        if len(Sigma_field.shape) < 2:
            # 0D case - just a single matrix
            Sigma = Sigma_field
            eigs = np.linalg.eigvalsh(Sigma)
            
            if not np.allclose(Sigma, Sigma.T, atol=1e-5):
                raise AssertionError(f"{name} not symmetric")
            
            if np.any(eigs < 1e-6):
                raise AssertionError(
                    f"{name} not positive-definite: min eigenvalue = {np.min(eigs):.6e}"
                )
            
       #     print(f"[Agent {self.agent_id}] {name} validation:")
        #    print(f"  Eigenvalue range: [{np.min(eigs):.4f}, {np.max(eigs):.4f}]")
            return
        
        # Spatial case
        spatial_shape = Sigma_field.shape[:-2]
        
        min_eig_overall = float('inf')
        max_eig_overall = float('-inf')
        
        for idx in np.ndindex(spatial_shape):
            Sigma = Sigma_field[idx]
            
            # Check symmetry
            if not np.allclose(Sigma, Sigma.T, atol=1e-5):
                raise AssertionError(f"{name} at {idx} not symmetric")
            
            # Check positive-definite
            eigs = np.linalg.eigvalsh(Sigma)
            min_eig_overall = min(min_eig_overall, np.min(eigs))
            max_eig_overall = max(max_eig_overall, np.max(eigs))
            
            if np.any(eigs < 1e-6):
                raise AssertionError(
                    f"{name} at {idx} not positive-definite: "
                    f"min eigenvalue = {np.min(eigs):.6e}"
                )
        
    
    

    # =============================================================================
    # REPLACEMENT: agent/agents.py - _initialize_gauge()
    # =============================================================================
    
    def _initialize_gauge(self):
        """
        Initialize gauge field φ(c) ∈ so(3) ~ ℝ³ with smooth spatial structure.
        
        Key changes from original:
        - Uses spatial smoothing (like mean fields)
        - Creates coherent gauge structure
        - Still enforces support constraints
        """
        spatial_shape = self.geometry.spatial_shape
        
        # ========== GENERATE SMOOTH RANDOM FIELD ==========
        if len(spatial_shape) == 0:
            # 0D particle: simple random
            phi_init = (
                self.config.phi_scale * 
                self.rng.standard_normal(3)
            ).astype(np.float32)
        
        else:
            # Spatial: smooth random field for each so(3) component
            phi_init = self._generate_smooth_gauge_field(
                spatial_shape=spatial_shape,
                scale=self.config.phi_scale,
                smoothness_scale=self._get_gauge_smoothness_scale(),
            )
        
        # ========== CREATE GAUGEFIELD ==========
        from gradients.gauge_fields import GaugeField
        self.gauge = GaugeField(phi_init, self.K, validate=True)
        
        # ========== ENFORCE SUPPORT CONSTRAINTS ==========
        if hasattr(self, 'support') and self.support is not None:
            try:
                from agent.masking import FieldEnforcer, SupportRegionSmooth
                
                # Ensure support is SupportRegionSmooth
                if not isinstance(self.support, SupportRegionSmooth):
                    
                    mask_config = (self.config.mask_config 
                                  if hasattr(self.config, 'mask_config') 
                                  else MaskConfig())
                    
                    self.support = SupportRegionSmooth(
                        mask_binary=self.support.mask,
                        base_shape=self.support.base_shape,
                        config=mask_config
                    )
                
                # Enforce: φ = 0 outside support
                self.gauge.phi = FieldEnforcer.enforce_gauge_field(
                    self.gauge.phi,
                    self.support,
                    fill_value=0.0
                )
                
                # Validate enforcement worked
                if len(spatial_shape) > 0:
                    mask = self.support.mask_continuous
                    outside_mask = mask < self.support.config.min_mask_for_normal_cov
                    
                    if np.any(outside_mask):
                        phi_norm_outside = np.mean(
                            np.linalg.norm(self.gauge.phi[outside_mask], axis=-1)
                        )
                        
                        if phi_norm_outside > 1e-6:
                            print(f"  ⚠️  Agent {self.agent_id}: φ not fully zero outside "
                                  f"(||φ||_outside = {phi_norm_outside:.2e})")
            
            except ImportError:
                # Fallback: manual masking
                if hasattr(self.support, 'mask_continuous'):
                    mask = self.support.mask_continuous[..., None]
                    self.gauge.phi = self.gauge.phi * mask
    
    
    # =============================================================================
    # NEW METHOD: Add to Agent class in agents.py
    # =============================================================================
    
    def _generate_smooth_gauge_field(
        self,
        spatial_shape: Tuple[int, ...],
        scale: float,
        smoothness_scale: float,
    ) -> np.ndarray:
        """
        Generate smooth random gauge field φ(c) ∈ so(3).
        
        Similar to _generate_smooth_mean_field but for gauge frames.
        
        Args:
            spatial_shape: Shape of base manifold
            scale: Overall magnitude (phi_scale)
            smoothness_scale: Spatial smoothing parameter (in grid units)
        
        Returns:
            phi: Smooth gauge field, shape (*spatial_shape, 3)
        
        Algorithm:
            1. Generate white noise for each so(3) component
            2. Apply Gaussian smoothing
            3. Normalize to desired scale
            4. Ensure stays within principal ball |φ| < π
        """
        from scipy.ndimage import gaussian_filter
        
        phi_field = np.zeros((*spatial_shape, 3), dtype=np.float32)
        
        # For each so(3) generator direction
        for a in range(3):
            # White noise
            noise = self.rng.standard_normal(spatial_shape)
            
            # Spatial smoothing
            smooth = gaussian_filter(
                noise,
                sigma=smoothness_scale,
                mode='wrap'  # Periodic boundaries
            )
            
            # Normalize: smooth has unit variance after filtering
            smooth_std = np.std(smooth)
            if smooth_std > 1e-8:
                smooth = smooth / smooth_std
            
            # Scale to desired magnitude
            phi_field[..., a] = scale * smooth
        
        # ========== ENSURE WITHIN PRINCIPAL BALL ==========
        # For SO(3), φ must satisfy |φ| < π (branch cut of exp map)
        phi_norm = np.linalg.norm(phi_field, axis=-1, keepdims=True)
        max_norm = np.pi * 0.9  # Safety margin
        
        # Scale down if any exceed limit
        exceeds = phi_norm > max_norm
        if np.any(exceeds):
            # Rescale only where needed
            scale_factor = np.where(exceeds, max_norm / (phi_norm + 1e-8), 1.0)
            phi_field = phi_field * scale_factor
        
        return phi_field
    
    
    def _get_gauge_smoothness_scale(self) -> float:
        """
        Determine smoothness scale for gauge field initialization.
        
        Returns smoothness parameter σ for Gaussian filtering.
        Larger σ → smoother fields.
        
        Heuristics:
        - For small grids (<10): σ = 1.0 (minimal smoothing)
        - For medium grids (10-50): σ = 2.0-3.0 (moderate)
        - For large grids (>50): σ = 0.1 * min(dimensions)
        """
        spatial_shape = self.geometry.spatial_shape
        
        if len(spatial_shape) == 0:
            return 0.0  # 0D particle
        
        # Use minimum dimension as reference
        min_dim = min(spatial_shape)
        
        if min_dim < 10:
            return 1.0
        elif min_dim < 50:
            return 2.0
        else:
            return max(2.0, 0.1 * min_dim)

    
    def set_observations(self, x_obs: np.ndarray):
        """Set observations for this agent."""
        expected_shape = (*self.geometry.spatial_shape, self.D)
        if x_obs.shape != expected_shape:
            raise ValueError(
                f"Observation shape {x_obs.shape} doesn't match "
                f"expected {expected_shape}"
            )
        self.x_obs = x_obs.astype(np.float32)
    
       
    def get_belief_distribution(self) -> GaussianDistribution:
        """Wrap belief in GaussianDistribution for transport operations."""
        return GaussianDistribution(mu=self.mu_q, Sigma=self.Sigma_q)
    
    def get_prior_distribution(self) -> GaussianDistribution:
        """Wrap prior in GaussianDistribution for transport operations."""
        return GaussianDistribution(mu=self.mu_p, Sigma=self.Sigma_p)
    
    def invalidate_caches(self):
        """Mark cached computations as stale after parameter updates."""
        self._transport_cache_dirty = True
    
    def summary(self) -> str:
        """Get agent summary string."""
        return (
            f"Agent {self.agent_id}:\n"
            f"  Base manifold: {self.base_manifold.ndim}D, shape={self.base_manifold.shape}\n"
            f"  Support: {self.support.n_active}/{self.base_manifold.n_points} points "
            f"({self.support.coverage:.1%} coverage)\n"
            f"  Latent dim: K={self.K}, Obs dim: D={self.D}\n"
            f"  Fields: μ_q{self.mu_q.shape}, Σ_q{self.Sigma_q.shape}, φ{self.gauge.phi.shape}\n"
            f"  Has observations: {self.x_obs is not None}"
        )
    
    def count_parameters(self) -> int:
        """Count total learnable parameters."""
        n_points = self.support.n_active
        
        # Belief: μ_q (K) + Σ_q (K*(K+1)/2 due to symmetry)
        belief_params = n_points * (self.K + self.K * (self.K + 1) // 2)
        
        # Prior: same as belief
        prior_params = belief_params
        
        # Gauge: φ (3 components for SO(3))
        gauge_params = n_points * 3
        
        return belief_params + prior_params + gauge_params




