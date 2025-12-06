#!/usr/bin/env python3
"""
Configuration System - PHASE 2
===============================

PHASE 2 CHANGES:
- Added @property methods for derived values
- NO other changes (still mutable, no frozen)

Benefits:
- config.D_effective instead of (config.D if config.D else config.K)
- config.is_particle instead of len(config.spatial_shape) == 0
- More readable code

Author: Phase 2 - Cached Properties
Date: November 2025
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict, Any, Literal
from pathlib import Path

from agent.masking import MaskConfig





@dataclass
class SystemConfig:
    """Configuration for multi-agent system."""
    
    # === EXISTING FIELDS (keep these!) ===
    lambda_self:         float = 1.0
    lambda_belief_align: float = 1.0
    lambda_prior_align:  float = 0.0
    lambda_phi:          float = 0.0
    lambda_obs:          float = 0.0
    lambda_obs_meta:     float = 1.0  # Meta-agent observation of constituents (bottom-up coupling)
    lambda_gauge_smooth: float = 0.0
    
    kappa_beta: float          = 1.0
    kappa_gamma: float         = 1.0
    
    overlap_threshold: float   = 0.1
    
    
    # Identical priors option
    identical_priors: Literal["off", "init_copy", "lock"] = "off"

    # (optional) choose the source of the shared prior
    identical_priors_source: Literal["first", "mean"] = "first"
    
    eps: float                 = 1e-8
    
    cache_transports: bool     = True
    cache_size: int            = 10000

    mask_config: MaskConfig = field(default_factory=lambda: MaskConfig(...))
    
    use_connection: bool = False
    connection_init_mode: Literal['flat', 'random', 'constant'] = 'flat'
    connection_scale: float = 1.0
    connection_const: Optional[np.ndarray] = None

    # =========================================================================
    # ⚡ ADD THESE NEW FIELDS FOR OBSERVATION MODEL
    # =========================================================================
    
    # Observation dimensions
    D_x: int           = 8                          # Observation space dimension
    
    # Observation matrix/covariance initialization
    obs_W_scale: float = 0.5              # W_obs ~ N(0, obs_W_scale²)
    obs_R_scale: float = 0.3              # R_obs covariance scale
    
    # Observation data generation
    obs_noise_scale: float = 0.2          # Measurement noise std
    obs_bias_scale: float  = 0.3           # Agent-specific bias std (KEY!)
    
    # Ground truth generation
    obs_ground_truth_modes: int       = 3       # Sinusoidal modes
    obs_ground_truth_amplitude: float = 1.0  # Field amplitude
    
    # Reproducibility
    seed: Optional[int] = None            # RNG seed

    # =========================================================================
    # END NEW FIELDS
    # =========================================================================

    def __post_init__(self):
        """Validate configuration."""
        self.validate()

    def validate(self) -> None:
        """Validate all parameters."""
        # EXISTING VALIDATION (keep this!)
        lambdas = [
            self.lambda_self, self.lambda_belief_align,
            self.lambda_prior_align, self.lambda_obs,
            self.lambda_obs_meta, self.lambda_gauge_smooth
        ]
        if any(lam < 0 for lam in lambdas):
            raise ValueError("All lambda weights must be non-negative")
        
        if self.kappa_beta <= 0:
            raise ValueError(f"kappa_beta must be positive, got {self.kappa_beta}")
        if self.kappa_gamma <= 0:
            raise ValueError(f"kappa_gamma must be positive, got {self.kappa_gamma}")
        
        if not 0 <= self.overlap_threshold <= 1:
            raise ValueError(
                f"overlap_threshold must be in [0, 1], got {self.overlap_threshold}"
            )
        # ... your existing validation ...
        if self.identical_priors not in ("off", "init_copy", "lock"):
            raise ValueError(f"identical_priors must be 'off' | 'init_copy' | 'lock', got {self.identical_priors}")
        if self.identical_priors_source not in ("first", "mean"):
            raise ValueError(f"identical_priors_source must be 'first' | 'mean', got {self.identical_priors_source}")
        if self.eps <= 0:
            raise ValueError(f"eps must be positive, got {self.eps}")
        
        if self.cache_size <= 0:
            raise ValueError(f"cache_size must be positive, got {self.cache_size}")
        
        # ⚡ ADD NEW VALIDATION FOR OBSERVATION PARAMETERS
        if self.D_x <= 0:
            raise ValueError(f"D_x must be positive, got {self.D_x}")
        
        if self.obs_W_scale <= 0:
            raise ValueError(f"obs_W_scale must be positive, got {self.obs_W_scale}")
        
        if self.obs_R_scale <= 0:
            raise ValueError(f"obs_R_scale must be positive, got {self.obs_R_scale}")
        
        if self.obs_noise_scale < 0:
            raise ValueError(f"obs_noise_scale must be non-negative, got {self.obs_noise_scale}")
        
        if self.obs_bias_scale < 0:
            raise ValueError(f"obs_bias_scale must be non-negative, got {self.obs_bias_scale}")
        
        if self.obs_ground_truth_modes < 1:
            raise ValueError(f"obs_ground_truth_modes must be >= 1, got {self.obs_ground_truth_modes}")
        
        if self.obs_ground_truth_amplitude <= 0:
            raise ValueError(f"obs_ground_truth_amplitude must be positive, got {self.obs_ground_truth_amplitude}")
    
    # EXISTING PROPERTIES (keep these!)
    @property
    def has_observations(self) -> bool:
        return self.lambda_obs > 0
    
    @property
    def has_belief_alignment(self) -> bool:
        return self.lambda_belief_align > 0
    
    @property
    def has_prior_alignment(self) -> bool:
        return self.lambda_prior_align > 0
    
    @property
    def has_gauge_smoothness(self) -> bool:
        return self.lambda_gauge_smooth > 0
    
    @property
    def has_self_energy(self) -> bool:
        return self.lambda_self > 0

    @property
    def trains_phi(self) -> bool:
        return self.lambda_phi > 0
    
    # ⚡ ADD NEW HELPER METHOD
    def get_obs_rng(self) -> np.random.Generator:
        """Get random generator for observation model."""
        seed = self.seed if self.seed is not None else 0
        return np.random.default_rng(seed + 10)


# =============================================================================
# THAT'S ALL! Just add those fields and validation to your existing class.
# =============================================================================



# =============================================================================
# Agent Configuration (PHASE 2: Added properties)
# =============================================================================

@dataclass
class AgentConfig:
    """Configuration for individual agents."""

    # === All existing fields (UNCHANGED) ===
    spatial_shape: Tuple[int, ...] = ()
    K: int = 3

    mu_scale: float = 0.2
    sigma_scale: float = 0.1
    phi_scale: float = 0.1

    covariance_strategy: str = "smooth"
    covariance_kwargs: Optional[dict] = None

    smooth_mean_fields: bool = True
    mean_smoothness_scale: Optional[float] = 1

    use_identity_observation: bool = False
    observation_noise: float = 0.1

    trust_region_sigma: float = 2e-1
    eps: float = 1e-8

    lr_mu_q: float = 0.0
    lr_sigma_q: float = 0.0
    lr_mu_p: float = 0.0
    lr_sigma_p: float = 0.0
    lr_phi: float = 0

    alpha: float = 1.0

    # Mask configuration for field agents
    mask_config: MaskConfig = field(default_factory=MaskConfig)

    def __post_init__(self):
        """Validate and fill defaults."""
        self.validate()

    def validate(self) -> None:
        """Validate and fill in any missing defaults."""
        if self.K <= 0:
            raise ValueError(f"K must be positive, got {self.K}")

       

        if (
            self.use_identity_observation
            and self.D is not None
            and self.D != self.K
        ):
            raise ValueError(
                f"use_identity_observation=True requires D == K, "
                f"got D={self.D}, K={self.K}"
            )

       
        # Validate learning rates
        if self.lr_mu_q < 0:
            raise ValueError(f"lr_mu must be positive, got {self.lr_mu}")
        if self.lr_sigma_q < 0:
            raise ValueError(f"lr_sigma must be positive, got {self.lr_sigma}")
        if self.lr_mu_p < 0:
            raise ValueError(f"lr_mu must be positive, got {self.lr_mu}")
        if self.lr_sigma_p < 0:
            raise ValueError(f"lr_sigma must be positive, got {self.lr_sigma}")
        if self.lr_phi < 0:
            raise ValueError(f"lr_phi must be positive, got {self.lr_phi}")

        # Validate scales
        if self.mu_scale < 0:
            raise ValueError(f"mu_scale must be non-negative, got {self.mu_scale}")
        if self.sigma_scale <= 0:
            raise ValueError(f"sigma_scale must be positive, got {self.sigma_scale}")
        if self.phi_scale < 0:
            raise ValueError(f"phi_scale must be non-negative, got {self.phi_scale}")

        # Validate trust region
        if self.trust_region_sigma <= 0:
            raise ValueError(
                f"trust_region_sigma must be positive, got {self.trust_region_sigma}"
            )

        # Validate eps
        if self.eps <= 0:
            raise ValueError(f"eps must be positive, got {self.eps}")

        # Default covariance_kwargs based on strategy
        if self.covariance_kwargs is None:
            if self.covariance_strategy == "smooth":
                self.covariance_kwargs = {"smoothness_scale": 5.0}
            elif self.covariance_strategy == "random":
                self.covariance_kwargs = {
                    "min_eigenvalue": 0.1,
                    "max_eigenvalue": 10.0,
                }
            elif self.covariance_strategy in {"gradient", "center", "random_centers"}:
                self.covariance_kwargs = {"variation_scale": 5.0}
            else:  # 'constant' or anything else
                self.covariance_kwargs = {}
    
    # =========================================================================
    # NEW PHASE 2: Convenience Properties
    # =========================================================================
    
   
    
    @property
    def ndim(self) -> int:
        """
        Spatial dimensionality.
        
        Returns number of spatial dimensions.
        
        Usage:
            # Old way:
            ndim = len(config.spatial_shape)
            
            # New way:
            ndim = config.ndim
        """
        return len(self.spatial_shape)
    
    @property
    def n_spatial_points(self) -> int:
        """
        Total number of spatial points.
        
        Returns product of spatial dimensions.
        
        Usage:
            # Old way:
            n_points = int(np.prod(config.spatial_shape)) if config.spatial_shape else 1
            
            # New way:
            n_points = config.n_spatial_points
        """
        return int(np.prod(self.spatial_shape)) if self.spatial_shape else 1
    
    @property
    def is_particle(self) -> bool:
        """
        True if this is a 0D particle agent.
        
        Returns True when spatial_shape is empty.
        
        Usage:
            # Old way:
            if len(config.spatial_shape) == 0:
                # 0D case
            
            # New way:
            if config.is_particle:
                # 0D case
        """
        return len(self.spatial_shape) == 0
    
    @property
    def covariance_kwargs_with_defaults(self) -> Dict[str, Any]:
        """
        Covariance kwargs with strategy-specific defaults.
        
        Always returns a dict (never None).
        
        Usage:
            # Old way:
            if config.covariance_kwargs is None:
                if config.covariance_strategy == "smooth":
                    kwargs = {"smoothness_scale": 5.0}
                # ... more if/else
            else:
                kwargs = config.covariance_kwargs
            
            # New way:
            kwargs = config.covariance_kwargs_with_defaults
        """
        if self.covariance_kwargs is not None:
            return self.covariance_kwargs
        
        # Strategy-specific defaults
        defaults = {
            "smooth": {"smoothness_scale": 5.0},
            "gradient": {"variation_scale": 5.0},
            "random_centers": {"variation_scale": 5.0},
            "random": {"min_eigenvalue": 0.1, "max_eigenvalue": 10.0},
            "constant": {},
        }
        
        return defaults.get(self.covariance_strategy, {})
    
    @property
    def mean_smoothness_scale_effective(self) -> float:
        if self.mean_smoothness_scale is not None:
            return self.mean_smoothness_scale
        if self.covariance_strategy == 'smooth':
            return self.covariance_kwargs_with_defaults.get('smoothness_scale', 1.0)
        if len(self.spatial_shape) > 0:
            min_dim = min(self.spatial_shape)
            return max(1, 0.1 * min_dim)
        return 1.0







# =============================================================================
# Training Configuration (PHASE 2: No new properties needed)
# =============================================================================

@dataclass
class TrainingConfig:
    """Configuration for training loop."""
    
    # Training duration
    n_steps: int = 200
    
    # Learning rates
    lr_mu_q: float = 0.00
    lr_sigma_q: float = 0.0
    lr_mu_p: float = 0
    lr_sigma_p: float = 0
    lr_phi: float = 0
    
    # SPD retraction parameters
    trust_region_sigma: float = 0.2  # Trust region for Σ updates
    retraction_mode_sigma: str = 'exp'  # 'exp' or 'cayley'
    
    # Gauge retraction parameters
    retraction_mode_phi: str = 'mod2pi'  # 'mod2pi' or 'project'
    gauge_margin: float = 1e-2  # Safety margin from branch cut
    
    # Logging and diagnostics
    log_every: int = 1
    save_history: bool = True

    # Snapshot storage for metric analysis
    save_snapshots: bool = False  # Enable agent state snapshots
    snapshot_every: int = 10  # Save snapshot every N steps

    # Early stopping
    early_stop_threshold: Optional[float] = None  # Min energy decrease
    early_stop_patience: int = 50  # Steps without improvement
    
    # Checkpointing
    checkpoint_every: int = 100
    checkpoint_dir: Optional[Path] = None

    save_checkpoints: bool = False
    



