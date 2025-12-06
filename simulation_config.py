"""
Simulation Configuration Dataclass

Consolidates all simulation parameters into a single, well-organized configuration.
Replaces 50+ global variables with a structured, type-safe configuration.
"""

from dataclasses import dataclass, field
from typing import Tuple, Optional
import numpy as np


@dataclass
class SimulationConfig:
    """Complete configuration for multi-agent simulation with emergence."""

    # =============================================================================
    # Experiment Metadata
    # =============================================================================
    experiment_name: str        = "_playground"
    experiment_description: str = "Multi-agent with smooth support boundaries"
    output_dir: str             = "_results"
    seed: int                   = 2

     # Phase space tracking
    track_phase_space: bool = True
    phase_space_track_interval: int = 1
    phase_space_track_sigma: bool = True
    phase_space_track_gauge: bool = True

    # =============================================================================
    # Training Loop
    # =============================================================================
    n_steps: int                = 1000
    log_every: int              = 1
    skip_initial_steps: int     = 0  #For analysis plots (ignore transients)

    # Early stopping conditions (any condition triggers stop)
    stop_if_n_scales_reached: Optional[int]  = 25  # Stop when this many scales exist
    stop_if_n_condensations: Optional[int]   = None  # Stop after this many meta-agents formed
    stop_if_max_active_agents: Optional[int] = 50  # Stop when total active agents reaches this
    stop_if_min_active_agents: Optional[int] = None  # Stop if active agents drops below this

    # =============================================================================
    # Agents & Latent Space
    # =============================================================================
    n_agents: int           = 6
    K_latent: int           = 11
    D_x: int                = 5  # Observation dimension

    # Field initialization scales
    mu_scale: float         = 1
    sigma_scale: float      = 1.0
    phi_scale: float        = 0.5
    mean_smoothness: float  = 1.0

    # Connection
    connection_type: str    = 'flat'  # flat, random, constant
    use_connection: bool    = False

    # =============================================================================
    # Hierarchical Emergence
    # =============================================================================
    enable_emergence: bool          = False
    consensus_threshold: float      = 0.1  # KL threshold for epistemic death
    consensus_check_interval: int   = 2  # Check every N steps
    min_cluster_size: int           = 2  # Min agents to form meta-agent
    max_scale: int                  = 20  # Highest scale (prevents runaway emergence)
    max_meta_membership: int        = 10  # Max constituents per meta-agent
    max_total_agents: int           = 50  # Hard cap across ALL scales

    enable_cross_scale_priors: bool = True  # Top-down prior propagation
    enable_timescale_sep: bool      = False  # Timescale separation
    info_metric: str                = "fisher_metric"  # Information change metric

    # Ouroboros Tower: Multi-scale hyperprior propagation
    enable_hyperprior_tower: bool = True  # Wheeler's "it from bit" extended
    max_hyperprior_depth: int     = 3  # How many levels up to receive priors
    hyperprior_decay: float       = 0.3  # Exponential decay for ancestral priors

    # =============================================================================
    # Energy Weights (Cultural/Hierarchical Tension)
    # =============================================================================
    lambda_self: float           = 1  # Individual identity (resist conformity)
    lambda_belief_align: float   = 1  # Peer pressure (social)
    lambda_prior_align: float    = 0  # Cultural authority (top-down)
    lambda_obs: float            = 0  # External observations
    lambda_phi: float            = 1  # Gauge coupling

    kappa_beta: float            = 1.0  # Softmax temperature (belief align)
    kappa_gamma: float           = 1.0  # Softmax temperature (prior align)

    identical_priors: str        = "lock"  # off, lock, init_copy
    identical_priors_source: str = "first"  # first or mean

    # =============================================================================
    # Learning Rates
    # =============================================================================
    lr_mu_q: float           = 0.001
    lr_sigma_q: float        = 0.0005
    lr_mu_p: float           = 0.0005
    lr_sigma_p: float        = 0.00025
    lr_phi: float            = 0.01
    
    # =============================================================================
    # Spatial Geometry
    # =============================================================================
    spatial_shape: Tuple                               = ()
    manifold_topology: str                             = "periodic"  # periodic, flat, sphere
    
    support_pattern: str                               = "point"  # point, circles_2d, full, intervals_1d
    agent_placement_2d: str                            = "center"  # center, random, grid
    agent_radius: float                                = 3.0  # For 2D circular supports
    random_radius_range: Optional[Tuple[float, float]] = None  # (min, max) or None
    interval_overlap_fraction: float                   = 0.25  # For 1D intervals

    # =============================================================================
    # Masking (Smooth Support Boundaries)
    # =============================================================================
    mask_type: str                  = "gaussian"  # hard, smooth, gaussian
    overlap_threshold: float        = 1e-1  # Ignore overlaps below this
    min_mask_for_normal_cov: float  = 1e-1  # Below this, use large Σ

    # Gaussian mask parameters
    gaussian_sigma: float           = field(init=False)  # Computed from overlap_threshold
    gaussian_cutoff_sigma: float    = 3.0  # Hard cutoff at N*σ

    # Smooth mask parameters
    smooth_width: float             = 0.1  # Transition width (relative to radius)

    # Covariance outside support
    covariance_strategy: str        = "smooth"  # Gaussian-filtered Cholesky
    outside_cov_scale: float        = 1e3  # Scale for diagonal Σ outside support
    use_smooth_cov_transition: bool = False  # Interpolate Σ at boundaries

    # =============================================================================
    # Observation Model
    # =============================================================================
    obs_bias_scale: float              = 0.5
    obs_noise_scale: float             = 1.0
    obs_w_scale: float                 = 0.5
    obs_r_scale: float                 = 1.0
    obs_ground_truth_amplitude: float  = 0.5
    obs_ground_truth_modes: int        = 3



    # =============================================================================
    # Hamiltonian Dynamics (Alternative to Gradient Flow)
    # =============================================================================
    enable_hamiltonian: bool           = True  # Enable Hamiltonian (underdamped) dynamics
    hamiltonian_integrator: str        = "SymplecticEuler"  # SymplecticEuler, Verlet, Ruth3, PEFRL
    hamiltonian_dt: float              = 0.025  # Time step for symplectic integration
    hamiltonian_friction: float        = 1  # Damping coefficient γ (0 = conservative)
    hamiltonian_mass_scale: float      = 1  # Mass scale for kinetic term
    hamiltonian_include_gauge: bool    = True  # Include gauge field φ in phase space (full field theory)
    
    
    # =============================================================================
    # Pullback Geometry Tracking (Emergent Spacetime)
    # =============================================================================
    track_pullback_geometry: bool         = False  # Enable pullback metric tracking
    geometry_track_interval: int          = 1  # Record geometry every N steps
    geometry_enable_consensus: bool       = False  # Compute consensus metrics (expensive!)
    geometry_enable_gauge_averaging: bool = False  # Gauge averaging (very expensive!)
    geometry_gauge_samples: int           = 1  # MC samples for gauge averaging
    geometry_lambda_obs: float            = 0.1  # Observable sector threshold
    geometry_lambda_dark: float           = 0.01  # Dark sector threshold



    # =============================================================================
    # Agent Field Visualization (2D Spatial Manifolds)
    # =============================================================================
   
    viz_track_interval: int                          = 10  # Record snapshots every N steps
    viz_scales: Tuple[int, ...]                      = (0,1,2)  # Which hierarchical scales to image
    viz_fields: Tuple[str, ...]                      = ("mu_q", "phi")  # Fields: mu_q, Sigma_q, mu_p, Sigma_p, phi
    viz_latent_components: Optional[Tuple[int, ...]] = None  # Which K components (None = all)

 
   
    # Comprehensive meta-agent visualizations (hierarchy, consensus, energy)
    generate_meta_visualizations: bool = True
    snapshot_interval: int             =   1  # Capture analyzer snapshots every N steps



    def __post_init__(self):
        """Compute derived parameters and validate settings."""
        # Compute gaussian_sigma from overlap_threshold
        if self.overlap_threshold > 0:
            self.gaussian_sigma = 1.0 / np.sqrt(-2 * np.log(self.overlap_threshold))
        else:
            self.gaussian_sigma = 1.0

        # Validate Hamiltonian integrator
        valid_integrators = {"SymplecticEuler", "Verlet", "StormerVerlet", "Ruth3", "PEFRL"}
        if self.hamiltonian_integrator not in valid_integrators:
            raise ValueError(
                f"Invalid hamiltonian_integrator '{self.hamiltonian_integrator}'. "
                f"Valid options: {valid_integrators}"
            )

    def to_dict(self) -> dict:
        """Convert config to dictionary for serialization."""
        return {
            k: v for k, v in self.__dict__.items()
            if not k.startswith('_')
        }

    def save(self, filepath: str):
        """Save configuration to text file for reproducibility."""
        with open(filepath, 'w') as f:
            f.write("# Simulation Configuration\n")
            f.write(f"# {'='*60}\n\n")

            sections = {
                "Experiment": ["experiment_name", "experiment_description", "seed"],
                "Manifold": ["spatial_shape", "manifold_topology"],
                "Training": ["n_steps", "log_every", "skip_initial_steps",
                            "stop_if_n_scales_reached", "stop_if_n_condensations",
                            "stop_if_max_active_agents", "stop_if_min_active_agents"],
                "Agents": ["n_agents", "K_latent", "D_x", "mu_scale", "sigma_scale",
                          "phi_scale", "mean_smoothness"],
                "Emergence": ["enable_emergence", "consensus_threshold", "consensus_check_interval",
                             "min_cluster_size", "max_scale", "max_meta_membership",
                             "max_total_agents", "enable_cross_scale_priors", "enable_timescale_sep"],
                "Energy": ["lambda_self", "lambda_belief_align", "lambda_prior_align",
                          "lambda_obs", "lambda_phi", "kappa_beta", "kappa_gamma"],
                "Learning Rates": ["lr_mu_q", "lr_sigma_q", "lr_mu_p", "lr_sigma_p", "lr_phi"],
                "Support": ["support_pattern", "agent_placement_2d", "agent_radius",
                           "random_radius_range", "interval_overlap_fraction"],
                "Masking": ["mask_type", "gaussian_sigma", "gaussian_cutoff_sigma",
                           "smooth_width", "overlap_threshold", "outside_cov_scale"],
                "Observations": ["obs_bias_scale", "obs_noise_scale", "obs_w_scale",
                               "obs_r_scale", "obs_ground_truth_amplitude", "obs_ground_truth_modes"],
                "Diagnostics": ["run_initial_diagnostics", "run_final_diagnostics",
                              "save_diagnostic_plots", "save_diagnostic_report"],
                "Hamiltonian": ["enable_hamiltonian", "hamiltonian_integrator",
                               "hamiltonian_dt", "hamiltonian_friction", "hamiltonian_mass_scale"],
                "Geometry": ["track_pullback_geometry", "geometry_track_interval",
                           "geometry_enable_consensus", "geometry_enable_gauge_averaging",
                           "geometry_gauge_samples", "geometry_lambda_obs", "geometry_lambda_dark"],
            }

            for section_name, keys in sections.items():
                f.write(f"[{section_name}]\n")
                for key in keys:
                    if hasattr(self, key):
                        value = getattr(self, key)
                        f.write(f"{key:<30} = {value}\n")
                f.write("\n")


# =============================================================================
# Preset Configurations
# =============================================================================

def default_config() -> SimulationConfig:
    """Default configuration for standard runs."""
    return SimulationConfig()


def emergence_demo_config() -> SimulationConfig:
    """Configuration optimized for demonstrating hierarchical emergence."""
    return SimulationConfig(
        experiment_name="_emergence_demo",
        experiment_description="Optimized for demonstrating meta-agent formation",
        n_agents=8,
        n_steps=100,
        enable_emergence=True,
        consensus_threshold=0.05,
        consensus_check_interval=5,
        lambda_self=3.0,
        lambda_belief_align=2.0,
        lambda_prior_align=2.5,
        enable_cross_scale_priors=True,
        # Early stopping: stop once we reach 5 scales or form 15 meta-agents
        stop_if_n_scales_reached=5,
        stop_if_n_condensations=15,
    )


def ouroboros_config() -> SimulationConfig:
    """Configuration with Ouroboros Tower (multi-scale hyperpriors)."""
    return SimulationConfig(
        experiment_name="_ouroboros_tower",
        experiment_description="Wheeler's 'it from bit' with ancestral priors",
        enable_emergence=True,
        enable_hyperprior_tower=True,
        max_hyperprior_depth=3,
        hyperprior_decay=0.3,
    )




def flat_agents_config() -> SimulationConfig:
    """Configuration for flat multi-agent system (no emergence)."""
    return SimulationConfig(
        experiment_name="_flat_agents",
        experiment_description="Standard multi-agent without emergence",
        enable_emergence=False,
        n_agents=5,
        n_steps=50,
    )


def hamiltonian_config() -> SimulationConfig:
    """
    Configuration for Hamiltonian (underdamped) dynamics.

    Uses symplectic integration to preserve phase space structure.
    Energy is approximately conserved (bounded drift).

    Key parameters:
    - hamiltonian_dt: Time step for integration (smaller = more accurate)
    - hamiltonian_friction: Damping coefficient (0 = conservative)
    - hamiltonian_mass_scale: Mass scaling for kinetic term

    Dynamics regimes:
    - friction=0: Pure Hamiltonian (underdamped, energy-conserving)
    - friction=0.1: Light damping (approaches equilibrium slowly)
    - friction=1.0: Critical damping (fastest convergence)
    - friction=10.0: Heavy damping (approaches gradient flow)
    """
    return SimulationConfig(
        experiment_name="_hamiltonian",
        experiment_description="Hamiltonian dynamics with symplectic integration",

        # Enable Hamiltonian dynamics
        enable_hamiltonian=True,
        hamiltonian_integrator="Verlet",  # Best balance of accuracy/speed
        hamiltonian_dt=0.01,  # Conservative time step
        hamiltonian_friction=0.0,  # Pure Hamiltonian (conservative)
        hamiltonian_mass_scale=1.0,

        # Standard agents
        n_agents=5,
        K_latent=3,
        n_steps=200,
        log_every=10,

        # Disable emergence for flat Hamiltonian
        enable_emergence=False,

        # Energy weights
        lambda_self=1.0,
        lambda_belief_align=1.0,
        lambda_prior_align=1.0,
        lambda_obs=0.0,
        lambda_phi=0.0,
    )


def hamiltonian_emergence_config() -> SimulationConfig:
    """
    Configuration for Hamiltonian dynamics WITH emergence.

    Combines:
    - Symplectic integration (energy-preserving dynamics)
    - Meta-agent emergence (hierarchical structure formation)
    - Ouroboros tower (cross-scale prior propagation)

    This is the most sophisticated training mode, enabling study of
    how energy conservation interacts with emergence phenomena.
    """
    return SimulationConfig(
        experiment_name="_hamiltonian_emergence",
        experiment_description="Hamiltonian dynamics with hierarchical emergence",

        # Enable Hamiltonian dynamics
        enable_hamiltonian=True,
        hamiltonian_integrator="Verlet",
        hamiltonian_dt=0.01,
        hamiltonian_friction=0.1,  # Light damping for stability
        hamiltonian_mass_scale=1.0,

        # Enable emergence
        enable_emergence=True,
        consensus_threshold=0.05,
        consensus_check_interval=5,
        min_cluster_size=2,
        max_scale=10,

        # Ouroboros Tower
        enable_cross_scale_priors=True,
        enable_hyperprior_tower=True,
        max_hyperprior_depth=3,
        hyperprior_decay=0.5,

        # Agents
        n_agents=8,
        K_latent=3,
        n_steps=300,
        log_every=10,

        # Balanced energy landscape
        lambda_self=1.0,
        lambda_belief_align=1.0,
        lambda_prior_align=1.0,
        lambda_obs=0.0,
        lambda_phi=0.0,

        # Visualization
        generate_meta_visualizations=True,
        snapshot_interval=5,
    )


def critical_damping_config() -> SimulationConfig:
    """
    Configuration for critically damped dynamics.

    Critical damping provides fastest convergence without oscillation.
    Intermediate between pure Hamiltonian and gradient flow.

    Good for:
    - Faster equilibration than gradient flow
    - More stability than pure Hamiltonian
    - Studying transition between dynamics regimes
    """
    return SimulationConfig(
        experiment_name="_critical_damping",
        experiment_description="Critically damped Hamiltonian dynamics",

        # Hamiltonian with critical friction
        enable_hamiltonian=True,
        hamiltonian_integrator="Verlet",
        hamiltonian_dt=0.01,
        hamiltonian_friction=1.0,  # Critical damping regime
        hamiltonian_mass_scale=1.0,

        # Standard setup
        n_agents=5,
        K_latent=3,
        n_steps=100,
        log_every=5,
        enable_emergence=False,

        # Energy weights
        lambda_self=1.0,
        lambda_belief_align=1.0,
        lambda_prior_align=1.0,
        lambda_obs=0.0,
        lambda_phi=0.0,
    )