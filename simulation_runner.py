#!/usr/bin/env python3
"""
Streamlined Simulation Runner

Clean, modular simulation orchestrator that replaces the bloated simulation_suite.py

Key improvements:
- Uses SimulationConfig dataclass (no more 50+ globals!)
- Extracted GradientSystemAdapter to meta/gradient_adapter.py
- Unified training interface (less duplication)
- Clear separation of concerns
- ~500 lines vs 1345 lines (62% reduction!)

Usage:
    python simulation_runner.py                    # Default config
    python simulation_runner.py --preset emergence # Emergence demo
    python simulation_runner.py --preset ouroboros # Ouroboros Tower
"""

import argparse
import pickle
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from simulation_config import (
    SimulationConfig,
    default_config,
    emergence_demo_config,
    ouroboros_config,
    flat_agents_config,
    hamiltonian_config,
    hamiltonian_emergence_config,
    critical_damping_config
)
from config import AgentConfig, SystemConfig, TrainingConfig
from agent.agents import Agent
from agent.system import MultiAgentSystem
from agent.trainer import Trainer, TrainingHistory
from agent.hamiltonian_trainer import HamiltonianTrainer, HamiltonianHistory
from geometry.geometry_base import BaseManifold, TopologyType
from agent.masking import SupportRegionSmooth, SupportPatternsSmooth, MaskConfig

COMP_FULL_ENERGY =  True
# =============================================================================
# System Building (Clean Helper Functions)
# =============================================================================

def build_manifold(cfg: SimulationConfig) -> BaseManifold:
    """Create BaseManifold from configuration."""
    topology_map = {
        "periodic": TopologyType.PERIODIC,
        "flat": TopologyType.FLAT,
        "sphere": TopologyType.SPHERE,
    }
    topology_key = cfg.manifold_topology.lower()
    if topology_key not in topology_map:
        raise ValueError(f"Unknown topology '{topology_key}'. Valid options: {list(topology_map.keys())}")

    return BaseManifold(
        shape=cfg.spatial_shape,
        topology=topology_map[topology_key]
    )


def build_supports(manifold: BaseManifold,
                  cfg: SimulationConfig,
                  rng: np.random.Generator):
    """Build support regions for all agents."""
    from geometry.geometry_base import SupportPatterns

    mask_config = MaskConfig(
        mask_type=cfg.mask_type,
        smooth_width=cfg.smooth_width,
        gaussian_sigma=cfg.gaussian_sigma,
        gaussian_cutoff_sigma=cfg.gaussian_cutoff_sigma,
        overlap_threshold=cfg.overlap_threshold,
        min_mask_for_normal_cov=cfg.min_mask_for_normal_cov,
        outside_cov_scale=cfg.outside_cov_scale,
        use_smooth_cov_transition=cfg.use_smooth_cov_transition
    )

    ndim = manifold.ndim

    # 0D: Point manifold (all agents at same point)
    if ndim == 0:
        return [
            SupportRegionSmooth(
                mask_binary=np.array(True),
                base_shape=(),
                config=mask_config
            ) for _ in range(cfg.n_agents)
        ]

    # 1D: Intervals or full support
    if ndim == 1:
        if cfg.support_pattern == "full":
            basic = [SupportPatterns.full(manifold) for _ in range(cfg.n_agents)]
        elif cfg.support_pattern == "intervals_1d":
            basic = _build_intervals_1d(manifold, cfg)
        else:
            raise ValueError(f"Unsupported 1D pattern: {cfg.support_pattern}")

        return [
            SupportRegionSmooth(s.mask, s.base_shape, mask_config) for s in basic
        ]

    # 2D: Circles or full support
    if ndim == 2:
        if cfg.support_pattern == "full":
            basic = [SupportPatterns.full(manifold) for _ in range(cfg.n_agents)]
            return [SupportRegionSmooth(s.mask, s.base_shape, mask_config) for s in basic]
        elif cfg.support_pattern == "circles_2d":
            return _build_circles_2d(manifold, cfg, mask_config, rng)
        else:
            raise ValueError(f"Unsupported 2D pattern: {cfg.support_pattern}")

    # Higher dimensions: Full support
    basic = [SupportPatterns.full(manifold) for _ in range(cfg.n_agents)]
    return [SupportRegionSmooth(s.mask, s.base_shape, mask_config) for s in basic]


def _build_intervals_1d(manifold, cfg):
    """Build overlapping intervals for 1D manifold."""
    from geometry.geometry_base import SupportPatterns

    n_points = manifold.shape[0]
    base_width = n_points / cfg.n_agents
    overlap = int(base_width * cfg.interval_overlap_fraction)

    supports = []
    for i in range(cfg.n_agents):
        start = int(max(0, round(i * base_width) - overlap // 2))
        end = int(min(n_points, round((i + 1) * base_width) + overlap // 2))
        supports.append(SupportPatterns.interval(manifold, start=start, end=end))
    return supports


def _build_circles_2d(manifold, cfg, mask_config, rng):
    """Build circular supports for 2D manifold."""
    H, W = manifold.shape
    supports = []

    if cfg.agent_placement_2d == "center":
        center = (H // 2, W // 2)
        for _ in range(cfg.n_agents):
            supports.append(SupportPatternsSmooth.circle(
                manifold_shape=manifold.shape,
                center=center,
                radius=cfg.agent_radius,
                config=mask_config
            ))
    elif cfg.agent_placement_2d == "random":
        for _ in range(cfg.n_agents):
            cy, cx = rng.uniform(0, H), rng.uniform(0, W)
            radius = (rng.uniform(*cfg.random_radius_range)
                     if cfg.random_radius_range else cfg.agent_radius)
            supports.append(SupportPatternsSmooth.circle(
                manifold_shape=manifold.shape,
                center=(cy, cx),
                radius=radius,
                config=mask_config
            ))
    else:
        raise ValueError(f"Unknown placement: {cfg.agent_placement_2d}")

    return supports


def build_agents(manifold, supports, cfg: SimulationConfig, rng):
    """Create Agent objects with support enforcement."""
    agent_cfg = AgentConfig(
        spatial_shape=cfg.spatial_shape,
        K=cfg.K_latent,
        mu_scale=cfg.mu_scale,
        sigma_scale=cfg.sigma_scale,
        phi_scale=cfg.phi_scale,
        mean_smoothness_scale=cfg.mean_smoothness
    )

    mask_config = MaskConfig(
        mask_type=cfg.mask_type,
        overlap_threshold=cfg.overlap_threshold,
        outside_cov_scale=cfg.outside_cov_scale
    )
    agent_cfg.mask_config = mask_config

    agents = []
    for i in range(cfg.n_agents):
        agent = Agent(agent_id=i, config=agent_cfg, rng=rng, base_manifold=manifold)
        agent.support = supports[i]

        # Re-initialize with support enforcement
        agent._initialize_belief_covariance()
        agent._initialize_prior_covariance()
        agent._initialize_gauge()
        agent.geometry.support = supports[i]
        agent.geometry.n_active = supports[i].n_active

        agents.append(agent)

    return agents


def build_system(agents, cfg: SimulationConfig, rng):
    """Create MultiAgentSystem or MultiScaleSystem."""
    system_cfg = SystemConfig(
        lambda_self=cfg.lambda_self,
        lambda_belief_align=cfg.lambda_belief_align,
        lambda_prior_align=cfg.lambda_prior_align,
        lambda_obs=cfg.lambda_obs,
        lambda_phi=cfg.lambda_phi,
        identical_priors=cfg.identical_priors,
        identical_priors_source=cfg.identical_priors_source,
        kappa_beta=cfg.kappa_beta,
        kappa_gamma=cfg.kappa_gamma,
        overlap_threshold=cfg.overlap_threshold,
        use_connection=cfg.use_connection,
        connection_init_mode=cfg.connection_type,
        D_x=cfg.D_x,
        obs_W_scale=cfg.obs_w_scale,
        obs_R_scale=cfg.obs_r_scale,
        obs_noise_scale=cfg.obs_noise_scale,
        obs_bias_scale=cfg.obs_bias_scale,
        obs_ground_truth_modes=cfg.obs_ground_truth_modes,
        obs_ground_truth_amplitude=cfg.obs_ground_truth_amplitude,
        seed=int(rng.integers(0, 2**31)),
    )

    if cfg.enable_emergence:
        return _build_hierarchical_system(agents, system_cfg, cfg)
    else:
        return _build_standard_system(agents, system_cfg)


def _build_standard_system(agents, system_cfg):
    """Build standard MultiAgentSystem."""
    print("  Mode: STANDARD (no emergence)")
    system = MultiAgentSystem(agents, system_cfg)
    if system.config.has_observations:
        system.ensure_observation_model()
    return system


def _build_hierarchical_system(agents, system_cfg, cfg):
    """Build hierarchical MultiScaleSystem."""
    from meta.emergence import MultiScaleSystem
    from math_utils.generators import generate_so3_generators

    print("  Mode: HIERARCHICAL (emergence enabled)")
    print(f"  Consensus threshold: {cfg.consensus_threshold}")
    print(f"  Max scales: {cfg.max_scale}")

    manifold = agents[0].base_manifold
    system = MultiScaleSystem(
        manifold,
        max_emergence_levels=cfg.max_scale,
        max_meta_membership=cfg.max_meta_membership,
        max_total_agents=cfg.max_total_agents
    )
    system.system_config = system_cfg

    # Add base agents
    generators = generate_so3_generators(cfg.K_latent)
    for agent in agents:
        h_agent = system.add_base_agent(agent.config, agent_id=agent.agent_id)
        h_agent.support = agent.support
        h_agent.generators = generators
        # Copy state
        h_agent.mu_q = agent.mu_q.copy()
        h_agent.Sigma_q = agent.Sigma_q.copy()
        h_agent.mu_p = agent.mu_p.copy()
        h_agent.Sigma_p = agent.Sigma_p.copy()
        if hasattr(agent, 'gauge'):
            h_agent.gauge.phi = agent.gauge.phi.copy()

    # Apply identical priors if configured
    if system_cfg.identical_priors in ("init_copy", "lock"):
        _apply_identical_priors(system, system_cfg)

    return system


def _apply_identical_priors(system, system_cfg):
    """Apply identical priors to base agents."""
    base_agents = system.agents[0]
    if not base_agents:
        return

    # Use Sigma_p (covariance) instead of L_p (Cholesky factor)
    # L_p is a computed property without a setter
    if system_cfg.identical_priors_source == "mean":
        mu_p_shared = sum(a.mu_p for a in base_agents) / len(base_agents)
        Sigma_p_shared = sum(a.Sigma_p for a in base_agents) / len(base_agents)
    else:
        mu_p_shared = base_agents[0].mu_p.copy()
        Sigma_p_shared = base_agents[0].Sigma_p.copy()

    for a in base_agents:
        a.mu_p = mu_p_shared.copy()
        a.Sigma_p = Sigma_p_shared.copy()  # This will invalidate L_p cache
        if hasattr(a, 'invalidate_caches'):
            a.invalidate_caches()


# =============================================================================
# Training (Unified Interface)
# =============================================================================

def run_training(system, cfg: SimulationConfig, output_dir: Path):
    """
    Unified training interface for all system/dynamics combinations.

    Dispatches based on:
    - enable_hamiltonian: Gradient flow vs Hamiltonian dynamics
    - enable_emergence: Flat vs hierarchical system

    Training modes:
    1. Standard gradient flow (default)
    2. Standard Hamiltonian (underdamped dynamics)
    3. Hierarchical gradient flow (with emergence)
    4. Hierarchical Hamiltonian (emergence + underdamped)
    """
    if cfg.enable_hamiltonian:
        if cfg.enable_emergence:
            return _run_hierarchical_hamiltonian_training(system, cfg, output_dir)
        else:
            return _run_hamiltonian_training(system, cfg, output_dir)
    else:
        if cfg.enable_emergence:
            return _run_hierarchical_training(system, cfg, output_dir)
        else:
            return _run_standard_training(system, cfg, output_dir)


def _run_standard_training(system, cfg, output_dir):
    """Run standard training with Trainer."""
    print(f"\n{'='*70}")
    print("STANDARD TRAINING")
    print(f"{'='*70}")

    training_cfg = TrainingConfig(
        n_steps=cfg.n_steps,
        log_every=cfg.log_every,
        lr_mu_q=cfg.lr_mu_q,
        lr_sigma_q=cfg.lr_sigma_q,
        lr_mu_p=cfg.lr_mu_p,
        lr_sigma_p=cfg.lr_sigma_p,
        lr_phi=cfg.lr_phi,
        checkpoint_every=1,
        checkpoint_dir=str(output_dir / "checkpoints"),
    )

    # Initialize geometry tracker if enabled
    geometry_tracker = None
    if cfg.track_pullback_geometry:
        from geometry.geometry_tracker import GeometryTracker

        print("\n  Initializing Pullback Geometry Tracker...")
        print(f"    Track interval: every {cfg.geometry_track_interval} steps")
        print(f"    Consensus metrics: {'ENABLED' if cfg.geometry_enable_consensus else 'DISABLED'}")
        print(f"    Gauge averaging: {'ENABLED' if cfg.geometry_enable_gauge_averaging else 'DISABLED'}")

        # Compute dx from spatial shape
        manifold = system.agents[0].base_manifold
        if manifold.ndim > 0:
            dx = 2 * np.pi / manifold.shape[0]  # Assuming periodic [0, 2π]
        else:
            dx = 1.0  # Point manifold

        geometry_tracker = GeometryTracker(
            agents=system.agents,
            track_interval=cfg.geometry_track_interval,
            dx=dx,
            enable_consensus=cfg.geometry_enable_consensus,
            enable_gauge_averaging=cfg.geometry_enable_gauge_averaging,
            gauge_samples=cfg.geometry_gauge_samples,
            lambda_obs=cfg.geometry_lambda_obs,
            lambda_dark=cfg.geometry_lambda_dark
        )

    trainer = Trainer(system, training_cfg)

    # Training loop with geometry tracking
    if geometry_tracker is not None:
        print("  Training with geometry tracking...")
        # Record initial geometry
        geometry_tracker.record(0, system.agents)

        # Train with manual loop to inject geometry tracking
        for step in range(cfg.n_steps):
            # Single training step
            trainer.step()

            # Record geometry
            if geometry_tracker.should_record(step + 1):
                geometry_tracker.record(step + 1, system.agents)

        history = trainer.history
    else:
        # Standard training without geometry tracking
        history = trainer.train()

    # Save history
    _save_history(history, output_dir)
    _plot_energy(history, output_dir)

    # Save and visualize geometry if tracked
    if geometry_tracker is not None:
        print("\n  Saving pullback geometry data...")
        geometry_tracker.save(output_dir / "geometry_history.pkl")
        geometry_tracker.plot_evolution(output_dir / "geometry_evolution.png")

        if cfg.geometry_enable_consensus:
            geometry_tracker.plot_consensus_evolution(output_dir / "geometry_consensus.png")

        from geometry.geometry_tracker import analyze_final_geometry
        analyze_final_geometry(geometry_tracker.history, save_dir=output_dir / "geometry_analysis")

    return history




def _run_hamiltonian_training(system, cfg, output_dir):
    """
    Enhanced Hamiltonian training with full phase space orbit tracking.
    
    Key additions:
    - PhaseSpaceTracker records (μ, π_μ, Σ_eigenvalues) each step
    - Orbit detection and classification
    - Phase portrait visualization
    """
    from pathlib import Path

    print(f"\n{'='*70}")
    print("HAMILTONIAN DYNAMICS WITH ORBIT TRACKING")
    print(f"{'='*70}")
    print(f"  Integrator: {cfg.hamiltonian_integrator}")
    print(f"  Time step (dt): {cfg.hamiltonian_dt}")
    print(f"  Friction (γ): {cfg.hamiltonian_friction}")
    print(f"  Mass scale: {cfg.hamiltonian_mass_scale}")
    regime = "Conservative (underdamped)" if cfg.hamiltonian_friction < 0.01 else "Damped"
    print(f"  Regime: {regime}")
    print(f"  Phase space tracking: ENABLED")
    
    # Standard training config
    from config import TrainingConfig
    training_cfg = TrainingConfig(
        n_steps=cfg.n_steps,
        log_every=cfg.log_every,
        lr_mu_q=cfg.lr_mu_q,
        lr_sigma_q=cfg.lr_sigma_q,
        lr_mu_p=cfg.lr_mu_p,
        lr_sigma_p=cfg.lr_sigma_p,
        lr_phi=cfg.lr_phi,
        checkpoint_every=1,
        checkpoint_dir=str(output_dir / "checkpoints"),
    )
    
    # Initialize Hamiltonian trainer with built-in phase space tracking
    from agent.hamiltonian_trainer import HamiltonianTrainer
    orbit_dir = output_dir / "orbit_analysis"
    trainer = HamiltonianTrainer(
        system,
        config=training_cfg,
        friction=cfg.hamiltonian_friction,
        mass_scale=cfg.hamiltonian_mass_scale,
        track_phase_space=True,
        phase_space_output_dir=orbit_dir
    )

    # Configure tracker with user settings if available
    if trainer.phase_space_tracker is not None:
        trainer.phase_space_tracker.track_interval = getattr(cfg, 'phase_space_track_interval', 1)
        trainer.phase_space_tracker.recurrence_threshold = 0.1

    print(f"\n  Training loop starting...")

    # Use trainer's train() method - it handles phase space recording automatically
    dt = cfg.hamiltonian_dt
    history = trainer.train(n_steps=cfg.n_steps, dt=dt)

    # Save history and plots
    _save_history(history, output_dir)
    _plot_hamiltonian_energy(history, output_dir)

    # Save phase space tracker for later analysis
    trainer.save_phase_space_tracker(output_dir / "phase_space_tracker.pkl")

    return history




def Q_run_hamiltonian_training(system, cfg, output_dir):
    """
    Run Hamiltonian (underdamped) training for flat systems.

    Uses symplectic integration to preserve phase space structure.
    Energy is approximately conserved (bounded drift).

    Dynamics regimes:
    - friction=0: Conservative (pure Hamiltonian, underdamped)
    - friction>0: Damped (approaches overdamped as friction increases)
    """
    print(f"\n{'='*70}")
    print("HAMILTONIAN DYNAMICS TRAINING")
    print(f"{'='*70}")
    print(f"  Integrator: {cfg.hamiltonian_integrator}")
    print(f"  Time step (dt): {cfg.hamiltonian_dt}")
    print(f"  Friction (γ): {cfg.hamiltonian_friction}")
    print(f"  Mass scale: {cfg.hamiltonian_mass_scale}")
    regime = "Conservative (underdamped)" if cfg.hamiltonian_friction < 0.01 else "Damped"
    print(f"  Regime: {regime}")

    training_cfg = TrainingConfig(
        n_steps=cfg.n_steps,
        log_every=cfg.log_every,
        lr_mu_q=cfg.lr_mu_q,
        lr_sigma_q=cfg.lr_sigma_q,
        lr_mu_p=cfg.lr_mu_p,
        lr_sigma_p=cfg.lr_sigma_p,
        lr_phi=cfg.lr_phi,
        checkpoint_every=1,
        checkpoint_dir=str(output_dir / "checkpoints"),
    )

    # Initialize Hamiltonian trainer
    trainer = HamiltonianTrainer(
        system,
        config=training_cfg,
        friction=cfg.hamiltonian_friction,
        mass_scale=cfg.hamiltonian_mass_scale
    )

    # Initialize geometry tracker if enabled
    geometry_tracker = None
    if cfg.track_pullback_geometry:
        from geometry.geometry_tracker import GeometryTracker

        print("\n  Initializing Pullback Geometry Tracker...")
        print(f"    Track interval: every {cfg.geometry_track_interval} steps")
        print(f"    Consensus metrics: {'ENABLED' if cfg.geometry_enable_consensus else 'DISABLED'}")
        print(f"    Gauge averaging: {'ENABLED' if cfg.geometry_enable_gauge_averaging else 'DISABLED'}")

        # Compute dx from spatial shape
        manifold = system.agents[0].base_manifold
        if manifold.ndim > 0:
            dx = 2 * np.pi / manifold.shape[0]  # Assuming periodic [0, 2π]
        else:
            dx = 1.0  # Point manifold

        geometry_tracker = GeometryTracker(
            agents=system.agents,
            track_interval=cfg.geometry_track_interval,
            dx=dx,
            enable_consensus=cfg.geometry_enable_consensus,
            enable_gauge_averaging=cfg.geometry_enable_gauge_averaging,
            gauge_samples=cfg.geometry_gauge_samples,
            lambda_obs=cfg.geometry_lambda_obs,
            lambda_dark=cfg.geometry_lambda_dark
        )

    # Training loop with geometry tracking
    if geometry_tracker is not None:
        print("  Training with geometry tracking...")
        # Record initial geometry
        geometry_tracker.record(0, system.agents)

        # Train with manual loop to inject geometry tracking
        for step in range(cfg.n_steps):
            # Single Hamiltonian step
            trainer.step(dt=cfg.hamiltonian_dt)

            # Record geometry
            if geometry_tracker.should_record(step + 1):
                geometry_tracker.record(step + 1, system.agents)

        history = trainer.history
    else:
        # Standard Hamiltonian training without geometry tracking
        history = trainer.train(n_steps=cfg.n_steps, dt=cfg.hamiltonian_dt)

    # Save history
    _save_history(history, output_dir)
    _plot_hamiltonian_energy(history, output_dir)

    # Save and visualize geometry if tracked
    if geometry_tracker is not None:
        print("\n  Saving pullback geometry data...")
        geometry_tracker.save(output_dir / "geometry_history.pkl")
        geometry_tracker.plot_evolution(output_dir / "geometry_evolution.png")

        if cfg.geometry_enable_consensus:
            geometry_tracker.plot_consensus_evolution(output_dir / "geometry_consensus.png")

        from geometry.geometry_tracker import analyze_final_geometry
        analyze_final_geometry(geometry_tracker.history, save_dir=output_dir / "geometry_analysis")

    return history


def _run_hierarchical_training(system, cfg, output_dir):
    """Run hierarchical training with emergence."""
    from meta.hierarchical_evolution import HierarchicalEvolutionEngine, HierarchicalConfig
    from meta.gradient_adapter import GradientSystemAdapter
    from gradients.gradient_engine import compute_natural_gradients
    from gradients.free_energy_clean import compute_total_free_energy

    print(f"\n{'='*70}")
    print("HIERARCHICAL TRAINING WITH EMERGENCE")
    print(f"{'='*70}")

    hier_config = HierarchicalConfig(
        enable_top_down_priors=cfg.enable_cross_scale_priors,
        enable_hyperprior_tower=cfg.enable_hyperprior_tower,
        max_hyperprior_depth=cfg.max_hyperprior_depth,
        hyperprior_decay=cfg.hyperprior_decay,
        enable_timescale_filtering=cfg.enable_timescale_sep,
        consensus_check_interval=cfg.consensus_check_interval,
        consensus_kl_threshold=cfg.consensus_threshold,
        min_cluster_size=cfg.min_cluster_size,
        lr_mu_q=cfg.lr_mu_q,
        lr_sigma_q=cfg.lr_sigma_q,
        lr_mu_p=cfg.lr_mu_p,
        lr_sigma_p=cfg.lr_sigma_p,
        lr_phi=cfg.lr_phi
    )

    engine = HierarchicalEvolutionEngine(system, hier_config)

    # Initialize geometry tracker if enabled
    geometry_tracker = None
    if cfg.track_pullback_geometry:
        from geometry.geometry_tracker import GeometryTracker

        print("\n  Initializing Pullback Geometry Tracker...")
        print(f"    Track interval: every {cfg.geometry_track_interval} steps")
        print(f"    Consensus metrics: {'ENABLED' if cfg.geometry_enable_consensus else 'DISABLED'}")
        print(f"    Gauge averaging: {'ENABLED' if cfg.geometry_enable_gauge_averaging else 'DISABLED'}")

        # Get base agents
        base_agents = system.agents[0]  # Scale 0 agents

        # Compute dx from spatial shape
        if len(base_agents) > 0:
            manifold = base_agents[0].base_manifold
            if manifold.ndim > 0:
                dx = 2 * np.pi / manifold.shape[0]  # Assuming periodic [0, 2π]
            else:
                dx = 1.0  # Point manifold
        else:
            dx = 1.0

        geometry_tracker = GeometryTracker(
            agents=base_agents,
            track_interval=cfg.geometry_track_interval,
            dx=dx,
            enable_consensus=cfg.geometry_enable_consensus,
            enable_gauge_averaging=cfg.geometry_enable_gauge_averaging,
            gauge_samples=cfg.geometry_gauge_samples,
            lambda_obs=cfg.geometry_lambda_obs,
            lambda_dark=cfg.geometry_lambda_dark
        )

        # Record initial geometry
        geometry_tracker.record(0, base_agents)

    # Initialize agent field visualizer if enabled
    field_visualizer = None
    visualize_agent_fields = True
    if cfg.visualize_agent_fields:
        from meta.meta_field_visualizer import AgentFieldVisualizer

        print("\n  Initializing Agent Field Visualizer...")
        print(f"    Track interval: every {cfg.viz_track_interval} steps")
        print(f"    Scales to image: {cfg.viz_scales}")
        print(f"    Fields to track: {cfg.viz_fields}")

        field_visualizer = AgentFieldVisualizer(
            output_dir=output_dir / "agent_fields",
            scales_to_track=list(cfg.viz_scales),
            fields_to_track=list(cfg.viz_fields),
            latent_components=list(cfg.viz_latent_components) if cfg.viz_latent_components else None,
            track_interval=cfg.viz_track_interval
        )

        # Record initial state
        field_visualizer.record(0, system)

    # Initialize comprehensive visualization tools
    analyzer = None
    diagnostics = None
    if cfg.generate_meta_visualizations:
        from meta.visualization import MetaAgentAnalyzer
        from meta.participatory_diagnostics import ParticipatoryDiagnostics

        print("  Initializing comprehensive visualization tools...")
        analyzer = MetaAgentAnalyzer(system)

        # Note: compute_full_energies=True enables belief/prior alignment energy tracking
        # This is VERY EXPENSIVE (10-100x slower) but gives detailed energy decomposition
        # Set to False for fast performance with only self-energy tracking
        diagnostics = ParticipatoryDiagnostics(
            system=system,
            track_agent_ids=None,  # Auto-selects first 3 scale-0 agents
            compute_full_energies=COMP_FULL_ENERGY  # Set True for detailed energy (SLOW!)
        )
        print(f"  Snapshot interval: every {cfg.snapshot_interval} steps")
        print(f"  Full energy computation: {'ENABLED (slow)' if diagnostics.compute_full_energies else 'DISABLED (fast, self-energy only)'}")

    # History tracking
    history = {
        'step': [],
        'total': [],
        'n_scales': [],
        'n_active_agents': [],
        'n_condensations': [],
        'emergence_events': []
    }

    # Training loop
    for step in range(cfg.n_steps):
        active_agents = system.get_all_active_agents()
        if not active_agents:
            break

        # Create adapter for gradient computation
        adapter = GradientSystemAdapter(active_agents, system.system_config)

        # Compute energy before updates
        energies = compute_total_free_energy(adapter)

        # Gradient computation wrapper
        def compute_grads(sys):
            return compute_natural_gradients(adapter)

        # Evolve one step
        metrics = engine.evolve_step(learning_rate=cfg.lr_mu_q, compute_gradients_fn=compute_grads)

        # Record geometry if tracking enabled
        if geometry_tracker is not None and geometry_tracker.should_record(step + 1):
            base_agents = system.agents[0]  # Track base agents only
            geometry_tracker.record(step + 1, base_agents)

        # Record agent fields if tracking enabled
        if field_visualizer is not None and field_visualizer.should_record(step + 1):
            field_visualizer.record(step + 1, system)

        # Capture visualization snapshots
        if cfg.generate_meta_visualizations:
            if step % cfg.snapshot_interval == 0 or step == cfg.n_steps - 1:
                analyzer.capture_snapshot()
            diagnostics.record_snapshot(step)

        # Record metrics - use ACTUAL system state, not potentially stale metrics
        actual_n_scales = len(system.agents)  # Number of scales = number of keys in agents dict
        actual_n_active = sum(sum(1 for a in agents if a.is_active)
                             for agents in system.agents.values())

        history['step'].append(step)
        history['total'].append(energies.total)
        history['n_scales'].append(actual_n_scales)
        history['n_active_agents'].append(actual_n_active)
        history['n_condensations'].append(metrics.get('n_condensations', 0))

        # Check early stopping conditions
        stop_reason = None
        if cfg.stop_if_n_scales_reached and actual_n_scales >= cfg.stop_if_n_scales_reached:
            stop_reason = f"Reached target scale count: {actual_n_scales}/{cfg.stop_if_n_scales_reached}"
        elif cfg.stop_if_n_condensations:
            total_condensations = sum(history['n_condensations'])
            if total_condensations >= cfg.stop_if_n_condensations:
                stop_reason = f"Reached target condensation count: {total_condensations}/{cfg.stop_if_n_condensations}"
        elif cfg.stop_if_max_active_agents and actual_n_active >= cfg.stop_if_max_active_agents:
            stop_reason = f"Reached maximum active agents: {actual_n_active}/{cfg.stop_if_max_active_agents}"
        elif cfg.stop_if_min_active_agents and actual_n_active < cfg.stop_if_min_active_agents:
            stop_reason = f"Active agents below minimum: {actual_n_active}/{cfg.stop_if_min_active_agents}"

        if stop_reason:
            print(f"\n⏹️  Early stop at step {step}: {stop_reason}")
            break

        # Log emergence events
        if metrics.get('n_condensations', 0) > 0:
            event = {
                'step': step,
                'n_condensations': metrics['n_condensations'],
                'n_scales': len(metrics.get('n_active', {}))
            }
            history['emergence_events'].append(event)
            print(f"\n🌟 EMERGENCE at step {step}! {metrics['n_condensations']} new meta-agents")

        if step % cfg.log_every == 0:
            print(f"Step {step:4d} | Energy: {energies.total:.4f} | "
                  f"Scales: {history['n_scales'][-1]} | Active: {history['n_active_agents'][-1]}")

    # Save history
    _save_history(history, output_dir)

    # Save and visualize geometry if tracked
    if geometry_tracker is not None:
        print("\n  Saving pullback geometry data...")
        geometry_tracker.save(output_dir / "geometry_history.pkl")
        geometry_tracker.plot_evolution(output_dir / "geometry_evolution.png")

        if cfg.geometry_enable_consensus:
            geometry_tracker.plot_consensus_evolution(output_dir / "geometry_consensus.png")

        from geometry.geometry_tracker import analyze_final_geometry
        analyze_final_geometry(geometry_tracker.history, save_dir=output_dir / "geometry_analysis")

    # Generate agent field visualizations if tracked
    if field_visualizer is not None:
        print("\n  Generating agent field visualizations...")
        field_visualizer.generate_summary_report()

    # Generate visualizations
    if cfg.generate_meta_visualizations and analyzer and diagnostics:
        _generate_comprehensive_visualizations(system, analyzer, diagnostics, output_dir)
    else:
        # Minimal visualization
        _plot_emergence(history, output_dir)

    return history


def _run_hierarchical_hamiltonian_training(system, cfg, output_dir):
    """
    Run Hamiltonian dynamics with hierarchical emergence.

    This is the most sophisticated training mode, combining:
    - Symplectic integration (energy-preserving)
    - Multi-scale hierarchy (meta-agent emergence)
    - Cross-scale prior propagation (Ouroboros tower)
    - Full gauge-invariant field theory (when hamiltonian_include_gauge=True)

    Key insight: Each agent at each scale has its own phase space:
    - (μ, π_μ): Mean dynamics with Fisher-Rao metric
    - (φ, π_φ): Gauge dynamics with Killing form metric (when enabled)

    The complete Hamiltonian H = T_μ + T_φ + V is gauge-invariant when
    all components transform properly under gauge transformations.

    EXPERIMENTAL: This mode is under active development.
    """
    from meta.hierarchical_evolution import HierarchicalEvolutionEngine, HierarchicalConfig
    from meta.gradient_adapter import GradientSystemAdapter
    from gradients.gradient_engine import compute_natural_gradients
    from gradients.free_energy_clean import compute_total_free_energy

    # Import gauge-invariant components if enabled
    include_gauge = cfg.hamiltonian_include_gauge
    if include_gauge:
        from geometry.lie_algebra import LieAlgebra, LieGroup

    print(f"\n{'='*70}")
    print("HIERARCHICAL HAMILTONIAN DYNAMICS")
    print(f"{'='*70}")
    print(f"  Integrator: {cfg.hamiltonian_integrator}")
    print(f"  Time step (dt): {cfg.hamiltonian_dt}")
    print(f"  Friction (γ): {cfg.hamiltonian_friction}")
    print(f"  Mass scale: {cfg.hamiltonian_mass_scale}")
    print(f"  Emergence: ENABLED")
    print(f"  Ouroboros Tower: {'ENABLED' if cfg.enable_hyperprior_tower else 'DISABLED'}")
    print(f"  Gauge Field (φ) Dynamics: {'ENABLED (gauge-invariant)' if include_gauge else 'DISABLED'}")
    regime = "Conservative (underdamped)" if cfg.hamiltonian_friction < 0.01 else "Damped"
    print(f"  Regime: {regime}")

    hier_config = HierarchicalConfig(
        enable_top_down_priors=cfg.enable_cross_scale_priors,
        enable_hyperprior_tower=cfg.enable_hyperprior_tower,
        max_hyperprior_depth=cfg.max_hyperprior_depth,
        hyperprior_decay=cfg.hyperprior_decay,
        enable_timescale_filtering=cfg.enable_timescale_sep,
        consensus_check_interval=cfg.consensus_check_interval,
        consensus_kl_threshold=cfg.consensus_threshold,
        min_cluster_size=cfg.min_cluster_size,
        lr_mu_q=cfg.lr_mu_q,
        lr_sigma_q=cfg.lr_sigma_q,
        lr_mu_p=cfg.lr_mu_p,
        lr_sigma_p=cfg.lr_sigma_p,
        lr_phi=cfg.lr_phi
    )

    engine = HierarchicalEvolutionEngine(system, hier_config)

    # Initialize Lie algebra for gauge dynamics
    gauge_algebra = None
    if include_gauge:
        gauge_algebra = LieAlgebra(LieGroup.SO3)  # SO(3) gauge group

    # Initialize phase space for all agents (θ, p)
    # Each agent gets its own momentum initialized to zero
    agent_momenta = {}  # Dict[scale][agent_id] -> {'mu': π_μ, 'phi': π_φ}
    for scale, agents in system.agents.items():
        agent_momenta[scale] = {}
        for agent in agents:
            # Initialize momenta to zero (start from rest)
            momenta = {
                'mu': np.zeros(agent.mu_q.size),  # π_μ (mean momentum)
            }
            # Add gauge momentum if enabled
            if include_gauge and hasattr(agent, 'gauge') and hasattr(agent.gauge, 'phi'):
                momenta['phi'] = np.zeros(agent.gauge.phi.size)  # π_φ (gauge momentum)

            agent_momenta[scale][agent.agent_id] = momenta

    # Initialize geometry tracker if enabled
    geometry_tracker = None
    if cfg.track_pullback_geometry:
        from geometry.geometry_tracker import GeometryTracker

        print("\n  Initializing Pullback Geometry Tracker...")
        base_agents = system.agents[0]  # Scale 0 agents

        if len(base_agents) > 0:
            manifold = base_agents[0].base_manifold
            if manifold.ndim > 0:
                dx = 2 * np.pi / manifold.shape[0]
            else:
                dx = 1.0
        else:
            dx = 1.0

        geometry_tracker = GeometryTracker(
            agents=base_agents,
            track_interval=cfg.geometry_track_interval,
            dx=dx,
            enable_consensus=cfg.geometry_enable_consensus,
            enable_gauge_averaging=cfg.geometry_enable_gauge_averaging,
            gauge_samples=cfg.geometry_gauge_samples,
            lambda_obs=cfg.geometry_lambda_obs,
            lambda_dark=cfg.geometry_lambda_dark
        )
        geometry_tracker.record(0, base_agents)

    # Initialize visualizers
    analyzer = None
    diagnostics = None
    if cfg.generate_meta_visualizations:
        from meta.visualization import MetaAgentAnalyzer
        from meta.participatory_diagnostics import ParticipatoryDiagnostics

        print("  Initializing comprehensive visualization tools...")
        analyzer = MetaAgentAnalyzer(system)
        diagnostics = ParticipatoryDiagnostics(
            system=system,
            track_agent_ids=None,
            compute_full_energies=True
        )

    # History tracking (extended for Hamiltonian with gauge decomposition)
    history = {
        'step': [],
        'total': [],
        'kinetic_energy': [],       # Total T = T_mu + T_phi
        'kinetic_energy_mu': [],    # T_μ: Mean dynamics kinetic energy
        'kinetic_energy_phi': [],   # T_φ: Gauge dynamics kinetic energy
        'potential_energy': [],
        'total_hamiltonian': [],
        'energy_drift': [],
        'n_scales': [],
        'n_active_agents': [],
        'n_condensations': [],
        'emergence_events': []
    }

    dt = cfg.hamiltonian_dt
    friction = cfg.hamiltonian_friction
    mass_scale = cfg.hamiltonian_mass_scale
    initial_H = None

    # Training loop
    for step in range(cfg.n_steps):
        active_agents = system.get_all_active_agents()
        if not active_agents:
            break

        # Create adapter for energy/gradient computation
        adapter = GradientSystemAdapter(active_agents, system.system_config)

        # Compute energy (potential V)
        energies = compute_total_free_energy(adapter)
        V = energies.total

        # Compute gradients (force = -∇V)
        agent_grads = compute_natural_gradients(adapter)

        # Leapfrog integration for each agent across all scales
        # Tracks kinetic energy decomposition: T = T_μ + T_φ
        total_kinetic_mu = 0.0   # Kinetic energy from mean dynamics
        total_kinetic_phi = 0.0  # Kinetic energy from gauge dynamics

        for agent, grads in zip(active_agents, agent_grads):
            scale = agent.scale if hasattr(agent, 'scale') else 0
            aid = agent.agent_id
            K = agent.config.K

            # Ensure momentum dict exists for this agent
            if scale not in agent_momenta:
                agent_momenta[scale] = {}
            if aid not in agent_momenta[scale]:
                # Initialize momentum dict
                momenta = {'mu': np.zeros(agent.mu_q.size)}
                if include_gauge and hasattr(agent, 'gauge') and hasattr(agent.gauge, 'phi'):
                    momenta['phi'] = np.zeros(agent.gauge.phi.size)
                agent_momenta[scale][aid] = momenta

            momenta = agent_momenta[scale][aid]
            p_mu = momenta['mu']

            # Get gradient for mu_q (full gradient, same shape as mu_q)
            grad_mu = grads.grad_mu_q.flatten()

            # ===== MEAN DYNAMICS (μ, π_μ) =====
            # Half-step momentum update: π_μ = π_μ - (dt/2) * ∂V/∂μ
            p_mu = p_mu - 0.5 * dt * grad_mu

            # Full-step position update: μ = μ + dt * M_μ^{-1} * π_μ
            # Use prior covariance as inverse mass (Σ_p ≈ M_μ^{-1})
            if agent.mu_q.ndim == 1:
                # 0D: Single Gaussian, Sigma_p is (K, K)
                M_inv = agent.Sigma_p  # Sigma_p as inverse mass
                velocity_mu = (M_inv @ p_mu) / mass_scale
            elif agent.mu_q.ndim == 2:
                # 1D field: Apply M^{-1} per spatial point
                n_spatial = agent.mu_q.shape[0]
                velocity_mu = np.zeros_like(p_mu)
                p_reshaped = p_mu.reshape(n_spatial, K)
                for i in range(n_spatial):
                    M_inv_i = agent.Sigma_p[i] if agent.Sigma_p.ndim == 3 else agent.Sigma_p
                    velocity_mu[i*K:(i+1)*K] = (M_inv_i @ p_reshaped[i]) / mass_scale
            else:
                # 2D+ field: Simplified - use identity
                velocity_mu = p_mu / mass_scale

            agent.mu_q = agent.mu_q + dt * velocity_mu.reshape(agent.mu_q.shape)

            # ===== GAUGE DYNAMICS (φ, π_φ) =====
            # Only evolve gauge if enabled and agent has gauge field
            if include_gauge and 'phi' in momenta and hasattr(agent, 'gauge'):
                p_phi = momenta['phi']
                phi = agent.gauge.phi

                # Compute gauge gradient: ∂V/∂φ
                # Use finite differences on the transported KL divergence
                eps = 1e-5
                grad_phi = np.zeros_like(phi)
                for i in range(len(phi)):
                    phi_plus = phi.copy()
                    phi_plus[i] += eps
                    phi_minus = phi.copy()
                    phi_minus[i] -= eps

                    # Temporarily perturb phi and compute energy difference
                    old_phi = agent.gauge.phi.copy()
                    agent.gauge.phi = phi_plus
                    adapter_plus = GradientSystemAdapter(active_agents, system.system_config)
                    E_plus = compute_total_free_energy(adapter_plus).total

                    agent.gauge.phi = phi_minus
                    adapter_minus = GradientSystemAdapter(active_agents, system.system_config)
                    E_minus = compute_total_free_energy(adapter_minus).total

                    agent.gauge.phi = old_phi  # Restore
                    grad_phi[i] = (E_plus - E_minus) / (2 * eps)

                # Half-step gauge momentum: π_φ = π_φ - (dt/2) * ∂V/∂φ
                p_phi = p_phi - 0.5 * dt * grad_phi

                # Full-step gauge position: φ = φ + dt * M_φ^{-1} * π_φ
                # For SO(3), the metric is the Killing form (identity in standard basis)
                velocity_phi = p_phi / mass_scale
                agent.gauge.phi = phi + dt * velocity_phi

                # Store updated gauge momentum (will be updated again after final half-step)
                momenta['phi'] = p_phi

            # Recompute gradient after position update
            adapter_new = GradientSystemAdapter(active_agents, system.system_config)
            agent_grads_new = compute_natural_gradients(adapter_new)

            # Find updated gradient for this agent
            grad_mu_new = grad_mu  # Default to old
            for a, g in zip(active_agents, agent_grads_new):
                if a.agent_id == aid:
                    grad_mu_new = g.grad_mu_q.flatten()
                    break

            # Half-step mean momentum update: π_μ = π_μ - (dt/2) * ∂V/∂μ_new
            p_mu = p_mu - 0.5 * dt * grad_mu_new

            # Final half-step for gauge momentum (if enabled)
            if include_gauge and 'phi' in momenta and hasattr(agent, 'gauge'):
                p_phi = momenta['phi']
                phi = agent.gauge.phi

                # Recompute gauge gradient at new position
                eps = 1e-5
                grad_phi_new = np.zeros_like(phi)
                for i in range(len(phi)):
                    phi_plus = phi.copy()
                    phi_plus[i] += eps
                    phi_minus = phi.copy()
                    phi_minus[i] -= eps

                    old_phi = agent.gauge.phi.copy()
                    agent.gauge.phi = phi_plus
                    adapter_plus = GradientSystemAdapter(active_agents, system.system_config)
                    E_plus = compute_total_free_energy(adapter_plus).total

                    agent.gauge.phi = phi_minus
                    adapter_minus = GradientSystemAdapter(active_agents, system.system_config)
                    E_minus = compute_total_free_energy(adapter_minus).total

                    agent.gauge.phi = old_phi
                    grad_phi_new[i] = (E_plus - E_minus) / (2 * eps)

                # Half-step gauge momentum: π_φ = π_φ - (dt/2) * ∂V/∂φ_new
                p_phi = p_phi - 0.5 * dt * grad_phi_new

                # Apply friction to gauge momentum
                if friction > 0:
                    p_phi = p_phi * np.exp(-friction * dt)

                momenta['phi'] = p_phi

                # Gauge kinetic energy: T_φ = (1/2) ⟨π_φ, π_φ⟩_𝔤
                # For SO(3), use Killing form (identity in standard basis)
                if gauge_algebra is not None:
                    T_phi_agent = gauge_algebra.kinetic_energy(p_phi)
                else:
                    T_phi_agent = 0.5 * np.dot(p_phi, p_phi) / mass_scale
                total_kinetic_phi += T_phi_agent

            # Apply friction to mean momentum
            if friction > 0:
                p_mu = p_mu * np.exp(-friction * dt)

            # Store updated mean momentum
            momenta['mu'] = p_mu
            agent_momenta[scale][aid] = momenta

            # Mean kinetic energy: T_μ = (1/2) π_μ^T M_μ^{-1} π_μ
            if agent.mu_q.ndim == 1:
                # 0D: T_μ = (1/2) π_μ^T Σ_p π_μ
                T_mu_agent = 0.5 * np.dot(p_mu, agent.Sigma_p @ p_mu) / mass_scale
            elif agent.mu_q.ndim == 2:
                # 1D: Sum over spatial points
                n_spatial = agent.mu_q.shape[0]
                p_reshaped = p_mu.reshape(n_spatial, K)
                T_mu_agent = 0.0
                for i in range(n_spatial):
                    M_inv_i = agent.Sigma_p[i] if agent.Sigma_p.ndim == 3 else agent.Sigma_p
                    T_mu_agent += 0.5 * np.dot(p_reshaped[i], M_inv_i @ p_reshaped[i]) / mass_scale
            else:
                # 2D+: Simplified
                T_mu_agent = 0.5 * np.dot(p_mu, p_mu) / mass_scale
            total_kinetic_mu += T_mu_agent

        # Total kinetic energy
        total_kinetic = total_kinetic_mu + total_kinetic_phi

        # Total Hamiltonian
        H = total_kinetic + V

        if initial_H is None:
            initial_H = H

        energy_drift = abs(H - initial_H) / (abs(initial_H) + 1e-10)

        # Check for consensus and form meta-agents (periodically)
        metrics = {'n_condensations': 0}
        if step % hier_config.consensus_check_interval == 0:
            new_condensations = engine._check_and_condense_all_scales()
            metrics['n_condensations'] = len(new_condensations)

        # Update priors from parents (top-down) via system method
        if hier_config.enable_top_down_priors:
            system.update_cross_scale_priors(
                enable_tower=hier_config.enable_hyperprior_tower,
                max_depth=hier_config.max_hyperprior_depth,
                decay=hier_config.hyperprior_decay
            )

        # Record geometry if tracking enabled
        if geometry_tracker is not None and geometry_tracker.should_record(step + 1):
            base_agents = system.agents[0]
            geometry_tracker.record(step + 1, base_agents)

        # Capture visualization snapshots
        if cfg.generate_meta_visualizations:
            if step % cfg.snapshot_interval == 0 or step == cfg.n_steps - 1:
                analyzer.capture_snapshot()
            diagnostics.record_snapshot(step)

        # Record metrics
        actual_n_scales = len(system.agents)
        actual_n_active = sum(sum(1 for a in agents if a.is_active)
                             for agents in system.agents.values())

        history['step'].append(step)
        history['total'].append(energies.total)
        history['kinetic_energy'].append(total_kinetic)
        history['kinetic_energy_mu'].append(total_kinetic_mu)
        history['kinetic_energy_phi'].append(total_kinetic_phi)
        history['potential_energy'].append(V)
        history['total_hamiltonian'].append(H)
        history['energy_drift'].append(energy_drift)
        history['n_scales'].append(actual_n_scales)
        history['n_active_agents'].append(actual_n_active)
        history['n_condensations'].append(metrics.get('n_condensations', 0))

        # Check early stopping
        stop_reason = None
        if cfg.stop_if_n_scales_reached and actual_n_scales >= cfg.stop_if_n_scales_reached:
            stop_reason = f"Reached target scale count: {actual_n_scales}/{cfg.stop_if_n_scales_reached}"
        elif cfg.stop_if_max_active_agents and actual_n_active >= cfg.stop_if_max_active_agents:
            stop_reason = f"Reached maximum active agents: {actual_n_active}/{cfg.stop_if_max_active_agents}"

        if stop_reason:
            print(f"\n  Early stop at step {step}: {stop_reason}")
            break

        # Log emergence events
        if metrics.get('n_condensations', 0) > 0:
            event = {
                'step': step,
                'n_condensations': metrics['n_condensations'],
                'n_scales': actual_n_scales
            }
            history['emergence_events'].append(event)
            print(f"\n  EMERGENCE at step {step}! {metrics['n_condensations']} new meta-agents")

        if step % cfg.log_every == 0:
            print(f"Step {step:4d} | H: {H:.4f} [T={total_kinetic:.3f}, V={V:.3f}] | "
                  f"drift={energy_drift:.2e} | Scales: {actual_n_scales} | Active: {actual_n_active}")

    # Save history
    _save_history(history, output_dir)
    _plot_hierarchical_hamiltonian(history, output_dir)

    # Save geometry
    if geometry_tracker is not None:
        print("\n  Saving pullback geometry data...")
        geometry_tracker.save(output_dir / "geometry_history.pkl")
        geometry_tracker.plot_evolution(output_dir / "geometry_evolution.png")

        if cfg.geometry_enable_consensus:
            geometry_tracker.plot_consensus_evolution(output_dir / "geometry_consensus.png")

        from geometry.geometry_tracker import analyze_final_geometry
        analyze_final_geometry(geometry_tracker.history, save_dir=output_dir / "geometry_analysis")

    # Generate visualizations
    if cfg.generate_meta_visualizations and analyzer and diagnostics:
        _generate_comprehensive_visualizations(system, analyzer, diagnostics, output_dir)
    else:
        _plot_emergence(history, output_dir)

    return history


def _plot_hierarchical_hamiltonian(history, output_dir):
    """
    Plot Hamiltonian + emergence evolution with gauge-invariant decomposition.

    Creates a 2x3 figure showing:
    - Hamiltonian components (T, V, H)
    - Kinetic energy decomposition (T_μ, T_φ)
    - Energy drift
    - Hierarchy growth
    - Emergence events
    - Phase space dynamics
    """
    # Check if we have gauge decomposition data
    has_gauge = 'kinetic_energy_phi' in history and any(
        x > 0 for x in history.get('kinetic_energy_phi', [0])
    )

    if has_gauge:
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    else:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        # Flatten for consistent indexing
        axes = np.array([[axes[0, 0], axes[0, 1], None],
                         [axes[1, 0], axes[1, 1], None]])

    steps = history['step']

    # 1. Hamiltonian components
    ax1 = axes[0, 0]
    ax1.plot(steps, history['total_hamiltonian'], 'k-', linewidth=2, label='H = T + V')
    ax1.plot(steps, history['kinetic_energy'], 'b--', linewidth=1.5, label='T (kinetic)')
    ax1.plot(steps, history['potential_energy'], 'r--', linewidth=1.5, label='V (potential)')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Energy')
    ax1.set_title('Hamiltonian Components')
    ax1.legend()
    ax1.grid(alpha=0.3)

    # 2. Energy drift
    ax2 = axes[0, 1]
    ax2.semilogy(steps, np.array(history['energy_drift']) + 1e-12, 'g-', linewidth=2)
    ax2.axhline(y=0.01, color='orange', linestyle='--', alpha=0.7, label='1% threshold')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('|H(t) - H(0)|/H(0)')
    ax2.set_title('Energy Conservation')
    ax2.legend()
    ax2.grid(alpha=0.3)

    # 3. Kinetic energy decomposition (only if gauge enabled)
    if has_gauge and axes[0, 2] is not None:
        ax3 = axes[0, 2]
        ax3.plot(steps, history['kinetic_energy_mu'], 'b-', linewidth=2, label=r'$T_\mu$ (mean)')
        ax3.plot(steps, history['kinetic_energy_phi'], 'm-', linewidth=2, label=r'$T_\phi$ (gauge)')
        ax3.plot(steps, history['kinetic_energy'], 'k--', linewidth=1.5, alpha=0.5, label='Total T')
        ax3.set_xlabel('Step')
        ax3.set_ylabel('Kinetic Energy')
        ax3.set_title(r'Kinetic Decomposition: $T = T_\mu + T_\phi$')
        ax3.legend()
        ax3.grid(alpha=0.3)

    # 4. Hierarchy evolution
    ax4 = axes[1, 0]
    ax4.plot(steps, history['n_scales'], 'g-', marker='o', markersize=3, label='# Scales')
    ax4.plot(steps, history['n_active_agents'], 'b-', marker='s', markersize=3, label='# Active Agents')
    ax4.set_xlabel('Step')
    ax4.set_ylabel('Count')
    ax4.set_title('Hierarchical Structure')
    ax4.legend()
    ax4.grid(alpha=0.3)

    # 5. Energy with emergence markers
    ax5 = axes[1, 1]
    ax5.plot(steps, history['total'], 'b-', linewidth=2, label='Free Energy')
    for event in history['emergence_events']:
        ax5.axvline(event['step'], color='red', alpha=0.3, linestyle='--')
    ax5.set_xlabel('Step')
    ax5.set_ylabel('Free Energy')
    ax5.set_title('Free Energy (red = emergence)')
    ax5.legend()
    ax5.grid(alpha=0.3)

    # 6. Gauge fraction (only if gauge enabled)
    if has_gauge and axes[1, 2] is not None:
        ax6 = axes[1, 2]
        total_T = np.array(history['kinetic_energy']) + 1e-10
        T_phi = np.array(history['kinetic_energy_phi'])
        gauge_fraction = T_phi / total_T
        ax6.plot(steps, gauge_fraction, 'm-', linewidth=2)
        ax6.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        ax6.set_xlabel('Step')
        ax6.set_ylabel(r'$T_\phi / T$')
        ax6.set_title('Gauge Kinetic Fraction')
        ax6.set_ylim(0, 1)
        ax6.grid(alpha=0.3)

    plt.tight_layout()

    fig_path = output_dir / "hierarchical_hamiltonian_evolution.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"✓ Saved {fig_path}")

    # Additional gauge-specific plot if enabled
    if has_gauge:
        _plot_gauge_dynamics(history, output_dir)


def _plot_gauge_dynamics(history, output_dir):
    """
    Detailed plot of gauge field dynamics.

    Shows the interplay between mean (μ) and gauge (φ) degrees of freedom
    in the full gauge-invariant Hamiltonian: H = T_μ + T_φ + V.

    Key insight: Energy can flow between mean and gauge sectors while
    the total Hamiltonian is approximately conserved.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    steps = history['step']

    T_mu = np.array(history['kinetic_energy_mu'])
    T_phi = np.array(history['kinetic_energy_phi'])
    T_total = T_mu + T_phi + 1e-10
    V = np.array(history['potential_energy'])
    H = np.array(history['total_hamiltonian'])

    # 1. Energy partition (stacked area)
    ax1 = axes[0, 0]
    ax1.fill_between(steps, 0, T_mu, alpha=0.7, color='blue', label=r'$T_\mu$ (mean)')
    ax1.fill_between(steps, T_mu, T_mu + T_phi, alpha=0.7, color='magenta', label=r'$T_\phi$ (gauge)')
    ax1.fill_between(steps, T_mu + T_phi, T_mu + T_phi + V, alpha=0.5, color='red', label='V (potential)')
    ax1.plot(steps, H, 'k--', linewidth=2, label='H (total)')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Energy')
    ax1.set_title('Energy Partition: H = T_μ + T_φ + V')
    ax1.legend(loc='upper right')
    ax1.grid(alpha=0.3)

    # 2. Kinetic energy ratio (reveals energy exchange dynamics)
    ax2 = axes[0, 1]
    ratio = T_phi / T_total
    ax2.plot(steps, ratio, 'm-', linewidth=2)
    ax2.fill_between(steps, 0, ratio, alpha=0.3, color='magenta')
    ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Equipartition')
    ax2.set_xlabel('Step')
    ax2.set_ylabel(r'$T_\phi / (T_\mu + T_\phi)$')
    ax2.set_title('Gauge Kinetic Fraction')
    ax2.set_ylim(0, 1)
    ax2.legend()
    ax2.grid(alpha=0.3)

    # 3. Energy flow rate (time derivative of kinetic energies)
    ax3 = axes[1, 0]
    if len(steps) > 1:
        dT_mu = np.diff(T_mu)
        dT_phi = np.diff(T_phi)
        dV = np.diff(V)
        ax3.plot(steps[1:], dT_mu, 'b-', linewidth=1.5, alpha=0.8, label=r'$dT_\mu/dt$')
        ax3.plot(steps[1:], dT_phi, 'm-', linewidth=1.5, alpha=0.8, label=r'$dT_\phi/dt$')
        ax3.plot(steps[1:], dV, 'r-', linewidth=1.5, alpha=0.8, label=r'$dV/dt$')
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax3.set_xlabel('Step')
        ax3.set_ylabel('Energy Flow Rate')
        ax3.set_title('Energy Exchange Between Sectors')
        ax3.legend()
        ax3.grid(alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'Insufficient data', ha='center', va='center', transform=ax3.transAxes)

    # 4. Virial-like ratio: 2T vs V (for insight into dynamics)
    ax4 = axes[1, 1]
    virial = 2 * T_total / (np.abs(V) + 1e-10)
    ax4.semilogy(steps, virial, 'g-', linewidth=2)
    ax4.axhline(y=1.0, color='orange', linestyle='--', alpha=0.7, label='Virial equilibrium')
    ax4.set_xlabel('Step')
    ax4.set_ylabel('2T/|V|')
    ax4.set_title('Virial Ratio (log scale)')
    ax4.legend()
    ax4.grid(alpha=0.3)

    plt.tight_layout()

    fig_path = output_dir / "gauge_dynamics.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"✓ Saved {fig_path}")


# =============================================================================
# Saving and Visualization
# =============================================================================

def _save_history(history, output_dir):
    """Save training history."""
    hist_path = output_dir / "history.pkl"
    with open(hist_path, "wb") as f:
        pickle.dump(history, f)
    print(f"✓ Saved {hist_path}")


def _plot_energy(history, output_dir):
    """Plot energy evolution for standard training."""
    if isinstance(history, TrainingHistory):
        steps, energy = history.steps, history.total_energy
    else:
        steps, energy = history['step'], history['total']

    plt.figure(figsize=(10, 6))
    plt.plot(steps, energy, linewidth=2, color='black')
    plt.xlabel('Step')
    plt.ylabel('Energy')
    plt.title('Energy Evolution')
    plt.grid(alpha=0.3)
    plt.tight_layout()

    fig_path = output_dir / "energy_evolution.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"✓ Saved {fig_path}")


def _plot_hamiltonian_energy(history, output_dir):
    """
    Plot Hamiltonian energy evolution with phase space diagnostics.

    Creates a 2x2 figure showing:
    - Total Hamiltonian H = T + V
    - Kinetic (T) and Potential (V) separately
    - Energy drift |H(t) - H(0)|/H(0)
    - Phase space norms (momentum & velocity)
    """
    if not isinstance(history, HamiltonianHistory):
        # Fallback to standard plot
        _plot_energy(history, output_dir)
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    steps = history.steps

    # 1. Total Hamiltonian
    ax1 = axes[0, 0]
    ax1.plot(steps, history.total_hamiltonian, 'k-', linewidth=2, label='H = T + V')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Hamiltonian')
    ax1.set_title('Total Hamiltonian (should be conserved)')
    ax1.legend()
    ax1.grid(alpha=0.3)

    # 2. Kinetic and Potential energy
    ax2 = axes[0, 1]
    ax2.plot(steps, history.kinetic_energy, 'b-', linewidth=2, label='T (kinetic)')
    ax2.plot(steps, history.potential_energy, 'r-', linewidth=2, label='V (potential)')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Energy')
    ax2.set_title('Energy Components')
    ax2.legend()
    ax2.grid(alpha=0.3)

    # 3. Energy drift
    ax3 = axes[1, 0]
    ax3.semilogy(steps, np.array(history.energy_drift) + 1e-12, 'g-', linewidth=2)
    ax3.axhline(y=0.01, color='orange', linestyle='--', alpha=0.7, label='1% threshold')
    ax3.axhline(y=0.001, color='red', linestyle='--', alpha=0.7, label='0.1% threshold')
    ax3.set_xlabel('Step')
    ax3.set_ylabel('|H(t) - H(0)|/H(0)')
    ax3.set_title('Energy Drift (symplectic should stay bounded)')
    ax3.legend()
    ax3.grid(alpha=0.3)

    # 4. Phase space norms
    ax4 = axes[1, 1]
    ax4.plot(steps, history.momentum_norms, 'purple', linewidth=2, label='||p|| (momentum)')
    ax4.plot(steps, history.velocity_norms, 'orange', linewidth=2, label='||θ̇|| (velocity)')
    ax4.set_xlabel('Step')
    ax4.set_ylabel('Norm')
    ax4.set_title('Phase Space Dynamics')
    ax4.legend()
    ax4.grid(alpha=0.3)

    plt.tight_layout()

    fig_path = output_dir / "hamiltonian_evolution.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"✓ Saved {fig_path}")

    # Also save standard energy plot for comparison
    plt.figure(figsize=(10, 6))
    plt.plot(steps, history.total_energy, 'b-', linewidth=2, label='Free Energy F')
    plt.plot(steps, history.total_hamiltonian, 'k--', linewidth=2, label='Hamiltonian H')
    plt.xlabel('Step')
    plt.ylabel('Energy')
    plt.title('Free Energy vs Hamiltonian')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    fig_path = output_dir / "energy_evolution.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"✓ Saved {fig_path}")


def _plot_emergence(history, output_dir):
    """Plot emergence evolution."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Energy with emergence markers
    ax1.plot(history['step'], history['total'], 'b-', linewidth=2)
    for event in history['emergence_events']:
        ax1.axvline(event['step'], color='red', alpha=0.3, linestyle='--')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Energy')
    ax1.set_title('Energy (red = emergence)')
    ax1.grid(alpha=0.3)

    # Hierarchy evolution
    ax2.plot(history['step'], history['n_scales'], 'g-', marker='o', label='# Scales')
    ax2.plot(history['step'], history['n_active_agents'], 'b-', marker='s', label='# Active')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Count')
    ax2.set_title('Hierarchical Structure')
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    fig_path = output_dir / "emergence_evolution.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"✓ Saved {fig_path}")


def _generate_comprehensive_visualizations(system, analyzer, diagnostics, output_dir):
    """
    Generate comprehensive meta-agent visualizations.

    Uses the new visualization toolkit to create:
    - Hierarchy graphs (static and interactive)
    - Consensus matrices
    - Scale occupancy heatmaps
    - Energy landscapes
    - Coherence trajectories
    - And more!
    """
    from meta.visualization import (
        HierarchyVisualizer,
        ConsensusVisualizer,
        DynamicsVisualizer,
        create_analysis_report
    )
    from meta.energy_visualization import EnergyVisualizer

    print(f"\n{'='*70}")
    print("GENERATING COMPREHENSIVE VISUALIZATIONS")
    print(f"{'='*70}")

    # Create output directories
    meta_dir = output_dir / "meta_analysis"
    energy_dir = output_dir / "energy_analysis"
    meta_dir.mkdir(exist_ok=True, parents=True)
    energy_dir.mkdir(exist_ok=True, parents=True)

    # Generate structure and dynamics visualizations
    print("\n1. Meta-Agent Structure and Dynamics...")
    try:
        create_analysis_report(analyzer, str(meta_dir))
    except Exception as e:
        print(f"  ⚠️  Error generating meta-agent analysis: {e}")

    # Generate energy visualizations
    print("\n2. Energy Landscapes and Thermodynamics...")
    try:
        energy_viz = EnergyVisualizer(diagnostics)
        energy_viz.create_energy_report(str(energy_dir))
    except Exception as e:
        print(f"  ⚠️  Error generating energy analysis: {e}")

    # Generate interactive hierarchy (if possible)
    print("\n3. Interactive Visualizations...")
    try:
        hierarchy_viz = HierarchyVisualizer(analyzer)
        interactive_fig = hierarchy_viz.plot_interactive_hierarchy()
        if interactive_fig:
            interactive_path = output_dir / "interactive_hierarchy.html"
            interactive_fig.write_html(str(interactive_path))
            print(f"  ✓ Saved interactive hierarchy to {interactive_path}")
    except Exception as e:
        print(f"  ⚠️  Plotly not available or error: {e}")

    # Export data for external analysis
    print("\n4. Exporting Raw Data...")
    try:
        data_path = output_dir / "snapshots.json"
        analyzer.export_to_json(str(data_path))
        print(f"  ✓ Saved raw data to {data_path}")
    except Exception as e:
        print(f"  ⚠️  Error exporting data: {e}")

    # Print summary
    print(f"\n{'='*70}")
    print("VISUALIZATION COMPLETE")
    print(f"{'='*70}")
    print(f"Meta-agent analysis: {meta_dir}/")
    print(f"Energy analysis:     {energy_dir}/")
    print(f"Interactive plots:   {output_dir}/interactive_*.html")
    print(f"Raw data:            {output_dir}/snapshots.json")
    print(f"{'='*70}\n")

    # Print analysis summary
    if analyzer.snapshots:
        final_snapshot = analyzer.snapshots[-1]
        print("Final System State:")
        print(f"  Total agents:    {final_snapshot.metrics['total_agents']}")
        print(f"  Active agents:   {final_snapshot.metrics['total_active']}")
        print(f"  Max scale:       {final_snapshot.metrics['max_scale']}")
        print(f"  Meta-agents:     {len(final_snapshot.meta_agents)}")
        print(f"  Condensations:   {len(system.condensation_events)}")
        print()


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Streamlined Simulation Runner")
    parser.add_argument('--preset', type=str, default='default',
                       choices=['default', 'emergence', 'ouroboros', 'flat',
                               'hamiltonian', 'hamiltonian_emergence', 'critical_damping'],
                       help='Configuration preset')
    args = parser.parse_args()

    # Load configuration
    preset_map = {
        'default': default_config,
        'emergence': emergence_demo_config,
        'ouroboros': ouroboros_config,
        'flat': flat_agents_config,
        'hamiltonian': hamiltonian_config,
        'hamiltonian_emergence': hamiltonian_emergence_config,
        'critical_damping': critical_damping_config
    }
    cfg = preset_map[args.preset]()

    # Setup
    np.random.seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    output_dir = Path(cfg.output_dir) / cfg.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print("MULTI-AGENT SIMULATION")
    print(f"{'='*70}")
    print(f"Preset: {args.preset}")
    print(f"Output: {output_dir}")
    print(f"{'='*70}\n")

    # Build and run
    manifold = build_manifold(cfg)
    supports = build_supports(manifold, cfg, rng)
    agents = build_agents(manifold, supports, cfg, rng)
    system = build_system(agents, cfg, rng)

    # Save config
    cfg.save(str(output_dir / "config.txt"))

    # Train
    history = run_training(system, cfg, output_dir)

    # Summary
    print(f"\n{'='*70}")
    print("✓ SIMULATION COMPLETE")
    print(f"{'='*70}")
    print(f"Results: {output_dir}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()