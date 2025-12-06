# -*- coding: utf-8 -*-
"""
Created on Wed Nov 12 18:41:41 2025

@author: chris and christine
"""

"""
Deep 0-D base manifold test module
==================================

End-to-end tests for the zero-dimensional ("point") case:

- BaseManifold geometry in 0-D
- SupportRegionSmooth behavior at a single point
- Overlap masks and neighbor bookkeeping in MultiAgentSystem
- Free energy computation on a 0-D system
- Observation model wiring in 0-D
- Gradient engine output shapes/finiteness in 0-D
- Identical priors behaviour in 0-D
- Click-to-run helpers, including a small training smoke test
"""
import simulation_suite as sim
from agent.trainer import Trainer  # type: ignore
from config import TrainingConfig, SystemConfig
import numpy as np
from math_utils.fisher_metric import natural_gradient_gaussian
from gradients.gradient_engine import compute_natural_gradients
from test_gauge_inv import apply_global_gauge, energy_and_grads
import copy

from math_utils.transport_cache import remove_cache_from_system

from math_utils.generators import generate_so3_generators
from math_utils.transport import _matrix_exponential_so3

from test_gauge_inv import rodrigues  # uses the same 3×3 adjoint helper

# ---------------------------------------------------------------------
# Helper: energy object → dict
# ---------------------------------------------------------------------

def _energy_to_dict(energies):
    """
    Accept either:
      - FreeEnergyBreakdown (attributes self_energy, belief_align, prior_align,
        observations, total)
      - or a dict-like object

    and return a dict with keys:
      'self', 'belief_align', 'prior_align', 'obs', 'total'
    """
    # Dict-like?
    if isinstance(energies, dict):
        out = {}
        mapping_candidates = {
            "self": "self",
            "self_energy": "self",
            "belief_align": "belief_align",
            "prior_align": "prior_align",
            "obs": "obs",
            "observations": "obs",
            "total": "total",
        }
        for k_in, k_out in mapping_candidates.items():
            if k_in in energies:
                out[k_out] = energies[k_in]
        return out

    # FreeEnergyBreakdown-like
    out = {}
    if hasattr(energies, "self_energy"):
        out["self"] = energies.self_energy
    if hasattr(energies, "belief_align"):
        out["belief_align"] = energies.belief_align
    if hasattr(energies, "prior_align"):
        out["prior_align"] = energies.prior_align
    if hasattr(energies, "observations"):
        out["obs"] = energies.observations
    if hasattr(energies, "total"):
        out["total"] = energies.total
    return out


# ---------------------------------------------------------------------
# Dimension-agnostic small system builders
# ---------------------------------------------------------------------

def build_small_system(
    spatial_shape=(),
    n_agents: int = 3,
    K: int = 5,
    *,
    lambda_self: float = 1.0,
    lambda_belief_align: float = 1.0,
    lambda_prior_align: float = 0.0,
    lambda_obs: float = 0.0,
    identical_priors: str = "off",
    support_pattern: str | None = None,
    seed: int = 123,
):
    """
    Build a SMALL MultiAgentSystem using the simulation_suite pipeline,
    in a base-dimension agnostic way.

    Args:
        spatial_shape: ()
            -> 0-D "point" manifold
                        (SPATIAL_SHAPE = ())
                        SUPPORT_PATTERN is ignored by build_supports.
                       (n,) 
            -> 1-D manifold with n points
                        default SUPPORT_PATTERN = "intervals_1d"
                       (H, W)
            -> 2-D manifold
                        default SUPPORT_PATTERN = "circles_2d"
        n_agents: number of agents
        K: latent dimension
        lambda_*: energy weights
        identical_priors: "off", "init_copy", etc.
        support_pattern: if not None, force this SUPPORT_PATTERN.
        seed: RNG seed
    """
    # Normalize spatial_shape to a tuple
    if spatial_shape in (None, ()):
        spatial_shape = ()
    elif isinstance(spatial_shape, int):
        spatial_shape = (spatial_shape,)

    ndim = 0 if spatial_shape == () else len(spatial_shape)

    # --- Override global sim config for a tiny, fast test system ---
    sim.SPATIAL_SHAPE = spatial_shape
    sim.MANIFOLD_TOPOLOGY = "periodic"
    sim.N_AGENTS = n_agents
    sim.K_LATENT = K

    sim.LAMBDA_SELF = lambda_self
    sim.LAMBDA_BELIEF_ALIGN = lambda_belief_align
    sim.LAMBDA_PRIOR_ALIGN = lambda_prior_align
    sim.LAMBDA_OBS = lambda_obs

    sim.IDENTICAL_PRIORS = identical_priors

    # Set a sensible SUPPORT_PATTERN for each dimension, unless overridden
    if support_pattern is not None:
        sim.SUPPORT_PATTERN = support_pattern
    else:
        if ndim == 0:
            # 0-D: build_supports ignores SUPPORT_PATTERN, so don't touch it
            # to avoid poisoning later tests.
            pass
        elif ndim == 1:
            # Default 1-D pattern
            sim.SUPPORT_PATTERN = getattr(sim, "SUPPORT_PATTERN", "intervals_1d")
            if sim.SUPPORT_PATTERN not in ("full", "intervals_1d"):
                sim.SUPPORT_PATTERN = "intervals_1d"
        elif ndim == 2:
            # Default 2-D pattern
            sim.SUPPORT_PATTERN = getattr(sim, "SUPPORT_PATTERN", "circles_2d")
            if sim.SUPPORT_PATTERN not in ("full", "circles_2d"):
                sim.SUPPORT_PATTERN = "circles_2d"
        else:
            raise ValueError(f"Unsupported ndim={ndim} in build_small_system")

    rng = np.random.default_rng(seed)

    manifold = sim.build_manifold()
    supports = sim.build_supports(manifold, rng)
    agents = sim.build_agents(manifold, supports, rng)
    system = sim.build_system(agents, rng)

    return manifold, supports, agents, system


def build_small_0d_system(
    n_agents: int = 3,
    K: int = 5,
    *,
    lambda_self: float = 1.0,
    lambda_belief_align: float = 1.0,
    lambda_prior_align: float = 0.0,
    lambda_obs: float = 0.0,
    identical_priors: str = "off",
    seed: int = 123,
):
    """
    Legacy 0-D convenience wrapper around build_small_system.
    """
    return build_small_system(
        spatial_shape=(),
        n_agents=n_agents,
        K=K,
        lambda_self=lambda_self,
        lambda_belief_align=lambda_belief_align,
        lambda_prior_align=lambda_prior_align,
        lambda_obs=lambda_obs,
        identical_priors=identical_priors,
        support_pattern=None,  # don't touch SUPPORT_PATTERN for 0-D
        seed=seed,
    )


def build_small_2d_system(
    spatial_shape=(32, 32),
    n_agents=3,
    K=5,
    *,
    lambda_self: float = 1.0,
    lambda_belief_align: float = 1.0,
    lambda_prior_align: float = 0.0,
    lambda_obs: float = 0.0,
    identical_priors: str = "off",
    seed: int = 123,
):
    """
    Legacy 2-D convenience wrapper around build_small_system.
    """
    return build_small_system(
        spatial_shape=spatial_shape,
        n_agents=n_agents,
        K=K,
        lambda_self=lambda_self,
        lambda_belief_align=lambda_belief_align,
        lambda_prior_align=lambda_prior_align,
        lambda_obs=lambda_obs,
        identical_priors=identical_priors,
        support_pattern="circles_2d",
        seed=seed,
    )


# ---------------------------------------------------------------------
# 0-D geometry & support
# ---------------------------------------------------------------------

def test_0d_base_manifold_and_support_geometry():
    """
    Basic sanity for 0-D manifold + smooth supports.

    Ensures:
      - manifold.ndim == 0 and n_points == 1
      - supports are SupportRegionSmooth over a single point
      - mask_continuous is scalar-like and non-negative
      - agent.geometry is particle-like in 0-D
    """
    manifold, supports, agents, system = build_small_0d_system(
        n_agents=2,
        K=3,
        lambda_self=0.0,
        lambda_belief_align=0.0,
        lambda_prior_align=0.0,
        lambda_obs=0.0,
    )

    # Base manifold geometry
    assert manifold.shape == ()
    assert manifold.ndim == 0
    assert manifold.n_points == 1

    # Each support should live on the single point
    for s in supports:
        assert s.base_shape == ()
        mask_c = np.asarray(s.mask_continuous)
        assert mask_c.size == 1
        assert np.all(mask_c >= 0.0)
        assert s.n_active == 1

    # Agent geometry sanity
    for a in agents:
        assert a.geometry.is_particle
        assert a.support_shape == ()
        assert a.geometry.ndim == 0
        assert a.geometry.total_points == 1
        assert a.geometry.n_active == 1


# ---------------------------------------------------------------------
# Overlaps & neighbors on a point manifold
# ---------------------------------------------------------------------

def test_0d_overlap_masks_and_neighbors():
    """
    On a point manifold with full supports, every ordered pair (i,j), i≠j,
    should have a non-zero overlap χ_ij.

    This checks:
      - overlap_masks keys and scalar shapes
      - neighbor lists for each agent
    """
    n_agents = 3
    manifold, supports, agents, system = build_small_0d_system(
        n_agents=n_agents,
        K=3,
        lambda_self=0.0,
        lambda_belief_align=0.0,
        lambda_prior_align=0.0,
        lambda_obs=0.0,
    )

    # There should be an overlap for every ordered pair (i, j), i != j
    expected_pairs = {(i, j) for i in range(n_agents) for j in range(n_agents) if i != j}
    found_pairs = set(system.overlap_masks.keys())
    assert found_pairs == expected_pairs

    # Each χ_ij should be scalar-ish and between 0 and 1
    for (i, j), chi_ij in system.overlap_masks.items():
        arr = np.asarray(chi_ij)
        assert arr.size == 1, f"χ_{i}{j} should be scalar-like in 0D, got shape {arr.shape}"
        val = float(arr.reshape(()))
        assert 0.0 <= val <= 1.0, f"χ_{i}{j} not in [0,1]: {val}"

    # Neighbor lists should reflect full overlap
    for i in range(n_agents):
        neighbors = system.get_neighbors(i)
        assert sorted(neighbors) == sorted([j for j in range(n_agents) if j != i])


# ---------------------------------------------------------------------
# Free energy in 0-D
# ---------------------------------------------------------------------

def test_0d_free_energy_finite():
    """
    Build a pure-coupling 0-D system and verify that compute_free_energy()
    returns finite scalars for all components.

    NOTE: We currently only have:
      self, belief_align, prior_align, obs, total
    """
    _, _, _, system = build_small_0d_system(
        n_agents=3,
        K=5,
        lambda_self=1.0,
        lambda_belief_align=1.0,
        lambda_prior_align=0.0,
        lambda_obs=0.0,
    )

    energies_raw = system.compute_free_energy()
    energies = _energy_to_dict(energies_raw)

    for key in ["total", "self", "belief_align", "prior_align", "obs"]:
        assert key in energies, f"Missing energy component '{key}'"
        val = float(energies[key])
        assert np.isfinite(val), f"Energy component '{key}' is not finite: {val}"


# ---------------------------------------------------------------------
# Observation model in 0-D
# ---------------------------------------------------------------------

def test_0d_observation_model_shapes_and_energy():
    """
    Turn on observations and check that:
      - system.config.D_x matches sim.D_X
      - W_obs, R_obs shapes are correct
      - Each agent gets a 0-D observation x_obs of shape (D_x,)
      - Observation energy is finite
    """
    _, _, agents, system = build_small_0d_system(
        n_agents=3,
        K=5,
        lambda_self=1.0,
        lambda_belief_align=1.0,
        lambda_prior_align=0.0,
        lambda_obs=1.0,  # turn on obs term
    )

    cfg: SystemConfig = system.config
    D_x_expected = getattr(sim, "D_X", cfg.D_x)
    assert cfg.D_x == D_x_expected

    # Lazily initialize observation model if needed
    system.ensure_observation_model()

    # Check W_obs, R_obs shapes
    assert system.W_obs is not None
    assert system.R_obs is not None
    assert system.W_obs.shape == (cfg.D_x, agents[0].K)
    assert system.R_obs.shape == (cfg.D_x, cfg.D_x)

    # Each agent should have a single 0-D observation vector
    for agent in system.agents:
        assert hasattr(agent, "x_obs")
        x_obs = np.asarray(agent.x_obs)
        assert x_obs.shape == (cfg.D_x,), f"0D x_obs should be (D_x,), got {x_obs.shape}"

    # Observation contribution to energy should be finite
    energies_raw = system.compute_free_energy()
    energies = _energy_to_dict(energies_raw)
    assert np.isfinite(float(energies["obs"]))


# ---------------------------------------------------------------------
# Gradient engine on 0-D (natural gradients)
# ---------------------------------------------------------------------

def test_0d_gradients_shapes_and_finiteness():
    """
    Run the gradient engine on a 0-D system and ensure that:
      - We get one AgentGradients object per agent
      - Shapes of grad_{mu,Sigma,phi} fields match agent fields
      - Natural gradients delta_L_* exist with correct shapes
      - All gradient entries are finite

    Uses compute_natural_gradients(system) directly (no system.step()).
    """
    _, _, agents, system = build_small_0d_system(
        n_agents=3,
        K=3,
        lambda_self=1.0,
        lambda_belief_align=1.0,
        lambda_prior_align=1.0,
        lambda_obs=1.0,
    )

    # Ensure observation model so obs gradients get included
    system.ensure_observation_model()

    grads_list = compute_natural_gradients(system)
    assert len(grads_list) == len(agents)

    for agent, g in zip(agents, grads_list):
        # Euclidean grads should always be present
        for name, arr in [
            ("grad_mu_q", g.grad_mu_q),
            ("grad_Sigma_q", g.grad_Sigma_q),
            ("grad_mu_p", g.grad_mu_p),
            ("grad_Sigma_p", g.grad_Sigma_p),
            ("grad_phi", g.grad_phi),
        ]:
            arr = np.asarray(arr)
            assert np.all(np.isfinite(arr)), f"Non-finite values in {name} for agent {agent.agent_id}"

        # Shape checks
        assert g.grad_mu_q.shape == agent.mu_q.shape
        assert g.grad_Sigma_q.shape == agent.Sigma_q.shape
        assert g.grad_mu_p.shape == agent.mu_p.shape
        assert g.grad_Sigma_p.shape == agent.Sigma_p.shape
        assert g.grad_phi.shape == agent.gauge.phi.shape

        # Natural gradients used by Trainer: delta_mu_q, delta_mu_p, delta_L_q, delta_L_p, delta_phi
        for name, arr, target in [
            ("delta_mu_q", g.delta_mu_q, agent.mu_q),
            ("delta_mu_p", g.delta_mu_p, agent.mu_p),
            ("delta_L_q", g.delta_L_q, agent.L_q),
            ("delta_L_p", g.delta_L_p, agent.L_p),
            ("delta_phi", g.delta_phi, agent.gauge.phi),
        ]:
            assert arr is not None, f"{name} is None for agent {agent.agent_id}"
            arr = np.asarray(arr)
            assert arr.shape == target.shape, f"{name} shape {arr.shape} != {target.shape}"
            assert np.all(np.isfinite(arr)), f"Non-finite values in {name} for agent {agent.agent_id}"


# ---------------------------------------------------------------------
# Identical priors behavior on 0-D (Trainer-based)
# ---------------------------------------------------------------------

def test_0d_identical_priors_enforced_by_trainer():
    """
    With identical_priors='lock', priors should be kept identical across agents
    even after a Trainer.step().

    NOTE:
      If this test fails, it means the lock is not being re-applied after
      Trainer updates (i.e. we need to call system._apply_identical_priors_now()
      inside Trainer after all agent updates).
    """
    _, _, agents, system = build_small_0d_system(
        n_agents=3,
        K=3,
        lambda_self=1.0,
        lambda_belief_align=1.0,
        lambda_prior_align=1.0,
        lambda_obs=0.0,
        identical_priors="lock",
    )

    # Capture initial priors
    mu0 = [np.array(a.mu_p, copy=True) for a in agents]
    L0  = [np.array(a.L_p,  copy=True) for a in agents]

    # All priors should already be identical after system init
    for k in range(1, len(agents)):
        assert np.allclose(mu0[0], mu0[k])
        assert np.allclose(L0[0],  L0[k])

    # Build a Trainer with non-zero prior learning rates so drift is possible
    train_cfg = TrainingConfig(
        n_steps=1,
        lr_mu_q=0.0,
        lr_sigma_q=0.0,
        lr_mu_p=0.05,
        lr_sigma_p=0.05,
        lr_phi=0.0,
        save_history=False,
    )
    trainer = Trainer(system, config=train_cfg)

    # One optimization step via Trainer (this is the *current* update path)
    trainer.step()

    mu1 = [np.array(a.mu_p, copy=True) for a in agents]
    L1  = [np.array(a.L_p,  copy=True) for a in agents]

    # EXPECTATION for a correct implementation:
    # priors STILL identical across agents.
    #
    # If these asserts fail, it's a *real* regression: lock mode is broken.
    for k in range(1, len(agents)):
        assert np.allclose(mu1[0], mu1[k]), "mu_p drifted apart under identical_priors='lock' with Trainer"
        assert np.allclose(L1[0],  L1[k]),  "L_p drifted apart under identical_priors='lock' with Trainer"


# ---------------------------------------------------------------------
# Click-to-run helpers
# ---------------------------------------------------------------------




def run_0d_training_smoke_test(
    *,
    n_steps: int = 20,
    n_agents: int = 3,
    K: int = 3,
    lr_mu_q: float = 0.05,
    lr_sigma_q: float = 0.01,
    lr_mu_p: float = 0.01,
    lr_sigma_p: float = 0.01,
    lr_phi: float = 0.01,
    lambda_self: float = 1.0,
    lambda_belief_align: float = 1.0,
    lambda_prior_align: float = 0.0,
    lambda_obs: float = 1.0,
    identical_priors: str = "off",
    seed: int = 123,
):
    """
    Small, configurable 0-D training run using the *current* Trainer.

    - Builds a 0-D system
    - Runs Trainer for a few steps
    - Prints initial/final energies and returns (system, history)

    Usage:
        from test_zero_dimensional_manifold import run_0d_training_smoke_test
        system, history = run_0d_training_smoke_test(n_steps=10)
    """
    # Build a fresh 0-D system with the desired energy weights
    _, _, _, system = build_small_0d_system(
        n_agents=n_agents,
        K=K,
        lambda_self=lambda_self,
        lambda_belief_align=lambda_belief_align,
        lambda_prior_align=lambda_prior_align,
        lambda_obs=lambda_obs,
        identical_priors=identical_priors,
        seed=seed,
    )

    # Set up a compact training config
    train_cfg = TrainingConfig(
        n_steps=n_steps,
        lr_mu_q=lr_mu_q,
        lr_sigma_q=lr_sigma_q,
        lr_mu_p=lr_mu_p,
        lr_sigma_p=lr_sigma_p,
        lr_phi=lr_phi,
        save_history=True,
    )

    trainer = Trainer(system, config=train_cfg)

    # Measure initial energy
    energies0 = _energy_to_dict(system.compute_free_energy())
    initial_E = float(energies0["total"])
    print("\n[0D TRAINING SMOKE TEST]")
    print(f"  Agents   : {system.n_agents}")
    print(f"  K        : {system.agents[0].K}")
    print(f"  Steps    : {n_steps}")
    print(f"  λ_self   : {lambda_self}")
    print(f"  λ_align  : belief={lambda_belief_align}, prior={lambda_prior_align}")
    print(f"  λ_obs    : {lambda_obs}")
    print(f"  Priors   : {identical_priors}")
    print(f"  Initial E: {initial_E:.6e}")

    history = trainer.train(n_steps=n_steps)

    energies1 = _energy_to_dict(system.compute_free_energy())
    final_E = float(energies1["total"])
    print(f"  Final   E: {final_E:.6e}")
    print("  ΔE        = {:.6e}".format(final_E - initial_E))

    return system, history

def test_0d_parameters_frozen_when_all_lrs_zero():
    """
    If all learning rates are zero, parameters MUST NOT move,
    even if the energy weights λ_* are non-zero (i.e., gradients exist).

    This catches subtle bugs where something is mutating fields
    outside of the Trainer's controlled updates.
    """
    # Non-zero lambdas so gradients are non-trivial
    _, _, agents, system = build_small_0d_system(
        n_agents=3,
        K=3,
        lambda_self=1.0,
        lambda_belief_align=1.0,
        lambda_prior_align=1.0,
        lambda_obs=1.0,
        identical_priors="off",
        seed=123,
    )

    # Snapshot all relevant parameters
    mu_q0  = [np.array(a.mu_q,  copy=True) for a in agents]
    L_q0   = [np.array(a.L_q,   copy=True) for a in agents]
    mu_p0  = [np.array(a.mu_p,  copy=True) for a in agents]
    L_p0   = [np.array(a.L_p,   copy=True) for a in agents]
    phi0   = [np.array(a.gauge.phi, copy=True) for a in agents]

    # All learning rates zero
    train_cfg = TrainingConfig(
        n_steps=10,
        lr_mu_q=0.0,
        lr_sigma_q=0.0,
        lr_mu_p=0.0,
        lr_sigma_p=0.0,
        lr_phi=0.0,
        save_history=False,
    )
    trainer = Trainer(system, config=train_cfg)
    trainer.train(n_steps=10)

    # Compare after training
    for idx, a in enumerate(agents):
        assert np.allclose(a.mu_q,  mu_q0[idx]), f"mu_q changed for agent {idx} despite zero LR"
        assert np.allclose(a.L_q,   L_q0[idx]),  f"L_q changed for agent {idx} despite zero LR"
        assert np.allclose(a.mu_p,  mu_p0[idx]), f"mu_p changed for agent {idx} despite zero LR"
        assert np.allclose(a.L_p,   L_p0[idx]),  f"L_p changed for agent {idx} despite zero LR"
        assert np.allclose(a.gauge.phi, phi0[idx]), f"phi changed for agent {idx} despite zero LR"

def test_0d_gradients_vanish_when_all_lambdas_zero():
    """
    When all λ_* terms are zero, free energy should be constant
    and the natural gradients should be (numerically) zero.

    This ensures the gradient engine is respecting the energy
    weights and not injecting stray forces.
    """
    # Build a 0-D system with ALL energy terms off
    _, _, agents, system = build_small_0d_system(
        n_agents=3,
        K=3,
        lambda_self=0.0,
        lambda_belief_align=0.0,
        lambda_prior_align=0.0,
        lambda_obs=0.0,
        identical_priors="off",
        seed=321,
    )

    grads_list = compute_natural_gradients(system)

    # Numerical tolerance for "zero"
    tol = 1e-8

    for agent, g in zip(agents, grads_list):
        for name, arr in [
            ("grad_mu_q", g.grad_mu_q),
            ("grad_Sigma_q", g.grad_Sigma_q),
            ("grad_mu_p", g.grad_mu_p),
            ("grad_Sigma_p", g.grad_Sigma_p),
            ("grad_phi", g.grad_phi),
            ("delta_mu_q", g.delta_mu_q),
            ("delta_mu_p", g.delta_mu_p),
            ("delta_L_q", g.delta_L_q),
            ("delta_L_p", g.delta_L_p),
            ("delta_phi", g.delta_phi),
        ]:
            arr = np.asarray(arr)
            norm = np.linalg.norm(arr.ravel())
            assert norm < tol, f"{name} has non-zero norm {norm:.3e} with all λ=0 for agent {agent.agent_id}"


def test_0d_dynamic_gauge_invariance_over_training():
    """
    Build a 0-D system and a randomly gauge-rotated copy, run the same
    Trainer config on both, and check that the total energy traces match
    (within tolerance) when no gauge-fixing or phi terms are active.
    """
    n_agents = 3
    K = 5

    # --- Base system (no priors/obs/phi in the energy) ---
    _, _, agents_base, system_base = build_small_0d_system(
        n_agents=n_agents,
        K=K,
        lambda_self=1.0,
        lambda_belief_align=1.0,
        lambda_prior_align=0.0,
        lambda_obs=0.0,
        identical_priors="init_copy",  # or "off"
        seed=123,
    )

    # Clone: build another system with the same seed/config
    _, _, agents_gauge, system_gauge = build_small_0d_system(
        n_agents=n_agents,
        K=K,
        lambda_self=1.0,
        lambda_belief_align=1.0,
        lambda_prior_align=0.0,
        lambda_obs=0.0,
        identical_priors="init_copy",
        seed=123,
    )

    # --- Apply a random SO(K) gauge to each agent in the second system ---
    rng = np.random.default_rng(42)
    for a_base, a_gauge in zip(agents_base, agents_gauge):
        # Random orthogonal matrix R (from QR)
        M = rng.standard_normal((K, K))
        Q, _ = np.linalg.qr(M)
        R = Q.astype(np.float32)

        # Rotate beliefs (mu_q, Sigma_q)
        a_gauge.mu_q = (R @ a_gauge.mu_q[..., None]).squeeze(-1)
        a_gauge.Sigma_q = R @ a_gauge.Sigma_q @ R.T

        # Rotate priors too (even though lambda_prior_align = 0, this is the
        # "physically correct" gauge action)
        a_gauge.mu_p = (R @ a_gauge.mu_p[..., None]).squeeze(-1)
        a_gauge.Sigma_p = R @ a_gauge.Sigma_p @ R.T

        # Rotate gauge generators if needed / or leave phi fixed if lambda_phi=0

    # --- Same Trainer config for both systems ---
    train_cfg = TrainingConfig(
        n_steps=50,
        lr_mu_q=0.05,
        lr_sigma_q=0.05,
        lr_mu_p=0.00,
        lr_sigma_p=0.00,
        lr_phi=0.00,
        save_history=False,
    )

    trainer_base = Trainer(system_base, config=train_cfg)
    trainer_gauge = Trainer(system_gauge, config=train_cfg)

    # Track energies
    E_base = []
    E_gauge = []

    for _ in range(train_cfg.n_steps):
        E0 = _energy_to_dict(system_base.compute_free_energy())["total"]
        E1 = _energy_to_dict(system_gauge.compute_free_energy())["total"]
        E_base.append(float(E0))
        E_gauge.append(float(E1))

        trainer_base.step()
        trainer_gauge.step()

    # Final comparison: energies should match up to small numerical noise
    E_base = np.array(E_base)
    E_gauge = np.array(E_gauge)

    diff = np.abs(E_base - E_gauge)
    max_diff = diff.max()
    assert max_diff < 1e-5, f"Dynamic gauge invariance failed, max ΔE = {max_diff:.3e}"



def test_0d_training_gauge_invariance_after_steps():
    # Build and train a small 0-D system through simulation_suite + Trainer
    manifold, supports, agents, system = build_small_0d_system(
        n_agents=3,
        K=5,
        lambda_self=1.0,
        lambda_belief_align=1.0,
        lambda_prior_align=0.0,
        lambda_obs=0.0,
        identical_priors="init_copy",
        seed=123,
    )

    train_cfg = TrainingConfig(
        n_steps=20,
        lr_mu_q=0.01,
        lr_sigma_q=0.01,
        lr_mu_p=0.0,
        lr_sigma_p=0.0,
        lr_phi=0.0,
        save_history=False,
    )
    trainer = Trainer(system, config=train_cfg)
    trainer.train(n_steps=20)

    # --- IMPORTANT: remove cache before deepcopy so the clone
    #     doesn't keep a closure bound to the original system
    remove_cache_from_system(system)

    # Now do the gauge test on *uncached* systems
    E0, grads0 = energy_and_grads(system)

    sys_g = copy.deepcopy(system)
    sys_g = apply_global_gauge(sys_g)

    # (optional) if you want caches for performance in the test:
    # add_cache_to_system(system, max_size=1000)
    # add_cache_to_system(sys_g, max_size=1000)

    Eg, gradsG = energy_and_grads(sys_g)

    # Quick assertion on total energy
    assert np.allclose(E0["total"], Eg["total"], rtol=1e-6, atol=1e-6)
    # You can also check components:
    assert np.allclose(E0["belief_align"], Eg["belief_align"], rtol=1e-6, atol=1e-6)
    assert np.allclose(E0["prior_align"],  Eg["prior_align"],  rtol=1e-6, atol=1e-6)




def apply_global_gauge_with_axis(system, axis: np.ndarray):
    """
    Apply a *fixed* global SO(3) rotation (axis-angle in R^3) to every agent.

    ND-safe version:
      • Works for μ_q, μ_p of shape (..., K).
      • Works for L_q, L_p of shape (..., K, K).
      • Uses Σ' = R Σ R^T, then Cholesky, with full broadcasting.
      • φ is rotated in adjoint space with broadcasting over spatial dims.
    """
    axis = np.asarray(axis, dtype=np.float64)

    for a in system.agents:
        K = a.K

        # K-dim irrep of this group element
        G = generate_so3_generators(K)
        R_K = _matrix_exponential_so3(axis, G)   # (K, K)

        # 3×3 adjoint rotation
        R_adj = rodrigues(axis)                 # (3, 3)

        # ------------------------------------------------------------------
        # Means μ: shape (..., K)
        # μ'(..., i) = R_K[i, j] μ(..., j)
        # ------------------------------------------------------------------
        a.mu_q = np.einsum("ij,...j->...i", R_K, a.mu_q)
        a.mu_p = np.einsum("ij,...j->...i", R_K, a.mu_p)

        # ------------------------------------------------------------------
        # Covariances via L → Σ → R Σ R^T → Cholesky over all spatial points
        # L has shape (..., K, K)
        # ------------------------------------------------------------------

        # Belief covariances
        L_q = a.L_q.astype(np.float64)
        Sigma_q = L_q @ np.swapaxes(L_q, -1, -2)       # (..., K, K)
        Sigma_q_rot = np.einsum("ab,...bc,dc->...ad", R_K, Sigma_q, R_K)
        a.L_q = np.linalg.cholesky(Sigma_q_rot).astype(np.float32)

        # Prior / model covariances
        L_p = a.L_p.astype(np.float64)
        Sigma_p = L_p @ np.swapaxes(L_p, -1, -2)       # (..., K, K)
        Sigma_p_rot = np.einsum("ab,...bc,dc->...ad", R_K, Sigma_p, R_K)
        a.L_p = np.linalg.cholesky(Sigma_p_rot).astype(np.float32)

        # ------------------------------------------------------------------
        # Gauge field φ: shape (..., 3)
        # φ'(..., i) = R_adj[i, j] φ(..., j)
        # ------------------------------------------------------------------
        if hasattr(a, "gauge") and hasattr(a.gauge, "phi"):
            a.gauge.phi = np.einsum("ij,...j->...i", R_adj, a.gauge.phi)

        # Clear per-agent caches if present
        if hasattr(a, "invalidate_caches"):
            a.invalidate_caches()

    # Rebuild masks / overlaps if needed
    if hasattr(system, "_compute_overlap_masks"):
        system.overlap_masks = system._compute_overlap_masks()

    # Leave system.transport_cache alone here; higher-level tests can
    # invalidate it explicitly if needed.
    return system






def test_training_gauge_covariant_dynamics_0d():
    """
    0-D gauge-covariant training test, via the dimension-agnostic runner.
    """
    _run_training_gauge_covariant_dynamics(
        spatial_shape=(),   # 0-D / point
        n_agents=3,
        K=5,
        n_steps=20,
        identical_priors="init_copy",
        axis_seed=999,
        train_seed=123,
        lr_mu_q=0.01,
        lr_sigma_q=0.01,
        lr_mu_p=0.0,
        lr_sigma_p=0.0,
        lr_phi=0.0,
    )
    print("✅ 0-D gauge-covariant dynamics test PASSED.")


def test_training_gauge_covariant_dynamics_2d():
    """
    2-D analogue of the 0-D gauge-covariant dynamics test,
    using the same dimension-agnostic runner.
    """
    _run_training_gauge_covariant_dynamics(
        spatial_shape=(16, 16),
        n_agents=3,
        K=5,
        n_steps=10,
        identical_priors="off",
        axis_seed=999,
        train_seed=123,
        lr_mu_q=0.01,
        lr_sigma_q=0.01,
        lr_mu_p=0.0,
        lr_sigma_p=0.0,
        lr_phi=0.0,
    )
    print("✅ 2-D gauge-covariant dynamics test PASSED.")



def random_spd(K, rng):
    A = rng.standard_normal((K, K))
    Sigma = A @ A.T
    # Add a bit of identity for conditioning
    Sigma += 0.5 * np.eye(K)
    return Sigma


def random_so(K, rng):
    """Random orthogonal matrix with det=+1 (SO(K))."""
    Q, _ = np.linalg.qr(rng.standard_normal((K, K)))
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1.0
    return Q


def test_natural_gradient_gaussian_gauge_invariance():
    rng = np.random.default_rng(0)
    K = 3

    # Base Gaussian + Euclidean grads
    mu = rng.standard_normal(K)
    Sigma = random_spd(K, rng)

    grad_mu = rng.standard_normal(K)

    G = rng.standard_normal((K, K))
    grad_Sigma = 0.5 * (G + G.T)  # symmetrize for clarity

    delta_mu, delta_Sigma = natural_gradient_gaussian(
        mu,
        Sigma,
        grad_mu,
        grad_Sigma,
        eps=1e-10,
        assume_symmetric=False,
    )

    # Random global gauge rotation R ∈ SO(K)
    R = random_so(K, rng)

    mu_R = R @ mu
    Sigma_R = R @ Sigma @ R.T
    grad_mu_R = R @ grad_mu
    grad_Sigma_R = R @ grad_Sigma @ R.T

    delta_mu_R, delta_Sigma_R = natural_gradient_gaussian(
        mu_R,
        Sigma_R,
        grad_mu_R,
        grad_Sigma_R,
        eps=1e-10,
        assume_symmetric=False,
    )

    # Check equivariance:
    #   δμ' ≈ R δμ
    #   δΣ' ≈ R δΣ Rᵀ
    assert np.allclose(delta_mu_R, R @ delta_mu, atol=1e-6, rtol=1e-6)
    assert np.allclose(delta_Sigma_R, R @ delta_Sigma @ R.T, atol=1e-6, rtol=1e-6)












def apply_global_gauge_spatial_with_axis(system, axis: np.ndarray):
    """
    Apply a *fixed* global SO(3) rotation (axis-angle in R^3) to every agent,
    broadcasting over all spatial points.

    This is the spatial analogue of test_gauge_inv.apply_global_gauge:
      • μ_q, μ_p transform in the K-dim irrep (R_K).
      • L_q, L_p transform in the same irrep (left-multiply by R_K).
      • φ transforms in the 3D adjoint (R_adj).
      • The generator basis a.generators stays fixed.
    """
    axis = np.asarray(axis, dtype=np.float64)

    for a in system.agents:
        K = a.K

        # K-dim irrep of this group element
        G = generate_so3_generators(K)
        R_K = _matrix_exponential_so3(axis, G)   # (K, K)

        # 3×3 adjoint rotation on algebra coords
        R_adj = rodrigues(axis)                 # (3, 3)

        # ----- μ fields: shape (*S, K) -----
        # μ'(..., i) = R_K[i, j] μ(..., j)
        a.mu_q = np.einsum("ij,...j->...i", R_K, a.mu_q)
        a.mu_p = np.einsum("ij,...j->...i", R_K, a.mu_p)

        # ----- L fields: shape (*S, K, K) -----
        # L'(..., i, k) = R_K[i, j] L(..., j, k)
        a.L_q = np.einsum("ij,...jk->...ik", R_K, a.L_q)
        a.L_p = np.einsum("ij,...jk->...ik", R_K, a.L_p)

        # ----- φ field: shape (..., 3) -----
        if hasattr(a, "gauge") and hasattr(a.gauge, "phi"):
            # φ'(..., i) = R_adj[i, j] φ(..., j)
            a.gauge.phi = np.einsum("ij,...j->...i", R_adj, a.gauge.phi)

        # Clear any per-agent caches
        if hasattr(a, "invalidate_caches"):
            a.invalidate_caches()

    # Rebuild overlaps and reset system-level cache
    if hasattr(system, "_compute_overlap_masks"):
        system.overlap_masks = system._compute_overlap_masks()

    if hasattr(system, "transport_cache") and hasattr(system.transport_cache, "invalidate"):
        system.transport_cache.invalidate()

    return system




def _run_training_gauge_covariant_dynamics(
    spatial_shape,
    *,
    n_agents=3,
    K=5,
    n_steps=10,
    identical_priors="off",
    axis_seed=999,
    train_seed=123,
    lr_mu_q=0.01,
    lr_sigma_q=0.01,
    lr_mu_p=0.0,
    lr_sigma_p=0.0,
    lr_phi=0.0,
    atol_energy=1e-7,
    rtol_energy=1e-5,
    atol_mu=1e-7,
    rtol_mu=1e-5,
    atol_sigma=1e-4,
    rtol_sigma=1e-4,
):
    """
    Dimension-agnostic gauge-covariant training test:

        Train(S₀ᵍ) ≈ (Train(S₀))ᵍ

    for a fixed global SO(3) rotation g, where:

        S₀       : baseline system with given spatial_shape
        S₀ᵍ      : deep copy of S₀, globally rotated by g
        Train(.) : Trainer with shared TrainingConfig

    This works for 0-D, 1-D, 2-D, ... as long as apply_global_gauge_with_axis
    is ND-safe (which it is).
    """
    # ------------------------------
    # 1. Build baseline system S₀
    # ------------------------------
    _, _, _, system0 = build_small_system(
        spatial_shape=spatial_shape,
        n_agents=n_agents,
        K=K,
        lambda_self=1.0,
        lambda_belief_align=1.0,
        lambda_prior_align=0.0,
        lambda_obs=0.0,
        identical_priors=identical_priors,
        seed=train_seed,
    )

    # Copy for rotated-branch initial condition
    system0_rot = copy.deepcopy(system0)

    # --------------------------------------
    # 2. Choose fixed global axis-angle g
    #    and apply to rotated branch
    # --------------------------------------
    rng_axis = np.random.default_rng(axis_seed)
    axis = rng_axis.standard_normal(3)

    # ND-safe global gauge (uses Σ → R Σ Rᵀ → chol)
    system0_rot = apply_global_gauge_with_axis(system0_rot, axis)

    # --------------------------------------
    # 3. Train both systems with same config
    # --------------------------------------
    train_cfg = TrainingConfig(
        n_steps=n_steps,
        lr_mu_q=lr_mu_q,
        lr_sigma_q=lr_sigma_q,
        lr_mu_p=lr_mu_p,
        lr_sigma_p=lr_sigma_p,
        lr_phi=lr_phi,       # often 0 for these tests
        save_history=False,
    )

    trainer_A = Trainer(system0, config=train_cfg)
    trainer_B = Trainer(system0_rot, config=train_cfg)

    trainer_A.train(n_steps=n_steps)
    trainer_B.train(n_steps=n_steps)

    # --------------------------------------
    # 4. Remove transport caches before
    #    applying *final* gauge
    # --------------------------------------
    remove_cache_from_system(system0)
    remove_cache_from_system(system0_rot)

    # --------------------------------------
    # 5. Gauge-transform S₀_final by SAME g
    # --------------------------------------
    system0_g = copy.deepcopy(system0)
    system0_g = apply_global_gauge_with_axis(system0_g, axis)

    # --------------------------------------
    # 6. Compare energies + fields:
    #      S₀_finalᵍ vs S₀ᵍ_final
    # --------------------------------------
    E_A  = _energy_to_dict(system0.compute_free_energy())
    E_B  = _energy_to_dict(system0_rot.compute_free_energy())
    E_Ag = _energy_to_dict(system0_g.compute_free_energy())

    for key in ["total", "self", "belief_align", "prior_align", "obs"]:
        if key in E_B and key in E_Ag:
            assert np.allclose(
                E_B[key], E_Ag[key],
                rtol=rtol_energy, atol=atol_energy
            ), f"[{spatial_shape}] Energy component '{key}' differs after gauge-covariant training"

    # Per-agent field comparison
    for idx, (a_g, b) in enumerate(zip(system0_g.agents, system0_rot.agents)):
        # μ fields
        assert np.allclose(
            a_g.mu_q, b.mu_q,
            rtol=rtol_mu, atol=atol_mu
        ), f"[{spatial_shape}] Agent {idx}: mu_q mismatch"
        assert np.allclose(
            a_g.mu_p, b.mu_p,
            rtol=rtol_mu, atol=atol_mu
        ), f"[{spatial_shape}] Agent {idx}: mu_p mismatch"

        # Σ fields (via properties)
        assert np.allclose(
            a_g.Sigma_q, b.Sigma_q,
            rtol=rtol_sigma, atol=atol_sigma
        ), f"[{spatial_shape}] Agent {idx}: Sigma_q mismatch"
        assert np.allclose(
            a_g.Sigma_p, b.Sigma_p,
            rtol=rtol_sigma, atol=atol_sigma
        ), f"[{spatial_shape}] Agent {idx}: Sigma_p mismatch"

        # Gauge field φ if present (should be identical because lr_phi often 0)
        if hasattr(a_g, "gauge") and hasattr(a_g.gauge, "phi"):
            assert np.allclose(
                a_g.gauge.phi, b.gauge.phi,
                rtol=rtol_mu, atol=atol_mu
            ), f"[{spatial_shape}] Agent {idx}: phi mismatch"




def run_0d_deep_suite(verbose=True):
    tests = [
       # test_0d_base_manifold_and_support_geometry,
       # test_0d_overlap_masks_and_neighbors,
       # test_0d_free_energy_finite,
       # test_0d_observation_model_shapes_and_energy,
       # test_0d_gradients_shapes_and_finiteness,
      #  test_0d_identical_priors_enforced_by_trainer,
      #  test_0d_parameters_frozen_when_all_lrs_zero,
      #  test_0d_gradients_vanish_when_all_lambdas_zero,
      #  test_natural_gradient_gaussian_gauge_invariance,
      #  test_0d_training_gauge_invariance_after_steps,
        # Dynamic-trajectory GI is intentionally *not* enforced.
        test_training_gauge_covariant_dynamics_0d,
        test_training_gauge_covariant_dynamics_2d,
    ]
    


    for fn in tests:
        print(f"\n=== Running {fn.__name__} ===")
        fn()

    print("\n✓ All 0-D deep tests passed.")



if __name__ == "__main__":
    # Simple click-run behaviour: deep suite + a short training smoke test
    run_0d_deep_suite(verbose=True)
    print("\n" + "="*70)
    print("Now running 0-D training smoke test...")
    print("="*70)
   # run_0d_training_smoke_test(n_steps=50)
    
