"""
Data Loading and Preprocessing for Analysis
============================================

Functions for loading training history, system state, and filtering data.
"""

import pickle
import numpy as np
from pathlib import Path
from copy import deepcopy
from typing import Optional, Dict, Any


# Default skip steps for analysis plots (ignore initial transients)
DEFAULT_SKIP_STEPS = 0


def load_history(run_dir: Path):
    """Load training history from pkl (preferred) or npz."""
    pkl_path = run_dir / "history.pkl"
    npz_path = run_dir / "history.npz"

    # PREFER pkl because it has mu_tracker!
    if pkl_path.exists():
        with open(pkl_path, "rb") as f:
            history = pickle.load(f)
        print(f"✓ Loaded history from {pkl_path}")

        # Check if it has mu_tracker
        mu_tracker = get_mu_tracker(history)
        if mu_tracker is not None:
            print(f"  ✓ Mu tracking data available: {len(mu_tracker.steps)} steps")
        else:
            print("  ⚠️  No mu tracking data in history")

        return history

    # Fallback to npz (but this won't have mu_tracker)
    if npz_path.exists():
        data = np.load(npz_path)
        history = {k: data[k] for k in data.files}
        print(f"✓ Loaded history from {npz_path}")
        print("  ⚠️  NPZ format doesn't include mu_tracker - use PKL for full data")
        return history

    print("⚠️  No history.(npz|pkl) found in run directory.")
    return None


def get_mu_tracker(history):
    """
    Extract mu_tracker from history regardless of format (dict or object).

    Args:
        history: Either a dict with 'mu_tracker' key or object with mu_tracker attribute

    Returns:
        MuCenterTracking instance or None
    """
    if history is None:
        return None

    # Dict format (hierarchical training)
    if isinstance(history, dict):
        return history.get('mu_tracker', None)

    # Object format (standard training)
    if hasattr(history, 'mu_tracker'):
        return history.mu_tracker

    return None


def filter_history_steps(history, skip_initial_steps=0):
    """
    Filter history to skip initial transient steps.

    Args:
        history: Either dict or TrainingHistory object
        skip_initial_steps: Number of initial steps to skip

    Returns:
        Filtered history in same format as input
    """
    if history is None or skip_initial_steps <= 0:
        return history

    # Handle dict format (hierarchical training)
    if isinstance(history, dict):
        filtered = {}
        for key, value in history.items():
            if key == 'mu_tracker':
                # Special handling for mu_tracker
                filtered[key] = filter_mu_tracker(value, skip_initial_steps)
            elif key == 'emergence_events':
                # Filter emergence events by step
                filtered[key] = [e for e in value if e.get('step', 0) >= skip_initial_steps]
            elif isinstance(value, (list, np.ndarray)) and len(value) > 0:
                # Filter list/array data
                if len(value) > skip_initial_steps:
                    filtered[key] = value[skip_initial_steps:]
                else:
                    filtered[key] = []
            else:
                # Keep non-sequence data as-is
                filtered[key] = value
        return filtered

    # Handle TrainingHistory object
    if hasattr(history, 'steps'):
        filtered = deepcopy(history)

        # Filter all list attributes
        for attr in ['steps', 'total_energy', 'self_energy', 'belief_align',
                     'prior_align', 'observations', 'grad_norm_mu_q',
                     'grad_norm_Sigma_q', 'grad_norm_phi']:
            if hasattr(filtered, attr):
                value = getattr(filtered, attr)
                if isinstance(value, list) and len(value) > skip_initial_steps:
                    setattr(filtered, attr, value[skip_initial_steps:])

        # Filter mu_tracker
        if hasattr(filtered, 'mu_tracker') and filtered.mu_tracker is not None:
            filtered.mu_tracker = filter_mu_tracker(filtered.mu_tracker, skip_initial_steps)

        return filtered

    return history


def filter_mu_tracker(tracker, skip_initial_steps):
    """Filter MuCenterTracking data to skip initial steps."""
    if tracker is None or skip_initial_steps <= 0:
        return tracker

    if not hasattr(tracker, 'steps') or len(tracker.steps) <= skip_initial_steps:
        return tracker

    filtered = deepcopy(tracker)

    # Filter steps
    filtered.steps = tracker.steps[skip_initial_steps:]

    # Filter per-agent data
    if hasattr(tracker, 'mu_components'):
        filtered.mu_components = [
            agent_data[skip_initial_steps:] if len(agent_data) > skip_initial_steps else []
            for agent_data in tracker.mu_components
        ]

    if hasattr(tracker, 'mu_norms'):
        filtered.mu_norms = [
            agent_data[skip_initial_steps:] if len(agent_data) > skip_initial_steps else []
            for agent_data in tracker.mu_norms
        ]

    return filtered


def normalize_history(history):
    """
    Convert TrainingHistory object to dict format for plotting.

    This allows plot functions to work with both pkl and npz formats.
    """
    if history is None:
        return None

    # If it's already a dict, return as-is
    if isinstance(history, dict):
        return history

    # If it's a TrainingHistory object, convert to dict
    if hasattr(history, 'steps'):
        hist_dict = {
            "step": history.steps if hasattr(history, 'steps') else [],
            "total": history.total_energy if hasattr(history, 'total_energy') else [],
            "self": history.self_energy if hasattr(history, 'self_energy') else [],
            "belief_align": history.belief_align if hasattr(history, 'belief_align') else [],
            "prior_align": history.prior_align if hasattr(history, 'prior_align') else [],
            "observations": history.observations if hasattr(history, 'observations') else [],
            "grad_norm_mu_q": history.grad_norm_mu_q if hasattr(history, 'grad_norm_mu_q') else [],
            "grad_norm_Sigma_q": history.grad_norm_Sigma_q if hasattr(history, 'grad_norm_Sigma_q') else [],
            "grad_norm_phi": history.grad_norm_phi if hasattr(history, 'grad_norm_phi') else [],
        }
        return hist_dict

    return None


def load_system(run_dir: Path):
    """Load final MultiAgentSystem from pickle."""
    state_path = run_dir / "final_state.pkl"
    if not state_path.exists():
        print("⚠ No final_state.pkl found in run directory.")
        return None

    with open(state_path, "rb") as f:
        system = pickle.load(f)
    print(f"✓ Loaded final system from {state_path}")
    return system