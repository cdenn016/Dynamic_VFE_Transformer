"""
Analysis Module
===============

Comprehensive analysis and visualization toolkit for multi-agent training runs.

Modules:
    - core: Data loading and preprocessing utilities
    - plots: Specialized plotting functions (energy, fields, mu tracking, etc.)
"""

from . import core
from . import plots

__all__ = ['core', 'plots']