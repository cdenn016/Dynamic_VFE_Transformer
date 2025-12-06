# -*- coding: utf-8 -*-
"""
Created on Fri Dec  5 20:30:32 2025

@author: chris and christine
"""

# -*- coding: utf-8 -*-
"""
Metric Signature Analysis
=========================

Analyze metric signatures to determine if a metric is:
- Riemannian (all positive eigenvalues)
- Lorentzian (one negative, rest positive)
- Indefinite (mixed signs)

Author: Chris
Date: November 2025
"""

import numpy as np
from enum import Enum
from dataclasses import dataclass
from typing import Tuple


class MetricSignature(Enum):
    """Classification of metric tensor signatures."""
    RIEMANNIAN = "riemannian"      # All positive eigenvalues (++...+)
    LORENTZIAN = "lorentzian"      # One negative, rest positive (-++...+)
    ANTI_LORENTZIAN = "anti_lorentzian"  # One positive, rest negative (+--...-)
    INDEFINITE = "indefinite"       # Mixed signature
    DEGENERATE = "degenerate"       # Has zero eigenvalues


@dataclass
class SignatureAnalysis:
    """Results of metric signature analysis."""
    signature: MetricSignature
    eigenvalues: np.ndarray
    positive_count: int
    negative_count: int
    zero_count: int

    @property
    def dimension(self) -> int:
        return len(self.eigenvalues)

    @property
    def signature_tuple(self) -> Tuple[int, int]:
        """Return (p, q) where p = positive eigenvalues, q = negative."""
        return (self.positive_count, self.negative_count)


def analyze_metric_signature(
    G: np.ndarray,
    tol: float = 1e-10
) -> SignatureAnalysis:
    """
    Analyze the signature of a metric tensor.

    Args:
        G: Metric tensor (symmetric matrix)
        tol: Tolerance for zero eigenvalues

    Returns:
        SignatureAnalysis with signature classification
    """
    # Compute eigenvalues
    eigenvalues = np.linalg.eigvalsh(G)

    # Count signs
    positive_count = np.sum(eigenvalues > tol)
    negative_count = np.sum(eigenvalues < -tol)
    zero_count = np.sum(np.abs(eigenvalues) <= tol)

    n = len(eigenvalues)

    # Classify signature
    if zero_count > 0:
        signature = MetricSignature.DEGENERATE
    elif positive_count == n:
        signature = MetricSignature.RIEMANNIAN
    elif negative_count == 1 and positive_count == n - 1:
        signature = MetricSignature.LORENTZIAN
    elif positive_count == 1 and negative_count == n - 1:
        signature = MetricSignature.ANTI_LORENTZIAN
    else:
        signature = MetricSignature.INDEFINITE

    return SignatureAnalysis(
        signature=signature,
        eigenvalues=eigenvalues,
        positive_count=positive_count,
        negative_count=negative_count,
        zero_count=zero_count
    )