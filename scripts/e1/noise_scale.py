"""McCandlish et al. (2018) "simple" gradient noise scale for the E1 pilot.

From one optimiser step's micro-batch gradients (each over ``b_small`` examples)
and their mean (over ``b_big`` examples):

    |G|^2 ~ (b_big |G_big|^2 - b_small |G_small|^2) / (b_big - b_small)
    S     ~ (|G_small|^2 - |G_big|^2) / (1/b_small - 1/b_big)

Both are unbiased but noisy per step, so B_simple = S/|G|^2 is the ratio of
their averages over many probes, as the paper recommends.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def noise_scale_terms(small_sq: float, big_sq: float,
                      b_small: float, b_big: float) -> tuple[float, float]:
    """(|G|^2, S) estimates from one probe.

    ``small_sq`` is the mean squared norm of the micro-batch gradients;
    ``big_sq`` is the squared norm of their mean.
    """
    if b_big == b_small:
        raise ValueError("b_big must differ from b_small")
    g2 = (b_big * big_sq - b_small * small_sq) / (b_big - b_small)
    s = (small_sq - big_sq) / (1.0 / b_small - 1.0 / b_big)
    return g2, s


def b_simple(g2s: Sequence[float], ss: Sequence[float]) -> float:
    """Ratio of the averaged S and |G|^2 terms, in the units of the batch sizes."""
    return float(np.mean(ss) / np.mean(g2s))
