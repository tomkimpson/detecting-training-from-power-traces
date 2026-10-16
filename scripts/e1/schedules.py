"""Micro-step schedules for the E1 learning-cost pilot (exploratory).

Each arm gives the number of gradient-accumulation micro-steps G_i taken before
optimiser step i. Honest training uses G = 8 (``BASE_ACCUM``, the same as
``powerladder.config.KoWorkloadParams.work_base_accum``), so the distributions here
are exactly those scored against the tracker in the E0 spike
(notes/results/e0-work-variation-spike-findings.md):

    shipped0.7          G = round(8 / max(1 + xi, 0.05)), xi ~ N(0, 0.7): the §6
                        attack, as in powerladder/ko_workload.py's work branch
    lognorm<s>          G = round(8 exp(s z - s^2/2)), capped at 160: a mean-8
                        attacker with lognormal spread s
    lognorm1.0_sqrtlr   the lognorm1.0 schedule with the learning rate scaled by
                        sqrt(G_i / 8), the attacker's best response

Rounding is stochastic with a floor of one micro-step, as on the hardware.
"""
from __future__ import annotations

import numpy as np

BASE_ACCUM = 8
FLOOR = 0.05                      # _MIN_JITTER_FACTOR in powerladder/ko_workload.py
CAP = BASE_ACCUM / FLOOR          # 160: the shipped form's ceiling, reused for lognormals

ARMS = ("honest", "shipped0.7", "lognorm0.8", "lognorm1.0", "lognorm1.0_sqrtlr")


def stochastic_round(g_real: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Round to an integer >= 1, up with probability equal to the fractional part."""
    lo = np.floor(g_real)
    g = lo + (rng.random(g_real.shape) < g_real - lo)
    return np.maximum(g, 1).astype(np.int64)


def _targets(arm: str, n: int, rng: np.random.Generator) -> np.ndarray:
    if arm == "honest":
        return np.full(n, float(BASE_ACCUM))
    if arm == "shipped0.7":
        xi = rng.normal(0.0, 0.7, n)
        return BASE_ACCUM / np.maximum(1.0 + xi, FLOOR)
    if arm.startswith("lognorm"):
        s = float(arm.removeprefix("lognorm").removesuffix("_sqrtlr"))
        z = rng.standard_normal(n)
        return np.minimum(BASE_ACCUM * np.exp(s * z - s * s / 2), CAP)
    raise ValueError(f"unknown arm {arm!r}; expected one of {ARMS}")


def make_schedule(arm: str, n: int, seed: int) -> np.ndarray:
    """``n`` micro-step counts for ``arm``. The best-response arm shares its
    base arm's schedule (same seed -> same G sequence)."""
    rng = np.random.default_rng([seed, 0xE1])
    return stochastic_round(_targets(arm, n, rng), rng)


def lr_multiplier(arm: str, G: np.ndarray) -> np.ndarray:
    """Per-step learning-rate multiplier: sqrt(G/8) for the best-response arm,
    else 1."""
    G = np.asarray(G, dtype=float)
    if arm.endswith("_sqrtlr"):
        return np.sqrt(G / BASE_ACCUM)
    return np.ones_like(G)
