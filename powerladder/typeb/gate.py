"""The B0 decision gate: run both detectors over a frequency-wander sweep.

One place computes the comparison the gate turns on (spec.md Sec. 4 B0; plan
Sec. 3 B0), so the figure script and the test suite agree by construction. For
each wander level it scores a training and an inference population with each
detector and records AUC and TPR at a fixed FAR. The verdict: keep the
HMM/Viterbi tracker iff it beats the spectral baseline on this metric, and the
plan's prediction is that it does so precisely as f0 wanders.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import TypeBParams
from .detectors import spectral_statistic, viterbi_statistic
from .roc import auc, roc_points, tpr_at_far
from .synth import make_population

# The two detectors under test (name -> statistic on a single (t, P_obs) trace).
DETECTORS = {
    "spectral": spectral_statistic,
    "viterbi": viterbi_statistic,
}


def score_population(traces, detector_fn, p: TypeBParams) -> np.ndarray:
    """Detection statistic for every trace under one detector."""
    return np.array(
        [detector_fn(tr.t, tr.P_obs, p.band_lo, p.band_hi) for tr in traces]
    )


@dataclass(frozen=True)
class GatePoint:
    """One wander level: per-detector AUC and TPR-at-fixed-FAR."""

    wander_hz: float
    auc: dict[str, float]
    tpr_at_far: dict[str, float]


def evaluate(
    p: TypeBParams,
    wanders: np.ndarray,
    *,
    n_each: int = 100,
    target_far: float = 0.05,
    seed: int | None = None,
    population_fn=make_population,
) -> list[GatePoint]:
    """Run the gate over ``wanders``; one GatePoint per wander level.

    Each wander level uses a fresh RNG seeded off ``p.seed`` (or ``seed``) plus the
    level index, so the populations are reproducible and independent across levels.

    ``population_fn(n_each, p, rng, wander_hz=...) -> (train, infer)`` is the trace
    source; it defaults to the B0-local :func:`code.typeb.synth.make_population`.
    B1 (task 3) passes a wrapper over :func:`code.typeb.ko_synth.ko_make_population`
    to re-run the same comparison on the faithful Ko generator -- the swept
    ``wander_hz`` then means Ko's ``f0_drift_hz`` (the same physical wander axis,
    b0-state-space §6). ``p`` need only expose ``band_lo``/``band_hi`` (the gate's
    ``score_population`` reads nothing else off it), so ``KoTypeBParams`` works here.
    """
    base = p.seed if seed is None else seed
    points: list[GatePoint] = []
    for i, w in enumerate(wanders):
        rng = np.random.default_rng(base + i)
        train, infer = population_fn(n_each, p, rng, wander_hz=float(w))
        auc_d, tpr_d = {}, {}
        for name, fn in DETECTORS.items():
            pos = score_population(train, fn, p)
            neg = score_population(infer, fn, p)
            auc_d[name] = auc(pos, neg)
            tpr_d[name] = tpr_at_far(pos, neg, target_far)
        points.append(GatePoint(wander_hz=float(w), auc=auc_d, tpr_at_far=tpr_d))
    return points


def roc_at(
    p: TypeBParams,
    wander_hz: float,
    *,
    n_each: int = 100,
    seed: int | None = None,
    population_fn=make_population,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Per-detector (far, tpr) ROC curves at a single wander level (for plotting).

    ``population_fn`` is the trace source (see :func:`evaluate`); B1 passes the Ko
    builder to draw the ROC on the faithful generator.
    """
    base = p.seed if seed is None else seed
    rng = np.random.default_rng(base)
    train, infer = population_fn(n_each, p, rng, wander_hz=wander_hz)
    out = {}
    for name, fn in DETECTORS.items():
        pos = score_population(train, fn, p)
        neg = score_population(infer, fn, p)
        far, tpr, _ = roc_points(pos, neg)
        out[name] = (far, tpr)
    return out
