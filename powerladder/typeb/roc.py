"""ROC bookkeeping for the B0 decision gate — no extra dependencies.

The gate's metric (spec.md Sec. 4 B0; plan Sec. 3 B0) is the detection rate (TPR)
at a *fixed* false-alarm rate (FAR). Given detection statistics for the training
("positive") and inference ("negative") populations, these helpers build the ROC
and read off TPR at a chosen FAR, the single number the two detectors are compared
on. Larger statistic = more training-like (a positive call is ``score >= thr``).
"""

from __future__ import annotations

import numpy as np


def roc_points(
    pos: np.ndarray, neg: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (far, tpr, thresholds) tracing the ROC as the threshold sweeps.

    Thresholds are every distinct statistic value (plus +inf), so the curve is
    exact for the given samples. ``far``/``tpr`` are sorted by increasing FAR.
    """
    pos = np.asarray(pos, dtype=float)
    neg = np.asarray(neg, dtype=float)
    thr = np.unique(np.concatenate([pos, neg]))
    thr = np.concatenate([thr, [np.inf]])
    tpr = np.array([(pos >= s).mean() for s in thr])
    far = np.array([(neg >= s).mean() for s in thr])
    order = np.argsort(far)
    return far[order], tpr[order], thr[order]


def tpr_at_far(pos: np.ndarray, neg: np.ndarray, target_far: float) -> float:
    """Best detection rate achievable within the false-alarm budget ``target_far``.

    The operational reading: among all thresholds whose FAR <= ``target_far``,
    report the largest TPR. This is the ROC's value at ``target_far`` (its upper
    envelope), robust to the several operating points that can share one FAR.
    Returns 0.0 if no threshold meets the budget.
    """
    far, tpr, _ = roc_points(pos, neg)
    within = far <= target_far
    return float(tpr[within].max()) if within.any() else 0.0


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Area under the ROC = P(score_pos > score_neg) (the Mann-Whitney statistic).

    Ties count as half. 1.0 = perfect separation, 0.5 = chance.
    """
    pos = np.asarray(pos, dtype=float)
    neg = np.asarray(neg, dtype=float)
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    wins = (diff > 0).sum() + 0.5 * (diff == 0).sum()
    return float(wins / (pos.size * neg.size))
