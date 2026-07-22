"""De-periodicisation measures from generator ground truth (ST2 task 20.5).

The ST2 frontier compares attack families on a SHARED PHYSICAL x-axis, not by
knob names (plan-for-paper-2 §5): how de-periodicised are the iteration
boundaries actually? Two measures, both computed from the ground-truth
boundary times ``iter_starts`` returned by
:func:`code.ko_workload.training_F_meta` (no estimation step, so the x-axis
carries no detector error):

- :func:`cadence_cv` — coefficient of variation of the iteration period, the
  dimensionless spread of the cadence;
- :func:`phase_diffusion_coeff` — the Brownian phase-diffusion coefficient D
  [rad^2/s] with Var[phi(t)] ~ D * t, which separates ACCUMULATING boundary
  randomness (i.i.d. period jitter, phase slip — both random-walk the
  boundaries) from non-accumulating spread.

Analytic anchor (tested): i.i.d. fractional period jitter sigma at cadence f0
gives CV ~ sigma and D ~ (2*pi)^2 * sigma^2 * f0.
"""

from __future__ import annotations

import numpy as np


def cadence_cv(iter_starts: np.ndarray) -> float:
    """Coefficient of variation of the iteration period: std/mean of
    ``diff(iter_starts)``. 0 for a perfectly periodic generator."""
    periods = np.diff(np.asarray(iter_starts, dtype=float))
    if periods.size < 2:
        raise ValueError("need at least 3 boundary times for a cadence CV")
    return float(periods.std() / periods.mean())


def phase_diffusion_coeff(
    iter_starts: np.ndarray,
    max_lag_frac: float = 0.1,
    n_lags: int = 10,
) -> float:
    """Phase-diffusion coefficient D [rad^2/s], Var[phi(t)] ~ D * t.

    Boundary i carries phase ``theta_i = 2*pi*i`` at time ``t_i``. The phases
    are detrended against the best-fit mean rate (least squares of theta on
    t), leaving the residual phase path ``r_i``; for Brownian boundary
    diffusion the residual's increment variance grows linearly with elapsed
    time, ``Var[r_{i+k} - r_i] ~ D * k * T_bar``. D is recovered by a
    through-origin regression of the windowed increment variances against
    elapsed time over ``n_lags`` log-spaced lags up to ``max_lag_frac`` of the
    series (lags << N keep the detrending bias negligible).

    Deterministic: a pure function of ``iter_starts``.
    """
    t_i = np.asarray(iter_starts, dtype=float)
    n = t_i.size
    if n < 20:
        raise ValueError("need at least 20 boundary times for a diffusion fit")
    theta = 2.0 * np.pi * np.arange(n)
    slope, intercept = np.polyfit(t_i, theta, 1)
    resid = theta - (intercept + slope * t_i)
    t_bar = (t_i[-1] - t_i[0]) / (n - 1)

    k_max = max(2, int(round(max_lag_frac * n)))
    lags = np.unique(np.round(np.geomspace(1, k_max, n_lags)).astype(int))
    elapsed = lags * t_bar
    growth = np.array([np.var(resid[k:] - resid[:-k]) for k in lags])
    # through-origin least squares: Var = D * elapsed
    return float(growth @ elapsed / (elapsed @ elapsed))
