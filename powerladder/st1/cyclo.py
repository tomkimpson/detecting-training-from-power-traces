"""Dandawate-Giannakis cyclostationary statistic (ST1 task 19.3; memo Sec. 3.2).

A repeated iteration schedule modulates second-order structure (variance,
autocorrelation) at the cycle frequency alpha even when the first-order
spectral line is shallow. For a candidate alpha, the cyclic correlations

    R_x^alpha(tau) = (1/N) sum_n x[n] x[n+tau] exp(-i 2 pi alpha t_n)

are estimated over a prespecified lag set, their real and imaginary parts
stacked into r_hat (2L-dim), the long-run covariance Sigma_hat of the stacked
per-sample series estimated from the same record, and

    Q = N r_hat^T Sigma_hat^{-1} r_hat  ->  chi2(2L)   under the null,

asymptotically, given weak dependence, consistent covariance estimation, and
regularity (Dandawate & Giannakis 1994). Whether that asymptotic null HOLDS at
operationally small levels on finite traces is exactly what the ST1 FAR
harness measures — nothing here assumes it does.

Two long-run covariance estimators are implemented (an open choice the gate
sweeps, not decides ex ante): Newey-West/Bartlett HAC and batch means. Both
are followed by symmetrisation, diagonal shrinkage, and a positive-definite
solve with an eigenvalue-floor fallback. Conditioning failures are COUNTED
(module-level counter + per-call flag) and surfaced in the harness summary —
never silently patched, because a quietly regularised Sigma changes the null.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from scipy import linalg
from scipy.stats import chi2

from ..config import St1DetectorParams

_LN10 = np.log(10.0)

# Eigenvalue floor for the fallback solve, relative to the largest eigenvalue.
_EIG_FLOOR_REL = 1e-10

# Module-level conditioning-failure counter. The FAR harness resets it before
# a cell and reads it after (each cell runs wholly inside one worker process,
# so the counter is cell-local under the pool-over-cells design). The per-call
# flag on DgResult carries the same information for single-trace callers.
_COV_FAILURES = 0


def reset_cov_failures() -> None:
    global _COV_FAILURES
    _COV_FAILURES = 0


def cov_failure_count() -> int:
    return _COV_FAILURES


def _z_series(
    x: np.ndarray, fs: float, alpha_hz: float, tau: int
) -> np.ndarray:
    """z_tau[n] = x[n] * x[n+tau] * exp(-i 2 pi alpha t_n), x demeaned.

    Length N - tau; t_n = n / fs. The cyclic correlation is its mean.
    """
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    n_eff = x.size - tau
    t_n = np.arange(n_eff) / fs
    return x[:n_eff] * x[tau:tau + n_eff] * np.exp(-2j * np.pi * alpha_hz * t_n)


def _stacked_z(
    x: np.ndarray, fs: float, alpha_hz: float, lags: tuple[int, ...]
) -> np.ndarray:
    """Per-sample stacked [Re; Im] series, aligned to length N - max(lags).

    Returns W of shape (n_eff, 2L): column j < L is Re z_{lags[j]}, column
    L + j is Im z_{lags[j]}. All lags share one alignment so the long-run
    covariance is estimated on a common time index.
    """
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    n_eff = x.size - max(lags)
    t_n = np.arange(n_eff) / fs
    e = np.exp(-2j * np.pi * alpha_hz * t_n)
    Z = np.empty((n_eff, len(lags)), dtype=complex)
    for j, tau in enumerate(lags):
        Z[:, j] = x[:n_eff] * x[tau:tau + n_eff] * e
    return np.concatenate([Z.real, Z.imag], axis=1)


def cyclic_corr(
    x: np.ndarray, fs: float, alpha_hz: float, lags: tuple[int, ...]
) -> np.ndarray:
    """Cyclic correlation estimates R_hat_x^alpha(tau) over the lag set (complex, L)."""
    W = _stacked_z(x, fs, alpha_hz, lags)
    m = W.mean(axis=0)
    L = len(lags)
    return m[:L] + 1j * m[L:]


def longrun_cov(
    W: np.ndarray,
    estimator: str,
    bandwidth_frac: float,
    shrinkage: float,
    bandwidth_b: int = 0,
) -> np.ndarray:
    """Long-run covariance of the sample mean of W (n, d), symmetrised + shrunk.

    "bartlett":     Newey-West HAC on the demeaned series with weights
                    w_j = 1 - j/(b+1); b = ``bandwidth_b`` when > 0 (the
                    explicit stage-8 sweep knob), else floor(n**(1/3)) when
                    ``bandwidth_frac == 0``, else floor(bandwidth_frac * n).
    "batch_means":  ~sqrt(n) non-overlapping batches; covariance of the batch
                    means times the batch length (bandwidth args unused).

    Both end with Sigma <- (Sigma + Sigma^T)/2, then diagonal shrinkage
    Sigma <- (1 - l) Sigma + l diag(Sigma).
    """
    W = np.asarray(W, dtype=float)
    n = W.shape[0]
    Wd = W - W.mean(axis=0, keepdims=True)

    if estimator == "bartlett":
        if bandwidth_b > 0:
            b = min(int(bandwidth_b), n - 1)
        elif bandwidth_frac == 0.0:
            b = int(np.floor(n ** (1.0 / 3.0)))
        else:
            b = max(1, int(np.floor(bandwidth_frac * n)))
        sigma = Wd.T @ Wd / n
        for j in range(1, b + 1):
            gamma_j = Wd[j:].T @ Wd[:-j] / n
            w_j = 1.0 - j / (b + 1.0)
            sigma = sigma + w_j * (gamma_j + gamma_j.T)
    elif estimator == "batch_means":
        n_batch = max(2, int(np.floor(np.sqrt(n))))
        ell = n // n_batch
        means = Wd[: n_batch * ell].reshape(n_batch, ell, -1).mean(axis=1)
        sigma = ell * np.cov(means, rowvar=False, ddof=1)
        sigma = np.atleast_2d(sigma)
    else:
        raise ValueError(f"unknown cov estimator {estimator!r} "
                         "(expected 'bartlett' or 'batch_means')")

    sigma = 0.5 * (sigma + sigma.T)
    d = np.diag(np.diag(sigma))
    return (1.0 - shrinkage) * sigma + shrinkage * d


def _solve_pos_def(sigma: np.ndarray, r: np.ndarray) -> tuple[np.ndarray, bool]:
    """Solve Sigma s = r assuming positive-definiteness; eigenvalue-floor fallback.

    Returns (s, failed). On a Cholesky failure (non-pos-def / singular Sigma)
    the fallback floors the eigenvalues at _EIG_FLOOR_REL times the largest
    and increments the module failure counter — the failure is surfaced, and
    the returned Q is then a diagnostic value, not a trusted chi2 draw.
    """
    global _COV_FAILURES
    try:
        s = linalg.solve(sigma, r, assume_a="pos")
        if not np.all(np.isfinite(s)):
            raise linalg.LinAlgError("non-finite solution")
        return s, False
    except (linalg.LinAlgError, ValueError):
        _COV_FAILURES += 1
        lam, V = np.linalg.eigh(sigma)
        floor = max(float(lam.max()), 0.0) * _EIG_FLOOR_REL + 1e-300
        lam = np.clip(lam, floor, None)
        return V @ ((V.T @ r) / lam), True


class DgResult(NamedTuple):
    """One fixed-alpha DG evaluation: the statistic, its asymptotic p, the flag."""

    Q: float
    p: float           # chi2.sf(Q, 2L) — the ASYMPTOTIC p; validity is ST1's question
    cov_failed: bool   # True if the pos-def solve fell back to the eigenvalue floor


def dg_q(
    x: np.ndarray,
    fs: float,
    alpha_hz: float,
    lags: tuple[int, ...],
    *,
    params: St1DetectorParams,
    mask: np.ndarray | None = None,
) -> DgResult:
    """DG statistic Q = N r_hat^T Sigma_hat^{-1} r_hat at one fixed alpha.

    Covariance estimator / bandwidth / shrinkage come from ``params``
    (cov_estimator, cov_bandwidth_frac, cov_shrinkage). Under the stationary
    weak-dependence null, Q -> chi2(2L) asymptotically.

    ``mask`` (boolean, length N over samples of x) restricts the estimate to
    rows whose EVERY constituent sample is masked-in: row n is retained iff
    mask[n + tau] for all tau in the lag set (0 included). N in Q is the
    retained count. The stage-3 held-out design uses this to score ONLY
    samples mapping back into TEST blocks. The long-run covariance treats the
    retained rows as contiguous — valid when the dropped gaps exceed the
    mixing length (the split guard assumption, stated at the call site).
    Demeaning stays global (the sample mean is a warp-independent scalar).
    If fewer than 4L rows survive, the trace is degenerate: counted as a
    conditioning failure and scored Q = 0 (p = 1), never a silent solve.
    """
    W = _stacked_z(x, fs, alpha_hz, lags)
    n_eff = W.shape[0]
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        valid = np.ones(n_eff, dtype=bool)
        for tau in set(lags) | {0}:
            valid &= mask[tau:tau + n_eff]
        W = W[valid]
        n_eff = W.shape[0]
        if n_eff < 2 * W.shape[1]:            # fewer rows than 2*(2L) dims
            global _COV_FAILURES
            _COV_FAILURES += 1
            return DgResult(Q=0.0, p=1.0, cov_failed=True)
    r = W.mean(axis=0)
    sigma = longrun_cov(W, params.cov_estimator, params.cov_bandwidth_frac,
                        params.cov_shrinkage,
                        bandwidth_b=getattr(params, "cov_bandwidth_b", 0))
    s, failed = _solve_pos_def(sigma, r)
    q = float(n_eff * r @ s)
    q = max(q, 0.0)                       # guard tiny negative round-off
    return DgResult(Q=q, p=float(chi2.sf(q, 2 * len(lags))), cov_failed=failed)


class DgMaxResult(NamedTuple):
    """DG scan over an alpha grid with Bonferroni bookkeeping."""

    Q_max: float
    alpha_best: float
    p_corr: float          # min(1, n_alphas * min_p) — Bonferroni over the grid
    n_alphas: int
    cov_failed_any: bool


def dg_q_max(
    x: np.ndarray,
    fs: float,
    alphas: np.ndarray,
    lags: tuple[int, ...],
    *,
    params: St1DetectorParams,
) -> DgMaxResult:
    """Max-Q over a candidate alpha grid, Bonferroni-corrected over grid size.

    The grid search is the LOOK-ELSEWHERE the staged ST1 design quantifies:
    after phase resampling (task 19.5) the tested alpha is fixed by
    construction and the search lives in the tracker instead — this helper
    exists for the fixed-time-domain comparison arm.
    """
    best = DgResult(Q=-np.inf, p=1.0, cov_failed=False)
    alpha_best = float(alphas[0])
    failed_any = False
    for a in np.asarray(alphas, dtype=float):
        res = dg_q(x, fs, a, lags, params=params)
        failed_any = failed_any or res.cov_failed
        if res.Q > best.Q:
            best = res
            alpha_best = float(a)
    p_corr = min(1.0, len(alphas) * best.p)
    return DgMaxResult(Q_max=best.Q, alpha_best=alpha_best, p_corr=p_corr,
                       n_alphas=len(alphas), cov_failed_any=failed_any)


def neg_log10_p(q: float, dof: int) -> float:
    """-log10 chi2.sf(q, dof) via logsf (finite even when sf underflows)."""
    return float(-chi2.logsf(q, dof) / _LN10)
