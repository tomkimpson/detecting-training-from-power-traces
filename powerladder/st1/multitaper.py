"""Thomson multitaper harmonic F-test (ST1 task 19.2; methods memo Sec. 3.1).

The first structural detector: orthogonal DPSS tapers give K approximately
independent views of the trace; a coherent sinusoid contributes consistently
across tapers while the local stochastic background lands in the residual.
At a FIXED frequency f the resulting F statistic is pivotal under a
stationary, locally-smooth spectral null:

    mu(f) = sum_k U_k y_k(f) / sum_k U_k^2          (line-amplitude estimate)
    F(f)  = (K-1) |mu(f)|^2 sum_k U_k^2
            / sum_k |y_k(f) - mu(f) U_k|^2   ~  F(2, 2K-2)

with y_k the k-th DPSS eigencoefficient and U_k = sum_n v_k[n] (the taper DC
gain — odd tapers have U_k = 0 and contribute only to the residual).

Searching the band breaks pivotality: the detectors below correct the band
max-F by Bonferroni over N_eff = (band_hi - band_lo) * duration Rayleigh
resolutions. The a-priori argument says that OVER-corrects (eigencoefficients
are correlated over the taper bandwidth 2W = 2*NW/duration, several Rayleigh
widths wide) — but the F RATIO fluctuates on a finer scale than its
ingredients, and the first M=10^4 calibration table (task 19.4,
results/st1/far_summary.json) measured the corrected statistic as
ANTI-conservative by ~3.6-6x on the zero-padded grid: the padded-grid max
behaves like ~5x more effective tests than the Rayleigh count. The analytic
p-values here are therefore NOT trustworthy at the band level in either
direction; the FAR harness (code.st1.far) provides the Monte-Carlo-calibrated
alternative for free, and that is the operative calibration for these
detectors.

Both detectors return a corrected -log10 p under the shared gate signature
``fn(t, p_obs, band_lo, band_hi, *, params) -> float`` (larger = more line-like).
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.signal.windows import dpss
from scipy.stats import chi2, f as f_dist

from ..config import St1DetectorParams

_LN10 = np.log(10.0)


@lru_cache(maxsize=16)
def _dpss_cached(n: int, nw: float, n_tapers: int) -> np.ndarray:
    """DPSS tapers (K, n), unit-energy (scipy default: sum v_k^2 = 1). Cached —
    the FAR harness evaluates thousands of same-length traces per cell."""
    return dpss(n, nw, Kmax=n_tapers)


def _next_pow2(n: int) -> int:
    return 1 << (int(n - 1).bit_length())


def dpss_eigencoef(
    x: np.ndarray,
    fs: float,
    *,
    params: St1DetectorParams,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """DPSS eigencoefficients of a demeaned trace, one vectorised rfft.

    Returns ``(freqs, y, tapers)`` with ``y[k, i]`` the k-th eigencoefficient
    at ``freqs[i]`` on a grid zero-padded to ``n_fft_pad * next_pow2(n)``.
    The mean is removed first (the DC board-power level, exactly as
    code.typeb.detectors._detrend does for the existing detectors); the
    slow trend beyond the mean is left to the low-f bins outside the band.
    """
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    n = x.size
    tapers = _dpss_cached(n, params.nw, params.n_tapers)
    n_fft = params.n_fft_pad * _next_pow2(n)
    y = np.fft.rfft(tapers * x[None, :], n=n_fft, axis=1)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / fs)
    return freqs, y, tapers


def harmonic_f(y: np.ndarray, tapers: np.ndarray) -> np.ndarray:
    """Thomson harmonic F statistic at every grid frequency.

    ``y`` is (K, n_freq) eigencoefficients, ``tapers`` (K, n). Under the
    stationary locally-smooth null, each F(f) at FIXED f follows F(2, 2K-2).
    """
    U = tapers.sum(axis=1)                       # (K,) taper DC gains
    U2 = float(np.sum(U**2))
    K = y.shape[0]
    mu = (U @ y) / U2                            # (n_freq,) line amplitude
    resid = y - mu[None, :] * U[:, None]
    denom = np.sum(np.abs(resid) ** 2, axis=0)
    return (K - 1) * np.abs(mu) ** 2 * U2 / np.maximum(denom, 1e-300)


def _band_and_correction(
    t: np.ndarray, band_lo: float, band_hi: float
) -> tuple[float, float]:
    """(fs, log10 N_eff): Bonferroni count = Rayleigh resolutions in the band."""
    fs = 1.0 / (t[1] - t[0])
    duration = t[-1] - t[0]
    n_eff = max((band_hi - band_lo) * duration, 1.0)
    return fs, float(np.log10(n_eff))


def f_test_statistic(
    t: np.ndarray,
    p_obs: np.ndarray,
    band_lo: float,
    band_hi: float,
    *,
    params: St1DetectorParams,
) -> float:
    """Bonferroni-corrected -log10 p of the band max-F (the "mtf" detector).

    p_corr = min(1, N_eff * p_min) with N_eff = (band_hi - band_lo) * duration
    Rayleigh resolutions. The a-priori expectation was over-correction (the F
    grid is correlated over the 2W taper bandwidth); the measured M=10^4
    calibration table found the OPPOSITE — ~3.6-6x anti-conservative on the
    padded grid (see the module docstring). Treat this analytic tail as a
    ranking statistic; the FAR harness's Monte-Carlo calibration of this same
    statistic is the operative threshold (the harness gives it for free).
    """
    fs, log10_neff = _band_and_correction(t, band_lo, band_hi)
    freqs, y, tapers = dpss_eigencoef(p_obs, fs, params=params)
    F = harmonic_f(y, tapers)
    band = (freqs >= band_lo) & (freqs <= band_hi)
    if not np.any(band):
        return 0.0
    f_max = float(F[band].max())
    dof2 = 2 * params.n_tapers - 2
    log10_p = f_dist.logsf(f_max, 2, dof2) / _LN10
    return float(max(0.0, -log10_p - log10_neff))


def comb_f_statistic(
    t: np.ndarray,
    p_obs: np.ndarray,
    band_lo: float,
    band_hi: float,
    *,
    params: St1DetectorParams,
) -> float:
    """Harmonic-comb variant: Fisher combination over (f0, 2f0, ..., H*f0).

    For each in-band fundamental f0 on the padded grid, combine the per-
    harmonic F-test p-values by Fisher's method, T(f0) = -2 sum_h ln p_h ~
    chi2(2H) at fixed f0 (harmonics fall in disjoint taper bandwidths for
    in-band f0, so the p_h are treated as independent); maximise T over the
    f0 grid and Bonferroni-correct by the same N_eff Rayleigh count as
    :func:`f_test_statistic` (same caveat: conservative, MC-calibrated by
    the FAR harness). Harmonics beyond Nyquist are dropped; H comes from
    ``params.n_harmonics_comb``.
    """
    fs, log10_neff = _band_and_correction(t, band_lo, band_hi)
    freqs, y, tapers = dpss_eigencoef(p_obs, fs, params=params)
    F = harmonic_f(y, tapers)
    dof2 = 2 * params.n_tapers - 2

    in_band = np.flatnonzero((freqs >= band_lo) & (freqs <= band_hi))
    if in_band.size == 0:
        return 0.0
    f0s = freqs[in_band]
    nyq = freqs[-1]
    df = freqs[1] - freqs[0]

    log_p = f_dist.logsf(F, 2, dof2)             # natural-log p per grid bin
    T = np.zeros(in_band.size)
    n_harm = np.zeros(in_band.size, dtype=int)
    for h in range(1, params.n_harmonics_comb + 1):
        target = h * f0s
        ok = target <= nyq
        idx = np.clip(np.round(target / df).astype(int), 0, freqs.size - 1)
        T[ok] += -2.0 * log_p[idx[ok]]
        n_harm[ok] += 1

    # chi2(2H) tail per candidate fundamental (H varies where harmonics clip).
    log10_p = chi2.logsf(T, 2 * n_harm) / _LN10
    best = float(np.min(log10_p))
    return float(max(0.0, -best - log10_neff))
