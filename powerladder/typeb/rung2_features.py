"""Rung-2 physics feature vector (plan §3.3): the Rung-1 quantities as one
fixed interpretable vector, computed from ``(t, P_obs)`` ONLY.

Rung 2 turns Rung-1 *structure* into a training-vs-inference decision. Its
features are the same physics statistics Rung 1 already computes — harmonic
energy, cyclic correlation, phase-folded repeatability, tracker stability — so
the classifier is interpretable and the reader can see which physics quantity
carries the discrimination. Every feature is oriented so **larger = more
training-like**, so the prespecified rule (:mod:`code.typeb.rung2`) can sum them
with unit weights and any fitted rule sees a consistent sign convention.

**Estimated-from-trace only (the central correctness constraint).** A deployable
Rung-2 classifier sees only the observed power trace; it must NOT read generator
internals (``training_F_meta``'s iteration boundaries). The ground-truth
de-periodicisation measures (:mod:`code.typeb.deperiod`) are the FRONTIER's
x-axis, never a feature here. The two timing features below are estimated from
the Viterbi tracker's recovered frequency path (:func:`code.typeb.detectors.
viterbi_best_path`), exactly as the ST1 adaptive stages estimate the warp:

- ``path_freq_stability`` — smoothness of the recovered best-path frequency
  (a coherent iteration line, even a slowly drifting one, moves smoothly frame
  to frame; the inference null has no line, so its best path meanders);
- ``fold_repeatability`` — fold the trace on the estimated phase and correlate
  each recovered cycle against the mean waveform (a repeating training iteration
  folds to a stable template; a diffuse null does not).

The other six features reuse the existing scalar detectors unchanged:
``mtf``/``mtf_comb`` (Thomson multitaper harmonic F-test and its comb),
``spectral`` (matched-filter peak-to-background), ``viterbi`` (tracked line
energy), ``dg_fixed`` (Dandawaté–Giannakis at the nominal cadence — the fixed
comparator), ``dg_order_full`` (the fully adaptive tracked-cyclostationary
statistic, the ST1 survivor).
"""

from __future__ import annotations

import numpy as np

from ..config import DEFAULT, KoTypeBParams, St1DetectorParams
from ..st1.multitaper import comb_f_statistic, f_test_statistic
from ..st1.pipeline import (KO_NOMINAL_F0_HZ, stage1_fixed_alpha,
                            stage4_full_adaptive)
from ..st1.resample import angle_resample, phase_from_path
from .detectors import spectral_statistic, viterbi_best_path, viterbi_statistic

# Feature order is the contract between this module, the physics score (rule i,
# which sums in this order) and the fitted rules' column layout. Every feature
# is oriented larger = more training-like.
FEATURE_NAMES: tuple[str, ...] = (
    "mtf", "mtf_comb", "spectral", "viterbi",
    "dg_fixed", "dg_order_full", "path_freq_stability", "fold_repeatability",
)


def _path_features(
    t: np.ndarray,
    p_obs: np.ndarray,
    band_lo: float,
    band_hi: float,
    params: St1DetectorParams,
) -> tuple[float, float]:
    """Estimated tracker-stability + phase-fold repeatability from ONE Viterbi run.

    Shares the single :func:`viterbi_best_path` call (the tracked line the ST1
    stages also ride) between both timing features. Returns
    ``(path_freq_stability, fold_repeatability)``, both in ``[0, 1]``, both
    ``0.0`` on a degenerate trace (no trackable path / too few folded cycles).
    """
    _, times, _, path_freqs = viterbi_best_path(t, p_obs, band_lo, band_hi)
    if path_freqs.size < 2:
        return 0.0, 0.0

    # (1) stability: smoothness of the recovered frequency path. Mean absolute
    # frame-to-frame step relative to the mean frequency; a coherent (even
    # drifting) line steps smoothly, a meandering null does not. Mapped to
    # (0, 1] by 1/(1+step) so larger = steadier = more training-like.
    mean_f = float(np.mean(path_freqs))
    if mean_f <= 0.0:
        stability = 0.0
    else:
        step = float(np.mean(np.abs(np.diff(path_freqs)))) / mean_f
        stability = 1.0 / (1.0 + step)

    # (2) fold repeatability: fold on the estimated phase and correlate each
    # recovered cycle against the mean waveform. Guarded against a too-short
    # or non-monotone path (a flat/degenerate load).
    fold = 0.0
    try:
        _, theta = phase_from_path(times, path_freqs, t,
                                   floor_hz=params.f_path_floor_hz)
        spc = params.samples_per_cycle
        _, x_ang = angle_resample(t, p_obs, theta,
                                  samples_per_cycle=spc,
                                  interp=params.resample_interp)
        n_cyc = x_ang.size // spc
        if n_cyc >= 2:
            W = x_ang[:n_cyc * spc].reshape(n_cyc, spc)
            W = W - W.mean(axis=1, keepdims=True)     # per-cycle DC out
            template = W.mean(axis=0)
            tnorm = float(np.linalg.norm(template))
            if tnorm > 0.0:
                corrs = []
                for k in range(n_cyc):
                    wn = float(np.linalg.norm(W[k]))
                    if wn > 0.0:
                        corrs.append(float(W[k] @ template) / (wn * tnorm))
                if corrs:
                    fold = max(0.0, float(np.mean(corrs)))
    except ValueError:
        fold = 0.0                                     # non-monotone phase guard

    return stability, fold


def feature_vector(
    t: np.ndarray,
    p_obs: np.ndarray,
    band_lo: float,
    band_hi: float,
    *,
    params: St1DetectorParams = DEFAULT.st1,
) -> tuple[np.ndarray, list[str]]:
    """The fixed physics feature vector of one trace, in :data:`FEATURE_NAMES` order.

    Consumes only ``(t, P_obs)`` and the band — never generator internals. The
    ``dg_fixed`` feature tests the nominal Ko cadence (the fixed-verifier
    comparator); ``dg_order_full`` is the adaptive tracked statistic. Returns
    ``(vec, names)`` with ``vec`` a length-8 float array.
    """
    mtf = f_test_statistic(t, p_obs, band_lo, band_hi, params=params)
    comb = comb_f_statistic(t, p_obs, band_lo, band_hi, params=params)
    spec = spectral_statistic(t, p_obs, band_lo, band_hi)
    vit = viterbi_statistic(t, p_obs, band_lo, band_hi)
    dg_fixed = stage1_fixed_alpha(t, p_obs, band_lo, band_hi,
                                  params=params, alpha_hz=KO_NOMINAL_F0_HZ)
    dg_full = stage4_full_adaptive(t, p_obs, band_lo, band_hi, params=params)
    stability, fold = _path_features(t, p_obs, band_lo, band_hi, params)
    vec = np.array([mtf, comb, spec, vit, dg_fixed, dg_full, stability, fold],
                   dtype=float)
    return vec, list(FEATURE_NAMES)


def feature_matrix(
    traces,
    glue: KoTypeBParams,
    *,
    params: St1DetectorParams = DEFAULT.st1,
) -> tuple[np.ndarray, list[str]]:
    """Feature matrix ``X[n_traces, 8]`` over a population, using the glue band.

    Each ``TypeTrace`` supplies only ``.t`` / ``.P_obs`` (the estimated-only
    contract). The search band comes from the ``KoTypeBParams`` glue, matching
    every other Rung-1/2 detector's operating band.
    """
    rows = [
        feature_vector(tr.t, tr.P_obs, glue.band_lo, glue.band_hi, params=params)[0]
        for tr in traces
    ]
    return np.asarray(rows, dtype=float), list(FEATURE_NAMES)
