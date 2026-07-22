"""Two Type detectors and the comparison they exist to settle.

Both answer "is a training iteration line present in this power trace?" and both
return a single scalar *detection statistic* (larger -> more training-like), so a
threshold sweep gives a ROC. They consume only ``(t, P_obs)`` arrays, so they are
agnostic to which generator produced the trace (the B0-local one here, or the
shared Ko et al. generator when it merges).

    spectral_statistic  — the yardstick. Welch PSD, then peak-to-background in the
                          f0 search band: a matched filter for a *narrow, fixed*
                          line. Best case for a stationary f0.

    viterbi_statistic   — CW-gravitational-wave-style line tracking. A spectrogram
                          (STFT) gives power vs (time, frequency); Viterbi finds the
                          highest-scoring frequency *path* through the band subject
                          to a slow-wander transition penalty. Follows a line that
                          drifts, where the fixed-bin matched filter loses SNR.

The decision gate (spec.md Sec. 4 B0): keep the HMM/Viterbi only if it beats the
spectral baseline on ROC at a fixed FAR. The plan (Sec. 3 B0) predicts it wins
*when f0 wanders* — which is simultaneously the honest-smearing regime and the
jitter-adversary regime, the strongest reason to keep it in scope.
"""

from __future__ import annotations

import numpy as np
from scipy import signal as sp_signal


def _detrend(p_obs: np.ndarray) -> np.ndarray:
    """Remove the DC board-power level so the line, not P_base, drives the PSD."""
    x = np.asarray(p_obs, dtype=float)
    return x - x.mean()


def spectral_statistic(
    t: np.ndarray,
    p_obs: np.ndarray,
    band_lo: float,
    band_hi: float,
    *,
    nperseg: int | None = None,
) -> float:
    """Matched-filter baseline: peak PSD in [band_lo, band_hi] / median PSD.

    The ratio is dimensionless and self-normalising (board power and broadband
    fluctuation cancel), so it is comparable across traces. A sharp stationary
    line gives a tall, narrow peak -> large statistic; a wandering line smears
    across bins -> the peak collapses, which is exactly where this baseline is
    expected to lose to the Viterbi tracker.
    """
    fs = 1.0 / (t[1] - t[0])
    x = _detrend(p_obs)
    if nperseg is None:
        nperseg = min(x.size, 1024)
    f, psd = sp_signal.welch(x, fs=fs, nperseg=nperseg)
    band = (f >= band_lo) & (f <= band_hi)
    if not np.any(band):
        return 0.0
    background = np.median(psd[psd > 0]) if np.any(psd > 0) else 1.0
    return float(psd[band].max() / background)


def _spectrogram_band(
    t: np.ndarray,
    p_obs: np.ndarray,
    band_lo: float,
    band_hi: float,
    nperseg: int,
    noverlap: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Log-power spectrogram restricted to the search band.

    Returns (freqs_in_band, frame_times, logP[freq, time]) with each time column
    normalised by its median over the band, so a slow overall power drift cannot
    masquerade as a tracked line and the score reflects line *contrast* against the
    local floor.
    """
    fs = 1.0 / (t[1] - t[0])
    x = _detrend(p_obs)
    f, tt, Sxx = sp_signal.spectrogram(
        x, fs=fs, nperseg=nperseg, noverlap=noverlap, mode="psd"
    )
    band = (f >= band_lo) & (f <= band_hi)
    fb = f[band]
    Pb = Sxx[band, :]
    col_floor = np.median(Pb, axis=0, keepdims=True)
    logP = np.log(Pb + 1e-30) - np.log(col_floor + 1e-30)
    return fb, tt, logP


def viterbi_path_scores(
    t: np.ndarray,
    p_obs: np.ndarray,
    band_lo: float,
    band_hi: float,
    *,
    nperseg: int | None = None,
    noverlap: int | None = None,
    jump_penalty: float = 1.0,
) -> np.ndarray:
    """Running Viterbi statistic after each spectrogram frame.

    Element ``j-1`` is the best cumulative path score over the first ``j``
    frames divided by ``j`` — the sequential view of ``viterbi_statistic``,
    whose forward recursion computes every prefix on the way to the final
    scalar (``out[-1]``). Frames are left-aligned at t=0, so ``out[j-1]``
    equals ``viterbi_statistic`` on a truncation covering exactly the first
    j frames (up to the full-trace DC detrend, which the band excludes).
    Consumed by the sequential verifier and as a diagnostic of how the
    statistic accumulates with observation time.
    """
    n = np.asarray(p_obs).size
    if nperseg is None:
        # ~16 s windows resolve a ~1 Hz line yet leave many frames to track over.
        fs = 1.0 / (t[1] - t[0])
        nperseg = int(min(n, max(64, round(16.0 * fs))))
    if noverlap is None:
        noverlap = nperseg // 2

    fb, _, logP = _spectrogram_band(t, p_obs, band_lo, band_hi, nperseg, noverlap)
    n_freq, n_time = logP.shape
    if n_freq == 0 or n_time == 0:
        return np.zeros(0)
    if n_freq == 1:
        return np.cumsum(logP[0]) / np.arange(1, n_time + 1)

    # Forward Viterbi over frequency-bin states.
    idx = np.arange(n_freq)
    score = logP[:, 0].copy()
    out = np.empty(n_time)
    out[0] = score.max()
    for j in range(1, n_time):
        # best[i] = max_k score[k] - penalty*|i-k|
        trans = score[None, :] - jump_penalty * np.abs(idx[:, None] - idx[None, :])
        score = logP[:, j] + trans.max(axis=1)
        out[j] = score.max() / (j + 1)
    return out


def viterbi_statistic(
    t: np.ndarray,
    p_obs: np.ndarray,
    band_lo: float,
    band_hi: float,
    *,
    nperseg: int | None = None,
    noverlap: int | None = None,
    jump_penalty: float = 1.0,
) -> float:
    """CW-style Viterbi line tracker: best mean log-power along a wandering path.

    Hidden state = frequency bin within the band; emission = column-normalised
    log-power; transition cost = ``jump_penalty`` per bin of frequency change,
    encoding the prior that a real iteration cadence wanders *slowly*. The returned
    statistic is the best path score averaged over time frames, so it is
    comparable across traces of different length. A drifting line keeps a high
    score (the path bends to follow it); broadband noise cannot, because no single
    slowly-varying path stays on top.
    """
    scores = viterbi_path_scores(t, p_obs, band_lo, band_hi, nperseg=nperseg,
                                 noverlap=noverlap, jump_penalty=jump_penalty)
    return float(scores[-1]) if scores.size else 0.0


def viterbi_best_path(
    t: np.ndarray,
    p_obs: np.ndarray,
    band_lo: float,
    band_hi: float,
    *,
    nperseg: int | None = None,
    noverlap: int | None = None,
    jump_penalty: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The Viterbi tracker's contrast map and the frequency path it follows.

    Returns ``(freqs, times, logP, path_freqs)``: the band-restricted,
    column-normalised log-contrast map ``logP[freq, time]`` that
    ``viterbi_statistic`` scores, and ``path_freqs`` (length ``times.size``), the
    frequency of the best-scoring path at each frame. Uses the same emission map,
    transition cost and window defaults as ``viterbi_statistic`` — the forward
    recursion here stores back-pointers and backtracks to recover the winning bin
    sequence — so the path is exactly the one the scored statistic rides. For
    visualisation (overlaying the tracked line on the spectrogram), not scoring.
    """
    n = np.asarray(p_obs).size
    if nperseg is None:
        fs = 1.0 / (t[1] - t[0])
        nperseg = int(min(n, max(64, round(16.0 * fs))))
    if noverlap is None:
        noverlap = nperseg // 2

    fb, times, logP = _spectrogram_band(t, p_obs, band_lo, band_hi, nperseg,
                                        noverlap)
    n_freq, n_time = logP.shape
    if n_freq == 0 or n_time == 0:
        return fb, times, logP, np.zeros(0)
    if n_freq == 1:
        return fb, times, logP, np.repeat(fb, n_time)

    idx = np.arange(n_freq)
    score = logP[:, 0].copy()
    back = np.empty((n_time, n_freq), dtype=int)
    back[0] = idx
    for j in range(1, n_time):
        # best[i] = max_k score[k] - penalty*|i-k|; remember the winning k.
        trans = score[None, :] - jump_penalty * np.abs(idx[:, None] - idx[None, :])
        back[j] = trans.argmax(axis=1)
        score = logP[:, j] + trans.max(axis=1)

    path = np.empty(n_time, dtype=int)
    path[-1] = int(score.argmax())
    for j in range(n_time - 1, 0, -1):
        path[j - 1] = back[j, path[j]]
    return fb, times, logP, fb[path]
