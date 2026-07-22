"""Sample splitting for the held-out-phase design (ST1 task 19.6; stage 3).

The stage-3 idea: break the estimate-then-test circularity of tacholess order
tracking by splitting the record into alternating EST / TEST blocks with
guard gaps. The frequency path is estimated ONLY from spectrogram frames
wholly supported inside EST blocks; the phase warp is interpolated across the
TEST blocks; the DG statistic is computed ONLY from angle samples that map
back into TEST blocks. Conditional on the EST samples the warp is a fixed
(non-random) time change of the TEST samples, so the fixed-order chi2 null
argument applies to the TEST-restricted statistic.

Three assumptions, stated not hidden (the FAR harness measures the MARGINAL
realised FAR over both EST and TEST randomness, not conditional-on-EST
validity):

    1. mixing length of the null process << guard_s, so EST and TEST blocks
       are effectively independent and the retained DG rows can be treated as
       contiguous by the long-run covariance;
    2. the true wander is smooth on the block scale (block_s), so a warp
       interpolated across a TEST block from flanking EST frames is not
       systematically misaligned;
    3. the warp is fixed given EST — no TEST sample influences the path
       (enforced here by the frame-support mask around the spectrogram).

Taper splitting (fundamental on one taper set, harmonics on another) is
deliberately NOT implemented: the memo (notes/power-verification-paths-
forward.md Sec. 3.4) flags it as non-equivalent — DPSS tapers overlap in
time, so taper-disjointness does not give sample-disjointness. Harmonic
splitting (fundamental -> estimate, harmonics -> test) is a comparison arm
deferred with the stage-8 sweeps.

The Viterbi machinery is duplicated (not imported wholesale) from
code.typeb.detectors — that module is the frozen B0/B1/B2 track; only the
private spectrogram helper is reused read-only.
"""

from __future__ import annotations

import numpy as np

from ..config import St1DetectorParams
from ..typeb.detectors import _spectrogram_band


def split_masks(
    n: int, fs: float, block_s: float, guard_s: float
) -> tuple[np.ndarray, np.ndarray]:
    """Alternating EST / TEST block masks with guards dropped at boundaries.

    Sample i (time t_i = i/fs) belongs to block b = floor(t_i / block_s);
    even blocks are EST, odd blocks are TEST. Samples within ``guard_s`` of
    ANY block edge (including the trace start, for a uniform rule) are
    dropped from both masks, so every EST/TEST transition is separated by a
    2*guard_s gap. Returns boolean ``(est_mask, test_mask)``: disjoint, and
    their union is strictly smaller than n whenever guard_s > 0.
    """
    t = np.arange(n) / fs
    block = np.floor(t / block_s).astype(int)
    pos = t - block * block_s
    core = (pos >= guard_s) & (pos < block_s - guard_s)
    est = core & (block % 2 == 0)
    test = core & (block % 2 == 1)
    return est, test


def _viterbi_path_bins(logP: np.ndarray, jump_penalty: float) -> np.ndarray:
    """Best path (bin indices per frame) through a log-contrast map.

    Same forward recursion / backtracking as typeb.detectors.viterbi_best_path
    — per-frame transition cost = jump_penalty per bin of frequency change.
    Frames may be non-uniformly spaced in time (the EST-only subsampling);
    the per-STEP penalty is kept constant regardless of the gap, a documented
    modelling choice (larger gaps could justify weaker penalties; the FAR
    harness calibrates whatever rule is used).
    """
    n_freq, n_time = logP.shape
    idx = np.arange(n_freq)
    if n_freq == 1:
        return np.zeros(n_time, dtype=int)
    score = logP[:, 0].copy()
    back = np.empty((n_time, n_freq), dtype=int)
    back[0] = idx
    for j in range(1, n_time):
        trans = score[None, :] - jump_penalty * np.abs(idx[:, None] - idx[None, :])
        back[j] = trans.argmax(axis=1)
        score = logP[:, j] + trans.max(axis=1)
    path = np.empty(n_time, dtype=int)
    path[-1] = int(score.argmax())
    for j in range(n_time - 1, 0, -1):
        path[j - 1] = back[j, path[j]]
    return path


def refine_path_subbin(
    fb: np.ndarray, logP: np.ndarray, bins: np.ndarray
) -> np.ndarray:
    """Parabolic sub-bin refinement of a bin-quantised frequency path.

    Fits a parabola through logP at (bin-1, bin, bin+1) per frame and shifts
    the path frequency to its vertex, clamped to +-half a bin; edge bins and
    non-concave triples are left unrefined. NECESSARY, not cosmetic: the DG
    statistic pools a coherent sum over the record, so the warp must be
    accurate to |delta f| <~ 1/(2 pi T) — a few mHz at T = 300 s — while the
    spectrogram bin width is fs/nperseg = 62.5 mHz at the defaults. A
    bin-quantised path leaves the angle-domain line smeared and the pipeline
    powerless at ANY SNR.
    """
    bins = np.asarray(bins, dtype=int)
    j = np.arange(bins.size)
    n_freq = fb.size
    f = fb[bins].astype(float)
    if n_freq < 3 or bins.size == 0:
        return f
    interior = (bins > 0) & (bins < n_freq - 1)
    a = logP[np.clip(bins - 1, 0, n_freq - 1), j]
    b = logP[bins, j]
    c = logP[np.clip(bins + 1, 0, n_freq - 1), j]
    den = a - 2.0 * b + c
    ok = interior & (den < 0.0)
    delta = np.zeros(bins.size)
    delta[ok] = np.clip(0.5 * (a[ok] - c[ok]) / den[ok], -0.5, 0.5)
    return f + delta * (fb[1] - fb[0])


def heldout_freq_path(
    t: np.ndarray,
    p_obs: np.ndarray,
    band_lo: float,
    band_hi: float,
    *,
    params: St1DetectorParams,
    nperseg: int | None = None,
    noverlap: int | None = None,
    jump_penalty: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Frequency path estimated ONLY from frames wholly inside EST blocks.

    Runs the spectrogram over the full trace (window defaults match
    typeb.detectors: ~16 s frames, 50% overlap), then masks OUT every frame
    whose support [centre - nperseg/2fs, centre + nperseg/2fs] contains ANY
    TEST sample — touching a guard gap is allowed (guards belong to neither
    set). Viterbi runs over the surviving frames only; its transitions bridge
    the TEST-block gaps. The winning path is sub-bin refined (see
    refine_path_subbin — without it the pooled DG sum is powerless). Returns
    ``(frame_times, path_freqs_hz)`` for the retained frames — the input
    phase_from_path interpolates across TEST.
    """
    t = np.asarray(t, dtype=float)
    n = np.asarray(p_obs).size
    fs = 1.0 / (t[1] - t[0])
    if nperseg is None:
        nperseg = int(min(n, max(64, round(16.0 * fs))))
    if noverlap is None:
        noverlap = nperseg // 2

    fb, times, logP = _spectrogram_band(t, p_obs, band_lo, band_hi,
                                        nperseg, noverlap)
    if fb.size == 0 or times.size == 0:
        return np.zeros(0), np.zeros(0)

    _, test_mask = split_masks(n, fs, params.split_block_s, params.split_guard_s)
    half = 0.5 * nperseg / fs
    i0 = np.clip(np.ceil((times - half) * fs).astype(int), 0, n - 1)
    i1 = np.clip(np.floor((times + half) * fs).astype(int), 0, n - 1)
    cs = np.concatenate([[0], np.cumsum(test_mask)])
    n_test_in_support = cs[i1 + 1] - cs[i0]
    est_frames = n_test_in_support == 0

    if not np.any(est_frames):
        return np.zeros(0), np.zeros(0)
    logP_est = logP[:, est_frames]
    bins = _viterbi_path_bins(logP_est, jump_penalty)
    return times[est_frames], refine_path_subbin(fb, logP_est, bins)
