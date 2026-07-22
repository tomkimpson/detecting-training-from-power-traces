"""ST1 staged detectors under the shared gate signature.

The ST1 design isolates each adaptive ingredient in stages (plan Sec. 6 ST1):

    stage 1  fixed known cycle frequency          (this module, task 19.3)
    stage 2  known externally-supplied phase path (task 19.5)
    stage 3  phase path estimated on held-out data (task 19.6)
    stage 4  fully adaptive pipeline              (task 19.7)

Detectors here follow the code.typeb.gate convention exactly —
``fn(t, P_obs, band_lo, band_hi) -> float`` with larger = more structured —
via ``functools.partial`` over the params (and the fixed alpha). They are NOT
inserted into ``code.typeb.gate.DETECTORS`` (that registry is the frozen
B0/B1/B2 track); callers merge the dicts explicitly where a joint bake-off
needs them.

Conditioning-failure bookkeeping: stage detectors built on the DG statistic
increment the module counter in code.st1.cyclo (reset/read by the FAR
harness around each cell); see cyclo._COV_FAILURES.
"""

from __future__ import annotations

from functools import partial

import numpy as np
from scipy.stats import chi2

from ..config import DEFAULT, St1DetectorParams
from .cyclo import dg_q, neg_log10_p
from .multitaper import comb_f_statistic, f_test_statistic
from ..typeb.detectors import viterbi_best_path
from .resample import (angle_resample, angle_sample_times, phase_from_path,
                       resample_at)
from .splitting import (heldout_freq_path, refine_path_subbin, split_masks)

# The nominal training cadence: centre of the Ko et al. f0 draw band
# (KoWorkloadParams.f0_lo / f0_hi) — the alpha a verifier who knows the
# declared iteration cadence would test.
KO_NOMINAL_F0_HZ: float = 0.5 * (DEFAULT.ko.f0_lo + DEFAULT.ko.f0_hi)


def _tracker_nperseg(n: int, fs: float, params: St1DetectorParams) -> int | None:
    """Spectrogram window for stages 3-4, honouring tracker_nperseg_factor.

    Returns None at factor 1.0 so the callee's default path (and hence the
    calibrated chunk-1/2 behaviour) is preserved bit-for-bit; otherwise scales
    the typeb default nperseg = min(n, max(64, round(16 s * fs))).
    """
    factor = getattr(params, "tracker_nperseg_factor", 1.0)
    if factor == 1.0:
        return None
    base = int(min(n, max(64, round(16.0 * fs))))
    return int(min(n, max(16, round(factor * base))))


def stage1_fixed_alpha(
    t: np.ndarray,
    p_obs: np.ndarray,
    band_lo: float,
    band_hi: float,
    *,
    params: St1DetectorParams,
    alpha_hz: float,
) -> float:
    """Stage 1: DG statistic at one FIXED, known cycle frequency.

    -log10 of the asymptotic chi2(2L) p-value of Q at ``alpha_hz``. No search,
    no resampling, no tracking — the base case whose null calibration is the
    earliest kill-signal for the gate. ``band_lo``/``band_hi`` are unused
    (kept for the shared signature; the band enters only searching stages).
    """
    fs = 1.0 / (t[1] - t[0])
    res = dg_q(p_obs, fs, alpha_hz, params.lag_set, params=params)
    return neg_log10_p(res.Q, 2 * len(params.lag_set))


def _dg_on_angle_series(x_ang: np.ndarray, params: St1DetectorParams,
                        mask: np.ndarray | None = None) -> float:
    """DG at the FIXED order on a uniform angle-domain series -> -log10 p.

    Bookkeeping: the angle grid is uniform, so it is treated as UNIT-SPACED
    samples (fs = 1) with cycle frequency alpha = 1/samples_per_cycle — one
    modulation period per samples_per_cycle samples, matching
    cyclo._z_series's t_n = n/fs phase convention exactly. Order 1 only:
    dg_q is a single-alpha statistic; a joint order-(1,2,3) stack would
    triple the covariance dimension (2L -> 6L), so higher orders are
    deferred to the stage-8 sweep axes rather than bolted on here.

    Degenerate-trace guard: a floor-clipped path over a short record can
    leave too few angle samples for the 2L-dim covariance; such traces score
    0.0 (p = 1) rather than produce a meaningless solve.
    """
    L = len(params.lag_set)
    if x_ang.size - max(params.lag_set) < 4 * L:
        return 0.0
    alpha = 1.0 / params.samples_per_cycle
    kw = {} if mask is None else {"mask": mask}
    res = dg_q(x_ang, 1.0, alpha, params.lag_set, params=params, **kw)
    return neg_log10_p(res.Q, 2 * L)


def stage2_known_phase(
    t: np.ndarray,
    p_obs: np.ndarray,
    band_lo: float,
    band_hi: float,
    *,
    params: St1DetectorParams,
    theta_true: np.ndarray,
) -> float:
    """Stage 2: angle-resample with a KNOWN phase, then fixed-order DG.

    The phase path is handed to the detector (in the FAR harness: an
    independently drawn random smooth warp — see
    resample.random_smooth_phase_path), so relative to stage 1 the ONLY new
    ingredient is the resampling operation itself: non-uniform interpolation
    of the trace onto the angle grid. The stage 1->2 FAR delta therefore
    isolates what interpolation does to the fixed-alpha chi2 null (e.g.
    interpolation-induced correlation between angle samples). Harness-only:
    needs ``theta_true``, so it is not in the ST1_DETECTORS registry.
    ``band_lo``/``band_hi`` unused (shared signature).
    """
    _, x_ang = angle_resample(t, p_obs, theta_true,
                              samples_per_cycle=params.samples_per_cycle,
                              interp=params.resample_interp)
    return _dg_on_angle_series(x_ang, params)


def stage3_heldout_phase(
    t: np.ndarray,
    p_obs: np.ndarray,
    band_lo: float,
    band_hi: float,
    *,
    params: St1DetectorParams,
) -> float:
    """Stage 3: warp estimated on held-out EST blocks, DG scored on TEST only.

    Pipeline: (i) Viterbi frequency path from spectrogram frames wholly inside
    EST blocks (splitting.heldout_freq_path — no TEST sample influences the
    path); (ii) phase_from_path interpolates the warp across the TEST blocks;
    (iii) the FULL trace is angle-resampled; (iv) dg_q at the fixed order
    alpha = 1/samples_per_cycle uses ONLY angle samples mapping back into
    TEST blocks — a row survives iff sample n AND n+tau for every lag land in
    TEST; N in Q is the retained count.

    Three assumptions (see splitting module docstring): mixing << guard_s;
    wander smooth on the block scale; warp fixed given EST. The FAR harness
    measures the MARGINAL realised FAR of this whole recipe — it does not
    certify conditional-on-EST validity.
    """
    frame_times, path_freqs = heldout_freq_path(
        t, p_obs, band_lo, band_hi, params=params,
        nperseg=_tracker_nperseg(t.size, 1.0 / (t[1] - t[0]), params))
    if frame_times.size < 2:
        return 0.0                               # degenerate: nothing tracked
    _, theta = phase_from_path(frame_times, path_freqs, t,
                               floor_hz=params.f_path_floor_hz)
    _, t_m = angle_sample_times(
        t, theta, samples_per_cycle=params.samples_per_cycle)
    x_ang = resample_at(t, p_obs, t_m, params.resample_interp)
    fs = 1.0 / (t[1] - t[0])
    _, test_mask = split_masks(t.size, fs, params.split_block_s,
                               params.split_guard_s)
    idx = np.clip(np.round(t_m * fs).astype(int), 0, t.size - 1)
    mask_ang = test_mask[idx]
    return _dg_on_angle_series(x_ang, params, mask=mask_ang)


def stage4_full_adaptive(
    t: np.ndarray,
    p_obs: np.ndarray,
    band_lo: float,
    band_hi: float,
    *,
    params: St1DetectorParams,
) -> float:
    """Stage 4: fully adaptive pipeline — no splitting, no protection.

    viterbi_best_path on the FULL trace (the tracker selects the best path
    through the data it will then be scored on), sub-bin refinement,
    phase_from_path, angle_resample, DG at the fixed order over ALL angle
    samples. This is the maximally optimistic, least-protected pipeline: the
    same samples choose the warp and are scored under it, so the tracker's
    effective search over paths is an unaccounted look-elsewhere. The
    stage 3 -> 4 FAR delta MEASURES that path-selection inflation — this
    function exists to be measured, not to be trusted.
    """
    fb, times, logP, path_freqs = viterbi_best_path(
        t, p_obs, band_lo, band_hi,
        nperseg=_tracker_nperseg(t.size, 1.0 / (t[1] - t[0]), params))
    if path_freqs.size < 2:
        return 0.0
    # recover bin indices from the returned (exact-bin-valued) path, then
    # refine sub-bin — the pooled DG sum needs mHz-scale warp accuracy.
    bins = np.abs(fb[:, None] - path_freqs[None, :]).argmin(axis=0)
    refined = refine_path_subbin(fb, logP, bins)
    _, theta = phase_from_path(times, refined, t,
                               floor_hz=params.f_path_floor_hz)
    _, x_ang = angle_resample(t, p_obs, theta,
                              samples_per_cycle=params.samples_per_cycle,
                              interp=params.resample_interp)
    return _dg_on_angle_series(x_ang, params)


def stage4_semicoherent(
    t: np.ndarray,
    p_obs: np.ndarray,
    band_lo: float,
    band_hi: float,
    *,
    params: St1DetectorParams,
) -> float:
    """Semi-coherent stage 4: per-block DG on the de-warped series, Fisher-combined.

    Motivation (chunk-2 implementation finding, notes/st1-findings.md): the
    POOLED coherent DG sum needs the warp accurate to ~1/(2 pi T) Hz over the
    whole record — a few mHz at 300 s — so tracker error decoheres it for fast
    wander. Splitting the ANGLE-domain series into blocks of
    ``params.semicoh_block_s`` seconds (by each angle sample's time preimage),
    computing the fixed-order DG per block (block-local long-run covariance),
    and Fisher-combining X = -2 sum_k ln p_k -> chi2(2K) relaxes the accuracy
    requirement to ~1/(2 pi * block_s): the classic coherence/robustness
    trade of semi-coherent CW searches.

    Same tracker and warp as stage 4 (fully adaptive, no splitting), so it
    inherits stage 4's selection effects; its null level is checked by its own
    FAR spot-check cells, not assumed. Fisher treats the per-block p-values as
    independent — the shared full-trace tracked path couples them, which is
    exactly what the spot-check measures. Blocks too short for the covariance
    (fewer than 4L usable rows) are skipped; no scorable block -> 0.0 (p = 1).
    """
    fb, times, logP, path_freqs = viterbi_best_path(
        t, p_obs, band_lo, band_hi,
        nperseg=_tracker_nperseg(t.size, 1.0 / (t[1] - t[0]), params))
    if path_freqs.size < 2:
        return 0.0
    bins = np.abs(fb[:, None] - path_freqs[None, :]).argmin(axis=0)
    refined = refine_path_subbin(fb, logP, bins)
    _, theta = phase_from_path(times, refined, t,
                               floor_hz=params.f_path_floor_hz)
    _, t_m = angle_sample_times(
        t, theta, samples_per_cycle=params.samples_per_cycle)
    x_ang = resample_at(t, p_obs, t_m, params.resample_interp)

    L = len(params.lag_set)
    alpha = 1.0 / params.samples_per_cycle
    block = np.floor(t_m / params.semicoh_block_s).astype(int)
    log_p_sum = 0.0
    n_blocks = 0
    for b in np.unique(block):
        seg = x_ang[block == b]                   # contiguous angle samples
        if seg.size - max(params.lag_set) < 4 * L:
            continue                              # too short to estimate Sigma
        res = dg_q(seg, 1.0, alpha, params.lag_set, params=params)
        log_p_sum += chi2.logsf(res.Q, 2 * L)     # ln p_b, underflow-safe
        n_blocks += 1
    if n_blocks == 0:
        return 0.0
    fisher_x = -2.0 * log_p_sum
    return float(-chi2.logsf(fisher_x, 2 * n_blocks) / np.log(10.0))


def st1_detectors(
    params: St1DetectorParams = DEFAULT.st1,
    *,
    alpha_hz: float = KO_NOMINAL_F0_HZ,
) -> dict:
    """Build the ST1 detector registry (gate signature, params bound by partial).

    dg_order_split / dg_order_full are stages 3 / 4 of the ladder. Stage 2
    (dg_order_known) is deliberately NOT here: it needs the true phase path
    (theta_true), so it exists only inside the FAR harness, which draws the
    known warp per replicate.
    """
    return {
        "mtf": partial(f_test_statistic, params=params),
        "mtf_comb": partial(comb_f_statistic, params=params),
        "dg_fixed": partial(stage1_fixed_alpha, params=params, alpha_hz=alpha_hz),
        "dg_order_split": partial(stage3_heldout_phase, params=params),
        "dg_order_full": partial(stage4_full_adaptive, params=params),
    }


# Default registry at the default params and the nominal Ko cadence.
ST1_DETECTORS = st1_detectors()
