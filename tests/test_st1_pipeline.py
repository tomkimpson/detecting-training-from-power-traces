"""ST1 staged-pipeline tests (tasks 19.6+): split masks disjoint with guards,
stage 3 finite on null and signal, and stage 3 separates a wandering-line
signal from the AR(1) null (small-population AUC)."""

from __future__ import annotations

import numpy as np

from powerladder.config import DEFAULT
from powerladder.st1.nulls import ar1, white
from powerladder.st1.pipeline import (ST1_DETECTORS, stage3_heldout_phase,
                               stage4_full_adaptive, stage4_semicoherent)
from powerladder.st1.resample import random_smooth_phase_path
from powerladder.st1.splitting import heldout_freq_path, split_masks
from powerladder.typeb.synth import _duty_wave

FS = 20.0
N = 6001  # 300 s at 20 Hz
BAND = (0.3, 1.7)


def _grid():
    return np.arange(N) / FS


def _wandering_signal(rng, amp=16.0, sigma=4.0, drift_hz=0.05,
                      drift_theta_per_cycle=0.002):
    """Wandering duty wave + white noise (the Ko-drift idiom, self-contained).

    Default operating point is deliberately favourable — SLOW smooth wander
    (excursion 0.05 Hz, correlation ~200 cycles) at a strong line. The pooled
    coherent DG sum needs the estimated warp accurate to ~1/(2*pi*T) Hz over
    the whole record; with the warp anchored only at EST frames, stage 3 has
    usable power ONLY in this slow-wander regime (the honest power boundary —
    quantified properly in the task-19.10 bake-off).
    """
    t = _grid()
    _, theta = random_smooth_phase_path(
        t, rng, drift_hz=drift_hz,
        drift_theta_per_cycle=drift_theta_per_cycle)
    x = amp * _duty_wave(theta, duty=0.6, n_harmonics=3) \
        + rng.normal(0.0, sigma, size=t.size)
    return t, x


def test_split_masks_disjoint_guarded_and_partial():
    n, fs = N, FS
    block_s, guard_s = 30.0, 2.0
    est, test = split_masks(n, fs, block_s, guard_s)
    assert est.dtype == bool and test.dtype == bool
    assert not np.any(est & test)                      # disjoint
    assert est.sum() + test.sum() < n                  # guards dropped
    t = np.arange(n) / fs
    # samples within guard_s of any block boundary are in neither mask
    pos = t - np.floor(t / block_s) * block_s
    guarded = (pos < guard_s) | (pos >= block_s - guard_s)
    assert not np.any((est | test)[guarded])
    # parity: EST in even blocks only, TEST in odd blocks only
    block = np.floor(t / block_s).astype(int)
    assert np.all(block[est] % 2 == 0)
    assert np.all(block[test] % 2 == 1)


def test_heldout_frames_avoid_test_blocks():
    """No retained frame's support may contain a TEST sample."""
    rng = np.random.default_rng(0)
    t, x = _wandering_signal(rng)
    p = DEFAULT.st1
    frame_times, path_freqs = heldout_freq_path(t, x, *BAND, params=p)
    assert frame_times.size >= 2
    # refined freqs may exceed the band edges by up to half a bin
    half_bin = 0.5 * FS / 320
    assert np.all((path_freqs >= BAND[0] - half_bin)
                  & (path_freqs <= BAND[1] + half_bin))
    _, test_mask = split_masks(N, FS, p.split_block_s, p.split_guard_s)
    half = 16.0 / 2.0                                  # default ~16 s frames
    for tc in frame_times:
        i0 = max(0, int(np.ceil((tc - half) * FS)))
        i1 = min(N - 1, int(np.floor((tc + half) * FS)))
        assert not test_mask[i0:i1 + 1].any()


def test_stage3_finite_on_null_and_signal():
    p = DEFAULT.st1
    rng = np.random.default_rng(1)
    tr = ar1(N, FS, DEFAULT.st1_null, rng)
    s_null = stage3_heldout_phase(tr.t, tr.P_obs, *BAND, params=p)
    t, x = _wandering_signal(rng)
    s_sig = stage3_heldout_phase(t, x, *BAND, params=p)
    assert np.isfinite(s_null) and np.isfinite(s_sig)
    assert s_sig > s_null


def test_stage3_separates_signal_from_ar1_null_auc():
    """AUC over a small population > 0.8 (30/class, kept fast)."""
    p = DEFAULT.st1
    rng = np.random.default_rng(2)
    n_per = 30
    s_null = np.array([
        stage3_heldout_phase(*(lambda tr: (tr.t, tr.P_obs))(
            ar1(N, FS, DEFAULT.st1_null, rng)), *BAND, params=p)
        for _ in range(n_per)
    ])
    s_sig = np.empty(n_per)
    for i in range(n_per):
        t, x = _wandering_signal(rng)
        s_sig[i] = stage3_heldout_phase(t, x, *BAND, params=p)
    # rank-sum AUC
    order = np.argsort(np.concatenate([s_null, s_sig]), kind="stable")
    ranks = np.empty(2 * n_per)
    ranks[order] = np.arange(1, 2 * n_per + 1)
    auc = (ranks[n_per:].sum() - n_per * (n_per + 1) / 2) / (n_per * n_per)
    assert auc > 0.8, f"stage-3 AUC only {auc:.3f}"


def test_stage4_finite_on_nulls():
    p = DEFAULT.st1
    rng = np.random.default_rng(3)
    for null_fn in (white, ar1):
        tr = null_fn(N, FS, DEFAULT.st1_null, rng)
        s = stage4_full_adaptive(tr.t, tr.P_obs, *BAND, params=p)
        assert np.isfinite(s)


def test_stage4_fires_on_wandering_line():
    """Full adaptive pipeline: signal score >> every null score."""
    p = DEFAULT.st1
    rng = np.random.default_rng(4)
    s_null = np.array([
        stage4_full_adaptive(*(lambda tr: (tr.t, tr.P_obs))(
            ar1(N, FS, DEFAULT.st1_null, rng)), *BAND, params=p)
        for _ in range(5)
    ])
    t, x = _wandering_signal(rng)
    s_sig = stage4_full_adaptive(t, x, *BAND, params=p)
    assert s_sig > s_null.max() + 5.0, \
        f"sig {s_sig:.2f} vs nulls max {s_null.max():.2f}"


def test_stage4_semicoherent_finite_on_nulls_and_fires_on_signal():
    """Semi-coherent variant: finite on nulls, signal >> nulls (like stage 4)."""
    p = DEFAULT.st1
    rng = np.random.default_rng(6)
    s_null = []
    for null_fn in (white, ar1):
        tr = null_fn(N, FS, DEFAULT.st1_null, rng)
        s = stage4_semicoherent(tr.t, tr.P_obs, *BAND, params=p)
        assert np.isfinite(s)
        s_null.append(s)
    t, x = _wandering_signal(rng)
    s_sig = stage4_semicoherent(t, x, *BAND, params=p)
    assert s_sig > max(s_null) + 5.0, \
        f"sig {s_sig:.2f} vs nulls max {max(s_null):.2f}"


def test_registry_has_staged_detectors():
    assert {"dg_order_split", "dg_order_full"} <= set(ST1_DETECTORS)
    rng = np.random.default_rng(5)
    tr = white(N, FS, DEFAULT.st1_null, rng)
    for name in ("dg_order_split", "dg_order_full"):
        assert np.isfinite(ST1_DETECTORS[name](tr.t, tr.P_obs, *BAND))
