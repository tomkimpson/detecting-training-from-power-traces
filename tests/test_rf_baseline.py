"""Task 9.1 RF power-only baseline (code.typeb.rf_baseline).

Exercises the Rahman-style power-channel feature vocabulary and the grouped-CV
out-of-fold RandomForest scorer on synthetic traces (CPU, no GPU). The headline
methodological trap — windows within one trace are highly correlated, so
splitting them across CV folds inflates AUC — is guarded by
``test_grouping_prevents_leakage``.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from powerladder.config import DEFAULT
from powerladder.typeb.rf_baseline import (
    LEVEL_FEATURES,
    RfResult,
    _autocorr,
    _fft_periodicity,
    power_features,
    rf_fit_predict,
    rf_oof_scores,
)
from powerladder.typeb.roc import auc
from powerladder.typeb.synth import TypeTrace

# Small forests keep the suite fast; the science does not depend on 400 trees.
FAST = dataclasses.replace(DEFAULT.rf, n_estimators=60)


def _line_trace(f, *, fs=20.0, dur=300.0, amp=30.0, base=300.0, noise=3.0,
                seed=0, label="train"):
    rng = np.random.default_rng(seed)
    t = np.arange(0.0, dur, 1.0 / fs)
    p = base + rng.normal(0.0, noise, t.size)
    if f is not None:
        p += amp * np.sin(2 * np.pi * f * t)
    return TypeTrace(t=t, P_obs=p, label=label, f0=f if f is not None else np.nan)


def _flat_trace(*, fs=20.0, dur=300.0, base=300.0, noise=3.0, seed=0,
                label="infer"):
    return _line_trace(None, fs=fs, dur=dur, base=base, noise=noise, seed=seed,
                       label=label)


# --------------------------------------------------------------------------
# feature extraction
# --------------------------------------------------------------------------

def test_power_features_shape_and_names():
    p = DEFAULT.rf
    tr = _line_trace(1.0, seed=1)
    X, names, starts = power_features(
        tr.t, tr.P_obs, fs=20.0, window_s=p.window_s, stride_s=p.stride_s,
        autocorr_lags_s=p.autocorr_lags_s, idle_frac_tol=p.idle_frac_tol)
    # 300 s @ 20 Hz, 30 s window / 15 s stride -> 19 full windows.
    assert X.shape == (19, 21)
    assert len(names) == 21
    assert len(starts) == 19
    for expected in ("mean", "std", "acf_1s", "acf_20s", "fft_periodicity",
                     "duty", "idle_frac"):
        assert expected in names
    assert np.all(np.isfinite(X))


def test_drop_level_removes_exactly_the_level_features():
    p = DEFAULT.rf
    tr = _line_trace(1.0, seed=2)
    X, names, _ = power_features(
        tr.t, tr.P_obs, fs=20.0, window_s=p.window_s, stride_s=p.stride_s,
        autocorr_lags_s=p.autocorr_lags_s, idle_frac_tol=p.idle_frac_tol,
        drop_level=True)
    assert X.shape[1] == 13
    assert len(names) == 13
    assert not (set(names) & LEVEL_FEATURES)          # no level features remain
    for kept in ("std", "iqr", "cv", "skew", "kurtosis", "acf_1s",
                 "fft_periodicity", "duty", "idle_frac"):
        assert kept in names


def test_autocorr_recovers_period():
    # pure sine, period = 20 samples; acf at one period ~ +1, at half ~ -1.
    n = 2000
    x = np.sin(2 * np.pi * np.arange(n) / 20.0)
    assert _autocorr(x, 20) > 0.9
    assert _autocorr(x, 10) < -0.8
    # flat signal has no autocorrelation structure.
    assert abs(_autocorr(np.full(n, 5.0), 20)) < 1e-6
    # lag longer than the window returns 0, not an error.
    assert _autocorr(np.arange(5.0), 20) == 0.0


def test_fft_periodicity_separates_line_from_noise():
    rng = np.random.default_rng(0)
    n = 600
    line = np.sin(2 * np.pi * np.arange(n) / 20.0) + rng.normal(0, 0.1, n)
    noise = rng.normal(0, 1.0, n)
    assert _fft_periodicity(line) > 5 * _fft_periodicity(noise)


# --------------------------------------------------------------------------
# grouped-CV out-of-fold RandomForest scoring
# --------------------------------------------------------------------------

def test_rf_separates_line_from_flat():
    traces, labels = [], []
    for i in range(12):
        traces.append(_line_trace(1.0 + 0.05 * i, seed=100 + i, label="train"))
        labels.append(True)
    for i in range(12):
        traces.append(_flat_trace(seed=200 + i, label="infer"))
        labels.append(False)
    res = rf_oof_scores(traces, labels, FAST, fs=20.0,
                        window_s=FAST.window_s, stride_s=FAST.stride_s,
                        seed=FAST.seed)
    assert isinstance(res, RfResult)
    assert res.pos.size == 12 and res.neg.size == 12
    assert auc(res.pos, res.neg) > 0.9
    assert auc(res.win_pos, res.win_neg) > 0.9
    assert res.importances.shape[0] == len(res.feature_names)


def test_grouping_prevents_leakage():
    # Each trace has a distinct constant offset (learnable trace identity) but a
    # RANDOM label. Windows within a trace share the offset, so a CV split that
    # leaked windows across folds would let the forest map offset -> trace ->
    # label and score ~1.0. Grouping by trace makes that impossible: held-out
    # traces have unseen offsets, so out-of-fold AUC must sit near chance.
    rng = np.random.default_rng(7)
    traces, labels = [], []
    offsets = np.linspace(100.0, 500.0, 20)
    for i, off in enumerate(offsets):
        t = np.arange(0.0, 120.0, 1.0 / 20.0)
        p = off + rng.normal(0.0, 1.0, t.size)
        traces.append(TypeTrace(t=t, P_obs=p, label="x", f0=np.nan))
        labels.append(bool(i % 2))                    # arbitrary, offset-independent
    res = rf_oof_scores(traces, labels, FAST, fs=20.0, window_s=30.0,
                        stride_s=15.0, seed=1)
    # Leakage would drive this to ~1.0; grouped CV keeps it near chance.
    assert auc(res.win_pos, res.win_neg) < 0.8


def test_zero_shot_fit_predict_generalises_and_is_deterministic():
    # train on clean line-vs-flat; score a held-out set the model never saw.
    train, tlab = [], []
    for i in range(8):
        train.append(_line_trace(1.0, seed=500 + i, label="train"))
        tlab.append(True)
        train.append(_flat_trace(seed=600 + i, label="infer"))
        tlab.append(False)
    test_pos = [_line_trace(1.0, seed=700 + i) for i in range(6)]
    test_neg = [_flat_trace(seed=800 + i) for i in range(6)]
    kw = dict(fs=20.0, window_s=FAST.window_s, stride_s=FAST.stride_s, seed=3)
    pos = rf_fit_predict(train, tlab, test_pos, FAST, **kw)
    neg = rf_fit_predict(train, tlab, test_neg, FAST, **kw)
    assert pos.size == 6 and neg.size == 6
    assert auc(pos, neg) > 0.9                     # generalises zero-shot
    # determinism under fixed seed
    assert np.allclose(pos, rf_fit_predict(train, tlab, test_pos, FAST, **kw))


def test_determinism_under_fixed_seed():
    traces, labels = [], []
    for i in range(8):
        traces.append(_line_trace(1.0, seed=300 + i, label="train"))
        labels.append(True)
        traces.append(_flat_trace(seed=400 + i, label="infer"))
        labels.append(False)
    kw = dict(fs=20.0, window_s=FAST.window_s, stride_s=FAST.stride_s, seed=42)
    a = rf_oof_scores(traces, labels, FAST, **kw)
    b = rf_oof_scores(traces, labels, FAST, **kw)
    assert np.allclose(a.pos, b.pos)
    assert np.allclose(a.neg, b.neg)
