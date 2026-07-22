"""Task 9.1: Rahman et al. (arXiv 2606.19262) RandomForest, power channel only.

Rahman detect hidden ML training with a RandomForest over 166 windowed features
from 9 NVML channels @ 1 Hz (headline 98.2%). This module replicates their
*method* — the App. A.3 feature vocabulary and RF hyperparameters
(:class:`code.config.RfBaselineParams`) — but **restricted to the one channel an
off-chip power meter sees**, trained on our existing B2 A100 power traces. Their
code is not public and expects their 1 Hz 9-channel format, so we reimplement
rather than reuse (notes/rahman2606.19262-comparison.md §6; tasks.md Task 9.1).

It exists to answer two questions on equal footing with our physics detectors
(``code.typeb.detectors``): (1) the *information gap* — how much of Rahman's
98.2% came from the non-power channels a meter is blind to; and (2) the
supervised ceiling on the trusted channel our Viterbi tracker must beat.

Design (mirrors ``code.typeb.gate.evaluate`` rather than the stateless
``(t, p, lo, hi) -> float`` detector contract, because a RandomForest must be
*fit*): features are extracted per sliding window (that is where the classifier
gets its sample count), scored by grouped-CV out-of-fold probability with **all
windows of one trace confined to a single fold** (the anti-leakage guard — see
``rf_oof_scores``), then aggregated to one score per trace so the result drops
straight into ``code.typeb.roc`` alongside the spectral and Viterbi statistics.

Pure numpy + scipy + scikit-learn; consumes only ``.t``/``.P_obs`` off a trace,
so it is generator-agnostic like the other detectors.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import RfBaselineParams

# Absolute-level features (Rahman's train-vs-infer stage drops these, keeping
# shape/periodicity/autocorrelation only; ``drop_level=True`` reproduces that).
LEVEL_FEATURES = {"mean", "min", "max", "p25", "p50", "p75", "p95", "range"}

_BASE_NAMES = ["mean", "std", "min", "max", "p25", "p50", "p75", "p95",
               "iqr", "range", "cv", "skew", "kurtosis"]
_TAIL_NAMES = ["fft_periodicity", "duty", "idle_frac"]


# ---------------------------------------------------------------------------
# feature primitives (pure numpy)
# ---------------------------------------------------------------------------

def _window_bounds(n: int, fs: float, window_s: float, stride_s: float
                   ) -> list[tuple[int, int]]:
    """Contiguous [start, end) sample-index pairs; a trailing partial window is
    dropped so every window has the same length."""
    w = int(round(window_s * fs))
    s = max(1, int(round(stride_s * fs)))
    if w <= 0 or n < w:
        return []
    return [(start, start + w) for start in range(0, n - w + 1, s)]


def _autocorr(x: np.ndarray, lag: int) -> float:
    """Normalised autocorrelation of ``x`` at integer ``lag`` samples.

    0 if the lag does not fit in the window or the window is constant (no
    structure), +1 at a full period of a pure sine, -1 at a half period.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    if lag <= 0 or lag >= n:
        return 0.0
    xc = x - x.mean()
    denom = float(np.dot(xc, xc))
    if denom <= 0.0:
        return 0.0
    return float(np.dot(xc[:-lag], xc[lag:]) / denom)


def _fft_periodicity(x: np.ndarray) -> float:
    """Rahman's epoch-periodicity feature: dominant-bin power / mean-bin power.

    Detrend (remove DC), rFFT, drop the DC bin: a concentrated cadence line
    gives one tall bin over a low floor (large ratio); broadband fluctuation
    spreads power evenly (ratio near 1).
    """
    x = np.asarray(x, dtype=float)
    if x.size < 4:
        return 0.0
    sp = np.abs(np.fft.rfft(x - x.mean())) ** 2
    sp = sp[1:]                                   # exclude DC
    if sp.size == 0:
        return 0.0
    m = float(sp.mean())
    return float(sp.max() / m) if m > 0.0 else 0.0


def _duty_idle(x: np.ndarray, tol: float) -> tuple[float, float]:
    """(duty, idle_frac): fraction above the window midpoint, and fraction
    within ``tol`` of the window's dynamic range above its minimum."""
    x = np.asarray(x, dtype=float)
    lo = float(x.min())
    hi = float(x.max())
    rng = hi - lo
    if rng <= 0.0:
        return 0.0, 1.0                           # flat window: all "idle"
    duty = float(np.mean(x > lo + 0.5 * rng))
    idle = float(np.mean(x <= lo + tol * rng))
    return duty, idle


def _window_features(w: np.ndarray, lags: tuple[int, ...], tol: float
                     ) -> list[float]:
    """The full (21-value) Rahman power-channel feature vector for one window."""
    from scipy import stats as sp_stats

    mean = float(w.mean())
    std = float(w.std())
    lo, hi = float(w.min()), float(w.max())
    p25, p50, p75, p95 = (float(v) for v in np.percentile(w, [25, 50, 75, 95]))
    iqr = p75 - p25
    rng = hi - lo
    cv = std / mean if mean != 0.0 else 0.0
    if std > 0.0:
        skew = float(sp_stats.skew(w))
        kurt = float(sp_stats.kurtosis(w))         # Fisher (normal -> 0)
    else:
        skew = kurt = 0.0
    base = [mean, std, lo, hi, p25, p50, p75, p95, iqr, rng, cv, skew, kurt]
    acf = [_autocorr(w, lag) for lag in lags]
    fft = _fft_periodicity(w)
    duty, idle = _duty_idle(w, tol)
    return base + acf + [fft, duty, idle]


def power_features(
    t: np.ndarray,
    p: np.ndarray,
    *,
    fs: float,
    window_s: float,
    stride_s: float,
    autocorr_lags_s: tuple[float, ...],
    idle_frac_tol: float,
    drop_level: bool = False,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Rahman App. A.3 power-channel feature matrix over sliding windows.

    ``p`` is assumed already on the uniform ``fs`` grid (resample/decimate
    upstream). Returns ``(X[n_windows, n_features], feature_names,
    window_start_s)``. ``drop_level`` removes the absolute-level features
    (:data:`LEVEL_FEATURES`), leaving Rahman's shape-only train-vs-infer subset.
    Autocorrelation lags are given in seconds and converted to samples at ``fs``,
    so they carry the same physical meaning at 1 Hz and 20 Hz.
    """
    p = np.asarray(p, dtype=float)
    lags = tuple(int(round(L * fs)) for L in autocorr_lags_s)
    acf_names = [f"acf_{L:g}s" for L in autocorr_lags_s]
    all_names = _BASE_NAMES + acf_names + _TAIL_NAMES
    keep = [i for i, nm in enumerate(all_names)
            if not (drop_level and nm in LEVEL_FEATURES)]
    names = [all_names[i] for i in keep]

    bounds = _window_bounds(p.size, fs, window_s, stride_s)
    if not bounds:
        return np.empty((0, len(names))), names, np.empty(0)
    rows = [_window_features(p[a:b], lags, idle_frac_tol) for a, b in bounds]
    X = np.asarray(rows, dtype=float)[:, keep]
    starts = np.array([a / fs for a, _ in bounds])
    return X, names, starts


# ---------------------------------------------------------------------------
# grouped-CV out-of-fold RandomForest scoring
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RfResult:
    """Out-of-fold RandomForest scores for a labelled trace population."""

    pos: np.ndarray            # per-trace OOF score (mean window prob), positives
    neg: np.ndarray            # per-trace OOF score, negatives
    win_pos: np.ndarray        # per-window OOF prob, positive-trace windows
    win_neg: np.ndarray        # per-window OOF prob, negative-trace windows
    feature_names: list[str]
    importances: np.ndarray    # mean over folds of clf.feature_importances_
    n_windows: int
    drop_level: bool


def rf_oof_scores(
    traces,
    labels,
    p: RfBaselineParams,
    *,
    fs: float,
    window_s: float,
    stride_s: float,
    drop_level: bool = False,
    seed: int | None = None,
) -> RfResult:
    """Grouped-CV out-of-fold RandomForest scores for a labelled population.

    ``traces``: sequence exposing ``.t``/``.P_obs`` (e.g. ``TypeTrace``) already
    at rate ``fs``; ``labels``: truthy = positive (training). Windows are
    grouped by trace under ``StratifiedGroupKFold`` so **all windows of a trace
    land in one fold** — without this the strongly correlated within-trace
    windows leak across the split and inflate AUC toward 1.0. Out-of-fold
    ``predict_proba`` gives one honest probability per window; the per-trace
    score is the mean over that trace's windows, so ``pos``/``neg`` feed
    ``code.typeb.roc`` exactly like the spectral/Viterbi per-trace statistics.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedGroupKFold

    labels = np.asarray([bool(v) for v in labels])
    seed = int(p.seed if seed is None else seed)

    Xs, ys, grps = [], [], []
    names: list[str] = []
    for g, (tr, lab) in enumerate(zip(traces, labels)):
        Xi, names, _ = power_features(
            tr.t, tr.P_obs, fs=fs, window_s=window_s, stride_s=stride_s,
            autocorr_lags_s=p.autocorr_lags_s, idle_frac_tol=p.idle_frac_tol,
            drop_level=drop_level)
        if Xi.shape[0] == 0:
            continue
        Xs.append(Xi)
        ys.append(np.full(Xi.shape[0], lab))
        grps.append(np.full(Xi.shape[0], g))
    if not Xs:
        raise ValueError("no traces long enough for even one window")

    X = np.vstack(Xs)
    y = np.concatenate(ys)
    grp = np.concatenate(grps)

    skf = StratifiedGroupKFold(n_splits=p.n_splits, shuffle=True,
                               random_state=seed)
    oof = np.full(y.size, np.nan)
    importances = np.zeros(X.shape[1])
    n_folds = 0
    for tr_idx, te_idx in skf.split(X, y.astype(int), grp):
        clf = RandomForestClassifier(
            n_estimators=p.n_estimators, max_depth=p.max_depth,
            min_samples_leaf=p.min_samples_leaf, max_features=p.max_features,
            class_weight=p.class_weight, random_state=seed, n_jobs=-1)
        clf.fit(X[tr_idx], y[tr_idx].astype(int))
        pos_cols = np.where(clf.classes_ == 1)[0]
        if pos_cols.size:                          # both classes seen in train
            oof[te_idx] = clf.predict_proba(X[te_idx])[:, pos_cols[0]]
        else:                                      # degenerate fold: no positives
            oof[te_idx] = 0.0
        importances += clf.feature_importances_
        n_folds += 1
    importances /= max(n_folds, 1)

    win_pos = oof[y]
    win_neg = oof[~y]
    uniq = np.unique(grp)
    trace_score = np.array([oof[grp == g].mean() for g in uniq])
    trace_label = np.array([y[grp == g][0] for g in uniq])
    return RfResult(
        pos=trace_score[trace_label], neg=trace_score[~trace_label],
        win_pos=win_pos, win_neg=win_neg, feature_names=names,
        importances=importances, n_windows=int(y.size), drop_level=drop_level)


def rf_fit_predict(
    train_traces,
    train_labels,
    test_traces,
    p: RfBaselineParams,
    *,
    fs: float,
    window_s: float,
    stride_s: float,
    drop_level: bool = False,
    seed: int | None = None,
) -> np.ndarray:
    """Zero-shot / leave-one-strategy-out scoring: fit on ``train_traces``,
    return one score per ``test_trace`` (mean over its window probabilities).

    Unlike :func:`rf_oof_scores` (which cross-validates *within* one population,
    i.e. the classifier sees each family in training — Rahman's retrain-per-round
    setting), this trains once and applies the frozen classifier to a population
    it never saw. ``test_traces`` need not resemble ``train_traces``: that is the
    point — it measures how a supervised detector generalises to a *novel*
    adversary, the honest comparison to our never-retrained physics tracker.
    """
    from sklearn.ensemble import RandomForestClassifier

    seed = int(p.seed if seed is None else seed)
    labels = np.asarray([bool(v) for v in train_labels])
    fkw = dict(fs=fs, window_s=window_s, stride_s=stride_s,
               autocorr_lags_s=p.autocorr_lags_s,
               idle_frac_tol=p.idle_frac_tol, drop_level=drop_level)

    Xtr, ytr = [], []
    for tr, lab in zip(train_traces, labels):
        Xi, _, _ = power_features(tr.t, tr.P_obs, **fkw)
        if Xi.shape[0]:
            Xtr.append(Xi)
            ytr.append(np.full(Xi.shape[0], int(lab)))
    if not Xtr:
        raise ValueError("no training traces long enough for even one window")
    clf = RandomForestClassifier(
        n_estimators=p.n_estimators, max_depth=p.max_depth,
        min_samples_leaf=p.min_samples_leaf, max_features=p.max_features,
        class_weight=p.class_weight, random_state=seed, n_jobs=-1)
    clf.fit(np.vstack(Xtr), np.concatenate(ytr))
    pos_cols = np.where(clf.classes_ == 1)[0]
    pos_col = int(pos_cols[0]) if pos_cols.size else None

    out = []
    for tr in test_traces:
        Xi, _, _ = power_features(tr.t, tr.P_obs, **fkw)
        if Xi.shape[0] == 0 or pos_col is None:
            out.append(0.0)
        else:
            out.append(float(clf.predict_proba(Xi)[:, pos_col].mean()))
    return np.array(out)
