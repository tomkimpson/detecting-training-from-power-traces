"""Duration-aware compute/communication event detection.

This module implements the E0/E1 smoke-test path in ``spec.md``.  It deliberately
does not estimate or consume a global cadence.  A blind detector finds downward
power transitions, pairs them with recoveries, summarizes the resulting marked
events, and compares those summaries with profiled training/null distributions.

Ground-truth phase metadata is accepted only by :func:`oracle_trace_features`, an
evaluation ceiling.  The deployable interfaces -- :func:`blind_trace_features`
and :meth:`RenewalLikelihood.score` -- accept only ``(t, P_obs)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np
from scipy.ndimage import gaussian_filter1d, median_filter
from scipy.signal import find_peaks


FEATURE_NAMES: tuple[str, ...] = (
    "event_rate_hz",
    "paired_fraction",
    "duration_median_s",
    "duration_cv",
    "amplitude_median_z",
    "amplitude_cv",
    "plateau_abs_slope",
    "plateau_noise",
    "edge_balance",
    "shape_repeatability",
    "interval_cv",
)


@dataclass(frozen=True)
class EventParams:
    """Physical-time knobs for the blind event extractor."""

    baseline_window_s: float = 5.0
    smooth_sigma_s: float = 0.05
    edge_window_s: float = 0.15
    min_event_separation_s: float = 0.20
    min_dwell_s: float = 0.08
    max_dwell_s: float = 1.20
    threshold_mad: float = 1.5
    min_contrast_z: float = 0.60
    shape_pre_n: int = 4
    shape_low_n: int = 8
    shape_post_n: int = 4


@dataclass(frozen=True)
class MarkedEvent:
    """One paired downward transition and recovery."""

    down_index: int
    up_index: int
    down_time: float
    up_time: float
    duration_s: float
    amplitude_z: float
    plateau_abs_slope: float
    plateau_noise: float
    down_strength: float
    up_strength: float
    shape: np.ndarray


@dataclass(frozen=True)
class EventExtraction:
    """Blind candidate bookkeeping and paired event marks."""

    events: tuple[MarkedEvent, ...]
    n_down_candidates: int
    n_up_candidates: int

    @property
    def down_times(self) -> np.ndarray:
        return np.asarray([event.down_time for event in self.events], dtype=float)


@dataclass(frozen=True)
class GaussianProfile:
    """Regularised Gaussian profile over trace-level event features."""

    mean: np.ndarray
    precision: np.ndarray
    logdet: float

    def logpdf(self, features: np.ndarray) -> float:
        x = np.asarray(features, dtype=float)
        delta = x - self.mean
        d = self.mean.size
        return float(-0.5 * (d * np.log(2.0 * np.pi) + self.logdet
                             + delta @ self.precision @ delta))


@dataclass(frozen=True)
class RenewalLikelihood:
    """Profiled marked-renewal likelihood ratio with a composite null."""

    training: GaussianProfile
    nulls: tuple[GaussianProfile, ...]
    params: EventParams = EventParams()

    def score(self, t: np.ndarray, p_obs: np.ndarray) -> float:
        """Training-vs-strongest-null score using no latent metadata."""
        features = blind_trace_features(t, p_obs, params=self.params)
        return self.score_features(features)

    def score_features(self, features: np.ndarray) -> float:
        log_train = self.training.logpdf(features)
        log_null = max(profile.logpdf(features) for profile in self.nulls)
        return float(log_train - log_null)


def _sampling_rate(t: np.ndarray) -> float:
    t = np.asarray(t, dtype=float)
    if t.ndim != 1 or t.size < 4:
        raise ValueError("t must be a one-dimensional grid with at least 4 samples")
    dt = np.diff(t)
    if not np.all(np.isfinite(dt)) or np.any(dt <= 0.0):
        raise ValueError("t must be finite and strictly increasing")
    dt_med = float(np.median(dt))
    if not np.allclose(dt, dt_med, rtol=1e-4, atol=1e-9):
        raise ValueError("event extraction requires a uniformly sampled trace")
    return 1.0 / dt_med


def _odd_samples(seconds: float, fs: float, *, minimum: int = 1) -> int:
    n = max(int(round(seconds * fs)), minimum)
    return n if n % 2 == 1 else n + 1


def robust_event_series(
    t: np.ndarray,
    p_obs: np.ndarray,
    *,
    params: EventParams = EventParams(),
) -> np.ndarray:
    """Remove a slow local baseline, robustly scale, and lightly smooth."""
    fs = _sampling_rate(t)
    x = np.asarray(p_obs, dtype=float)
    if x.shape != np.asarray(t).shape or not np.all(np.isfinite(x)):
        raise ValueError("p_obs must be finite and have the same shape as t")

    baseline_n = _odd_samples(params.baseline_window_s, fs, minimum=3)
    baseline = median_filter(x, size=baseline_n, mode="nearest")
    residual = x - baseline
    centre = float(np.median(residual))
    mad = 1.4826 * float(np.median(np.abs(residual - centre)))
    std = float(np.std(residual))
    scale = max(mad, 0.1 * std, np.finfo(float).eps)
    z = (residual - centre) / scale
    sigma = params.smooth_sigma_s * fs
    return gaussian_filter1d(z, sigma=sigma, mode="nearest") if sigma > 0.25 else z


def _edge_contrasts(z: np.ndarray, edge_n: int) -> tuple[np.ndarray, np.ndarray]:
    """Mean-before minus mean-after contrast and its valid-index mask."""
    n = z.size
    contrast = np.zeros(n, dtype=float)
    valid = np.zeros(n, dtype=bool)
    if n < 2 * edge_n + 1:
        return contrast, valid
    csum = np.concatenate([[0.0], np.cumsum(z)])
    idx = np.arange(edge_n, n - edge_n)
    before = (csum[idx] - csum[idx - edge_n]) / edge_n
    after = (csum[idx + edge_n] - csum[idx]) / edge_n
    contrast[idx] = before - after
    valid[idx] = True
    return contrast, valid


def _resample_segment(values: np.ndarray, n_out: int) -> np.ndarray:
    if values.size == 0:
        return np.zeros(n_out, dtype=float)
    if values.size == 1:
        return np.full(n_out, float(values[0]))
    src = np.linspace(0.0, 1.0, values.size)
    dst = np.linspace(0.0, 1.0, n_out)
    return np.interp(dst, src, values)


def _make_event(
    t: np.ndarray,
    z: np.ndarray,
    down: int,
    up: int,
    down_strength: float,
    up_strength: float,
    edge_n: int,
    params: EventParams,
) -> MarkedEvent | None:
    if up <= down or down < edge_n or up + edge_n > z.size:
        return None
    pre = z[down - edge_n:down]
    low = z[down:up]
    post = z[up:up + edge_n]
    if pre.size == 0 or low.size == 0 or post.size == 0:
        return None

    high_level = 0.5 * (float(np.median(pre)) + float(np.median(post)))
    low_level = float(np.median(low))
    amplitude = high_level - low_level
    if not np.isfinite(amplitude) or amplitude <= 0.0:
        return None

    if low.size >= 3:
        low_t = t[down:up] - t[down]
        slope, intercept = np.polyfit(low_t, low, 1)
        residual = low - (slope * low_t + intercept)
        plateau_slope = abs(float(slope)) / amplitude
        plateau_noise = (1.4826 * float(np.median(
            np.abs(residual - np.median(residual))
        ))) / amplitude
    else:
        plateau_slope = 0.0
        plateau_noise = 0.0

    raw_shape = np.concatenate([
        _resample_segment(pre, params.shape_pre_n),
        _resample_segment(low, params.shape_low_n),
        _resample_segment(post, params.shape_post_n),
    ])
    shape = (raw_shape - high_level) / amplitude
    return MarkedEvent(
        down_index=int(down),
        up_index=int(up),
        down_time=float(t[down]),
        up_time=float(t[up]),
        duration_s=float(t[up] - t[down]),
        amplitude_z=float(amplitude),
        plateau_abs_slope=float(plateau_slope),
        plateau_noise=float(plateau_noise),
        down_strength=float(down_strength),
        up_strength=float(up_strength),
        shape=shape,
    )


def extract_events(
    t: np.ndarray,
    p_obs: np.ndarray,
    *,
    params: EventParams = EventParams(),
) -> EventExtraction:
    """Blindly extract paired down/up events without estimating a cadence."""
    fs = _sampling_rate(t)
    z = robust_event_series(t, p_obs, params=params)
    edge_n = max(int(round(params.edge_window_s * fs)), 1)
    drop, valid = _edge_contrasts(z, edge_n)
    valid_drop = drop[valid]
    centre = float(np.median(valid_drop)) if valid_drop.size else 0.0
    mad = 1.4826 * float(np.median(np.abs(valid_drop - centre))) \
        if valid_drop.size else 0.0
    threshold = max(params.min_contrast_z, centre + params.threshold_mad * mad)
    min_sep = max(int(round(params.min_event_separation_s * fs)), 1)
    down_idx, _ = find_peaks(drop, height=threshold, distance=min_sep)
    up_idx, _ = find_peaks(-drop, height=threshold, distance=min_sep)

    min_dwell = max(int(round(params.min_dwell_s * fs)), 1)
    max_dwell = max(int(round(params.max_dwell_s * fs)), min_dwell + 1)
    used_up: set[int] = set()
    events: list[MarkedEvent] = []
    for j, down in enumerate(down_idx):
        stop = min(down + max_dwell, z.size - edge_n)
        if j + 1 < down_idx.size:
            stop = min(stop, int(down_idx[j + 1]))
        candidates = [int(up) for up in up_idx
                      if down + min_dwell <= up <= stop and int(up) not in used_up]
        if not candidates:
            continue
        up = max(candidates, key=lambda idx: -drop[idx])
        event = _make_event(t, z, int(down), up, float(drop[down]),
                            float(-drop[up]), edge_n, params)
        if event is not None:
            events.append(event)
            used_up.add(up)
    return EventExtraction(tuple(events), int(down_idx.size), int(up_idx.size))


def _events_from_boundaries(
    t: np.ndarray,
    p_obs: np.ndarray,
    communication_starts: np.ndarray,
    communication_durations: np.ndarray,
    *,
    params: EventParams,
) -> tuple[MarkedEvent, ...]:
    """Construct oracle event marks from latent boundaries for E0 only."""
    fs = _sampling_rate(t)
    z = robust_event_series(t, p_obs, params=params)
    edge_n = max(int(round(params.edge_window_s * fs)), 1)
    drop, _ = _edge_contrasts(z, edge_n)
    events: list[MarkedEvent] = []
    for start, duration in zip(communication_starts, communication_durations):
        down = int(np.searchsorted(t, start, side="left"))
        up = int(np.searchsorted(t, start + duration, side="left"))
        if 0 <= down < drop.size and 0 <= up < drop.size:
            event = _make_event(t, z, down, up, float(drop[down]),
                                float(-drop[up]), edge_n, params)
            if event is not None:
                events.append(event)
    return tuple(events)


def _coefficient_of_variation(values: np.ndarray) -> float:
    if values.size < 2:
        return 0.0
    mean = abs(float(np.mean(values)))
    return float(np.std(values) / mean) if mean > 1e-12 else 0.0


def _shape_repeatability(events: Sequence[MarkedEvent]) -> float:
    if not events:
        return 0.0
    shapes = np.stack([event.shape for event in events])
    centre = np.median(shapes, axis=0)
    rmse = np.sqrt(np.mean((shapes - centre) ** 2, axis=1))
    return float(1.0 / (1.0 + np.median(rmse)))


def summarize_events(
    t: np.ndarray,
    events: Sequence[MarkedEvent],
    *,
    paired_fraction: float,
) -> np.ndarray:
    """Fixed, interpretable trace-level feature vector."""
    duration = max(float(t[-1] - t[0]), np.finfo(float).eps)
    if not events:
        return np.zeros(len(FEATURE_NAMES), dtype=float)
    durations = np.asarray([event.duration_s for event in events])
    amplitudes = np.asarray([event.amplitude_z for event in events])
    slopes = np.asarray([event.plateau_abs_slope for event in events])
    noises = np.asarray([event.plateau_noise for event in events])
    balances = np.asarray([
        min(event.down_strength, event.up_strength)
        / max(event.down_strength, event.up_strength, 1e-12)
        for event in events
    ])
    times = np.asarray([event.down_time for event in events])
    intervals = np.diff(times)
    return np.asarray([
        len(events) / duration,
        paired_fraction,
        float(np.median(durations)),
        _coefficient_of_variation(durations),
        float(np.median(amplitudes)),
        _coefficient_of_variation(amplitudes),
        float(np.median(slopes)),
        float(np.median(noises)),
        float(np.median(balances)),
        _shape_repeatability(events),
        _coefficient_of_variation(intervals),
    ], dtype=float)


def blind_trace_features(
    t: np.ndarray,
    p_obs: np.ndarray,
    *,
    params: EventParams = EventParams(),
) -> np.ndarray:
    """E1 features from ``(t, P_obs)`` only; no phase/cadence input exists."""
    extracted = extract_events(t, p_obs, params=params)
    paired = len(extracted.events) / max(extracted.n_down_candidates, 1)
    return summarize_events(t, extracted.events, paired_fraction=paired)


def oracle_trace_features(
    t: np.ndarray,
    p_obs: np.ndarray,
    communication_starts: np.ndarray,
    communication_durations: np.ndarray,
    *,
    params: EventParams = EventParams(),
) -> np.ndarray:
    """E0 information-ceiling features; never a deployable score."""
    events = _events_from_boundaries(
        t, p_obs, np.asarray(communication_starts, dtype=float),
        np.asarray(communication_durations, dtype=float), params=params,
    )
    return summarize_events(t, events, paired_fraction=1.0 if events else 0.0)


def event_timing_diagnostics(
    predicted_down_times: np.ndarray,
    true_down_times: np.ndarray,
    *,
    tolerance_s: float = 0.1,
) -> tuple[float, float, float]:
    """Return (recall, precision, median absolute timing error) with greedy matches."""
    predicted = np.sort(np.asarray(predicted_down_times, dtype=float))
    truth = np.sort(np.asarray(true_down_times, dtype=float))
    used: set[int] = set()
    errors: list[float] = []
    for target in truth:
        candidates = [(abs(float(value - target)), idx)
                      for idx, value in enumerate(predicted) if idx not in used]
        if not candidates:
            continue
        error, idx = min(candidates)
        if error <= tolerance_s:
            used.add(idx)
            errors.append(error)
    recall = len(errors) / max(truth.size, 1)
    precision = len(errors) / max(predicted.size, 1)
    median_error = float(np.median(errors)) if errors else float("inf")
    return float(recall), float(precision), median_error


def fit_gaussian_profile(
    features: np.ndarray,
    *,
    shrinkage: float = 0.25,
) -> GaussianProfile:
    """Fit a regularised full-covariance Gaussian feature profile."""
    x = np.asarray(features, dtype=float)
    if x.ndim != 2 or x.shape[0] < 2 or x.shape[1] != len(FEATURE_NAMES):
        raise ValueError("features must have shape (n>=2, len(FEATURE_NAMES))")
    if not np.all(np.isfinite(x)):
        raise ValueError("features must be finite")
    mean = x.mean(axis=0)
    cov = np.atleast_2d(np.cov(x, rowvar=False, ddof=1))
    diagonal = np.diag(np.diag(cov))
    cov = (1.0 - shrinkage) * cov + shrinkage * diagonal
    scale = max(float(np.trace(cov) / cov.shape[0]), 1.0)
    cov = cov + np.eye(cov.shape[0]) * (1e-6 * scale)
    sign, logdet = np.linalg.slogdet(cov)
    if sign <= 0:
        raise ValueError("regularised covariance is not positive definite")
    return GaussianProfile(mean=mean, precision=np.linalg.inv(cov),
                           logdet=float(logdet))


def fit_likelihood_from_features(
    training_features: np.ndarray,
    null_feature_groups: Iterable[np.ndarray],
    *,
    params: EventParams = EventParams(),
) -> RenewalLikelihood:
    """Fit the profiled E1 likelihood from disjoint feature matrices."""
    nulls = tuple(fit_gaussian_profile(group) for group in null_feature_groups)
    if not nulls:
        raise ValueError("at least one null feature group is required")
    return RenewalLikelihood(fit_gaussian_profile(training_features), nulls,
                             params=params)


def fit_renewal_likelihood(
    training_traces: Sequence,
    null_trace_groups: Mapping[str, Sequence],
    *,
    params: EventParams = EventParams(),
) -> RenewalLikelihood:
    """Fit a blind E1 model from trace objects carrying ``t`` and ``P_obs``."""
    training = np.stack([
        blind_trace_features(trace.t, trace.P_obs, params=params)
        for trace in training_traces
    ])
    nulls = [
        np.stack([
            blind_trace_features(trace.t, trace.P_obs, params=params)
            for trace in traces
        ])
        for traces in null_trace_groups.values()
    ]
    return fit_likelihood_from_features(training, nulls, params=params)
