"""Hurdle likelihood for marked events, with honest missing-feature handling.

The E0/E1 feature path (:mod:`powerladder.typeb.renewal`) represents "no events
were extracted" as eleven continuous zeros and evaluates that under a Gaussian
density.  That is wrong in two separate ways:

1. **A point mass is not an eleven-dimensional signature.**  ``N = 0`` is a single
   discrete observation; scoring it as a vector in mark space lets the Gaussian
   manufacture arbitrary evidence from a coordinate that carries none.
2. **Undefined is not zero.**  With one event a coefficient of variation, a
   repeatability and an inter-event interval do not exist.  The legacy code
   returns ``0.0`` for the CVs and ``1.0`` -- *perfect* regularity -- for shape
   repeatability, so an under-determined statistic is read as strong evidence.

A high no-event rate among physical nulls is a **genuine property** of those
nulls at short records, not an artefact to be engineered away (18.6% of composite
-null traces at 10 s).  It should be legitimate, explicit, auditable evidence.
This module makes it so:

.. math::

    \\ell_M(x) = \\begin{cases}
      \\log p_M(N=0) & N = 0 \\\\
      \\log(1 - p_M(N=0)) + \\log p_M(N \\mid N>0, T)
        + \\log p_M(z_{\\text{obs}} \\mid N) & N > 0
    \\end{cases}

with the composite structure retained,
``S(x) = l_train(x) - max_j l_null_j(x)``.

* ``p_M(N=0)`` is fitted with Jeffreys smoothing, so it is never 0 or 1.
* ``p_M(N \\mid N>0, T)`` is a zero-truncated negative binomial (Poisson when the
  sample is under-dispersed), on counts at a recorded exposure ``T``.
* ``p_M(z_obs \\mid N)`` evaluates the Gaussian **marginal** over exactly the
  marks that are defined at that event count.  Undefined marks are marginalised
  out, never replaced by a sentinel.

The legacy path is deliberately left untouched: the pre-fix confirmation in
``notes/results/hsmm-cell-confirmation-and-e2-precommitment.md`` stays frozen as
the audited result, and its constants must not be silently mixed with corrected
ones.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.stats import nbinom, poisson

from powerladder.typeb.renewal import (
    EventParams,
    MarkedEvent,
    _events_from_boundaries,
    extract_events,
)


#: Mark summaries, with the minimum event count at which each is *defined*.
#: ``event_rate_hz`` is absent by design -- the count is modelled explicitly by
#: the hurdle's count term, so keeping a rate feature would double-count it.
MARK_SPEC: tuple[tuple[str, int], ...] = (
    ("paired_fraction", 1),
    ("duration_median_s", 1),
    ("amplitude_median_z", 1),
    ("plateau_abs_slope", 1),
    ("plateau_noise", 1),
    ("edge_balance", 1),
    # A coefficient of variation needs two observations to vary.
    ("duration_cv", 2),
    ("amplitude_cv", 2),
    # Repeatability needs a second event to be repeatable against.
    ("shape_repeatability", 2),
    # Interval variation needs two intervals, hence three events.
    ("interval_cv", 3),
)

MARK_NAMES: tuple[str, ...] = tuple(name for name, _ in MARK_SPEC)
MARK_MIN_EVENTS: tuple[int, ...] = tuple(minimum for _, minimum in MARK_SPEC)


def _cv(values: np.ndarray) -> float:
    """Coefficient of variation, or NaN when there is nothing to vary."""
    if values.size < 2:
        return float("nan")
    mean = abs(float(np.mean(values)))
    return float(np.std(values, ddof=1) / mean) if mean > 1e-12 else float("nan")


def _repeatability(events: Sequence[MarkedEvent]) -> float:
    """Within-trace shape repeatability, or NaN for a single event.

    The legacy implementation returns 1.0 for one event -- perfect repeatability
    inferred from one observation.  It is an unavailable statistic, not evidence.
    """
    if len(events) < 2:
        return float("nan")
    shapes = np.stack([event.shape for event in events])
    centre = np.median(shapes, axis=0)
    rmse = np.sqrt(np.mean((shapes - centre) ** 2, axis=1))
    return float(1.0 / (1.0 + np.median(rmse)))


def mark_vector(events: Sequence[MarkedEvent], *,
                paired_fraction: float) -> np.ndarray:
    """Marks defined at this event count; ``np.nan`` where undefined."""
    marks = np.full(len(MARK_NAMES), np.nan, dtype=float)
    if not events:
        return marks
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

    values = {
        "paired_fraction": float(paired_fraction),
        "duration_median_s": float(np.median(durations)),
        "amplitude_median_z": float(np.median(amplitudes)),
        "plateau_abs_slope": float(np.median(slopes)),
        "plateau_noise": float(np.median(noises)),
        "edge_balance": float(np.median(balances)),
        "duration_cv": _cv(durations),
        "amplitude_cv": _cv(amplitudes),
        "shape_repeatability": _repeatability(events),
        "interval_cv": _cv(np.diff(times)) if times.size >= 3 else float("nan"),
    }
    for j, name in enumerate(MARK_NAMES):
        marks[j] = values[name]
    return marks


@dataclass(frozen=True)
class Observation:
    """One trace reduced to a count, an exposure and its defined marks."""

    n_events: int
    exposure_s: float
    marks: np.ndarray

    @property
    def available(self) -> np.ndarray:
        return np.isfinite(self.marks)


def observe(t: np.ndarray, p_obs: np.ndarray, *,
            params: EventParams = EventParams()) -> Observation:
    """Blind observation: consumes ``(t, P_obs)`` only, as E1 must."""
    extracted = extract_events(t, p_obs, params=params)
    paired = len(extracted.events) / max(extracted.n_down_candidates, 1)
    exposure = float(np.asarray(t)[-1] - np.asarray(t)[0])
    return Observation(
        n_events=len(extracted.events),
        exposure_s=max(exposure, np.finfo(float).eps),
        marks=mark_vector(extracted.events, paired_fraction=paired),
    )


def observation_from_events(events: Sequence[MarkedEvent], exposure_s: float, *,
                            paired_fraction: float) -> Observation:
    """Observation from pre-built events (the E0 aligned-window path)."""
    return Observation(
        n_events=len(events),
        exposure_s=max(float(exposure_s), np.finfo(float).eps),
        marks=mark_vector(events, paired_fraction=paired_fraction),
    )


def aligned_observation(t: np.ndarray, p_obs: np.ndarray,
                        starts: np.ndarray, durations: np.ndarray, *,
                        params: EventParams = EventParams()) -> Observation:
    """E0's windowed observation, for true *or* random alignments.

    ``paired_fraction`` is 1.0 by construction here, exactly as in the legacy
    oracle path -- but both sides of the E0 comparison are built this way, so it
    takes the same value for positives and nulls and therefore carries no
    discriminative information.  That is the point of the matched-window control.
    """
    events = _events_from_boundaries(
        t, p_obs, np.asarray(starts, dtype=float),
        np.asarray(durations, dtype=float), params=params,
    )
    exposure = float(np.asarray(t)[-1] - np.asarray(t)[0])
    return observation_from_events(events, exposure,
                                   paired_fraction=1.0 if events else 0.0)


def random_aligned_observations(traces, counts: np.ndarray,
                                durations: np.ndarray, seed: int, *,
                                params: EventParams = EventParams()) -> list:
    """Windows at uniformly random locations, counts/durations resampled."""
    rng = np.random.default_rng(seed)
    out = []
    for trace in traces:
        n = int(rng.choice(counts))
        drawn = rng.choice(durations, size=max(n, 1), replace=True)
        latest = float(np.asarray(trace.t)[-1] - drawn.max())
        if n == 0 or latest <= float(np.asarray(trace.t)[0]):
            out.append(Observation(0, max(float(np.asarray(trace.t)[-1]
                                          - np.asarray(trace.t)[0]), 1e-12),
                                   np.full(len(MARK_NAMES), np.nan)))
            continue
        starts = np.sort(rng.uniform(float(np.asarray(trace.t)[0]), latest, n))
        out.append(aligned_observation(trace.t, trace.P_obs, starts, drawn,
                                       params=params))
    return out


@dataclass(frozen=True)
class CountModel:
    """``p(N | N > 0, T)`` -- zero-truncated NB, or Poisson if under-dispersed."""

    kind: str
    mean: float
    r: float
    p: float
    exposure_s: float

    def logpmf(self, n: int) -> float:
        if n < 1:
            return -np.inf
        if self.kind == "poisson":
            log_p = poisson.logpmf(n, self.mean)
            log_zero = poisson.logpmf(0, self.mean)
        else:
            log_p = nbinom.logpmf(n, self.r, self.p)
            log_zero = nbinom.logpmf(0, self.r, self.p)
        # Renormalise over N >= 1.
        return float(log_p - np.log1p(-np.exp(log_zero)))


def fit_count_model(counts: np.ndarray, exposure_s: float) -> CountModel:
    """Method-of-moments fit on the positive counts."""
    positive = np.asarray(counts, dtype=float)
    positive = positive[positive > 0]
    if positive.size == 0:
        return CountModel("poisson", 1.0, np.nan, np.nan, exposure_s)
    mean = float(positive.mean())
    variance = float(positive.var(ddof=1)) if positive.size > 1 else mean
    if variance <= mean or not np.isfinite(variance):
        return CountModel("poisson", max(mean, 1e-6), np.nan, np.nan, exposure_s)
    p = mean / variance
    r = mean * p / (1.0 - p)
    return CountModel("nbinom", mean, float(r), float(p), exposure_s)


@dataclass(frozen=True)
class MarkModel:
    """Gaussian over marks, evaluated on the marginal that is defined."""

    mean: np.ndarray
    cov: np.ndarray

    def logpdf(self, marks: np.ndarray) -> float:
        available = np.isfinite(marks)
        if not available.any():
            return 0.0          # nothing observed contributes nothing
        idx = np.flatnonzero(available)
        mu = self.mean[idx]
        sigma = self.cov[np.ix_(idx, idx)]
        delta = marks[idx] - mu
        sign, logdet = np.linalg.slogdet(sigma)
        if sign <= 0:
            sigma = sigma + np.eye(idx.size) * 1e-9
            sign, logdet = np.linalg.slogdet(sigma)
        solved = np.linalg.solve(sigma, delta)
        return float(-0.5 * (idx.size * np.log(2.0 * np.pi) + logdet
                             + delta @ solved))


def fit_mark_model(mark_rows: np.ndarray, *, shrinkage: float = 0.25) -> MarkModel:
    """Fit the mark Gaussian, using each column's available observations.

    Complete rows give the covariance when there are enough of them; otherwise a
    diagonal covariance is built from per-column available observations.  Either
    way the mean is per-column, so a mark is never informed by rows where it was
    undefined.
    """
    rows = np.atleast_2d(np.asarray(mark_rows, dtype=float))
    if rows.shape[1] != len(MARK_NAMES):
        raise ValueError(f"expected {len(MARK_NAMES)} mark columns")
    available = np.isfinite(rows)
    if not available.any():
        raise ValueError("no available marks to fit")

    mean = np.zeros(rows.shape[1])
    variance = np.ones(rows.shape[1])
    for j in range(rows.shape[1]):
        column = rows[available[:, j], j]
        if column.size == 0:
            continue
        mean[j] = float(column.mean())
        variance[j] = float(column.var(ddof=1)) if column.size > 1 else 1.0
    variance = np.maximum(variance, 1e-12)

    complete = np.all(available, axis=1)
    if complete.sum() >= max(2 * rows.shape[1], 10):
        cov = np.atleast_2d(np.cov(rows[complete], rowvar=False, ddof=1))
        cov = (1.0 - shrinkage) * cov + shrinkage * np.diag(np.diag(cov))
        # Keep the per-column means, which use every available observation.
        np.fill_diagonal(cov, np.maximum(np.diag(cov), 1e-12))
    else:
        cov = np.diag(variance)
    scale = max(float(np.trace(cov) / cov.shape[0]), 1.0)
    return MarkModel(mean=mean, cov=cov + np.eye(cov.shape[0]) * 1e-6 * scale)


@dataclass(frozen=True)
class HurdleProfile:
    """One class's hurdle likelihood."""

    log_p_zero: float
    log_p_nonzero: float
    counts: CountModel
    marks: MarkModel
    zero_event_rate: float

    def loglike(self, observation: Observation, *,
                counts_only: bool = False) -> float:
        if observation.n_events == 0:
            return self.log_p_zero
        total = self.log_p_nonzero + self.counts.logpmf(observation.n_events)
        if counts_only:
            return total
        return total + self.marks.logpdf(observation.marks)


def fit_hurdle_profile(observations: Sequence[Observation], *,
                       shrinkage: float = 0.25) -> HurdleProfile:
    """Fit a class profile; the zero rate uses Jeffreys smoothing."""
    if not observations:
        raise ValueError("at least one observation is required")
    counts = np.asarray([o.n_events for o in observations], dtype=float)
    n_zero = int((counts == 0).sum())
    n_total = counts.size
    # Jeffreys (Beta(1/2, 1/2)) posterior mean: never exactly 0 or 1, so a class
    # that happened to produce no empty traces cannot veto one at scoring time.
    p_zero = (n_zero + 0.5) / (n_total + 1.0)
    positive = [o for o in observations if o.n_events > 0]
    exposure = float(np.median([o.exposure_s for o in observations]))
    if positive:
        mark_rows = np.stack([o.marks for o in positive])
        mark_model = fit_mark_model(mark_rows, shrinkage=shrinkage)
        count_model = fit_count_model(counts, exposure)
    else:
        mark_model = MarkModel(np.zeros(len(MARK_NAMES)),
                               np.eye(len(MARK_NAMES)))
        count_model = fit_count_model(np.asarray([1.0]), exposure)
    return HurdleProfile(
        log_p_zero=float(np.log(p_zero)),
        log_p_nonzero=float(np.log1p(-p_zero)),
        counts=count_model,
        marks=mark_model,
        zero_event_rate=float(n_zero / n_total),
    )


@dataclass(frozen=True)
class HurdleLikelihood:
    """Composite likelihood ratio over hurdle profiles."""

    training: HurdleProfile
    nulls: tuple[HurdleProfile, ...]
    params: EventParams = EventParams()

    def score_observation(self, observation: Observation, *,
                          counts_only: bool = False) -> float:
        train = self.training.loglike(observation, counts_only=counts_only)
        null = max(profile.loglike(observation, counts_only=counts_only)
                   for profile in self.nulls)
        return float(train - null)

    def score(self, t: np.ndarray, p_obs: np.ndarray) -> float:
        """Deployable score: consumes ``(t, P_obs)`` only."""
        return self.score_observation(observe(t, p_obs, params=self.params))


def fit_hurdle_likelihood(training: Sequence[Observation],
                          nulls: Sequence[Sequence[Observation]], *,
                          shrinkage: float = 0.25,
                          params: EventParams = EventParams()
                          ) -> HurdleLikelihood:
    profiles = tuple(fit_hurdle_profile(group, shrinkage=shrinkage)
                     for group in nulls)
    if not profiles:
        raise ValueError("at least one null group is required")
    return HurdleLikelihood(fit_hurdle_profile(training, shrinkage=shrinkage),
                            profiles, params=params)


def zero_event_rate(observations: Sequence[Observation]) -> float:
    """Reported per class and per null family, replacing the degeneracy gate."""
    if not observations:
        return 0.0
    return float(np.mean([o.n_events == 0 for o in observations]))
