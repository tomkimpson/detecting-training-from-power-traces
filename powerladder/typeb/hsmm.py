"""E2: a two-state explicit-duration HSMM, and its duration-neutral controls.

The question this exists to answer (agreed 2026-08-08):

> At the fixed 10 s, 20%-share operating point, does explicit-duration
> marginalisation separate training from an IAAFT phase-randomised surrogate
> materially better than E1 and otherwise identical duration-neutral sequence
> models?

Structure follows ``spec.md`` §E2: two mandatory-alternating states,

    COMPUTE -> COMMUNICATION -> COMPUTE -> ...

robust emissions on normalised power and local slope, a broad compute-state
duration law and a separate communication-state law, and a decision statistic
that is the **forward marginal likelihood over all admissible segmentations** --
not the maximum-likelihood path.  The composite structure is retained,
``S(x) = log p(x | M_train) - max_j log p(x | M_null,j)``.

Three model variants share one code path, so "otherwise identical" is literal --
only the duration law differs:

* :data:`EXPLICIT` -- the fitted per-state duration pmf;
* :data:`GEOMETRIC` -- a geometric law with the same mean.  An HSMM whose
  advantage survives here is not using duration *shape*; a plain HMM has
  geometric dwell times by construction, so this is the duration-neutral
  sequence model.
* :data:`SHUFFLED` -- the fitted pmf with its probabilities permuted across the
  support.  Same entropy, same support, same parameter count, wrong durations.

**Fitting is symmetric.**  Both the training and the null models are fitted from
the *blind* event extractor's segmentation on a disjoint fitting split.
``spec.md`` permits profiling the training model from latent labels for a first
smoke; that is deliberately not the default here, because giving one class ground
truth and the other an estimator is precisely the asymmetry that invalidated the
original E0 gate.  Latent-label fitting remains available, explicitly flagged, as
a "best plausible chance" diagnostic.

Nothing in the scoring path consumes metadata: :meth:`HSMMComposite.score` takes
``(t, P_obs)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.special import logsumexp

from powerladder.typeb.renewal import (
    EventParams,
    extract_events,
    robust_event_series,
)


COMPUTE, COMMUNICATION = 0, 1
N_STATES = 2

EXPLICIT = "explicit"
GEOMETRIC = "geometric"
SHUFFLED = "shuffled"
DURATION_KINDS = (EXPLICIT, GEOMETRIC, SHUFFLED)

_MIN_VARIANCE = 1e-6


def observation_matrix(t: np.ndarray, p_obs: np.ndarray, *,
                       params: EventParams = EventParams()) -> np.ndarray:
    """``(2, T)`` observations: robust normalised power and its local slope.

    Both channels come from ``(t, P_obs)`` alone.  The slope is the gradient of
    the same robustly-normalised series the event extractor uses, so the HSMM and
    E1 see the same preprocessing and differ only in what they do with it.
    """
    z = robust_event_series(t, p_obs, params=params)
    fs = 1.0 / float(np.median(np.diff(np.asarray(t, dtype=float))))
    slope = np.gradient(gaussian_filter1d(z, sigma=max(0.05 * fs, 0.5),
                                          mode="nearest")) * fs
    return np.vstack([z, slope])


@dataclass(frozen=True)
class Emissions:
    """Per-state diagonal Gaussian on the two observation channels."""

    mean: np.ndarray          # (2 states, 2 channels)
    variance: np.ndarray      # (2 states, 2 channels)

    def loglike(self, observations: np.ndarray) -> np.ndarray:
        """``(2, T)`` per-sample log density under each state."""
        x = observations.T[None, :, :]                     # (1, T, 2)
        mean = self.mean[:, None, :]                       # (2, 1, 2)
        variance = self.variance[:, None, :]
        quad = (x - mean) ** 2 / variance
        return -0.5 * (np.log(2.0 * np.pi * variance) + quad).sum(axis=2)


@dataclass(frozen=True)
class DurationLaw:
    """Log-pmf and log-survival over dwell lengths ``1..support``."""

    logpmf: np.ndarray
    logsurv: np.ndarray

    @property
    def support(self) -> int:
        return int(self.logpmf.size)


def _normalise_log(values: np.ndarray) -> np.ndarray:
    return values - logsumexp(values)


def _survival(logpmf: np.ndarray) -> np.ndarray:
    """``log P(D >= d)`` for d = 1..support."""
    pmf = np.exp(logpmf)
    tail = np.cumsum(pmf[::-1])[::-1]
    return np.log(np.maximum(tail, 1e-300))


def duration_law(durations: np.ndarray, support: int, kind: str,
                 rng: np.random.Generator | None = None) -> DurationLaw:
    """Fit a duration law of the requested kind over ``1..support`` samples."""
    if kind not in DURATION_KINDS:
        raise ValueError(f"unknown duration kind {kind!r}")
    counts = np.bincount(np.clip(np.asarray(durations, dtype=int), 1, support),
                         minlength=support + 1)[1:].astype(float)
    # Total pseudo-count mass of one, spread over the support.  Half a count per
    # bin -- the usual Jeffreys choice -- would add ~100 pseudo-counts across a
    # 200-sample support and flatten the very duration law being measured.
    counts += 1.0 / support
    logpmf = _normalise_log(np.log(counts))

    if kind == GEOMETRIC:
        # Same mean, geometric shape: the duration-neutral sequence model.
        mean = float(np.exp(logsumexp(logpmf + np.log(np.arange(1, support + 1)))))
        p = float(np.clip(1.0 / max(mean, 1.0 + 1e-6), 1e-6, 1.0 - 1e-6))
        d = np.arange(1, support + 1)
        logpmf = _normalise_log((d - 1) * np.log1p(-p) + np.log(p))
    elif kind == SHUFFLED:
        # Same entropy, same support, wrong durations.
        generator = rng if rng is not None else np.random.default_rng(0)
        logpmf = logpmf[generator.permutation(support)]
        logpmf = _normalise_log(logpmf)
    return DurationLaw(logpmf=logpmf, logsurv=_survival(logpmf))


@dataclass(frozen=True)
class TwoStateHSMM:
    """Explicit-duration HSMM with mandatory alternation."""

    emissions: Emissions
    durations: tuple[DurationLaw, DurationLaw]
    log_initial: np.ndarray

    def loglike(self, observations: np.ndarray) -> float:
        """Forward marginal over every admissible segmentation, right-censored."""
        logb = self.emissions.loglike(observations)
        n = logb.shape[1]
        cum = np.concatenate([np.zeros((N_STATES, 1)),
                              np.cumsum(logb, axis=1)], axis=1)

        # alpha[j, t]: a state-j segment *ends* exactly at sample t.
        alpha = np.full((N_STATES, n + 1), -np.inf)
        for t in range(1, n + 1):
            for j in range(N_STATES):
                law = self.durations[j]
                d_max = min(law.support, t)
                d = np.arange(1, d_max + 1)
                starts = t - d
                segment = cum[j, t] - cum[j, starts]
                previous = np.where(starts == 0, self.log_initial[j],
                                    alpha[1 - j, starts])
                alpha[j, t] = logsumexp(law.logpmf[:d_max] + segment + previous)

        # The record ends mid-segment: weight the final dwell by its survival.
        terms = []
        for j in range(N_STATES):
            law = self.durations[j]
            d_max = min(law.support, n)
            d = np.arange(1, d_max + 1)
            starts = n - d
            segment = cum[j, n] - cum[j, starts]
            previous = np.where(starts == 0, self.log_initial[j],
                                alpha[1 - j, starts])
            terms.append(logsumexp(law.logsurv[:d_max] + segment + previous))
        return float(logsumexp(terms))


@dataclass(frozen=True)
class Segmentation:
    """Per-state sample assignments and dwell lengths from one trace."""

    samples: list[np.ndarray]       # per state, concatenated observation columns
    durations: list[list[int]]      # per state, dwell lengths in samples


def blind_segmentation(t: np.ndarray, p_obs: np.ndarray, *,
                       params: EventParams = EventParams()) -> Segmentation:
    """Segment a trace using the blind extractor -- no metadata, either class."""
    observations = observation_matrix(t, p_obs, params=params)
    extracted = extract_events(t, p_obs, params=params)
    n = observations.shape[1]
    labels = np.zeros(n, dtype=int)
    for event in extracted.events:
        labels[event.down_index:event.up_index] = COMMUNICATION

    samples = [observations[:, labels == j].T for j in range(N_STATES)]
    durations: list[list[int]] = [[], []]
    if n:
        edges = np.flatnonzero(np.diff(labels)) + 1
        for start, end in zip(np.r_[0, edges], np.r_[edges, n]):
            durations[int(labels[start])].append(int(end - start))
    return Segmentation(samples=samples, durations=durations)


def latent_segmentation(trace, *, params: EventParams = EventParams()
                        ) -> Segmentation:
    """Ground-truth segmentation -- fitting diagnostic only, never for scoring.

    Using this for the training class while the nulls are fitted blindly makes
    the comparison asymmetric in exactly the way that invalidated the original
    E0 gate.  It is provided so an E2 failure can be checked against the model's
    best plausible chance, and must be reported as such.
    """
    observations = observation_matrix(trace.t, trace.P_obs, params=params)
    times = np.asarray(trace.t, dtype=float)
    labels = np.zeros(observations.shape[1], dtype=int)
    for start, dwell in zip(trace.meta.communication_starts,
                            trace.meta.communication_durations):
        lo = int(np.searchsorted(times, start, side="left"))
        hi = int(np.searchsorted(times, start + dwell, side="left"))
        labels[lo:hi] = COMMUNICATION
    samples = [observations[:, labels == j].T for j in range(N_STATES)]
    durations: list[list[int]] = [[], []]
    n = labels.size
    if n:
        edges = np.flatnonzero(np.diff(labels)) + 1
        for lo, hi in zip(np.r_[0, edges], np.r_[edges, n]):
            durations[int(labels[lo])].append(int(hi - lo))
    return Segmentation(samples=samples, durations=durations)


def fit_hsmm(segmentations: Sequence[Segmentation], *, kind: str = EXPLICIT,
             support: int | None = None,
             rng: np.random.Generator | None = None) -> TwoStateHSMM:
    """Fit emissions and duration laws from pooled segmentations."""
    mean = np.zeros((N_STATES, 2))
    variance = np.ones((N_STATES, 2))
    pooled: list[np.ndarray] = []
    for j in range(N_STATES):
        rows = [s.samples[j] for s in segmentations if s.samples[j].size]
        if rows:
            stacked = np.concatenate(rows, axis=0)
            mean[j] = stacked.mean(axis=0)
            variance[j] = np.maximum(stacked.var(axis=0), _MIN_VARIANCE)
        pooled.append(np.asarray(
            [d for s in segmentations for d in s.durations[j]], dtype=float))

    if support is None:
        observed = np.concatenate([p for p in pooled if p.size] or
                                  [np.asarray([1.0])])
        # A dwell cannot outlast the record, so the observed maximum is the
        # physical cap; without it the tail of the law is unreachable mass.
        support = int(max(4, min(np.percentile(observed, 99.5) * 3.0,
                                 observed.max())))

    laws = []
    for j in range(N_STATES):
        durations = pooled[j] if pooled[j].size else np.asarray([1.0])
        laws.append(duration_law(durations, support, kind, rng))

    total = sum(p.size for p in pooled) or 1
    initial = np.asarray([max(pooled[j].size, 0.5) for j in range(N_STATES)])
    return TwoStateHSMM(
        emissions=Emissions(mean=mean, variance=variance),
        durations=(laws[0], laws[1]),
        log_initial=np.log(initial / initial.sum()) if total else
        np.log(np.full(N_STATES, 0.5)),
    )


@dataclass(frozen=True)
class HSMMComposite:
    """``S(x) = log p(x | M_train) - max_j log p(x | M_null,j)``."""

    training: TwoStateHSMM
    nulls: tuple[TwoStateHSMM, ...]
    params: EventParams = EventParams()

    def score(self, t: np.ndarray, p_obs: np.ndarray) -> float:
        """Deployable score: consumes ``(t, P_obs)`` only."""
        observations = observation_matrix(t, p_obs, params=self.params)
        train = self.training.loglike(observations)
        null = max(model.loglike(observations) for model in self.nulls)
        return float(train - null)


def fit_composite(training: Sequence[Segmentation],
                  nulls: Sequence[Sequence[Segmentation]], *,
                  kind: str = EXPLICIT,
                  support: int | None = None,
                  seed: int = 0,
                  params: EventParams = EventParams()) -> HSMMComposite:
    """Fit training and null models with one estimator and one duration kind."""
    if not nulls:
        raise ValueError("at least one null group is required")
    # A shared support keeps the three duration kinds strictly comparable.
    if support is None:
        pooled = np.asarray(
            [d for group in [training, *nulls] for s in group
             for j in range(N_STATES) for d in s.durations[j]], dtype=float)
        support = int(max(4, min(np.percentile(pooled, 99.5) * 3.0,
                                 pooled.max()))) if pooled.size else 8
    return HSMMComposite(
        training=fit_hsmm(training, kind=kind, support=support,
                          rng=np.random.default_rng(seed)),
        nulls=tuple(fit_hsmm(group, kind=kind, support=support,
                             rng=np.random.default_rng(seed + 1 + j))
                    for j, group in enumerate(nulls)),
        params=params,
    )
