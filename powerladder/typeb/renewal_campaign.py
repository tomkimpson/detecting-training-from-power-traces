"""Shared harness for the duration-aware event-detection campaigns.

Population construction, feature extraction, feature-set ablations, and the
composite-null threshold machinery used by every campaign in the ``spec.md``
duration-aware extension:

* ``scripts/smoke_hsmm.py``          -- the E0/E1 viability smoke;
* ``scripts/hsmm_generalization.py`` -- stage 1, generalization falsification;
* ``scripts/hsmm_meter.py``          -- stage 2, the observation-channel sweep.

Two invariants hold throughout and are covered by ``tests/test_renewal_campaign.py``:

1. **Latent metadata never reaches a deployable score.**  :class:`AnnotatedTrace`
   carries generator phase boundaries, but only :func:`oracle_matrix` and
   :func:`timing_metrics` read them; :func:`blind_matrix` consumes ``(t, P_obs)``.
2. **Channel symmetry.**  A campaign that degrades the observation channel passes
   every positive *and* every null through the identical ``MeterParams``, so no
   population enjoys a bandwidth or noise advantage over another.

Trace features are model-independent, so a campaign extracts features once and
every leave-one-out or ablation configuration is a cheap re-fit of Gaussian
profiles over the cached matrices.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import numpy as np

from powerladder.config import DEFAULT, MeterParams
from powerladder.forward import TraceSpec, make_time_grid
from powerladder.ko_workload import (
    TrainingPhaseMetadata,
    aggregate_F_phase_meta,
    training_F_phase_meta,
)
from powerladder.observation import apply_meter
from powerladder.st1.nulls import ar1, ar2_resonant, controller, controller_ar1
from powerladder.st1.surrogates import fourier_phase_surrogate
from powerladder.typeb.ko_synth import _observe
from powerladder.typeb.renewal import (
    FEATURE_NAMES,
    blind_trace_features,
    event_timing_diagnostics,
    extract_events,
    oracle_trace_features,
)
from powerladder.typeb.roc import auc
from powerladder.typeb.rung2_scenarios import make_control_population
from powerladder.typeb.st2_attacks import _ratio_for_share, make_negative_population


#: Positive conditions: (name, family, level).  ``honest`` is the zero-level
#: member of the drift family, so it is the no-regression anchor for both.
CONDITIONS: tuple[tuple[str, str, float], ...] = (
    ("honest", "drift", 0.0),
    ("work_0.5", "work", 0.5),
    ("work_0.7", "work", 0.7),
    ("drift_0.8", "drift", 0.8),
    ("drift_1.5", "drift", 1.5),
)

ST1_NULLS = {
    "ar1": ar1,
    "ar2_resonant": ar2_resonant,
    "controller": controller,
    "controller_ar1": controller_ar1,
}

#: Every structural null family, in the order the campaigns report them.
NULL_FAMILIES: tuple[str, ...] = (
    "inference",
    *ST1_NULLS,
    "periodic_inference",
)

_CADENCE_FEATURES = tuple(
    FEATURE_NAMES.index(name) for name in ("event_rate_hz", "interval_cv")
)
_AMPLITUDE_FEATURES = tuple(
    FEATURE_NAMES.index(name)
    for name in ("amplitude_median_z", "amplitude_cv")
)


@dataclass(frozen=True)
class AnnotatedTrace:
    """Observed positive trace plus evaluation-only latent phase metadata."""

    t: np.ndarray
    P_obs: np.ndarray
    meta: TrainingPhaseMetadata


def _zero_columns(features: np.ndarray, columns: Sequence[int]) -> np.ndarray:
    out = np.asarray(features, dtype=float).copy()
    out[..., list(columns)] = 0.0
    return out


def _identity(features: np.ndarray) -> np.ndarray:
    return np.asarray(features, dtype=float)


def _local_only(features: np.ndarray) -> np.ndarray:
    """Ablation: remove event rate and interval variability from the score."""
    return _zero_columns(features, _CADENCE_FEATURES)


def _shape_duration(features: np.ndarray) -> np.ndarray:
    """Stricter ablation: remove cadence and absolute-amplitude information."""
    return _zero_columns(features, _CADENCE_FEATURES + _AMPLITUDE_FEATURES)


#: Feature sets each configuration is evaluated under.  ``shape_duration`` is the
#: most diagnostic: it is the only one with measured headroom in the E0/E1 smoke.
FEATURE_SETS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "full": _identity,
    "local_only": _local_only,
    "shape_duration": _shape_duration,
}

FEATURE_SET_REMOVED: dict[str, tuple[str, ...]] = {
    "full": (),
    "local_only": tuple(FEATURE_NAMES[i] for i in _CADENCE_FEATURES),
    "shape_duration": tuple(
        FEATURE_NAMES[i] for i in _CADENCE_FEATURES + _AMPLITUDE_FEATURES
    ),
}


def training_kwargs(family: str, level: float) -> dict[str, float]:
    if family == "work":
        return {"work_sigma": level}
    if family == "drift":
        return {"f0_drift_hz": level}
    raise ValueError(f"unsupported positive family {family!r}")


def make_annotated_population(
    family: str,
    level: float,
    n: int,
    glue,
    seed: int,
    *,
    meter: MeterParams | None = None,
) -> list[AnnotatedTrace]:
    """Draw ``n`` positive traces with their latent compute/communication phases.

    ``meter`` is the observation channel; ``None`` keeps the pre-meter path used
    by the E0/E1 smoke.  Callers degrading the channel must pass the *same*
    ``meter`` to :func:`make_null_populations` (see the module docstring).
    """
    rng = np.random.default_rng(seed)
    t = make_time_grid(glue.duration_s, 1.0 / glue.fs)
    f_peak = glue.f_peak_frac * DEFAULT.floor.F_max
    traces: list[AnnotatedTrace] = []
    for _ in range(n):
        f0 = float(rng.uniform(DEFAULT.ko.f0_lo, DEFAULT.ko.f0_hi))
        F, meta = training_F_phase_meta(
            t, DEFAULT.ko, rng, f_peak=f_peak, f0=f0,
            eta_scale=glue.eta_scale, **training_kwargs(family, level),
        )
        spec = TraceSpec(t=t, F=F, r=np.full_like(t, glue.r),
                         P0=np.full_like(t, glue.P0))
        t_obs, p_obs = _observe(spec, glue, meter, rng)
        traces.append(AnnotatedTrace(t_obs, p_obs, meta))
    return traces


@dataclass(frozen=True)
class _NullTrace:
    """A structural-null realisation in the shape the feature code expects."""

    t: np.ndarray
    P_obs: np.ndarray


def _st1_null_population(
    name: str,
    n: int,
    glue,
    seed: int,
    meter: MeterParams | None,
) -> list:
    """Realise a coloured/controller null, then apply the shared channel.

    The ST1 generators emit an already-observed process on the device grid, so
    channel symmetry is enforced by pushing them through the same
    :func:`apply_meter` map as the positives.  Meter noise is added to both, and
    the generator's own process amplitude plays the role the positive's workload
    modulation plays -- neither population is filtered while the other is not.
    """
    generator = ST1_NULLS[name]
    n_samples = int(round(glue.duration_s * glue.fs)) + 1
    rng = np.random.default_rng(seed)
    traces = [generator(n_samples, glue.fs, DEFAULT.st1_null, rng)
              for _ in range(n)]
    if meter is None:
        return traces
    observed: list[_NullTrace] = []
    for trace in traces:
        mt = apply_meter(trace.t, trace.P_obs, meter, rng)
        observed.append(_NullTrace(mt.t, mt.P_meter))
    return observed


def make_null_populations(
    n: int,
    glue,
    seed: int,
    *,
    meter: MeterParams | None = None,
    families: Sequence[str] = NULL_FAMILIES,
) -> dict[str, list]:
    """Draw ``n`` traces for each requested structural-null family."""
    unknown = set(families) - set(NULL_FAMILIES)
    if unknown:
        raise ValueError(f"unknown null families: {sorted(unknown)}")
    groups: dict[str, list] = {}
    for name in families:
        if name == "inference":
            groups[name] = make_negative_population(
                "work", n, DEFAULT.ko, glue, np.random.default_rng([seed, 0]),
                meter=meter,
            )
        elif name == "periodic_inference":
            groups[name] = make_control_population(
                "periodic_inference", n, DEFAULT.ko, glue,
                np.random.default_rng([seed, 10]), DEFAULT.rung2, meter=meter,
            )
        else:
            # Index within ST1_NULLS fixes the stream, so dropping a family from
            # `families` never shifts another family's realisations.
            j = list(ST1_NULLS).index(name) + 1
            groups[name] = _st1_null_population(name, n, glue, [seed, j], meter)
    return groups


def blind_matrix(traces) -> np.ndarray:
    """E1 features from ``(t, P_obs)`` only, one row per trace."""
    return np.stack([
        blind_trace_features(trace.t, trace.P_obs) for trace in traces
    ])


def oracle_matrix(traces: Sequence[AnnotatedTrace]) -> np.ndarray:
    """E0 information-ceiling features; never a deployable score."""
    return np.stack([
        oracle_trace_features(
            trace.t, trace.P_obs, trace.meta.communication_starts,
            trace.meta.communication_durations,
        )
        for trace in traces
    ])


def make_diluted_population(
    share: float,
    n: int,
    glue,
    seed: int,
    *,
    family: str = "drift",
    level: float = 0.0,
    meter: MeterParams | None = None,
) -> list[AnnotatedTrace]:
    """Positives whose training workload is a ``share`` fraction of an aggregate.

    The dominant training line is buried under Ko's eq-11 background of small
    trainings and fine-tunings.  ``share`` is the dominant slot's fraction of the
    total; ``share = 1`` is unreachable by construction, so the single-workload
    case stays with :func:`make_annotated_population`.
    """
    if not 0.0 < share < 1.0:
        raise ValueError("share must lie strictly between 0 and 1")
    rng = np.random.default_rng(seed)
    t = make_time_grid(glue.duration_s, 1.0 / glue.fs)
    f_peak = glue.f_peak_frac * DEFAULT.floor.F_max
    ko = dataclasses.replace(DEFAULT.ko, aggregate_ratio=_ratio_for_share(share))
    traces: list[AnnotatedTrace] = []
    for _ in range(n):
        f0 = float(rng.uniform(ko.f0_lo, ko.f0_hi))
        F, meta = aggregate_F_phase_meta(
            t, ko, rng, f_peak=f_peak, f0=f0, eta_scale=glue.eta_scale,
            **training_kwargs(family, level),
        )
        spec = TraceSpec(t=t, F=F, r=np.full_like(t, glue.r),
                         P0=np.full_like(t, glue.P0))
        t_obs, p_obs = _observe(spec, glue, meter, rng)
        traces.append(AnnotatedTrace(t_obs, p_obs, meta))
    return traces


def _iaaft(x: np.ndarray, rng: np.random.Generator, n_iter: int = 200,
           tol: float = 1e-10) -> np.ndarray:
    """Iterative amplitude-adjusted FT surrogate (Schreiber & Schmitz).

    Alternates between imposing the original amplitude spectrum and imposing the
    original marginal, converging to a series that matches *both* to within a
    small residual -- unlike one-pass AAFT, whose final rank remap perturbs the
    spectrum it just imposed.
    """
    x = np.asarray(x, dtype=float)
    target_mag = np.abs(np.fft.rfft(x))
    sorted_x = np.sort(x)
    y = rng.permutation(x)
    previous = None
    for _ in range(n_iter):
        spectrum = np.fft.rfft(y)
        phases = np.angle(spectrum)
        y = np.fft.irfft(target_mag * np.exp(1j * phases), n=x.size)
        y = sorted_x[np.argsort(np.argsort(y))]
        if previous is not None and np.max(np.abs(y - previous)) < tol:
            break
        previous = y.copy()
    return y


def surrogate_spectral_error(original: np.ndarray, surrogate: np.ndarray) -> float:
    """Relative L2 error between two amplitude spectra (0 = exactly matched)."""
    a = np.abs(np.fft.rfft(np.asarray(original, dtype=float)))
    b = np.abs(np.fft.rfft(np.asarray(surrogate, dtype=float)))
    denominator = float(np.linalg.norm(a))
    return float(np.linalg.norm(a - b) / denominator) if denominator > 0 else 0.0


def matched_surrogate_population(traces, seed: int, *, kind: str = "ft") -> list:
    """Nulls matched to the positives on mean, variance and power spectrum.

    ``fourier_phase_surrogate`` preserves every ``|rfft|`` magnitude exactly --
    hence the mean (DC), the variance (Parseval) and the whole autocovariance --
    while replacing each interior phase with an independent draw.  That destroys
    all phase coupling, which is where localised compute-to-communication events
    live.

    This is the hardest available structural null for the locality question: a
    detector that cannot separate training from its own phase-randomised
    surrogate is using the spectrum and the marginal, not local events.

Three kinds, differing in what they match:

    * ``'ft'`` -- spectrum matched exactly; marginal **not** matched.
    * ``'aaft'`` -- marginal matched exactly, spectrum only *approximately*: the
      final rank remap perturbs the amplitude spectrum it has just imposed.  Use
      :func:`surrogate_spectral_error` to quantify the residual before describing
      such a null as spectrum-matched.
    * ``'iaaft'`` -- iterates the two constraints to convergence, matching the
      marginal exactly and the spectrum to a small measured residual.  This is
      the one to use when a claim depends on *both* being matched.
    """
    if kind not in {"ft", "aaft", "iaaft"}:
        raise ValueError("kind must be 'ft', 'aaft' or 'iaaft'")
    rng = np.random.default_rng(seed)
    out: list[_NullTrace] = []
    for trace in traces:
        x = np.asarray(trace.P_obs, dtype=float)
        if kind == "iaaft":
            y = _iaaft(x, rng)
        else:
            y = fourier_phase_surrogate(x, rng)
            if kind == "aaft":
                y = np.sort(x)[np.argsort(np.argsort(y))]
        out.append(_NullTrace(np.asarray(trace.t, dtype=float), y))
    return out


def alignment_pool(traces: Sequence[AnnotatedTrace]) -> tuple[np.ndarray, np.ndarray]:
    """Event counts and communication durations pooled over a positive population."""
    counts = np.asarray([trace.meta.communication_starts.size for trace in traces],
                        dtype=int)
    durations = np.concatenate(
        [trace.meta.communication_durations for trace in traces]
    )
    return counts, durations


def oracle_null_matrix(traces, counts: np.ndarray, durations: np.ndarray,
                       seed: int) -> np.ndarray:
    """E0's null side: the *same* windowing procedure at random locations.

    ``oracle_trace_features`` windows a positive trace at its true communication
    boundaries.  Scoring nulls with :func:`blind_matrix` instead would compare two
    different measurement procedures, and several oracle features are read
    straight off the latent schedule -- ``paired_fraction`` is the constant 1.0,
    and the duration features are the generator's own durations -- so such a
    comparison separates the populations even on a pure-noise trace.  It measures
    the metadata, not the channel.

    ``spec.md`` specifies the control this implements: *"compare with randomly
    aligned or best-matched windows from each null"*.  Each null trace is
    windowed at uniformly random start times, with the event count and durations
    resampled from the positive population, so every metadata-derived feature is
    matched by construction and only genuine local waveform structure can
    separate the two.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for trace in traces:
        n = int(rng.choice(counts))
        drawn = rng.choice(durations, size=max(n, 1), replace=True)
        latest = float(trace.t[-1] - drawn.max())
        if latest <= float(trace.t[0]) or n == 0:
            rows.append(np.zeros(len(FEATURE_NAMES), dtype=float))
            continue
        starts = np.sort(rng.uniform(float(trace.t[0]), latest, n))
        rows.append(oracle_trace_features(trace.t, trace.P_obs, starts, drawn))
    return np.stack(rows)


def profile_scores(model, features: np.ndarray) -> np.ndarray:
    return np.asarray([model.score_features(row) for row in features])


def threshold_at_far(scores: np.ndarray, far: float) -> float:
    """Most permissive observed-score threshold whose empirical FAR is valid."""
    x = np.asarray(scores, dtype=float)
    unique = np.unique(x)
    # With the >= decision rule, a point mass at one score may be larger than
    # the FAR budget.  Thresholds immediately above each observed value are
    # therefore genuine operating points (excluding the whole tied mass), not
    # merely a numerical trick.  Omitting them can incorrectly leave +inf as
    # the only valid threshold for a perfectly separable discrete score.
    candidates = np.unique(np.concatenate([
        unique,
        np.nextafter(unique, np.inf),
        [np.inf],
    ]))
    valid = candidates[np.asarray([(x >= value).mean() <= far
                                   for value in candidates])]
    return float(np.min(valid))


def composite_threshold(groups: Mapping[str, np.ndarray], far: float) -> float:
    """Most conservative per-family threshold over a finite union of nulls."""
    return max(threshold_at_far(scores, far) for scores in groups.values())


def null_rates(groups: Mapping[str, np.ndarray],
               threshold: float) -> dict[str, float]:
    return {name: float((scores >= threshold).mean())
            for name, scores in groups.items()}


def metrics(pos: np.ndarray, neg_groups: Mapping[str, np.ndarray],
            threshold: float) -> dict:
    union = np.concatenate(list(neg_groups.values()))
    aucs = {name: auc(pos, scores) for name, scores in neg_groups.items()}
    return {
        "auc_union": auc(pos, union),
        "auc_by_null": aucs,
        "min_auc": min(aucs.values()),
        "tpr_at_composite_far": float((pos >= threshold).mean()),
        "score_median": float(np.median(pos)),
    }


def timing_metrics(traces: Sequence[AnnotatedTrace],
                   *, tolerance_s: float = 0.1) -> dict:
    """Blind-vs-truth event recall, precision and timing error (diagnostic only).

    ``tolerance_s`` defaults to the 100 ms figure ``spec.md`` states for the
    nominal 20 Hz channel.  A campaign that degrades the sample rate should also
    report a channel-adaptive tolerance: 100 ms is unreachable once the sample
    period exceeds it, so a recall of zero there measures the tolerance, not the
    detector.
    """
    rows = []
    for trace in traces:
        events = extract_events(trace.t, trace.P_obs)
        rows.append(event_timing_diagnostics(
            events.down_times, trace.meta.communication_starts,
            tolerance_s=tolerance_s,
        ))
    values = np.asarray(rows)
    finite_error = values[np.isfinite(values[:, 2]), 2]
    return {
        "recall_mean": float(values[:, 0].mean()),
        "recall_median": float(np.median(values[:, 0])),
        "precision_mean": float(values[:, 1].mean()),
        "precision_median": float(np.median(values[:, 1])),
        "timing_error_median_s": (float(np.median(finite_error))
                                  if finite_error.size else None),
    }


def glue_for(duration_s: float, *, fs: float | None = None,
             eta_scale: float | None = None,
             f_peak_frac: float | None = None):
    """Type-B glue at a given record length, device fs and event contrast.

    ``f_peak_frac`` is the workload's peak share of ``F_max`` and is the honest
    **event-contrast / SNR** knob: lowering it shrinks the compute-to-
    communication power excursion against a fixed meter noise ``sigma_eta``.

    ``eta_scale`` is *not* a contrast knob -- it rescales each component's
    sub-step noise, which the meter averages away, so lowering it makes traces
    cleaner rather than harder.  It is exposed only for completeness.
    """
    glue = dataclasses.replace(DEFAULT.ko_typeb, duration_s=duration_s)
    if fs is not None:
        glue = dataclasses.replace(glue, fs=fs)
    if eta_scale is not None:
        glue = dataclasses.replace(glue, eta_scale=eta_scale)
    if f_peak_frac is not None:
        glue = dataclasses.replace(glue, f_peak_frac=f_peak_frac)
    return glue
