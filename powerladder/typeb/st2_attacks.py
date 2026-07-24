"""ST2 attack registry: named de-periodicisation families -> trace populations.

Task 20.6 (plan-for-paper-2 §4: every attack is an OPERATOR with a REPORTED
BUDGET). Each :class:`AttackFamily` names one family, its budget units, the
swept levels (from :class:`code.config.St2Params`) and — where the b2 hardware
campaign measured the same operator — the results file its systems-cost anchor
is read from (task 20.8). Families with ``cost_anchor=None`` are
analytic/qualitative only.

The two population builders map a (family, level) onto the right generator
knobs, reusing the Ko generator + forward-model glue via
:func:`code.typeb.ko_synth._observe` (single owner of the noise-ownership
rule):

    jitter    -> KoWorkloadParams.sigma_jitter        (i.i.d. period jitter)
    work      -> training_F(work_sigma=...)           (real-work variation,
                                                       the measured ~zero-cost
                                                       attack, task 20.3)
    drift     -> training_F(f0_drift_hz=...)          (OU centre-freq drift)
    phase     -> training_F(phase_slip_sigma=...)     (Brownian boundary slip)
    relocate  -> training_F(f0=level)                 (cadence moved toward the
                                                       search-band edges)
    harmonic  -> training_F(harmonic_smooth_s=...)    (comb suppression)
    shape     -> training_F(shape_fill_frac=...)      (amplitude shaping, b2 phi)
    dilute    -> ko_make_aggregate_trace at dominant  (b1agg superposition
                 share = level                         pattern)
    meter     -> nominal training under a NAMED hostile MeterParams variant
                 (level is the variant name; the caller resolves it to a
                 MeterParams and passes it as ``meter``)

Class channel symmetry: :func:`make_negative_population` builds the inference
null through the SAME observation channel as the positives (``meter`` kwarg) —
for the meter family in particular, positives and negatives must share the
hostile channel or the comparison measures the channel, not the workload.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np

from ..config import DEFAULT, KoTypeBParams, KoWorkloadParams, MeterParams, St2Params
from ..forward import TraceSpec, make_time_grid
from ..ko_workload import training_F
from .ko_synth import _observe, ko_make_aggregate_trace, ko_make_trace
from .synth import TypeTrace

# Canonical family order (stable indices for seed derivation in code.typeb.st2).
FAMILY_ORDER = (
    "jitter", "work", "drift", "phase", "relocate",
    "harmonic", "shape", "dilute", "meter",
)

# Background weights held at Ko's nominal 0.5:0.5 while the dominant share is
# swept (== scripts/plot_b1agg_roc.py's _BG_RATIO; d = 0.9 reproduces Ko's
# nominal 9:0.5:0.5 exactly).
_BG_RATIO = (0.5, 0.5)


def _ratio_for_share(d: float) -> tuple[float, float, float]:
    """aggregate_ratio whose dominant share of the total is ``d`` (b1agg pattern)."""
    w_bg = sum(_BG_RATIO)
    return (d / (1.0 - d) * w_bg, *_BG_RATIO)


@dataclass(frozen=True)
class AttackFamily:
    """One de-periodicisation attack family (plan §4: operator + budget).

    ``cost_anchor`` names the b2 measured-summary file the family's systems-
    cost (throughput-overhead) anchor is read from, or None when no hardware
    measurement of this operator exists (labelled "analytic/qualitative only"
    on the frontier).
    """

    name: str
    budget_units: str
    levels: tuple
    cost_anchor: str | None


def attack_families(p: St2Params) -> dict[str, AttackFamily]:
    """The registry, with level grids read from ``p`` (single source of truth)."""
    fams = {
        "jitter": AttackFamily(
            "jitter", "sigma_jitter (fractional i.i.d. period jitter)",
            p.jitter_levels, "data/measured_cost_anchors/spoof_summary.json"),
        "work": AttackFamily(
            "work", "work_sigma (fractional micro-step-count jitter)",
            p.work_levels, "data/measured_cost_anchors/workjitter_summary.json"),
        "drift": AttackFamily(
            "drift", "f0_drift_hz (OU centre-frequency excursion, Hz)",
            p.drift_levels, "data/measured_cost_anchors/spoof_summary.json"),
        "phase": AttackFamily(
            "phase", "phase_slip_sigma (boundary slip std, fraction of period)",
            p.phase_levels, None),
        "relocate": AttackFamily(
            "relocate", "f0 (relocated cadence, Hz; search band 0.3-1.7)",
            p.relocate_f0s, None),
        "harmonic": AttackFamily(
            "harmonic", "harmonic_smooth_s (transition ramp width, s)",
            p.harmonic_levels, None),
        "shape": AttackFamily(
            "shape", "shape_fill_frac (phi, down-level fill fraction)",
            p.shape_levels, "data/measured_cost_anchors/shaped_summary.json"),
        "dilute": AttackFamily(
            "dilute", "dominant training share of the aggregate",
            p.dilute_shares, None),
        "meter": AttackFamily(
            "meter", "named hostile MeterParams variant",
            tuple(name for name, _ in p.meter_variants), None),
    }
    assert tuple(fams) == FAMILY_ORDER
    return fams


# The registry at the default level grids (scripts import this; the harness
# rebuilds from its own St2Params so overridden grids stay authoritative).
ATTACKS = attack_families(DEFAULT.st2)


def _train_trace(
    ko_p: KoWorkloadParams,
    glue: KoTypeBParams,
    rng: np.random.Generator,
    *,
    meter: MeterParams | None,
    f0: float | None = None,
    **train_kwargs,
) -> TypeTrace:
    """One training trace with arbitrary ``training_F`` attack kwargs.

    Mirrors :func:`code.typeb.ko_synth.ko_make_trace` (same glue, same
    noise-ownership rule via ``_observe``) but forwards the ST2 attack knobs
    (work_sigma, phase_slip_sigma, harmonic_smooth_s, shape_fill_frac, ...)
    that the B1-era builder does not expose.
    """
    f_peak = glue.f_peak_frac * DEFAULT.floor.F_max
    t = make_time_grid(glue.duration_s, 1.0 / glue.fs)
    if f0 is None:
        f0 = float(rng.uniform(ko_p.f0_lo, ko_p.f0_hi))
    F = training_F(t, ko_p, rng, f_peak=f_peak, f0=f0,
                   eta_scale=glue.eta_scale, **train_kwargs)
    spec = TraceSpec(t=t, F=F,
                     r=np.full_like(t, glue.r), P0=np.full_like(t, glue.P0))
    t_obs, P_obs = _observe(spec, glue, meter, rng)
    return TypeTrace(t=t_obs, P_obs=P_obs, label="train", f0=f0)


# family -> the training_F kwarg its level sets (the plain single-knob families).
_TRAIN_KNOB = {
    "work": "work_sigma",
    "drift": "f0_drift_hz",
    "phase": "phase_slip_sigma",
    "harmonic": "harmonic_smooth_s",
    "shape": "shape_fill_frac",
}


def make_positive_population(
    family: str,
    level,
    n: int,
    ko_params: KoWorkloadParams,
    glue: KoTypeBParams,
    rng: np.random.Generator,
    *,
    meter: MeterParams | None,
) -> list[TypeTrace]:
    """``n`` attacked-training traces of ``family`` at ``level``.

    ``meter`` is the observation channel (None = the pre-ST2 glue-noise path).
    For the meter family the caller resolves the variant name (== ``level``)
    to its hostile MeterParams and passes it here; the workload itself is the
    NOMINAL honest training (the attack lives in the channel).
    """
    if family == "jitter":
        kp = dataclasses.replace(ko_params, sigma_jitter=float(level))
        return [_train_trace(kp, glue, rng, meter=meter) for _ in range(n)]
    if family in _TRAIN_KNOB:
        kw = {_TRAIN_KNOB[family]: float(level)}
        return [_train_trace(ko_params, glue, rng, meter=meter, **kw)
                for _ in range(n)]
    if family == "relocate":
        return [_train_trace(ko_params, glue, rng, meter=meter, f0=float(level))
                for _ in range(n)]
    if family == "dilute":
        kp = dataclasses.replace(
            ko_params, aggregate_ratio=_ratio_for_share(float(level)))
        return [ko_make_aggregate_trace("train", kp, glue, rng, meter=meter)
                for _ in range(n)]
    if family == "meter":
        return [_train_trace(ko_params, glue, rng, meter=meter)
                for _ in range(n)]
    raise ValueError(f"unknown attack family {family!r}")


def make_negative_population(
    family: str,
    n: int,
    ko_params: KoWorkloadParams,
    glue: KoTypeBParams,
    rng: np.random.Generator,
    *,
    meter: MeterParams | None,
) -> list[TypeTrace]:
    """``n`` inference-null traces through the SAME channel as the positives.

    Single-workload families score against the hard inference null
    (:func:`code.ko_workload.inference_F` via ``ko_make_trace``). The dilute
    family scores against the AGGREGATE null (inference-dominant eq-11 mix) at
    Ko's NOMINAL 9:0.5:0.5 ratio: the Verifier's null hypothesis is a nominal
    inference-dominant facility, held fixed while the positives sweep their
    internal mix (deliberate deviation from plot_b1agg_roc.py, where both
    classes moved together — here negatives are generated once per family and
    reused across levels). The meter family passes the hostile variant's
    MeterParams via ``meter`` so both classes share the observation channel —
    otherwise the gate measures the channel, not the workload.
    """
    if family == "dilute":
        return [ko_make_aggregate_trace("infer", ko_params, glue, rng,
                                        meter=meter) for _ in range(n)]
    return [ko_make_trace("infer", ko_params, glue, rng, meter=meter)
            for _ in range(n)]
