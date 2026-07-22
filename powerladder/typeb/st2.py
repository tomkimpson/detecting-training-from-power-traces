"""ST2 sweep harness: one attack family -> detector power + physical measures.

Task 20.6. Generalises :func:`code.typeb.gate.evaluate` for the frontier
(plan-for-paper-2 §5): injected detector dict (the ST1 plug-in point, task
20.9), multiple FARs, negatives generated ONCE per family and reused across
levels, and per-level PHYSICAL de-periodicisation measures (cadence CV, phase
diffusion D) from generator ground truth (:mod:`code.typeb.deperiod`) so the
frontier's x-axis is shared across families rather than knob-specific.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np

from ..config import KoTypeBParams, KoWorkloadParams, MeterParams, St2Params
from ..forward import make_time_grid
from ..ko_workload import training_F_meta
from .deperiod import cadence_cv, phase_diffusion_coeff
from .gate import DETECTORS, score_population
from .roc import auc, tpr_at_far
from .st2_attacks import (
    FAMILY_ORDER,
    attack_families,
    make_negative_population,
    make_positive_population,
)

# Seed-stream tags (third entry of the default_rng entropy triple).
_NEG_STREAM = 0        # the once-per-family negative population
_POS_STREAM = 1        # + level index
_META_STREAM = 1000    # + level index (physical-measure traces)

# Meta traces per level for the CV/D measures (cheap: boundaries only).
_N_META = 8

# Minimum meta-trace window [s]. CV and D are stationary properties of the
# boundary process (duration-independent), so a longer window only tightens
# the estimate — and guarantees the >= 20 boundaries the diffusion fit needs
# even at the slowest relocated cadence under heavy jitter.
_META_MIN_T = 240.0

# family -> the training_F_meta kwarg its level sets for the CV/D measures.
# jitter is handled via dataclasses.replace (sigma_jitter is a KoWorkloadParams
# field, not a kwarg); relocate via f0; families absent here (shape, harmonic,
# dilute, meter) do not touch timing and run the measure at the honest baseline.
_META_KNOB = {
    "work": "work_sigma",
    "drift": "f0_drift_hz",
    "phase": "phase_slip_sigma",
}


@dataclass(frozen=True)
class St2Point:
    """One (family, level) cell of the frontier.

    ``tpr_at_far`` is keyed detector -> {"%g" % far -> TPR}; ``auc`` is keyed
    detector -> AUC. ``level`` is a float for every family except meter, where
    it is the hostile-variant name.
    """

    family: str
    level: float | str
    cadence_cv: float
    phase_diffusion_D: float
    auc: dict[str, float]
    tpr_at_far: dict[str, dict[str, float]]


def _effective_meter(p: St2Params) -> MeterParams | None:
    """The builders' meter argument for non-meter families.

    The exact-no-op default ``MeterParams()`` maps to ``meter=None`` (the
    pre-ST2 glue-noise path) so default sweeps stay byte-comparable with the
    b0/b1 gates; any non-default meter is passed through as the channel.
    """
    return None if p.meter == MeterParams() else p.meter


def _measures(
    family: str,
    level,
    ko_params: KoWorkloadParams,
    glue: KoTypeBParams,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Mean (cadence CV, phase-diffusion D) over ``_N_META`` fresh meta traces.

    The measures come from generator GROUND-TRUTH boundaries
    (:func:`training_F_meta`), so they need not — and do not — reuse the
    scored population's traces: a handful of independent draws per level
    estimates the family-level physics without re-running the observation
    channel. Families that do not touch timing (shape, harmonic, dilute,
    meter) still get CV/D computed; they sit at the honest Ko baseline by
    construction, which is the point — their attack axis is orthogonal to
    de-periodicisation.
    """
    t = make_time_grid(max(glue.duration_s, _META_MIN_T), 1.0 / glue.fs)
    kp = ko_params
    kwargs: dict = {}
    f0 = None
    if family == "jitter":
        kp = dataclasses.replace(ko_params, sigma_jitter=float(level))
    elif family in _META_KNOB:
        kwargs[_META_KNOB[family]] = float(level)
    elif family == "relocate":
        f0 = float(level)

    cvs, ds = [], []
    for _ in range(_N_META):
        _, iter_starts = training_F_meta(t, kp, rng, f0=f0, eta_scale=0.0,
                                         **kwargs)
        cvs.append(cadence_cv(iter_starts))
        # extreme jitter draws can (rarely) leave too few boundaries for the
        # diffusion fit even on the long meta window; skip those traces.
        if iter_starts.size >= 20:
            ds.append(phase_diffusion_coeff(iter_starts))
    return float(np.mean(cvs)), float(np.mean(ds)) if ds else float("nan")


def run_family(
    family: str,
    p: St2Params,
    ko_params: KoWorkloadParams,
    glue: KoTypeBParams,
    *,
    detectors: dict | None = None,
    seed: int | None = None,
) -> list[St2Point]:
    """Sweep one attack family over its levels; one :class:`St2Point` each.

    ``detectors`` maps name -> statistic fn ``(t, P_obs, band_lo, band_hi) ->
    float`` (default: a copy of :data:`code.typeb.gate.DETECTORS`; ST1
    detectors register here, task 20.9).

    Seeding (all reproducible off ``seed`` or ``p.seed``): negatives draw from
    the fixed spawned stream ``(base, family_index, _NEG_STREAM)`` and are
    generated ONCE per family, scored once per detector, and reused across
    levels — so level-to-level TPR differences come from the positives alone.
    EXCEPTION: the meter family's channel differs per level (each level IS a
    different hostile channel, and both classes must share it), so its
    negatives are regenerated per level from the SAME fixed stream — identical
    device-side draws, only the observation channel varies. Positives use
    stream ``(base, family_index, _POS_STREAM + level_index)``.

    Physical measures (CV, D) come from ``_N_META`` independent meta traces
    per level (see :func:`_measures`) — deliberately NOT the scored traces.
    """
    detectors = dict(DETECTORS) if detectors is None else detectors
    fam = attack_families(p)[family]
    base = p.seed if seed is None else seed
    fidx = FAMILY_ORDER.index(family)
    meter = _effective_meter(p)
    variants = dict(p.meter_variants)

    def neg_scores_for(m: MeterParams | None) -> dict[str, np.ndarray]:
        rng = np.random.default_rng([base, fidx, _NEG_STREAM])
        negs = make_negative_population(family, p.n_each, ko_params, glue,
                                        rng, meter=m)
        return {name: score_population(negs, fn, glue)
                for name, fn in detectors.items()}

    if family != "meter":
        shared_neg = neg_scores_for(meter)

    points: list[St2Point] = []
    for i, level in enumerate(fam.levels):
        m = variants[level] if family == "meter" else meter
        neg = neg_scores_for(m) if family == "meter" else shared_neg

        rng_pos = np.random.default_rng([base, fidx, _POS_STREAM + i])
        pos_traces = make_positive_population(family, level, p.n_each,
                                              ko_params, glue, rng_pos,
                                              meter=m)
        rng_meta = np.random.default_rng([base, fidx, _META_STREAM + i])
        cv, d = _measures(family, level, ko_params, glue, rng_meta)

        auc_d: dict[str, float] = {}
        tpr_d: dict[str, dict[str, float]] = {}
        for name, fn in detectors.items():
            pos = score_population(pos_traces, fn, glue)
            auc_d[name] = auc(pos, neg[name])
            tpr_d[name] = {f"{far:g}": tpr_at_far(pos, neg[name], far)
                           for far in p.target_fars}
        points.append(St2Point(family=family, level=level, cadence_cv=cv,
                               phase_diffusion_D=d, auc=auc_d,
                               tpr_at_far=tpr_d))
    return points
