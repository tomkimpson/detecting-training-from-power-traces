"""Ko et al. F(t) -> labeled power ``TypeTrace`` populations for the B1 gate.

B1 (tasks.md Task 3) re-runs the B0 Type detectors on the FAITHFUL Ko generator
(:mod:`code.ko_workload`) instead of the B0-local placeholder
(:mod:`code.typeb.synth`), turning B0's *method decision* into real single-GPU
*performance* numbers. The detectors consume only ``(t, P_obs)``, so this module
just supplies the missing glue: it converts a Ko training ``F(t)`` (with optional
slow ``f0`` drift, task 3.2) or the hard inference null (:func:`inference_F`, task
3.3) into an observed power trace ``P = r*F + P0 + eta`` via the forward model
(:mod:`code.forward`), and packages it as the same :class:`TypeTrace` the gate
already understands. So :func:`ko_make_population` drops into the gate's population
slot in place of :func:`code.typeb.synth.make_population` with no detector change.

B1-agg (Task 4) reuses the same glue at PDU granularity:
:func:`ko_make_aggregate_population` swaps the single workload for the Ko eq-11
superposition (:func:`aggregate_F` vs the inference-dominant
:func:`aggregate_null_F`), asking whether the dominant training line survives
being buried under other workloads at a shared meter.

The glue constants live in :class:`code.config.KoTypeBParams` (== the values
validated in ``test_detectors_fire_on_ko_training`` and used in
:mod:`code.scenarios`); ``F_peak`` reads :data:`code.config.DEFAULT`'s
``floor.F_max`` at call time rather than copying it.

ST2 meter wiring (task 20.2): every builder takes ``meter: MeterParams | None``.
``meter=None`` (default) is the existing path, byte-identical. With a meter the
device trace is pushed through :func:`code.observation.apply_meter`, and NOISE
OWNERSHIP moves to the meter: ``simulate`` runs with ``WhiteNoise(0.0)`` so the
glue's white ``sigma_eta`` is NOT applied — the meter map is the single owner of
observation noise (``meter.sigma_eta``), never stacked on top of the glue's
(the double-noise guard in tests/test_ko_synth_meter.py). The returned
``TypeTrace`` carries the METER grid, which differs from the device grid when
``meter.sample_hz`` decimates.
"""

from __future__ import annotations

import numpy as np

from ..config import DEFAULT, KoTypeBParams, KoWorkloadParams, MeterParams
from ..forward import TraceSpec, make_time_grid, simulate
from ..ko_workload import aggregate_F, aggregate_null_F, inference_F, training_F
from ..noise import WhiteNoise
from ..observation import apply_meter
from .synth import TypeTrace


def _observe(
    spec: TraceSpec,
    glue: KoTypeBParams,
    meter: MeterParams | None,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Device spec -> (t, P_obs) under the noise-ownership rule.

    meter=None: the pre-ST2 path — glue white noise, device grid (byte-
    identical to before the meter existed). meter set: simulate noiselessly
    (``WhiteNoise(0.0)`` keeps the RNG stream aligned with the legacy path up
    to this point) and let the meter map own all observation noise.
    """
    if meter is None:
        trace = simulate(spec, WhiteNoise(glue.sigma_eta), rng)
        return trace.t, trace.P_obs
    trace = simulate(spec, WhiteNoise(0.0), rng)
    mt = apply_meter(trace.t, trace.P_obs, meter, rng)
    return mt.t, mt.P_meter


def ko_make_trace(
    label: str,
    ko_p: KoWorkloadParams,
    glue: KoTypeBParams,
    rng: np.random.Generator,
    *,
    f0_drift_hz: float = 0.0,
    f_max: float | None = None,
    meter: MeterParams | None = None,
) -> TypeTrace:
    """One labeled Ko power trace. ``label`` is "train" or "infer".

    Training draws its own ``f0`` from the Ko band (recorded on the trace) and may
    drift it (``f0_drift_hz`` > 0, the honest-smearing axis of task 3.2). Inference
    is the hard quasi-periodic null (no concentrated line; ``f0`` = NaN).
    """
    f_max = DEFAULT.floor.F_max if f_max is None else f_max
    f_peak = glue.f_peak_frac * f_max
    t = make_time_grid(glue.duration_s, 1.0 / glue.fs)

    if label == "train":
        f0 = float(rng.uniform(ko_p.f0_lo, ko_p.f0_hi))
        F = training_F(t, ko_p, rng, f_peak=f_peak, f0=f0,
                       eta_scale=glue.eta_scale, f0_drift_hz=f0_drift_hz)
    elif label == "infer":
        f0 = float("nan")
        F = inference_F(t, ko_p, rng, f_peak=f_peak, eta_scale=glue.eta_scale)
    else:
        raise ValueError(f"unknown label {label!r} (expected 'train' or 'infer')")

    spec = TraceSpec(
        t=t, F=F,
        r=np.full_like(t, glue.r),
        P0=np.full_like(t, glue.P0),
    )
    t_obs, P_obs = _observe(spec, glue, meter, rng)
    return TypeTrace(t=t_obs, P_obs=P_obs, label=label, f0=f0)


def ko_make_population(
    n_each: int,
    ko_p: KoWorkloadParams,
    glue: KoTypeBParams,
    rng: np.random.Generator,
    *,
    f0_drift_hz: float = 0.0,
    f_max: float | None = None,
    meter: MeterParams | None = None,
) -> tuple[list[TypeTrace], list[TypeTrace]]:
    """Return (training traces, inference traces), ``n_each`` of each regime."""
    train = [
        ko_make_trace("train", ko_p, glue, rng,
                      f0_drift_hz=f0_drift_hz, f_max=f_max, meter=meter)
        for _ in range(n_each)
    ]
    infer = [
        ko_make_trace("infer", ko_p, glue, rng, f_max=f_max, meter=meter)
        for _ in range(n_each)
    ]
    return train, infer


def ko_make_aggregate_trace(
    label: str,
    ko_p: KoWorkloadParams,
    glue: KoTypeBParams,
    rng: np.random.Generator,
    *,
    f0_drift_hz: float = 0.0,
    f_max: float | None = None,
    meter: MeterParams | None = None,
) -> TypeTrace:
    """One labeled PDU-level aggregate power trace (B1-agg, task 4).

    Same glue as :func:`ko_make_trace`, but the F(t) source is the Ko eq-11
    superposition: ``label`` "train" is the positive class (dominant training,
    with its ``f0`` drawn from the Ko band and recorded, plus the small-training/
    fine-tune background); "infer" is the null (dominant slot replaced by the hard
    inference null over the SAME background; ``f0`` = NaN). The dominance ratio is
    read from ``ko_p.aggregate_ratio`` -- sweep it via ``dataclasses.replace``.
    ``f_peak`` is the aggregate TOTAL, so the two classes are power-matched and
    the dominant line weakens as its share shrinks.
    """
    f_max = DEFAULT.floor.F_max if f_max is None else f_max
    f_peak = glue.f_peak_frac * f_max
    t = make_time_grid(glue.duration_s, 1.0 / glue.fs)

    if label == "train":
        f0 = float(rng.uniform(ko_p.f0_lo, ko_p.f0_hi))
        F = aggregate_F(t, ko_p, rng, f_peak=f_peak, f0=f0,
                        eta_scale=glue.eta_scale, f0_drift_hz=f0_drift_hz)
    elif label == "infer":
        f0 = float("nan")
        F = aggregate_null_F(t, ko_p, rng, f_peak=f_peak,
                             eta_scale=glue.eta_scale)
    else:
        raise ValueError(f"unknown label {label!r} (expected 'train' or 'infer')")

    spec = TraceSpec(
        t=t, F=F,
        r=np.full_like(t, glue.r),
        P0=np.full_like(t, glue.P0),
    )
    t_obs, P_obs = _observe(spec, glue, meter, rng)
    return TypeTrace(t=t_obs, P_obs=P_obs, label=label, f0=f0)


def ko_make_aggregate_population(
    n_each: int,
    ko_p: KoWorkloadParams,
    glue: KoTypeBParams,
    rng: np.random.Generator,
    *,
    f0_drift_hz: float = 0.0,
    f_max: float | None = None,
    meter: MeterParams | None = None,
) -> tuple[list[TypeTrace], list[TypeTrace]]:
    """Return (aggregate-with-training, aggregate-null) traces, ``n_each`` each.

    Drop-in for the gate's ``population_fn`` slot, as :func:`ko_make_population`.
    """
    train = [
        ko_make_aggregate_trace("train", ko_p, glue, rng,
                                f0_drift_hz=f0_drift_hz, f_max=f_max,
                                meter=meter)
        for _ in range(n_each)
    ]
    infer = [
        ko_make_aggregate_trace("infer", ko_p, glue, rng, f_max=f_max,
                                meter=meter)
        for _ in range(n_each)
    ]
    return train, infer
