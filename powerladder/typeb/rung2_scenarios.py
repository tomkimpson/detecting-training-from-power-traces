"""Rung-2 semantic falsification controls (plan §3.3): the paper's central question.

Rung 2 certifies *physical training-likeness* against a stated inference null. But
what does the meter actually pin down — the SEMANTICS of training, or only its
physical schedule? These seven controls answer that by construction. Each is a
labelled population the calibrated Rung-2 rules (:mod:`code.typeb.rung2`) score;
the reported quantity is the fraction scored as training at the stated operating
point, plus AUC against the inference null. Two axes are annotated per control:

- ``is_training`` — the SEMANTIC ground truth (is retained model training
  actually happening?);
- ``expect`` — the physical prediction (how the schedule-level detector *should*
  score it: ``"train"`` / ``"not_train"``).

The interesting rows are where the two disagree:

  discarded_update_decoy  is_training=False, expect="train"  — full training
      compute whose optimizer update is discarded: physically identical to
      training, so it SHOULD score as training. The meter certifies physics,
      not semantics.
  async_training          is_training=True,  expect="not_train" — genuine
      training strongly de-periodicised (OU drift + Brownian slip): the
      scoped-OUT efficient-family edge the schedule detector misses.

The other five:

  gradient_only        is_training=False, expect="train"     — forward/backward
      without the optimizer swing (attenuated level deltas), same cadence.
  nonml_kernel_loop    is_training=False, expect="train"     — a training-SHAPED
      non-ML kernel loop (matched cadence, comb-suppressed waveform).
  controller_cycle     is_training=False, expect="train"     — a bare periodic
      NON-compute load (power-controller limit cycle). Expected to FIRE the
      schedule detector (plan §3.3: "controllers triggering Rung 1 ⇒ the
      structural test works and causal attribution needs the stated nuisance
      model"): a periodic line is training-LIKE at the physical level.
  periodic_inference   is_training=False, expect="train"     — the inference
      null driven by a periodic request generator: a periodic line without the
      training iteration semantics, expected to fire (periodicity alone is not
      training — the meter certifies the schedule, not the workload).
  coresident_mixture   is_training=True,  expect="train"     — genuine training
      diluted among unrelated co-resident loads at the shared meter.

Builders reuse the Ko generator + forward-model glue via
:func:`code.typeb.ko_synth._observe` (the single owner of the noise-ownership
rule) exactly as :mod:`code.typeb.st2_attacks` does; the two non-Ko loads
(controller_cycle, periodic_inference) synthesise ``F(t)`` directly. Class
channel symmetry (the st2_attacks rule) is preserved by the harness, which pushes
the inference-null reference through the same ``meter`` as the controls.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np

from ..config import (DEFAULT, KoTypeBParams, KoWorkloadParams, MeterParams,
                      Rung2Params)
from ..forward import TraceSpec, make_time_grid
from ..ko_workload import inference_F, training_F
from .ko_synth import _observe, ko_make_aggregate_trace
from .st2_attacks import _ratio_for_share
from .synth import TypeTrace


@dataclass(frozen=True)
class Control:
    """One semantic falsification control (plan §3.3).

    ``is_training`` is the semantic ground truth; ``expect`` is the predicted
    physical (schedule-level) score, ``"train"`` or ``"not_train"``.
    """

    name: str
    is_training: bool
    expect: str
    note: str


CONTROL_ORDER: tuple[str, ...] = (
    "discarded_update_decoy",
    "gradient_only",
    "nonml_kernel_loop",
    "controller_cycle",
    "periodic_inference",
    "async_training",
    "coresident_mixture",
)

CONTROLS: dict[str, Control] = {
    "discarded_update_decoy": Control(
        "discarded_update_decoy", is_training=False, expect="train",
        note="full training compute, optimizer update discarded — "
             "physically identical to training"),
    "gradient_only": Control(
        "gradient_only", is_training=False, expect="train",
        note="forward/backward without the optimizer-step swing"),
    "nonml_kernel_loop": Control(
        "nonml_kernel_loop", is_training=False, expect="train",
        note="training-shaped non-ML kernel loop (matched cadence)"),
    "controller_cycle": Control(
        "controller_cycle", is_training=False, expect="train",
        note="bare periodic non-compute load (power-controller limit cycle) — "
             "fires the schedule detector; attribution needs the nuisance model"),
    "periodic_inference": Control(
        "periodic_inference", is_training=False, expect="train",
        note="inference null driven by a periodic request generator — "
             "periodicity alone is not training semantics"),
    "async_training": Control(
        "async_training", is_training=True, expect="not_train",
        note="genuine training strongly de-periodicised (scoped-out edge)"),
    "coresident_mixture": Control(
        "coresident_mixture", is_training=True, expect="train",
        note="genuine training diluted among co-resident loads"),
}

assert tuple(CONTROLS) == CONTROL_ORDER


def _f_peak(glue: KoTypeBParams) -> float:
    return glue.f_peak_frac * DEFAULT.floor.F_max


def _wrap(F: np.ndarray, t: np.ndarray, label: str, glue: KoTypeBParams,
          meter: MeterParams | None, rng: np.random.Generator,
          f0: float = float("nan")) -> TypeTrace:
    """Forward-model F(t) -> observed TypeTrace (shared noise-ownership rule)."""
    spec = TraceSpec(t=t, F=F,
                     r=np.full_like(t, glue.r), P0=np.full_like(t, glue.P0))
    t_obs, P_obs = _observe(spec, glue, meter, rng)
    return TypeTrace(t=t_obs, P_obs=P_obs, label=label, f0=f0)


def _training_control(
    ko_params: KoWorkloadParams,
    glue: KoTypeBParams,
    rng: np.random.Generator,
    label: str,
    *,
    meter: MeterParams | None,
    **train_kwargs,
) -> TypeTrace:
    """A training-schedule control: nominal cadence with the given training_F knobs."""
    t = make_time_grid(glue.duration_s, 1.0 / glue.fs)
    f0 = float(rng.uniform(ko_params.f0_lo, ko_params.f0_hi))
    F = training_F(t, ko_params, rng, f_peak=_f_peak(glue), f0=f0,
                   eta_scale=glue.eta_scale, **train_kwargs)
    return _wrap(F, t, label, glue, meter, rng, f0=f0)


def _controller_cycle(
    glue: KoTypeBParams, p: Rung2Params, rng: np.random.Generator,
    *, meter: MeterParams | None,
) -> TypeTrace:
    """Bare periodic non-compute load: base level + a duty-cycle square limit cycle."""
    t = make_time_grid(glue.duration_s, 1.0 / glue.fs)
    fp = _f_peak(glue)
    ph = (t * p.controller_hz) % 1.0
    square = (ph < p.controller_duty).astype(float)
    eta = rng.normal(0.0, 1.0, size=t.shape) * (glue.eta_scale * 0.03)
    F = fp * (p.controller_base + p.controller_amp * square + eta)
    return _wrap(np.clip(F, 0.0, None), t, "controller_cycle", glue, meter, rng)


def _periodic_inference(
    ko_params: KoWorkloadParams, glue: KoTypeBParams, p: Rung2Params,
    rng: np.random.Generator, *, meter: MeterParams | None,
) -> TypeTrace:
    """Inference null modulated by a periodic request envelope (a periodic line)."""
    t = make_time_grid(glue.duration_s, 1.0 / glue.fs)
    F_inf = inference_F(t, ko_params, rng, f_peak=_f_peak(glue),
                        eta_scale=glue.eta_scale)
    sq = np.where(np.sin(2.0 * np.pi * p.periodic_inf_req_hz * t) >= 0.0,
                  1.0, -1.0)
    env = 1.0 + p.periodic_inf_amp * sq
    return _wrap(np.clip(F_inf * env, 0.0, None), t, "periodic_inference",
                 glue, meter, rng)


def make_control_trace(
    name: str,
    ko_params: KoWorkloadParams,
    glue: KoTypeBParams,
    rng: np.random.Generator,
    p: Rung2Params,
    *,
    meter: MeterParams | None = None,
) -> TypeTrace:
    """One trace of the named semantic control, observed through ``meter``."""
    if name == "discarded_update_decoy":
        # Physically identical to honest training (nominal knobs).
        return _training_control(ko_params, glue, rng, name, meter=meter)
    if name == "gradient_only":
        kp = dataclasses.replace(ko_params,
                                 mu_delta_tr=p.grad_only_mu_delta,
                                 sigma_delta_tr=p.grad_only_sigma_delta)
        return _training_control(kp, glue, rng, name, meter=meter)
    if name == "nonml_kernel_loop":
        return _training_control(ko_params, glue, rng, name, meter=meter,
                                 harmonic_smooth_s=p.nonml_harmonic_smooth_s)
    if name == "controller_cycle":
        return _controller_cycle(glue, p, rng, meter=meter)
    if name == "periodic_inference":
        return _periodic_inference(ko_params, glue, p, rng, meter=meter)
    if name == "async_training":
        return _training_control(ko_params, glue, rng, name, meter=meter,
                                 f0_drift_hz=p.async_f0_drift_hz,
                                 phase_slip_sigma=p.async_phase_slip_sigma)
    if name == "coresident_mixture":
        kp = dataclasses.replace(
            ko_params, aggregate_ratio=_ratio_for_share(p.coresident_share))
        tr = ko_make_aggregate_trace("train", kp, glue, rng, meter=meter)
        return dataclasses.replace(tr, label="coresident_mixture")
    raise ValueError(f"unknown control {name!r}")


def make_control_population(
    name: str,
    n: int,
    ko_params: KoWorkloadParams,
    glue: KoTypeBParams,
    rng: np.random.Generator,
    p: Rung2Params,
    *,
    meter: MeterParams | None = None,
) -> list[TypeTrace]:
    """``n`` traces of the named semantic control (observed through ``meter``)."""
    if name not in CONTROLS:
        raise ValueError(f"unknown control {name!r} (expected one of {CONTROL_ORDER})")
    return [make_control_trace(name, ko_params, glue, rng, p, meter=meter)
            for _ in range(n)]
