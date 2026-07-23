"""Observation-channel (meter) map: device power -> external-sensor reading.

ST2 task 1 (plan-for-paper-2 §2). The layer stack has clean provenance:

    layer 3  generator   F(t)          code.ko_workload   (Ko et al.'s model)
    layer 2  device map  P = r F + P0  code.forward       (our forward model)
    layer 4  meter map   P_meter       THIS MODULE        (the observation channel)

so `P_meter(t) = (h * P_device)(t) + P_other(t) + eta(t)` with h the LTI
power-delivery/reporting stage, P_other the interference terms and eta the
meter noise. ST1's stage-8 LTI arm imports this module unchanged.

Composition order (matches plan-for-paper-2 §2):

    1. LTI stage on the DEVICE grid:   x = gain * notch(lowpass(P_device))
    2. integrate + sample:             trailing boxcar mean over integ_window_s,
                                       then ZOH decimation onto the sample_hz
                                       grid (decimation WITHOUT a low-pass is
                                       the deliberate aliasing case)
    3. additive terms on the METER grid:
                                       + controller interference
                                       + OU baseline wander
                                       + AR(1)/white meter noise eta

CRITICAL INVARIANT: the default :class:`code.config.MeterParams` is an exact
no-op — ``apply_meter(t, P, MeterParams(), rng)`` returns (t, P) unchanged and
draws ZERO RNG (guarded by tests/test_observation.py). Every stage is skipped,
not applied-with-neutral-values, so the byte-identical convention of
code.ko_workload's knobs carries through the whole pipeline.

RNG is an injected ``np.random.Generator`` (repo convention; no globals);
noise models are reused from :mod:`code.noise`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch

from .config import MeterParams
from .noise import AR1Noise, WhiteNoise


@dataclass(frozen=True)
class MeterTrace:
    """Output of the meter map. ``t`` is the METER grid — it differs from the
    device grid whenever ``sample_hz`` decimates."""

    t: np.ndarray        # meter time grid [s]
    P_meter: np.ndarray  # sensor reading  [W]


def _fs_of(t: np.ndarray) -> float:
    """Sample rate of a uniform grid."""
    return 1.0 / (t[1] - t[0]) if t.size > 1 else 1.0


def _lowpass(x: np.ndarray, fs: float, cutoff_hz: float, order: int) -> np.ndarray:
    """Zero-phase Butterworth low-pass (butter + filtfilt).

    filtfilt applies the filter forward and backward, so the effective
    magnitude response is |H(f)|^2 = 1 / (1 + (f/fc)^(2*order))  — the analytic
    factor the tests check against.
    """
    b, a = butter(order, cutoff_hz / (fs / 2.0), btype="low")
    return filtfilt(b, a, x)


def _notch(x: np.ndarray, fs: float, notch_hz: float, q: float,
           depth: float) -> np.ndarray:
    """Zero-phase IIR notch at ``notch_hz`` with quality factor ``q``, blended
    to ``depth``.

    ``depth`` linearly interpolates between the unfiltered signal and the full
    ``iirnotch`` null: ``x + depth * (notch(x) - x)``. ``depth == 1.0`` is the
    full null (equal up to rounding to a bare filtfilt — the blend reassociates
    the arithmetic, so ~1e-16 off, far below any physical scale); ``depth ==
    0.0`` is the identity; intermediate values give a partial anti-resonance.
    """
    b, a = iirnotch(notch_hz, q, fs=fs)
    notched = filtfilt(b, a, x)
    return x + depth * (notched - x)


def _integrate_sample(
    t: np.ndarray,
    x: np.ndarray,
    integ_window_s: float,
    sample_hz: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Trailing boxcar mean over ``integ_window_s``, then ZOH pick onto the
    ``sample_hz`` grid.

    The boxcar is TRAILING (each reading averages the recent past — how a real
    meter integrates); partial windows at the start average what is available.
    The ZOH pick takes, for each meter-grid time, the latest integrated sample
    at or before it. ``integ_window_s == 0`` skips the boxcar; ``sample_hz is
    None`` keeps the device grid. Decimating without a preceding low-pass
    aliases out-of-band power into band — deliberately reachable.
    """
    fs = _fs_of(t)
    if integ_window_s > 0.0:
        n_w = max(1, int(round(integ_window_s * fs)))
        c = np.concatenate(([0.0], np.cumsum(x)))
        i = np.arange(x.size)
        lo = np.maximum(i - n_w + 1, 0)
        x = (c[i + 1] - c[lo]) / (i + 1 - lo)
    if sample_hz is not None:
        n_out = int(np.floor((t[-1] - t[0]) * sample_hz)) + 1
        t_m = t[0] + np.arange(n_out) / sample_hz
        idx = np.clip(np.searchsorted(t, t_m, side="right") - 1, 0, t.size - 1)
        return t_m, x[idx]
    return t, x


def _controller(t: np.ndarray, mp: MeterParams, rng: np.random.Generator) -> np.ndarray:
    """Periodic controller interference on the meter grid.

    A zero-mean square-ish pulse train (template: the measured A100 power-
    management limit cycle): level ``amp*(1-duty)`` for the first ``duty``
    fraction of each cycle, ``-amp*duty`` for the rest — peak-to-peak
    ``controller_amp``, zero mean at any duty. The initial phase is drawn
    uniformly (one draw). With ``controller_wander_hz > 0`` the instantaneous
    frequency wanders as an OU process with stationary std
    ``controller_wander_hz`` and correlation time 10 controller periods (a
    documented modelling choice: slow against the line, fast against the
    window), so the interference line smears like a drifting cadence.
    """
    dt = 1.0 / _fs_of(t)
    phase0 = rng.uniform(0.0, 1.0)
    if mp.controller_wander_hz > 0.0:
        tau_s = 10.0 / mp.controller_hz
        phi = np.exp(-dt / tau_s)
        innov = mp.controller_wander_hz * np.sqrt(1.0 - phi * phi)
        w = np.empty(t.size)
        w[0] = rng.normal(0.0, mp.controller_wander_hz)
        for i in range(1, t.size):
            w[i] = phi * w[i - 1] + rng.normal(0.0, innov)
        f_inst = np.maximum(mp.controller_hz + w, 0.0)
    else:
        f_inst = np.full(t.size, mp.controller_hz)
    phase = phase0 + np.concatenate(([0.0], np.cumsum(f_inst[:-1] * dt)))
    frac = np.mod(phase, 1.0)
    duty = mp.controller_duty
    return mp.controller_amp * np.where(frac < duty, 1.0 - duty, -duty)


def _baseline(t: np.ndarray, mp: MeterParams, rng: np.random.Generator) -> np.ndarray:
    """Zero-mean OU baseline wander, marginal std ``baseline_sigma``,
    correlation time ``baseline_tau_s`` (mirrors code.ko_workload._ou_envelope:
    a unit-std OU path, empirically normalised, then scaled)."""
    n = t.size
    dt = 1.0 / _fs_of(t)
    phi = np.exp(-dt / mp.baseline_tau_s) if mp.baseline_tau_s > 0 else 0.0
    x = np.empty(n)
    x[0] = rng.normal()
    innov = np.sqrt(1.0 - phi * phi)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + rng.normal(0.0, innov)
    x = (x - x.mean()) / (x.std() + 1e-12)
    return mp.baseline_sigma * x


def apply_meter(
    t: np.ndarray,
    p_device: np.ndarray,
    mp: MeterParams,
    rng: np.random.Generator,
) -> MeterTrace:
    """Map a device power trace through the observation channel (plan §2).

    Composition (module docstring):
        P = gain * notch(lowpass(p_device))     [device grid]
          -> boxcar-integrate + ZOH-sample      [-> meter grid]
          -> + controller + baseline + eta      [meter grid]

    Each stage runs only when its knob departs from the default, so
    ``MeterParams()`` returns (t, p_device) unchanged and draws zero RNG.
    """
    t = np.asarray(t, dtype=float)
    x = np.asarray(p_device, dtype=float)
    fs = _fs_of(t)

    # 1. LTI stage (deterministic — never touches the RNG).
    if mp.lp_cutoff_hz is not None:
        x = _lowpass(x, fs, mp.lp_cutoff_hz, mp.lp_order)
    if mp.notch_hz is not None:
        x = _notch(x, fs, mp.notch_hz, mp.notch_q, mp.notch_depth)
    if mp.gain != 1.0:
        x = mp.gain * x

    # 2. Integrate + sample (deterministic; may change the grid).
    t_m, x = _integrate_sample(t, x, mp.integ_window_s, mp.sample_hz)

    # 3. Additive interference + noise on the meter grid (RNG only when on).
    if mp.controller_amp > 0.0 and mp.controller_hz > 0.0:
        x = x + _controller(t_m, mp, rng)
    if mp.baseline_sigma > 0.0:
        x = x + _baseline(t_m, mp, rng)
    if mp.sigma_eta > 0.0:
        noise = (AR1Noise(mp.sigma_eta, mp.eta_tau_s) if mp.eta_tau_s > 0.0
                 else WhiteNoise(mp.sigma_eta))
        x = x + noise.sample(t_m, rng)

    return MeterTrace(t=t_m, P_meter=x)
