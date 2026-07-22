"""Minimal, B0-LOCAL synthetic power traces for the Type decision gate.

Generates a labeled power trace P_obs(t) for one of two regimes:

    "train"  — a periodic iteration signature at cadence f0 in [0.5, 1.5] Hz with
               a compute/comm duty cycle (a square-ish wave -> a fundamental line
               at f0 plus harmonics: the f_step training line, spec.md Sec. 4 B0).
    "infer"  — a *structured* null: slow, heavily-jittered batch / prefill-decode
               fluctuation with no concentrated line in the training f0 band
               (the false-positive risk of spec.md Sec. 4 B1, premise 2).

PLACEHOLDER. This is NOT the shared Ko et al. trace generator (arXiv 2508.16457,
eqs 1-10) — that is Task 1.1, built in a separate worktree and reused by B1. This
module exists only to drive the B0 detector comparison and exercise the
frequency-wander regime; its bands/duty follow Ko et al. so the structure is
faithful, but the detectors (code.typeb.detectors) consume only (t, P_obs), so
the real generator drops in with a one-line import change in the figure scripts.

The only physics that matters for the gate: training concentrates power into a
narrow line at f0; honest smearing and the jitter adversary *wander* that line
(``wander_hz`` > 0), which is the regime that separates the two detectors
(plan Sec. 3 B0: matched filter loses SNR as the line smears; the Viterbi
tracker follows it).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import TypeBParams


@dataclass(frozen=True)
class TypeTrace:
    """One labeled synthetic power trace (mirrors code.forward.Trace fields)."""

    t: np.ndarray       # time grid                       [s]
    P_obs: np.ndarray   # observed power P + meter noise   [W]
    label: str          # "train" or "infer" (ground truth)
    f0: float           # nominal iteration cadence        [Hz] (NaN for infer)


def _duty_wave(phase: np.ndarray, duty: float, n_harmonics: int) -> np.ndarray:
    """Zero-mean square-ish wave from a band-limited Fourier sum of a duty cycle.

    A pulse train of duty ``duty`` has Fourier coefficients
    ``a_k = (2/(k pi)) sin(k pi duty)``; summing the first ``n_harmonics`` gives a
    smooth wave whose fundamental sits at the phase frequency. The DC term is
    dropped so the wave is zero-mean (the baseline lives in ``P_base``).
    """
    out = np.zeros_like(phase)
    for k in range(1, n_harmonics + 1):
        a_k = (2.0 / (k * np.pi)) * np.sin(k * np.pi * duty)
        out += a_k * np.cos(k * phase)
    return out


def _band_noise(
    n: int, fs: float, band_lo: float, band_hi: float, rms: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Diffuse band-limited noise: white noise spectrum kept only in [lo, hi].

    The inference null's in-band power. It has no coherent frequency path (every
    in-band bin is independent), so it stresses concentration-vs-diffuseness: a
    matched filter sees it as a raised floor, a line tracker finds no path through
    it. Scaled to a target RMS so the null's in-band power can be set comparable
    to the training line's.
    """
    w = rng.normal(0.0, 1.0, size=n)
    W = np.fft.rfft(w)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    W[(freqs < band_lo) | (freqs > band_hi)] = 0.0
    x = np.fft.irfft(W, n=n)
    return x / (x.std() + 1e-12) * rms


def _wandering_phase(
    t: np.ndarray, f0: float, wander_hz: float, jitter: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Accumulated phase 2*pi*int f(tau) dtau for a slowly-wandering frequency.

    ``f(t)`` is a bounded random walk around ``f0`` with total excursion scale
    ``wander_hz`` (0 -> a stationary line). ``jitter`` adds a small white phase
    perturbation per sample (the e^{-w^2 sigma^2} line attenuation, spec Sec. 4 B1).
    """
    n = t.size
    dt = t[1] - t[0] if n > 1 else 1.0
    if wander_hz > 0:
        # Slow random walk in frequency, scaled so the std over the window ~ wander_hz.
        steps = rng.normal(0.0, 1.0, size=n)
        walk = np.cumsum(steps)
        walk = walk - walk.mean()
        walk = walk / (walk.std() + 1e-12) * wander_hz
        f_t = np.clip(f0 + walk, 0.05, None)
    else:
        f_t = np.full(n, f0)
    phase = 2.0 * np.pi * np.cumsum(f_t) * dt
    if jitter > 0:
        phase = phase + rng.normal(0.0, jitter, size=n)
    return phase


def make_trace(
    label: str,
    p: TypeBParams,
    rng: np.random.Generator,
    *,
    wander_hz: float | None = None,
) -> TypeTrace:
    """Generate one labeled trace. ``wander_hz`` overrides ``p.wander_hz`` if given.

    ``label`` is "train" or "infer". The training f0, duty cycle, and inference
    cadence are drawn per-trace from the Ko et al. ranges in ``p`` so a population
    of traces spans the band rather than sitting at one frequency.
    """
    n = int(round(p.duration_s * p.fs)) + 1
    t = np.arange(n) / p.fs
    wander = p.wander_hz if wander_hz is None else wander_hz
    eta = rng.normal(0.0, p.sigma_eta, size=n)
    intra = rng.normal(0.0, p.intra_sigma, size=n)

    if label == "train":
        f0 = rng.uniform(p.f0_train_lo, p.f0_train_hi)
        duty = rng.uniform(p.duty_train_lo, p.duty_train_hi)
        phase = _wandering_phase(t, f0, wander, p.phase_jitter, rng)
        signal = p.amp_train * _duty_wave(phase, duty, p.n_harmonics)
        P = p.P_base + signal + intra
        return TypeTrace(t=t, P_obs=P + eta, label="train", f0=f0)

    if label == "infer":
        # Diffuse in-band power (no concentrated line) + a slow out-of-band batch
        # cadence. The in-band part is the hard part of the null: comparable power
        # to training, but spread across the band with no trackable path.
        f0_slow = rng.uniform(p.f0_infer_lo, p.f0_infer_hi)
        slow = p.amp_infer_slow * np.cos(2.0 * np.pi * f0_slow * t + rng.uniform(0, 2 * np.pi))
        diffuse = _band_noise(n, p.fs, p.band_lo, p.band_hi, p.amp_infer_band, rng)
        P = p.P_base + slow + diffuse + intra
        return TypeTrace(t=t, P_obs=P + eta, label="infer", f0=np.nan)

    raise ValueError(f"unknown label {label!r} (expected 'train' or 'infer')")


def make_population(
    n_each: int,
    p: TypeBParams,
    rng: np.random.Generator,
    *,
    wander_hz: float | None = None,
) -> tuple[list[TypeTrace], list[TypeTrace]]:
    """Return (training traces, inference traces), ``n_each`` of each regime."""
    train = [make_trace("train", p, rng, wander_hz=wander_hz) for _ in range(n_each)]
    infer = [make_trace("infer", p, rng, wander_hz=wander_hz) for _ in range(n_each)]
    return train, infer
