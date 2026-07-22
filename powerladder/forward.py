"""The forward model.

A synthetic accelerator emits true power P(t) = r(t) F(t) + P0(t), and an
off-chip meter reads P_obs(t) = P(t) + eta(t) (paper Sec. 2). State is carried as
full time-series arrays for F, r and P0 (not scalars) so the later Sec. 4.1
EKF/UKF tracker can reuse :func:`simulate` unchanged, including the adversarial
case where r(t) co-modulates with F(t).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .noise import NoiseModel


@dataclass(frozen=True)
class TraceSpec:
    """Inputs defining one synthetic experiment (all arrays share shape (N,))."""

    t: np.ndarray   # time grid       [s]
    F: np.ndarray   # FLOP rate       [FLOP/s]
    r: np.ndarray   # exchange rate   [J/FLOP]
    P0: np.ndarray  # overhead power  [W]


@dataclass(frozen=True)
class Trace:
    """Output of the forward model."""

    t: np.ndarray       # time grid           [s]
    P: np.ndarray       # true power r F + P0  [W]
    P_obs: np.ndarray   # observed P + eta     [W]
    spec: TraceSpec     # provenance


def make_time_grid(T: float, dt: float) -> np.ndarray:
    """Uniform grid [0, T] with step ``dt`` (endpoint included)."""
    n = int(round(T / dt)) + 1
    return np.linspace(0.0, T, n)


def constant_spec(T: float, dt: float, F: float, r: float, P0: float) -> TraceSpec:
    """Build a flat (constant F, r, P0) spec — the A0 / energy-integral regime."""
    t = make_time_grid(T, dt)
    ones = np.ones_like(t)
    return TraceSpec(t=t, F=F * ones, r=r * ones, P0=P0 * ones)


def simulate(spec: TraceSpec, noise: NoiseModel, rng: np.random.Generator) -> Trace:
    """Evaluate P = r F + P0 elementwise and add a noise realisation."""
    P = spec.r * spec.F + spec.P0
    P_obs = P + noise.sample(spec.t, rng)
    return Trace(t=spec.t, P=P, P_obs=P_obs, spec=spec)
