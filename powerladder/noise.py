"""Meter-noise models for the synthetic power trace.

The forward model adds a zero-mean, non-forgeable noise term eta(t) to the true
power (paper Sec. 2: P_obs = P + eta). Stage 1 only needs white noise.
:class:`AR1Noise` (correlated) and :class:`AR1DriftNoise` (correlated +
non-stationary) drive the Floor-A study (notes/development-notes/phase1-plan.md
Sec. 3, spec A1): meter
noise averages away as 1/sqrt(N_eff) with N_eff = T / (2 tau_corr) for AR(1), so
a correlated noise model with a tunable correlation time tau_corr is what that
study exercises (see :mod:`code.floor_a`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class NoiseModel(Protocol):
    """A meter-noise generator.

    Implementations draw one realisation over the supplied time grid and expose a
    correlation time so downstream code can compute N_eff = T / (2 tau_corr).
    """

    @property
    def tau_corr(self) -> float:
        """Noise correlation time [s]; 0 for white noise."""
        ...

    def sample(self, t: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Return a noise realisation eta(t) [W] on grid ``t``."""
        ...


@dataclass(frozen=True)
class WhiteNoise:
    """Zero-mean i.i.d. Gaussian meter noise."""

    sigma: float  # std [W]

    @property
    def tau_corr(self) -> float:
        return 0.0

    def sample(self, t: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        return rng.normal(0.0, self.sigma, size=np.asarray(t).shape)


@dataclass(frozen=True)
class AR1Noise:
    """First-order autoregressive (correlated) meter noise.

    Drives the Floor-A scaling study. Stationary AR(1) with correlation time
    ``tau_corr``: the per-step retention is phi = exp(-dt / tau_corr) and the
    innovation variance is scaled so the marginal std equals ``sigma``.
    """

    sigma: float      # marginal std [W]
    tau_corr: float   # correlation time [s]

    def sample(self, t: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        t = np.asarray(t, dtype=float)
        n = t.size
        out = np.empty(n)
        if n == 0:
            return out
        dt = t[1] - t[0] if n > 1 else 1.0
        phi = np.exp(-dt / self.tau_corr) if self.tau_corr > 0 else 0.0
        innov_std = self.sigma * np.sqrt(1.0 - phi * phi)
        out[0] = rng.normal(0.0, self.sigma)
        for i in range(1, n):
            out[i] = phi * out[i - 1] + rng.normal(0.0, innov_std)
        return out


@dataclass(frozen=True)
class AR1DriftNoise:
    """Correlated, *non-stationary* meter noise: AR(1) with a drifting std.

    Same retention phi = exp(-dt / tau_corr) as :class:`AR1Noise`, but the
    marginal std drifts linearly across the window,
    sigma(t) = sigma * (1 + drift * t / T), with the innovation scaled per step
    so the local marginal std tracks sigma(t). It carries no closed form — its
    role in the Floor-A study is to show the empirical sqrt(T) energy scaling
    (hence reducibility) survives mild non-stationarity. Keep ``drift`` modest.

    ``tau_corr`` reports the stationary correlation time (for N_eff bookkeeping).
    """

    sigma: float      # base marginal std at t=0 [W]
    tau_corr: float   # correlation time [s]
    drift: float      # fractional change in std across the window [dimensionless]

    def sample(self, t: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        t = np.asarray(t, dtype=float)
        n = t.size
        out = np.empty(n)
        if n == 0:
            return out
        dt = t[1] - t[0] if n > 1 else 1.0
        T = t[-1] if n > 1 and t[-1] > 0 else 1.0
        phi = np.exp(-dt / self.tau_corr) if self.tau_corr > 0 else 0.0
        sigma_t = self.sigma * (1.0 + self.drift * t / T)  # local marginal std
        # The innovation std is sigma_t(i) * sqrt(1 - phi^2); the sqrt factor is
        # phi-invariant, so hoist it out of the loop (cf. AR1Noise.sample, where
        # the whole innovation std is constant).
        innov_factor = np.sqrt(1.0 - phi * phi)
        out[0] = rng.normal(0.0, sigma_t[0])
        for i in range(1, n):
            out[i] = phi * (sigma_t[i] / sigma_t[i - 1]) * out[i - 1] \
                + rng.normal(0.0, sigma_t[i] * innov_factor)
        return out
