"""Whittle spectral LRT — the NP-optimal detector ceiling for the ST1 bake-off.

Phase 2 (optional, non-blocking; ``tasks.md``). The strategy memo
(``notes/discussion/method-soundness-and-prior-art.md`` §2.2) asks: since we own the
generators, compute the Neyman–Pearson optimal likelihood-ratio detector between the
training and null generators, and report each corpus-free detector as a *fraction* of
it — "X% of NP-optimal power at Y% of the information cost". Neither generator has a
closed-form likelihood (both are black-box hierarchical stochastic samplers —
:mod:`powerladder.ko_workload`), so the NP-optimal test is approximated in the
frequency domain under the stationary-Gaussian **Whittle** model.

The statistic
-------------
For a detrended in-band periodogram ``I(f)`` the Whittle approximation treats the
ordinates as independent ``Exponential(mean = S(f))`` (S the PSD), giving

    ℓ_H(x) = −Σ_{f∈band} [ log S_H(f) + I_x(f) / S_H(f) ].

The negative-class PSD ``S_neg`` has no nuisance and is estimated by Monte-Carlo (the
mean periodogram over many null traces). The training class carries the unknown line
frequency ``f0``; it is the KNOWN generator nuisance, so we MARGINALISE it: build a
per-f₀ template bank ``S_tr(f; f0_k)`` (each an MC mean periodogram of training traces
at fixed ``f0_k``, pooled over the drift grid), and take

    LR(x) = logsumexp_k ℓ_tr(x | f0_k)  −  ℓ_neg(x)          (uniform f₀ prior).

Larger ``LR`` ⇒ more training-like, so a ``WhittleCeiling`` scores like every other
detector (:mod:`powerladder.typeb.roc`) and drops into the bake-off.

The caveat (state it wherever the number is reported)
-----------------------------------------------------
This is NP-optimal *under the Whittle model only*. It is a 2nd-order/periodogram
statistic: it discards harmonic-PHASE coherence and the null's non-Gaussian OU/
Poisson-burst structure, so the true optimum can exceed it. Where a tracking detector
(Viterbi / DG-order) approaches or beats this ceiling at high drift, that reflects
wandering-line structure a stationary spectral template cannot see — a finding, not a
bug. The template bank is built drift-AGNOSTIC (pooled over the drift grid), so the
ceiling knows no more about the adversary's drift than a deployable detector does.

This module is generator-aware only through :func:`train_obs_at_f0` (it must fix the
training f₀ to build the bank); PSD estimation and scoring consume plain ``(t, P)``
arrays, so any sampler can feed them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.special import logsumexp

from ..config import DEFAULT, KoTypeBParams, KoWorkloadParams
from ..forward import TraceSpec, make_time_grid, simulate
from ..ko_workload import training_F
from ..noise import WhiteNoise

# A sampler yields one observed trace as (t, P_obs); the corpus streams through these
# so the MC corpus is never held in memory all at once.
Sampler = Callable[[np.random.Generator], "tuple[np.ndarray, np.ndarray]"]


# ---------------------------------------------------------------------------
# periodogram + PSD estimation
# ---------------------------------------------------------------------------

def band_periodogram(
    t: np.ndarray, p_obs: np.ndarray, band_lo: float, band_hi: float
) -> tuple[np.ndarray, np.ndarray]:
    """In-band one-sided periodogram of a DC-removed trace.

    ``I(f) = |rFFT(P − mean P)|² / n`` restricted to ``[band_lo, band_hi]``. The
    ``1/n`` scale is arbitrary but MUST match between the estimated PSD ``S`` and the
    scored trace's ``I`` (it does — both go through this function), so the Whittle
    ratio ``I/S`` and the ``log S`` template contrasts are self-consistent. Returns
    ``(I_band, freqs_band)``. The DC bin is dropped by the band mask (band_lo > 0).
    """
    x = np.asarray(p_obs, dtype=float)
    x = x - x.mean()
    n = x.size
    fs = 1.0 / (t[1] - t[0])
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    ordinates = (np.abs(np.fft.rfft(x)) ** 2) / n
    mask = (freqs >= band_lo) & (freqs <= band_hi)
    return ordinates[mask], freqs[mask]


def mean_periodogram(
    sampler: Sampler,
    n_mc: int,
    rng: np.random.Generator,
    *,
    band_lo: float,
    band_hi: float,
) -> tuple[np.ndarray, np.ndarray]:
    """MC estimate of a class PSD: the mean in-band periodogram over ``n_mc`` draws.

    Streams (accumulates the periodogram sum; never holds the corpus), so ``n_mc`` can
    be large. Returns ``(S_band, freqs_band)``.
    """
    if n_mc < 1:
        raise ValueError("n_mc must be >= 1")
    acc: np.ndarray | None = None
    freqs: np.ndarray | None = None
    for _ in range(n_mc):
        t, p_obs = sampler(rng)
        ordinates, f = band_periodogram(t, p_obs, band_lo, band_hi)
        if acc is None:
            acc = ordinates.copy()
            freqs = f
        else:
            acc += ordinates
    assert acc is not None and freqs is not None
    return acc / float(n_mc), freqs


# ---------------------------------------------------------------------------
# training trace at a FIXED f0 (the one generator-aware helper)
# ---------------------------------------------------------------------------

def train_obs_at_f0(
    f0: float,
    f0_drift_hz: float,
    ko_p: KoWorkloadParams,
    glue: KoTypeBParams,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """One observed TRAINING power trace with the line frequency pinned to ``f0``.

    Reproduces the ``meter=None`` observation path of
    :func:`powerladder.typeb.ko_synth.ko_make_trace` (``P = r·F + P0 + white η``,
    ``η`` owned by the glue) EXACTLY, except it fixes ``f0`` instead of drawing it —
    the template bank needs the class-conditional periodogram at a known ``f0``.
    """
    f_max = DEFAULT.floor.F_max
    f_peak = glue.f_peak_frac * f_max
    t = make_time_grid(glue.duration_s, 1.0 / glue.fs)
    F = training_F(
        t, ko_p, rng, f_peak=f_peak, f0=f0,
        eta_scale=glue.eta_scale, f0_drift_hz=f0_drift_hz,
    )
    spec = TraceSpec(t=t, F=F, r=np.full_like(t, glue.r), P0=np.full_like(t, glue.P0))
    trace = simulate(spec, WhiteNoise(glue.sigma_eta), rng)
    return trace.t, trace.P_obs


# ---------------------------------------------------------------------------
# the ceiling: Whittle log-likelihoods, template bank, scoring
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class WhittleCeiling:
    """A fitted f₀-marginalised Whittle LRT for one (training, negative) pair.

    Holds the negative PSD and the training per-f₀ template bank, with the per-bin
    reciprocals and the ``Σ log S`` constants precomputed so scoring is a single
    matvec per trace. ``neg_label`` names the null the ceiling was fit against (the
    bank is shared across negatives; ``S_neg`` is not).
    """

    freqs: np.ndarray                 # in-band frequency grid (F,)
    f0_grid: np.ndarray               # training template centres (K,)
    band_lo: float
    band_hi: float
    neg_label: str
    # precomputed Whittle terms
    inv_S_bank: np.ndarray            # 1/S_tr(f; f0_k)   (K, F)
    c_bank: np.ndarray                # Σ_f log S_tr(·; f0_k)   (K,)
    inv_S_neg: np.ndarray             # 1/S_neg(f)   (F,)
    c_neg: float                      # Σ_f log S_neg(f)

    def score(self, t: np.ndarray, p_obs: np.ndarray) -> float:
        """NP-ceiling log-LR for one trace (larger ⇒ more training-like)."""
        ordinates, _ = band_periodogram(t, p_obs, self.band_lo, self.band_hi)
        # ℓ_tr(x | f0_k) = −(c_k + Σ_f I(f)/S_k(f)); marginalise f₀ by logsumexp.
        loglik_tr = -(self.c_bank + self.inv_S_bank @ ordinates)     # (K,)
        marg_tr = float(logsumexp(loglik_tr))
        loglik_neg = -(self.c_neg + float(self.inv_S_neg @ ordinates))
        return float(marg_tr - loglik_neg)


def score_ceiling(ceiling: WhittleCeiling, traces) -> np.ndarray:
    """Score an iterable of traces (objects with ``.t`` / ``.P_obs``)."""
    return np.array([ceiling.score(tr.t, tr.P_obs) for tr in traces])


def _bank_from_psds(
    freqs: np.ndarray,
    f0_grid: np.ndarray,
    S_bank: np.ndarray,
    S_neg: np.ndarray,
    band_lo: float,
    band_hi: float,
    neg_label: str,
    psd_floor: float,
) -> WhittleCeiling:
    """Assemble a :class:`WhittleCeiling` from raw PSDs (floors, precomputes terms)."""
    S_bank = np.maximum(S_bank, psd_floor)
    S_neg = np.maximum(S_neg, psd_floor)
    return WhittleCeiling(
        freqs=freqs,
        f0_grid=f0_grid,
        band_lo=band_lo,
        band_hi=band_hi,
        neg_label=neg_label,
        inv_S_bank=1.0 / S_bank,
        c_bank=np.sum(np.log(S_bank), axis=1),
        inv_S_neg=1.0 / S_neg,
        c_neg=float(np.sum(np.log(S_neg))),
    )


def build_training_bank(
    ko_p: KoWorkloadParams,
    glue: KoTypeBParams,
    *,
    f0_grid: np.ndarray,
    drift_grid,
    n_mc: int,
    rng: np.random.Generator,
    band_lo: float,
    band_hi: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-f₀ training template bank, pooled over ``drift_grid``.

    For each ``f0_k`` the template is the MC mean periodogram of ``n_mc`` training
    traces at that fixed ``f0_k``, each drawing its drift level uniformly from
    ``drift_grid`` (so the bank is drift-agnostic). Returns ``(S_bank[K,F], freqs)``.
    """
    drift_grid = np.asarray(drift_grid, dtype=float)
    bank = []
    freqs: np.ndarray | None = None
    for f0 in f0_grid:
        def sampler(r: np.random.Generator, _f0=float(f0)) -> tuple[np.ndarray, np.ndarray]:
            drift = float(drift_grid[r.integers(drift_grid.size)])
            return train_obs_at_f0(_f0, drift, ko_p, glue, r)
        S, f = mean_periodogram(sampler, n_mc, rng, band_lo=band_lo, band_hi=band_hi)
        bank.append(S)
        freqs = f
    assert freqs is not None
    return np.array(bank), freqs


def fit_ceiling(
    neg_sampler: Sampler,
    neg_label: str,
    *,
    S_bank: np.ndarray,
    f0_grid: np.ndarray,
    bank_freqs: np.ndarray,
    n_mc: int,
    rng: np.random.Generator,
    band_lo: float,
    band_hi: float,
    psd_floor: float = 1e-12,
) -> WhittleCeiling:
    """Fit a ceiling for one negative class, reusing a prebuilt training bank.

    Estimates ``S_neg`` from ``neg_sampler`` (``n_mc`` draws) on the SAME in-band grid
    as the bank (asserted), then assembles the :class:`WhittleCeiling`.
    """
    S_neg, freqs = mean_periodogram(neg_sampler, n_mc, rng, band_lo=band_lo, band_hi=band_hi)
    if not np.array_equal(freqs, bank_freqs):
        raise ValueError("negative-PSD frequency grid differs from the training bank")
    return _bank_from_psds(
        freqs, np.asarray(f0_grid, dtype=float), S_bank, S_neg,
        band_lo, band_hi, neg_label, psd_floor,
    )
