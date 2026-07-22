"""Within-trace surrogate calibration (ST1 task 19.8; memo Sec. 3.5; stage 9).

Corpus-free per-trace calibrators: transform the OBSERVED record into
surrogates that preserve a stated invariance and destroy the structure the
detector keys on, then read the p-value off the surrogate ensemble,

    p = (1 + #{T_s >= T_obs}) / (S + 1),

exact under the invariance null when the surrogates are exchangeable with
the observation. Two essentials, both enforced here:

    1. the statistic_fn passed to :func:`surrogate_pvalue` must be the FULL
       pipeline — tracker, phase resampling, DG, everything — run inside
       EVERY surrogate, so the selection steps are part of the calibrated
       null (a surrogate p on the final statistic with the path held fixed
       would smuggle the look-elsewhere back in);
    2. each surrogate's null is the INVARIANCE it preserves, not "not
       training" — that scope limit is stated per generator below and in the
       findings note.

Fourier-phase surrogates test a LINEAR STATIONARY GAUSSIAN-ish null (any
process fully described by its PSD): |rfft| preserved exactly, phases
randomised, so all second-order time-domain structure survives while
phase coupling — including cyclostationary modulation and a trackable
ridge's cross-frame coherence — is destroyed. Block permutation tests BLOCK
EXCHANGEABILITY: dependence within blocks survives, any structure with
continuity across block boundaries (a drifting path, long-memory
modulation) is destroyed; blocks must exceed the mixing length for the
exchangeability premise to hold.
"""

from __future__ import annotations

import numpy as np


def fourier_phase_surrogate(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Phase-randomised surrogate: |rfft| (and DC / Nyquist) preserved EXACTLY.

    Invariance null: a linear stationary process — the observed periodogram
    with independent uniform phases. DC is untouched; for even n the Nyquist
    coefficient (necessarily real) is untouched too; every interior bin keeps
    its magnitude and receives a fresh U(0, 2 pi) phase. Destroys ALL
    phase coupling (cyclostationarity, ridge coherence, non-Gaussian phase
    structure) while preserving the PSD, hence the autocovariance.
    """
    x = np.asarray(x, dtype=float)
    X = np.fft.rfft(x)
    mag = np.abs(X)
    n_bins = X.size
    hi = n_bins - 1 if x.size % 2 == 0 else n_bins   # even n: exclude Nyquist
    phases = rng.uniform(0.0, 2.0 * np.pi, size=n_bins)
    Y = X.copy()
    Y[1:hi] = mag[1:hi] * np.exp(1j * phases[1:hi])
    return np.fft.irfft(Y, n=x.size)


def block_permutation_surrogate(
    x: np.ndarray, block_len: int, rng: np.random.Generator
) -> np.ndarray:
    """Permute whole blocks of length ``block_len``; remainder stays trailing.

    Invariance null: exchangeability of the blocks — within-block dependence
    (up to lags < block_len) survives, continuity ACROSS boundaries (a
    drifting frequency path, slow modulation) is destroyed. Valid as a
    calibrator only when block_len exceeds the process mixing length; a
    partial trailing block (n not a multiple of block_len) is left in place,
    documented rather than hidden.
    """
    x = np.asarray(x, dtype=float)
    n_blocks = x.size // block_len
    if n_blocks < 2:
        raise ValueError(f"need >= 2 complete blocks, got {n_blocks} "
                         f"(n={x.size}, block_len={block_len})")
    head = x[: n_blocks * block_len].reshape(n_blocks, block_len)
    out = np.concatenate([head[rng.permutation(n_blocks)].ravel(),
                          x[n_blocks * block_len:]])
    return out


def surrogate_pvalue(
    statistic_fn,
    t: np.ndarray,
    x: np.ndarray,
    rng: np.random.Generator,
    *,
    n_surrogates: int,
    surrogate_fn=fourier_phase_surrogate,
) -> float:
    """Surrogate p-value p = (1 + #{T_s >= T_obs}) / (S + 1).

    ``statistic_fn(t, x) -> float`` must be the FULL pipeline including the
    tracker — it is re-run on every surrogate, so path selection and
    covariance estimation are inside the calibrated null. ``surrogate_fn(x,
    rng) -> x_s`` defaults to the Fourier-phase generator (see module
    docstring for each generator's invariance null). The minimum attainable
    p is 1/(S+1) — surrogate calibration cannot certify levels below that
    (the pre-registered scope limit: S = 999 supports 0.05 and 1e-2 only).
    """
    t_obs = float(statistic_fn(t, x))
    n_ge = 0
    for _ in range(n_surrogates):
        if float(statistic_fn(t, surrogate_fn(x, rng))) >= t_obs:
            n_ge += 1
    return (1.0 + n_ge) / (n_surrogates + 1.0)
