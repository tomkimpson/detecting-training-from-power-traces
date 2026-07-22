"""Sequential detection: calibrated per-block nulls and anytime-valid tests (6.4).

Every gate up to 6.2 is a supervised window-level ROC whose FAR presumes a
trusted pooled null. A deployed verifier gets one stream, no labels, and an
adversary who chooses when to cheat — the right objects are a calibrated
per-trace probability and a sequential test with error control at *every*
stopping time, not a fixed horizon. This module supplies both, detector- and
generator-agnostic (anything scored by ``code.typeb.detectors``).

Design (notes/development-notes/b2-extensions.md E4):

  Calibration is conformal, not parametric. A trace/block score s gets
  p = (1 + #{null >= s}) / (n_null + 1) against same-length null blocks —
  marginally valid for exchangeable blocks with no distributional model,
  the honest choice at n_null ~ 10^2. Resolution floor: 1/(n_null + 1).

  The headline sequential test is an e-value martingale ("testing by
  betting"): each non-overlapping block's conformal p becomes the e-value
  kappa * p^(kappa-1) (kappa in (0,1); a p-to-e calibrator), and the running
  wealth W_k = prod e_j is a nonnegative supermartingale under the null, so
  by Ville's inequality P(sup W >= 1/alpha) <= alpha — anytime-valid: the
  verifier may stop (or keep watching) whenever it likes. Detection is the
  first block where W crosses 1/alpha; time-to-detection follows.

  The Wald SPRT is carried as the faster, model-based comparison: Gaussian
  likelihoods fitted to null/alternative block scores (fit on scores AS
  GIVEN — Viterbi contrasts can be negative, so any log-transform is the
  caller's choice), boundaries log((1-beta)/alpha) / log(beta/(1-alpha)).
  Its error control is exact only if the fitted model is; the e-process
  carries the finite-sample guarantee.

  Observation longer than one capture is simulated by trace-level bootstrap
  chaining: traces sampled with replacement, block sequences concatenated in
  within-trace order. Chained captures are treated as independent
  continuations — same card, same session, so cross-trace thermal
  correlation is assumed away (stated as a caveat wherever reported).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def block_scores(
    t: np.ndarray,
    p_obs: np.ndarray,
    band_lo: float,
    band_hi: float,
    detector_fn,
    *,
    block_s: float,
) -> np.ndarray:
    """Detection statistic per consecutive non-overlapping ``block_s`` block.

    Blocks are contiguous in time and any trailing partial block is dropped,
    so scores from equal-length traces are exchangeable with same-length null
    blocks (the requirement conformal calibration rests on). Each block is
    scored independently — ``detector_fn(t_blk, x_blk, band_lo, band_hi)`` —
    so its detrend/PSD sees only that block.
    """
    t = np.asarray(t, dtype=float)
    x = np.asarray(p_obs, dtype=float)
    fs = 1.0 / (t[1] - t[0])
    n_blk = int(round(block_s * fs))
    n_blocks = x.size // n_blk
    return np.array([
        detector_fn(t[k * n_blk:(k + 1) * n_blk],
                    x[k * n_blk:(k + 1) * n_blk], band_lo, band_hi)
        for k in range(n_blocks)
    ])


def conformal_p(scores: np.ndarray, null_scores: np.ndarray) -> np.ndarray:
    """Conformal p-value of each score against a null population.

    p_i = (1 + #{null >= s_i}) / (n_null + 1): under exchangeability of
    ``s_i`` with the null scores, P(p_i <= a) <= a with no model assumption.
    Larger statistic = more training-like, so small p = training-like.
    """
    s = np.atleast_1d(np.asarray(scores, dtype=float))
    null = np.asarray(null_scores, dtype=float)
    counts = (null[None, :] >= s[:, None]).sum(axis=1)
    return (1.0 + counts) / (null.size + 1.0)


def e_calibrator(p: np.ndarray, kappa: float = 0.5) -> np.ndarray:
    """p-to-e calibrator e = kappa * p^(kappa-1), kappa in (0, 1).

    For any p-value that is superuniform under the null, E[e] <= 1, so
    products of independent-block e-values form a supermartingale. kappa=0.5
    is the standard square-root calibrator: e = 1/(2 sqrt(p)).
    """
    if not 0.0 < kappa < 1.0:
        raise ValueError(f"kappa must be in (0, 1), got {kappa}")
    p = np.asarray(p, dtype=float)
    return kappa * p ** (kappa - 1.0)


def wealth_process(p_values: np.ndarray, *, kappa: float = 0.5) -> np.ndarray:
    """Running wealth W_k = prod_{j<=k} e_j over a block p-value sequence.

    Nonnegative supermartingale under the null, so Ville gives the anytime
    guarantee P(sup_k W_k >= 1/alpha) <= alpha: rejecting at the first
    crossing of 1/alpha controls the false-alarm rate at every stopping time.
    """
    return np.cumprod(e_calibrator(p_values, kappa=kappa))


def first_crossing(wealth: np.ndarray, alpha: float) -> int | None:
    """Index of the first block where wealth >= 1/alpha (None if never)."""
    hits = np.nonzero(np.asarray(wealth) >= 1.0 / alpha)[0]
    return int(hits[0]) if hits.size else None


@dataclass(frozen=True)
class SprtModel:
    """Gaussian block-score models for the Wald SPRT (H0: null, H1: alt)."""

    mu0: float
    s0: float
    mu1: float
    s1: float


def fit_sprt(null_scores: np.ndarray, alt_scores: np.ndarray) -> SprtModel:
    """Fit the two Gaussians on the scores exactly as given.

    Callers choose the scale: log-scores are natural for the strictly
    positive spectral peak/median ratio, raw scores for the Viterbi contrast
    (which can be negative). ddof=1; a zero spread is floored so a degenerate
    fit fails loudly in the LLR rather than dividing by zero.
    """
    null = np.asarray(null_scores, dtype=float)
    alt = np.asarray(alt_scores, dtype=float)
    return SprtModel(
        mu0=float(null.mean()), s0=float(max(null.std(ddof=1), 1e-12)),
        mu1=float(alt.mean()), s1=float(max(alt.std(ddof=1), 1e-12)),
    )


def sprt_llr(scores: np.ndarray, m: SprtModel) -> np.ndarray:
    """Cumulative log-likelihood ratio log N(s|mu1,s1) - log N(s|mu0,s0)."""
    s = np.asarray(scores, dtype=float)
    ll1 = -np.log(m.s1) - 0.5 * ((s - m.mu1) / m.s1) ** 2
    ll0 = -np.log(m.s0) - 0.5 * ((s - m.mu0) / m.s0) ** 2
    return np.cumsum(ll1 - ll0)


def sprt_decision(
    llr: np.ndarray, alpha: float, beta: float
) -> tuple[int | None, int]:
    """First Wald-boundary crossing: (block index, +1 accept-H1 / -1 accept-H0).

    Boundaries A = log((1-beta)/alpha), B = log(beta/(1-alpha)). Returns
    (None, 0) if the LLR stays between them for the whole sequence.
    """
    a_hi = np.log((1.0 - beta) / alpha)
    b_lo = np.log(beta / (1.0 - alpha))
    llr = np.asarray(llr, dtype=float)
    for k, v in enumerate(llr):
        if v >= a_hi:
            return k, +1
        if v <= b_lo:
            return k, -1
    return None, 0


def chain_blocks(
    per_trace_blocks: list[np.ndarray],
    horizon_blocks: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """One bootstrap observation chain: traces resampled, block order kept.

    Traces are drawn with replacement and their block-score sequences
    concatenated in within-trace order until the chain reaches
    ``horizon_blocks`` (then truncated) — the "keep watching the same prover"
    simulation for populations captured as fixed-length traces.
    """
    per_trace_blocks = [b for b in per_trace_blocks if b.size]
    if not per_trace_blocks:
        raise ValueError("no non-empty block sequences to chain")
    chain: list[np.ndarray] = []
    total = 0
    while total < horizon_blocks:
        blocks = per_trace_blocks[int(rng.integers(len(per_trace_blocks)))]
        chain.append(blocks)
        total += blocks.size
    return np.concatenate(chain)[:horizon_blocks]


def _crossing_blocks(
    per_trace_blocks: list[np.ndarray],
    null_blocks: np.ndarray,
    *,
    alpha: float,
    kappa: float,
    horizon_blocks: int,
    n_boot: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """First-crossing block index of the e-process, per bootstrap chain.

    Each of ``n_boot`` chains is resampled, calibrated against ``null_blocks``,
    bet through the wealth process, and stopped at the first ``1/alpha``
    crossing; a chain that never crosses within the horizon contributes
    ``inf``. Shared by :func:`time_to_detection` and :func:`detection_curve`
    so their per-chain outcomes agree by construction.
    """
    hits = np.empty(n_boot)
    for b in range(n_boot):
        chain = chain_blocks(per_trace_blocks, horizon_blocks, rng)
        wealth = wealth_process(conformal_p(chain, null_blocks), kappa=kappa)
        hit = first_crossing(wealth, alpha)
        hits[b] = hit if hit is not None else np.inf
    return hits


def time_to_detection(
    per_trace_blocks: list[np.ndarray],
    null_blocks: np.ndarray,
    *,
    alpha: float,
    kappa: float,
    block_s: float,
    horizon_blocks: int,
    n_boot: int,
    rng: np.random.Generator,
) -> dict:
    """Bootstrap time-to-detection of the e-process on chained observations.

    Each of ``n_boot`` chains is calibrated against ``null_blocks``, bet
    through the wealth process, and stopped at the first 1/alpha crossing;
    a chain that never crosses within the horizon contributes ``inf``.
    Returns median/quartiles in seconds and the fraction of chains that
    detect at all — the pair of numbers the paper quotes per level.
    """
    hits = _crossing_blocks(per_trace_blocks, null_blocks, alpha=alpha,
                            kappa=kappa, horizon_blocks=horizon_blocks,
                            n_boot=n_boot, rng=rng)
    ttd = (hits + 1.0) * block_s  # inf propagates: never-detect stays inf
    # quantile interpolation between two inf order statistics warns and
    # yields nan; the honest reading of "both neighbours never detect" is inf
    with np.errstate(invalid="ignore"):
        q25, med, q75 = np.quantile(ttd, [0.25, 0.5, 0.75])
    return {
        "median_s": float(med if not np.isnan(med) else np.inf),
        "q25_s": float(q25 if not np.isnan(q25) else np.inf),
        "q75_s": float(q75 if not np.isnan(q75) else np.inf),
        "detect_frac": float(np.isfinite(ttd).mean()),
    }


def detection_curve(
    per_trace_blocks: list[np.ndarray],
    null_blocks: np.ndarray,
    *,
    alpha: float,
    kappa: float,
    block_s: float,
    horizon_blocks: int,
    n_boot: int,
    rng: np.random.Generator,
) -> dict:
    """Empirical detection probability of the e-process versus observation time.

    The survival face of :func:`time_to_detection`: instead of quantiles of
    the crossing time it returns, for each observation horizon
    ``t = (k+1)*block_s``, the fraction of the ``n_boot`` bootstrap chains that
    have crossed ``1/alpha`` by block ``k``. This is the deployment-latency
    curve — "watch for t seconds, detect with probability p_detect(t)" — at a
    fixed anytime-valid level ``alpha``. Monotone non-decreasing in ``[0, 1]``;
    ``p_detect[-1]`` equals ``detect_frac`` from :func:`time_to_detection` for
    the same rng and ``n_boot``.
    """
    hits = _crossing_blocks(per_trace_blocks, null_blocks, alpha=alpha,
                            kappa=kappa, horizon_blocks=horizon_blocks,
                            n_boot=n_boot, rng=rng)
    k = np.arange(horizon_blocks)
    # inf hits are never <= a finite block index, so only chains that have
    # already crossed by block k count — a cumulative (non-decreasing) fraction
    p_detect = np.mean(hits[None, :] <= k[:, None], axis=1)
    return {
        "t_s": block_s * (k + 1.0),
        "p_detect": p_detect,
        "detect_frac": float(p_detect[-1]),
    }
