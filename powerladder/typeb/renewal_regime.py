"""Measuring whether a regime is *identifiable* for a duration-aware detector.

Stage 3 of the duration-aware extension asks a different question from stages 1
and 2.  Not "can an HSMM beat E1", but:

> can we construct a non-saturated regime in which knowledge of the true event
> locations measurably helps?

Until that answer is yes, an explicit-duration model has no identified target.
See ``notes/plans/hsmm-e2-preconditions-plan.md`` and
``notes/results/hsmm-e0-defect-and-fair-ceiling.md``.

Three quantities are measured at every candidate operating point:

1. **Blind discrimination** ``E1`` -- the deployable score, applied symmetrically
   to positives and nulls.
2. **Alignment benefit** ``align+`` -- true versus randomised alignments *within
   the same positive traces*.  This is the clean comparison: it differences away
   every global positive-vs-null difference, leaving only what knowing the event
   locations buys.
3. **Estimation gap** ``E0 - E1`` -- true-aligned positive windows against
   matched random null windows, compared with the blind score.

All three carry bootstrap intervals over the evaluation draw.  Those intervals
hold the fitted profiles fixed, so they do **not** include fitting-set
variability, and they are pointwise rather than corrected for the number of
operating points searched.  Confirming a selected cell therefore requires
replication over fresh seeds (``scripts/hsmm_confirm_cell.py``), not a wider
interval here.

``E0`` is an **alignment-informed benchmark for this feature family**, not a
universal oracle ceiling: a sequence model could exploit ordering information
that these trace-level summaries discard, and could in principle exceed it.
Matching the null windows' counts and durations to the positives deliberately
conditions away metadata-derived duration evidence, so what remains is whether
the *waveform at the true locations* matters.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.stats import rankdata

from powerladder.typeb.renewal import fit_likelihood_from_features
from powerladder.typeb.renewal_campaign import (
    FEATURE_SETS,
    alignment_pool,
    blind_matrix,
    oracle_matrix,
    oracle_null_matrix,
    profile_scores,
    timing_metrics,
)


def rank_auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Mann-Whitney AUC with correct tie handling; fast enough to bootstrap."""
    pos = np.asarray(pos, dtype=float)
    neg = np.asarray(neg, dtype=float)
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    ranks = rankdata(np.concatenate([pos, neg]))
    n1 = pos.size
    return float((ranks[:n1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * neg.size))


def benchmark_gap_recovery(auc_e2: float, auc_e1: float,
                           auc_e0: float) -> float:
    """``R``: the share of the measured E0-E1 gap that a new model recovers.

    Deliberately *not* called a fraction of an oracle ceiling: ``E0`` is specific
    to this feature family, so ``R`` can legitimately exceed 1.
    """
    span = auc_e0 - auc_e1
    if span <= 0.0:
        return float("nan")
    return float((auc_e2 - auc_e1) / span)


@dataclass(frozen=True)
class PointResult:
    """Everything the E2 admission rule needs at one operating point."""

    e1_auc: float
    e1_auc_lo: float
    e1_auc_hi: float
    e0_true_auc: float
    e0_random_auc: float
    align_plus: float
    align_plus_lo: float
    align_plus_hi: float
    e0_minus_e1: float
    e0_minus_e1_lo: float
    e0_minus_e1_hi: float
    event_recall: float
    event_timing_error_s: float | None
    events_per_trace: float
    degenerate_fraction_positive: float
    degenerate_fraction_null: float
    n_positive: int
    n_null: int

    def as_dict(self) -> dict:
        return asdict(self)


def _fit_and_score(train: np.ndarray, null_fit: np.ndarray,
                   eval_pos: np.ndarray, eval_null: np.ndarray,
                   feature_set: str) -> tuple[np.ndarray, np.ndarray]:
    transform = FEATURE_SETS[feature_set]
    model = fit_likelihood_from_features(transform(train), [transform(null_fit)])
    return (profile_scores(model, transform(eval_pos)),
            profile_scores(model, transform(eval_null)))


def _degenerate_fraction(features: np.ndarray) -> float:
    if features.size == 0:
        return 0.0
    return float(np.mean(np.all(features == 0.0, axis=1)))


def _bootstrap(scores: dict[str, np.ndarray], rng: np.random.Generator,
               n_boot: int, alpha: float = 0.05) -> dict[str, tuple[float, float]]:
    """Percentile intervals for E1, align+ and the estimation gap.

    One resampling of positives and one of nulls per replicate, shared by all
    three statistics.  That preserves the trace-level pairing that makes align+
    and the gap clean comparisons: the same evaluation traces feed the blind, the
    true-aligned and the random-aligned arms, and the same null traces feed all
    three featurisations.
    """
    n_pos = scores["e1_pos"].size
    n_null = scores["e1_neg"].size
    e1 = np.empty(n_boot)
    align = np.empty(n_boot)
    gap = np.empty(n_boot)
    for b in range(n_boot):
        pi = rng.integers(0, n_pos, n_pos)
        ni = rng.integers(0, n_null, n_null)
        e1_b = rank_auc(scores["e1_pos"][pi], scores["e1_neg"][ni])
        true_b = rank_auc(scores["true_pos"][pi], scores["true_neg"][ni])
        random_b = rank_auc(scores["random_pos"][pi], scores["random_neg"][ni])
        e1[b], align[b], gap[b] = e1_b, true_b - random_b, true_b - e1_b
    q = [alpha / 2.0, 1.0 - alpha / 2.0]
    return {
        "e1": tuple(float(v) for v in np.quantile(e1, q)),
        "align_plus": tuple(float(v) for v in np.quantile(align, q)),
        "e0_minus_e1": tuple(float(v) for v in np.quantile(gap, q)),
    }


def measure_point(
    fit_pos,
    eval_pos,
    fit_nulls,
    eval_nulls,
    *,
    seed: int,
    feature_set: str = "full",
    n_random: int = 8,
    n_boot: int = 1000,
) -> PointResult:
    """Measure E1, align+ and the estimation gap at one operating point.

    ``fit_*`` and ``eval_*`` must be disjoint draws.  The alignment distribution
    (event counts and durations) is taken from the *fitting* positives only, so no
    evaluation metadata leaks into the null windows.
    """
    fit_null_list = [trace for group in fit_nulls.values() for trace in group]
    eval_null_list = [trace for group in eval_nulls.values() for trace in group]

    # --- 1. blind discrimination, both sides through the same procedure ------
    blind_eval_pos = blind_matrix(eval_pos)
    blind_eval_null = blind_matrix(eval_null_list)
    e1_pos, e1_neg = _fit_and_score(
        blind_matrix(fit_pos), blind_matrix(fit_null_list),
        blind_eval_pos, blind_eval_null, feature_set,
    )

    # --- 2/3. alignment-informed benchmark ------------------------------------
    counts, durations = alignment_pool(fit_pos)
    null_fit_windows = oracle_null_matrix(fit_null_list, counts, durations,
                                          seed + 101)
    null_eval_windows = oracle_null_matrix(eval_null_list, counts, durations,
                                           seed + 202)

    true_pos, true_neg = _fit_and_score(
        oracle_matrix(fit_pos), null_fit_windows,
        oracle_matrix(eval_pos), null_eval_windows, feature_set,
    )

    random_fit = np.concatenate(
        [oracle_null_matrix(fit_pos, counts, durations, seed + 303 + k)
         for k in range(n_random)], axis=0,
    )
    # One score per evaluation trace, averaged over independent alignments, so
    # the random arm pairs 1:1 with the true arm for the bootstrap.
    random_draws = [
        _fit_and_score(random_fit, null_fit_windows,
                       oracle_null_matrix(eval_pos, counts, durations,
                                          seed + 404 + k),
                       null_eval_windows, feature_set)
        for k in range(n_random)
    ]
    random_pos = np.mean([draw[0] for draw in random_draws], axis=0)
    random_neg = random_draws[0][1]

    scores = {"e1_pos": e1_pos, "e1_neg": e1_neg,
              "true_pos": true_pos, "true_neg": true_neg,
              "random_pos": random_pos, "random_neg": random_neg}
    intervals = _bootstrap(scores, np.random.default_rng(seed + 909), n_boot)

    e1_auc = rank_auc(e1_pos, e1_neg)
    e0_true = rank_auc(true_pos, true_neg)
    e0_random = rank_auc(random_pos, random_neg)
    timing = timing_metrics(eval_pos)
    return PointResult(
        e1_auc=e1_auc,
        e1_auc_lo=intervals["e1"][0],
        e1_auc_hi=intervals["e1"][1],
        e0_true_auc=e0_true,
        e0_random_auc=e0_random,
        align_plus=float(e0_true - e0_random),
        align_plus_lo=intervals["align_plus"][0],
        align_plus_hi=intervals["align_plus"][1],
        e0_minus_e1=float(e0_true - e1_auc),
        e0_minus_e1_lo=intervals["e0_minus_e1"][0],
        e0_minus_e1_hi=intervals["e0_minus_e1"][1],
        event_recall=float(timing["recall_mean"]),
        event_timing_error_s=timing["timing_error_median_s"],
        events_per_trace=float(np.median(
            [trace.meta.communication_starts.size for trace in eval_pos])),
        # Gated separately: a regime can be degenerate on one side only, and
        # pooling would let a clean positive population mask empty nulls.
        degenerate_fraction_positive=_degenerate_fraction(blind_eval_pos),
        degenerate_fraction_null=_degenerate_fraction(blind_eval_null),
        n_positive=len(eval_pos),
        n_null=len(eval_null_list),
    )


#: E2 admission rule (final form, agreed 2026-08-08).
#:
#: Stated conceptually rather than by tuning thresholds to an observed value:
#:
#: * ``E0_true >= 0.90`` -- the channel demonstrably contains learnable
#:   information at the true event locations.
#: * ``E1`` non-trivial *inferentially* -- its lower 95% bound clears chance.
#:   There is deliberately **no upper bound**: it is redundant once the
#:   estimation gap must exceed 0.10, and an upper bound set near an observed
#:   value would look tuned to it.
#: * ``align+`` and ``E0 - E1`` each at least 0.10 with intervals excluding zero.
#: * event-count and degeneracy guards, the latter applied to positives and nulls
#:   separately.
#:
#: ``E0_random`` saturation is retained as a reported diagnostic only: it is
#: implied by ``align+ >= 0.10``, since ``E0_true`` cannot exceed 1.
ADMISSION = {
    "min_e0_true": 0.90,
    "e1_lower_bound_exceeds": 0.50,
    "min_align_plus": 0.10,
    "min_e0_minus_e1": 0.10,
    "min_events_per_trace": 5.0,
    "max_degenerate_fraction": 0.05,
}


def admits_e2(result: PointResult, *, credible_channel: bool = True) -> dict:
    """Apply the admission rule, returning each clause's verdict separately."""
    clauses = {
        "e0_true_learnable": result.e0_true_auc >= ADMISSION["min_e0_true"],
        "e1_non_trivial": (result.e1_auc_lo
                           > ADMISSION["e1_lower_bound_exceeds"]),
        "align_plus_large_enough": result.align_plus >= ADMISSION["min_align_plus"],
        "align_plus_interval_excludes_zero": result.align_plus_lo > 0.0,
        "estimation_gap_large_enough": (result.e0_minus_e1
                                        >= ADMISSION["min_e0_minus_e1"]),
        "estimation_gap_interval_excludes_zero": result.e0_minus_e1_lo > 0.0,
        "enough_events_per_trace": (result.events_per_trace
                                    >= ADMISSION["min_events_per_trace"]),
        "positives_not_degenerate": (result.degenerate_fraction_positive
                                     <= ADMISSION["max_degenerate_fraction"]),
        "nulls_not_degenerate": (result.degenerate_fraction_null
                                 <= ADMISSION["max_degenerate_fraction"]),
        "credible_channel": credible_channel,
    }
    return {
        **clauses,
        "admitted": all(clauses.values()),
        "diagnostic_e0_random_auc": result.e0_random_auc,
    }
