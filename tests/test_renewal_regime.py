"""Unit checks for the stage-3 identifiable-regime measurement.

The quantities here decide whether E2 is ever built, so the things worth pinning
are the ones a plausible bug would silently corrupt: the AUC estimator, the
pairing that makes ``align+`` a clean within-positive comparison, the surrogate
nulls' matching properties, and each clause of the admission rule.
"""

from __future__ import annotations

import numpy as np
import pytest

from powerladder.typeb.renewal_campaign import (
    alignment_pool,
    glue_for,
    make_annotated_population,
    make_diluted_population,
    make_null_populations,
    matched_surrogate_population,
    surrogate_spectral_error,
)
from powerladder.typeb.renewal_regime import (
    ADMISSION,
    PointResult,
    admits_e2,
    benchmark_gap_recovery,
    measure_point,
    rank_auc,
)


def _result(**overrides) -> PointResult:
    base = dict(
        e1_auc=0.85, e1_auc_lo=0.78, e1_auc_hi=0.91,
        e0_true_auc=0.95, e0_random_auc=0.70,
        align_plus=0.25, align_plus_lo=0.10, align_plus_hi=0.40,
        e0_minus_e1=0.10, e0_minus_e1_lo=0.04, e0_minus_e1_hi=0.17,
        event_recall=0.30, event_timing_error_s=0.04, events_per_trace=9.0,
        degenerate_fraction_positive=0.01, degenerate_fraction_null=0.01,
        n_positive=50, n_null=50,
    )
    return PointResult(**{**base, **overrides})


# --- the AUC estimator ------------------------------------------------------

def test_rank_auc_matches_the_project_estimator():
    from powerladder.typeb.roc import auc as project_auc
    rng = np.random.default_rng(0)
    pos, neg = rng.normal(1.0, 1.0, 200), rng.normal(0.0, 1.0, 300)
    assert rank_auc(pos, neg) == pytest.approx(project_auc(pos, neg), abs=1e-9)


def test_rank_auc_handles_perfect_separation_and_ties():
    assert rank_auc(np.asarray([2.0, 3.0]), np.asarray([0.0, 1.0])) == 1.0
    assert rank_auc(np.asarray([0.0, 1.0]), np.asarray([2.0, 3.0])) == 0.0
    assert rank_auc(np.ones(10), np.ones(10)) == pytest.approx(0.5)


def test_rank_auc_is_nan_on_an_empty_side():
    assert np.isnan(rank_auc(np.asarray([]), np.asarray([1.0])))


# --- surrogate nulls match the properties they claim to ---------------------

def test_ft_surrogate_matches_mean_variance_and_spectrum():
    traces = make_annotated_population("drift", 0.0, 3, glue_for(20.0), 11)
    surrogates = matched_surrogate_population(traces, 5, kind="ft")
    for original, surrogate in zip(traces, surrogates):
        assert np.mean(surrogate.P_obs) == pytest.approx(
            np.mean(original.P_obs), rel=1e-9)
        assert np.var(surrogate.P_obs) == pytest.approx(
            np.var(original.P_obs), rel=1e-6)
        np.testing.assert_allclose(
            np.abs(np.fft.rfft(surrogate.P_obs)),
            np.abs(np.fft.rfft(original.P_obs)), rtol=1e-6, atol=1e-6,
        )


def test_aaft_surrogate_matches_the_marginal_distribution_exactly():
    traces = make_annotated_population("drift", 0.0, 3, glue_for(20.0), 11)
    surrogates = matched_surrogate_population(traces, 5, kind="aaft")
    for original, surrogate in zip(traces, surrogates):
        np.testing.assert_allclose(np.sort(surrogate.P_obs),
                                   np.sort(original.P_obs))


def test_surrogate_is_not_the_original_trace():
    """Guards against a surrogate that trivially reproduces its input."""
    traces = make_annotated_population("drift", 0.0, 2, glue_for(20.0), 11)
    for kind in ("ft", "aaft"):
        for original, surrogate in zip(traces,
                                       matched_surrogate_population(traces, 5,
                                                                    kind=kind)):
            assert not np.allclose(surrogate.P_obs, original.P_obs)


def test_surrogate_kind_is_validated():
    traces = make_annotated_population("drift", 0.0, 1, glue_for(20.0), 11)
    with pytest.raises(ValueError, match="kind must be"):
        matched_surrogate_population(traces, 5, kind="nonsense")


# --- dilution ---------------------------------------------------------------

def test_diluted_population_carries_the_dominant_workload_metadata():
    traces = make_diluted_population(0.35, 2, glue_for(20.0), 7)
    for trace in traces:
        assert trace.meta.communication_starts.size > 0
        assert np.all(trace.meta.communication_durations > 0.0)
        assert trace.P_obs.shape == trace.t.shape


def test_dilution_share_is_validated():
    for share in (0.0, 1.0, -0.2, 1.5):
        with pytest.raises(ValueError, match="share must lie"):
            make_diluted_population(share, 1, glue_for(20.0), 7)


def test_lower_share_buries_the_line_deeper():
    """A smaller dominant share must reduce the training modulation depth."""
    glue = glue_for(30.0)
    strong = make_diluted_population(0.8, 4, glue, 7)
    weak = make_diluted_population(0.1, 4, glue, 7)

    def depth(traces):
        return float(np.median([np.std(trace.P_obs) for trace in traces]))

    assert depth(weak) < depth(strong)


# --- the measurement itself -------------------------------------------------

def test_measure_point_is_deterministic_and_well_formed():
    glue = glue_for(10.0)
    fit_pos = make_annotated_population("drift", 0.0, 12, glue, 11)
    eval_pos = make_annotated_population("drift", 0.0, 12, glue, 12)
    fit_nulls = make_null_populations(4, glue, 21, families=("ar1", "inference"))
    eval_nulls = make_null_populations(4, glue, 22, families=("ar1", "inference"))
    kwargs = dict(seed=3, n_random=2, n_boot=50)
    first = measure_point(fit_pos, eval_pos, fit_nulls, eval_nulls, **kwargs)
    second = measure_point(fit_pos, eval_pos, fit_nulls, eval_nulls, **kwargs)
    assert first == second
    assert 0.0 <= first.e1_auc <= 1.0
    assert first.align_plus_lo <= first.align_plus_hi
    assert first.e0_minus_e1 == pytest.approx(first.e0_true_auc - first.e1_auc)
    assert first.align_plus == pytest.approx(
        first.e0_true_auc - first.e0_random_auc)
    assert first.n_positive == 12
    assert first.n_null == 8


def test_alignment_pool_draws_only_on_the_fitting_split():
    """Evaluation metadata must not shape the null windows."""
    glue = glue_for(20.0)
    fit_pos = make_annotated_population("drift", 0.0, 4, glue, 11)
    counts, durations = alignment_pool(fit_pos)
    assert counts.size == 4
    assert durations.size == sum(
        trace.meta.communication_durations.size for trace in fit_pos)
    assert np.all(durations > 0.0)


# --- the admission rule -----------------------------------------------

def test_admission_accepts_a_fully_qualifying_point():
    assert admits_e2(_result())["admitted"]


@pytest.mark.parametrize("override,failing_clause", [
    ({"e0_true_auc": 0.80}, "e0_true_learnable"),
    ({"e1_auc_lo": 0.48}, "e1_non_trivial"),
    ({"align_plus": 0.05}, "align_plus_large_enough"),
    ({"align_plus_lo": -0.01}, "align_plus_interval_excludes_zero"),
    ({"e0_minus_e1": 0.02}, "estimation_gap_large_enough"),
    ({"e0_minus_e1_lo": -0.02}, "estimation_gap_interval_excludes_zero"),
    ({"events_per_trace": 2.0}, "enough_events_per_trace"),
    ({"degenerate_fraction_positive": 0.25}, "positives_not_degenerate"),
    ({"degenerate_fraction_null": 0.25}, "nulls_not_degenerate"),
])
def test_admission_rejects_when_a_single_clause_fails(override, failing_clause):
    verdict = admits_e2(_result(**override))
    assert not verdict["admitted"]
    assert verdict[failing_clause] is False


def test_admission_requires_a_credible_channel():
    verdict = admits_e2(_result(), credible_channel=False)
    assert not verdict["admitted"]
    assert verdict["credible_channel"] is False


def test_degeneracy_is_gated_per_side_not_pooled():
    """Clean positives must not mask degenerate nulls (or vice versa)."""
    for side in ("degenerate_fraction_positive", "degenerate_fraction_null"):
        # Pooling 0.25 and 0.00 would average to 0.125; either side alone fails.
        assert not admits_e2(_result(**{side: 0.25}))["admitted"]


def test_there_is_no_upper_bound_on_e1():
    """Redundant once the estimation gap must exceed 0.10, and would look tuned."""
    verdict = admits_e2(_result(e1_auc=0.97, e1_auc_lo=0.94, e0_true_auc=0.99,
                                e0_minus_e1=0.10, e0_minus_e1_lo=0.03))
    assert verdict["admitted"]


def test_e0_random_saturation_is_a_diagnostic_not_a_clause():
    verdict = admits_e2(_result(e0_random_auc=0.999))
    assert verdict["diagnostic_e0_random_auc"] == 0.999
    assert "e0_random_not_saturated" not in verdict


def test_admission_thresholds_are_the_agreed_ones():
    assert ADMISSION["min_e0_true"] == 0.90
    assert ADMISSION["e1_lower_bound_exceeds"] == 0.50
    assert ADMISSION["min_align_plus"] == 0.10
    assert ADMISSION["min_e0_minus_e1"] == 0.10
    assert ADMISSION["min_events_per_trace"] == 5.0
    assert ADMISSION["max_degenerate_fraction"] == 0.05


# --- the fixed E2 normalisation ---------------------------------------------

def test_benchmark_gap_recovery_is_the_agreed_ratio():
    assert benchmark_gap_recovery(0.827, 0.661, 0.993) == pytest.approx(0.5)
    assert benchmark_gap_recovery(0.661, 0.661, 0.993) == 0.0
    assert benchmark_gap_recovery(0.993, 0.661, 0.993) == pytest.approx(1.0)


def test_benchmark_gap_recovery_may_exceed_one():
    """E0 is feature-family-specific, so a sequence model could pass it."""
    assert benchmark_gap_recovery(1.0, 0.661, 0.993) > 1.0


def test_benchmark_gap_recovery_is_nan_without_a_gap():
    assert np.isnan(benchmark_gap_recovery(0.9, 0.99, 0.99))


# --- surrogate constructions differ in what they match ----------------------

def test_iaaft_matches_the_marginal_exactly_and_the_spectrum_more_closely():
    traces = make_annotated_population("drift", 0.0, 4, glue_for(20.0), 11)
    aaft = matched_surrogate_population(traces, 5, kind="aaft")
    iaaft = matched_surrogate_population(traces, 5, kind="iaaft")
    for original, surrogate in zip(traces, iaaft):
        np.testing.assert_allclose(np.sort(surrogate.P_obs),
                                   np.sort(original.P_obs))
    aaft_error = np.median([surrogate_spectral_error(o.P_obs, s.P_obs)
                            for o, s in zip(traces, aaft)])
    iaaft_error = np.median([surrogate_spectral_error(o.P_obs, s.P_obs)
                             for o, s in zip(traces, iaaft)])
    assert iaaft_error < aaft_error


def test_aaft_spectral_error_is_nonzero_so_the_null_is_only_approximately_matched():
    """The wording guard: AAFT is marginal-matched, spectrum-*approximately*."""
    traces = make_annotated_population("drift", 0.0, 4, glue_for(20.0), 11)
    aaft = matched_surrogate_population(traces, 5, kind="aaft")
    errors = [surrogate_spectral_error(o.P_obs, s.P_obs)
              for o, s in zip(traces, aaft)]
    assert all(e > 1e-6 for e in errors)


def test_ft_surrogate_spectral_error_is_essentially_zero():
    traces = make_annotated_population("drift", 0.0, 4, glue_for(20.0), 11)
    ft = matched_surrogate_population(traces, 5, kind="ft")
    for original, surrogate in zip(traces, ft):
        assert surrogate_spectral_error(original.P_obs, surrogate.P_obs) < 1e-9
