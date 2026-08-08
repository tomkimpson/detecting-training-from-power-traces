"""Unit checks for the duration-aware campaign harness.

The two properties worth pinning are the ones every campaign built on this module
relies on and that no individual result would reveal if broken:

1. **latent metadata never reaches a deployable score** -- the blind path must be
   a function of ``(t, P_obs)`` alone, so changing a trace's phase metadata must
   not move a blind feature; and
2. **channel symmetry** -- degrading the observation channel must put every
   positive and every null on the identical time grid.

Plus the threshold machinery, which sets every reported operating point, and the
fit-plan splitting in ``scripts/hsmm_generalization.py`` that keeps fit-set size
constant across configurations.

Loaded by path via importlib because scripts/ is not an importable package --
the same idiom as tests/test_st2_pareto.py.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import pathlib

import numpy as np
import pytest

from powerladder.config import MeterParams
from powerladder.ko_workload import TrainingPhaseMetadata
from powerladder.typeb.renewal import (
    FEATURE_NAMES,
    fit_gaussian_profile,
    widen_profile,
)
from powerladder.typeb.renewal_campaign import (
    NULL_FAMILIES,
    AnnotatedTrace,
    blind_matrix,
    composite_threshold,
    glue_for,
    make_annotated_population,
    make_null_populations,
    alignment_pool,
    null_rates,
    oracle_matrix,
    oracle_null_matrix,
    threshold_at_far,
    timing_metrics,
)

_SCRIPT = (pathlib.Path(__file__).resolve().parent.parent
           / "scripts" / "hsmm_generalization.py")
_spec = importlib.util.spec_from_file_location("hsmm_generalization", _SCRIPT)
generalization = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(generalization)

_DURATION_S = 20.0


def _glue():
    return glue_for(_DURATION_S)


# --- 1. the blind path is a function of (t, P_obs) alone --------------------

def test_blind_features_ignore_latent_phase_metadata():
    traces = make_annotated_population("drift", 0.0, 2, _glue(), 11)
    corrupted = [
        AnnotatedTrace(
            trace.t, trace.P_obs,
            TrainingPhaseMetadata(
                compute_starts=trace.meta.compute_starts + 3.7,
                communication_starts=trace.meta.communication_starts + 3.7,
                compute_durations=trace.meta.compute_durations * 2.0,
                communication_durations=trace.meta.communication_durations * 2.0,
            ),
        )
        for trace in traces
    ]
    np.testing.assert_array_equal(blind_matrix(traces), blind_matrix(corrupted))


def test_oracle_features_do_depend_on_phase_metadata():
    """Guards the test above from passing because the oracle is also blind."""
    traces = make_annotated_population("drift", 0.0, 2, _glue(), 11)
    shifted = [
        AnnotatedTrace(
            trace.t, trace.P_obs,
            dataclasses.replace(
                trace.meta,
                communication_starts=trace.meta.communication_starts + 1.3,
            ),
        )
        for trace in traces
    ]
    assert not np.allclose(oracle_matrix(traces), oracle_matrix(shifted))


# --- 2. channel symmetry ----------------------------------------------------

@pytest.mark.parametrize("sample_hz,integ_window_s",
                         [(20.0, 0.0), (5.0, 0.25), (2.0, 0.5)])
def test_meter_puts_positives_and_every_null_on_one_grid(sample_hz,
                                                         integ_window_s):
    glue = _glue()
    meter = MeterParams(integ_window_s=integ_window_s, sample_hz=sample_hz,
                        sigma_eta=glue.sigma_eta)
    positive = make_annotated_population("drift", 0.0, 1, glue, 3, meter=meter)[0]
    nulls = make_null_populations(1, glue, 5, meter=meter)
    assert set(nulls) == set(NULL_FAMILIES)
    for name, group in nulls.items():
        np.testing.assert_allclose(
            group[0].t, positive.t, atol=1e-9,
            err_msg=f"null {name!r} is not on the positive's grid",
        )
    assert np.isclose(1.0 / np.median(np.diff(positive.t)), sample_hz)


def test_dropping_a_null_family_does_not_shift_the_others():
    """Leave-one-null-out must not perturb the retained families' draws."""
    glue = _glue()
    full = make_null_populations(1, glue, 5)
    subset = make_null_populations(
        1, glue, 5, families=tuple(n for n in NULL_FAMILIES if n != "ar1"),
    )
    assert "ar1" not in subset
    for name, group in subset.items():
        np.testing.assert_array_equal(group[0].P_obs, full[name][0].P_obs)


def test_unknown_null_family_is_rejected():
    with pytest.raises(ValueError, match="unknown null families"):
        make_null_populations(1, _glue(), 5, families=("ar1", "not_a_family"))


# --- 2b. the E0 ceiling must not be scored on latent metadata ---------------
# Regression guard for the defect in notes/results/hsmm-e0-defect-and-fair-ceiling.md:
# E0 originally compared oracle-windowed positives against blind-extracted nulls,
# and several oracle features are read off the generator's schedule, so the
# ceiling saturated on metadata alone.

_METADATA_DERIVED = ("paired_fraction", "duration_median_s", "duration_cv")


def test_oracle_features_are_metadata_derived_hence_need_a_matched_null():
    """Why the null side must use the same windowing: these ignore the trace."""
    traces = make_annotated_population("drift", 0.0, 3, _glue(), 11)
    rng = np.random.default_rng(9)
    noise = [
        AnnotatedTrace(
            trace.t,
            rng.normal(float(np.mean(trace.P_obs)), float(np.std(trace.P_obs)),
                       trace.t.size),
            trace.meta,
        )
        for trace in traces
    ]
    real, scrambled = oracle_matrix(traces), oracle_matrix(noise)

    # paired_fraction is a hardcoded constant: identical whatever the trace holds.
    paired = FEATURE_NAMES.index("paired_fraction")
    np.testing.assert_array_equal(real[:, paired], scrambled[:, paired])
    np.testing.assert_allclose(scrambled[:, paired], 1.0)

    # The duration features are the *generator's* durations, filtered only by
    # which windows survive the amplitude check -- so on a signal-free trace they
    # still report true durations rather than anything measured from the data.
    column = FEATURE_NAMES.index("duration_median_s")
    for trace, value in zip(traces, scrambled[:, column]):
        true = trace.meta.communication_durations
        assert true.min() - 0.05 <= value <= true.max() + 0.05, (
            "duration_median_s on a noise trace is outside the true duration "
            "support, so it is not purely metadata-derived"
        )


def test_oracle_null_side_matches_the_positive_windowing_procedure():
    glue = _glue()
    positives = make_annotated_population("drift", 0.0, 4, glue, 11)
    nulls = make_null_populations(4, glue, 5)["inference"]
    counts, durations = alignment_pool(positives)
    fair = oracle_null_matrix(nulls, counts, durations, 3)
    blind = blind_matrix(nulls)
    paired = FEATURE_NAMES.index("paired_fraction")
    # The fair null side uses the oracle's own windowing, so it reproduces the
    # constant the positives get; blind extraction cannot.
    np.testing.assert_allclose(fair[:, paired], 1.0)
    assert np.all(blind[:, paired] < 1.0)


def test_oracle_null_alignment_is_deterministic_and_shaped_correctly():
    glue = _glue()
    positives = make_annotated_population("drift", 0.0, 3, glue, 11)
    nulls = make_null_populations(3, glue, 5)["ar1"]
    counts, durations = alignment_pool(positives)
    first = oracle_null_matrix(nulls, counts, durations, 17)
    second = oracle_null_matrix(nulls, counts, durations, 17)
    np.testing.assert_array_equal(first, second)
    assert first.shape == (3, len(FEATURE_NAMES))
    assert np.all(np.isfinite(first))


# --- 3. threshold machinery -------------------------------------------------

def test_threshold_at_far_respects_the_budget_and_is_most_permissive():
    scores = np.arange(100.0)
    threshold = threshold_at_far(scores, 0.05)
    assert (scores >= threshold).mean() <= 0.05
    below = np.nextafter(threshold, -np.inf)
    assert (scores >= below).mean() > 0.05


def test_threshold_at_far_handles_a_point_mass_larger_than_the_budget():
    """A perfectly separable discrete score must not yield +inf."""
    scores = np.zeros(100)
    threshold = threshold_at_far(scores, 0.05)
    assert np.isfinite(threshold)
    assert (scores >= threshold).mean() == 0.0


def test_composite_threshold_is_the_most_conservative_family():
    groups = {"a": np.arange(100.0), "b": np.arange(100.0) + 50.0}
    composite = composite_threshold(groups, 0.05)
    assert composite == max(threshold_at_far(g, 0.05) for g in groups.values())
    assert all((g >= composite).mean() <= 0.05 for g in groups.values())


def test_null_rates_counts_at_or_above_the_threshold():
    assert null_rates({"x": np.asarray([0.0, 1.0, 2.0, 3.0])}, 2.0) == {"x": 0.5}


# --- 4. profile width -------------------------------------------------------

def test_widen_profile_scales_covariance_isotropically():
    rng = np.random.default_rng(0)
    features = rng.normal(size=(50, len(FEATURE_NAMES)))
    profile = fit_gaussian_profile(features)
    widened = widen_profile(profile, 4.0)
    np.testing.assert_allclose(widened.precision, profile.precision / 4.0)
    np.testing.assert_allclose(
        widened.logdet, profile.logdet + len(FEATURE_NAMES) * np.log(4.0),
    )
    # A point far from the mean is relatively less improbable once widened.
    far = profile.mean + 5.0
    assert widened.logpdf(far) > profile.logpdf(far)


def test_widen_profile_rejects_a_non_positive_factor():
    profile = fit_gaussian_profile(
        np.random.default_rng(1).normal(size=(50, len(FEATURE_NAMES))),
    )
    with pytest.raises(ValueError, match="factor must be positive"):
        widen_profile(profile, 0.0)


# --- 5. timing diagnostics --------------------------------------------------

def test_timing_tolerance_is_configurable_and_monotone():
    traces = make_annotated_population("drift", 0.0, 2, _glue(), 11)
    tight = timing_metrics(traces, tolerance_s=0.01)["recall_mean"]
    loose = timing_metrics(traces, tolerance_s=0.5)["recall_mean"]
    assert loose >= tight


# --- 6. constant fit-set size across configurations -------------------------

def test_every_positive_fit_configuration_draws_the_same_total():
    for total in (250, 100, 7):
        for name, plan in generalization._positive_fit_configs(total).items():
            assert sum(plan.values()) == total, name


def test_even_split_is_balanced_to_within_one():
    counts = generalization._even_split(250, 3)
    assert sum(counts) == 250
    assert max(counts) - min(counts) <= 1


def test_held_out_conditions_are_the_complement_of_the_fit_plan():
    configs = generalization._positive_fit_configs(250)
    assert generalization._held_out_conditions(configs["honest_only"]) == [
        "work_0.5", "work_0.7", "drift_0.8", "drift_1.5",
    ]
    assert generalization._held_out_conditions(configs["pooled_baseline"]) == []


def test_stack_fit_features_takes_a_prefix_of_each_pool():
    cache = {"honest": np.arange(20.0).reshape(10, 2),
             "work_0.5": np.arange(100.0, 120.0).reshape(10, 2)}
    stacked = generalization._stack_fit_features(cache,
                                                 {"honest": 3, "work_0.5": 2})
    assert stacked.shape == (5, 2)
    np.testing.assert_array_equal(stacked[:3], cache["honest"][:3])
    np.testing.assert_array_equal(stacked[3:], cache["work_0.5"][:2])
