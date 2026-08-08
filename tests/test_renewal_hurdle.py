"""Unit checks for the hurdle likelihood and its missing-feature handling.

Two artefacts motivated this module, and both produced *better* numbers, which is
the direction that does not prompt suspicion.  These tests pin the fixes:

* "no events extracted" must be a point mass, not eleven continuous zeros;
* an under-determined statistic must be unavailable, not evidence -- the legacy
  path reports *perfect* shape repeatability from a single event and a
  coefficient of variation of zero from one observation.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import multivariate_normal

from powerladder.typeb.renewal import EventParams, extract_events
from powerladder.typeb.renewal_campaign import (
    glue_for,
    make_annotated_population,
    make_null_populations,
)
from powerladder.typeb.renewal_hurdle import (
    MARK_MIN_EVENTS,
    MARK_NAMES,
    CountModel,
    MarkModel,
    Observation,
    fit_count_model,
    fit_hurdle_likelihood,
    fit_hurdle_profile,
    fit_mark_model,
    mark_vector,
    observe,
    zero_event_rate,
)


def _events(n: int):
    """``n`` synthetic marked events with varying durations and shapes."""
    traces = make_annotated_population("drift", 0.0, 1, glue_for(60.0), 11)
    extracted = extract_events(traces[0].t, traces[0].P_obs)
    assert len(extracted.events) >= n, "fixture needs more events"
    return list(extracted.events[:n])


def _observation(n_events: int, marks: np.ndarray) -> Observation:
    return Observation(n_events=n_events, exposure_s=10.0, marks=marks)


# --- the count is modelled explicitly, not as a rate feature ----------------

def test_event_rate_is_not_a_mark():
    """Keeping a rate feature alongside the count term would double-count it."""
    assert "event_rate_hz" not in MARK_NAMES
    assert len(MARK_NAMES) == 10


# --- availability by event count -------------------------------------------

@pytest.mark.parametrize("n_events", [1, 2, 3, 5])
def test_marks_are_available_exactly_when_defined(n_events):
    marks = mark_vector(_events(n_events), paired_fraction=1.0)
    available = np.isfinite(marks)
    for j, (name, minimum) in enumerate(zip(MARK_NAMES, MARK_MIN_EVENTS)):
        assert available[j] == (n_events >= minimum), (
            f"{name} availability wrong at N={n_events}"
        )


def test_single_event_gives_no_variation_statistics():
    """The specific legacy artefact: CV 0.0 and repeatability 1.0 from one event."""
    marks = mark_vector(_events(1), paired_fraction=1.0)
    for name in ("duration_cv", "amplitude_cv", "shape_repeatability",
                 "interval_cv"):
        assert np.isnan(marks[MARK_NAMES.index(name)]), name
    # Medians are genuinely defined from one event.
    assert np.isfinite(marks[MARK_NAMES.index("duration_median_s")])


def test_two_events_give_repeatability_but_not_interval_variation():
    marks = mark_vector(_events(2), paired_fraction=1.0)
    assert np.isfinite(marks[MARK_NAMES.index("shape_repeatability")])
    assert np.isfinite(marks[MARK_NAMES.index("duration_cv")])
    assert np.isnan(marks[MARK_NAMES.index("interval_cv")])


def test_no_events_gives_all_unavailable_marks():
    marks = mark_vector([], paired_fraction=0.0)
    assert marks.shape == (len(MARK_NAMES),)
    assert np.all(np.isnan(marks))


# --- the hurdle: N = 0 is a point mass --------------------------------------

def test_zero_event_observation_scores_exactly_the_point_mass():
    observations = [_observation(0, np.full(len(MARK_NAMES), np.nan))
                    for _ in range(3)]
    observations += [_observation(5, np.zeros(len(MARK_NAMES)))
                     for _ in range(7)]
    profile = fit_hurdle_profile(observations)
    empty = _observation(0, np.full(len(MARK_NAMES), np.nan))
    assert profile.loglike(empty) == profile.log_p_zero
    # and it does not depend on the mark model at all
    assert profile.loglike(empty, counts_only=True) == profile.log_p_zero


def test_zero_probability_is_smoothed_away_from_zero_and_one():
    """A class that happened to produce no empty traces must not veto one."""
    never = fit_hurdle_profile([_observation(4, np.zeros(len(MARK_NAMES)))
                                for _ in range(50)])
    always = fit_hurdle_profile([_observation(0, np.full(len(MARK_NAMES), np.nan))
                                 for _ in range(50)])
    for profile in (never, always):
        assert np.isfinite(profile.log_p_zero)
        assert np.isfinite(profile.log_p_nonzero)
        assert -np.inf < profile.log_p_zero < 0.0
    assert never.zero_event_rate == 0.0
    assert always.zero_event_rate == 1.0


def test_nonzero_observation_uses_all_three_terms():
    observations = [_observation(0, np.full(len(MARK_NAMES), np.nan))
                    for _ in range(4)]
    rng = np.random.default_rng(0)
    observations += [_observation(6, rng.normal(size=len(MARK_NAMES)))
                     for _ in range(30)]
    profile = fit_hurdle_profile(observations)
    marks = np.zeros(len(MARK_NAMES))
    full = profile.loglike(_observation(6, marks))
    counts = profile.loglike(_observation(6, marks), counts_only=True)
    expected = (profile.log_p_nonzero + profile.counts.logpmf(6))
    assert counts == pytest.approx(expected)
    assert full == pytest.approx(expected + profile.marks.logpdf(marks))


# --- the count model --------------------------------------------------------

def test_zero_truncated_count_model_excludes_zero_and_normalises():
    model = fit_count_model(np.asarray([3, 4, 5, 6, 7, 4, 5, 9, 2, 5]), 10.0)
    assert model.logpmf(0) == -np.inf
    total = sum(np.exp(model.logpmf(n)) for n in range(1, 400))
    assert total == pytest.approx(1.0, abs=1e-3)


def test_count_model_falls_back_to_poisson_when_underdispersed():
    model = fit_count_model(np.full(20, 5.0), 10.0)
    assert model.kind == "poisson"
    assert np.isfinite(model.logpmf(5))


def test_count_model_uses_negative_binomial_when_overdispersed():
    counts = np.asarray([1, 2, 3, 20, 30, 1, 2, 40, 3, 25], dtype=float)
    model = fit_count_model(counts, 10.0)
    assert model.kind == "nbinom"


def test_count_model_ignores_zeros_when_fitting_the_truncated_part():
    with_zeros = fit_count_model(np.asarray([0, 0, 0, 4, 5, 6, 4, 5]), 10.0)
    without = fit_count_model(np.asarray([4, 5, 6, 4, 5]), 10.0)
    assert with_zeros.mean == pytest.approx(without.mean)


# --- marks are marginalised, never imputed ----------------------------------

def test_mark_logpdf_equals_the_true_gaussian_marginal():
    rng = np.random.default_rng(3)
    d = len(MARK_NAMES)
    a = rng.normal(size=(d, d))
    cov = a @ a.T + np.eye(d)
    mean = rng.normal(size=d)
    model = MarkModel(mean=mean, cov=cov)

    marks = np.full(d, np.nan)
    idx = [0, 2, 5]
    values = rng.normal(size=len(idx))
    marks[idx] = values
    expected = multivariate_normal(mean[idx], cov[np.ix_(idx, idx)]).logpdf(values)
    assert model.logpdf(marks) == pytest.approx(expected, rel=1e-9)


def test_unavailable_marks_do_not_shift_the_score_like_a_sentinel():
    """A missing mark must be marginalised, not scored as if it were zero."""
    rng = np.random.default_rng(4)
    d = len(MARK_NAMES)
    mean = np.full(d, 5.0)
    model = MarkModel(mean=mean, cov=np.eye(d))
    partial = np.full(d, np.nan)
    partial[0] = 5.0
    sentinel = np.zeros(d)
    sentinel[0] = 5.0
    # The sentinel puts every other coordinate 5 sigma from the mean; the
    # marginal ignores them entirely.
    assert model.logpdf(partial) > model.logpdf(sentinel)
    assert model.logpdf(partial) == pytest.approx(
        multivariate_normal(mean[:1], np.eye(1)).logpdf([5.0]), rel=1e-9)
    assert rng is not None


def test_all_unavailable_marks_contribute_nothing():
    model = MarkModel(np.zeros(len(MARK_NAMES)), np.eye(len(MARK_NAMES)))
    assert model.logpdf(np.full(len(MARK_NAMES), np.nan)) == 0.0


def test_mark_model_fits_each_column_from_its_available_rows():
    rows = np.full((30, len(MARK_NAMES)), np.nan)
    rows[:, 0] = 7.0
    rows[:10, 1] = 3.0
    model = fit_mark_model(rows)
    assert model.mean[0] == pytest.approx(7.0)
    assert model.mean[1] == pytest.approx(3.0)


def test_mark_model_rejects_a_wrong_width():
    with pytest.raises(ValueError, match="mark columns"):
        fit_mark_model(np.zeros((5, 3)))


# --- the composite score ----------------------------------------------------

def test_score_is_blind_and_composite():
    glue = glue_for(20.0)
    positives = make_annotated_population("drift", 0.0, 8, glue, 11)
    nulls = make_null_populations(4, glue, 21, families=("ar1", "inference"))
    model = fit_hurdle_likelihood(
        [observe(tr.t, tr.P_obs) for tr in positives],
        [[observe(tr.t, tr.P_obs) for tr in group] for group in nulls.values()],
    )
    assert len(model.nulls) == 2
    trace = positives[0]
    direct = model.score(trace.t, trace.P_obs)
    observation = observe(trace.t, trace.P_obs)
    expected = (model.training.loglike(observation)
                - max(p.loglike(observation) for p in model.nulls))
    assert direct == pytest.approx(expected)
    # Deterministic, and a function of (t, P_obs) alone.
    assert direct == pytest.approx(model.score(trace.t, trace.P_obs))


def test_fit_requires_at_least_one_null_group():
    observations = [_observation(3, np.zeros(len(MARK_NAMES))) for _ in range(4)]
    with pytest.raises(ValueError, match="at least one null group"):
        fit_hurdle_likelihood(observations, [])


def test_fit_profile_requires_observations():
    with pytest.raises(ValueError, match="at least one observation"):
        fit_hurdle_profile([])


# --- reporting replaces the degeneracy gate ---------------------------------

def test_zero_event_rate_reports_the_share_of_empty_traces():
    observations = ([_observation(0, np.full(len(MARK_NAMES), np.nan))] * 3
                    + [_observation(4, np.zeros(len(MARK_NAMES)))] * 7)
    assert zero_event_rate(observations) == pytest.approx(0.3)
    assert zero_event_rate([]) == 0.0


def test_observe_counts_events_and_records_exposure():
    trace = make_annotated_population("drift", 0.0, 1, glue_for(30.0), 11)[0]
    observation = observe(trace.t, trace.P_obs, params=EventParams())
    assert observation.n_events == len(extract_events(trace.t,
                                                      trace.P_obs).events)
    assert observation.exposure_s == pytest.approx(trace.t[-1] - trace.t[0])
    assert observation.available.shape == (len(MARK_NAMES),)


def test_count_model_dataclass_is_frozen():
    model = CountModel("poisson", 3.0, np.nan, np.nan, 10.0)
    with pytest.raises(Exception):
        model.mean = 4.0
