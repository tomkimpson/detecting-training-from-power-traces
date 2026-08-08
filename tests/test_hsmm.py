"""Unit checks for the E2 explicit-duration HSMM and its controls.

The claim this code exists to test is narrow -- that *explicit durations* buy
something over duration-neutral sequence models -- so the properties worth
pinning are the ones that would let a gain be misattributed:

* the three variants must differ **only** in the duration law;
* the statistic must be the forward marginal over segmentations, not a best path;
* the duration laws must be what they claim (geometric matched on mean; shuffled
  preserving entropy and support);
* scoring must consume ``(t, P_obs)`` alone.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from scipy.special import logsumexp

from powerladder.typeb.hsmm import (
    COMMUNICATION,
    COMPUTE,
    DURATION_KINDS,
    EXPLICIT,
    GEOMETRIC,
    SHUFFLED,
    Emissions,
    HSMMComposite,
    TwoStateHSMM,
    blind_segmentation,
    duration_law,
    fit_composite,
    fit_hsmm,
    latent_segmentation,
    observation_matrix,
)
from powerladder.typeb.renewal_campaign import (
    glue_for,
    make_annotated_population,
    make_diluted_population,
    make_null_populations,
)


def _traces(n=6, duration_s=20.0, seed=11):
    return make_annotated_population("drift", 0.0, n, glue_for(duration_s), seed)


def _segmentations(traces):
    return [blind_segmentation(t.t, t.P_obs) for t in traces]


# --- observations -----------------------------------------------------------

def test_observation_matrix_is_two_channels_from_t_and_power_only():
    trace = _traces(1)[0]
    obs = observation_matrix(trace.t, trace.P_obs)
    assert obs.shape == (2, trace.t.size)
    assert np.all(np.isfinite(obs))
    # A pure time/power function: identical inputs give identical outputs.
    np.testing.assert_array_equal(obs, observation_matrix(trace.t, trace.P_obs))


# --- duration laws ----------------------------------------------------------

def test_duration_laws_are_normalised_over_their_support():
    durations = np.asarray([2, 3, 3, 4, 5, 3, 7, 2])
    for kind in DURATION_KINDS:
        law = duration_law(durations, 12, kind, np.random.default_rng(0))
        assert law.support == 12
        assert np.exp(logsumexp(law.logpmf)) == pytest.approx(1.0)


def test_survival_is_the_upper_tail_of_the_pmf():
    law = duration_law(np.asarray([1, 2, 3, 4]), 6, EXPLICIT)
    pmf = np.exp(law.logpmf)
    surv = np.exp(law.logsurv)
    assert surv[0] == pytest.approx(1.0)
    for d in range(6):
        assert surv[d] == pytest.approx(pmf[d:].sum())


def test_geometric_law_matches_the_explicit_law_mean():
    rng = np.random.default_rng(0)
    durations = rng.integers(2, 9, size=400)
    support = 40
    explicit = duration_law(durations, support, EXPLICIT)
    geometric = duration_law(durations, support, GEOMETRIC)
    d = np.arange(1, support + 1)
    mean_explicit = float((np.exp(explicit.logpmf) * d).sum())
    mean_geometric = float((np.exp(geometric.logpmf) * d).sum())
    assert mean_geometric == pytest.approx(mean_explicit, rel=0.15)


def test_geometric_law_is_actually_geometric():
    """Its log-pmf must be linear in d, which the explicit law's is not."""
    durations = np.asarray([3, 4, 5, 6, 4, 5, 5, 4, 12])
    geometric = duration_law(durations, 30, GEOMETRIC)
    steps = np.diff(geometric.logpmf)
    np.testing.assert_allclose(steps, steps[0], rtol=1e-9)


def test_shuffled_law_preserves_entropy_and_support_but_not_order():
    durations = np.asarray([3, 4, 5, 6, 4, 5, 5, 4, 12])
    support = 30
    explicit = duration_law(durations, support, EXPLICIT)
    shuffled = duration_law(durations, support, SHUFFLED,
                            np.random.default_rng(1))
    assert shuffled.support == explicit.support
    np.testing.assert_allclose(np.sort(shuffled.logpmf),
                               np.sort(explicit.logpmf))
    assert not np.allclose(shuffled.logpmf, explicit.logpmf)


def test_unknown_duration_kind_is_rejected():
    with pytest.raises(ValueError, match="unknown duration kind"):
        duration_law(np.asarray([1, 2]), 4, "nonsense")


# --- the forward recursion --------------------------------------------------

def _uniform_model(support=3, n_channels_mean=0.0):
    emissions = Emissions(mean=np.full((2, 2), n_channels_mean),
                          variance=np.ones((2, 2)))
    laws = tuple(duration_law(np.asarray([1, 2, 3]), support, EXPLICIT)
                 for _ in range(2))
    return TwoStateHSMM(emissions=emissions, durations=laws,
                        log_initial=np.log(np.full(2, 0.5)))


def test_forward_marginal_exceeds_any_single_segmentation():
    """It marginalises over segmentations, so it must beat the best single path."""
    model = _uniform_model()
    rng = np.random.default_rng(0)
    observations = rng.normal(size=(2, 12))
    total = model.loglike(observations)

    # One admissible segmentation, scored by hand: alternate fixed dwells.
    logb = model.emissions.loglike(observations)
    best = -np.inf
    for start_state in (COMPUTE, COMMUNICATION):
        for dwell in (1, 2, 3):
            state, t, path = start_state, 0, 0.0
            path += model.log_initial[start_state]
            while t < observations.shape[1]:
                length = min(dwell, observations.shape[1] - t)
                law = model.durations[state]
                term = (law.logsurv[length - 1]
                        if t + length == observations.shape[1]
                        else law.logpmf[length - 1])
                path += term + logb[state, t:t + length].sum()
                t += length
                state = 1 - state
            best = max(best, path)
    assert total > best


def test_forward_likelihood_is_finite_and_deterministic():
    model = _uniform_model()
    observations = np.random.default_rng(1).normal(size=(2, 25))
    first = model.loglike(observations)
    assert np.isfinite(first)
    assert first == model.loglike(observations)


def test_a_model_prefers_data_from_its_own_emissions():
    high = TwoStateHSMM(
        emissions=Emissions(mean=np.array([[5.0, 0.0], [5.0, 0.0]]),
                            variance=np.ones((2, 2))),
        durations=_uniform_model().durations,
        log_initial=np.log(np.full(2, 0.5)))
    low = _uniform_model()
    observations = np.vstack([np.full(20, 5.0), np.zeros(20)])
    assert high.loglike(observations) > low.loglike(observations)


# --- segmentation -----------------------------------------------------------

def test_blind_segmentation_uses_no_metadata_and_alternates():
    trace = _traces(1)[0]
    segmentation = blind_segmentation(trace.t, trace.P_obs)
    assert len(segmentation.samples) == 2
    assert len(segmentation.durations) == 2
    total = sum(sum(d) for d in segmentation.durations)
    assert total == trace.t.size
    assert all(d > 0 for state in segmentation.durations for d in state)


def test_latent_segmentation_covers_the_record_and_uses_metadata():
    trace = _traces(1)[0]
    latent = latent_segmentation(trace)
    assert sum(sum(d) for d in latent.durations) == trace.t.size
    # It reads the metadata, so shifting the boundaries must move the result.
    shifted = dataclasses.replace(
        trace, meta=dataclasses.replace(
            trace.meta,
            communication_starts=trace.meta.communication_starts + 0.7))
    assert latent_segmentation(shifted).durations != latent.durations


def test_blind_and_latent_segmentations_diverge_where_extraction_is_hard():
    """On easy traces they coincide; the E2 target cell is where they do not."""
    trace = make_diluted_population(0.2, 1, glue_for(10.0), 7)[0]
    blind = blind_segmentation(trace.t, trace.P_obs)
    latent = latent_segmentation(trace)
    assert blind.durations != latent.durations


# --- fitting ----------------------------------------------------------------

def test_variants_differ_only_in_the_duration_law():
    segmentations = _segmentations(_traces(8))
    models = {kind: fit_hsmm(segmentations, kind=kind, support=30,
                             rng=np.random.default_rng(0))
              for kind in DURATION_KINDS}
    reference = models[EXPLICIT]
    for kind, model in models.items():
        np.testing.assert_array_equal(model.emissions.mean,
                                      reference.emissions.mean)
        np.testing.assert_array_equal(model.emissions.variance,
                                      reference.emissions.variance)
        np.testing.assert_array_equal(model.log_initial, reference.log_initial)
        if kind != EXPLICIT:
            assert not np.allclose(model.durations[0].logpmf,
                                   reference.durations[0].logpmf)


def test_fitted_support_cannot_outlast_the_record():
    traces = _traces(6, duration_s=10.0)
    segmentations = _segmentations(traces)
    model = fit_hsmm(segmentations, kind=EXPLICIT)
    assert model.durations[0].support <= traces[0].t.size


def test_composite_shares_one_support_across_classes():
    glue = glue_for(20.0)
    positives = _segmentations(_traces(6))
    nulls = [[blind_segmentation(t.t, t.P_obs)
              for t in make_null_populations(4, glue, 21,
                                             families=("ar1",))["ar1"]]]
    composite = fit_composite(positives, nulls, kind=EXPLICIT, seed=0)
    supports = {composite.training.durations[0].support,
                composite.training.durations[1].support}
    for model in composite.nulls:
        supports |= {model.durations[0].support, model.durations[1].support}
    assert len(supports) == 1, "duration kinds must be strictly comparable"


def test_composite_requires_a_null_group():
    with pytest.raises(ValueError, match="at least one null group"):
        fit_composite(_segmentations(_traces(4)), [], kind=EXPLICIT)


# --- scoring ----------------------------------------------------------------

def test_score_consumes_only_time_and_power():
    glue = glue_for(20.0)
    positives = _traces(6)
    nulls = make_null_populations(4, glue, 21, families=("ar1",))
    composite = fit_composite(
        _segmentations(positives),
        [[blind_segmentation(t.t, t.P_obs) for t in nulls["ar1"]]],
        kind=EXPLICIT, seed=0)
    trace = positives[0]
    score = composite.score(trace.t, trace.P_obs)
    assert np.isfinite(score)
    assert score == composite.score(trace.t, trace.P_obs)
    expected = (composite.training.loglike(
        observation_matrix(trace.t, trace.P_obs))
        - max(m.loglike(observation_matrix(trace.t, trace.P_obs))
              for m in composite.nulls))
    assert score == pytest.approx(expected)


def test_composite_takes_the_strongest_null():
    model = _uniform_model()
    strong = TwoStateHSMM(
        emissions=Emissions(mean=np.zeros((2, 2)), variance=np.ones((2, 2))),
        durations=model.durations, log_initial=model.log_initial)
    composite = HSMMComposite(training=model, nulls=(model, strong))
    observations = np.random.default_rng(2).normal(size=(2, 15))
    best = max(m.loglike(observations) for m in composite.nulls)
    assert (model.loglike(observations) - best) == pytest.approx(
        composite.training.loglike(observations) - best)
