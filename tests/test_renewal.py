"""Fast deterministic tests for the E0/E1 duration-aware smoke detector."""

from __future__ import annotations

import inspect

import numpy as np

from powerladder.typeb.renewal import (
    FEATURE_NAMES,
    blind_trace_features,
    event_timing_diagnostics,
    extract_events,
    fit_likelihood_from_features,
)


def _irregular_dip_trace(seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    fs = 20.0
    t = np.arange(0.0, 30.0, 1.0 / fs)
    x = 250.0 + rng.normal(0.0, 0.4, size=t.size)
    starts = np.asarray([1.0, 2.4, 4.1, 5.0, 7.8, 9.2, 12.5, 14.0,
                         17.3, 18.2, 21.6, 24.0, 27.5])
    for j, start in enumerate(starts):
        dwell = 0.25 + 0.05 * (j % 3)
        x[(t >= start) & (t < start + dwell)] -= 12.0
    return t, x, starts


def test_blind_interface_cannot_accept_phase_metadata():
    sig = inspect.signature(blind_trace_features)
    assert tuple(sig.parameters) == ("t", "p_obs", "params")
    assert sig.parameters["params"].kind is inspect.Parameter.KEYWORD_ONLY


def test_blind_events_recover_irregular_local_dips_without_a_cadence():
    t, x, truth = _irregular_dip_trace()
    extracted = extract_events(t, x)
    recall, precision, median_error = event_timing_diagnostics(
        extracted.down_times, truth, tolerance_s=0.1,
    )
    assert recall >= 0.9
    assert precision >= 0.9
    assert median_error <= 0.05
    features = blind_trace_features(t, x)
    assert features.shape == (len(FEATURE_NAMES),)
    assert np.all(np.isfinite(features))


def test_profiled_likelihood_prefers_training_feature_cloud():
    rng = np.random.default_rng(4)
    d = len(FEATURE_NAMES)
    train = rng.normal(1.0, 0.15, size=(30, d))
    null_a = rng.normal(-1.0, 0.2, size=(30, d))
    null_b = rng.normal(3.0, 0.2, size=(30, d))
    model = fit_likelihood_from_features(train, [null_a, null_b])
    assert model.score_features(np.ones(d)) > 0.0
    assert model.score_features(-np.ones(d)) < 0.0


def test_timing_diagnostics_do_not_double_match_predictions():
    recall, precision, error = event_timing_diagnostics(
        np.asarray([1.01]), np.asarray([1.0, 1.04]), tolerance_s=0.1,
    )
    assert recall == 0.5
    assert precision == 1.0
    assert error <= 0.01 + 1e-12
