"""ST1 DG-statistic tests (task 19.3): chi2(2L) null behaviour on white noise,
firing at the true alpha only, LTI-filter invariance, estimator agreement, and
the counted (not silent) covariance-conditioning fallback."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from scipy import signal as sp_signal
from scipy.stats import chi2, kstest

from powerladder.config import DEFAULT
from powerladder.st1.cyclo import (cov_failure_count, cyclic_corr, dg_q, dg_q_max,
                            reset_cov_failures)
from powerladder.st1.pipeline import ST1_DETECTORS, stage1_fixed_alpha

FS = 20.0
ALPHA = 1.0        # true cycle frequency used by the AM signals [Hz]


@pytest.fixture
def params():
    return DEFAULT.st1


def _am_noise(n, fs, alpha_hz, rng, depth=0.5):
    """Amplitude-modulated white noise: variance cycles at alpha_hz."""
    t = np.arange(n) / fs
    return (1.0 + depth * np.cos(2 * np.pi * alpha_hz * t)) * rng.normal(0, 1, n)


def test_q_on_white_noise_has_chi2_moments_and_ks(params):
    rng = np.random.default_rng(0)
    dof = 2 * len(params.lag_set)
    n_mc, n = 400, 4000
    qs = np.array([
        dg_q(rng.normal(0, 1, n), FS, ALPHA, params.lag_set, params=params).Q
        for _ in range(n_mc)
    ])
    assert abs(qs.mean() - dof) < 0.15 * dof, f"mean {qs.mean():.2f} vs dof {dof}"
    assert 0.5 * 2 * dof < qs.var() < 2.0 * 2 * dof, f"var {qs.var():.2f} vs {2*dof}"
    res = kstest(qs, lambda q: chi2.cdf(q, dof))
    assert res.pvalue > 1e-3, f"KS p = {res.pvalue:.2e}"


def test_q_fires_at_true_alpha_not_mismatched(params):
    rng = np.random.default_rng(1)
    x = _am_noise(6001, FS, ALPHA, rng)
    dof = 2 * len(params.lag_set)
    hit = dg_q(x, FS, ALPHA, params.lag_set, params=params)
    miss = dg_q(x, FS, 0.61, params.lag_set, params=params)   # not harmonically related
    assert hit.Q > chi2.isf(1e-8, dof), f"Q at true alpha only {hit.Q:.1f}"
    assert miss.Q < chi2.isf(1e-3, dof), f"Q at mismatched alpha {miss.Q:.1f}"


def test_q_survives_lti_filter(params):
    """An unknown LTI transfer function must not erase the cyclic signature."""
    rng = np.random.default_rng(2)
    x = _am_noise(6001, FS, ALPHA, rng)
    b, a = sp_signal.butter(4, 3.0 / (FS / 2))
    y = sp_signal.lfilter(b, a, x)
    dof = 2 * len(params.lag_set)
    res = dg_q(y, FS, ALPHA, params.lag_set, params=params)
    assert res.Q > chi2.isf(1e-8, dof), f"Q after LTI filter only {res.Q:.1f}"


def test_cov_estimators_roughly_agree_on_white_noise(params):
    rng = np.random.default_rng(3)
    p_bart = params
    p_batch = dataclasses.replace(params, cov_estimator="batch_means")
    q_bart, q_batch = [], []
    for _ in range(60):
        x = rng.normal(0, 1, 4000)
        q_bart.append(dg_q(x, FS, ALPHA, params.lag_set, params=p_bart).Q)
        q_batch.append(dg_q(x, FS, ALPHA, params.lag_set, params=p_batch).Q)
    ratio = np.mean(q_batch) / np.mean(q_bart)
    assert 0.7 < ratio < 1.4, f"estimator mean-Q ratio {ratio:.2f}"


def test_conditioning_failure_is_counted_not_silent(params):
    """A degenerate (constant) short trace must trip the fallback and be counted."""
    reset_cov_failures()
    x = np.ones(64)
    res = dg_q(x, FS, ALPHA, params.lag_set, params=params)
    assert res.cov_failed
    assert cov_failure_count() == 1
    assert np.isfinite(res.Q)
    reset_cov_failures()
    assert cov_failure_count() == 0


def test_unknown_estimator_raises(params):
    bad = dataclasses.replace(params, cov_estimator="bogus")
    with pytest.raises(ValueError):
        dg_q(np.random.default_rng(0).normal(0, 1, 500), FS, ALPHA,
             params.lag_set, params=bad)


def test_cyclic_corr_shape_and_zero_mean_under_null(params):
    rng = np.random.default_rng(4)
    r = cyclic_corr(rng.normal(0, 1, 20000), FS, ALPHA, params.lag_set)
    assert r.shape == (len(params.lag_set),)
    assert np.all(np.abs(r) < 0.05)


def test_dg_q_max_bonferroni_and_alpha_recovery(params):
    rng = np.random.default_rng(5)
    x = _am_noise(6001, FS, ALPHA, rng)
    alphas = np.array([0.5, 0.75, ALPHA, 1.25])
    res = dg_q_max(x, FS, alphas, params.lag_set, params=params)
    assert res.alpha_best == ALPHA
    assert res.n_alphas == 4
    assert res.p_corr <= 1.0


def test_stage1_detector_signature_and_registry(params):
    """dg_fixed obeys the gate signature and separates AM signal from noise."""
    rng = np.random.default_rng(6)
    n = 6001
    t = np.arange(n) / FS
    x_null = rng.normal(0, 1, n)
    x_sig = _am_noise(n, FS, ALPHA, rng)
    assert set(ST1_DETECTORS) == {"mtf", "mtf_comb", "dg_fixed",
                                  "dg_order_split", "dg_order_full"}
    s_null = ST1_DETECTORS["dg_fixed"](t, x_null, 0.3, 1.7)
    s_sig = ST1_DETECTORS["dg_fixed"](t, x_sig, 0.3, 1.7)
    assert s_sig > 8.0 > s_null
    # explicit-alpha call matches the registry default (alpha = Ko nominal 1.0 Hz)
    s_direct = stage1_fixed_alpha(t, x_sig, 0.3, 1.7, params=params, alpha_hz=ALPHA)
    assert np.isclose(s_direct, s_sig)
