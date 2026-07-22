"""Sequential-detection machinery (6.4; CPU/numpy, no GPU).

Guards the four claims the sequential verifier rests on:
  1. block scoring is a clean partition (count, length, per-block detrend);
  2. conformal p-values are valid (superuniform) on exchangeable blocks;
  3. the e-process respects Ville under the null and detects training fast;
  4. the Wald SPRT helpers implement Wald's boundaries and error rates.

Synthetic traces come from the deterministic Ko generator so every rate
checked here is a fixed-seed number, not a flake.
"""

from __future__ import annotations

import numpy as np
import pytest

from powerladder.config import DEFAULT
from powerladder.typeb.detectors import spectral_statistic, viterbi_statistic
from powerladder.typeb.ko_synth import ko_make_population
from powerladder.typeb.sequential import (SprtModel, block_scores, chain_blocks,
                                   conformal_p, detection_curve, e_calibrator,
                                   first_crossing, fit_sprt, sprt_decision,
                                   sprt_llr, time_to_detection, wealth_process)

GLUE = DEFAULT.ko_typeb
KO = DEFAULT.ko
BLOCK_S = 60.0


@pytest.fixture(scope="module")
def populations():
    """(train, infer) Ko populations, one rng, module-cached (they're slow)."""
    rng = np.random.default_rng(20260704)
    return ko_make_population(12, KO, GLUE, rng)


def _blocks(traces, detector, band_lo=None, band_hi=None):
    lo = GLUE.band_lo if band_lo is None else band_lo
    hi = GLUE.band_hi if band_hi is None else band_hi
    return [block_scores(tr.t, tr.P_obs, lo, hi, detector, block_s=BLOCK_S)
            for tr in traces]


# --- block scoring ----------------------------------------------------------

def test_block_scores_partition_count_and_independence(populations):
    train, _ = populations
    tr = train[0]
    fs = 1.0 / (tr.t[1] - tr.t[0])
    scores = block_scores(tr.t, tr.P_obs, GLUE.band_lo, GLUE.band_hi,
                          spectral_statistic, block_s=BLOCK_S)
    assert scores.size == int(tr.t.size // round(BLOCK_S * fs))
    # per-block scoring: the first block's score must not depend on the rest
    n_blk = int(round(BLOCK_S * fs))
    solo = spectral_statistic(tr.t[:n_blk], tr.P_obs[:n_blk],
                              GLUE.band_lo, GLUE.band_hi)
    assert scores[0] == pytest.approx(solo)


# --- conformal calibration --------------------------------------------------

def test_conformal_p_is_superuniform_on_exchangeable_blocks(populations):
    _, infer = populations
    null_blocks = np.concatenate(_blocks(infer, spectral_statistic))
    # leave-one-out: each null block scored against the others
    p = np.array([
        conformal_p(null_blocks[i], np.delete(null_blocks, i))[0]
        for i in range(null_blocks.size)
    ])
    for a in (0.05, 0.1, 0.25):
        assert (p <= a).mean() <= a + 2.0 / null_blocks.size
    # resolution floor: never below 1/(n+1)
    assert p.min() >= 1.0 / null_blocks.size


def test_conformal_p_orders_by_score():
    null = np.arange(100, dtype=float)
    p = conformal_p(np.array([150.0, 50.0, -10.0]), null)
    assert p[0] < p[1] < p[2]
    assert p[0] == pytest.approx(1.0 / 101.0)
    assert p[2] == pytest.approx(1.0)


# --- e-process --------------------------------------------------------------

def test_e_calibrator_is_a_valid_bet():
    rng = np.random.default_rng(0)
    u = rng.uniform(size=100_000)
    e = e_calibrator(u, kappa=0.5)
    assert e.mean() == pytest.approx(1.0, abs=0.02)  # E[e] <= 1 under the null
    with pytest.raises(ValueError):
        e_calibrator(u, kappa=1.5)


def test_wealth_respects_ville_under_null(populations):
    _, infer = populations
    per_trace = _blocks(infer, spectral_statistic)
    null_blocks = np.concatenate(per_trace)
    rng = np.random.default_rng(1)
    alpha, horizon = 0.05, 30
    false_alarms = 0
    n_chains = 200
    for _ in range(n_chains):
        chain = chain_blocks(per_trace, horizon, rng)
        wealth = wealth_process(conformal_p(chain, null_blocks))
        if first_crossing(wealth, alpha) is not None:
            false_alarms += 1
    # anytime false-alarm <= alpha (finite-null slack: chains reuse blocks)
    assert false_alarms / n_chains <= alpha + 0.03


def test_wealth_detects_training_within_one_trace(populations):
    # the headline pairing: e-process over *Viterbi* block scores. The
    # conformal floor (1/61 here) makes 3 near-perfect blocks the minimum
    # path to 1/alpha=20, so detection inside one 300 s trace is the bar.
    # (Spectral block scores are too noisy at 60 s for this — measured
    # per-level power is the analysis script's business, not this test's.)
    train, infer = populations
    null_blocks = np.concatenate(_blocks(infer, viterbi_statistic))
    detections = 0
    for blocks in _blocks(train, viterbi_statistic):
        wealth = wealth_process(conformal_p(blocks, null_blocks))
        if first_crossing(wealth, 0.05) is not None:
            detections += 1
    assert detections >= len(train) - 1  # nominal SNR: essentially always


def test_time_to_detection_full_band_beats_line_band(populations):
    train, infer = populations
    rng = np.random.default_rng(2)
    kw = dict(alpha=0.05, kappa=0.5, block_s=BLOCK_S, horizon_blocks=30,
              n_boot=50, rng=rng)
    line_lo = 0.55
    full = time_to_detection(
        _blocks(train, viterbi_statistic),
        np.concatenate(_blocks(infer, viterbi_statistic)), **kw)
    line = time_to_detection(
        _blocks(train, viterbi_statistic, band_lo=line_lo),
        np.concatenate(_blocks(infer, viterbi_statistic, band_lo=line_lo)),
        **kw)
    assert full["detect_frac"] >= line["detect_frac"]
    assert full["median_s"] <= line["median_s"]
    assert full["q25_s"] <= full["median_s"] <= full["q75_s"]


def test_detection_curve_is_monotone_bounded_and_consistent(populations):
    # the deployment-latency (9.4) primitive: P(detect by t). It must be a
    # valid CDF over observation time, and agree with time_to_detection's
    # detect_frac chain-for-chain (shared _crossing_blocks helper).
    train, infer = populations
    null = np.concatenate(_blocks(infer, viterbi_statistic))
    per_trace = _blocks(train, viterbi_statistic)
    kw = dict(alpha=0.05, kappa=0.5, block_s=BLOCK_S, horizon_blocks=30,
              n_boot=50)
    curve = detection_curve(per_trace, null, rng=np.random.default_rng(7),
                            **kw)
    p = curve["p_detect"]
    assert p.shape == (30,)
    assert np.all(p >= 0.0) and np.all(p <= 1.0)
    assert np.all(np.diff(p) >= 0.0)            # non-decreasing CDF
    np.testing.assert_allclose(curve["t_s"], BLOCK_S * np.arange(1, 31))
    assert curve["detect_frac"] == pytest.approx(p[-1])
    # identical rng + n_boot -> identical bootstrap chains -> same detect_frac
    ttd = time_to_detection(per_trace, null, rng=np.random.default_rng(7),
                            **kw)
    assert curve["detect_frac"] == pytest.approx(ttd["detect_frac"])


# --- SPRT -------------------------------------------------------------------

def test_sprt_boundaries_and_fast_decision_on_separated_gaussians():
    m = SprtModel(mu0=0.0, s0=1.0, mu1=5.0, s1=1.0)
    rng = np.random.default_rng(3)
    # H1 stream: decides H1, quickly
    k1, d1 = sprt_decision(sprt_llr(rng.normal(5.0, 1.0, 50), m), 0.05, 0.05)
    assert d1 == +1 and k1 <= 3
    # H0 stream: decides H0
    k0, d0 = sprt_decision(sprt_llr(rng.normal(0.0, 1.0, 50), m), 0.05, 0.05)
    assert d0 == -1
    # undecidable stream: stays between the boundaries
    assert sprt_decision(np.zeros(10), 0.05, 0.05) == (None, 0)


def test_sprt_error_rates_near_nominal():
    m = SprtModel(mu0=0.0, s0=1.0, mu1=1.0, s1=1.0)
    rng = np.random.default_rng(4)
    alpha = beta = 0.05
    n_trials, wrong_h0, wrong_h1 = 400, 0, 0
    for _ in range(n_trials):
        _, d = sprt_decision(sprt_llr(rng.normal(0.0, 1.0, 400), m),
                             alpha, beta)
        wrong_h0 += d == +1
        _, d = sprt_decision(sprt_llr(rng.normal(1.0, 1.0, 400), m),
                             alpha, beta)
        wrong_h1 += d == -1
    assert wrong_h0 / n_trials <= alpha + 0.02   # Wald: ~alpha false alarms
    assert wrong_h1 / n_trials <= beta + 0.02    # ~beta misses


def test_fit_sprt_recovers_the_moments():
    rng = np.random.default_rng(5)
    m = fit_sprt(rng.normal(0.0, 1.0, 20_000), rng.normal(3.0, 2.0, 20_000))
    assert m.mu0 == pytest.approx(0.0, abs=0.05)
    assert m.s0 == pytest.approx(1.0, abs=0.05)
    assert m.mu1 == pytest.approx(3.0, abs=0.05)
    assert m.s1 == pytest.approx(2.0, abs=0.05)


# --- chaining ---------------------------------------------------------------

def test_chain_blocks_is_deterministic_ordered_and_bounded():
    per_trace = [np.array([1.0, 2.0, 3.0]), np.array([10.0, 20.0, 30.0])]
    c1 = chain_blocks(per_trace, 7, np.random.default_rng(6))
    c2 = chain_blocks(per_trace, 7, np.random.default_rng(6))
    np.testing.assert_array_equal(c1, c2)          # deterministic given rng
    assert c1.size == 7                            # truncated to horizon
    # within-trace order preserved: every aligned triple is one of the traces
    for k in range(0, 6, 3):
        assert list(c1[k:k + 3]) in ([1.0, 2.0, 3.0], [10.0, 20.0, 30.0])
    with pytest.raises(ValueError):
        chain_blocks([np.zeros(0)], 5, np.random.default_rng(0))
