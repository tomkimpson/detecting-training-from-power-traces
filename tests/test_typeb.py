"""B0 Type-detector tests (CPU/numpy; no GPU).

Guards the three things the decision gate rests on:
  1. the generator stamps a line in the f0 band for training and not for inference;
  2. both detectors rank training above inference, and self-normalise sanely;
  3. the gate's headline ordering holds — stationary: tie; wandering: Viterbi wins;
  4. the ROC helpers are correct on hand-built scores.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import signal as sp_signal

import dataclasses

from powerladder.config import DEFAULT
from powerladder.forward import make_time_grid
from powerladder.ko_workload import aggregate_F, aggregate_null_F, inference_F, training_F
from powerladder.typeb.detectors import (forward_path_scores, forward_statistic,
                                  spectral_statistic, viterbi_best_path,
                                  viterbi_path_scores, viterbi_statistic)
from powerladder.typeb.gate import evaluate, score_population
from powerladder.typeb.roc import auc, roc_points, tpr_at_far
from powerladder.typeb.ko_synth import (
    ko_make_aggregate_population,
    ko_make_aggregate_trace,
    ko_make_population,
    ko_make_trace,
)
from powerladder.typeb.synth import make_population, make_trace


@pytest.fixture
def p():
    return DEFAULT.typeb


@pytest.fixture
def ko_p():
    return DEFAULT.ko


@pytest.fixture
def glue():
    return DEFAULT.ko_typeb


@pytest.fixture
def rng():
    return np.random.default_rng(0)


# --- generator -------------------------------------------------------------

def _band_peak_freq(tr, p):
    x = tr.P_obs - tr.P_obs.mean()
    f, psd = sp_signal.welch(x, fs=p.fs, nperseg=min(x.size, 1024))
    band = (f >= p.band_lo) & (f <= p.band_hi)
    return f[band][np.argmax(psd[band])]


def test_training_line_sits_at_its_f0(p, rng):
    tr = make_trace("train", p, rng, wander_hz=0.0)
    assert p.f0_train_lo <= tr.f0 <= p.f0_train_hi
    # the PSD peak in the band lands near the trace's own f0
    assert abs(_band_peak_freq(tr, p) - tr.f0) < 0.1


def test_inference_has_no_concentrated_band_line(p, rng):
    train = make_trace("train", p, rng, wander_hz=0.0)
    infer = make_trace("infer", p, rng, wander_hz=0.0)
    # training's in-band peak towers over its background; inference's does not
    assert spectral_statistic(train.t, train.P_obs, p.band_lo, p.band_hi) > \
        5 * spectral_statistic(infer.t, infer.P_obs, p.band_lo, p.band_hi)


def test_make_population_shapes_and_labels(p, rng):
    train, infer = make_population(5, p, rng)
    assert len(train) == 5 and len(infer) == 5
    assert all(t.label == "train" for t in train)
    assert all(t.label == "infer" for t in infer)


def test_unknown_label_raises(p, rng):
    with pytest.raises(ValueError):
        make_trace("bogus", p, rng)


# --- detectors -------------------------------------------------------------

@pytest.mark.parametrize("detector",
                         [spectral_statistic, viterbi_statistic, forward_statistic])
def test_detector_ranks_training_above_inference(detector, p, rng):
    train, infer = make_population(30, p, rng, wander_hz=0.0)
    pos = score_population(train, detector, p)
    neg = score_population(infer, detector, p)
    assert pos.mean() > neg.mean()
    assert auc(pos, neg) > 0.9


def test_detectors_fire_on_ko_training(p, ko_p, rng):
    """Generator-agnostic claim: the detectors fire on a real Ko training trace.

    The B0 detectors consume only (t, P_obs), so the shared Ko et al. generator
    (Task 1.1) drives them with no change — feed Ko training_F through the forward
    model P = r*F + P0. Guards against the migration to ko_workload (B1/Task 3)
    silently breaking. Ko has no inference workload, so the null is the synth one.
    """
    dt = 1.0 / p.fs
    t = make_time_grid(p.duration_s, dt)
    r, P0, f_peak = 1.0e-12, 250.0, DEFAULT.floor.F_max * 0.85
    F = training_F(t, ko_p, rng, f_peak=f_peak, eta_scale=0.15)
    P = r * F + P0 + rng.normal(0.0, p.sigma_eta, size=t.size)
    null = make_trace("infer", p, rng, wander_hz=0.0)
    assert spectral_statistic(t, P, p.band_lo, p.band_hi) > \
        5 * spectral_statistic(null.t, null.P_obs, p.band_lo, p.band_hi)
    assert viterbi_statistic(t, P, p.band_lo, p.band_hi) > \
        viterbi_statistic(null.t, null.P_obs, p.band_lo, p.band_hi)


# --- forward (sum-over-paths) statistic: PROTOTYPE --------------------------
# notes/discussion/track-before-detect-forward-statistic.md. Not in any frozen
# result; these tests pin the contract, not a bake-off verdict.

def test_forward_path_scores_last_equals_statistic(p, rng):
    tr = make_trace("train", p, rng, wander_hz=0.3)
    scores = forward_path_scores(tr.t, tr.P_obs, p.band_lo, p.band_hi)
    assert scores.size > 1
    assert forward_statistic(tr.t, tr.P_obs, p.band_lo, p.band_hi) == scores[-1]


def test_forward_empty_band_is_zero(p, rng):
    """A band beyond Nyquist has no bins: empty sequential array, scalar 0.0."""
    tr = make_trace("infer", p, rng)
    assert forward_path_scores(tr.t, tr.P_obs, 50.0, 60.0).size == 0
    assert forward_statistic(tr.t, tr.P_obs, 50.0, 60.0) == 0.0


def test_forward_holds_under_wander(p, rng):
    """The regime the sum-over-paths statistic exists for: a wandering line.

    Under heavy wander the fixed-bin matched filter smears out; the forward
    statistic, like Viterbi, integrates along wander paths and keeps separating.
    """
    train, infer = make_population(30, p, rng, wander_hz=0.5)
    pos = score_population(train, forward_statistic, p)
    neg = score_population(infer, forward_statistic, p)
    fwd_auc = auc(pos, neg)
    spec_auc = auc(score_population(train, spectral_statistic, p),
                   score_population(infer, spectral_statistic, p))
    assert fwd_auc > 0.9
    assert fwd_auc > spec_auc


def test_forward_gains_over_viterbi_at_low_snr_heavy_wander(p, rng):
    """The regime the sum-over-paths upgrade is FOR: weak line, heavy wander.

    Per-frame SNR too low for one path to dominate -> the MAP path undersells
    the evidence that many near-optimal paths carry jointly. Robust across
    seeds at n_each=100 (AUC gap 0.06-0.12); pinned here at the fixture seed.
    """
    weak = dataclasses.replace(p, amp_train=8.0)
    train, infer = make_population(40, weak, rng, wander_hz=0.6)
    fwd_auc = auc(score_population(train, forward_statistic, weak),
                  score_population(infer, forward_statistic, weak))
    vit_auc = auc(score_population(train, viterbi_statistic, weak),
                  score_population(infer, viterbi_statistic, weak))
    assert fwd_auc > vit_auc


def test_forward_agrees_with_viterbi_when_one_path_dominates(p, rng):
    """Strong stationary line: the MAP path carries the mass, so the two
    statistics must rank a training trace identically far above the null."""
    tr = make_trace("train", p, rng, wander_hz=0.0)
    null = make_trace("infer", p, rng, wander_hz=0.0)
    for stat in (viterbi_statistic, forward_statistic):
        assert stat(tr.t, tr.P_obs, p.band_lo, p.band_hi) > \
            stat(null.t, null.P_obs, p.band_lo, p.band_hi)


# --- Ko f0 drift (B1 / task 3.2) -------------------------------------------

def _ko_train_power(t, ko_p, p, rng, *, f0, f0_drift_hz=0.0):
    """Ko training F(t) -> observed power P = r*F + P0 + meter noise.

    The inline glue convention validated in test_detectors_fire_on_ko_training
    (r=1e-12, P0=250, f_peak=F_max*0.85, eta_scale=0.15); used here so the drift
    tests stay self-contained ahead of the ko_synth builder (task 3).
    """
    r, P0, f_peak = 1.0e-12, 250.0, DEFAULT.floor.F_max * 0.85
    F = training_F(t, ko_p, rng, f_peak=f_peak, f0=f0, eta_scale=0.15,
                   f0_drift_hz=f0_drift_hz)
    return r * F + P0 + rng.normal(0.0, p.sigma_eta, size=t.size)


# --- Ko -> TypeTrace population builder (B1 / task 3) -----------------------

def test_ko_population_shapes_and_labels(ko_p, glue, rng):
    train, infer = ko_make_population(4, ko_p, glue, rng)
    assert len(train) == 4 and len(infer) == 4
    assert all(tr.label == "train" for tr in train)
    assert all(tr.label == "infer" for tr in infer)
    n = int(round(glue.duration_s * glue.fs)) + 1
    assert all(tr.P_obs.shape == (n,) and tr.t.shape == (n,) for tr in train + infer)
    # training records its f0 in the Ko band; inference has none.
    assert all(ko_p.f0_lo <= tr.f0 <= ko_p.f0_hi for tr in train)
    assert all(np.isnan(tr.f0) for tr in infer)


def test_ko_make_trace_unknown_label_raises(ko_p, glue, rng):
    with pytest.raises(ValueError):
        ko_make_trace("bogus", ko_p, glue, rng)


def _ko_infer_power(t, ko_p, p, rng):
    """Ko hard inference null F(t) -> observed power (same glue as training)."""
    r, P0, f_peak = 1.0e-12, 250.0, DEFAULT.floor.F_max * 0.85
    F = inference_F(t, ko_p, rng, f_peak=f_peak, eta_scale=0.15)
    return r * F + P0 + rng.normal(0.0, p.sigma_eta, size=t.size)


def test_inference_F_runs_and_in_band(p, ko_p, rng):
    """The null produces in-band power and no pathological single-bin spike."""
    t = make_time_grid(p.duration_s, 1.0 / p.fs)
    F = inference_F(t, ko_p, rng, f_peak=DEFAULT.floor.F_max * 0.85)
    assert F.shape == t.shape
    assert np.all(F >= 0.0)
    f, psd = sp_signal.welch(F - F.mean(), fs=p.fs, nperseg=min(F.size, 1024))
    band = (f >= p.band_lo) & (f <= p.band_hi)
    assert psd[band].sum() > 0.0
    # No concentrated line: the largest in-band bin is not a giant outlier (guards
    # the _MIN_JITTER_FACTOR clip from manufacturing a spurious spike under jitter).
    assert psd[band].max() < 50.0 * np.median(psd[band])


def test_inference_null_has_no_trackable_line(p, ko_p):
    """The hard null carries in-band power but no temporally-coherent line (task 3.3).

    The null is deliberately concentrated enough to fool a peak-to-background filter
    (it is NOT 5x below a stationary line, unlike the easy B0 null) -- so the
    discriminator is the *temporal* one: a stationary training line gives a high
    Viterbi path score, the null does not. A stationary line is still spectrally
    sharper than the null, just by a smaller margin.
    """
    t = make_time_grid(p.duration_s, 1.0 / p.fs)
    tr_spec, tr_vit, in_spec, in_vit = [], [], [], []
    for s in range(6):
        train = _ko_train_power(t, ko_p, p, np.random.default_rng(s), f0=1.0)
        infer = _ko_infer_power(t, ko_p, p, np.random.default_rng(100 + s))
        tr_spec.append(spectral_statistic(t, train, p.band_lo, p.band_hi))
        in_spec.append(spectral_statistic(t, infer, p.band_lo, p.band_hi))
        tr_vit.append(viterbi_statistic(t, train, p.band_lo, p.band_hi))
        in_vit.append(viterbi_statistic(t, infer, p.band_lo, p.band_hi))
    # A stationary line is sharper than the null (separable at stationary)...
    assert np.median(tr_spec) > 1.5 * np.median(in_spec)
    # ...but the decisive gap is temporal: the null has no trackable path.
    assert np.median(tr_vit) > 2.0 * np.median(in_vit)


def test_ko_drift_zero_is_byte_identical(ko_p):
    """f0_drift_hz=0.0 must not perturb the RNG stream (scenarios/tracker rely on it)."""
    t = make_time_grid(300.0, 0.05)
    f_peak = DEFAULT.floor.F_max * 0.85
    f_old = training_F(t, ko_p, np.random.default_rng(0), f_peak=f_peak)
    f_new = training_F(t, ko_p, np.random.default_rng(0), f_peak=f_peak,
                       f0_drift_hz=0.0)
    assert np.array_equal(f_old, f_new)


def test_ko_drift_smears_spectral_peak_viterbi_holds(p, ko_p):
    """Drift collapses the matched filter while the Viterbi tracker degrades little."""
    t = make_time_grid(p.duration_s, 1.0 / p.fs)
    stationary = [
        spectral_statistic(t, _ko_train_power(t, ko_p, p,
                           np.random.default_rng(s), f0=1.0, f0_drift_hz=0.0),
                           p.band_lo, p.band_hi) for s in range(8)
    ]
    drifted = [
        spectral_statistic(t, _ko_train_power(t, ko_p, p,
                           np.random.default_rng(s), f0=1.0, f0_drift_hz=0.4),
                           p.band_lo, p.band_hi) for s in range(8)
    ]
    assert np.median(drifted) < 0.6 * np.median(stationary)

    vit_stat = [
        viterbi_statistic(t, _ko_train_power(t, ko_p, p,
                          np.random.default_rng(s), f0=1.0, f0_drift_hz=0.0),
                          p.band_lo, p.band_hi) for s in range(8)
    ]
    vit_drift = [
        viterbi_statistic(t, _ko_train_power(t, ko_p, p,
                          np.random.default_rng(s), f0=1.0, f0_drift_hz=0.4),
                          p.band_lo, p.band_hi) for s in range(8)
    ]
    # Viterbi keeps a far larger fraction of its statistic than the matched filter.
    spectral_ratio = np.median(drifted) / np.median(stationary)
    viterbi_ratio = np.median(vit_drift) / np.median(vit_stat)
    assert viterbi_ratio > spectral_ratio


def test_evaluate_accepts_population_fn(p):
    """Injectable population factory: default == explicit; a custom fn is honoured."""
    wanders = np.array([0.0])
    default = evaluate(p, wanders, n_each=10, target_far=0.05)
    explicit = evaluate(p, wanders, n_each=10, target_far=0.05,
                        population_fn=make_population)
    # Same seed convention -> byte-identical statistics -> identical TPR.
    assert default[0].tpr_at_far == explicit[0].tpr_at_far

    calls = {"n": 0}

    def fake_pop(n_each, params, rng, *, wander_hz):
        calls["n"] += 1
        return make_population(n_each, params, rng, wander_hz=wander_hz)

    evaluate(p, wanders, n_each=5, population_fn=fake_pop)
    assert calls["n"] == 1  # one wander level -> one population draw


# --- Ko gate: the real numbers (B1 / task 3.1, 3.2, 3.3) -------------------

def _ko_pop_fn(ko_p, glue):
    """Gate population_fn over the Ko generator: wander_hz -> Ko f0_drift_hz."""
    def fn(n_each, _p, rng, *, wander_hz):
        return ko_make_population(n_each, ko_p, glue, rng, f0_drift_hz=wander_hz)
    return fn


def test_inference_inband_power_comparable(p, ko_p, glue, rng):
    """The hard null's in-band power is comparable to training's (within ~2x).

    Pins inf_amp_frac: the null must carry COMPARABLE in-band power to a training
    line (not a weak strawman), so the detectors discriminate concentration, not
    power (spec.md Sec. 4 B1 premise 2).
    """
    t = make_time_grid(glue.duration_s, 1.0 / glue.fs)

    def inband_rms(P):
        x = P - P.mean()
        f, psd = sp_signal.welch(x, fs=glue.fs, nperseg=min(x.size, 1024))
        b = (f >= glue.band_lo) & (f <= glue.band_hi)
        return np.sqrt(np.trapezoid(psd[b], f[b]))

    tr = [inband_rms(_ko_train_power(t, ko_p, glue, np.random.default_rng(s), f0=1.0))
          for s in range(8)]
    inf = [inband_rms(_ko_infer_power(t, ko_p, glue, np.random.default_rng(100 + s)))
           for s in range(8)]
    ratio = np.median(inf) / np.median(tr)
    assert 0.5 < ratio < 2.0


def test_ko_gate_stationary_tie(ko_p, glue):
    """Stationary Ko line: both detectors near-perfect (matched filter not strawmanned)."""
    pts = evaluate(glue, np.array([0.0]), n_each=80, target_far=0.05,
                   population_fn=_ko_pop_fn(ko_p, glue))
    assert pts[0].tpr_at_far["spectral"] > 0.9
    assert pts[0].tpr_at_far["viterbi"] > 0.9


def test_ko_gate_drift_viterbi_wins(ko_p, glue):
    """Honest f0 drift: spectral collapses on Ko, Viterbi holds (B0 verdict, real numbers)."""
    pts = evaluate(glue, np.array([0.0, 0.5]), n_each=80, target_far=0.05,
                   population_fn=_ko_pop_fn(ko_p, glue))
    drift = pts[1]
    assert drift.tpr_at_far["viterbi"] - drift.tpr_at_far["spectral"] > 0.3
    assert drift.auc["viterbi"] > drift.auc["spectral"]


def test_spectral_collapses_under_wander_viterbi_holds(p, rng):
    """The crux of the gate: stationary -> tie; wandering -> Viterbi wins."""
    pts = evaluate(p, np.array([0.0, 0.5]), n_each=60, target_far=0.05)
    stat, wander = pts[0], pts[1]
    # stationary: both essentially perfect
    assert stat.tpr_at_far["spectral"] > 0.95
    assert stat.tpr_at_far["viterbi"] > 0.95
    # wandering: spectral falls well below Viterbi
    assert wander.tpr_at_far["viterbi"] - wander.tpr_at_far["spectral"] > 0.3
    assert wander.auc["viterbi"] > wander.auc["spectral"]


# --- Ko aggregate (B1-agg / task 4) -----------------------------------------

def _with_dominant_frac(ko_p, frac):
    """Ko params whose aggregate_ratio gives the dominant a ``frac`` share
    (background held at Ko's 0.5:0.5)."""
    w_dom = frac * 1.0 / (1.0 - frac)
    return dataclasses.replace(ko_p, aggregate_ratio=(w_dom, 0.5, 0.5))


def test_aggregate_default_kwargs_unchanged(ko_p):
    """The new B1-agg knobs at their defaults draw the same RNG stream as before
    they existed (the aggregate counterpart of the f0_drift byte-identical guard)."""
    t = make_time_grid(120.0, 0.05)
    base = aggregate_F(t, ko_p, np.random.default_rng(7), f_peak=1.0)
    expl = aggregate_F(t, ko_p, np.random.default_rng(7), f_peak=1.0,
                       f0=None, eta_scale=1.0, f0_drift_hz=0.0)
    assert np.array_equal(base, expl)


def test_aggregate_drift_passthrough(ko_p):
    """f0_drift_hz reaches the dominant workload (and 0.0 stays byte-identical)."""
    t = make_time_grid(120.0, 0.05)
    a = aggregate_F(t, ko_p, np.random.default_rng(3), f_peak=1.0, f0=1.0)
    b = aggregate_F(t, ko_p, np.random.default_rng(3), f_peak=1.0, f0=1.0,
                    f0_drift_hz=0.5)
    c = aggregate_F(t, ko_p, np.random.default_rng(3), f_peak=1.0, f0=1.0,
                    f0_drift_hz=0.0)
    assert not np.array_equal(a, b)
    assert np.array_equal(a, c)


def test_aggregate_null_power_matched(ko_p):
    """The two B1-agg classes carry comparable total power (not a power detector)."""
    t = make_time_grid(300.0, 0.05)
    pos = np.median([aggregate_F(t, ko_p, np.random.default_rng(s),
                                 f_peak=1.0).mean() for s in range(5)])
    neg = np.median([aggregate_null_F(t, ko_p, np.random.default_rng(100 + s),
                                      f_peak=1.0).mean() for s in range(5)])
    assert 0.5 < neg / pos < 2.0


def test_ko_aggregate_population_shapes_and_labels(ko_p, glue, rng):
    train, infer = ko_make_aggregate_population(3, ko_p, glue, rng)
    assert len(train) == 3 and len(infer) == 3
    assert all(tr.label == "train" for tr in train)
    assert all(tr.label == "infer" for tr in infer)
    n = int(round(glue.duration_s * glue.fs)) + 1
    assert all(tr.P_obs.shape == (n,) and tr.t.shape == (n,) for tr in train + infer)
    # the DOMINANT training's f0 is recorded; the null aggregate has none.
    assert all(ko_p.f0_lo <= tr.f0 <= ko_p.f0_hi for tr in train)
    assert all(np.isnan(tr.f0) for tr in infer)


def test_ko_make_aggregate_trace_unknown_label_raises(ko_p, glue, rng):
    with pytest.raises(ValueError):
        ko_make_aggregate_trace("bogus", ko_p, glue, rng)


def test_aggregate_dominant_line_detectable_at_nominal(ko_p, glue):
    """At Ko's nominal 9:0.5:0.5 the dominant line stands above the aggregate null
    for both detectors -- even though the null carries the background's weak lines."""
    tr_spec, tr_vit, in_spec, in_vit = [], [], [], []
    for s in range(5):
        tr = ko_make_aggregate_trace("train", ko_p, glue, np.random.default_rng(s))
        inf = ko_make_aggregate_trace("infer", ko_p, glue,
                                      np.random.default_rng(100 + s))
        tr_spec.append(spectral_statistic(tr.t, tr.P_obs, glue.band_lo, glue.band_hi))
        in_spec.append(spectral_statistic(inf.t, inf.P_obs, glue.band_lo, glue.band_hi))
        tr_vit.append(viterbi_statistic(tr.t, tr.P_obs, glue.band_lo, glue.band_hi))
        in_vit.append(viterbi_statistic(inf.t, inf.P_obs, glue.band_lo, glue.band_hi))
    assert np.median(tr_spec) > 1.5 * np.median(in_spec)
    assert np.median(tr_vit) > 2.0 * np.median(in_vit)


def test_aggregate_detection_degrades_as_dominance_falls(ko_p, glue):
    """Burying the training run (dominant share 0.9 -> 0.3) erodes both detector
    statistics -- the superposition axis the B1-agg gate sweeps."""
    def med_stats(frac):
        kp = _with_dominant_frac(ko_p, frac)
        spec, vit = [], []
        for s in range(5):
            tr = ko_make_aggregate_trace("train", kp, glue, np.random.default_rng(s))
            spec.append(spectral_statistic(tr.t, tr.P_obs, glue.band_lo, glue.band_hi))
            vit.append(viterbi_statistic(tr.t, tr.P_obs, glue.band_lo, glue.band_hi))
        return np.median(spec), np.median(vit)

    spec_hi, vit_hi = med_stats(0.9)
    spec_lo, vit_lo = med_stats(0.3)
    assert spec_lo < 0.8 * spec_hi
    assert vit_lo < 0.8 * vit_hi


def _ko_agg_pop_fn(ko_p, glue):
    """Gate population_fn over the aggregate builders (wander_hz -> f0_drift_hz)."""
    def fn(n_each, _p, rng, *, wander_hz):
        return ko_make_aggregate_population(n_each, ko_p, glue, rng,
                                            f0_drift_hz=wander_hz)
    return fn


def test_ko_aggregate_gate_nominal_stationary_and_drift(ko_p, glue):
    """The B1-single verdict survives superposition at the nominal ratio:
    stationary -> both detectors near-perfect; drift -> spectral collapses,
    Viterbi holds."""
    pts = evaluate(glue, np.array([0.0, 0.5]), n_each=30, target_far=0.05,
                   population_fn=_ko_agg_pop_fn(ko_p, glue))
    stat, drift = pts[0], pts[1]
    assert stat.tpr_at_far["spectral"] > 0.9
    assert stat.tpr_at_far["viterbi"] > 0.9
    assert drift.tpr_at_far["viterbi"] - drift.tpr_at_far["spectral"] > 0.3
    assert drift.auc["viterbi"] > drift.auc["spectral"]


# --- ROC helpers -----------------------------------------------------------

def test_auc_perfect_and_chance():
    assert auc(np.array([3.0, 4.0]), np.array([1.0, 2.0])) == 1.0
    # identical populations -> all ties -> 0.5
    assert auc(np.array([1.0, 2.0]), np.array([1.0, 2.0])) == 0.5


def test_roc_endpoints_monotonic():
    far, tpr, _ = roc_points(np.array([2.0, 3.0, 4.0]), np.array([0.0, 1.0]))
    assert far[0] == 0.0 and far[-1] == 1.0
    assert np.all(np.diff(far) >= 0)


def test_tpr_at_far_reads_the_curve():
    # perfectly separable: TPR = 1 at any FAR >= 0
    assert tpr_at_far(np.array([5.0, 6.0, 7.0]), np.array([0.0, 1.0]), 0.05) == 1.0


# --- running Viterbi path score --------------------------------------------

def test_viterbi_path_scores_prefix_consistency():
    # The forward recursion computes every prefix on the way to the final
    # scalar: out[j-1] must equal the whole detector run on a truncation
    # covering exactly the first j frames. Two tones with integer periods in
    # every tested prefix keep the trace zero-mean, so the full-trace detrend
    # matches the truncated-trace detrend and equality is tight.
    fs = 20.0
    t = np.arange(0, 60.0, 1.0 / fs)
    x = np.sin(2 * np.pi * 1.0 * t) + 0.5 * np.sin(2 * np.pi * 0.5 * t)
    kw = dict(nperseg=80, noverlap=40)          # hop 40 -> j frames = 40j+40 samples
    running = viterbi_path_scores(t, x, 0.3, 1.7, **kw)
    assert running.size == (t.size - 40) // 40
    for j in (1, 3, running.size):
        m = 40 * j + 40
        assert running[j - 1] == pytest.approx(
            viterbi_statistic(t[:m], x[:m], 0.3, 1.7, **kw), rel=1e-9)


def test_viterbi_path_scores_final_element_is_the_statistic(p, rng):
    tr = make_trace("train", p, rng)
    running = viterbi_path_scores(tr.t, tr.P_obs, p.band_lo, p.band_hi)
    assert running.size > 1
    assert viterbi_statistic(tr.t, tr.P_obs, p.band_lo, p.band_hi) == \
        pytest.approx(float(running[-1]))


def test_viterbi_path_scores_empty_band_is_empty():
    t = np.arange(0, 30.0, 0.05)
    x = np.sin(2 * np.pi * 1.0 * t)
    # band entirely above Nyquist -> no bins -> empty running score, 0 statistic
    assert viterbi_path_scores(t, x, 50.0, 60.0).size == 0
    assert viterbi_statistic(t, x, 50.0, 60.0) == 0.0


def test_viterbi_best_path_shapes_and_band(p, rng):
    tr = make_trace("train", p, rng)
    freqs, times, logP, path = viterbi_best_path(tr.t, tr.P_obs,
                                                 p.band_lo, p.band_hi)
    # map/path shapes are mutually consistent and the path stays in the band
    assert logP.shape == (freqs.size, times.size)
    assert path.size == times.size
    assert np.all((path >= p.band_lo) & (path <= p.band_hi))


def test_viterbi_best_path_tracks_the_line(p, rng):
    # a clean stationary line: the recovered path should sit near f0 most of the
    # time (median tracked frequency within one FFT bin of the true cadence).
    tr = make_trace("train", p, rng)
    freqs, _times, _logP, path = viterbi_best_path(tr.t, tr.P_obs,
                                                   p.band_lo, p.band_hi)
    bin_hz = freqs[1] - freqs[0]
    assert abs(np.median(path) - tr.f0) <= 2 * bin_hz
