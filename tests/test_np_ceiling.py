"""Tests for the Whittle spectral LRT ceiling (powerladder.typeb.np_ceiling).

Fast: short-duration glue and tiny banks/populations — the real numbers freeze on
Slurm. Covers the periodogram contract, the Whittle-LR ordering on a controlled
synthetic input, corpus/eval seed hygiene, and the headline sanity property that the
ceiling dominates the spectral matched filter at drift 0.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import pathlib

import numpy as np
import pytest

from powerladder.config import DEFAULT
from powerladder.typeb import np_ceiling as npc
from powerladder.typeb.detectors import spectral_statistic
from powerladder.typeb.ko_synth import ko_make_trace
from powerladder.typeb.roc import auc

_DRIVER_PATH = (pathlib.Path(__file__).resolve().parent.parent
                / "scripts" / "st1_np_ceiling.py")
_spec = importlib.util.spec_from_file_location("st1_np_ceiling", _DRIVER_PATH)
driver = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(driver)


@pytest.fixture
def glue():
    """Short-duration glue so bank building / trace generation stay fast."""
    return dataclasses.replace(DEFAULT.ko_typeb, duration_s=60.0)


BAND = (DEFAULT.st1.band_lo, DEFAULT.st1.band_hi)


# ---------------------------------------------------------------------------
# periodogram contract
# ---------------------------------------------------------------------------

def test_band_periodogram_shape_and_determinism():
    fs, n = 20.0, 1201
    t = np.arange(n) / fs
    p = np.sin(2 * np.pi * 1.0 * t) + 250.0            # a 1 Hz line on a DC pedestal
    I, freqs = npc.band_periodogram(t, p, *BAND)
    assert I.shape == freqs.shape
    assert freqs.min() >= BAND[0] and freqs.max() <= BAND[1]
    assert np.all(freqs > 0.0)                          # DC bin excluded by the band
    # DC pedestal is removed -> the 1 Hz line dominates the in-band periodogram.
    assert abs(freqs[np.argmax(I)] - 1.0) < 0.05
    I2, _ = npc.band_periodogram(t, p, *BAND)
    assert np.array_equal(I, I2)


# ---------------------------------------------------------------------------
# Whittle log-LR ordering on a controlled synthetic input
# ---------------------------------------------------------------------------

def test_whittle_lr_prefers_power_under_the_training_template():
    """A tone under the training template's spectral bump scores higher than a tone
    where the template is flat — the core Whittle discrimination, RNG-free."""
    fs, n = 20.0, 400
    t = np.arange(n) / fs
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    lo, hi = BAND
    mask = (freqs >= lo) & (freqs <= hi)
    fband = freqs[mask]
    # pick two exact in-band FFT bins: one the template favours, one it does not
    j_bump = len(fband) // 3
    j_off = 2 * len(fband) // 3
    f_bump, f_off = float(fband[j_bump]), float(fband[j_off])

    # training template: flat floor with a tall bump at f_bump; negative: flat.
    S_tr = np.ones_like(fband)
    S_tr[j_bump] = 50.0
    S_neg = np.ones_like(fband)
    ceil = npc._bank_from_psds(
        fband, np.array([1.0]), S_tr[None, :], S_neg,
        lo, hi, "synthetic", psd_floor=1e-12)

    tone_bump = np.cos(2 * np.pi * f_bump * t)
    tone_off = np.cos(2 * np.pi * f_off * t)
    assert ceil.score(t, tone_bump) > ceil.score(t, tone_off)


# ---------------------------------------------------------------------------
# corpus / eval seed hygiene (leakage guard)
# ---------------------------------------------------------------------------

def test_corpus_stream_disjoint_from_eval():
    p = DEFAULT.np_ceiling
    assert p.corpus_seed_offset != 0
    corpus = np.random.default_rng(p.eval_seed + p.corpus_seed_offset)
    evalr = np.random.default_rng(p.eval_seed)
    # the two streams are distinct (astronomically unlikely to collide by chance)
    assert not np.array_equal(corpus.random(64), evalr.random(64))


def test_eval_population_reproduction_is_deterministic():
    """build_eval_populations replays a fixed RNG order -> byte-identical each call."""
    a_neg, a_alpha, a_pos = driver.build_eval_populations(8, [0.0, 0.4], 20260721)
    b_neg, b_alpha, b_pos = driver.build_eval_populations(8, [0.0, 0.4], 20260721)
    assert np.array_equal(a_alpha["inference"], b_alpha["inference"])
    for d in (0.0, 0.4):
        assert np.array_equal([tr.f0 for tr in a_pos[d]], [tr.f0 for tr in b_pos[d]])
        assert np.array_equal(a_pos[d][0].P_obs, b_pos[d][0].P_obs)


def test_structural_neg_sampler_covers_the_full_mix(monkeypatch):
    """The structural negative-PSD sampler must draw ALL of ar1/ar1_t/controller
    over many MC draws, not just ar1 -- guards the round-robin n=1 -> [1,0,0] trap
    that would silently fit the 'structural' ceiling against an ar1-only null."""
    seen = []
    orig = driver.bakeoff._make_structural_negatives

    def spy(n, r, names=driver.bakeoff.STRUCTURAL_MIX):
        seen.append(tuple(names))
        return orig(n, r, names=names)

    monkeypatch.setattr(driver.bakeoff, "_make_structural_negatives", spy)
    rng = np.random.default_rng(0)
    for _ in range(200):
        driver.NEG_SAMPLERS["structural"](rng)
    drawn = {nm for names in seen for nm in names}
    assert drawn == set(driver.bakeoff.STRUCTURAL_MIX)


# ---------------------------------------------------------------------------
# headline sanity: the ceiling dominates the spectral matched filter at drift 0
# ---------------------------------------------------------------------------

def test_ceiling_dominates_spectral_at_drift_zero(glue):
    ko = DEFAULT.ko
    lo, hi = BAND
    rng = np.random.default_rng(0)
    f0_grid = np.linspace(ko.f0_lo, ko.f0_hi, 9)
    S_bank, fb = npc.build_training_bank(
        ko, glue, f0_grid=f0_grid, drift_grid=(0.0,), n_mc=25, rng=rng,
        band_lo=lo, band_hi=hi)

    def infer_sampler(r):
        tr = ko_make_trace("infer", ko, glue, r)
        return tr.t, tr.P_obs

    ceil = npc.fit_ceiling(
        infer_sampler, "inference", S_bank=S_bank, f0_grid=f0_grid, bank_freqs=fb,
        n_mc=60, rng=rng, band_lo=lo, band_hi=hi)

    pos = [ko_make_trace("train", ko, glue, rng, f0_drift_hz=0.0) for _ in range(40)]
    neg = [ko_make_trace("infer", ko, glue, rng) for _ in range(40)]
    auc_ceil = auc(npc.score_ceiling(ceil, pos), npc.score_ceiling(ceil, neg))
    auc_spec = auc(
        np.array([spectral_statistic(t.t, t.P_obs, lo, hi) for t in pos]),
        np.array([spectral_statistic(t.t, t.P_obs, lo, hi) for t in neg]))
    assert auc_ceil >= auc_spec - 1e-9
    assert auc_ceil > 0.9        # the optimal spectral test separates the clean case
