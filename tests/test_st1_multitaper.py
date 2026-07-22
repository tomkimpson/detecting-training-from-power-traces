"""ST1 multitaper tests (task 19.2): DPSS orthonormality, F peaking at an
injected line, the fixed-frequency F(2, 2K-2) null distribution (Monte-Carlo
KS), and the comb variant beating the single line on a harmonic-rich wave."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import f as f_dist, kstest

from powerladder.config import DEFAULT
from powerladder.st1.multitaper import (comb_f_statistic, dpss_eigencoef,
                                 f_test_statistic, harmonic_f)
from powerladder.typeb.synth import _duty_wave

FS = 20.0


@pytest.fixture
def params():
    return DEFAULT.st1


def test_dpss_tapers_orthonormal(params):
    x = np.zeros(1024)
    _, _, tapers = dpss_eigencoef(x, FS, params=params)
    gram = tapers @ tapers.T
    np.testing.assert_allclose(gram, np.eye(params.n_tapers), atol=1e-8)


def test_f_peaks_at_injected_sinusoid(params):
    rng = np.random.default_rng(0)
    n = 6001
    t = np.arange(n) / FS
    f_line = 0.9
    x = 5.0 * np.cos(2 * np.pi * f_line * t) + rng.normal(0, 8.0, n)
    freqs, y, tapers = dpss_eigencoef(x, FS, params=params)
    F = harmonic_f(y, tapers)
    band = (freqs >= 0.3) & (freqs <= 1.7)
    f_peak = freqs[band][np.argmax(F[band])]
    assert abs(f_peak - f_line) < 0.02


def test_fixed_bin_f_follows_f_2_2km2(params):
    """~2000 white-noise draws of the max-free FIXED-BIN F: KS vs F(2, 2K-2)."""
    rng = np.random.default_rng(1)
    n, n_mc = 256, 2000
    draws = np.empty(n_mc)
    i_bin = None
    for i in range(n_mc):
        x = rng.normal(0.0, 1.0, n)
        freqs, y, tapers = dpss_eigencoef(x, FS, params=params)
        if i_bin is None:
            # a fixed bin well away from DC and Nyquist (no search, no max)
            i_bin = int(np.argmin(np.abs(freqs - 0.4 * FS / 2)))
        F = harmonic_f(y, tapers)
        draws[i] = F[i_bin]
    dof2 = 2 * params.n_tapers - 2
    res = kstest(draws, lambda q: f_dist.cdf(q, 2, dof2))
    assert res.pvalue > 1e-3, f"KS p = {res.pvalue:.2e} (stat {res.statistic:.4f})"


def test_comb_beats_single_line_on_duty_wave(params):
    """A square-ish duty wave spreads power into harmonics; Fisher-combining
    them must recover evidence the single-line max-F leaves on the table."""
    rng = np.random.default_rng(2)
    n = 6001
    t = np.arange(n) / FS
    phase = 2 * np.pi * 0.9 * t
    wins = 0
    n_trials = 6
    for _ in range(n_trials):
        x = 6.0 * _duty_wave(phase, duty=0.3, n_harmonics=3) + rng.normal(0, 12.0, n)
        s_line = f_test_statistic(t, x, 0.3, 1.7, params=params)
        s_comb = comb_f_statistic(t, x, 0.3, 1.7, params=params)
        wins += s_comb > s_line
    assert wins >= n_trials - 1, f"comb won only {wins}/{n_trials}"


def test_statistics_near_zero_on_white_noise(params):
    rng = np.random.default_rng(3)
    n = 6001
    t = np.arange(n) / FS
    x = rng.normal(0.0, 8.0, n)
    # -log10 of a Bonferroni-corrected p on a null trace: typically ~0
    assert f_test_statistic(t, x, 0.3, 1.7, params=params) < 3.0
    assert comb_f_statistic(t, x, 0.3, 1.7, params=params) < 3.0
