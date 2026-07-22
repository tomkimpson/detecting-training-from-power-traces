"""ST1 surrogate tests (task 19.8): exact |rfft| preservation, DG-killing on
cyclostationary input, and rough uniformity of full-pipeline surrogate
p-values on a Gaussian null (small M/S to stay fast)."""

from __future__ import annotations

import numpy as np
import pytest

from powerladder.config import DEFAULT
from powerladder.st1.cyclo import dg_q
from powerladder.st1.pipeline import stage4_full_adaptive
from powerladder.st1.surrogates import (block_permutation_surrogate,
                                 fourier_phase_surrogate, surrogate_pvalue)

FS = 20.0


def test_fourier_surrogate_preserves_rfft_magnitude_exactly():
    rng = np.random.default_rng(0)
    for n in (4096, 4097):                      # even (Nyquist bin) and odd
        x = rng.normal(0.0, 1.0, n) + 0.7       # nonzero DC
        xs = fourier_phase_surrogate(x, rng)
        assert xs.shape == x.shape
        np.testing.assert_allclose(np.abs(np.fft.rfft(xs)),
                                   np.abs(np.fft.rfft(x)), rtol=1e-12,
                                   atol=1e-9)
        assert not np.allclose(xs, x)           # phases actually randomised


def test_block_surrogate_permutes_blocks_and_validates():
    rng = np.random.default_rng(1)
    x = np.arange(10.0)
    xs = block_permutation_surrogate(x, 3, rng)
    assert xs.shape == x.shape
    assert xs[-1] == x[-1]                      # partial trailing block fixed
    assert sorted(xs.tolist()) == sorted(x.tolist())
    blocks = {tuple(xs[i:i + 3]) for i in (0, 3, 6)}
    assert blocks == {(0., 1., 2.), (3., 4., 5.), (6., 7., 8.)}
    with pytest.raises(ValueError):
        block_permutation_surrogate(np.zeros(5), 3, rng)


def test_surrogate_kills_dg_on_cyclostationary_input():
    """AM noise: observed DG statistic far above every surrogate's."""
    rng = np.random.default_rng(2)
    n = 6001
    t = np.arange(n) / FS
    x = (1.0 + 0.5 * np.cos(2 * np.pi * 1.0 * t)) * rng.normal(0, 1, n)
    p = DEFAULT.st1
    q_obs = dg_q(x, FS, 1.0, p.lag_set, params=p).Q
    q_surr = np.array([
        dg_q(fourier_phase_surrogate(x, rng), FS, 1.0, p.lag_set, params=p).Q
        for _ in range(20)
    ])
    assert q_obs > 5.0 * q_surr.max(), \
        f"obs {q_obs:.1f} vs surrogate max {q_surr.max():.1f}"


def test_pipeline_surrogate_pvalues_roughly_uniform_on_gaussian_null():
    """Full stage-4 pipeline inside every surrogate; loose uniformity check."""
    p = DEFAULT.st1
    n = 3001                                    # 150 s traces to keep it fast
    t = np.arange(n) / FS
    rng = np.random.default_rng(3)
    n_traces, S = 16, 15

    def stat(tt, xx):
        return stage4_full_adaptive(tt, xx, 0.3, 1.7, params=p)

    pv = np.array([
        surrogate_pvalue(stat, t, rng.normal(0.0, 8.0, n), rng,
                         n_surrogates=S)
        for _ in range(n_traces)
    ])
    # p lives on {1/(S+1), ..., 1}; under the null its mean is ~ (S+2)/(2(S+1))
    assert pv.min() >= 1.0 / (S + 1) and pv.max() <= 1.0
    assert 0.3 < pv.mean() < 0.75, f"mean surrogate p {pv.mean():.3f}"
    assert np.unique(pv).size > 3               # not degenerate
