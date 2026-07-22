"""ST1 null-suite tests (task 19.1): every null runs, is reproducible, and has
the structural property it exists to contribute (skew, resonance peak,
nonstationary variance, controller band)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import signal as sp_signal
from scipy import stats

from powerladder.config import DEFAULT
from powerladder.st1.nulls import NULLS, STAGE5, STAGE6, STAGE7

N = 6001   # standard 300 s @ 20 Hz grid (matches TypeBParams)
FS = 20.0


@pytest.fixture
def p():
    return DEFAULT.st1_null


def test_registry_covers_all_stages():
    assert set(STAGE5) | set(STAGE6) | set(STAGE7) == set(NULLS)


@pytest.mark.parametrize("name", sorted(NULLS))
def test_null_runs_with_expected_shape_and_label(name, p):
    tr = NULLS[name](N, FS, p, np.random.default_rng(0))
    assert tr.t.shape == (N,) and tr.P_obs.shape == (N,)
    assert np.all(np.isfinite(tr.P_obs))
    assert tr.label == "null" and np.isnan(tr.f0)
    assert np.isclose(tr.t[1] - tr.t[0], 1.0 / FS)
    # fluctuation rides on the shared baseline
    assert abs(tr.P_obs.mean() - p.P_base) < 5.0 * p.sigma


@pytest.mark.parametrize("name", sorted(NULLS))
def test_null_seed_reproducibility(name, p):
    a = NULLS[name](N, FS, p, np.random.default_rng(42))
    b = NULLS[name](N, FS, p, np.random.default_rng(42))
    c = NULLS[name](N, FS, p, np.random.default_rng(43))
    np.testing.assert_array_equal(a.P_obs, b.P_obs)
    assert not np.array_equal(a.P_obs, c.P_obs)


def test_lognormal_is_positively_skewed(p):
    tr = NULLS["lognormal"](N, FS, p, np.random.default_rng(1))
    assert stats.skew(tr.P_obs) > 0.5


def _psd_peak_hz(x, fs):
    f, psd = sp_signal.welch(x - x.mean(), fs=fs, nperseg=min(x.size, 2048))
    keep = f > 0.05          # ignore the DC/drift bins
    return f[keep][np.argmax(psd[keep])]


def test_ar2_resonant_peaks_near_configured_frequency(p):
    tr = NULLS["ar2_resonant"](N, FS, p, np.random.default_rng(2))
    assert abs(_psd_peak_hz(tr.P_obs, FS) - p.ar2_peak_hz) < 0.15


def test_controller_psd_peak_in_measured_band(p):
    for seed in (0, 1, 2):
        tr = NULLS["controller"](N, FS, p, np.random.default_rng(seed))
        peak = _psd_peak_hz(tr.P_obs, FS)
        assert p.ctrl_f_lo <= peak <= p.ctrl_f_hi, f"seed {seed}: peak {peak} Hz"


def test_tvar_variance_grows_between_halves(p):
    tr = NULLS["tvar"](N, FS, p, np.random.default_rng(3))
    x = tr.P_obs - p.P_base
    v1, v2 = x[: N // 2].var(), x[N // 2 :].var()
    assert v2 > 2.0 * v1     # phi ramp 0.5 -> 0.95 more than doubles the variance
