"""ST1 FAR-harness tests (task 19.4): Clopper-Pearson CI against known values,
and the harness recovering nominal FAR on a fake detector emitting uniform
p-values — the self-check the whole calibration table rests on."""

from __future__ import annotations

import json

import numpy as np
import pytest
from scipy.stats import norm

from powerladder.config import DEFAULT
from powerladder.st1.far import (binomial_ci, params_hash, run_cell, save_cell_raw,
                          update_summary)
from powerladder.st1.nulls import white


def test_binomial_ci_known_values():
    # k=5, n=100, 95%: standard Clopper-Pearson tables give [0.0164, 0.1128]
    lo, hi = binomial_ci(5, 100, 0.95)
    assert abs(lo - 0.01657) < 5e-4
    assert abs(hi - 0.11283) < 5e-4
    # k=0: lo is exactly 0; hi = 1 - (a/2)^(1/n)
    lo, hi = binomial_ci(0, 50, 0.95)
    assert lo == 0.0
    assert abs(hi - (1.0 - 0.025 ** (1.0 / 50.0))) < 1e-9
    # k=n: hi is exactly 1
    lo, hi = binomial_ci(50, 50, 0.95)
    assert hi == 1.0 and lo > 0.9


def test_binomial_ci_rejects_bad_k():
    with pytest.raises(ValueError):
        binomial_ci(5, 4)


def _uniform_p_detector(t, p_obs, band_lo, band_hi):
    """Fake detector whose p-value is exactly U(0,1) on the white null.

    The white null's first sample is P_base + N(0, sigma), so its normal CDF
    is uniform — a deterministic function of the trace, no extra RNG.
    """
    p_null = DEFAULT.st1_null
    u = norm.cdf((p_obs[0] - p_null.P_base) / p_null.sigma)
    u = min(max(u, 1e-300), 1.0)
    return -np.log10(u)


def test_harness_recovers_nominal_far_on_uniform_p():
    M = 2000
    cell = run_cell(
        _uniform_p_detector, white, M, 123,
        stage="stage0", null_name="white", calib="fake",
        n=64, fs=20.0, null_params=DEFAULT.st1_null,
        band_lo=0.3, band_hi=1.7,
        nominal_levels=(0.05, 1e-2), ci_conf=0.95,
    )
    for lvl in (0.05, 1e-2):
        rec = cell.levels[lvl]
        lo, hi = rec["ci"]
        assert lo <= lvl <= hi, f"nominal {lvl} outside CI [{lo}, {hi}]"
    assert cell.ks_pvalue > 1e-3
    assert cell.cov_failures == 0
    assert cell.p.shape == (M,)


def test_run_cell_is_seed_reproducible():
    kwargs = dict(stage="s", null_name="white", calib="fake", n=64, fs=20.0,
                  null_params=DEFAULT.st1_null, band_lo=0.3, band_hi=1.7,
                  nominal_levels=(0.05,), ci_conf=0.95)
    a = run_cell(_uniform_p_detector, white, 50, 7, **kwargs)
    b = run_cell(_uniform_p_detector, white, 50, 7, **kwargs)
    c = run_cell(_uniform_p_detector, white, 50, 8, **kwargs)
    np.testing.assert_array_equal(a.p, b.p)
    assert not np.array_equal(a.p, c.p)


def test_save_and_summary_roundtrip(tmp_path):
    cell = run_cell(
        _uniform_p_detector, white, 50, 0,
        stage="stage1", null_name="white", calib="fake",
        n=64, fs=20.0, null_params=DEFAULT.st1_null,
        band_lo=0.3, band_hi=1.7, nominal_levels=(0.05, 1e-2),
        phash=params_hash(DEFAULT.st1, DEFAULT.st1_null),
    )
    raw = save_cell_raw(cell, tmp_path)
    assert raw.name == "stage1_white_fake.npz"
    loaded = np.load(raw)
    np.testing.assert_array_equal(loaded["p"], cell.p)

    summary_path = tmp_path / "far_summary.json"
    update_summary(cell, summary_path)
    summary = json.loads(summary_path.read_text())
    entry = summary["stage1"]["white"]["fake"]
    assert entry["M"] == 50
    assert set(entry["levels"]) == {"0.05", "0.01"}
    assert entry["params_hash"] == cell.params_hash
