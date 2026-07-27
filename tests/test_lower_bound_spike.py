"""Unit checks for the covertness-bound statistics (Phase 4).

`scripts/lower_bound_spike.py` supplies the sigma*(eps) thresholds quoted in paper
§3 / App. B, so its estimator and its fit mask are load-bearing: flip a comparison
or a polarity in `_youden` and every published threshold moves with nothing to
catch it. These tests pin the arithmetic, the sign convention, the H0 floor that
sets the fit window, and the inversion. Loaded by path via importlib because
scripts/ is not an importable package -- the idiom of tests/test_st2_sweeps.py.

Deliberately no test of the sweep itself: that runs the full generator for ~100 s
and is the script's own job (see README).
"""

from __future__ import annotations

import importlib.util
import math
import pathlib

import numpy as np
import pytest

_SCRIPT = (pathlib.Path(__file__).resolve().parent.parent
           / "scripts" / "lower_bound_spike.py")
_spec = importlib.util.spec_from_file_location("lower_bound_spike", _SCRIPT)
spike = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(spike)


# --- the achieved-advantage estimator ---------------------------------------

def test_youden_perfect_separation_is_one():
    """Training scores strictly above the null => advantage 1."""
    assert spike._youden(np.array([3.0, 4, 5]), np.array([0.0, 1, 2])) == 1.0


def test_youden_sign_convention_training_scores_high():
    """The statistic is one-sided: reversing the classes must NOT report 1.

    The fixed-bin SNR is large for training traces, so only sup(F_train - F_null)
    is a detection advantage. If this ever returns 1.0 the polarity has flipped
    and every sigma* would silently change.
    """
    assert spike._youden(np.array([0.0, 1, 2]), np.array([3.0, 4, 5])) == 0.0


def test_youden_is_never_negative():
    """J >= 0 by construction (the threshold below every point gives 0)."""
    rng = np.random.default_rng(0)
    for _ in range(20):
        assert spike._youden(rng.normal(size=30), rng.normal(size=30)) >= 0.0


def test_youden_hand_computable_case_with_ties():
    """Ties are handled at the shared threshold, not skipped."""
    # thr=2: TPR = P(pos>=2) = 2/3, FPR = P(neg>=2) = 0 -> J = 2/3
    assert spike._youden(np.array([1.0, 2, 3]),
                         np.array([0.0, 1, 1])) == pytest.approx(2.0 / 3.0)


def test_youden_matches_one_sided_ks_oracle():
    """Independent oracle: the one-sided two-sample KS statistic."""
    ks = pytest.importorskip("scipy.stats").ks_2samp
    rng = np.random.default_rng(1)
    for _ in range(25):
        pos, neg = rng.normal(0.7, 1, 40), rng.normal(0, 1, 55)
        assert spike._youden(pos, neg) == pytest.approx(
            ks(pos, neg, alternative="less").statistic)


def test_gini_can_exceed_youden():
    """Why the bound uses J and not 2*AUC-1: the Gini index is not attained.

    2*AUC-1 can exceed the best threshold test's advantage, so it is not the
    advantage of any single-trace verifier and cannot certify non-covertness.
    """
    pos, neg = np.array([3.0, 4, 5, 6]), np.array([0.0, 1, 2, 3.5])
    assert 2.0 * spike._auc(pos, neg) - 1.0 > spike._youden(pos, neg)


# --- the H0 floor that sets the fit window ----------------------------------

def test_ks_null_floor_scales_as_inverse_root_n():
    """Floor ~ 1/sqrt(n): quadrupling n must halve it."""
    assert spike._ks_null_floor(400) == pytest.approx(
        spike._ks_null_floor(100) / 2.0)


def test_ks_null_floor_brackets_the_empirical_null():
    """The floor must sit above the typical H0 draw and below its extreme tail."""
    n = 120
    rng = np.random.default_rng(2)
    draws = [spike._youden(rng.normal(size=n), rng.normal(size=n))
             for _ in range(300)]
    floor = spike._ks_null_floor(n)
    assert float(np.median(draws)) < floor          # excludes the typical null
    assert float(np.max(draws)) > floor / 3.0       # but is the right order


def test_fit_mask_excludes_null_and_saturated_points():
    """The window keeps only points carrying slope information."""
    sig = np.array([0.0, 0.1, 0.2, 0.5])
    y = np.array([1.0, 0.99, 0.6, 0.02])       # saturated, then informative, then null
    m = spike._fit_mask(sig, y, 1.0, floor=0.05)
    assert m.tolist() == [False, False, True, False]


# --- the fit and the inversion ----------------------------------------------

def test_fit_kappa_recovers_a_planted_rolloff():
    """Exact data through the model must return the planted kappa."""
    kappa, sig = 25.0, np.array([0.0, 0.05, 0.1, 0.15, 0.2, 0.3])
    y = 1.0 / (1.0 + kappa * sig ** 2)
    got, mask = spike._fit_kappa(sig, y, 1.0, floor=0.0)
    assert got == pytest.approx(kappa)
    assert not mask[0]                              # sigma=0 never enters the fit


def test_fit_kappa_refuses_an_underdetermined_window():
    """Too few informative points must fail loudly, not fit noise."""
    sig = np.array([0.0, 0.4, 0.5])
    y = np.array([1.0, 0.01, 0.005])               # everything below the floor
    with pytest.raises(SystemExit):
        spike._fit_kappa(sig, y, 1.0, floor=0.1)


def test_covertness_thresholds_invert_the_rolloff():
    """sigma* solves y0/(1+kappa sigma^2) = eps, and flags extrapolation."""
    kappa, y0 = 25.0, 1.0
    for th in spike._covertness_thresholds(kappa, y0, sig_max=0.3):
        expected = math.sqrt((y0 / th["epsilon"] - 1.0) / kappa)
        assert th["sigma_star"] == pytest.approx(expected)
        assert th["extrapolated"] == (expected > 0.3)


def test_covertness_thresholds_increase_as_epsilon_falls():
    """Stronger covertness demands more distortion (the bound's whole point)."""
    ths = spike._covertness_thresholds(25.0, 1.0, sig_max=1.0)
    stars = [t["sigma_star"] for t in
             sorted(ths, key=lambda t: -t["epsilon"])]
    assert all(b > a for a, b in zip(stars, stars[1:]))


def test_sigma_star_scales_as_inverse_root_epsilon():
    """The small-eps form quoted in App. B: sigma* ~ eps^(-1/2), not 1/eps.

    Only asymptotic: sigma* = sqrt((J0/eps - 1)/kappa), so the -1 is negligible
    only once J0/eps >> 1. Compared between two SMALL epsilons for that reason --
    at eps = 0.5 the -1 is half the value and the exponent has not settled.
    """
    ths = {t["epsilon"]: t["sigma_star"]
           for t in spike._covertness_thresholds(25.0, 1.0, sig_max=99.0)}
    # halving eps moves sigma* by sqrt(2) (~1.41), not by 2
    assert ths[0.01] / ths[0.02] == pytest.approx(math.sqrt(2.0), rel=0.02)
