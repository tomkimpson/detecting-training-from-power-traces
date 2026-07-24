"""Rung-2 tests: physics feature vector, decision rules, semantic controls, grid.

Mirrors tests/test_st2_meter_boundary.py: schema/uniqueness, smoke+determinism,
and anchor tests reproducing the design's committed expectations. All sizes are
tiny (short duration, few traces) so the suite stays fast — the real numbers are
frozen on Slurm.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from powerladder.config import DEFAULT
from powerladder.typeb.ko_synth import ko_make_population, ko_make_trace
from powerladder.typeb.rung2 import (FIXED_COMPARATOR_FEATURES, physics_score,
                                     physics_weights, run_rung2_cell,
                                     run_stated_cell)
from powerladder.typeb.rung2_features import (FEATURE_NAMES, feature_matrix,
                                              feature_vector)
from powerladder.typeb.rung2_scenarios import (CONTROL_ORDER, CONTROLS,
                                               make_control_population)


@pytest.fixture
def glue():
    """Short-duration glue so feature extraction is fast in tests."""
    return dataclasses.replace(DEFAULT.ko_typeb, duration_s=90.0)


@pytest.fixture
def small_p():
    return dataclasses.replace(DEFAULT.rung2, n_each=16, n_folds=3)


# --- feature vector --------------------------------------------------------

def test_feature_vector_shape_and_names(glue):
    rng = np.random.default_rng(0)
    tr = ko_make_trace("train", DEFAULT.ko, glue, rng)
    vec, names = feature_vector(tr.t, tr.P_obs, glue.band_lo, glue.band_hi)
    assert names == list(FEATURE_NAMES)
    assert vec.shape == (len(FEATURE_NAMES),)
    assert np.all(np.isfinite(vec))


def test_feature_vector_deterministic(glue):
    rng = np.random.default_rng(1)
    tr = ko_make_trace("train", DEFAULT.ko, glue, rng)
    a, _ = feature_vector(tr.t, tr.P_obs, glue.band_lo, glue.band_hi)
    b, _ = feature_vector(tr.t, tr.P_obs, glue.band_lo, glue.band_hi)
    assert np.array_equal(a, b)


def test_features_estimated_only_ignore_label(glue):
    """Features depend on (t, P_obs) alone — never on the generator label/f0."""
    rng = np.random.default_rng(2)
    train, infer = ko_make_population(4, DEFAULT.ko, glue, rng)
    X, _ = feature_matrix(train, glue)
    # Relabel the same traces (and blank f0); the feature matrix must not move.
    relabelled = [dataclasses.replace(tr, label="infer", f0=float("nan"))
                  for tr in train]
    X2, _ = feature_matrix(relabelled, glue)
    assert np.array_equal(X, X2)


# --- prespecified physics score -------------------------------------------

def test_physics_weights_exclude_fixed_comparator():
    w = physics_weights(list(FEATURE_NAMES))
    for name, wi in zip(FEATURE_NAMES, w):
        expected = 0.0 if name in FIXED_COMPARATOR_FEATURES else 1.0
        assert wi == expected


def test_physics_score_deterministic_no_fit(glue):
    rng = np.random.default_rng(3)
    train, infer = ko_make_population(6, DEFAULT.ko, glue, rng)
    Xt, names = feature_matrix(train, glue)
    Xi, _ = feature_matrix(infer, glue)
    a = physics_score(Xt, Xi, names)
    b = physics_score(Xt, Xi, names)
    assert np.array_equal(a, b)          # no fitting, pure function of inputs


# --- anchor: the stated population is separable by all three rules ----------

def test_stated_population_separable(glue, small_p):
    rng = np.random.default_rng(11)
    train, infer = ko_make_population(small_p.n_each, DEFAULT.ko, glue, rng)
    rep = run_stated_cell(train, infer, glue, small_p, seed=11)
    for rule in ("physics_score", "fitted_discriminant", "learned_reference"):
        assert rep[rule]["auc"] >= 0.8, (rule, rep[rule]["auc"])


# --- semantic falsification controls ---------------------------------------

def test_control_registry_wellformed():
    assert tuple(CONTROLS) == CONTROL_ORDER
    for c in CONTROLS.values():
        assert c.expect in ("train", "not_train")


def test_control_populations_generate(glue):
    """Every control yields well-formed traces (finite power, right length)."""
    p = dataclasses.replace(DEFAULT.rung2, n_each=2)
    rng = np.random.default_rng(9)
    for name in CONTROL_ORDER:
        pop = make_control_population(name, 2, DEFAULT.ko, glue, rng, p, meter=None)
        assert len(pop) == 2
        for tr in pop:
            assert tr.P_obs.size == tr.t.size > 0
            assert np.all(np.isfinite(tr.P_obs))


def test_decoy_scores_as_training_async_missed(glue):
    """The two central findings, under the physics rule:

    the discarded-update decoy (physically identical to training) scores AS
    training; strongly de-periodicised async training is MISSED (the scoped-out
    efficient-family edge).
    """
    p = dataclasses.replace(DEFAULT.rung2, n_each=10, n_folds=3)
    mv = DEFAULT.st2.meter_variants
    decoy = run_rung2_cell("control", "discarded_update_decoy", DEFAULT.ko,
                           glue, p, meter_variants=mv, seed=21)
    async_ = run_rung2_cell("control", "async_training", DEFAULT.ko,
                            glue, p, meter_variants=mv, seed=21)
    # decoy is a non-training load that nonetheless scores as training
    assert decoy["is_training"] is False
    assert decoy["rules"]["physics_score"]["frac_training"]["0.05"] >= 0.5
    # async is genuine training the physics rule fails to flag
    assert async_["is_training"] is True
    assert async_["rules"]["physics_score"]["frac_training"]["0.05"] <= 0.5


def test_transfer_cell_schema(glue):
    p = dataclasses.replace(DEFAULT.rung2, n_each=8, n_folds=3)
    mv = DEFAULT.st2.meter_variants
    cell = run_rung2_cell("transfer", "f_peak_lo", DEFAULT.ko, glue, p,
                          meter_variants=mv, seed=31)
    assert cell["kind"] == "transfer" and cell["condition"] == "f_peak_lo"
    for rule in ("physics_score", "fitted_discriminant", "learned_reference"):
        blk = cell["rules"][rule]
        assert 0.0 <= blk["auc"] <= 1.0
        assert set(blk["fpr"]) == {f"{f:g}" for f in p.target_fars}
