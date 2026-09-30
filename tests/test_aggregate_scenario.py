"""The section-2 aggregate scenario of the headline drivers.

The bake-off, the Whittle ceiling and the meter boundary each take a
``scenario`` switch that swaps the single workload for the eq-11 aggregate. These
tests pin what the switch must get right: the ceiling's pinned-f0 bank sampler
is the aggregate generator exactly, the scenario changes the eval populations
deterministically, and the aggregate meter-boundary cell runs.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import pathlib

import numpy as np

from powerladder.config import DEFAULT, St2MeterBoundaryParams
from powerladder.typeb import np_ceiling as npc
from powerladder.typeb.detectors import viterbi_statistic
from powerladder.typeb.ko_synth import ko_make_aggregate_trace
from powerladder.typeb.meter_boundary import meter_grid, run_meter_cell

_SCRIPTS = pathlib.Path(__file__).resolve().parent.parent / "scripts"
_spec = importlib.util.spec_from_file_location("st1_np_ceiling", _SCRIPTS / "st1_np_ceiling.py")
driver = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(driver)

KO = DEFAULT.ko
GLUE = dataclasses.replace(DEFAULT.ko_typeb, duration_s=60.0)


def test_aggregate_bank_sampler_matches_aggregate_generator():
    """train_obs_at_f0(aggregate=True) is ko_make_aggregate_trace("train") with
    the dominant f0 pinned: same stream after the f0 draw -> identical trace."""
    for drift in (0.0, 0.4):
        rng_a = np.random.default_rng(11)
        ref = ko_make_aggregate_trace("train", KO, GLUE, rng_a, f0_drift_hz=drift)
        rng_b = np.random.default_rng(11)
        f0 = float(rng_b.uniform(KO.f0_lo, KO.f0_hi))
        assert f0 == ref.f0
        t, p = npc.train_obs_at_f0(f0, drift, KO, GLUE, rng_b, aggregate=True)
        assert np.array_equal(t, ref.t)
        assert np.array_equal(p, ref.P_obs)


def test_aggregate_bank_differs_from_single_bank():
    lo, hi = DEFAULT.st1.band_lo, DEFAULT.st1.band_hi
    kw = dict(f0_grid=np.array([0.8, 1.2]), drift_grid=(0.0,), n_mc=3,
              band_lo=lo, band_hi=hi)
    s_single, f1 = npc.build_training_bank(KO, GLUE, rng=np.random.default_rng(0), **kw)
    s_agg, f2 = npc.build_training_bank(KO, GLUE, rng=np.random.default_rng(0),
                                        aggregate=True, **kw)
    assert np.array_equal(f1, f2)
    assert s_agg.shape == s_single.shape == (2, f1.size)
    assert not np.allclose(s_agg, s_single)


def test_eval_populations_scenario_swaps_only_workload_classes():
    single = driver.build_eval_populations(4, [0.0], 20260721)
    agg = driver.build_eval_populations(4, [0.0], 20260721, scenario="aggregate")
    agg2 = driver.build_eval_populations(4, [0.0], 20260721, scenario="aggregate")
    (s_neg, _, s_pos), (a_neg, _, a_pos), (b_neg, _, b_pos) = single, agg, agg2
    # deterministic
    assert np.array_equal(a_pos[0.0][0].P_obs, b_pos[0.0][0].P_obs)
    assert np.array_equal(a_neg["inference"][0].P_obs, b_neg["inference"][0].P_obs)
    # the workload classes change: aggregate null (no line) and aggregate positive
    assert not np.array_equal(a_neg["inference"][0].P_obs, s_neg["inference"][0].P_obs)
    assert not np.array_equal(a_pos[0.0][0].P_obs, s_pos[0.0][0].P_obs)
    assert np.isnan(a_neg["inference"][0].f0)
    assert np.isfinite(a_pos[0.0][0].f0)


def test_meter_cell_aggregate_well_formed_and_deterministic():
    small = St2MeterBoundaryParams(
        n_each=4, sample_hz_grid=(20.0,), integ_window_grid=(0.0,),
        notch_hz_grid=(1.0,), notch_depth_grid=(1.0,))
    name, mp = meter_grid(small)[0]
    dets = {"viterbi": viterbi_statistic}
    a = run_meter_cell(name, mp, small, KO, GLUE, detectors=dets, aggregate=True)
    b = run_meter_cell(name, mp, small, KO, GLUE, detectors=dets, aggregate=True)
    assert a == b
    assert 0.0 <= a["auc"]["viterbi"] <= 1.0
    assert set(a["tpr_at_far"]["viterbi"]) == {f"{f:g}" for f in small.target_fars}
