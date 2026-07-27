"""Unit checks for the cost-vs-hiding Pareto reducers (Phase 4).

The plot script is a pure reader of the frozen frontier summary, so the things
worth testing are its reductions, not its rendering: the class envelope, the
anchored/unpriced split, and the monotonicity of the Pareto staircase. Loaded by
path via importlib because scripts/ is not an importable package (the scripts
carry sys.path shims instead) — the same idiom as tests/test_st2_sweeps.py.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib

import pytest

from powerladder.typeb.meter_boundary import FIXED, TRACKING

_SCRIPT = (pathlib.Path(__file__).resolve().parent.parent
           / "scripts" / "plot_st2_pareto.py")
_spec = importlib.util.spec_from_file_location("plot_st2_pareto", _SCRIPT)
pareto = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pareto)


def _cell(family, level, cost, tprs):
    """A frontier cell with the fields the reducers read."""
    return {
        "family": family,
        "level": level,
        "cost_overhead_pct": cost,
        "tpr_at_far": {d: {"0.05": v, "0.01": v} for d, v in tprs.items()},
    }


def _summary(cells):
    return {"detector_classes": {"tracking": list(TRACKING),
                                 "fixed": list(FIXED)},
            "cells": cells}


_ALL_HIGH = {d: 1.0 for d in TRACKING + FIXED}


def test_detector_classes_read_from_summary():
    """Classes come from the accompanying summary, cross-checked to the library."""
    s = _summary([])
    assert pareto.detector_classes(s) == {"tracking": TRACKING, "fixed": FIXED}


def test_detector_classes_reject_mismatch():
    """A summary written under a different class definition must not be plotted."""
    s = _summary([])
    s["detector_classes"]["fixed"] = ["spectral"]
    with pytest.raises(SystemExit):
        pareto.detector_classes(s)


def test_hiding_is_one_minus_class_envelope():
    """Hiding uses the BEST class member, so hiding from a class needs all of it."""
    tprs = dict(_ALL_HIGH)
    tprs["viterbi"] = 0.4
    tprs["dg_order_semicoh"] = 0.2
    tprs["dg_order_full"] = 0.9      # this one still sees it, so nothing is hidden
    c = _cell("work", 0.5, 0.0, tprs)
    assert pareto.hiding(c, TRACKING) == pytest.approx(1.0 - 0.9)
    # the fixed class is untouched at 1.0, so hiding from it is nil
    assert pareto.hiding(c, FIXED) == pytest.approx(0.0)


def test_split_by_cost_counts_unpriced_and_keeps_them():
    """Null-cost cells are separated out, not dropped."""
    cells = [_cell("jitter", 0.2, 66.5, _ALL_HIGH),
             _cell("work", 0.7, None, _ALL_HIGH),
             _cell("meter", "integrating_1hz", None, _ALL_HIGH)]
    anchored, unpriced = pareto.split_by_cost(cells, TRACKING)
    assert [p["family"] for p in anchored] == ["jitter"]
    assert [p["family"] for p in unpriced] == ["work", "meter"]
    assert len(anchored) + len(unpriced) == len(cells)
    # unpriced cells keep their hiding value; only the x coordinate is missing
    assert all(p["cost"] is None for p in unpriced)
    assert unpriced[0]["hiding"] == pytest.approx(0.0)   # _ALL_HIGH: nothing hidden


def test_pareto_envelope_is_monotone_and_cost_ordered():
    """The staircase is a running max over ascending cost."""
    pts = [{"family": "a", "level": 1, "cost": 200.0, "hiding": 0.2},
           {"family": "a", "level": 2, "cost": 0.0, "hiding": 0.7},
           {"family": "a", "level": 3, "cost": 100.0, "hiding": 0.1}]
    env = pareto.pareto_envelope(pts)
    assert [c for c, _ in env] == [0.0, 100.0, 200.0]
    hid = [h for _, h in env]
    assert hid == [0.7, 0.7, 0.7]                      # running max, not raw
    assert all(b >= a for a, b in zip(hid, hid[1:]))   # monotone non-decreasing


def test_pareto_envelope_ignores_unpriced():
    """Cells with no cost anchor cannot enter the envelope (no x coordinate)."""
    cells = [_cell("jitter", 0.35, 158.9, {**_ALL_HIGH, "viterbi": 0.84,
                                           "dg_order_full": 0.84,
                                           "dg_order_semicoh": 0.84}),
             _cell("work", 0.7, None, {**_ALL_HIGH, "viterbi": 0.0,
                                       "dg_order_full": 0.0,
                                       "dg_order_semicoh": 0.0})]
    anchored, unpriced = pareto.split_by_cost(cells, TRACKING)
    env = pareto.pareto_envelope(anchored)
    # the unpriced cell hides completely, but must not lift the measured staircase
    assert unpriced[0]["hiding"] == pytest.approx(1.0)
    assert max(h for _, h in env) == pytest.approx(1.0 - 0.84)


def test_frozen_summary_reproduces_the_headline_asymmetry():
    """Regression on the real artefact: ~zero cost hides from fixed, not tracking."""
    summary = json.loads(
        (pathlib.Path(__file__).resolve().parent.parent / "results" / "st2"
         / "frontier_summary.json").read_text())
    classes = pareto.detector_classes(summary)
    track, _ = pareto.split_by_cost(summary["cells"], classes["tracking"])
    fixed, _ = pareto.split_by_cost(summary["cells"], classes["fixed"])

    cheap = [p for p in fixed if abs(p["cost"]) < 1.0]
    assert max(p["hiding"] for p in cheap) > 0.7      # fixed tests fall for free
    assert max(p["hiding"] for p in track) < 0.2      # tracking holds everywhere
