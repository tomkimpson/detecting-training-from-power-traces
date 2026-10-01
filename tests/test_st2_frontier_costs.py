"""Cost join of the ST2 frontier assembly (scripts/plot_st2_frontier.py)."""

from __future__ import annotations

import importlib.util
import pathlib

_SCRIPT = (pathlib.Path(__file__).resolve().parent.parent
           / "scripts" / "plot_st2_frontier.py")
_spec = importlib.util.spec_from_file_location("plot_st2_frontier", _SCRIPT)
frontier = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(frontier)

_B2 = pathlib.Path(__file__).resolve().parent.parent / "data" / "measured_cost_anchors"


def _point(level):
    return {"level": level, "cadence_cv": 0.0, "phase_diffusion_D": 0.0,
            "auc": {}, "tpr_at_far": {}}


def _cells(family, levels):
    summaries = {family: {"points": [_point(lv) for lv in levels]}}
    cells = frontier.build_cells(summaries, frontier._cost_anchors(_B2))
    return {c["level"]: c for c in cells}


def test_drift_zero_is_the_honest_schedule_at_zero_cost():
    c = _cells("drift", [0.0])[0.0]
    assert c["cost_overhead_pct"] == 0.0
    assert c["cost_overhead_pct_std"] == 0.0
    assert "honest" in c["cost_anchor_source"]


def test_measured_drift_levels_keep_their_anchor():
    c = _cells("drift", [0.2])[0.2]
    assert c["cost_overhead_pct"] > 0.0
    assert c["cost_anchor_source"] == frontier._ANCHOR_SOURCE["drift"]


def test_unmeasured_level_stays_unpriced():
    c = _cells("drift", [0.3])[0.3]
    assert c["cost_overhead_pct"] is None
