"""Schema check for the ST2 sweep script's per-family summaries — task 20.7."""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import pathlib

from powerladder.config import DEFAULT

_SCRIPT = (pathlib.Path(__file__).resolve().parent.parent
           / "scripts" / "plot_st2_sweeps.py")
_spec = importlib.util.spec_from_file_location("plot_st2_sweeps", _SCRIPT)
sweeps = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sweeps)


def test_summary_schema_smoke(tmp_path):
    p = sweeps.smoke_params(dataclasses.replace(DEFAULT.st2, seed=5))
    returned = sweeps.run_and_write("work", p, results_dir=tmp_path,
                                    fig_dir=tmp_path)

    out = tmp_path / "work_summary.json"
    assert out.exists()
    d = json.loads(out.read_text())
    assert d == returned

    # params block (echoes the results/b1/gate_summary.json style)
    for key in ("family", "generator", "budget_units", "cost_anchor",
                "n_each", "target_fars", "seed", "meter", "detectors",
                "levels", "points"):
        assert key in d, key
    assert d["family"] == "work"
    assert d["cost_anchor"] == "results/b2/workjitter_summary.json"
    assert d["n_each"] == 8 and d["target_fars"] == [0.05, 0.01]
    # default detector set is "full" since task 20.9 (ST1 detectors registered)
    assert d["detectors"] == ["dg_order_full", "dg_order_semicoh", "mtf",
                              "spectral", "viterbi"]
    assert d["levels"] == list(p.work_levels)

    # per-level entries carry the frontier fields
    assert len(d["points"]) == len(p.work_levels)
    for pt, level in zip(d["points"], p.work_levels):
        assert pt["level"] == level
        assert isinstance(pt["cadence_cv"], float)
        assert isinstance(pt["phase_diffusion_D"], float)
        for det in d["detectors"]:
            assert 0.0 <= pt["auc"][det] <= 1.0
            assert set(pt["tpr_at_far"][det]) == {"0.05", "0.01"}

    # figure pair written alongside
    assert (tmp_path / "st2_work.pdf").exists()
    assert (tmp_path / "st2_work.png").exists()
