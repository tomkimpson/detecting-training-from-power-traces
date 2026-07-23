"""Meter-requirement boundary sweep tests (Phase 2).

Tiny-n smokes prove the grid builder and the per-cell runner are well-formed;
one slower anchor test reproduces the committed ST2 finding INSIDE the grid —
the honest 20 Hz reference channel keeps the tracked order detector at full
power while the 1 Hz integrating sampler collapses it toward chance — so the
sweep is measuring the right thing.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from powerladder.config import DEFAULT, MeterParams, St2MeterBoundaryParams
from powerladder.typeb.meter_boundary import (
    FIXED,
    TRACKING,
    full_detector_set,
    meter_grid,
    run_meter_cell,
)

KO = DEFAULT.ko
# 60 s glue keeps the schema/smoke runs fast; the anchor test uses a longer
# window so the multitaper/DG detectors operate in their designed regime.
GLUE = dataclasses.replace(DEFAULT.ko_typeb, duration_s=60.0)

SMALL = St2MeterBoundaryParams(
    n_each=6,
    sample_hz_grid=(20.0, 1.0),
    integ_window_grid=(0.0, 1.0),
    notch_hz_grid=(1.0,),
    notch_depth_grid=(1.0,),
)


def test_full_detector_set_is_the_five_frontier_detectors():
    dets = full_detector_set()
    assert set(dets) == set(TRACKING) | set(FIXED)
    assert set(TRACKING).isdisjoint(FIXED)


def test_meter_grid_schema_and_uniqueness():
    cells = meter_grid(SMALL)
    names = [name for name, _ in cells]
    assert len(names) == len(set(names))                # unique cell names
    # 2x2 main grid + 1x1 notch sub-sweep
    assert len(cells) == 4 + 1
    main = [mp for name, mp in cells if name.startswith("main_")]
    notch = [mp for name, mp in cells if name.startswith("notch_")]
    assert len(main) == 4 and len(notch) == 1
    # the honest reference channel (20 Hz, no integration, no notch) is present
    assert any(mp.sample_hz == 20.0 and mp.integ_window_s == 0.0
               and mp.notch_hz is None for mp in main)
    # every cell carries the meter-owned noise and the notch cell its blend knob
    assert all(mp.sigma_eta == SMALL.sigma_eta for _, mp in cells)
    assert notch[0].notch_hz == 1.0 and notch[0].notch_depth == 1.0
    assert notch[0].notch_q == SMALL.notch_q


def test_run_meter_cell_smoke_well_formed_and_deterministic():
    name, mp = meter_grid(SMALL)[0]
    a = run_meter_cell(name, mp, SMALL, KO, GLUE)
    b = run_meter_cell(name, mp, SMALL, KO, GLUE)
    assert a == b                                        # pure function of seed
    assert a["cell"] == name
    assert set(a["auc"]) == set(TRACKING) | set(FIXED)
    for det, auc in a["auc"].items():
        assert 0.0 <= auc <= 1.0
        assert set(a["tpr_at_far"][det]) == {"0.05", "0.01"}
        for v in a["tpr_at_far"][det].values():
            assert 0.0 <= v <= 1.0


def test_reference_beats_integrating_1hz_for_tracked_order_detector():
    """The committed anchor, reproduced inside the grid: honest 20 Hz keeps the
    DG order detector at full power; the 1 Hz integrating sampler kills it."""
    glue = dataclasses.replace(DEFAULT.ko_typeb, duration_s=180.0)
    p = dataclasses.replace(SMALL, n_each=16)
    reference = MeterParams(sample_hz=20.0, integ_window_s=0.0,
                            sigma_eta=p.sigma_eta)
    integrating = MeterParams(sample_hz=1.0, integ_window_s=1.0,
                              sigma_eta=p.sigma_eta)   # == integrating_1hz
    ref = run_meter_cell("reference", reference, p, KO, glue)
    itg = run_meter_cell("integrating_1hz", integrating, p, KO, glue)

    det = "dg_order_full"
    assert ref["auc"][det] > 0.8                          # near full power
    assert itg["auc"][det] < ref["auc"][det]              # the channel killed it
    assert itg["auc"][det] < 0.8
