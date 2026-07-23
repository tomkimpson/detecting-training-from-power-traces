"""Meter-requirement boundary sweep (Phase 2; spec.md "minimum meter spec").

Grids the observation channel over sample_hz x integ_window_s (plus a notch
depth sub-sweep), HONEST TRAINING ONLY, and records where each detector class
dies as the channel degrades between the nominal 20 Hz meter and the 1 Hz
integrating sampler. The cell grid, level axes and detector wiring live in the
library (powerladder.typeb.meter_boundary + config.St2MeterBoundaryParams); this
script only drives the sweep and persists results.

Each cell is one MeterParams channel, scored independently and crc-seeded, so
the sweep runs EITHER as a Slurm array (SLURM_ARRAY_TASK_ID selects one cell)
or single-node (--jobs multiprocessing.Pool over cells). Results persist
immediately, crash-safe: results/st2/meter_boundary_raw/<cell>.json (unique per
cell, no race) plus the tracked aggregate results/st2/meter_boundary_summary.json
(merged under an advisory flock; a torn merge is rebuildable from the raws).

Usage:
    python scripts/st2_meter_boundary.py --jobs 8          # single node, full grid
    SLURM_ARRAY_TASK_ID=0 python scripts/st2_meter_boundary.py  # one cell (array)
    python scripts/st2_meter_boundary.py --smoke           # tiny end-to-end check
                                                           # (-> meter_boundary_smoke_*)
    python scripts/st2_meter_boundary.py --list            # print cells + count
"""

from __future__ import annotations

import argparse
import dataclasses
import fcntl
import json
import os
import pathlib
import sys
import time
from collections import namedtuple
from multiprocessing import Pool

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from powerladder.config import DEFAULT, St2MeterBoundaryParams  # noqa: E402
from powerladder.typeb.meter_boundary import (  # noqa: E402
    FIXED,
    TRACKING,
    full_detector_set,
    meter_grid,
    run_meter_cell,
)

RESULTS_DIR = pathlib.Path(__file__).resolve().parent.parent / "results" / "st2"

# Output-path bundle: the real sweep and the --smoke plumbing check write to
# DISJOINT locations so a smoke run can never touch the tracked full-grid
# artifact (its truncated grid shares cell names with the real grid).
OutPaths = namedtuple("OutPaths", ["raw_dir", "summary_path", "lock_path"])

REAL_PATHS = OutPaths(
    raw_dir=RESULTS_DIR / "meter_boundary_raw",
    summary_path=RESULTS_DIR / "meter_boundary_summary.json",
    lock_path=RESULTS_DIR / ".meter_boundary.lock",
)
SMOKE_PATHS = OutPaths(
    raw_dir=RESULTS_DIR / "meter_boundary_smoke_raw",
    summary_path=RESULTS_DIR / "meter_boundary_smoke_summary.json",
    lock_path=RESULTS_DIR / ".meter_boundary_smoke.lock",
)


def smoke_params(p: St2MeterBoundaryParams) -> St2MeterBoundaryParams:
    """Truncate every grid to a 2x2 main block + one notch cell at tiny n."""
    return dataclasses.replace(
        p, n_each=8,
        sample_hz_grid=p.sample_hz_grid[:1] + p.sample_hz_grid[-1:],
        integ_window_grid=p.integ_window_grid[:1] + p.integ_window_grid[-1:],
        notch_hz_grid=p.notch_hz_grid[2:3],   # 1.0 Hz (band centre)
        notch_depth_grid=p.notch_depth_grid[-1:],
    )


def summary_skeleton(p: St2MeterBoundaryParams) -> dict:
    """The metadata header shared by every cell (schema echoes the per-family
    ST2 summaries)."""
    return {
        "sweep": "meter_boundary",
        "generator": "ko_workload honest training vs hard inference null",
        "n_each": p.n_each,
        "target_fars": list(p.target_fars),
        "seed": p.seed,
        "sigma_eta": p.sigma_eta,
        "detector_classes": {"tracking": list(TRACKING), "fixed": list(FIXED)},
        "detectors": sorted(full_detector_set()),
        "environment": (
            "canonical: MATS `compute` partition, single-threaded OpenBLAS. "
            "Marginal/dead-cell AUCs are BLAS-build-conditional (up to ~0.3 "
            "swing on a different OpenBLAS at the same seed); the live-region "
            "boundary (the minimum meter specification) is not."
        ),
        "grid": {
            "sample_hz_grid": list(p.sample_hz_grid),
            "integ_window_grid": list(p.integ_window_grid),
            "notch_hz_grid": list(p.notch_hz_grid),
            "notch_depth_grid": list(p.notch_depth_grid),
            "notch_q": p.notch_q,
        },
        "cells": [],
    }


def _merge_cell(record: dict, p: St2MeterBoundaryParams, paths: OutPaths) -> None:
    """flock-guarded merge of one cell record into the summary.

    The summary is a deterministic aggregate keyed by cell name — merging is
    idempotent (a re-run replaces the cell in place), so a crashed array leaves
    a rebuildable partial file.

    Refuses to merge into a summary whose header ``n_each`` disagrees with this
    run's: cell values are only comparable at a common ``n_each``, so a mixed-n
    merge would silently corrupt the aggregate under an inconsistent header.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(paths.lock_path, "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        try:
            if paths.summary_path.exists():
                summary = json.loads(paths.summary_path.read_text())
                if summary.get("n_each") != p.n_each:
                    raise SystemExit(
                        f"refusing to merge into {paths.summary_path.name}: "
                        f"existing n_each={summary.get('n_each')} != run "
                        f"n_each={p.n_each} (mixed-n merge would corrupt it)")
            else:
                summary = summary_skeleton(p)
            cells = [c for c in summary["cells"] if c["cell"] != record["cell"]]
            cells.append(record)
            cells.sort(key=lambda c: c["cell"])
            summary["cells"] = cells
            paths.summary_path.write_text(json.dumps(summary, indent=2))
        finally:
            fcntl.flock(lk, fcntl.LOCK_UN)


def _run_and_persist(spec) -> dict:
    """Worker: run one cell, persist its raw JSON + merge into the summary."""
    cell_name, mp, p, paths = spec
    t0 = time.time()
    record = run_meter_cell(cell_name, mp, p, DEFAULT.ko, DEFAULT.ko_typeb)
    paths.raw_dir.mkdir(parents=True, exist_ok=True)
    (paths.raw_dir / f"{cell_name}.json").write_text(json.dumps(record, indent=2))
    _merge_cell(record, p, paths)
    tpr = {det: per_far[f"{p.target_fars[0]:g}"]
           for det, per_far in record["tpr_at_far"].items()}
    print(f"  cell {cell_name}: {time.time() - t0:.1f}s  "
          f"TPR@{p.target_fars[0]:g} " +
          ", ".join(f"{k}={tpr[k]:.2f}" for k in sorted(tpr)), flush=True)
    return record


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--jobs", type=int, default=1,
                    help="multiprocessing pool size over cells (single node)")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny grid (2x2 + 1 notch, n_each=8) — plumbing check; "
                         "writes to meter_boundary_smoke_* (never the tracked "
                         "full-grid summary)")
    ap.add_argument("--list", action="store_true",
                    help="print the cell list and count, then exit")
    ap.add_argument("--array-id", type=int, default=None,
                    help="run only cell index N (overrides SLURM_ARRAY_TASK_ID)")
    args = ap.parse_args(argv)

    p = DEFAULT.st2_meter_boundary
    paths = REAL_PATHS
    if args.smoke:
        p = smoke_params(p)
        paths = SMOKE_PATHS
    cells = meter_grid(p)

    if args.list:
        for i, (name, mp) in enumerate(cells):
            print(f"{i:3d}  {name}")
        print(f"{len(cells)} cells")
        return

    # Array mode: run exactly one cell selected by --array-id or the Slurm env.
    env_id = os.environ.get("SLURM_ARRAY_TASK_ID")
    array_id = args.array_id if args.array_id is not None else (
        int(env_id) if env_id is not None else None)

    if array_id is not None:
        if not 0 <= array_id < len(cells):
            raise SystemExit(f"array id {array_id} out of range "
                             f"[0, {len(cells)})")
        name, mp = cells[array_id]
        print(f"array cell {array_id}/{len(cells)}: {name}")
        _run_and_persist((name, mp, p, paths))
        return

    specs = [(name, mp, p, paths) for name, mp in cells]
    print(f"running {len(specs)} cells at n_each={p.n_each} "
          f"with {args.jobs} job(s)")
    t0 = time.time()
    if args.jobs > 1:
        with Pool(args.jobs) as pool:
            pool.map(_run_and_persist, specs)
    else:
        for s in specs:
            _run_and_persist(s)
    print(f"grid done in {time.time() - t0:.1f}s; wrote {paths.summary_path}")


if __name__ == "__main__":
    main()
