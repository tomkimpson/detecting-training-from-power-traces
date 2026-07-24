"""Rung-2 training-vs-inference classification sweep (Phase 2; plan §3.3).

Evaluates the three decision rules (prespecified physics score / fitted
discriminant / RF learned reference) on ONE physics feature vector, over three
kinds of cell (powerladder.typeb.rung2.rung2_grid + config.Rung2Params):

    stated     the Rung-2 FPR/FNR under the STATED workload population
               (training vs the hard inference null), cross-validated;
    transfer   fit on the nominal population, evaluate ZERO-SHOT when hardware /
               meter / workload parameters leave it (domain shift);
    control    a SEMANTIC falsification control vs a held-out nominal null — the
               paper's central question of what the meter can vs cannot certify.

The rule wiring, feature vector, scenarios and cell grid live in the library
(powerladder.typeb.rung2 / rung2_features / rung2_scenarios); this script only
drives the sweep and persists results.

Each cell is scored independently and crc-seeded, so the sweep runs EITHER as a
Slurm array (SLURM_ARRAY_TASK_ID selects one cell) or single-node (--jobs
multiprocessing.Pool over cells). Results persist immediately, crash-safe:
results/rung2/rung2_raw/<cell>.json (unique per cell, no race) plus the tracked
aggregate results/rung2/rung2_summary.json (merged under an advisory flock; a
torn merge is rebuildable from the raws).

Usage:
    python scripts/rung2_eval.py --jobs 8            # single node, full grid
    SLURM_ARRAY_TASK_ID=0 python scripts/rung2_eval.py   # one cell (array)
    python scripts/rung2_eval.py --smoke             # tiny end-to-end check
                                                     # (-> rung2_smoke_*)
    python scripts/rung2_eval.py --list              # print cells + count
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
import zlib
from collections import namedtuple
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from powerladder.config import DEFAULT, KoTypeBParams, Rung2Params  # noqa: E402
from powerladder.typeb.rung2 import RULES, run_rung2_cell, rung2_grid  # noqa: E402
from powerladder.typeb.rung2_scenarios import CONTROL_ORDER  # noqa: E402

RESULTS_DIR = pathlib.Path(__file__).resolve().parent.parent / "results" / "rung2"

# Disjoint output paths so a --smoke run can never touch the tracked artifact.
OutPaths = namedtuple("OutPaths", ["raw_dir", "summary_path", "lock_path"])
REAL_PATHS = OutPaths(
    raw_dir=RESULTS_DIR / "rung2_raw",
    summary_path=RESULTS_DIR / "rung2_summary.json",
    lock_path=RESULTS_DIR / ".rung2.lock",
)
SMOKE_PATHS = OutPaths(
    raw_dir=RESULTS_DIR / "rung2_smoke_raw",
    summary_path=RESULTS_DIR / "rung2_smoke_summary.json",
    lock_path=RESULTS_DIR / ".rung2_smoke.lock",
)

# The observation channel & generators are fixed inputs (like st2_meter_boundary).
KO = DEFAULT.ko
GLUE = DEFAULT.ko_typeb
MV = DEFAULT.st2.meter_variants


def smoke_params(p: Rung2Params) -> Rung2Params:
    """Tiny n + one transfer shift (the full control set stays, cheaply at n=6)."""
    return dataclasses.replace(p, n_each=6, transfer_shifts=p.transfer_shifts[:1])


def smoke_glue(glue: KoTypeBParams) -> KoTypeBParams:
    """Short-duration glue so the smoke grid runs in ~a minute."""
    return dataclasses.replace(glue, duration_s=90.0)


def _cell_seed(base: int, cell_name: str) -> int:
    """Well-mixed per-cell seed from (base, crc(cell_name)) — order-independent."""
    ss = np.random.SeedSequence([int(base), zlib.crc32(cell_name.encode())])
    return int(ss.generate_state(1)[0])


def summary_skeleton(p: Rung2Params, glue: KoTypeBParams) -> dict:
    """Metadata header shared by every cell (schema echoes the ST2 summaries)."""
    return {
        "sweep": "rung2",
        "generator": "ko_workload training vs hard inference null (plan §3.3)",
        "n_each": p.n_each,
        "target_fars": list(p.target_fars),
        "seed": p.seed,
        "n_folds": p.n_folds,
        "duration_s": glue.duration_s,
        "band": [glue.band_lo, glue.band_hi],
        "rules": list(RULES),
        "transfer_shifts": [name for name, _ in p.transfer_shifts],
        "controls": list(CONTROL_ORDER),
        "environment": (
            "canonical: MATS `compute` partition, single-threaded OpenBLAS. "
            "Marginal AUCs are BLAS-build-conditional; the qualitative "
            "verdicts (decoy scores as training; async training missed by the "
            "physics rule) are not."
        ),
        "cells": [],
    }


def _merge_cell(record: dict, p: Rung2Params, glue: KoTypeBParams,
                paths: OutPaths) -> None:
    """flock-guarded, idempotent merge of one cell record into the summary."""
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
                summary = summary_skeleton(p, glue)
            cells = [c for c in summary["cells"] if c["cell"] != record["cell"]]
            cells.append(record)
            cells.sort(key=lambda c: c["cell"])
            summary["cells"] = cells
            paths.summary_path.write_text(json.dumps(summary, indent=2))
        finally:
            fcntl.flock(lk, fcntl.LOCK_UN)


def _fmt_cell(record: dict) -> str:
    """One-line progress digest (AUCs, plus frac-training for control cells)."""
    aucs = ", ".join(f"{r}={record['rules'][r]['auc']:.2f}" for r in RULES)
    if record["kind"] == "control":
        f0 = {r: record["rules"][r]["frac_training"][
            list(record["rules"][r]["frac_training"])[0]] for r in RULES}
        extra = "  fracTrain " + ", ".join(f"{r}={f0[r]:.2f}" for r in RULES)
        return f"AUC {aucs}{extra}"
    return f"AUC {aucs}"


def _run_and_persist(spec) -> dict:
    """Worker: run one cell, persist its raw JSON + merge into the summary."""
    cell_name, kind, name, p, glue, paths = spec
    t0 = time.time()
    record = run_rung2_cell(kind, name, KO, glue, p, meter_variants=MV,
                            seed=_cell_seed(p.seed, cell_name))
    record["cell"] = cell_name
    paths.raw_dir.mkdir(parents=True, exist_ok=True)
    (paths.raw_dir / f"{cell_name}.json").write_text(json.dumps(record, indent=2))
    _merge_cell(record, p, glue, paths)
    print(f"  cell {cell_name}: {time.time() - t0:.1f}s  {_fmt_cell(record)}",
          flush=True)
    return record


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--jobs", type=int, default=1,
                    help="multiprocessing pool size over cells (single node)")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny grid (n_each=6, 90 s, 1 transfer shift) — "
                         "plumbing check; writes to rung2_smoke_*")
    ap.add_argument("--list", action="store_true",
                    help="print the cell list and count, then exit")
    ap.add_argument("--array-id", type=int, default=None,
                    help="run only cell index N (overrides SLURM_ARRAY_TASK_ID)")
    args = ap.parse_args(argv)

    p = DEFAULT.rung2
    glue = GLUE
    paths = REAL_PATHS
    if args.smoke:
        p = smoke_params(p)
        glue = smoke_glue(glue)
        paths = SMOKE_PATHS
    cells = rung2_grid(p)

    if args.list:
        for i, (name, kind, cond) in enumerate(cells):
            print(f"{i:3d}  {name}")
        print(f"{len(cells)} cells")
        return

    env_id = os.environ.get("SLURM_ARRAY_TASK_ID")
    array_id = args.array_id if args.array_id is not None else (
        int(env_id) if env_id is not None else None)

    if array_id is not None:
        if not 0 <= array_id < len(cells):
            raise SystemExit(f"array id {array_id} out of range [0, {len(cells)})")
        cell_name, kind, cond = cells[array_id]
        print(f"array cell {array_id}/{len(cells)}: {cell_name}")
        _run_and_persist((cell_name, kind, cond, p, glue, paths))
        return

    specs = [(cell_name, kind, cond, p, glue, paths)
             for cell_name, kind, cond in cells]
    print(f"running {len(specs)} cells at n_each={p.n_each} "
          f"(dur {glue.duration_s:g}s) with {args.jobs} job(s)")
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
