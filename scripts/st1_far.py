"""ST1 realised-FAR grid: null calibration tables for the gate (tasks 19.4+).

Runs (stage x null x calibration) cells of the ST1 FAR harness and writes
results/st1/raw/{stage}_{null}_{calib}.npz (raw p-values; gitignored) plus the
tracked aggregate results/st1/far_summary.json.

Stages (the ST1 ladder — each adds one adaptive ingredient):
    stage1  fixed alpha at the nominal Ko cadence (time domain)
    stage2  angle resampling under a KNOWN, independently drawn warp
            (per-replicate OU wander at the pre-registered 0.2 Hz scale)
    stage3  warp estimated on held-out EST blocks, DG scored on TEST only
    stage4  fully adaptive (full-trace Viterbi; no splitting)

Calibration arms: dg_bartlett / dg_batch (the two long-run covariance
estimators; asymptotic chi2 tail), mtf (stage 1 only), and surrogate
(task 19.8: full-pipeline Fourier-phase surrogate p-values; M capped at
St1FarParams.n_null_surrogate_cells, levels meaningful at 0.05/1e-2 only).

The three pre-registered deltas read off this grid (notes/results/st1-findings.md):
resampling-alone (stage 1->2), estimated-warp (2->3), path-selection (3->4).

Usage:
    python scripts/st1_far.py --stages stage1 --nulls white,ar1,ar2_resonant \
        --M 10000 --jobs 8
    python scripts/st1_far.py --stages stage2,stage3,stage4 --calibs dg_bartlett \
        --M 10000 --jobs 8
    python scripts/st1_far.py --smoke        # tiny end-to-end check

Slurm array (one cell per task): --list prints the grid + count; --array-id N
runs only cell N (falls back to SLURM_ARRAY_TASK_ID). See
scripts/slurm/st1_far_surrogate.sbatch.
"""

from __future__ import annotations

import argparse
import dataclasses
import fcntl
import os
import pathlib
import sys
import time
import zlib
from functools import partial
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from powerladder.config import DEFAULT  # noqa: E402
from powerladder.st1 import far  # noqa: E402
from powerladder.st1.multitaper import f_test_statistic  # noqa: E402
from powerladder.st1.nulls import NULLS  # noqa: E402
from powerladder.st1.pipeline import (KO_NOMINAL_F0_HZ, stage1_fixed_alpha,  # noqa: E402
                               stage2_known_phase, stage3_heldout_phase,
                               stage4_full_adaptive, stage4_semicoherent)
from powerladder.st1.resample import random_smooth_phase_path  # noqa: E402
from powerladder.st1.surrogates import surrogate_pvalue  # noqa: E402

RESULTS_DIR = pathlib.Path(__file__).resolve().parent.parent / "results" / "st1"

# Calibration arms per stage. dg_bartlett / dg_batch = the two long-run
# covariance estimators (asymptotic chi2 tail); mtf only has a stage-1 (fixed
# search) form; "surrogate" = full-pipeline Fourier-phase surrogate p-values
# (task 19.8) — available for the adaptive stages, opt-in via --calibs.
STAGE_CALIBS = {
    "stage1": ("dg_bartlett", "dg_batch", "mtf"),
    "stage2": ("dg_bartlett", "dg_batch"),
    "stage3": ("dg_bartlett", "dg_batch"),
    "stage4": ("dg_bartlett", "dg_batch"),
    # task 19.10: the semi-coherent per-block Fisher variant's own null level
    # (its Fisher chi2(2K) combination assumes independent blocks that the
    # shared full-trace tracked path couples -- measured, not assumed).
    "stage4_semicoh": ("dg_bartlett",),
}

# Stage-2 known-warp scale: the pre-registered wander-regime drift
# (notes/results/st1-findings.md GO-reframe criterion, drift >~ 0.2 Hz).
STAGE2_DRIFT_HZ = 0.2


class _Stage2KnownWarp:
    """Stage-2 harness arm: a fresh, independent known warp per replicate.

    Under the null the trace has no line, so the warp is just a random time
    change — drawn from this object's OWN spawned rng stream (independent of
    the trace stream) so the (trace, warp) pairs are reproducible per cell.
    """

    def __init__(self, params, seed):
        self._params = params
        self._rng = np.random.default_rng(seed)

    def __call__(self, t, p_obs, band_lo, band_hi) -> float:
        _, theta = random_smooth_phase_path(
            t, self._rng, drift_hz=STAGE2_DRIFT_HZ,
            floor_hz=self._params.f_path_floor_hz)
        return stage2_known_phase(t, p_obs, band_lo, band_hi,
                                  params=self._params, theta_true=theta)


class _SurrogateCalibrated:
    """Full-pipeline surrogate calibration: score = -log10 surrogate p.

    The wrapped detector (tracker included) runs inside EVERY surrogate, so
    the selection steps are part of the calibrated null. Fourier-phase
    surrogates: the invariance null is a linear stationary process with the
    observed PSD (see code.st1.surrogates).
    """

    def __init__(self, detector_pfn, seed, n_surrogates):
        self._det = detector_pfn
        self._rng = np.random.default_rng(seed)
        self._S = n_surrogates

    def __call__(self, t, p_obs, band_lo, band_hi) -> float:
        p = surrogate_pvalue(
            lambda tt, xx: self._det(tt, xx, band_lo, band_hi),
            t, p_obs, self._rng, n_surrogates=self._S)
        return float(-np.log10(p))


def _detector_for(stage: str, calib: str, warp_seed, n_surrogates: int,
                  det_overrides: dict | None = None):
    """Build the (t, P_obs, band_lo, band_hi) -> -log10 p callable for one cell.

    ``det_overrides`` are St1DetectorParams replacements applied on top of the
    calib's estimator choice — used to re-run the grid at a candidate
    configuration from the stage-8 sweeps (--cov-shrinkage/--cov-bandwidth-b),
    with the cell RECORDED under a tagged calib name so the default-config
    table is never overwritten.
    """
    p_det = DEFAULT.st1
    if calib.split("+")[0] in ("dg_bartlett", "dg_batch", "surrogate"):
        est = "batch_means" if calib.split("+")[0] == "dg_batch" else "bartlett"
        p_det = dataclasses.replace(p_det, cov_estimator=est,
                                    **(det_overrides or {}))
        base = {
            "stage1": partial(stage1_fixed_alpha, params=p_det,
                              alpha_hz=KO_NOMINAL_F0_HZ),
            "stage2": _Stage2KnownWarp(p_det, warp_seed),
            "stage3": partial(stage3_heldout_phase, params=p_det),
            "stage4": partial(stage4_full_adaptive, params=p_det),
            "stage4_semicoh": partial(stage4_semicoherent, params=p_det),
        }[stage]
        if calib.split("+")[0] == "surrogate":
            if stage == "stage2":
                raise SystemExit("surrogate calibration for stage2 is not "
                                 "supported (the known warp must stay fixed "
                                 "across surrogates; run stages 1/3/4)")
            return p_det, _SurrogateCalibrated(base, warp_seed, n_surrogates)
        return p_det, base
    if calib == "mtf":
        if stage != "stage1":
            raise SystemExit("mtf is a stage-1 (fixed-search) arm only")
        return p_det, partial(f_test_statistic, params=p_det)
    raise SystemExit(f"unknown calibration arm {calib!r}")


def _run_one(spec) -> far.CellResult:
    """Worker: run one (stage, null, calib) cell. Top-level for picklability."""
    stage, null_name, calib, M, base_seed, n_surrogates, det_overrides = spec
    # Stable per-cell seed spec: base seed + a CRC of the cell name MINUS any
    # +tag suffix, so a tagged (candidate-configuration) cell scores the SAME
    # trace stream as its default-config counterpart — the comparison is then
    # paired, not two independent Monte Carlo draws. The warp / surrogate
    # stream gets a distinct child (trailing 1).
    crc = zlib.crc32(f"{stage}/{null_name}/{calib.split('+')[0]}".encode())
    cell_seed = [base_seed, crc]
    p_det, detector = _detector_for(stage, calib, [base_seed, crc, 1],
                                    n_surrogates, det_overrides)
    p_null = DEFAULT.st1_null
    n = int(round(p_det.duration_s * p_det.fs)) + 1
    t0 = time.time()
    cell = far.run_cell(
        detector, NULLS[null_name], M, cell_seed,
        stage=stage, null_name=null_name, calib=calib,
        n=n, fs=p_det.fs, null_params=p_null,
        band_lo=p_det.band_lo, band_hi=p_det.band_hi,
        nominal_levels=DEFAULT.st1_far.nominal_levels,
        ci_conf=DEFAULT.st1_far.ci_conf,
        phash=far.params_hash(p_det, p_null),
    )
    print(f"  cell {stage}/{null_name}/{calib}: M={M} in {time.time()-t0:.1f} s "
          f"(cov_failures={cell.cov_failures})", flush=True)
    # Persist IMMEDIATELY (crash safety): long grids can be killed mid-run and
    # a completed cell must survive. The raw npz is unique per cell (no race);
    # the shared far_summary.json merge is serialised across workers with an
    # advisory flock. The summary stays a deterministic aggregate of the raws,
    # so even a torn merge is re-buildable from the npz files.
    far.save_cell_raw(cell, RESULTS_DIR)
    lock_path = RESULTS_DIR / ".far_summary.lock"
    with open(lock_path, "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        try:
            far.update_summary(cell, RESULTS_DIR / "far_summary.json")
        finally:
            fcntl.flock(lk, fcntl.LOCK_UN)
    return cell


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stages", default="stage1",
                    help=f"comma list from {sorted(STAGE_CALIBS)}")
    ap.add_argument("--nulls", default="white,ar1,ar2_resonant",
                    help=f"comma list from {sorted(NULLS)}")
    ap.add_argument("--calibs", default="",
                    help="comma list; empty = the per-stage defaults "
                         "(dg_bartlett,dg_batch[,mtf]); 'surrogate' opt-in")
    ap.add_argument("--M", type=int, default=DEFAULT.st1_far.n_null,
                    help="null traces per cell (surrogate cells are capped "
                         f"at {DEFAULT.st1_far.n_null_surrogate_cells})")
    ap.add_argument("--n-surrogates", type=int,
                    default=DEFAULT.st1_far.n_surrogates,
                    help="S per trace for surrogate-calibrated cells")
    ap.add_argument("--seed", type=int, default=DEFAULT.st1_far.seed)
    ap.add_argument("--jobs", type=int, default=1,
                    help="multiprocessing pool size over cells")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny run (M=50, S=19) to exercise the plumbing")
    ap.add_argument("--cov-shrinkage", type=float, default=None,
                    help="override St1DetectorParams.cov_shrinkage (candidate-"
                         "configuration re-run; requires --tag)")
    ap.add_argument("--cov-bandwidth-b", type=int, default=None,
                    help="override St1DetectorParams.cov_bandwidth_b (explicit "
                         "Bartlett lag bandwidth; requires --tag)")
    ap.add_argument("--tag", default="",
                    help="suffix recorded as calib+tag so an overridden "
                         "configuration never overwrites the default table")
    ap.add_argument("--list", action="store_true",
                    help="print the (stage/null/calib) cell list and count, "
                         "then exit (use to size a Slurm --array range)")
    ap.add_argument("--array-id", type=int, default=None,
                    help="run only cell index N of the built grid (overrides "
                         "SLURM_ARRAY_TASK_ID) — one cell per Slurm array task")
    args = ap.parse_args(argv)

    det_overrides = {}
    if args.cov_shrinkage is not None:
        det_overrides["cov_shrinkage"] = args.cov_shrinkage
    if args.cov_bandwidth_b is not None:
        det_overrides["cov_bandwidth_b"] = args.cov_bandwidth_b
    if det_overrides and not args.tag:
        raise SystemExit("--cov-* overrides require --tag (so the candidate "
                         "configuration is recorded under its own calib name)")

    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    nulls = [s.strip() for s in args.nulls.split(",") if s.strip()]
    M = 50 if args.smoke else args.M
    n_surr = 19 if args.smoke else args.n_surrogates

    unknown = [s for s in stages if s not in STAGE_CALIBS]
    if unknown:
        raise SystemExit(f"unknown stages {unknown}; "
                         f"choose from {sorted(STAGE_CALIBS)}")
    unknown = [n_ for n_ in nulls if n_ not in NULLS]
    if unknown:
        raise SystemExit(f"unknown nulls {unknown}; choose from {sorted(NULLS)}")

    specs = []
    for stage in stages:
        calibs = ([c.strip() for c in args.calibs.split(",") if c.strip()]
                  or STAGE_CALIBS[stage])
        for null_name in nulls:
            for calib in calibs:
                m_cell = M
                if calib == "surrogate":
                    m_cell = min(M, DEFAULT.st1_far.n_null_surrogate_cells)
                if args.tag:
                    calib = f"{calib}+{args.tag}"
                specs.append((stage, null_name, calib, m_cell, args.seed,
                              n_surr, det_overrides))

    if args.list:
        for i, spec in enumerate(specs):
            stage, null_name, calib, m_cell = spec[0], spec[1], spec[2], spec[3]
            print(f"{i:3d}  {stage}/{null_name}/{calib}  M={m_cell}")
        print(f"{len(specs)} cells")
        return

    # Array mode: run exactly one cell selected by --array-id or the Slurm env.
    # Each cell persists its own raw npz + flock-merges far_summary.json, so
    # concurrent array tasks are safe (no shared write races).
    env_id = os.environ.get("SLURM_ARRAY_TASK_ID")
    array_id = args.array_id if args.array_id is not None else (
        int(env_id) if env_id is not None else None)
    if array_id is not None:
        if not 0 <= array_id < len(specs):
            raise SystemExit(f"array id {array_id} out of range "
                             f"[0, {len(specs)})")
        spec = specs[array_id]
        print(f"array cell {array_id}/{len(specs)}: "
              f"{spec[0]}/{spec[1]}/{spec[2]}  M={spec[3]}")
        specs = [spec]

    print(f"running {len(specs)} cells at M={M} with {args.jobs} job(s)")

    t0 = time.time()
    if args.jobs > 1:
        with Pool(args.jobs) as pool:
            cells = pool.map(_run_one, specs)
    else:
        cells = [_run_one(s) for s in specs]
    print(f"grid done in {time.time()-t0:.1f} s")

    summary_path = RESULTS_DIR / "far_summary.json"
    for cell in cells:
        far.save_cell_raw(cell, RESULTS_DIR)
        far.update_summary(cell, summary_path)
    print(f"wrote {summary_path}")

    # Console table: realised FAR vs nominal per cell.
    for cell in cells:
        rows = ", ".join(
            f"{lvl:g}: {rec['far']:.4g} [{rec['ci'][0]:.4g}, {rec['ci'][1]:.4g}]"
            for lvl, rec in sorted(cell.levels.items(), reverse=True)
        )
        print(f"{cell.stage}/{cell.null}/{cell.calib}  {rows}  "
              f"KS p={cell.ks_pvalue:.3g}  cov_fail={cell.cov_failures}")


if __name__ == "__main__":
    main()
