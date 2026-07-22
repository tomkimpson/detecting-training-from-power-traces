"""ST1 stage-8 one-at-a-time sweeps (task 19.9b): the swept open choices.

Each axis perturbs ONE knob of St1DetectorParams around the reference cells —
stage4 x ar1 (the coloured-null conservatism cell, the fix-or-reframe axis)
and stage1 x white (the clean anchor whose calibration must not be broken) —
and reruns the realised-FAR cell at M traces/point. The axes are the open
choices the plan carries explicitly (swept, not decided ex ante):

    duration  duration_s in {75, 150, 300, 600} s      x both reference cells
    lags      lag_set {(0,1,2), default, dense 0..12}  x both reference cells
    cov       bartlett: shrinkage {0, .01, .05} x b {6, 18, 54};
              batch_means: shrinkage {0, .01, .05}
                     x {stage4 x ar1, stage4 x white, stage1 x white}
              — THE decision axis: is there a (shrinkage, b) that brings
              coloured-null FAR into [1/3x, 3x] at 1e-2 without inflating
              white? (stage4 x white added beyond the two reference cells
              because that is where the resampling deflation lives, and a
              candidate GO-analytic configuration must hold there too.)
    nperseg   tracker_nperseg_factor {0.5, 1, 2}       x stage4 x ar1
    band      search-band width {half, nominal, double} x stage4 x ar1
    interp    resample_interp {linear, cubic, sinc}    x {stage2 x white,
              stage4 x white} — the 1->2 deflation was isolated on white, so
              white is where an interp effect is readable (ar1 sits at
              FAR ~ 0 for every interp; a null result there is uninformative)

PRE-REGISTERED SELECTION RULE (publication-bias guard, notes/results/st1-findings.md):
any configuration promoted from these sweeps is chosen on NULL CALIBRATION
QUALITY ACROSS THE NULL SUITE ONLY — never on signal power. Power enters only
afterwards, in the 19.10 bake-off, at whatever configuration the calibration
already fixed.

Usage:
    python scripts/st1_sweeps.py --jobs 8                 # all axes, M=5000
    python scripts/st1_sweeps.py --axes cov --M 5000
Writes results/st1/sweeps/{axis}.json (one file per axis).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
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
from powerladder.st1.nulls import NULLS  # noqa: E402
from powerladder.st1.pipeline import (KO_NOMINAL_F0_HZ, stage1_fixed_alpha,  # noqa: E402
                               stage2_known_phase, stage3_heldout_phase,
                               stage4_full_adaptive)
from powerladder.st1.resample import random_smooth_phase_path  # noqa: E402

SWEEP_DIR = pathlib.Path(__file__).resolve().parent.parent / "results" / "st1" / "sweeps"

# Reference cells (stage, null): the coloured-conservatism cell and the clean
# anchor (see module docstring).
REF_COLOURED = ("stage4", "ar1")
REF_ANCHOR = ("stage1", "white")

# Stage-2 known-warp scale — matches scripts/st1_far.py (the pre-registered
# wander-regime drift).
STAGE2_DRIFT_HZ = 0.2

# Axis grids (documented here, not magic): the default operating point is
# always included so every sweep contains the chunk-2 reference row.
DURATIONS_S = (75.0, 150.0, 300.0, 600.0)
LAG_SETS = {
    "short": (0, 1, 2),
    "default": (0, 1, 2, 3, 5, 8),
    "dense": tuple(range(13)),
}
COV_SHRINKAGES = (0.0, 0.01, 0.05)
COV_BANDWIDTHS = (6, 18, 54)          # b=18 ~ floor(N**1/3) at N ~ 6000, the default
NPERSEG_FACTORS = (0.5, 1.0, 2.0)
BAND_WIDTH_FACTORS = {"half": 0.5, "nominal": 1.0, "double": 2.0}
BAND_LO_CLIP_HZ = 0.1                 # doubled band would cross 0; clip above DC
INTERPS = ("linear", "cubic", "sinc")


class _Stage2KnownWarp:
    """Fresh independent known warp per replicate (mirrors scripts/st1_far.py)."""

    def __init__(self, params, seed):
        self._params = params
        self._rng = np.random.default_rng(seed)

    def __call__(self, t, p_obs, band_lo, band_hi) -> float:
        _, theta = random_smooth_phase_path(
            t, self._rng, drift_hz=STAGE2_DRIFT_HZ,
            floor_hz=self._params.f_path_floor_hz)
        return stage2_known_phase(t, p_obs, band_lo, band_hi,
                                  params=self._params, theta_true=theta)


def _detector(stage: str, p_det, warp_seed):
    return {
        "stage1": partial(stage1_fixed_alpha, params=p_det,
                          alpha_hz=KO_NOMINAL_F0_HZ),
        "stage2": _Stage2KnownWarp(p_det, warp_seed),
        "stage3": partial(stage3_heldout_phase, params=p_det),
        "stage4": partial(stage4_full_adaptive, params=p_det),
    }[stage]


def _points_duration():
    for stage, null in (REF_COLOURED, REF_ANCHOR):
        for dur in DURATIONS_S:
            yield (f"{stage}/{null}/dur{dur:g}", stage, null,
                   {"duration_s": dur})


def _points_lags():
    for stage, null in (REF_COLOURED, REF_ANCHOR):
        for name, lags in LAG_SETS.items():
            yield (f"{stage}/{null}/lags_{name}", stage, null,
                   {"lag_set": lags})


def _points_cov():
    cells = (REF_COLOURED, ("stage4", "white"), REF_ANCHOR)
    for stage, null in cells:
        for lam in COV_SHRINKAGES:
            for b in COV_BANDWIDTHS:
                yield (f"{stage}/{null}/bartlett_s{lam:g}_b{b}", stage, null,
                       {"cov_estimator": "bartlett", "cov_shrinkage": lam,
                        "cov_bandwidth_b": b})
            yield (f"{stage}/{null}/batch_means_s{lam:g}", stage, null,
                   {"cov_estimator": "batch_means", "cov_shrinkage": lam})


def _points_nperseg():
    stage, null = REF_COLOURED
    for f in NPERSEG_FACTORS:
        yield (f"{stage}/{null}/nperseg{f:g}", stage, null,
               {"tracker_nperseg_factor": f})


def _points_band():
    stage, null = REF_COLOURED
    centre = 0.5 * (DEFAULT.st1.band_lo + DEFAULT.st1.band_hi)
    width = DEFAULT.st1.band_hi - DEFAULT.st1.band_lo
    for name, wf in BAND_WIDTH_FACTORS.items():
        lo = max(BAND_LO_CLIP_HZ, centre - 0.5 * wf * width)
        hi = centre + 0.5 * wf * width
        yield (f"{stage}/{null}/band_{name}", stage, null,
               {"band_lo": lo, "band_hi": hi})


def _points_interp():
    for stage, null in (("stage2", "white"), ("stage4", "white")):
        for interp in INTERPS:
            yield (f"{stage}/{null}/interp_{interp}", stage, null,
                   {"resample_interp": interp})


AXES = {
    "duration": _points_duration,
    "lags": _points_lags,
    "cov": _points_cov,
    "nperseg": _points_nperseg,
    "band": _points_band,
    "interp": _points_interp,
}


def _run_point(spec):
    """Worker: one sweep point = one FAR cell at perturbed params."""
    axis, label, stage, null_name, delta, M, base_seed = spec
    p_det = dataclasses.replace(DEFAULT.st1, **delta)
    crc = zlib.crc32(f"sweep/{axis}/{label}".encode())
    detector = _detector(stage, p_det, [base_seed, crc, 1])
    n = int(round(p_det.duration_s * p_det.fs)) + 1
    t0 = time.time()
    cell = far.run_cell(
        detector, NULLS[null_name], M, [base_seed, crc],
        stage=stage, null_name=null_name, calib=label,
        n=n, fs=p_det.fs, null_params=DEFAULT.st1_null,
        band_lo=p_det.band_lo, band_hi=p_det.band_hi,
        nominal_levels=DEFAULT.st1_far.nominal_levels,
        ci_conf=DEFAULT.st1_far.ci_conf,
        phash=far.params_hash(p_det, DEFAULT.st1_null),
    )
    print(f"  {axis}: {label} M={M} in {time.time()-t0:.1f} s "
          f"(cov_failures={cell.cov_failures})", flush=True)
    rec = {
        "label": label, "stage": stage, "null": null_name,
        "params_delta": {k: (list(v) if isinstance(v, tuple) else v)
                         for k, v in delta.items()},
        "M": M,
        "levels": {f"{lvl:g}": r for lvl, r in sorted(cell.levels.items(),
                                                      reverse=True)},
        "ks": {"stat": cell.ks_stat, "pvalue": cell.ks_pvalue},
        "cov_failures": cell.cov_failures,
        "params_hash": cell.params_hash,
    }
    return axis, rec


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--axes", default=",".join(AXES),
                    help=f"comma list from {sorted(AXES)}")
    ap.add_argument("--M", type=int, default=5000,
                    help="null traces per sweep point")
    ap.add_argument("--seed", type=int, default=DEFAULT.st1_far.seed)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--smoke", action="store_true", help="M=50 plumbing check")
    args = ap.parse_args(argv)

    axes = [a.strip() for a in args.axes.split(",") if a.strip()]
    unknown = [a for a in axes if a not in AXES]
    if unknown:
        raise SystemExit(f"unknown axes {unknown}; choose from {sorted(AXES)}")
    M = 50 if args.smoke else args.M

    specs = []
    for axis in axes:
        for label, stage, null, delta in AXES[axis]():
            specs.append((axis, label, stage, null, delta, M, args.seed))
    print(f"running {len(specs)} sweep points at M={M} with {args.jobs} job(s)")

    t0 = time.time()
    if args.jobs > 1:
        with Pool(args.jobs) as pool:
            results = pool.map(_run_point, specs)
    else:
        results = [_run_point(s) for s in specs]
    print(f"sweeps done in {time.time()-t0:.1f} s")

    SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    for axis in axes:
        recs = [rec for ax, rec in results if ax == axis]
        out = SWEEP_DIR / f"{axis}.json"
        out.write_text(json.dumps(
            {"axis": axis, "M": M, "seed": args.seed,
             "selection_rule": ("configurations are promoted on null "
                                "calibration quality across the null suite "
                                "ONLY, never on signal power"),
             "points": recs}, indent=2) + "\n")
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
