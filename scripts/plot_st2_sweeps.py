"""ST2 per-family sweeps — detector power vs attack level, all nine families.

Task 20.7. For each attack family in the ST2 registry
(code.typeb.st2_attacks) this runs the sweep harness (code.typeb.st2) over the
St2Params level grid: n_each traces per class per level, negatives generated
once per family, TPR at both operational FARs, plus the ground-truth physical
measures (cadence CV, phase diffusion D) the frontier plots on (task 20.8).

Reproduce:
    python scripts/plot_st2_sweeps.py                # all families, n_each=200
    python scripts/plot_st2_sweeps.py --jobs 8       # parallel over families
    python scripts/plot_st2_sweeps.py --smoke        # 2 levels x 8 traces
Output:
    results/st2/<family>_summary.json ; figures/st2_<family>.png/.pdf

Slurm array (one family per task): --list prints the families + count;
--array-id N sweeps only family N (falls back to SLURM_ARRAY_TASK_ID). See
scripts/slurm/st2_sweeps.sbatch.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import multiprocessing
import os
import pathlib
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from powerladder.config import DEFAULT, MeterParams, St2Params  # noqa: E402
from powerladder.plotstyle import C, WIDTH_WIDE, apply_house_style  # noqa: E402
from powerladder.typeb.st2 import run_family  # noqa: E402
from powerladder.typeb.st2_attacks import FAMILY_ORDER, attack_families  # noqa: E402

_ROOT = pathlib.Path(__file__).resolve().parent.parent
_RESULTS = _ROOT / "results" / "st2"
_FIGURES = _ROOT / "figures"

# detector -> (colour, FAR-0.05 linestyle, label); the second FAR is drawn
# dashed in the same colour.
_STYLE = {"spectral": (C["orange"], "spectral (matched filter)"),
          "viterbi": (C["blue"], "Viterbi (line tracker)"),
          "mtf": (C["green"], "multitaper F (fixed)"),
          "dg_order_full": (C["vermillion"], "DG order (adaptive)"),
          "dg_order_semicoh": (C["purple"], "DG order (semi-coherent)")}


def detector_set(which: str) -> dict:
    """The injected detector dict for the sweep (task 20.9).

    "base" is the pre-ST1 pair; "full" adds the ST1-gate detectors that
    survived calibration: the fixed multitaper F comparator and the tracked
    order/cyclostationary statistics (pooled and semi-coherent).
    """
    from powerladder.typeb.gate import DETECTORS
    dets = dict(DETECTORS)
    if which == "full":
        from functools import partial

        from powerladder.st1.pipeline import ST1_DETECTORS, stage4_semicoherent
        dets["mtf"] = ST1_DETECTORS["mtf"]
        dets["dg_order_full"] = ST1_DETECTORS["dg_order_full"]
        dets["dg_order_semicoh"] = partial(stage4_semicoherent,
                                           params=DEFAULT.st1)
    return dets

_XLABEL = {
    "jitter": r"i.i.d. period jitter $\sigma_\xi$",
    "work": r"work variation $\sigma_w$ (micro-step count)",
    "drift": r"$f_0$ drift excursion (Hz)",
    "phase": r"phase slip $\sigma_{\rm slip}$ (fraction of period)",
    "relocate": r"relocated cadence $f_0$ (Hz)",
    "harmonic": "transition smoothing (s)",
    "shape": r"amplitude-shaping fill $\phi$",
    "dilute": "dominant training share",
    "meter": "hostile meter variant",
}


def smoke_params(p: St2Params) -> St2Params:
    """Truncate every level grid to 2 levels at tiny n (timing / schema runs)."""
    return dataclasses.replace(
        p, n_each=8,
        jitter_levels=p.jitter_levels[:2], work_levels=p.work_levels[:2],
        drift_levels=p.drift_levels[:2], phase_levels=p.phase_levels[:2],
        harmonic_levels=p.harmonic_levels[:2], shape_levels=p.shape_levels[:2],
        relocate_f0s=p.relocate_f0s[:2], dilute_shares=p.dilute_shares[:2],
        meter_variants=p.meter_variants[:2],
    )


def summarise(family: str, points, p: St2Params) -> dict:
    """The per-family summary record (schema echoing results/b1/gate_summary)."""
    fam = attack_families(p)[family]
    detectors = sorted(points[0].auc)
    return {
        "family": family,
        "generator": ("ko_workload eq-11 aggregate vs inference-dominant "
                      "aggregate null" if family == "dilute" else
                      "ko_workload training vs hard inference null"),
        "budget_units": fam.budget_units,
        "cost_anchor": fam.cost_anchor,
        "n_each": p.n_each,
        "target_fars": list(p.target_fars),
        "seed": p.seed,
        "meter": ("per-level hostile variants (see St2Params.meter_variants)"
                  if family == "meter" else
                  "off (exact no-op MeterParams)" if p.meter == MeterParams()
                  else "custom (St2Params.meter)"),
        "detectors": detectors,
        "levels": [pt.level for pt in points],
        "points": [
            {
                "level": pt.level,
                "cadence_cv": pt.cadence_cv,
                "phase_diffusion_D": pt.phase_diffusion_D,
                "auc": pt.auc,
                "tpr_at_far": pt.tpr_at_far,
            }
            for pt in points
        ],
    }


def plot_family(family: str, points, p: St2Params,
                fig_dir: pathlib.Path) -> pathlib.Path:
    """TPR at both FARs vs attack level, one curve pair per detector."""
    apply_house_style()
    fig, ax = plt.subplots(figsize=(0.62 * WIDTH_WIDE, 2.5))
    categorical = family == "meter"
    xs = (list(range(len(points))) if categorical
          else [float(pt.level) for pt in points])

    detectors = sorted(points[0].auc)
    for name in detectors:
        col, lab = _STYLE.get(name, (C["green"], name))
        for far, ls, mk in zip(p.target_fars, ("-", "--"), ("o", "s")):
            ys = [pt.tpr_at_far[name][f"{far:g}"] for pt in points]
            ax.plot(xs, ys, color=col, ls=ls, lw=1.4, marker=mk, ms=2.5,
                    label=f"{lab}, FAR {far:g}")
    if categorical:
        ax.set_xticks(xs)
        ax.set_xticklabels([str(pt.level).replace("_", " ") for pt in points],
                           rotation=20, ha="right", fontsize=6)
        ax.tick_params(axis="x", which="minor", bottom=False)
    ax.set_ylim(0, 1.04)
    ax.set_xlabel(_XLABEL[family])
    ax.set_ylabel("Detection rate")
    ax.set_title(f"ST2 {family} family", fontsize=8)
    ax.legend(frameon=False, fontsize=5.5, loc="best")
    fig.tight_layout()

    fig_dir.mkdir(parents=True, exist_ok=True)
    pdf = fig_dir / f"st2_{family}.pdf"
    fig.savefig(pdf)
    fig.savefig(fig_dir / f"st2_{family}.png")
    plt.close(fig)
    return pdf


def run_and_write(family: str, p: St2Params,
                  results_dir: pathlib.Path = _RESULTS,
                  fig_dir: pathlib.Path = _FIGURES,
                  detectors: str = "full") -> dict:
    """Sweep one family, write its summary JSON + figure; return the summary."""
    t0 = time.time()
    points = run_family(family, p, DEFAULT.ko, DEFAULT.ko_typeb,
                        detectors=detector_set(detectors))
    summary = summarise(family, points, p)
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / f"{family}_summary.json").write_text(
        json.dumps(summary, indent=2))
    plot_family(family, points, p, fig_dir)
    print(f"  {family}: {len(points)} levels in {time.time() - t0:.1f}s")
    return summary


def _worker(args_tuple):
    family, p, detectors = args_tuple
    return run_and_write(family, p, detectors=detectors)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--families", nargs="+", choices=FAMILY_ORDER,
                    default=list(FAMILY_ORDER), metavar="FAM",
                    help="attack families to sweep (default: all)")
    ap.add_argument("--n-each", type=int, default=DEFAULT.st2.n_each,
                    help="traces per class per level")
    ap.add_argument("--seed", type=int, default=DEFAULT.st2.seed)
    ap.add_argument("--jobs", type=int, default=1,
                    help="parallel worker processes (over families)")
    ap.add_argument("--smoke", action="store_true",
                    help="2 levels x 8 traces per family (timing/schema run)")
    ap.add_argument("--detectors", choices=("base", "full"), default="full",
                    help="'base' = spectral+viterbi (pre-ST1); 'full' adds "
                         "mtf and the DG order detectors (task 20.9)")
    ap.add_argument("--list", action="store_true",
                    help="print the family list and count, then exit (use to "
                         "size a Slurm --array range)")
    ap.add_argument("--array-id", type=int, default=None,
                    help="sweep only family index N of FAMILY_ORDER (overrides "
                         "--families / SLURM_ARRAY_TASK_ID) — one family per task")
    args = ap.parse_args()

    if args.list:
        for i, fam in enumerate(FAMILY_ORDER):
            print(f"{i:3d}  {fam}")
        print(f"{len(FAMILY_ORDER)} families")
        return

    # Array mode: one family per Slurm task. Each family writes its own summary
    # JSON independently, so concurrent array tasks never race.
    env_id = os.environ.get("SLURM_ARRAY_TASK_ID")
    array_id = args.array_id if args.array_id is not None else (
        int(env_id) if env_id is not None else None)
    if array_id is not None:
        if not 0 <= array_id < len(FAMILY_ORDER):
            raise SystemExit(f"array id {array_id} out of range "
                             f"[0, {len(FAMILY_ORDER)})")
        args.families = [FAMILY_ORDER[array_id]]
        print(f"array family {array_id}/{len(FAMILY_ORDER)}: {args.families[0]}")

    p = dataclasses.replace(DEFAULT.st2, n_each=args.n_each, seed=args.seed)
    if args.smoke:
        p = smoke_params(p)

    t0 = time.time()
    work = [(f, p, args.detectors) for f in args.families]
    if args.jobs > 1:
        with multiprocessing.Pool(args.jobs) as pool:
            summaries = pool.map(_worker, work)
    else:
        summaries = [_worker(w) for w in work]

    for s in summaries:
        far0 = f"{p.target_fars[0]:g}"
        worst = {d: min(pt["tpr_at_far"][d][far0] for pt in s["points"])
                 for d in s["detectors"]}
        print(f"{s['family']:<9} min TPR@{far0}: "
              + "  ".join(f"{d}={v:.2f}" for d, v in worst.items()))
    print(f"total {time.time() - t0:.1f}s -> {_RESULTS} ; {_FIGURES}/st2_*")


if __name__ == "__main__":
    main()
