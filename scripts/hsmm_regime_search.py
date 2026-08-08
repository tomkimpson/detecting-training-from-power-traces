#!/usr/bin/env python3
"""Stage 3: search for a regime where knowing the true event locations helps.

Stages 1 and 2 established that the marked-renewal detector saturates: every
population separates at AUC ~1.000 on the pristine channel, so no comparison --
E0 vs E1, E1 vs E2, feature set vs feature set -- has room to resolve anything.
With the corrected (metadata-free) alignment benchmark there is no measured gap
anywhere that an explicit-duration model could close.

So this campaign does not develop E2.  It searches for an *identifiable regime*:
an operating point where the detector is off its ceiling AND true alignment
measurably beats random alignment.  Axes, cheapest first:

1. **record length** -- 5/10/20/45/90 s;
2. **dilution / superposition** -- the training line buried in a Ko eq-11
   aggregate; operationally realistic and a direct challenge to locality;
3. **event contrast** (``f_peak_frac``) -- a sensitivity analysis on event SNR,
   shrinking the compute-to-communication excursion against fixed meter noise;
4. **harder nulls** -- phase-randomised surrogates of the positives themselves,
   matched on mean, variance and power spectrum, so only local phase structure
   can separate them.

This is a coordinate search, not a factorial: each axis is swept from the
baseline, then a small targeted grid combines the axes that actually desaturate.

Outputs land in ``results/hsmm_regime_search/`` and never touch frozen ST1/ST2
artefacts or the earlier campaigns.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from powerladder.typeb.renewal_campaign import (  # noqa: E402
    NULL_FAMILIES,
    glue_for,
    make_annotated_population,
    make_diluted_population,
    make_null_populations,
    matched_surrogate_population,
)
from powerladder.typeb.renewal_regime import (  # noqa: E402
    ADMISSION,
    admits_e2,
    measure_point,
)


_ROOT = pathlib.Path(__file__).resolve().parent.parent
_DEFAULT_OUTPUT = _ROOT / "results" / "hsmm_regime_search"

_BASELINE = {"duration_s": 90.0, "share": None, "f_peak_frac": None,
             "nulls": "composite"}

_DURATIONS = (3.0, 5.0, 10.0, 20.0, 45.0, 90.0)
_SHARES = (0.5, 0.35, 0.2, 0.1)
#: Fractions of the default 0.85 peak share -- the event-contrast axis.
_F_PEAK_FRACS = (0.425, 0.21, 0.106, 0.053)
_NULL_SETS = ("composite", "surrogate_ft", "surrogate_aaft")

#: A channel the paper could credibly claim.  Nothing in this campaign degrades
#: the meter, so every point here is on the nominal 20 Hz channel.
_CREDIBLE_CHANNEL = True


def _point_name(spec: dict) -> str:
    bits = [f"T{spec['duration_s']:g}"]
    if spec["share"] is not None:
        bits.append(f"share{spec['share']:g}")
    if spec["f_peak_frac"] is not None:
        bits.append(f"fpk{spec['f_peak_frac']:g}")
    if spec["nulls"] != "composite":
        bits.append(spec["nulls"])
    return "_".join(bits)


def _positives(spec: dict, n: int, seed: int):
    glue = glue_for(spec["duration_s"], f_peak_frac=spec["f_peak_frac"])
    if spec["share"] is None:
        return make_annotated_population("drift", 0.0, n, glue, seed), glue
    return make_diluted_population(spec["share"], n, glue, seed), glue


def _nulls(spec: dict, glue, n: int, seed: int, positives) -> dict:
    """Null populations for one operating point.

    The surrogate families are built from an *independent* positive draw, never
    from the traces being scored, so a positive is never compared against a
    surrogate of itself.
    """
    if spec["nulls"] == "composite":
        return make_null_populations(n, glue, seed)
    kind = "ft" if spec["nulls"] == "surrogate_ft" else "aaft"
    return {spec["nulls"]: matched_surrogate_population(positives, seed,
                                                        kind=kind)}


def _run_point(spec: dict, args, seed: int) -> dict:
    fit_pos, glue = _positives(spec, args.n_fit, seed + 1)
    eval_pos, _ = _positives(spec, args.n_eval, seed + 2)
    # Independent positive draws feeding the surrogate nulls.
    surrogate_source_fit, _ = _positives(spec, args.n_fit, seed + 3)
    surrogate_source_eval, _ = _positives(spec, args.n_eval, seed + 4)

    n_null = max(args.n_eval // len(NULL_FAMILIES), 20)
    fit_nulls = _nulls(spec, glue, n_null, seed + 5, surrogate_source_fit)
    eval_nulls = _nulls(spec, glue, n_null, seed + 6, surrogate_source_eval)

    result = measure_point(fit_pos, eval_pos, fit_nulls, eval_nulls,
                           seed=seed, n_random=args.n_random,
                           n_boot=args.n_boot)
    verdict = admits_e2(result, credible_channel=_CREDIBLE_CHANNEL)
    return {"spec": spec, **result.as_dict(), "verdict": verdict}


def _sweeps() -> list[dict]:
    """Coordinate sweep: one axis at a time from the baseline."""
    points = [dict(_BASELINE, duration_s=d) for d in _DURATIONS]
    points += [dict(_BASELINE, share=s) for s in _SHARES]
    points += [dict(_BASELINE, f_peak_frac=f) for f in _F_PEAK_FRACS]
    points += [dict(_BASELINE, nulls=n) for n in _NULL_SETS if n != "composite"]
    seen, unique = set(), []
    for spec in points:
        name = _point_name(spec)
        if name not in seen:
            seen.add(name)
            unique.append(spec)
    return unique


def _targeted_grid() -> list[dict]:
    """Combine the axes most likely to desaturate, at their harder levels."""
    # Record length is the axis that actually desaturates, so the grid is
    # centred on the short-record end rather than spread evenly.
    grid = []
    for duration in (3.0, 5.0, 10.0, 20.0):
        for f_peak in (0.21, 0.106):
            grid.append(dict(_BASELINE, duration_s=duration, f_peak_frac=f_peak))
        for share in (0.35, 0.2):
            grid.append(dict(_BASELINE, duration_s=duration, share=share))
        grid.append(dict(_BASELINE, duration_s=duration, nulls="surrogate_aaft"))
    # The hardest null against the hardest single-workload settings.
    for duration in (5.0, 10.0):
        grid.append(dict(_BASELINE, duration_s=duration, f_peak_frac=0.21,
                         nulls="surrogate_aaft"))
        grid.append(dict(_BASELINE, duration_s=duration, share=0.2,
                         nulls="surrogate_aaft"))
    return grid


def run(args: argparse.Namespace) -> dict:
    t0 = time.time()
    output = pathlib.Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    points: dict[str, dict] = {}
    specs = _sweeps() + ([] if args.sweeps_only else _targeted_grid())
    for j, spec in enumerate(specs):
        name = _point_name(spec)
        if name in points:
            continue
        points[name] = _run_point(spec, args, args.seed + 1000 * j)
        r = points[name]
        print(f"  {name:<28s} E1={r['e1_auc']:.3f} "
              f"E0true={r['e0_true_auc']:.3f} E0rand={r['e0_random_auc']:.3f} "
              f"align+={r['align_plus']:+.3f} "
              f"[{r['align_plus_lo']:+.3f},{r['align_plus_hi']:+.3f}] "
              f"recall={r['event_recall']:.2f} ev={r['events_per_trace']:.0f} "
              f"deg={r['degenerate_fraction']:.2f}"
              f"{'  <== ADMITTED' if r['verdict']['admitted'] else ''}",
              flush=True)

    admitted = [name for name, r in points.items() if r["verdict"]["admitted"]]
    admitted_strict = [name for name, r in points.items()
                       if r["verdict"]["admitted_strict"]]
    desaturated = [
        name for name, r in points.items()
        if ADMISSION["e1_band"][0] <= r["e1_auc"] <= ADMISSION["e1_band"][1]
    ]
    summary = {
        "status": "stage-3 identifiable-regime search; not a frozen paper result",
        "question": ("is there a non-saturated regime where true event alignment "
                     "measurably beats random alignment?"),
        "sizes": {"fit": args.n_fit, "evaluation": args.n_eval,
                  "random_alignments": args.n_random, "bootstrap": args.n_boot},
        "seed": args.seed,
        "admission_rule": ADMISSION,
        "baseline": _BASELINE,
        "points": points,
        "n_points": len(points),
        "desaturated_points": desaturated,
        "admitted_points": admitted,
        "admitted_points_strict_rule": admitted_strict,
        "any_admitted": bool(admitted),
        "elapsed_s": time.time() - t0,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    _plot(points, output)
    return summary


def _plot(points: dict[str, dict], output: pathlib.Path) -> None:
    """align+ with its bootstrap interval against blind discrimination."""
    names = list(points)
    e1 = np.asarray([points[n]["e1_auc"] for n in names])
    ap = np.asarray([points[n]["align_plus"] for n in names])
    lo = np.asarray([points[n]["align_plus_lo"] for n in names])
    hi = np.asarray([points[n]["align_plus_hi"] for n in names])
    admitted = np.asarray([points[n]["verdict"]["admitted"] for n in names])

    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    ax.errorbar(e1, ap, yerr=[ap - lo, hi - ap], fmt="o", ms=4, lw=0.8,
                color="0.35", ecolor="0.7", label="operating point")
    if admitted.any():
        ax.plot(e1[admitted], ap[admitted], "o", ms=7, mfc="none",
                mec="C3", mew=1.6, label="admits E2")
    ax.axhline(0.0, color="0.5", lw=0.8)
    ax.axhline(ADMISSION["min_align_plus"], color="C0", ls="--", lw=0.9,
               label=f"align+ = {ADMISSION['min_align_plus']}")
    ax.axvspan(*ADMISSION["e1_band"], color="C2", alpha=0.10,
               label="measurable E1 band")
    ax.set_xlabel("blind discrimination, E1 AUC")
    ax.set_ylabel("alignment benefit, align+ (true - random)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "align_benefit.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-fit", type=int, default=150)
    parser.add_argument("--n-eval", type=int, default=150)
    parser.add_argument("--n-random", type=int, default=8,
                        help="random alignments drawn per trace")
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260808)
    parser.add_argument("--output-dir", default=str(_DEFAULT_OUTPUT))
    parser.add_argument("--sweeps-only", action="store_true",
                        help="skip the targeted combination grid")
    parser.add_argument("--quick", action="store_true",
                        help="plumbing run: 30 fit / 30 eval / 3 draws / 200 boot")
    args = parser.parse_args()
    if args.quick:
        args.n_fit, args.n_eval, args.n_random, args.n_boot = 30, 30, 3, 200
        args.output_dir = str(pathlib.Path(args.output_dir) / "quick")

    summary = run(args)
    print(f"\ndesaturated: {summary['desaturated_points']}")
    print(f"admitted   : {summary['admitted_points']}")
    print(f"elapsed {summary['elapsed_s']:.1f}s -> {args.output_dir}")


if __name__ == "__main__":
    main()
