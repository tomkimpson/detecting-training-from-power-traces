"""Headline detection figure: power vs cadence drift against the inference null.

The Viterbi-led main-text view of the ST1 bake-off. Detection rate at an
empirical false-alarm rate of 1e-2 (the level of the bake-off table) against the
modelled inference null, over the cadence-drift axis, for the Viterbi tracker and
the two fixed-frequency comparators (spectral matched filter, multitaper F),
with the Whittle Neyman--Pearson ceiling overlaid.

The full grid -- the DG order family, the structural-confuser negatives, the
controller-only hard case -- stays in figures/st1_bakeoff and
figures/st1_np_ceiling (appendix).

READ-ONLY: a pure reader of the tracked results/st1/np_ceiling_summary.json,
whose detector rows replay the bake-off evaluation populations exactly
(parity_check.max_tpr_delta_vs_bakeoff = 0).

--scenario aggregate reads the section-2 aggregate run
(results/st1/np_ceiling_aggregate_summary.json): dominant training over the
small-training + fine-tuning background, against the inference-dominant
aggregate null. That is the main-text view.

Reproduce:
    python scripts/plot_drift_ceiling.py [--scenario aggregate] [--far 0.01]
Outputs:
    figures/drift_ceiling{,_aggregate}.{pdf,png}
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from powerladder.plotstyle import C, WIDTH_ICML_COL, apply_house_style, save  # noqa: E402

_SUMMARY = (pathlib.Path(__file__).resolve().parent.parent
            / "results" / "st1" / "np_ceiling_summary.json")

_NEGATIVE = "inference"
_SERIES = [  # (summary key, label, colour, linestyle)
    ("viterbi", "Viterbi tracker", C["blue"], "-"),
    ("spectral", "spectral matched filter", C["orange"], "-"),
    ("mtf", "multitaper F-test", C["green"], "-"),
]


def plot(summary: dict, far: str, stem: str = "drift_ceiling") -> pathlib.Path:
    apply_house_style()
    drift = summary["drifts_hz"]
    block = summary["by_negative"][_NEGATIVE]
    fig, ax = plt.subplots(figsize=(WIDTH_ICML_COL, 2.2))

    ax.plot(drift, block["ceiling"]["tpr"][far], color=C["black"], ls="--",
            lw=1.0, label="Neyman\u2013Pearson ceiling (Whittle)", zorder=3)
    for key, label, col, ls in _SERIES:
        ax.plot(drift, block["detectors"][key]["tpr"][far], color=col, ls=ls,
                lw=1.3, marker="o", ms=2.8, label=label, zorder=2)

    ax.set_xlabel(r"cadence drift $\sigma_f$ [Hz]")
    ax.set_ylabel(f"detection rate at FAR {float(far):g}")
    ax.set_ylim(-0.03, 1.08)
    ax.legend(frameon=False, fontsize=5.5, loc="center right",
              bbox_to_anchor=(1.0, 0.62))
    fig.tight_layout()
    return save(fig, stem)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scenario", choices=("single", "aggregate"), default="single")
    ap.add_argument("--summary", type=pathlib.Path, default=None)
    ap.add_argument("--far", default="0.01", choices=["0.05", "0.01"])
    args = ap.parse_args()
    suffix = "" if args.scenario == "single" else f"_{args.scenario}"
    path = args.summary or _SUMMARY.with_name(f"np_ceiling{suffix}_summary.json")
    summary = json.loads(path.read_text())
    block = summary["by_negative"][_NEGATIVE]
    print(f"drift {summary['drifts_hz']}")
    for key, *_ in _SERIES:
        print(f"  {key:9s} tpr@{args.far} {block['detectors'][key]['tpr'][args.far]}")
    out = plot(summary, args.far, stem=f"drift_ceiling{suffix}")
    print(f"-> {out.with_suffix('')}.*")


if __name__ == "__main__":
    main()
