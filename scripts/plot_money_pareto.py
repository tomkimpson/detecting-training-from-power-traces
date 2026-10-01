"""Headline cost-of-hiding figure: the Viterbi tracker vs the fixed tests.

Single-panel companion to scripts/plot_st2_pareto.py. That script draws one
panel per detector *class* (tracking = best of Viterbi and two DG order
statistics; fixed = best of spectral matched filter and multitaper F). The paper
is Viterbi-led, so this figure overlays the two Pareto staircases that carry
its claim on one axis:

    against the fixed tests, varying real work hides ~0.74 at ~zero cost;
    against the Viterbi tracker, no cost-anchored attack hides more than 0.16.

Colour encodes the detector, marker shape the attack family. Cells with no
measured hardware cost anchor are kept in a hatched strip (work=0.7, where the
tracker bends, is among them), exactly as in plot_st2_pareto.py.

READ-ONLY: a pure reader of the tracked results/st2/frontier_summary.json; it
reuses the reductions of plot_st2_pareto.py so the two figures cannot disagree.

Reproduce:
    python scripts/plot_money_pareto.py [--summary PATH] [--far 0.01] [--out STEM]
Outputs:
    figures/money_pareto.{pdf,png}   (or figures/<STEM>.*)
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from powerladder.plotstyle import (C, WIDTH_ICML_COL,  # noqa: E402
                                   apply_house_style, save)

_HERE = pathlib.Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("plot_st2_pareto",
                                               _HERE / "plot_st2_pareto.py")
_pp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_pp)

_SUMMARY = _HERE.parent / "results" / "st2" / "frontier_summary.json"

# The two detectors the figure compares, and their colours.
_SERIES = {
    "Viterbi tracker": (("viterbi",), C["blue"]),
    "fixed tests": (None, C["orange"]),
}


def series(summary: dict) -> dict[str, tuple[tuple[str, ...], str]]:
    """Detector tuples per series; the fixed class is read from the summary."""
    fixed = _pp.detector_classes(summary)["fixed"]
    return {name: (dets if dets is not None else fixed, col)
            for name, (dets, col) in _SERIES.items()}


def plot(summary: dict, far: str = _pp._FAR_KEY,
         name: str = "money_pareto") -> pathlib.Path:
    apply_house_style()
    cells = summary["cells"]
    fig, ax = plt.subplots(figsize=(WIDTH_ICML_COL, 2.45))

    costs = [c["cost_overhead_pct"] for c in cells
             if c["cost_overhead_pct"] is not None]
    x_hi = max(costs)
    strip_lo, strip_hi = x_hi * 1.06, x_hi * 1.24

    work_zero = None
    for k, (label, (dets, col)) in enumerate(series(summary).items()):
        anchored, unpriced = _pp.split_by_cost(cells, dets, far)
        if label == "fixed tests":
            # the cheap attack's best hiding against the fixed tests, for the
            # annotation: the most a ~zero-cost work-variation cell hides
            work_zero = max(p["hiding"] for p in anchored
                            if p["family"] == "work" and abs(p["cost"]) < 1.0)
        env = _pp.pareto_envelope(anchored)
        # extend the staircase to the edge of the measured range
        xs = [p[0] for p in env] + [x_hi]
        ys = [p[1] for p in env] + [env[-1][1]]
        ax.step(xs, ys, where="post", color=col, lw=1.4, zorder=2)
        for p in anchored:
            ax.plot(p["cost"], p["hiding"], ls="none",
                    marker=_pp._FAMILY_MARKER[p["family"]], ms=3.2,
                    mfc=col, mec="white", mew=0.3, alpha=0.9, zorder=3)
        # unpriced cells: short ticks in the hatched strip, one column per series
        xs_u = strip_lo + (strip_hi - strip_lo) * (0.3 + 0.4 * k)
        ax.plot([xs_u] * len(unpriced), [p["hiding"] for p in unpriced],
                ls="none", marker="_", ms=4, color=col, alpha=0.8, zorder=2)

    ax.axvspan(strip_lo, strip_hi, facecolor="none", edgecolor=C["grey"],
               hatch="////", lw=0.4, alpha=0.5, zorder=0)
    ax.text((strip_lo + strip_hi) / 2, 1.05, "cost not\nmeasured",
            ha="center", va="bottom", fontsize=4.5, color=C["grey"],
            clip_on=False)

    ax.annotate("vary real work\nper iteration: ~0% cost",
                xy=(0, work_zero), xytext=(40, 0.95), fontsize=5.5,
                color=C["orange"], ha="left", va="center",
                arrowprops=dict(arrowstyle="-", lw=0.5, color=C["orange"]))

    ax.set_xlim(-0.05 * x_hi, strip_hi + 0.02 * x_hi)
    ax.set_ylim(-0.04, 1.04)
    ax.set_xlabel("measured throughput overhead [%]")
    ax.set_ylabel(f"hiding at FAR {far}")

    handles = [Line2D([], [], color=col, lw=1.4, label=label)
               for label, (_, col) in series(summary).items()]
    handles += [Line2D([], [], ls="none", marker=m, ms=3.2, mfc=C["grey"],
                       mec="white", mew=0.3, label=f)
                for f, m in _pp._FAMILY_MARKER.items()]
    # the region right of the drift point and between the staircases is empty
    ax.legend(handles=handles, frameon=False, fontsize=5, loc="center left",
              bbox_to_anchor=(0.37, 0.44), ncol=2, columnspacing=0.8,
              handlelength=1.6)
    fig.tight_layout()
    return save(fig, name)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--summary", type=pathlib.Path, default=_SUMMARY)
    ap.add_argument("--far", type=float, default=_pp._DEFAULT_FAR,
                    help="FAR the hiding axis is read at (must be stored)")
    ap.add_argument("--out", default="money_pareto",
                    help="figure stem under figures/")
    args = ap.parse_args()
    summary = json.loads(args.summary.read_text())
    far = _pp.far_key(summary, args.far)
    for name, (dets, _) in series(summary).items():
        anchored, unpriced = _pp.split_by_cost(summary["cells"], dets, far)
        env = _pp.pareto_envelope(anchored)
        print(f"{name}: staircase {[(round(c, 1), round(h, 2)) for c, h in env]}")
        print("  unpriced hiding > 0.3: " + ", ".join(
            f"{p['family']}={p['level']} ({p['hiding']:.2f})"
            for p in unpriced if p["hiding"] > 0.3))
    out = plot(summary, far, args.out)
    print(f"-> {out.with_suffix('')}.*")


if __name__ == "__main__":
    main()
