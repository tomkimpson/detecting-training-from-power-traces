"""Cost of hiding per adversarial strategy: one Pareto panel per priced family.

Once work variation at level 0.7 was priced (~0% throughput, the OzSTAR top-up),
the pooled Pareto staircase of plot_money_pareto.py went flat from zero cost: the
cheapest attack is also the best one, so the envelope says little. This figure
keeps the trade-off visible by splitting it by strategy. Each panel is one of the
four cost-anchored families (work variation, period jitter, cadence drift, power
shaping) and shows, against the Viterbi tracker and against the fixed tests, every
attack level as a point, that family's own Pareto staircase (solid: the best
hiding the family buys at or below each cost, drawn only over the family's
measured cost range) and the attack dial (dotted: the points joined in order of
increasing level). For work variation every cost is zero within noise, so its
staircase is a point and the dial is the informative curve.

Each panel has its own x-axis scale, fitted to that family's cost range (see
_SCALE): linear for work variation (all within noise of zero) and power shaping
(a narrow 170-230 % band), log for jitter, symlog for drift (which includes the
0 % honest level). The linear panels carry the measured cost spread (std across
traces) as horizontal error bars.

READ-ONLY: a pure reader of a frozen frontier_summary.json, reusing the
reductions of plot_st2_pareto.py so the figures cannot disagree.

Reproduce:
    python scripts/plot_pareto_by_strategy.py \
        --summary results/st2/aggregate/frontier_summary.json --far 0.01
Outputs:
    figures/pareto_by_strategy_aggregate.{pdf,png}   (or figures/<STEM>.*)
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

from powerladder.plotstyle import C, WIDTH_WIDE, apply_house_style, save  # noqa: E402

_HERE = pathlib.Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("plot_st2_pareto",
                                               _HERE / "plot_st2_pareto.py")
_pp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_pp)

_SUMMARY = _HERE.parent / "results" / "st2" / "aggregate" / "frontier_summary.json"

# Panel order and titles: the free attack first, then the costly ones.
_PANELS = {
    "work": "vary real work per iteration",
    "jitter": "period jitter (idle padding)",
    "drift": "cadence drift (idle padding)",
    "shape": "power shaping",
}
# Per-panel x scale, fitted to each family's cost range.
_SCALE = {"work": "linear", "jitter": "log", "drift": "symlog", "shape": "linear"}
_LINTHRESH = 10.0     # drift symlog: linear within +-10 % overhead
_LABEL_GAP = 0.06     # min vertical spacing [hiding units] of nearby labels
_LABEL_NEAR = 0.08    # labels closer than this [axes fraction in x] collide


def _xlim(scale: str, xs: list[float], errs: list[float]) -> tuple[float, float]:
    """Padded x-limits for one panel."""
    lo = min(x - e for x, e in zip(xs, errs))
    hi = max(x + e for x, e in zip(xs, errs))
    if scale == "log":
        return lo / 1.6, hi * 1.6
    if scale == "symlog":
        return min(lo, 0.0) - 2.0, hi * 1.6
    pad = 0.12 * (hi - lo)
    return lo - pad, hi + 1.5 * pad          # extra room on the right for labels


def spread_labels(ax, pts: list[dict]) -> list[tuple[dict, float]]:
    """(point, label y) with nearby labels pushed apart vertically.

    Nearness is judged in axes-fraction x, so it works on any scale; the axis
    limits must already be set.
    """
    to_ax = ax.transData + ax.transAxes.inverted()
    fx = {id(p): to_ax.transform((p["cost"], 0.0))[0] for p in pts}
    placed: list[tuple[dict, float]] = []
    for p in sorted(pts, key=lambda q: q["hiding"]):
        y = p["hiding"]
        for q, yq in placed:
            if abs(fx[id(q)] - fx[id(p)]) < _LABEL_NEAR:
                y = max(y, yq + _LABEL_GAP)
        placed.append((p, y))
    return placed

_SERIES = {
    "Viterbi tracker": (("viterbi",), C["blue"]),
    "fixed tests": (None, C["orange"]),
}


def series(summary: dict) -> dict[str, tuple[tuple[str, ...], str]]:
    fixed = _pp.detector_classes(summary)["fixed"]
    return {name: (dets if dets is not None else fixed, col)
            for name, (dets, col) in _SERIES.items()}


def plot(summary: dict, far: str, name: str) -> pathlib.Path:
    apply_house_style()
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH_WIDE, 3.9), sharey=True)
    std = {(c["family"], c["level"]): c.get("cost_overhead_pct_std") or 0.0
           for c in summary["cells"]}

    for ax, (fam, title) in zip(axes.flat, _PANELS.items()):
        scale = _SCALE[fam]
        fam_cells = [c for c in summary["cells"] if c["family"] == fam
                     and c["cost_overhead_pct"] is not None]
        xs = [c["cost_overhead_pct"] for c in fam_cells]
        errs = ([std[(c["family"], c["level"])] for c in fam_cells]
                if scale == "linear" else [0.0] * len(xs))
        if scale == "symlog":
            ax.set_xscale("symlog", linthresh=_LINTHRESH)
        else:
            ax.set_xscale(scale)
        ax.set_xlim(*_xlim(scale, xs, errs))
        ax.set_ylim(-0.04, 1.04)

        for label, (dets, col) in series(summary).items():
            anchored, _ = _pp.split_by_cost(summary["cells"], dets, far)
            pts = [p for p in anchored if p["family"] == fam]
            env = _pp.pareto_envelope(pts)
            ax.step([p[0] for p in env], [p[1] for p in env], where="post",
                    color=col, lw=1.2, zorder=2)
            dial = sorted(pts, key=lambda q: q["level"])
            ax.plot([p["cost"] for p in dial], [p["hiding"] for p in dial],
                    color=col, lw=0.6, ls=":", alpha=0.8, zorder=1)
            if scale == "linear":
                ax.errorbar([p["cost"] for p in pts], [p["hiding"] for p in pts],
                            xerr=[std[(fam, p["level"])] for p in pts],
                            ls="none", ecolor=col, elinewidth=0.5, capsize=1.0,
                            alpha=0.6, zorder=2)
            ax.plot([p["cost"] for p in pts], [p["hiding"] for p in pts],
                    ls="none", marker="o", ms=3.0, mfc=col, mec="white",
                    mew=0.3, zorder=3)
            if label == "fixed tests":
                # level labels once per cell, beside the fixed-test point (the
                # higher of the pair in every cell), nudged apart where crowded
                for p, y in spread_labels(ax, pts):
                    ax.annotate(f"{p['level']:g}", (p["cost"], y),
                                textcoords="offset points", xytext=(4.0, -2.0),
                                fontsize=4.5, color=C["grey"], zorder=4,
                                bbox=dict(fc="white", ec="none", pad=0.4,
                                          alpha=0.85))
        ax.set_title(title, fontsize=7)
        if scale != "log":
            ax.axvline(0.0, color=C["grey"], lw=0.5, ls=":", zorder=0)

    for ax in axes[1]:
        ax.set_xlabel("measured throughput overhead [%]")
    for ax in axes[:, 0]:
        ax.set_ylabel(f"hiding at FAR {far}")
    handles = [Line2D([], [], color=col, lw=1.2, marker="o", ms=3, label=lab)
               for lab, (_, col) in series(summary).items()]
    handles += [Line2D([], [], color=C["grey"], lw=1.2, label="Pareto staircase"),
                Line2D([], [], color=C["grey"], lw=0.6, ls=":",
                       label="increasing attack level")]
    # the drift panel is empty above the floor at low cost
    axes[1, 0].legend(handles=handles, frameon=False, fontsize=5,
                      loc="upper left")
    fig.tight_layout()
    return save(fig, name)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--summary", type=pathlib.Path, default=_SUMMARY)
    ap.add_argument("--far", type=float, default=_pp._DEFAULT_FAR,
                    help="FAR the hiding axis is read at (must be stored)")
    ap.add_argument("--out", default="pareto_by_strategy_aggregate",
                    help="figure stem under figures/")
    args = ap.parse_args()
    summary = json.loads(args.summary.read_text())
    far = _pp.far_key(summary, args.far)
    for fam in _PANELS:
        print(f"{fam}:")
        for label, (dets, _) in series(summary).items():
            anchored, _ = _pp.split_by_cost(summary["cells"], dets, far)
            env = _pp.pareto_envelope([p for p in anchored if p["family"] == fam])
            print(f"  {label}: {[(round(c, 1), round(h, 2)) for c, h in env]}")
    out = plot(summary, far, args.out)
    print(f"-> {out.with_suffix('')}.*")


if __name__ == "__main__":
    main()
