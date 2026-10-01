"""Attack response curves: hiding against budget knob, one panel per strategy.

Small multiples over the eight schedule strategies of Tab. 1: the four with a
measured single-A100 throughput cost c on the top row, the four unpriced ones
below. Each panel plots hiding (1 - detection rate at the chosen FAR) against the
strategy's own budget knob, for the Viterbi tracker (blue, solid) and the fixed
tests (grey, dashed; the better of matched filter and multitaper F per cell),
with Wilson 95% intervals for n traces per class. Levels are evenly spaced, weakest on the left, and labelled with
their knob value on the bottom axis and, on priced panels, its measured c on
the top axis.
Dilution's levels run in decreasing training share (lower share is the stronger
attack); relocation plots the cadence itself, so its attack grows toward both
edges.

--x cv instead puts every cell of every strategy on one shared axis, the
measured cadence coefficient of variation, as a single scatter (one panel per
detector).

READ-ONLY: a pure reader of a frozen frontier_summary.json.

Reproduce:
    python scripts/plot_attack_curves.py            # small multiples
    python scripts/plot_attack_curves.py --x cv     # shared-axis scatter
Outputs:
    figures/attack_curves_aggregate.{pdf,png}
    figures/attack_scatter_cv_aggregate.{pdf,png}
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from powerladder.plotstyle import (C, FAMILY_COLOR, WIDTH_WIDE,  # noqa: E402
                                   apply_house_style, save)

_SUMMARY = (pathlib.Path(__file__).resolve().parent.parent
            / "results" / "st2" / "aggregate" / "frontier_summary.json")

# (family, title, knob label, reverse level order)
_PANELS = [  # priced strategies (Tab. 3 order) on the top row, unpriced below
    ("jitter", "Period jitter", r"$\sigma_\xi$", False),
    ("drift", "Cadence drift", r"$\sigma_f$ [Hz]", False),
    ("work", "Work variation", r"$\sigma_G$", False),
    ("shape", "Shape filling", r"fill $\phi$", False),
    ("phase", "Phase slip", r"$\sigma_s$ [periods]", False),
    ("relocate", "Relocation", r"$f_0$ [Hz]", False),
    ("harmonic", "Harmonic smoothing", r"ramp $w$ [s]", False),
    ("dilute", "Dilution", r"training share", True),
]
_MARKER = {"jitter": "o", "work": "s", "drift": "^", "shape": "D",
           "phase": "v", "harmonic": "P", "relocate": "X", "dilute": "*"}
# headline cell annotated in the figure: (family, level)
_HEADLINE = ("work", 0.7)
_Z95 = 1.959964


def _wilson(p: float, n: int) -> tuple[float, float]:
    """Wilson 95% interval for a binomial proportion p observed on n trials."""
    den = 1 + _Z95**2 / n
    mid = (p + _Z95**2 / (2 * n)) / den
    half = _Z95 * np.sqrt(p * (1 - p) / n + _Z95**2 / (4 * n**2)) / den
    return mid - half, mid + half


def _cells(summary: dict, fam: str, fixed: tuple[str, ...], far: str) -> list[dict]:
    out = []
    for c in summary["cells"]:
        if c["family"] != fam:
            continue
        out.append({
            "level": c["level"], "cv": c["cadence_cv"],
            "cost": c.get("cost_overhead_pct"),
            "viterbi": 1 - c["tpr_at_far"]["viterbi"][far],
            "fixed": 1 - max(c["tpr_at_far"][d][far] for d in fixed),
        })
    return sorted(out, key=lambda p: p["level"])


def _cost_label(c: float) -> str:
    return "0" if abs(c) < 1.0 else f"{c:.0f}"


def _cost_axis(ax, pts: list[dict], upper_bound: bool) -> None:
    """Top axis giving each level's measured throughput cost c [%]."""
    top = ax.secondary_xaxis("top")
    top.set_xticks(range(len(pts)))
    labels = []
    for p in pts:
        c = p["cost"]
        txt = "" if c is None else _cost_label(c)
        if upper_bound and txt not in ("", "0"):
            txt = r"$\leq$" + txt
        labels.append(txt)
    top.set_xticklabels(labels, fontsize=6)
    top.tick_params(axis="x", which="both", direction="in", pad=1.5)
    top.minorticks_off()
    top.set_xlabel("$c$ [%]", fontsize=6.5, labelpad=2)


def curves(summary: dict, fixed, far: str, far_txt: str) -> plt.Figure:
    n = int(summary["n_each"])
    fig = plt.figure(figsize=(WIDTH_WIDE * 1.3, 4.5), layout="constrained")
    fig.get_layout_engine().set(h_pad=0.04, hspace=0.04, wspace=0.015)
    rows = fig.subfigures(2, 1, hspace=0.03, height_ratios=(1.08, 1.0))
    labels = iter("abcdefgh")
    col_v, col_f = C["blue"], "#7f7f7f"
    for r, (sub, row_title) in enumerate(zip(
            rows, ("Measured on an A100: throughput cost $c$ on the top axis",
                   "No hardware implementation, so no measured cost"))):
        axes = sub.subplots(1, 4, sharey=True)
        sub.suptitle(row_title, x=0.0, ha="left", fontsize=7.5,
                     fontweight="bold", color="#333333")
        for ax, (fam, title, xlab, inv) in zip(axes, _PANELS[4 * r:4 * r + 4]):
            pts = _cells(summary, fam, fixed, far)
            if inv:
                pts = pts[::-1]
            x = np.arange(len(pts))
            v = np.array([p["viterbi"] for p in pts])
            f = np.array([p["fixed"] for p in pts])
            for y, col in ((f, col_f), (v, col_v)):
                lo, hi = zip(*(_wilson(p, n) for p in y))
                ax.fill_between(x, lo, hi, color=col, alpha=0.18, lw=0, zorder=2)
            ax.plot(x, f, "--o", color=col_f, ms=3, mfc="white", mew=0.8,
                    lw=1.0, zorder=3)
            ax.plot(x, v, "-o", color=col_v, ms=3.2, mec="white", mew=0.4,
                    lw=1.4, zorder=4)
            if fam == _HEADLINE[0]:
                i = [p["level"] for p in pts].index(_HEADLINE[1])
                ax.plot(x[i], v[i], "o", ms=7, mfc="none", mec=C["vermillion"],
                        mew=1.0, zorder=5)
                hid = round(v[i] + 1e-9, 2)  # 0.545 -> 0.55, as quoted in the text
                ax.annotate(f"hides {hid:.2f}\nat $c\\approx0$",
                            (x[i], v[i]), xytext=(3.3, 0.25), textcoords="data",
                            ha="center", va="center", fontsize=6,
                            color=C["vermillion"], zorder=6,
                            arrowprops=dict(arrowstyle="-", lw=0.6,
                                            color=C["vermillion"],
                                            shrinkB=4))
            ax.axhline(0, color="#cccccc", lw=0.5, zorder=0)
            ax.set_ylim(-0.03, 1.03)
            ax.set_xlim(-0.5, len(pts) - 0.5)
            ax.set_yticks([0, 0.5, 1])
            ax.tick_params(axis="y", which="minor", left=False)
            ax.set_xticks(x)
            ax.set_xticklabels([f"{p['level']:g}" for p in pts], fontsize=6)
            ax.tick_params(axis="x", which="minor", bottom=False)
            ax.set_xlabel(xlab + (r"  (stronger $\rightarrow$)" if inv else ""),
                          labelpad=1.5, fontsize=7)
            pad = 4
            if r == 0:
                _cost_axis(ax, pts, upper_bound=(fam == "shape"))
            ax.set_title(title, fontsize=7.5, pad=pad)
            ax.set_title(next(labels), loc="left", fontsize=8.5,
                         fontweight="bold", pad=pad, x=-0.12)
        axes[0].set_ylabel("fraction of runs hidden")
    handles = [
        Line2D([], [], color=col_v, lw=1.4, marker="o", ms=3.2,
               label="Viterbi tracker"),
        Line2D([], [], color=col_f, lw=1.0, ls="--", marker="o", ms=3,
               mfc="white", label="fixed tests (best of MF, MTF)"),
        Patch(fc="#999999", alpha=0.35, lw=0, label="95% interval"),
    ]
    fig.legend(handles=handles, loc="outside upper center", ncol=3,
               frameon=False, fontsize=6.5,
               title=f"hiding = 1 $-$ detection rate at false-alarm rate {far_txt}",
               title_fontsize=6.5)
    return fig


def scatter(summary: dict, fixed, far: str, far_txt: str) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH_WIDE, 2.4),
                             layout="constrained", sharey=True)
    for ax, det, title in zip(axes, ("viterbi", "fixed"),
                              ("against the Viterbi tracker",
                               "against the fixed tests")):
        for fam, label, _, _ in _PANELS:
            pts = _cells(summary, fam, fixed, far)
            col = FAMILY_COLOR[fam]
            ax.plot([p["cv"] for p in pts], [p[det] for p in pts],
                    ls="none", marker=_MARKER[fam], ms=3.5, mfc=col, mec="k",
                    mew=0.3, alpha=0.9, label=label)
        ax.set_xscale("log")
        ax.set_xlabel("cadence coefficient of variation")
        ax.set_title(title)
        ax.set_ylim(-0.04, 1.04)
    axes[0].set_ylabel(f"hiding at FAR {far_txt}")
    axes[0].legend(loc="upper left", frameon=False, fontsize=5.5, ncol=1)
    return fig


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--summary", type=pathlib.Path, default=_SUMMARY)
    ap.add_argument("--far", type=float, default=0.01)
    ap.add_argument("--x", choices=("level", "cv"), default="level")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    summary = json.loads(args.summary.read_text())
    far = f"{args.far:g}"
    fixed = tuple(summary["detector_classes"]["fixed"])
    apply_house_style()
    if args.x == "level":
        fig = curves(summary, fixed, far, far)
        out = args.out or "attack_curves_aggregate"
    else:
        fig = scatter(summary, fixed, far, far)
        out = args.out or "attack_scatter_cv_aggregate"
    print(save(fig, out))


if __name__ == "__main__":
    main()
