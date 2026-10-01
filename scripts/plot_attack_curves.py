"""Attack response curves: hiding against attack level, one panel per strategy.

Small multiples over the eight schedule strategies of Tab. 1 (priced or not).
Each panel plots hiding (1 - detection rate at the chosen FAR) against the
strategy's own budget knob, for the Viterbi tracker (solid) and the fixed tests
(dashed; the better of matched filter and multitaper F per cell). Levels are
evenly spaced, weakest on the left; each is labelled with its knob value on the
bottom axis and, where the single-A100 campaign priced it, its measured
throughput overhead on the top axis. Dilution's levels run in decreasing
training share (lower share is the stronger attack); relocation plots the cadence
itself, so its attack grows toward both edges.

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

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from powerladder.plotstyle import (C, FAMILY_COLOR, WIDTH_WIDE,  # noqa: E402
                                   apply_house_style, save)

_SUMMARY = (pathlib.Path(__file__).resolve().parent.parent
            / "results" / "st2" / "aggregate" / "frontier_summary.json")

# (family, title, knob label, reverse level order)
_PANELS = [  # Tab. 3 (tab:attacks) order, read row by row
    ("jitter", "period jitter", r"$\sigma_\xi$", False),
    ("drift", "cadence drift", r"$\sigma_f$ [Hz]", False),
    ("work", "work variation", r"$\sigma_G$", False),
    ("phase", "phase slip", r"$\sigma_s$ [periods]", False),
    ("relocate", "relocation", r"$f_0$ [Hz]", False),
    ("harmonic", "harmonic smoothing", r"ramp $w$ [s]", False),
    ("shape", "shape filling", r"fill $\phi$", False),
    ("dilute", "dilution", "training share", True),
]
_MARKER = {"jitter": "o", "work": "s", "drift": "^", "shape": "D",
           "phase": "v", "harmonic": "P", "relocate": "X", "dilute": "*"}


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


def curves(summary: dict, fixed, far: str, far_txt: str) -> plt.Figure:
    fig, axes = plt.subplots(2, 4, figsize=(WIDTH_WIDE * 1.3, 4.0),
                             layout="constrained", sharey=True)
    for ax, (fam, title, xlab, inv) in zip(axes.flat, _PANELS):
        pts = _cells(summary, fam, fixed, far)
        if inv:
            pts = pts[::-1]
        x = range(len(pts))  # levels evenly spaced, weakest on the left
        ax.plot(x, [p["viterbi"] for p in pts], "-o", color=C["black"],
                ms=2.5, lw=1.0, label="Viterbi tracker", zorder=3)
        ax.plot(x, [p["fixed"] for p in pts], "--o", color=C["grey"],
                ms=2.5, mfc="white", lw=1.0, label="fixed tests", zorder=2)
        ax.set_ylim(-0.04, 1.04)
        ax.set_xlim(-0.4, len(pts) - 0.6)
        # knob value on the bottom axis, measured cost on the top axis
        ax.set_xticks(list(x))
        ax.set_xticklabels([f"{p['level']:g}" for p in pts], fontsize=5.5)
        ax.tick_params(axis="x", which="minor", bottom=False)
        ax.set_xlabel(xlab, labelpad=2)
        # unpriced strategies get no cost axis at all (caption: cf. Tab. 3)
        if any(p["cost"] is not None for p in pts):
            top = ax.secondary_xaxis("top")
            top.set_xticks(list(x))
            top.set_xticklabels([_cost_label(p["cost"]) if p["cost"] is not None
                                 else "" for p in pts], fontsize=5.5)
            top.set_xlabel("throughput cost [%]", labelpad=2)
            top.tick_params(axis="x", which="both", direction="in")
            top.minorticks_off()
        ax.text(0.04, 0.96, f"Strategy: {title}", transform=ax.transAxes,
                ha="left", va="top", fontsize=6.5)
    for ax in axes[:, 0]:
        ax.set_ylabel(f"hiding at FAR {far_txt}")
    # the harmonic-smoothing panel is empty above its floor line
    axes[1, 1].legend(loc="center left", frameon=False, fontsize=6)
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
