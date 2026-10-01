"""Cost of hiding, one line per priced strategy family, on a single panel.

Companion to plot_money_pareto.py (pooled staircase per detector). Here each of
the four cost-anchored families (jitter, work variation, drift, power shaping)
gets its own line, coloured by family (the house FAMILY_COLOR map, as in the
App. D cost-of-hiding figure); the detector is encoded by line style: solid
lines and filled markers for the Viterbi tracker, dashed lines and hollow
markers for the fixed tests (better of matched filter and multitaper F).

--lines dial      joins each family's points in order of increasing attack level
--lines staircase draws each family's own Pareto staircase (best hiding the
                  family buys at or below each cost), over its measured range

The x-axis is symmetric-log (linear within +-10 %), as Fig. 7.

--split puts the two detectors in side-by-side panels (tracker left, fixed
tests right, shared y); line style then carries no information, so all lines
are solid and markers filled.

READ-ONLY: a pure reader of a frozen frontier_summary.json.

Reproduce:
    python scripts/plot_pareto_families.py --lines dial
Outputs:
    figures/pareto_families_<lines>.{pdf,png}   (or figures/<STEM>.*)
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

from powerladder.plotstyle import (C, WIDTH_ICML_COL, WIDTH_WIDE,  # noqa: E402
                                   apply_house_style, save)

_HERE = pathlib.Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("plot_st2_pareto",
                                               _HERE / "plot_st2_pareto.py")
_pp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_pp)

_SUMMARY = _HERE.parent / "results" / "st2" / "aggregate" / "frontier_summary.json"
_LINTHRESH = 10.0


def detectors(summary: dict) -> dict[str, tuple[tuple[str, ...], str, bool]]:
    """name -> (detector tuple, linestyle, filled markers)."""
    fixed = _pp.detector_classes(summary)["fixed"]
    return {"Viterbi tracker": (("viterbi",), "-", True),
            "fixed tests": (fixed, "--", False)}


def _draw(ax, anchored: list[dict], lines: str, ls: str, filled: bool) -> None:
    """One line per priced family on ``ax``."""
    for fam, col in _pp._FAMILY_COLOR.items():
        pts = [p for p in anchored if p["family"] == fam]
        if lines == "staircase":
            env = _pp.pareto_envelope(pts)
            ax.step([p[0] for p in env], [p[1] for p in env], where="post",
                    color=col, ls=ls, lw=1.1, zorder=2)
        else:
            dial = sorted(pts, key=lambda q: q["level"])
            ax.plot([p["cost"] for p in dial], [p["hiding"] for p in dial],
                    color=col, ls=ls, lw=1.1, zorder=2)
        ax.plot([p["cost"] for p in pts], [p["hiding"] for p in pts],
                ls="none", marker=_pp._FAMILY_MARKER[fam], ms=3.0,
                mfc=col if filled else "white", mec=col, mew=0.7, zorder=3)


def _axes_style(ax, x_hi: float) -> None:
    ax.set_xscale("symlog", linthresh=_LINTHRESH, linscale=0.6)
    ax.set_xlim(-3.0, x_hi * 1.6)
    ax.set_ylim(-0.04, 1.04)
    ax.set_xlabel("measured throughput overhead [%]")


def plot_split(summary: dict, far: str, lines: str, name: str,
               stack: bool = False) -> pathlib.Path:
    """Tracker and fixed tests in separate panels: side by side (full width) or,
    with ``stack``, one above the other at single-column width (shared x)."""
    apply_house_style()
    if stack:
        fig, axes = plt.subplots(2, 1, figsize=(WIDTH_ICML_COL, 3.9),
                                 sharex=True)
    else:
        fig, axes = plt.subplots(1, 2, figsize=(WIDTH_WIDE, 2.4), sharey=True)
    x_hi = max(c["cost_overhead_pct"] for c in summary["cells"]
               if c["cost_overhead_pct"] is not None)
    for ax, (det_name, (dets, _, _)) in zip(axes, detectors(summary).items()):
        anchored, _ = _pp.split_by_cost(summary["cells"], dets, far)
        _draw(ax, anchored, lines, "-", True)
        _axes_style(ax, x_hi)
        ax.set_title(f"against the {det_name}", fontsize=7)
        if stack:
            ax.set_ylabel(f"hiding at FAR {far}")
    if stack:
        axes[0].set_xlabel("")
    else:
        axes[0].set_ylabel(f"hiding at FAR {far}")
    handles = [Line2D([], [], color=col, lw=1.1,
                      marker=_pp._FAMILY_MARKER[f], ms=3, label=f)
               for f, col in _pp._FAMILY_COLOR.items()]
    # the tracker panel is empty above hiding ~0.6 away from zero cost
    axes[0].legend(handles=handles, frameon=False, fontsize=5.5,
                   loc="upper right", ncol=2)
    fig.tight_layout()
    return save(fig, name)


def plot(summary: dict, far: str, lines: str, name: str) -> pathlib.Path:
    apply_house_style()
    fig, ax = plt.subplots(figsize=(WIDTH_ICML_COL, 2.6))
    x_hi = max(c["cost_overhead_pct"] for c in summary["cells"]
               if c["cost_overhead_pct"] is not None)

    for det_name, (dets, ls, filled) in detectors(summary).items():
        anchored, _ = _pp.split_by_cost(summary["cells"], dets, far)
        for fam, col in _pp._FAMILY_COLOR.items():
            pts = [p for p in anchored if p["family"] == fam]
            if lines == "staircase":
                env = _pp.pareto_envelope(pts)
                ax.step([p[0] for p in env], [p[1] for p in env], where="post",
                        color=col, ls=ls, lw=1.1, zorder=2)
            else:
                dial = sorted(pts, key=lambda q: q["level"])
                ax.plot([p["cost"] for p in dial], [p["hiding"] for p in dial],
                        color=col, ls=ls, lw=1.1, zorder=2)
            ax.plot([p["cost"] for p in pts], [p["hiding"] for p in pts],
                    ls="none", marker=_pp._FAMILY_MARKER[fam], ms=3.0,
                    mfc=col if filled else "white", mec=col, mew=0.7,
                    zorder=3)

    ax.set_xscale("symlog", linthresh=_LINTHRESH, linscale=0.6)
    ax.set_xlim(-3.0, x_hi * 1.6)
    ax.set_ylim(-0.04, 1.04)
    ax.set_xlabel("measured throughput overhead [%]")
    ax.set_ylabel(f"hiding at FAR {far}")

    handles = [Line2D([], [], color=col, lw=1.1,
                      marker=_pp._FAMILY_MARKER[f], ms=3, label=f)
               for f, col in _pp._FAMILY_COLOR.items()]
    handles += [Line2D([], [], color=C["grey"], ls=ls, lw=1.1, marker="o",
                       ms=3, mfc=C["grey"] if filled else "white",
                       mec=C["grey"], label=n)
                for n, (_, ls, filled) in detectors(summary).items()]
    ax.legend(handles=handles, frameon=False, fontsize=5, loc="lower center",
              bbox_to_anchor=(0.5, 1.01), ncol=3, columnspacing=0.8,
              handlelength=2.0)
    fig.tight_layout()
    return save(fig, name)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--summary", type=pathlib.Path, default=_SUMMARY)
    ap.add_argument("--far", type=float, default=_pp._DEFAULT_FAR)
    ap.add_argument("--lines", choices=("dial", "staircase"), default="dial")
    ap.add_argument("--split", action="store_true",
                    help="tracker and fixed tests in separate panels")
    ap.add_argument("--stack", action="store_true",
                    help="with --split: panels stacked vertically, column width")
    ap.add_argument("--out", default=None,
                    help="figure stem (default pareto_families_<lines>)")
    args = ap.parse_args()
    summary = json.loads(args.summary.read_text())
    far = _pp.far_key(summary, args.far)
    if args.split:
        out = plot_split(summary, far, args.lines,
                         args.out or (f"pareto_families_{args.lines}_split"
                                      + ("_stacked" if args.stack else "")),
                         stack=args.stack)
    else:
        out = plot(summary, far, args.lines,
                   args.out or f"pareto_families_{args.lines}")
    print(f"-> {out.with_suffix('')}.*")


if __name__ == "__main__":
    main()
