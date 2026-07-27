"""Cost-vs-hiding Pareto view of the frozen ST2 frontier (plan-for-paper-2 §5).

The frontier figure (scripts/plot_st2_frontier.py) plots detection against a
PHYSICAL de-periodicisation axis (cadence CV, phase diffusion D) and carries the
measured systems cost only as small text annotations. The plan's stated
deliverable is the trade-off itself:

    how much adversary cost buys how much hiding, and from WHICH detector class?

so this script re-plots the same frozen cells with measured throughput overhead
on the x-axis and hiding = 1 - TPR@0.05 on the y-axis, one panel per detector
class, with the Pareto staircase (best hiding available at or below a given cost)
drawn per panel.

READ-ONLY. This is a pure reader of the tracked results/st2/frontier_summary.json
-- it computes nothing and writes no results artefact. (plot_st2_frontier.py, by
contrast, ASSEMBLES that summary from the per-family sweeps plus the measured
cost anchors, so it must not be re-run casually; the frozen numbers are the
slurm freeze recorded in notes/results/st2-frontier-freeze-findings.md.) There is
deliberately no --smoke path: nothing is computed, so there is nothing to
plumbing-check and no way to corrupt the tracked artefact.

Only the four cost-anchored families carry a measured overhead (jitter, drift,
work, shape -- see the summary's cost_anchor_mapping); the rest are
analytic/qualitative only and cannot be placed on a cost axis. Those cells are
NOT silently dropped: they are drawn in a hatched "cost not measured" strip and
counted on stdout, because one of them (work=0.7) is the level at which the
tracker finally bends, and hiding that would misread as "the tracker never
fails". The learning-efficiency cost leg is descoped (2026-07-22) and stays the
stated open empirical question; this figure plots the measured systems-cost axis
only.

Reproduce:
    python scripts/plot_st2_pareto.py [--summary PATH]
Outputs:
    figures/st2_cost_pareto.{pdf,png}
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from powerladder.plotstyle import (C, FAMILY_COLOR, WIDTH_WIDE,  # noqa: E402
                                   apply_house_style, save)
from powerladder.typeb.meter_boundary import FIXED, TRACKING  # noqa: E402

_ROOT = pathlib.Path(__file__).resolve().parent.parent
_SUMMARY = _ROOT / "results" / "st2" / "frontier_summary.json"

_FAR_KEY = "0.05"     # the operational FAR the frontier's verdict is stated at

# Only the cost-anchored families can appear on a cost axis; the marker map
# enumerates them, and the colours come from the shared house map so this figure
# and the frontier figure cannot drift apart.
_FAMILY_MARKER = {"jitter": "o", "work": "s", "drift": "^", "shape": "D"}
_FAMILY_COLOR = {f: FAMILY_COLOR[f] for f in _FAMILY_MARKER}

# Attack levels that hide less than this are left unlabelled in the figure: they
# sit on the hiding=0 floor where per-point labels overprint. Nothing is dropped
# from the data or the stdout table -- this is a legibility threshold only.
_LABEL_MIN_HIDING = 0.05


def detector_classes(summary: dict) -> dict[str, tuple[str, ...]]:
    """Read the pre-registered classes from the summary it accompanies.

    Taken from the frozen file rather than hardcoded, so the figure cannot
    disagree with the verdict computed over those same classes. Cross-checked
    against the library constants to catch a summary written by a different
    class definition.
    """
    cls = summary["detector_classes"]
    tracking, fixed = tuple(cls["tracking"]), tuple(cls["fixed"])
    if tracking != TRACKING or fixed != FIXED:
        raise SystemExit(
            f"summary detector_classes {cls} disagree with the library "
            f"constants (tracking={TRACKING}, fixed={FIXED})")
    return {"tracking": tracking, "fixed": fixed}


def hiding(cell: dict, dets: tuple[str, ...], far: str = _FAR_KEY) -> float:
    """Hiding achieved against a detector class = 1 - best detection rate.

    The class envelope (best member) is the same reduction the frontier verdict
    uses, so a cell hides from the class only if it hides from every member.
    """
    return 1.0 - max(cell["tpr_at_far"][d][far] for d in dets)


def split_by_cost(cells: list[dict], dets: tuple[str, ...],
                  far: str = _FAR_KEY) -> tuple[list[dict], list[dict]]:
    """(anchored, unpriced) points, each as {family, level, cost, hiding}.

    A cell is anchored when the measured-cost join in plot_st2_frontier.py found
    a hardware anchor at exactly its attack level; otherwise cost_overhead_pct is
    None ("analytic/qualitative only") and it cannot go on a cost axis.
    """
    anchored, unpriced = [], []
    for c in cells:
        pt = {"family": c["family"], "level": c["level"],
              "cost": c["cost_overhead_pct"], "hiding": hiding(c, dets, far)}
        (anchored if pt["cost"] is not None else unpriced).append(pt)
    # An anchored family with no marker would move the Pareto staircase while its
    # own points went undrawn -- the staircase IS the figure's claim, so fail loudly
    # rather than show a curve nothing accounts for.
    unknown = {p["family"] for p in anchored} - set(_FAMILY_MARKER)
    if unknown:
        raise SystemExit(
            f"cost-anchored families {sorted(unknown)} have no marker/colour; "
            "add them to _FAMILY_MARKER or they will silently skew the staircase")
    return anchored, unpriced


def pareto_envelope(points: list[dict]) -> list[tuple[float, float]]:
    """The cost-vs-hiding Pareto staircase: best hiding at or below each cost.

    Sorted by cost with a running maximum, so the result is monotone
    non-decreasing in hiding -- literally "this much cost buys at most this much
    hiding". Unpriced points cannot participate (no x coordinate).
    """
    best, out = -1.0, []
    for p in sorted(points, key=lambda q: q["cost"]):
        best = max(best, p["hiding"])
        out.append((p["cost"], best))
    return out


def plot(summary: dict) -> pathlib.Path:
    """Two panels: hiding vs measured cost, against each detector class."""
    apply_house_style()
    classes = detector_classes(summary)
    cells = summary["cells"]

    fig, axes = plt.subplots(1, 2, figsize=(WIDTH_WIDE, 2.7), sharey=True)
    titles = {"tracking": "vs the tracking class",
              "fixed": "vs the fixed class"}

    # A common "cost not measured" strip to the right of the measured range, so
    # the unpriced cells stay visible instead of being silently dropped.
    costs = [c["cost_overhead_pct"] for c in cells
             if c["cost_overhead_pct"] is not None]
    x_hi = max(costs)
    strip_lo, strip_hi = x_hi * 1.06, x_hi * 1.26

    # unpriced-ness is a property of the cost anchor alone, not of the detector
    # class, so it is the same in both panels -- count it once, outside the loop
    n_unpriced = sum(1 for c in cells if c["cost_overhead_pct"] is None)

    for ax, key in zip(axes, ("tracking", "fixed")):
        dets = classes[key]
        anchored, unpriced = split_by_cost(cells, dets)

        env = pareto_envelope(anchored)
        ax.step([p[0] for p in env], [p[1] for p in env], where="post",
                color=C["grey"], lw=1.0, ls="-", zorder=1)

        for fam, col in _FAMILY_COLOR.items():
            # order the family polyline by ATTACK LEVEL (the adversary's dial),
            # not by cost: the work family's measured overheads are all ~zero and
            # not monotone in level, so a cost ordering would scramble it.
            pts = sorted([p for p in anchored if p["family"] == fam],
                         key=lambda q: q["level"])
            if not pts:
                continue
            ax.plot([p["cost"] for p in pts], [p["hiding"] for p in pts],
                    color=col, marker=_FAMILY_MARKER[fam], ms=3, lw=1.0,
                    alpha=0.9, zorder=2)
            # Label only the levels that hide something. The rest pile up on the
            # hiding=0 floor where the labels would overprint illegibly; every
            # cell is listed on stdout and in the summary regardless.
            labelled = [p for p in pts if p["hiding"] > _LABEL_MIN_HIDING]
            for i, p in enumerate(labelled):
                # alternate the offset: adjacent levels can land on nearly the
                # same point (the work family's costs are all ~zero)
                dy = 2.0 if i % 2 == 0 else -6.0
                ax.annotate(f"{p['level']:g}", (p["cost"], p["hiding"]),
                            textcoords="offset points", xytext=(2.5, dy),
                            fontsize=4.5, color=col)

        # the unpriced cells, in the hatched strip
        ax.axvspan(strip_lo, strip_hi, facecolor="none", edgecolor=C["grey"],
                   hatch="////", lw=0.4, alpha=0.6, zorder=0)
        span = strip_hi - strip_lo
        fams = sorted({p["family"] for p in unpriced})
        for i, fam in enumerate(fams):
            xs = strip_lo + span * (i + 0.5) / len(fams)
            pts = [p for p in unpriced if p["family"] == fam]
            ax.plot([xs] * len(pts), [p["hiding"] for p in pts], ls="none",
                    marker="_", ms=3.5, color=_FAMILY_COLOR.get(fam, C["grey"]),
                    alpha=0.8, zorder=2)

        ax.set_xlim(-0.06 * x_hi, strip_hi + 0.03 * x_hi)
        ax.set_ylim(-0.04, 1.04)
        ax.set_xlabel("measured throughput overhead [%]")
        ax.set_title(titles[key], fontsize=7)

    axes[0].set_ylabel(f"hiding  $1-$ detection rate at FAR $= {_FAR_KEY}$")
    handles = [Line2D([], [], color=_FAMILY_COLOR[f], lw=1.0,
                      marker=_FAMILY_MARKER[f], ms=3, label=f)
               for f in _FAMILY_COLOR]
    handles += [Line2D([], [], color=C["grey"], lw=1.0, label="Pareto staircase"),
                Line2D([], [], color=C["grey"], lw=0, marker="_", ms=4,
                       label=f"cost not measured ({n_unpriced} cells, hatched)")]
    # the tracking panel is empty above hiding ~0.2, so the legend goes there
    axes[0].legend(handles=handles, frameon=False, fontsize=5, loc="upper left",
                   ncol=2)
    fig.suptitle("Cost of hiding: what measured adversary cost buys against "
                 "each detector class", fontsize=7)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return save(fig, "st2_cost_pareto")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--summary", type=pathlib.Path, default=_SUMMARY,
                    help="path to the frozen frontier_summary.json (read only)")
    args = ap.parse_args()

    if not args.summary.exists():
        raise SystemExit(f"{args.summary} not found — it is a tracked artefact; "
                         "see README (ST2 frontier) to regenerate on slurm")
    summary = json.loads(args.summary.read_text())
    classes = detector_classes(summary)

    for key in ("tracking", "fixed"):
        anchored, unpriced = split_by_cost(summary["cells"], classes[key])
        print(f"{key} class (best of {'/'.join(classes[key])}):")
        for p in sorted(anchored, key=lambda q: q["cost"]):
            print(f"  {p['family']:7s} {p['level']:<5g} "
                  f"cost={p['cost']:+8.2f}%  hiding={p['hiding']:.2f}")
        notable = ", ".join(f"{p['family']}={p['level']}"
                            for p in unpriced if p["hiding"] > 0.3)
        print(f"  ({len(unpriced)} of {len(summary['cells'])} cells carry no "
              "measured cost anchor and are drawn in the hatched strip; those "
              f"hiding > 0.3 from this class: {notable or 'none'})")

    out = plot(summary)
    print(f"-> {out.with_suffix('')}.*")


if __name__ == "__main__":
    main()
