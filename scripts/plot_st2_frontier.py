"""ST2 frontier — detector power vs physical de-periodicisation, all families.

Task 20.8 (plan-for-paper-2 §5: the Pareto frontier is a central result). This
assembles the per-family sweeps (results/st2/*_summary.json, task 20.7) onto
the SHARED physical x-axis (cadence CV; phase diffusion D in the companion
panel), attaches the measured b2 systems-cost anchors where the same operator
was run on hardware, and evaluates the PRE-REGISTERED ST2 go/no-go criteria
mechanically.

Cost-anchor mapping (families without an entry are analytic/qualitative only):

    jitter  -> data/measured_cost_anchors/spoof_summary.json
               throughput_overhead["jitter=L"]
               (idle-PAD realisation of timing jitter: the expensive way)
    drift   -> data/measured_cost_anchors/spoof_summary.json
               throughput_overhead["drift=L"]
               (pad realisation; measured grid 0.2/0.5/1 Hz, so only shared
               levels get an anchor)
    work    -> data/measured_cost_anchors/workjitter_summary.json
               throughput_overhead["jitter=L"]
               (REAL-WORK realisation of the same sigma grid: ~zero cost)
    shape   -> data/measured_cost_anchors/shaped_summary.json
               levels["shaped-jitter=L"]
               (phi grid matches; NOTE the measured anchor was captured on a
               sigma=0.35 timing-spoof base, so it UPPER-bounds shaping-only)

The anchors are the MEASURED single-A100 B2-campaign throughput overheads,
carried in as static, non-regenerable input data (see
data/measured_cost_anchors/README.md). Fractional overheads (mean, std),
reported here as percent. Only the CLEAN cost-side field (throughput_overhead)
is consumed; the detection-side line-band fields in those same files ride the
signature-B side channel and carry the issue #54 requalification — but the
frontier's own detection numbers come from results/st2/*_summary.json, so the
cost axis is unaffected.

The verdict is PROVISIONAL only when the inputs carry just {spectral, viterbi}
(then "tracking" = viterbi, "fixed" = spectral); it is definitive once the ST1
detectors (mtf, dg_order_*) are present in the per-family summaries — which is
the full-resolution freeze this assembles.

Reproduce (full-resolution numbers frozen on Slurm — the per-family sweeps run
as an array via scripts/slurm/st2_sweeps.sbatch, then this assembly step runs
via scripts/slurm/st2_frontier.sbatch, chained afterok):
    python scripts/plot_st2_sweeps.py   # (inputs; on Slurm for the freeze)
    python scripts/plot_st2_frontier.py
Output:
    results/st2/frontier_summary.json ; figures/st2_frontier.png/.pdf
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

from powerladder.plotstyle import C, WIDTH_WIDE, apply_house_style, save  # noqa: E402
from powerladder.typeb.st2_attacks import FAMILY_ORDER  # noqa: E402

_ROOT = pathlib.Path(__file__).resolve().parent.parent
_ST2 = _ROOT / "results" / "st2"
# The measured cost anchors are committed static input (NOT under results/;
# this repo is CPU-only and cannot regenerate them). See the module docstring
# and data/measured_cost_anchors/README.md.
_B2 = _ROOT / "data" / "measured_cost_anchors"

# --- pre-registered gate constants (plan we-are-working-on §"ST2 go/no-go") --
_FAR_KEY = "0.05"            # the operational FAR the criteria are stated at
_TRACK_MIN = 0.8             # tracking detector must hold TPR >= this
_FIXED_MAX = 0.5             # ...while EVERY fixed test <= this
_CHEAP_PCT = 20.0            # "modest cost" anchor threshold [%]
_ERASED = 0.5                # "erased": all detectors below this
_MATCH_MARGIN = 0.1          # NO_GO: tracking never beats fixed by more

# Detector classes. The verdict uses whichever members are present in the
# loaded summaries: with only the pre-ST1 pair {spectral, viterbi} the verdict
# is PROVISIONAL; once the ST1 detectors are registered (task 20.9) it is
# definitive. Candidate membership is fixed here, pre-registered.
_TRACKING_ALL = ("viterbi", "dg_order_full", "dg_order_semicoh")
_FIXED_ALL = ("spectral", "mtf")
# effective sets, filtered to the summaries' detectors at load time
_TRACKING: tuple = _TRACKING_ALL
_FIXED: tuple = _FIXED_ALL


def _set_detector_classes(available: set[str]) -> bool:
    """Filter the pre-registered classes to what the summaries contain.

    Returns provisional=True when none of the ST1 detectors are present.
    """
    global _TRACKING, _FIXED
    _TRACKING = tuple(d for d in _TRACKING_ALL if d in available)
    _FIXED = tuple(d for d in _FIXED_ALL if d in available)
    if not _TRACKING or not _FIXED:
        raise SystemExit(f"summaries carry detectors {sorted(available)} — "
                         "need at least one tracking and one fixed detector")
    st1_present = {"mtf", "dg_order_full", "dg_order_semicoh"} & available
    return not st1_present

_FAMILY_COLOR = {
    "jitter": C["orange"], "work": C["blue"], "drift": C["green"],
    "phase": C["vermillion"], "relocate": C["purple"], "harmonic": C["yellow"],
    "shape": C["skyblue"], "dilute": C["black"], "meter": C["grey"],
}


def _load_family_summaries(st2_dir: pathlib.Path) -> dict[str, dict]:
    out = {}
    for fam in FAMILY_ORDER:
        f = st2_dir / f"{fam}_summary.json"
        if f.exists():
            out[fam] = json.loads(f.read_text())
    if not out:
        raise SystemExit(f"no <family>_summary.json under {st2_dir} — "
                         "run scripts/plot_st2_sweeps.py first")
    return out


def _cost_anchors(b2_dir: pathlib.Path) -> dict[str, dict[float, tuple]]:
    """family -> {level -> (mean_frac, std_frac)} from the measured b2 sweeps.

    Only levels shared with the measured grids appear (missing measured file
    or level -> no anchor for that cell).
    """
    anchors: dict[str, dict[float, tuple]] = {}

    def read(name):
        f = b2_dir / name
        return json.loads(f.read_text()) if f.exists() else None

    spoof = read("spoof_summary.json")
    if spoof:
        oh = spoof["throughput_overhead"]
        anchors["jitter"] = {float(k.split("=")[1]): tuple(v)
                             for k, v in oh.items() if k.startswith("jitter=")}
        anchors["drift"] = {float(k.split("=")[1]): tuple(v)
                            for k, v in oh.items() if k.startswith("drift=")}
    workj = read("workjitter_summary.json")
    if workj:
        anchors["work"] = {float(k.split("=")[1]): tuple(v)
                           for k, v in workj["throughput_overhead"].items()
                           if k.startswith("jitter=")}
    shaped = read("shaped_summary.json")
    if shaped:
        anchors["shape"] = {
            float(k.split("=")[1]): tuple(v["throughput_overhead"])
            for k, v in shaped["levels"].items()
            if k.startswith("shaped-jitter=")}
    return anchors


_ANCHOR_SOURCE = {
    "jitter": "data/measured_cost_anchors/spoof_summary.json (idle-pad realisation)",
    "drift": "data/measured_cost_anchors/spoof_summary.json (idle-pad realisation)",
    "work": "data/measured_cost_anchors/workjitter_summary.json (real-work realisation)",
    "shape": ("data/measured_cost_anchors/shaped_summary.json (shaped-jitter basis, "
              "sigma=0.35 timing base -> upper bound on shaping-only cost)"),
}


def build_cells(summaries: dict[str, dict],
                anchors: dict[str, dict[float, tuple]]) -> list[dict]:
    """One frontier cell per family x level."""
    cells = []
    for fam, s in summaries.items():
        fam_anchor = anchors.get(fam, {})
        for pt in s["points"]:
            level = pt["level"]
            anchor = (fam_anchor.get(float(level))
                      if not isinstance(level, str) else None)
            cells.append({
                "family": fam,
                "level": level,
                "cadence_cv": pt["cadence_cv"],
                "phase_diffusion_D": pt["phase_diffusion_D"],
                "auc": pt["auc"],
                "tpr_at_far": pt["tpr_at_far"],
                "cost_overhead_pct": (100.0 * anchor[0] if anchor else None),
                "cost_overhead_pct_std": (100.0 * anchor[1] if anchor
                                          else None),
                "cost_anchor_source": (_ANCHOR_SOURCE[fam] if anchor
                                       else "analytic/qualitative only"),
                "learning_cost": None,   # reserved (Phase 2, user decision)
            })
    return cells


def _track_tpr(cell) -> float:
    return max(cell["tpr_at_far"][d][_FAR_KEY] for d in _TRACKING)


def _fixed_ok(cell) -> bool:
    return all(cell["tpr_at_far"][d][_FAR_KEY] <= _FIXED_MAX for d in _FIXED)


def _is_win(cell) -> bool:
    return _track_tpr(cell) >= _TRACK_MIN and _fixed_ok(cell)


def _is_cheap(cell) -> bool:
    """~zero/modest measured cost: the work family (measured ~zero within
    noise, plan §5) or any cell whose measured anchor is < _CHEAP_PCT %."""
    if cell["family"] == "work":
        return True
    pct = cell["cost_overhead_pct"]
    return pct is not None and pct < _CHEAP_PCT


def _cell_id(cell) -> str:
    return f"{cell['family']}={cell['level']}"


def evaluate_verdict(cells: list[dict],
                     summaries: dict[str, dict]) -> dict:
    """Apply the pre-registered ST2 gate criteria mechanically.

    GO      exists a contiguous (in level order) run of cells, within a
            ~zero/modest-cost family, where the best tracking detector holds
            TPR@0.05 >= 0.8 while every fixed test <= 0.5.
    REFRAME no GO, tracking wins only in expensive families, and some
            zero-cost (work) attack level erases ALL detectors (< 0.5).
    NO_GO   fixed matches tracking everywhere (margin < 0.1 at every cell).
    Precedence GO > REFRAME > NO_GO; anything else is UNDETERMINED.
    """
    by_family: dict[str, list[dict]] = {}
    for c in cells:
        by_family.setdefault(c["family"], []).append(c)
    # preserve the swept level order within each family
    for fam, s in summaries.items():
        order = {json.dumps(lv): i for i, lv in enumerate(s["levels"])}
        by_family[fam].sort(key=lambda c: order[json.dumps(c["level"])])

    # --- GO: contiguous cheap winning run ------------------------------------
    go_runs = []
    for fam, fam_cells in by_family.items():
        run: list[dict] = []
        for c in fam_cells + [None]:
            if c is not None and _is_win(c) and _is_cheap(c):
                run.append(c)
            else:
                if run:
                    go_runs.append((fam, [_cell_id(x) for x in run]))
                run = []
    go = bool(go_runs)

    # --- REFRAME: wins only where expensive; zero-cost erases everything -----
    expensive_wins = [_cell_id(c) for c in cells
                      if _is_win(c) and not _is_cheap(c)]
    all_dets = _TRACKING + _FIXED
    erased_zero_cost = [
        _cell_id(c) for c in by_family.get("work", [])
        if all(c["tpr_at_far"][d][_FAR_KEY] < _ERASED for d in all_dets)]
    reframe = (not go) and bool(expensive_wins) and bool(erased_zero_cost)

    # --- NO_GO: fixed ~ tracking everywhere ----------------------------------
    margins = {
        _cell_id(c): _track_tpr(c) - max(c["tpr_at_far"][d][_FAR_KEY]
                                         for d in _FIXED)
        for c in cells}
    no_go = max(margins.values()) < _MATCH_MARGIN

    if go:
        verdict, supporting = "GO", go_runs
        fired = (f"GO: contiguous run(s) in a ~zero/modest-cost family where "
                 f"best tracking TPR@{_FAR_KEY} >= {_TRACK_MIN} while every "
                 f"fixed test <= {_FIXED_MAX}")
    elif reframe:
        verdict = "REFRAME"
        supporting = {"expensive_wins": expensive_wins,
                      "zero_cost_erasure": erased_zero_cost}
        fired = ("REFRAME: tracking wins only in expensive families while a "
                 "zero-cost attack erases all detectors")
    elif no_go:
        verdict, supporting = "NO_GO", {"max_margin": max(margins.values())}
        fired = (f"NO_GO: fixed matches tracking everywhere "
                 f"(max TPR margin < {_MATCH_MARGIN})")
    else:
        verdict, supporting = "UNDETERMINED", {"margins": margins}
        fired = "no pre-registered criterion fired cleanly"

    return {
        "verdict": verdict,
        "provisional": True,  # overwritten by main() from detector presence
        "note": ("verdict is PROVISIONAL when only the pre-ST1 pair "
                 "{spectral, viterbi} is registered; definitive once the ST1 "
                 "detectors are in the summaries (task 20.9)"),
        "criterion_fired": fired,
        "supporting_cells": supporting,
        "detector_classes": {"tracking": list(_TRACKING),
                             "fixed": list(_FIXED)},
        "thresholds": {"far": float(_FAR_KEY), "tracking_min": _TRACK_MIN,
                       "fixed_max": _FIXED_MAX, "cheap_pct": _CHEAP_PCT,
                       "erased_below": _ERASED,
                       "match_margin": _MATCH_MARGIN},
    }


def plot_frontier(cells: list[dict]) -> pathlib.Path:
    """Headline figure: TPR@0.05 vs cadence CV (left) and vs D (right)."""
    apply_house_style()
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(WIDTH_WIDE, 2.7),
                                   sharey=True)
    by_family: dict[str, list[dict]] = {}
    for c in cells:
        by_family.setdefault(c["family"], []).append(c)

    # curves show the class envelopes the verdict is stated over: best
    # tracking detector (solid) and best fixed test (dashed) per cell
    def _cls_tpr(cell, dets):
        return max(cell["tpr_at_far"][d][_FAR_KEY] for d in dets)

    for fam, fam_cells in by_family.items():
        col = _FAMILY_COLOR[fam]
        for dets, ls in [(_TRACKING, "-"), (_FIXED, "--")]:
            for ax, xkey in [(axL, "cadence_cv"), (axR, "phase_diffusion_D")]:
                pts = sorted(fam_cells, key=lambda c: c[xkey])
                xs = [c[xkey] for c in pts]
                ys = [_cls_tpr(c, dets) for c in pts]
                ax.plot(xs, ys, color=col, ls=ls, lw=1.1, marker="o", ms=2.2,
                        alpha=0.9)
        # annotate measured cost anchors on the tracking curve (CV panel);
        # only where the CV axis separates points — the non-timing families
        # cluster at the honest baseline CV and their labels would overprint
        # (their anchors stay in frontier_summary.json).
        for c in fam_cells:
            pct = c["cost_overhead_pct"]
            if pct is not None and c["cadence_cv"] > 0.12:
                txt = "~0%" if abs(pct) < 1.0 else f"{pct:.0f}%"
                axL.annotate(txt,
                             (c["cadence_cv"], _cls_tpr(c, _TRACKING)),
                             textcoords="offset points", xytext=(2, -7),
                             fontsize=4.5, color=col)

    for ax, xlab in [(axL, "cadence CV (ground truth)"),
                     (axR, r"phase diffusion $D$ (rad$^2$/s)")]:
        ax.set_xscale("log")
        ax.set_ylim(0, 1.04)
        ax.set_xlabel(xlab)
        ax.axhline(_TRACK_MIN, color=C["grey"], lw=0.6, ls=":", zorder=0)
        ax.axhline(_FIXED_MAX, color=C["grey"], lw=0.6, ls=":", zorder=0)
    axL.set_ylabel(f"Detection rate at FAR = {_FAR_KEY}")

    handles = [Line2D([], [], color=_FAMILY_COLOR[f], lw=1.2, label=f)
               for f in by_family]
    handles += [Line2D([], [], color=C["black"], ls="-", lw=1.1,
                       label="tracking (best of "
                             + "/".join(_TRACKING) + ")"),
                Line2D([], [], color=C["black"], ls="--", lw=1.1,
                       label="fixed (best of " + "/".join(_FIXED) + ")")]
    axR.legend(handles=handles, frameon=False, fontsize=5,
               loc="lower left", ncol=2)
    st1_in = any(d != "viterbi" for d in _TRACKING)
    tag = "" if st1_in else "PROVISIONAL: pre-ST1 detector set; "
    fig.suptitle("ST2 de-periodicisation frontier "
                 f"({tag}% = measured throughput overhead)", fontsize=7)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return save(fig, "st2_frontier")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--st2-dir", default=str(_ST2),
                    help="directory of <family>_summary.json inputs")
    ap.add_argument("--b2-dir", default=str(_B2),
                    help="directory of measured b2 cost-anchor summaries")
    args = ap.parse_args()

    st2_dir, b2_dir = pathlib.Path(args.st2_dir), pathlib.Path(args.b2_dir)
    summaries = _load_family_summaries(st2_dir)
    available = set(next(iter(summaries.values()))["detectors"])
    provisional = _set_detector_classes(available)
    cells = build_cells(summaries, _cost_anchors(b2_dir))
    verdict = evaluate_verdict(cells, summaries)
    verdict["provisional"] = provisional

    any_s = next(iter(summaries.values()))
    frontier = {
        "provisional": provisional,
        "n_each": any_s["n_each"],
        "target_fars": any_s["target_fars"],
        "seed": any_s["seed"],
        "detector_classes": verdict["detector_classes"],
        "cost_anchor_mapping": _ANCHOR_SOURCE,
        "cells": cells,
        "verdict": verdict,
    }
    st2_dir.mkdir(parents=True, exist_ok=True)
    (st2_dir / "frontier_summary.json").write_text(
        json.dumps(frontier, indent=2))

    fig_pdf = plot_frontier(cells)
    print(f"{verdict['verdict']} ({verdict['criterion_fired']})")
    print("supporting:", json.dumps(verdict["supporting_cells"]))
    print(f"-> {st2_dir / 'frontier_summary.json'} ; "
          f"{fig_pdf.with_suffix('')}.*")


if __name__ == "__main__":
    main()
