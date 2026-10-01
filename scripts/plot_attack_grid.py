"""Attack grid: hiding for every schedule strategy at every swept level.

One row per schedule strategy of Tab. 1 (all eight, priced or not), one column
per attack level, weakest on the left. Cell colour is hiding (1 - detection rate
at the chosen FAR). Each cell carries its knob value and, where the single-A100
campaign priced it, the measured throughput overhead; unpriced cells show a dash.
Two stacked panels: the Viterbi tracker (top) and the fixed tests (bottom; the
better of matched filter and multitaper F per cell). The meter strategy degrades
the channel rather than the schedule (Sec. 5.2) and is left out.

Levels are ordered by attack strength: dilution by decreasing training share,
relocation by distance of the cadence from the band centre.

READ-ONLY: a pure reader of a frozen frontier_summary.json.

Reproduce:
    python scripts/plot_attack_grid.py
Outputs:
    figures/attack_grid_aggregate.{pdf,png}
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

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from powerladder.plotstyle import WIDTH_WIDE, apply_house_style, save  # noqa: E402

_SUMMARY = (pathlib.Path(__file__).resolve().parent.parent
            / "results" / "st2" / "aggregate" / "frontier_summary.json")

_BAND_CENTRE_HZ = 1.0  # centre of the 0.3-1.7 Hz search band (relocation ordering)

# (family, row label, knob symbol, strength key: level -> sort key)
_ROWS = [
    ("work", "work variation", r"$\sigma_G$", lambda v: v),
    ("jitter", "period jitter", r"$\sigma_\xi$", lambda v: v),
    ("drift", "cadence drift", r"$\sigma_f$", lambda v: v),
    ("shape", "shape filling", r"$\phi$", lambda v: v),
    ("phase", "phase slip", r"$\sigma_s$", lambda v: v),
    ("harmonic", "harmonic smoothing", r"$w$", lambda v: v),
    ("relocate", "relocation", r"$f_0$", lambda v: abs(v - _BAND_CENTRE_HZ)),
    ("dilute", "dilution", "share", lambda v: -v),
]


def _cost_label(c: float | None) -> str:
    if c is None:
        return "—"
    if c == 0.0:
        return "0%"
    if abs(c) < 1.0:
        return "≈0%"
    return f"{c:.0f}%"


def _rows(summary: dict, fixed: tuple[str, ...], far: str) -> list[dict]:
    rows = []
    for fam, label, sym, key in _ROWS:
        cells = sorted((c for c in summary["cells"] if c["family"] == fam),
                       key=lambda c: key(c["level"]))
        rows.append({
            "label": label, "sym": sym,
            "levels": [c["level"] for c in cells],
            "cost": [c.get("cost_overhead_pct") for c in cells],
            "viterbi": [1 - c["tpr_at_far"]["viterbi"][far] for c in cells],
            "fixed": [1 - max(c["tpr_at_far"][d][far] for d in fixed)
                      for c in cells],
        })
    return rows


def _panel(ax, rows: list[dict], det: str, ncol: int, cmap, title: str):
    grid = np.full((len(rows), ncol), np.nan)
    for i, r in enumerate(rows):
        grid[i, :len(r[det])] = r[det]
    im = ax.imshow(np.ma.masked_invalid(grid), cmap=cmap, vmin=0, vmax=1,
                   aspect="auto")
    for i, r in enumerate(rows):
        for j, (lev, cost, h) in enumerate(zip(r["levels"], r["cost"], r[det])):
            col = "white" if h > 0.55 else "black"
            ax.text(j, i - 0.17, f"{lev:g}", ha="center", va="center",
                    fontsize=6, color=col)
            ax.text(j, i + 0.2, _cost_label(cost), ha="center", va="center",
                    fontsize=5, color=col, alpha=0.85)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{r['label']} ({r['sym']})" for r in rows])
    ax.set_xticks([])
    ax.set_xticks(np.arange(-0.5, ncol), minor=True)
    ax.set_yticks(np.arange(-0.5, len(rows)), minor=True)
    ax.grid(which="minor", color="white", lw=1.2)
    ax.tick_params(which="both", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title, loc="left")
    return im


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--summary", type=pathlib.Path, default=_SUMMARY)
    ap.add_argument("--far", type=float, default=0.01)
    ap.add_argument("--out", default="attack_grid_aggregate")
    args = ap.parse_args()

    summary = json.loads(args.summary.read_text())
    far = f"{args.far:g}"
    fixed = tuple(summary["detector_classes"]["fixed"])
    rows = _rows(summary, fixed, far)
    ncol = max(len(r["levels"]) for r in rows)

    apply_house_style()
    cmap = plt.get_cmap("magma_r").copy()
    cmap.set_bad("white")
    fig, axes = plt.subplots(2, 1, figsize=(WIDTH_WIDE, 5.0),
                             layout="constrained", sharex=True)
    _panel(axes[0], rows, "viterbi", ncol, cmap, "against the Viterbi tracker")
    im = _panel(axes[1], rows, "fixed", ncol, cmap, "against the fixed tests")
    axes[1].set_xlabel("attack level (weakest → strongest)")
    cb = fig.colorbar(im, ax=axes, shrink=0.5, aspect=25, pad=0.02)
    cb.set_label(f"hiding at FAR {args.far:g}")
    cb.ax.minorticks_off()
    print(save(fig, args.out))


if __name__ == "__main__":
    main()
