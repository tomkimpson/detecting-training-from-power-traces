"""Minimum meter specification (Viterbi panel) -- v2 layout.

Same summary, same cells and same numbers as the ``plot_single`` panel of
``plot_st2_meter_boundary.py`` (its grid reader is imported, not re-derived);
only the presentation differs:

  * a single-hue sequential map (white -> house blue) in place of viridis, with
    thin white gaps between cells;
  * the live region (cells at or above LIVE_TPR) is outlined along cell edges,
    so the specification the text states is drawn rather than read off numbers;
  * no spines or tick marks, and a plain "detection rate" colour-bar label (the
    FAR is in the caption), as in scenario_meter_v2 and viterbi_track_v2.

Reproduce:
    python scripts/plot_st2_meter_boundary_v2.py [--scenario aggregate] [--far 0.01]
Outputs:
    figures/st2_meter_boundary[_aggregate]_viterbi_v2.{pdf,png}
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
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import plot_st2_meter_boundary as v1  # noqa: E402
from powerladder.plotstyle import C, WIDTH_ICML_COL, apply_house_style, save  # noqa: E402

# --- parameters (no magic numbers below) -------------------------------------
# Cells at or above this detection rate form the live region (the text's
# "full power, detection rate >~ 0.9").
LIVE_TPR = 0.9
CELL_GAP = 1.0           # white gap between cells [pt]
TEXT_DARK_ABOVE = 0.55   # cell value above which annotations switch to white
OUTLINE_C = C["black"]


def _live_outline(ax, mask: np.ndarray) -> None:
    """Draw the boundary of ``mask`` along cell edges (cell centres at ints)."""
    ny, nx = mask.shape
    pad = np.zeros((ny + 2, nx + 2), bool)
    pad[1:-1, 1:-1] = mask
    kw = dict(color=OUTLINE_C, lw=1.0, solid_capstyle="projecting")
    for i in range(ny + 1):          # horizontal edges between rows i-1 and i
        for j in range(nx):
            if pad[i, j + 1] != pad[i + 1, j + 1]:
                ax.plot([j - 0.5, j + 0.5], [i - 0.5, i - 0.5], **kw)
    for j in range(nx + 1):          # vertical edges between cols j-1 and j
        for i in range(ny):
            if pad[i + 1, j] != pad[i + 1, j + 1]:
                ax.plot([j - 0.5, j - 0.5], [i - 0.5, i + 0.5], **kw)


def plot(summary: dict, stem: str, far: float | None = None,
         detector: str = "viterbi") -> pathlib.Path:
    apply_house_style()
    far = v1._far_key(summary, far)
    Z, fs_axis, iw_axis = v1._main_grid(summary, detector, far)
    cmap = LinearSegmentedColormap.from_list("white_blue", ["white", C["blue"]])

    fig, ax = plt.subplots(figsize=(WIDTH_ICML_COL, 2.2))
    im = ax.pcolormesh(np.arange(len(fs_axis)), np.arange(len(iw_axis)), Z,
                       shading="nearest", cmap=cmap, vmin=0.0, vmax=1.0,
                       edgecolors="white", linewidth=CELL_GAP)
    for i in range(len(iw_axis)):
        for j in range(len(fs_axis)):
            if np.isfinite(Z[i, j]):
                ax.text(j, i, f"{Z[i, j]:.2f}", ha="center", va="center",
                        fontsize=5.5,
                        color="white" if Z[i, j] > TEXT_DARK_ABOVE else C["black"])
    _live_outline(ax, np.nan_to_num(Z) >= LIVE_TPR)

    ax.set_xticks(range(len(fs_axis)))
    ax.set_xticklabels([f"{v:g}" for v in fs_axis])
    ax.set_yticks(range(len(iw_axis)))
    ax.set_yticklabels([f"{v:g}" for v in iw_axis])
    ax.tick_params(which="both", length=0)
    ax.minorticks_off()
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xlabel(r"sample rate $f_s$ (Hz)")
    ax.set_ylabel(r"integration window $\tau$ (s)")

    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
    cb.set_label("detection rate")
    cb.outline.set_visible(False)
    cb.ax.minorticks_off()
    cb.ax.tick_params(length=0)
    fig.tight_layout()
    return save(fig, f"{stem}_viterbi_v2")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scenario", choices=("single", "aggregate"), default="aggregate")
    ap.add_argument("--far", type=float, default=0.01)
    args = ap.parse_args(argv)
    stem = "st2_meter_boundary" + ("" if args.scenario == "single"
                                   else f"_{args.scenario}")
    path = v1._SUMMARY.with_name(stem.removeprefix("st2_") + "_summary.json")
    print(plot(json.loads(path.read_text()), stem, far=args.far))


if __name__ == "__main__":
    main()
