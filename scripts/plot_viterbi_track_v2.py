"""How the Viterbi tracker works -- v2 layout.

Same traces, same paths and same scores as ``plot_viterbi_track.py`` (imported,
not re-derived); only the presentation differs:

  * the contrast map is drawn in greyscale, capped at mid-grey, from L = 0 (the
    frame median) up, so only above-median bins carry ink and the path and true
    cadence, in colour, read on top of it;
  * bold panel letters, outward ticks, and no spines, as in scenario_meter_v2;
  * each panel states its score V against the detection threshold.

Reproduce:
    python scripts/plot_viterbi_track_v2.py [--seed 4]
Outputs:
    figures/viterbi_track_v2.{pdf,png}
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import plot_viterbi_track as v1  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
import numpy as np  # noqa: E402

from powerladder.plotstyle import C, WIDTH_WIDE, apply_house_style, save  # noqa: E402

# --- parameters (no magic numbers below) -------------------------------------
# Colour range of the contrast map [log-contrast]. Below-median bins (L < 0)
# carry no evidence for a line, so they are drawn blank.
L_MIN, L_MAX = 0.0, 3.5
# Greys truncated at mid-grey, so the coloured lines stay legible over the
# brightest bins.
CMAP_TOP = 0.7
PATH_C = C["blue"]
TRUTH_C = C["vermillion"]
T_MAX_S = 300.0
# Viterbi threshold at FAR 1e-2 on the aggregate inference null (paper
# subsec:threshold), shown beside each panel's score.
V_ALPHA = 0.91

PANEL_TITLE = {
    "stationary": "training, stationary",
    "wandering": f"training, {v1.WANDER_HZ:g} Hz wander",
    "inference": "inference null",
}


def plot(panels: dict[str, dict]) -> pathlib.Path:
    apply_house_style()
    cmap = LinearSegmentedColormap.from_list(
        "greys_top", plt.get_cmap("Greys")(np.linspace(0.0, CMAP_TOP, 256)))
    fig, axs = plt.subplots(1, 3, figsize=(WIDTH_WIDE * 1.35, 2.2), sharey=True,
                            constrained_layout=True)
    for i, (ax, kind) in enumerate(zip(axs, v1.PANELS)):
        rec = panels[kind]
        im = ax.pcolormesh(rec["times"], rec["freqs"], rec["L"], shading="nearest",
                           cmap=cmap, vmin=L_MIN, vmax=L_MAX, rasterized=True)
        if rec["truth"] is not None:
            ax.plot(rec["times"], rec["truth"], color=TRUTH_C, ls="--", lw=1.0,
                    label="true cadence")
        ax.plot(rec["times"], rec["path"], color=PATH_C, lw=1.4,
                label="Viterbi path")
        verdict = ">" if rec["V"] > V_ALPHA else "<"
        # Bottom-left: the one corner no path or cadence visits in any panel.
        ax.text(0.03, 0.04, rf"$V = {rec['V']:.2f} {verdict} v_\alpha$",
                transform=ax.transAxes, ha="left", va="bottom", fontsize=7,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none"))
        ax.text(0.0, 1.04, PANEL_TITLE[kind], transform=ax.transAxes,
                ha="left", va="bottom", fontsize=8)
        ax.text(-0.04, 1.04, "abc"[i], transform=ax.transAxes,
                fontweight="bold", fontsize=9, ha="right", va="bottom")
        ax.set_xlim(0, T_MAX_S)
        ax.set_xticks([0, 100, 200, 300])
        ax.set_xlabel("time (s)")
        ax.tick_params(which="both", top=False, right=False, direction="out")
        ax.minorticks_off()
        for s in ax.spines.values():
            s.set_visible(False)
    axs[0].set_ylabel("frequency (Hz)")
    handles, labels = axs[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2,
               fontsize=7, frameon=False)
    cb = fig.colorbar(im, ax=axs, pad=0.02, fraction=0.035)
    cb.set_label(r"$L_{kj}$")
    cb.outline.set_visible(False)
    cb.ax.minorticks_off()
    return save(fig, "viterbi_track_v2")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=v1.SEED)
    args = ap.parse_args()
    print(plot(v1.build(seed=args.seed)))


if __name__ == "__main__":
    main()
