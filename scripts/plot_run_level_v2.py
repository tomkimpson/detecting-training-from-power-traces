"""Run-level detection figure -- v2 layout.

Same summary and same numbers as ``run_level_detection.py --replot`` (its
constants, labels and colours are imported, not re-derived); only the
presentation differs:

  * open spines (left + bottom), outward ticks, no minor ticks, and units in
    parentheses, as in scenario_meter_v2;
  * both panels share the same y-range, and the counting-rule circles at the
    24 h checkpoint are no longer cut by the axis edge;

Compute-free: reads results/st2/run_level_summary.json.

Reproduce:
    python scripts/plot_run_level_v2.py
Outputs:
    figures/run_level_detection_v2.{pdf,png}
"""

from __future__ import annotations

import json
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import run_level_detection as v1  # noqa: E402
from powerladder.plotstyle import C, WIDTH_WIDE, apply_house_style, save  # noqa: E402

# --- parameters (no magic numbers below) -------------------------------------
YLIM = (-0.02, 1.04)
X_PAD_H = 0.6            # room past the last checkpoint so its marker is whole [h]
MARKER = dict(ls="none", marker="o", ms=3.2, mfc="white", mew=0.9)


def _style(ax) -> None:
    ax.tick_params(which="both", top=False, right=False, direction="out")
    ax.minorticks_off()
    ax.set_ylim(*YLIM)
    ax.set_yticks(np.linspace(0, 1, 6))


def plot(summary: dict) -> pathlib.Path:
    apply_house_style()
    fig, (a, b) = plt.subplots(1, 2, figsize=(WIDTH_WIDE, 2.3))
    arms = summary["arms"]
    attacks = [arm for arm in v1.ARMS if arm != "honest"]

    for arm in attacks:
        a.scatter(np.array(arms[arm]["f0"], dtype=float),
                  arms[arm]["per_run_window_rate"], s=5, color=v1._COLOR[arm],
                  label=v1._LABEL[arm], alpha=0.8, lw=0)
    a.axhline(v1.WINDOW_FAR, color=C["grey"], lw=0.8, ls=":")
    a.text(0.02, v1.WINDOW_FAR + 0.02, "null rate", color=C["grey"], fontsize=6,
           ha="left", va="bottom", transform=a.get_yaxis_transform())
    a.set_xlabel(r"run cadence $f_0$ (Hz)")
    a.set_ylabel("fraction of windows detected")
    a.legend(loc="upper right", fontsize=6, frameon=False, handletextpad=0.2)
    _style(a)

    run_h = summary["params"]["run_hours"]
    w_h = summary["params"]["window_s"] / 3600
    for arm in attacks:
        c = arms[arm]["p_detected_by_window"]
        hrs = w_h * np.arange(1, len(c["eprocess"]) + 1)
        b.step(hrs, c["eprocess"], where="post", color=v1._COLOR[arm], lw=1.2)
        pts = arms[arm]["p_detected_by_hours"]
        b.plot([float(h) for h in pts], [v["count"] for v in pts.values()],
               color=v1._COLOR[arm], clip_on=False, **MARKER)
    b.set_xlim(0, run_h + X_PAD_H)
    b.set_xticks(np.arange(0, run_h + 1, 6))
    b.set_xlabel("observation time (h)")
    b.set_ylabel("fraction of runs flagged")
    b.plot([], [], color=C["black"], lw=1.2, label="anytime-valid test")
    b.plot([], [], color=C["black"], label="counting rule", **MARKER)
    b.legend(loc="lower right", fontsize=6, frameon=False)
    _style(b)

    for ax, lab in ((a, "a"), (b, "b")):
        ax.text(-0.02, 1.04, lab, transform=ax.transAxes, fontweight="bold",
                fontsize=9, ha="right", va="bottom")
    fig.tight_layout()
    return save(fig, "run_level_detection_v2")


def main() -> None:
    path = v1._RESULTS / "run_level_summary.json"
    print(plot(json.loads(path.read_text())))


if __name__ == "__main__":
    main()
