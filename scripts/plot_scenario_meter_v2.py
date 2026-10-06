"""Sensitivity of the cadence signature to the observation map -- v2 layout.

Same sweep and same numbers as ``plot_scenario_meter.py`` (imported, not
re-derived); only the presentation differs:

  * panels are grouped by stage of the power chain (fig:power_chain): one column
    each for the delivery path, the meter, and the additive terms;
  * x-axes are drawn on a true (log or linear) scale, oriented so that the
    channel gets harsher left to right; an "off"/zero setting sits on a stub left
    of an axis break;
  * the y-axis extends to 1e-5 so small-but-nonzero values are drawn where they
    are, and only exact zeros (f0 at or past Nyquist after decimation) are marked
    as not observable, in a band at the bottom of the panel;

The interquartile range over realisations is under 5 per cent at every grid
point -- narrower than the markers -- so it is not drawn.

Reproduce:
    python scripts/plot_scenario_meter_v2.py
Outputs:
    figures/scenario_meter_v2.{pdf,png}
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import plot_scenario_meter as v1  # noqa: E402
from powerladder.plotstyle import C, WIDTH_WIDE, apply_house_style, save  # noqa: E402

# --- parameters (no magic numbers below) -------------------------------------
YMIN, YMAX = 2e-6, 60.0
DEAD_Y = 6e-6            # where exact zeros are drawn, inside the bottom band
DEAD_BAND_TOP = 1.4e-5   # top of the "not observable" band
STUB_GAP = 0.5           # log10-decades between an off/zero stub and the grid
LINE_C = C["blue"]

# Panel layout: (row, col) -> AXES key. Columns follow the power chain.
LAYOUT = {
    (0, 0): "lp_cutoff_hz", (1, 0): "notch_depth",
    (0, 1): "integ_window_s", (1, 1): "sample_hz",
    (0, 2): "sigma_eta", (1, 2): "controller_amp",
}
# Per-axis x mapping. "log": position = sign * log10(value); "lin": value.
# sign = -1 for axes where a SMALLER value is harsher (cutoff, sample rate), so
# every panel reads harsher to the right.
XMAP = {
    "lp_cutoff_hz":   {"scale": "log", "sign": -1},
    "integ_window_s": {"scale": "log", "sign": +1},
    "sample_hz":      {"scale": "log", "sign": -1,
                       "none_value": 20.0},  # no decimation == the 20 Hz grid
    "notch_depth":    {"scale": "lin", "sign": +1},
    "sigma_eta":      {"scale": "log", "sign": +1},
    "controller_amp": {"scale": "log", "sign": +1},
}
XLABELS = {
    "lp_cutoff_hz": "low-pass cutoff (Hz)",
    "integ_window_s": "integration window (s)",
    "sample_hz": "sample rate (Hz)",
    "notch_depth": rf"notch depth at $f_0$ ($Q = {v1._MB.notch_q:g}$)",
    "sigma_eta": r"meter noise $\sigma_\eta$ (W)",
    "controller_amp": "controller amplitude (W pk-pk)",
}


def _xpos(key: str, levels: list) -> tuple[np.ndarray, list[str], int | None]:
    """x positions, tick labels, and index of an off/zero stub (or None)."""
    m = XMAP[key]
    if m["scale"] == "lin":
        return (np.asarray(levels, float),
                [f"{lv:g}" for lv in levels], None)
    vals = [m.get("none_value") if lv is None else lv for lv in levels]
    pos, labels, stub = [], [], None
    for i, (lv, v) in enumerate(zip(levels, vals)):
        if v is None or v == 0.0:
            stub = i
            pos.append(np.nan)
            labels.append("off" if lv is None else "0")
        else:
            pos.append(m["sign"] * np.log10(v))
            labels.append(f"{v:g}")
    pos = np.asarray(pos)
    if stub is not None:
        finite = pos[np.isfinite(pos)]
        pos[stub] = finite.min() - STUB_GAP
    return pos, labels, stub


def _break_marks(ax, x: float) -> None:
    """Two short slanted strokes on the bottom spine at data-x ``x``."""
    trans = ax.get_xaxis_transform()
    for dx in (-0.06, 0.06):
        ax.plot([x + dx - 0.05, x + dx + 0.05], [-0.025, 0.025], transform=trans,
                color="black", lw=0.6, clip_on=False, zorder=5)
    ax.plot([x - 0.06, x + 0.06], [0, 0], transform=trans, color="white",
            lw=1.6, clip_on=False, zorder=4)


def _panel(ax, key: str, rec: dict) -> None:
    levels = rec["levels"]
    xs, labels, stub = _xpos(key, levels)
    med = np.asarray(rec["median"], float)
    dead = med <= 0.0
    live = ~dead

    ax.axhspan(YMIN, DEAD_BAND_TOP, color="0.93", lw=0, zorder=0)
    ax.axhline(1.0, color=C["grey"], ls=(0, (2, 2)), lw=0.6, zorder=1)

    # Curve through the observable points; a dotted drop to the band where the
    # cadence leaves the observable band entirely.
    ax.plot(xs[live], med[live], color=LINE_C, lw=1.2, zorder=2)
    if dead.any():
        first_dead = int(np.argmax(dead))
        if first_dead > 0:
            ax.plot([xs[first_dead - 1], xs[first_dead]],
                    [med[first_dead - 1], DEAD_Y], color=LINE_C, lw=0.8,
                    ls=(0, (1, 1.5)), zorder=2)
        ax.plot(xs[dead], np.full(dead.sum(), DEAD_Y), ls="none", marker="o",
                ms=4.0, mfc="white", mec=C["vermillion"], mew=1.0, zorder=3)
    ax.plot(xs[live], med[live], ls="none", marker="o", ms=3.2, color=LINE_C,
            mec="white", mew=0.5, zorder=3)

    ax.set_yscale("log")
    ax.set_ylim(YMIN, YMAX)
    ax.set_yticks([1e-4, 1e-2, 1.0])
    ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    pad = 0.25 if XMAP[key]["scale"] == "log" else 0.08
    ax.set_xlim(np.nanmin(xs) - pad, np.nanmax(xs) + pad)
    ax.set_xlabel(XLABELS[key])
    ax.tick_params(which="both", top=False, right=False, direction="out",
                   length=2.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if stub is not None:
        others = np.delete(xs, stub)
        _break_marks(ax, 0.5 * (xs[stub] + others.min()))


def plot(sweeps: dict[str, dict]) -> pathlib.Path:
    apply_house_style()
    fig, axarr = plt.subplots(2, 3, figsize=(WIDTH_WIDE * 1.35, 3.6),
                              sharey=True)
    for (r, c), key in LAYOUT.items():
        ax = axarr[r, c]
        _panel(ax, key, sweeps[key])
        ax.text(-0.02, 1.04, "abcdef"[3 * r + c], transform=ax.transAxes,
                fontweight="bold", fontsize=9, ha="right", va="bottom")
        if c == 0:
            ax.set_ylabel("cadence SNR\n(relative to nominal)")

    axarr[1, 1].text(0.03, 0.035, r"not observable ($f_0 \geq$ Nyquist)",
                     transform=axarr[1, 1].transAxes, fontsize=6,
                     color=C["vermillion"], ha="left", va="bottom")
    axarr[0, 0].text(0.97, 0.80, "nominal", transform=axarr[0, 0].transAxes,
                     fontsize=6, color=C["grey"], ha="right", va="bottom")

    fig.tight_layout(h_pad=1.2, w_pad=1.0)
    return save(fig, "scenario_meter_v2")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-seeds", type=int, default=v1.N_SEEDS,
                    help="workload realisations per grid point")
    args = ap.parse_args()
    out = plot(v1.build(n_seeds=args.n_seeds))
    print(f"-> {out.with_suffix('')}.*")


if __name__ == "__main__":
    main()
