"""Plot the meter-requirement boundary sweep (Phase 2; "minimum meter spec").

Reads results/st2/meter_boundary_summary.json and renders:

  figures/st2_meter_boundary.{pdf,png}        one TPR@0.05 heatmap per detector
      over the sample_hz x integ_window_s main grid, with a "death" contour
      (TPR = 0.5) marking where that detector stops separating training from
      inference — the boundary of the region in which the rung is available.
  figures/st2_meter_boundary_notch.{pdf,png}  TPR@0.05 vs notch centre for each
      blend depth (the in-band transfer-function sub-sweep at the 20 Hz sampler).

Compute-free: run scripts/st2_meter_boundary.py first.

Usage:
    python scripts/plot_st2_meter_boundary.py
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

from powerladder.plotstyle import C, WIDTH_WIDE, apply_house_style, save  # noqa: E402

_ROOT = pathlib.Path(__file__).resolve().parent.parent
_SUMMARY = _ROOT / "results" / "st2" / "meter_boundary_summary.json"

# Death threshold: below this TPR the detector no longer separates the classes
# at the operating FAR (== the frontier's _FIXED_MAX gate constant).
_DEATH_TPR = 0.5

_LABEL = {"spectral": "spectral (matched filter)", "viterbi": "Viterbi (tracker)",
          "mtf": "multitaper F (fixed)", "dg_order_full": "DG order (adaptive)",
          "dg_order_semicoh": "DG order (semi-coherent)"}
_NOTCH_COL = (C["blue"], C["orange"], C["vermillion"], C["green"], C["purple"])


def _far_key(summary: dict) -> str:
    """The primary operating FAR as its summary key (first of target_fars)."""
    return f"{summary['target_fars'][0]:g}"


def _main_grid(summary: dict, detector: str, far: str):
    """(Z, sample_hz axis, integ_window axis) of TPR for one detector.

    Rows = integ_window_grid (ascending), cols = sample_hz_grid (descending, so
    the honest 20 Hz reference is at the left edge)."""
    fs_axis = sorted(summary["grid"]["sample_hz_grid"], reverse=True)
    iw_axis = sorted(summary["grid"]["integ_window_grid"])
    Z = np.full((len(iw_axis), len(fs_axis)), np.nan)
    by_key = {(c["sample_hz"], c["integ_window_s"]): c
              for c in summary["cells"] if c["notch_hz"] is None}
    for i, iw in enumerate(iw_axis):
        for j, fs in enumerate(fs_axis):
            c = by_key.get((fs, iw))
            if c is not None:
                Z[i, j] = c["tpr_at_far"][detector][far]
    return Z, fs_axis, iw_axis


def plot_main(summary: dict, fig_dir_stem: str = "st2_meter_boundary") -> None:
    """One TPR heatmap per detector over the sample_hz x integ_window grid."""
    apply_house_style()
    far = _far_key(summary)
    detectors = summary["detectors"]
    ncol = len(detectors)
    fig, axes = plt.subplots(1, ncol, figsize=(WIDTH_WIDE * 1.9, 2.4),
                             squeeze=False)
    im = None
    for ax, det in zip(axes[0], detectors):
        Z, fs_axis, iw_axis = _main_grid(summary, det, far)
        im = ax.imshow(Z, origin="lower", aspect="auto", vmin=0.0, vmax=1.0,
                       cmap="viridis")
        # Death contour at TPR = 0.5 (interpolated on the cell grid).
        if np.isfinite(Z).sum() >= 4 and np.nanmin(Z) < _DEATH_TPR < np.nanmax(Z):
            ax.contour(np.arange(len(fs_axis)), np.arange(len(iw_axis)), Z,
                       levels=[_DEATH_TPR], colors="white", linewidths=1.2)
        ax.set_xticks(range(len(fs_axis)))
        ax.set_xticklabels([f"{v:g}" for v in fs_axis], fontsize=5, rotation=45)
        ax.set_yticks(range(len(iw_axis)))
        ax.set_yticklabels([f"{v:g}" for v in iw_axis], fontsize=5)
        ax.set_xlabel("sample rate (Hz)", fontsize=6)
        ax.set_title(_LABEL.get(det, det), fontsize=6)
    axes[0][0].set_ylabel("integration window (s)", fontsize=6)
    fig.suptitle(f"Detection rate (TPR @ FAR {far}); white line = death "
                 f"contour (TPR {_DEATH_TPR})", fontsize=7)
    if im is not None:
        fig.colorbar(im, ax=axes[0], fraction=0.015, pad=0.01)
    save(fig, fig_dir_stem)
    plt.close(fig)


def plot_notch(summary: dict,
               fig_dir_stem: str = "st2_meter_boundary_notch") -> None:
    """TPR vs notch centre for each blend depth, per detector class rep."""
    notch_cells = [c for c in summary["cells"] if c["notch_hz"] is not None]
    if not notch_cells:
        print("  (no notch cells in summary; skipping notch figure)")
        return
    apply_house_style()
    far = _far_key(summary)
    depths = sorted({c["notch_depth"] for c in notch_cells})
    centres = sorted({c["notch_hz"] for c in notch_cells})
    # One panel per detector class representative: the strongest tracker and the
    # strongest fixed test at the reference channel.
    reps = ("dg_order_full", "viterbi", "mtf")
    reps = [d for d in reps if d in summary["detectors"]]
    fig, axes = plt.subplots(1, len(reps), figsize=(WIDTH_WIDE * 1.2, 2.2),
                             squeeze=False, sharey=True)
    for ax, det in zip(axes[0], reps):
        for depth, col in zip(depths, _NOTCH_COL):
            ys = []
            for fn in centres:
                c = next((c for c in notch_cells
                          if c["notch_hz"] == fn and c["notch_depth"] == depth),
                         None)
                ys.append(c["tpr_at_far"][det][far] if c else np.nan)
            ax.plot(centres, ys, marker="o", ms=3, lw=1.3, color=col,
                    label=f"depth {depth:g}")
        ax.set_ylim(0, 1.04)
        ax.set_xlabel("notch centre (Hz)", fontsize=6)
        ax.set_title(_LABEL.get(det, det), fontsize=6)
    axes[0][0].set_ylabel(f"TPR @ FAR {far}", fontsize=6)
    axes[0][-1].legend(frameon=False, fontsize=5.5, loc="lower left")
    fig.suptitle("Notch sub-sweep (20 Hz sampler): detection vs in-band "
                 "anti-resonance", fontsize=7)
    fig.tight_layout()
    save(fig, fig_dir_stem)
    plt.close(fig)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--summary", type=pathlib.Path, default=_SUMMARY)
    args = ap.parse_args(argv)
    if not args.summary.exists():
        raise SystemExit(f"{args.summary} not found — run "
                         f"scripts/st2_meter_boundary.py first")
    summary = json.loads(args.summary.read_text())
    plot_main(summary)
    plot_notch(summary)
    print(f"wrote figures/st2_meter_boundary*.pdf from {args.summary}")


if __name__ == "__main__":
    main()
