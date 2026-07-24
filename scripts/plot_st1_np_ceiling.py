"""Figure for the NP-optimal LRT ceiling in the ST1 bake-off (Phase 2, optional).

Reads results/st1/np_ceiling_summary.json (from scripts/st1_np_ceiling.py) and draws,
against the two bake-off negative columns:

  panels 1-2  TPR at empirical FAR 0.05 vs f0 drift, with the Whittle ceiling as a
              bold envelope and the shortfall (ceiling - detector) shaded — how much
              of the achievable power each corpus-free detector delivers;
  panel 3     rho_auc = (AUC_det - 0.5) / (AUC_ceiling - 0.5) vs drift (inference
              column) — the threshold-free "fraction of separating power".

Reproduce:  python scripts/plot_st1_np_ceiling.py [--summary PATH] [--smoke]
Outputs:    figures/st1_np_ceiling.png/.pdf
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

apply_house_style()

_ROOT = pathlib.Path(__file__).resolve().parent.parent
_RESULTS = _ROOT / "results" / "st1"

# Reuse the bake-off's detector styling so the two figures read as one family.
_STYLE = {
    "mtf": (C["black"], ":", "multitaper F (fixed)"),
    "spectral": (C["orange"], "--", "spectral matched filter"),
    "viterbi": (C["blue"], "-", "Viterbi tracker"),
    "dg_order_split": (C["green"], "-", "DG order (held-out)"),
    "dg_order_full": (C["vermillion"], "-", "DG order (adaptive)"),
    "dg_fixed_oracle": (C["grey"], "-.", r"DG fixed-$\alpha$ (oracle $f_0$)"),
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--summary", default=None, help="path to np_ceiling_summary.json")
    ap.add_argument("--smoke", action="store_true",
                    help="read np_ceiling_smoke_summary.json + write st1_np_ceiling_smoke")
    ap.add_argument("--stem", default=None, help="override the output figure stem")
    args = ap.parse_args()

    if args.summary:
        path = pathlib.Path(args.summary)
    elif args.smoke:
        path = _RESULTS / "np_ceiling_smoke_summary.json"
    else:
        path = _RESULTS / "np_ceiling_summary.json"
    stem = args.stem or ("st1_np_ceiling_smoke" if args.smoke else "st1_np_ceiling")

    s = json.loads(path.read_text())
    drifts = s["drifts_hz"]
    far = "0.05"
    det_names = s["detectors"]

    fig, axes = plt.subplots(1, 3, figsize=(WIDTH_WIDE * 1.35, 2.3))

    for ax, cls, title in zip(
            axes[:2], ("inference", "structural"),
            ("vs inference null", "vs mixed structural nulls")):
        col = s["by_negative"][cls]
        ceil_tpr = col["ceiling"]["tpr"][far]
        ax.plot(drifts, ceil_tpr, color=C["black"], lw=2.2, marker="s", ms=3.0,
                label="NP ceiling (Whittle)", zorder=5)
        for name in det_names:
            colr, ls, lab = _STYLE.get(name, (C["grey"], "-", name))
            y = col["detectors"][name]["tpr"][far]
            ax.plot(drifts, y, color=colr, ls=ls, lw=1.1, marker="o", ms=2.2, label=lab)
            ax.fill_between(drifts, y, ceil_tpr, color=colr, alpha=0.04)
        ax.set_ylim(-0.02, 1.05)
        ax.set_xlabel(r"$f_0$ drift (Hz)")
        ax.set_title(title, fontsize=7)
    axes[0].set_ylabel(r"TPR at empirical FAR $=0.05$")
    axes[0].legend(frameon=False, fontsize=5.0, loc="lower left")

    axR = axes[2]
    col = s["by_negative"]["inference"]
    for name in det_names:
        colr, ls, lab = _STYLE.get(name, (C["grey"], "-", name))
        rho = [np.nan if v is None else v for v in col["rho_auc"][name]]
        axR.plot(drifts, rho, color=colr, ls=ls, lw=1.1, marker="o", ms=2.2, label=lab)
    axR.axhline(1.0, color=C["black"], lw=1.0, ls="-")
    axR.axhline(0.0, color=C["grey"], lw=0.6, ls=":")
    axR.set_ylim(-0.35, 1.1)
    axR.set_xlabel(r"$f_0$ drift (Hz)")
    axR.set_ylabel(r"$\rho_{\mathrm{AUC}}$ (fraction of ceiling)")
    axR.set_title("fraction of NP-optimal power\n(vs inference null)", fontsize=7)

    fig.tight_layout()
    out = save(fig, stem)
    print(f"-> {out.with_suffix('')}.*")


if __name__ == "__main__":
    main()
