"""ST1 calibration figures (task 19.9b): the gate's evidence, drawn.

Three figures from the tracked calibration records:

  st1_calibration_far   realised vs nominal FAR (log-log, Clopper-Pearson CIs),
                        one panel per pipeline stage, every null in the suite,
                        dg/bartlett arm; nulls coloured by stage group
                        (5 = stationary Gaussian/coloured, 6 = stationary
                        non-Gaussian, 7 = locally stationary/controller).
                        Cells with zero exceedances are drawn at the plot floor
                        with open markers (the CI upper bound is still real).
  st1_calibration_qq    -log10 p QQ panels for the reference cells (raw
                        per-trace p-values from results/st1/raw/*.npz).
  st1_calibration_cov   the stage-8 covariance sweep: realised FAR@1e-2 vs
                        Bartlett bandwidth b for each shrinkage, on
                        stage4 x ar1 (the conservatism cell), stage4 x white,
                        and stage1 x white — the fix-or-reframe picture for
                        the GO-analytic decision.

Reproduce:
    python scripts/plot_st1_calibration.py
Inputs:  results/st1/far_summary.json, results/st1/raw/*.npz,
         results/st1/sweeps/cov.json
Outputs: figures/st1_calibration_{far,qq,cov}.png/.pdf
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

from powerladder.plotstyle import C, WIDTH_WIDE, apply_house_style, save  # noqa: E402
from powerladder.st1.nulls import STAGE5, STAGE6, STAGE7  # noqa: E402

apply_house_style()

_ROOT = pathlib.Path(__file__).resolve().parent.parent
_RESULTS = _ROOT / "results" / "st1"

STAGES = ("stage1", "stage2", "stage3", "stage4")
STAGE_TITLES = {
    "stage1": "stage 1: fixed $\\alpha$",
    "stage2": "stage 2: known warp",
    "stage3": "stage 3: held-out warp",
    "stage4": "stage 4: fully adaptive",
}
GROUP_COLOUR = {5: C["blue"], 6: C["green"], 7: C["vermillion"]}
GROUP_OF = {**{n: 5 for n in STAGE5}, **{n: 6 for n in STAGE6},
            **{n: 7 for n in STAGE7}}

# Reference cells for the QQ panels: the clean anchor, the resampling-deflated
# white cells, the coloured-conservatism cell, and the controller confuser.
QQ_CELLS = (("stage1", "white"), ("stage4", "white"),
            ("stage4", "ar1"), ("stage4", "controller"))


def _fig_far(summary: dict):
    fig, axes = plt.subplots(1, 4, figsize=(WIDTH_WIDE * 1.35, 2.1),
                             sharex=True, sharey=True)
    rng = np.random.default_rng(4)             # tiny jitter to unstack markers
    for ax, stage in zip(axes, STAGES):
        cells = summary.get(stage, {})
        for null, arms in cells.items():
            if "dg_bartlett" not in arms or null not in GROUP_OF:
                continue
            rec = arms["dg_bartlett"]
            M = rec["M"]
            floor = 0.5 / M
            col = GROUP_COLOUR[GROUP_OF[null]]
            for lvl_s, r in rec["levels"].items():
                lvl = float(lvl_s)
                x = lvl * 10 ** (rng.uniform(-0.05, 0.05))
                y = r["far"]
                lo, hi = r["ci"]
                if y <= 0.0:
                    ax.errorbar(x, floor, yerr=[[0.0], [max(hi - floor, 0.0)]],
                                fmt="v", ms=2.6, mfc="none", color=col,
                                lw=0.6, capsize=0, alpha=0.85)
                else:
                    ax.errorbar(x, y, yerr=[[max(y - lo, 0.0)], [hi - y]],
                                fmt="o", ms=2.2, color=col, lw=0.6,
                                capsize=0, alpha=0.85)
        lv = np.array([1e-4, 0.2])
        ax.plot(lv, lv, color=C["black"], lw=0.7, ls=":", zorder=0)
        ax.fill_between(lv, lv / 3.0, lv * 3.0, color=C["grey"], alpha=0.18,
                        lw=0, zorder=0)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(4e-4, 0.15)
        ax.set_ylim(2e-5, 0.3)
        ax.set_title(STAGE_TITLES[stage], fontsize=7)
        ax.set_xlabel("nominal level")
    axes[0].set_ylabel("realised FAR")
    handles = [plt.Line2D([], [], color=GROUP_COLOUR[g], marker="o", ms=3,
                          lw=0, label=lab)
               for g, lab in ((5, "stage-5 nulls (Gauss/coloured)"),
                              (6, "stage-6 (non-Gaussian)"),
                              (7, "stage-7 (nonstationary)"))]
    handles.append(plt.Line2D([], [], color=C["grey"], marker="v", ms=3,
                              mfc="none", lw=0, label="0 hits (at plot floor)"))
    axes[-1].legend(handles=handles, frameon=False, fontsize=5.4,
                    loc="upper left", handletextpad=0.3)
    fig.tight_layout()
    return save(fig, "st1_calibration_far")


def _fig_qq():
    fig, axes = plt.subplots(1, len(QQ_CELLS),
                             figsize=(WIDTH_WIDE * 1.35, 2.0), sharex=True)
    for ax, (stage, null) in zip(axes, QQ_CELLS):
        path = _RESULTS / "raw" / f"{stage}_{null}_dg_bartlett.npz"
        p = np.sort(np.load(path)["p"])
        M = p.size
        exp = -np.log10((np.arange(1, M + 1) - 0.5) / M)[::-1]
        obs = -np.log10(np.clip(p, 1e-300, None))[::-1]
        ax.plot(exp, obs, color=C["blue"], lw=1.0)
        lim = max(exp.max(), 4.2)
        ax.plot([0, lim], [0, lim], color=C["black"], lw=0.7, ls=":", zorder=0)
        ax.set_xlim(0, lim)
        ax.set_ylim(0, max(obs.max() * 1.05, lim))
        ax.set_title(f"{STAGE_TITLES[stage].split(':')[0]} / {null}",
                     fontsize=7)
        ax.set_xlabel(r"expected $-\log_{10} p$")
    axes[0].set_ylabel(r"observed $-\log_{10} p$")
    fig.tight_layout()
    return save(fig, "st1_calibration_qq")


def _fig_cov(cov: dict):
    cells = (("stage4", "ar1", "-"), ("stage4", "white", "--"),
             ("stage1", "white", ":"))
    shrink_colour = {0.0: C["blue"], 0.01: C["orange"], 0.05: C["vermillion"]}
    fig, ax = plt.subplots(figsize=(0.62 * WIDTH_WIDE, 2.4))
    pts = cov["points"]
    for stage, null, ls in cells:
        for lam, col in shrink_colour.items():
            xs, ys = [], []
            for b in (6, 18, 54):
                rec = next((r for r in pts
                            if r["stage"] == stage and r["null"] == null
                            and r["params_delta"].get("cov_estimator") == "bartlett"
                            and r["params_delta"].get("cov_shrinkage") == lam
                            and r["params_delta"].get("cov_bandwidth_b") == b),
                           None)
                if rec is None:
                    continue
                xs.append(b)
                ys.append(max(rec["levels"]["0.01"]["far"],
                              0.5 / rec["M"]))       # plot floor for 0 hits
            if xs:
                ax.plot(xs, ys, color=col, ls=ls, lw=1.1, marker="o", ms=2.6)
    ax.axhline(1e-2, color=C["black"], lw=0.7, ls=":", zorder=0)
    ax.axhspan(1e-2 / 3.0, 3e-2, color=C["grey"], alpha=0.18, lw=0, zorder=0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks([6, 18, 54])
    ax.set_xticklabels(["6", "18", "54"])
    ax.set_xlabel("Bartlett bandwidth $b$ (lags)")
    ax.set_ylabel("realised FAR at nominal $10^{-2}$")
    colour_handles = [plt.Line2D([], [], color=c, lw=1.1,
                                 label=fr"shrinkage $\lambda={l:g}$")
                      for l, c in shrink_colour.items()]
    style_handles = [plt.Line2D([], [], color=C["black"], ls=ls, lw=1.1,
                                label=f"{st} / {nu}")
                     for st, nu, ls in cells]
    ax.legend(handles=colour_handles + style_handles, frameon=False,
              fontsize=5.4, loc="lower right", ncol=1)
    fig.tight_layout()
    return save(fig, "st1_calibration_cov")


def main() -> None:
    summary = json.loads((_RESULTS / "far_summary.json").read_text())
    out = [_fig_far(summary), _fig_qq()]
    cov_path = _RESULTS / "sweeps" / "cov.json"
    if cov_path.exists():
        out.append(_fig_cov(json.loads(cov_path.read_text())))
    else:
        print("NOTE: sweeps/cov.json missing -- run scripts/st1_sweeps.py "
              "--axes cov first; skipping the covariance panel")
    for p in out:
        print(f"-> {p.with_suffix('')}.*")


if __name__ == "__main__":
    main()
