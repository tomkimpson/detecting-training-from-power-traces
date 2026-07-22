"""ST1 detector bake-off (task 19.10): power on the wander axis, honest nulls.

The pre-registered detector set, all scored on identical trace populations:

    mtf              fixed multitaper F (Thomson line test, Bonferroni-corrected)
    spectral         Welch matched filter (code.typeb.detectors.spectral_statistic)
    viterbi          CW-style line tracker score (code.typeb.detectors.viterbi_statistic)
    dg_order_split   stage-3 held-out-phase order-folded DG
    dg_order_full    stage-4 fully adaptive order-folded DG
    dg_fixed_oracle  fixed-alpha DG handed the TRUE per-trace f0 on positives
                     (negatives score at an alpha drawn from the same Ko f0
                     band, matching the positive alpha distribution) — the
                     oracle-frequency reference, an upper anchor no deployable
                     detector can reach
    dg_order_semicoh (--semicoh) stage-4 tracker + per-block DG Fisher-combined
                     — the semi-coherent variant for the fast-wander regime

Signal class: Ko training populations (code.typeb.ko_synth) over the wander
axis f0_drift_hz in {0, 0.1, 0.2, 0.4, 0.8, 1.5} Hz (the b1 grid), n/class.
Negative classes, BOTH reported separately:

    inference   the Ko inference null (the b1 comparison, backward-compatible)
    structural  a mixed structural-null population from the ST1 suite —
                equal parts ar1, ar1_t, controller

plus the pre-registered hard case: signal at drift 0.4 Hz vs CONTROLLER-ONLY
negatives (can the order pipeline distinguish a wandering line from a
controller limit cycle where the matched filter / Viterbi cannot?).

TPR is reported at EMPIRICAL FAR 0.05 and 0.01 per negative class (threshold =
the (1-FAR) quantile of that class's negative scores, method="higher" i.e.
conservative; with n=200 negatives the 0.01 threshold sits on the 2nd-highest
score — granularity stated, not hidden).

Reproduce:
    python scripts/plot_st1_bakeoff.py [--n-each 200] [--semicoh]
Outputs: figures/st1_bakeoff.png/.pdf ; results/st1/bakeoff_summary.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from functools import partial

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from powerladder.config import DEFAULT  # noqa: E402
from powerladder.plotstyle import C, WIDTH_WIDE, apply_house_style, save  # noqa: E402
from powerladder.st1.multitaper import f_test_statistic  # noqa: E402
from powerladder.st1.nulls import NULLS  # noqa: E402
from powerladder.st1.pipeline import (stage1_fixed_alpha, stage3_heldout_phase,  # noqa: E402
                               stage4_full_adaptive, stage4_semicoherent)
from powerladder.typeb.detectors import spectral_statistic, viterbi_statistic  # noqa: E402
from powerladder.typeb.ko_synth import ko_make_trace  # noqa: E402

apply_house_style()

_ROOT = pathlib.Path(__file__).resolve().parent.parent
_RESULTS = _ROOT / "results" / "st1"

DRIFTS_HZ = (0.0, 0.1, 0.2, 0.4, 0.8, 1.5)
FARS = (0.05, 0.01)
HARD_CASE_DRIFT_HZ = 0.4
STRUCTURAL_MIX = ("ar1", "ar1_t", "controller")   # equal parts

_STYLE = {
    "mtf": (C["black"], ":", "multitaper F (fixed)"),
    "spectral": (C["orange"], "--", "spectral matched filter"),
    "viterbi": (C["blue"], "-", "Viterbi tracker"),
    "dg_order_split": (C["green"], "-", "DG order (held-out, st.3)"),
    "dg_order_full": (C["vermillion"], "-", "DG order (adaptive, st.4)"),
    "dg_order_semicoh": (C["purple"], "-", "DG order (semi-coherent)"),
    "dg_fixed_oracle": (C["grey"], "-.", r"DG fixed-$\alpha$ (oracle $f_0$)"),
}


def _detectors(include_semicoh: bool) -> dict:
    p = DEFAULT.st1
    det = {
        "mtf": partial(f_test_statistic, params=p),
        "spectral": spectral_statistic,
        "viterbi": viterbi_statistic,
        "dg_order_split": partial(stage3_heldout_phase, params=p),
        "dg_order_full": partial(stage4_full_adaptive, params=p),
    }
    if include_semicoh:
        det["dg_order_semicoh"] = partial(stage4_semicoherent, params=p)
    return det


def _score(traces, detectors, oracle_alphas) -> dict:
    """Score every trace with every detector (+ the oracle fixed-alpha DG).

    ``oracle_alphas[i]`` is the alpha handed to dg_fixed_oracle for trace i:
    the trace's true f0 for positives, a fresh draw from the Ko f0 band for
    negatives (matching the positive alpha distribution so the oracle's null
    threshold is well defined).
    """
    p = DEFAULT.st1
    out = {name: np.empty(len(traces)) for name in detectors}
    out["dg_fixed_oracle"] = np.empty(len(traces))
    for i, tr in enumerate(traces):
        for name, fn in detectors.items():
            out[name][i] = fn(tr.t, tr.P_obs, p.band_lo, p.band_hi)
        out["dg_fixed_oracle"][i] = stage1_fixed_alpha(
            tr.t, tr.P_obs, p.band_lo, p.band_hi, params=p,
            alpha_hz=float(oracle_alphas[i]))
    return out


def _tpr_at_far(pos: np.ndarray, neg: np.ndarray, far: float) -> float:
    """TPR at the empirical-FAR threshold (conservative 'higher' quantile)."""
    thr = np.quantile(neg, 1.0 - far, method="higher")
    return float(np.mean(pos > thr))


def _make_positives(n_each, rng, drift):
    return [ko_make_trace("train", DEFAULT.ko, DEFAULT.ko_typeb, rng,
                          f0_drift_hz=drift) for _ in range(n_each)]


def _make_inference_negatives(n_each, rng):
    return [ko_make_trace("infer", DEFAULT.ko, DEFAULT.ko_typeb, rng)
            for _ in range(n_each)]


def _make_structural_negatives(n_each, rng, names=STRUCTURAL_MIX):
    p = DEFAULT.st1
    n = int(round(p.duration_s * p.fs)) + 1
    counts = [n_each // len(names)] * len(names)
    for i in range(n_each - sum(counts)):
        counts[i] += 1
    traces = []
    for name, k in zip(names, counts):
        traces += [NULLS[name](n, p.fs, DEFAULT.st1_null, rng)
                   for _ in range(k)]
    return traces


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-each", type=int, default=200,
                    help="traces per class (per drift level / negative class)")
    ap.add_argument("--seed", type=int, default=20260721)
    ap.add_argument("--semicoh", action="store_true",
                    help="include the semi-coherent DG variant")
    ap.add_argument("--drifts", default=",".join(f"{d:g}" for d in DRIFTS_HZ))
    args = ap.parse_args()
    drifts = tuple(float(s) for s in args.drifts.split(","))

    rng = np.random.default_rng(args.seed)
    detectors = _detectors(args.semicoh)
    det_names = list(detectors) + ["dg_fixed_oracle"]

    def oracle_null_alphas(n):
        return rng.uniform(DEFAULT.ko.f0_lo, DEFAULT.ko.f0_hi, size=n)

    t0 = time.time()
    # Negative classes (drift-independent) and their scores.
    negatives = {
        "inference": _make_inference_negatives(args.n_each, rng),
        "structural": _make_structural_negatives(args.n_each, rng),
        "controller_only": _make_structural_negatives(
            args.n_each, rng, names=("controller",)),
    }
    neg_scores = {
        cls: _score(tr, detectors, oracle_null_alphas(len(tr)))
        for cls, tr in negatives.items()
    }
    print(f"negatives scored in {time.time()-t0:.0f} s", flush=True)

    # Positive classes over the wander axis.
    tpr = {cls: {f"{far:g}": {n: [] for n in det_names} for far in FARS}
           for cls in ("inference", "structural")}
    hard = {}
    for drift in drifts:
        t1 = time.time()
        pos = _make_positives(args.n_each, rng, drift)
        pos_scores = _score(pos, detectors,
                            np.array([tr.f0 for tr in pos]))
        for cls in ("inference", "structural"):
            for far in FARS:
                for name in det_names:
                    tpr[cls][f"{far:g}"][name].append(
                        _tpr_at_far(pos_scores[name], neg_scores[cls][name],
                                    far))
        if drift == HARD_CASE_DRIFT_HZ:
            hard = {
                f"{far:g}": {
                    name: _tpr_at_far(pos_scores[name],
                                      neg_scores["controller_only"][name], far)
                    for name in det_names}
                for far in FARS
            }
        print(f"drift {drift:g} Hz scored in {time.time()-t1:.0f} s",
              flush=True)

    # ---- figure -------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH_WIDE * 1.35, 2.3))
    for ax, cls, title in zip(
            axes[:2], ("inference", "structural"),
            ("vs inference null (b1 comparison)",
             "vs mixed structural nulls (ar1 / ar1$_t$ / controller)")):
        for name in det_names:
            col, ls, lab = _STYLE[name]
            ax.plot(drifts, tpr[cls]["0.01"][name], color=col, ls=ls,
                    lw=1.2, marker="o", ms=2.4, label=lab)
        ax.set_ylim(-0.02, 1.05)
        ax.set_xlabel(r"$f_0$ drift (Hz)")
        ax.set_title(title, fontsize=7)
    axes[0].set_ylabel("TPR at empirical FAR $=10^{-2}$")
    axes[0].legend(frameon=False, fontsize=5.2, loc="upper right")

    axH = axes[2]
    names = det_names
    x = np.arange(len(names))
    for k, far in enumerate(FARS):
        ys = [hard.get(f"{far:g}", {}).get(n, np.nan) for n in names]
        axH.bar(x + 0.4 * k - 0.2, ys, width=0.36,
                color=[_STYLE[n][0] for n in names],
                alpha=1.0 if k == 0 else 0.45,
                label=f"FAR {far:g}")
    axH.set_xticks(x)
    axH.set_xticklabels([_STYLE[n][2] for n in names], rotation=40,
                        ha="right", fontsize=5.2)
    axH.set_ylim(0, 1.05)
    axH.set_title(f"hard case: drift {HARD_CASE_DRIFT_HZ:g} Hz vs "
                  "controller-only nulls", fontsize=7)
    axH.legend(frameon=False, fontsize=5.4)
    fig.tight_layout()
    fig_pdf = save(fig, "st1_bakeoff")

    # ---- summary ------------------------------------------------------------
    summary = {
        "n_each": args.n_each,
        "seed": args.seed,
        "drifts_hz": list(drifts),
        "fars": list(FARS),
        "structural_mix": list(STRUCTURAL_MIX),
        "hard_case": {"drift_hz": HARD_CASE_DRIFT_HZ,
                      "negatives": "controller_only", "tpr": hard},
        "tpr": tpr,
        "notes": (
            "TPR at empirical-FAR thresholds from each negative class's own "
            "scores (quantile method='higher', conservative); with n=200 "
            "negatives the 1e-2 threshold rests on the 2nd-highest negative "
            "score. dg_fixed_oracle is handed the true f0 per positive trace "
            "and a Ko-band draw per negative -- an oracle upper anchor, not "
            "a deployable detector."
        ),
    }
    _RESULTS.mkdir(parents=True, exist_ok=True)
    (_RESULTS / "bakeoff_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n")
    print(f"-> {fig_pdf.with_suffix('')}.* ; "
          f"{_RESULTS / 'bakeoff_summary.json'}")


if __name__ == "__main__":
    main()
