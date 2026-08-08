#!/usr/bin/env python3
"""E2 at the locked cell: explicit-duration HSMM against duration-neutral controls.

The question (agreed 2026-08-08):

> At the fixed 10 s, 20%-share operating point, does explicit-duration
> marginalisation separate training from an IAAFT phase-randomised surrogate
> materially better than E1 and otherwise identical duration-neutral sequence
> models?

**Scoring contract.** Raw model comparisons are the primary endpoint. The
pre-fix absolute targets are retired as gates -- a precommitment cannot require
preserving a measurement once that measurement is known to be biased -- and are
reported as explicitly pre-fix diagnostics only.

| Arm | Role | Primary criterion |
|---|---|---|
| IAAFT | primary mechanistic test | HSMM AUC >= 0.75 **and** >= +0.10 over E1 |
| AAFT | surrogate robustness | raw AUC and paired improvement; no gate |
| composite | operational safeguard | non-inferiority to E1 at composite FAR 0.05 |

The 0.75 threshold is the corrected ``R = 0.5`` target (0.749) expressed as a
stable raw endpoint: a per-replicate ``R`` is ill-behaved because the corrected
gap crosses zero across seeds (AAFT -0.108..+0.329), so ratios are reported in
aggregate and never averaged per seed.

**The duration claim** additionally requires the explicit-duration model to beat
*both* duration-neutral controls by a material margin (>= +0.05 AUC) with a
paired interval excluding zero. Without that, any gain is attributable to the
sequence model or to avoiding E1's brittle extraction step, not to durations.

Every score consumes only ``(t, P_obs)``. Outputs land in ``results/hsmm_e2/``.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from powerladder.typeb.hsmm import (  # noqa: E402
    DURATION_KINDS,
    EXPLICIT,
    blind_segmentation,
    fit_composite,
)
from powerladder.typeb.renewal_campaign import (  # noqa: E402
    NULL_FAMILIES,
    alignment_pool,
    glue_for,
    make_diluted_population,
    make_null_populations,
    matched_surrogate_population,
)
from powerladder.typeb.renewal_hurdle import (  # noqa: E402
    aligned_observation,
    fit_hurdle_likelihood,
    observe,
    random_aligned_observations,
    zero_event_rate,
)
from powerladder.typeb.renewal_regime import rank_auc  # noqa: E402


_ROOT = pathlib.Path(__file__).resolve().parent.parent
_DEFAULT_OUTPUT = _ROOT / "results" / "hsmm_e2"

CELL = {"duration_s": 10.0, "share": 0.2}
ARMS = ("iaaft", "aaft", "composite")

CONTRACT = {
    "primary_arm": "iaaft",
    "iaaft_min_auc": 0.75,
    "iaaft_min_improvement_over_e1": 0.10,
    "duration_claim_min_margin": 0.05,
    "composite_noninferiority_auc": 0.03,
    "composite_noninferiority_tpr": 0.05,
    "composite_far": 0.05,
    "retired_prefix_targets": {"aaft": 0.8066, "iaaft": 0.8094},
}


def _threshold_at_far(scores: np.ndarray, far: float) -> float:
    x = np.asarray(scores, dtype=float)
    unique = np.unique(x)
    candidates = np.unique(np.concatenate([unique, np.nextafter(unique, np.inf),
                                           [np.inf]]))
    valid = candidates[np.asarray([(x >= v).mean() <= far for v in candidates])]
    return float(np.min(valid))


def _paired_diff_interval(a_pos, a_neg, b_pos, b_neg, rng, n_boot=1000):
    """Interval for AUC(a) - AUC(b) with shared resampling (paired by trace)."""
    n_pos, n_neg = a_pos.size, a_neg.size
    diffs = np.empty(n_boot)
    for k in range(n_boot):
        pi = rng.integers(0, n_pos, n_pos)
        ni = rng.integers(0, n_neg, n_neg)
        diffs[k] = (rank_auc(a_pos[pi], a_neg[ni])
                    - rank_auc(b_pos[pi], b_neg[ni]))
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    return float(lo), float(hi)


def _populations(arm: str, seed: int, args):
    glue = glue_for(CELL["duration_s"])
    fit_pos = make_diluted_population(CELL["share"], args.n_fit, glue, seed + 1)
    eval_pos = make_diluted_population(CELL["share"], args.n_eval, glue, seed + 2)
    if arm == "composite":
        n = max(args.n_eval // len(NULL_FAMILIES), 20)
        return (fit_pos, eval_pos,
                make_null_populations(n, glue, seed + 5),
                make_null_populations(n, glue, seed + 6))
    src_fit = make_diluted_population(CELL["share"], args.n_fit, glue, seed + 3)
    src_eval = make_diluted_population(CELL["share"], args.n_eval, glue, seed + 4)
    return (fit_pos, eval_pos,
            {arm: matched_surrogate_population(src_fit, seed + 5, kind=arm)},
            {arm: matched_surrogate_population(src_eval, seed + 6, kind=arm)})


def _replicate(arm: str, seed: int, args) -> dict:
    fit_pos, eval_pos, fit_nulls, eval_nulls = _populations(arm, seed, args)
    fit_null_traces = [t for g in fit_nulls.values() for t in g]
    eval_null_traces = [t for g in eval_nulls.values() for t in g]

    # --- E1: the hurdle likelihood, the corrected baseline -------------------
    e1 = fit_hurdle_likelihood(
        [observe(t.t, t.P_obs) for t in fit_pos],
        [[observe(t.t, t.P_obs) for t in g] for g in fit_nulls.values()],
    )
    scores = {"e1": (
        np.asarray([e1.score(t.t, t.P_obs) for t in eval_pos]),
        np.asarray([e1.score(t.t, t.P_obs) for t in eval_null_traces]),
    )}

    # --- E2 and its duration-neutral controls, one code path ----------------
    fit_seg = [blind_segmentation(t.t, t.P_obs) for t in fit_pos]
    null_seg = [[blind_segmentation(t.t, t.P_obs) for t in g]
                for g in fit_nulls.values()]
    for kind in DURATION_KINDS:
        model = fit_composite(fit_seg, null_seg, kind=kind, seed=seed)
        scores[f"hsmm_{kind}"] = (
            np.asarray([model.score(t.t, t.P_obs) for t in eval_pos]),
            np.asarray([model.score(t.t, t.P_obs) for t in eval_null_traces]),
        )

    # --- corrected align+, recomputed under the hurdle representation --------
    counts, durations = alignment_pool(fit_pos)

    def true_aligned(traces):
        return [aligned_observation(x.t, x.P_obs, x.meta.communication_starts,
                                    x.meta.communication_durations)
                for x in traces]

    null_fit_windows = random_aligned_observations(fit_null_traces, counts,
                                                   durations, seed + 101)
    null_eval_windows = random_aligned_observations(eval_null_traces, counts,
                                                    durations, seed + 202)
    e0_true = fit_hurdle_likelihood(true_aligned(fit_pos), [null_fit_windows])
    e0_rand = fit_hurdle_likelihood(
        random_aligned_observations(fit_pos, counts, durations, seed + 303),
        [null_fit_windows],
    )
    auc_true = rank_auc(
        np.asarray([e0_true.score_observation(o) for o in true_aligned(eval_pos)]),
        np.asarray([e0_true.score_observation(o) for o in null_eval_windows]))
    auc_rand = rank_auc(
        np.asarray([e0_rand.score_observation(o) for o in
                    random_aligned_observations(eval_pos, counts, durations,
                                                seed + 404)]),
        np.asarray([e0_rand.score_observation(o) for o in null_eval_windows]))

    rng = np.random.default_rng(seed + 909)
    aucs = {name: rank_auc(*value) for name, value in scores.items()}
    improvements = {}
    for reference in ("e1", "hsmm_geometric", "hsmm_shuffled"):
        lo, hi = _paired_diff_interval(*scores["hsmm_explicit"],
                                       *scores[reference], rng, args.n_boot)
        improvements[f"explicit_minus_{reference}"] = {
            "delta": float(aucs["hsmm_explicit"] - aucs[reference]),
            "lo": lo, "hi": hi, "excludes_zero": lo > 0.0,
        }

    out = {
        "seed": seed, "auc": aucs, "improvements": improvements,
        "e0_true_auc": auc_true, "e0_random_auc": auc_rand,
        "align_plus_corrected": float(auc_true - auc_rand),
        "zero_event_rate_positive": zero_event_rate(
            [observe(t.t, t.P_obs) for t in eval_pos]),
    }
    if arm == "composite":
        # Operational safeguard: shared composite-null FAR calibration.
        out["far_calibrated"] = _composite_operating_point(
            scores, eval_nulls, eval_null_traces)
    return out


def _composite_operating_point(scores, eval_nulls, eval_null_traces) -> dict:
    """TPR at the most conservative per-family threshold, plus per-family FPR."""
    sizes = {name: len(group) for name, group in eval_nulls.items()}
    out = {}
    for model, (pos, neg) in scores.items():
        offset, by_family = 0, {}
        for name, size in sizes.items():
            by_family[name] = neg[offset:offset + size]
            offset += size
        threshold = max(_threshold_at_far(v, CONTRACT["composite_far"])
                        for v in by_family.values())
        out[model] = {
            "threshold": threshold,
            "tpr": float((pos >= threshold).mean()),
            "fpr_by_family": {name: float((v >= threshold).mean())
                              for name, v in by_family.items()},
        }
    return out


def _summarise(replicates: list[dict], arm: str) -> dict:
    def stats(values):
        v = np.asarray(values, dtype=float)
        return {"mean": float(v.mean()),
                "sd": float(v.std(ddof=1)) if v.size > 1 else 0.0,
                "min": float(v.min()), "max": float(v.max())}

    models = list(replicates[0]["auc"])
    auc = {m: stats([r["auc"][m] for r in replicates]) for m in models}
    improvements = {
        key: {**stats([r["improvements"][key]["delta"] for r in replicates]),
              "n_intervals_excluding_zero": int(sum(
                  r["improvements"][key]["excludes_zero"] for r in replicates)),
              "n_replicates": len(replicates)}
        for key in replicates[0]["improvements"]
    }
    summary = {
        "auc": auc,
        "improvements": improvements,
        "e0_true_auc": stats([r["e0_true_auc"] for r in replicates]),
        "align_plus_corrected": stats([r["align_plus_corrected"]
                                       for r in replicates]),
        "zero_event_rate_positive": stats([r["zero_event_rate_positive"]
                                           for r in replicates]),
    }
    # Aggregate R only -- never an average of per-seed ratios, whose denominator
    # crosses zero across seeds.
    gap = summary["e0_true_auc"]["mean"] - auc["e1"]["mean"]
    summary["aggregate_R"] = {
        "note": "computed from aggregate means; per-seed ratios are ill-behaved",
        "gap": float(gap),
        "explicit": (float((auc["hsmm_explicit"]["mean"] - auc["e1"]["mean"]) / gap)
                     if abs(gap) > 1e-9 else None),
    }
    duration_claim = {
        key: (improvements[key]["mean"] >= CONTRACT["duration_claim_min_margin"]
              and improvements[key]["n_intervals_excluding_zero"]
              == improvements[key]["n_replicates"])
        for key in ("explicit_minus_hsmm_geometric", "explicit_minus_hsmm_shuffled")
    }
    summary["duration_claim_supported"] = all(duration_claim.values())
    summary["duration_claim_detail"] = duration_claim

    if arm == CONTRACT["primary_arm"]:
        summary["primary_criterion"] = {
            "auc_at_least": CONTRACT["iaaft_min_auc"],
            "auc": auc["hsmm_explicit"]["mean"],
            "auc_passes": auc["hsmm_explicit"]["mean"] >= CONTRACT["iaaft_min_auc"],
            "improvement_over_e1": improvements["explicit_minus_e1"]["mean"],
            "improvement_passes": (improvements["explicit_minus_e1"]["mean"]
                                   >= CONTRACT["iaaft_min_improvement_over_e1"]),
        }
        summary["primary_criterion"]["passes"] = (
            summary["primary_criterion"]["auc_passes"]
            and summary["primary_criterion"]["improvement_passes"])
    if arm == "composite":
        far = [r["far_calibrated"] for r in replicates]
        summary["far_calibrated"] = {
            model: {
                "tpr": stats([f[model]["tpr"] for f in far]),
                "fpr_by_family": {
                    name: float(np.mean([f[model]["fpr_by_family"][name]
                                         for f in far]))
                    for name in far[0][model]["fpr_by_family"]
                },
            }
            for model in far[0]
        }
        e1_tpr = summary["far_calibrated"]["e1"]["tpr"]["mean"]
        hs_tpr = summary["far_calibrated"]["hsmm_explicit"]["tpr"]["mean"]
        summary["non_inferior_to_e1"] = bool(
            hs_tpr >= e1_tpr - CONTRACT["composite_noninferiority_tpr"]
            and auc["hsmm_explicit"]["mean"]
            >= auc["e1"]["mean"] - CONTRACT["composite_noninferiority_auc"])
    if arm in CONTRACT["retired_prefix_targets"]:
        target = CONTRACT["retired_prefix_targets"][arm]
        summary["retired_prefix_diagnostic"] = {
            "note": "pre-fix bar, retired as a gate; reported for continuity",
            "target": target,
            "cleared": bool(auc["hsmm_explicit"]["mean"] >= target),
        }
    return summary


def run(args: argparse.Namespace) -> dict:
    t0 = time.time()
    output = pathlib.Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    seeds = [args.seed + 7919 * k for k in range(args.n_seeds)]

    by_arm = {}
    for arm in ARMS:
        replicates = []
        for seed in seeds:
            replicates.append(_replicate(arm, seed, args))
            r = replicates[-1]
            print(f"  {arm:<10s} seed={seed:<10d} "
                  + "  ".join(f"{k.replace('hsmm_','')}={v:.3f}"
                              for k, v in r["auc"].items())
                  + f"  align+={r['align_plus_corrected']:+.3f}", flush=True)
        by_arm[arm] = {"replicates": replicates,
                       "summary": _summarise(replicates, arm)}

    summary = {
        "status": "E2 prototype at the locked cell; not a paper result",
        "question": ("does explicit-duration marginalisation beat E1 and "
                     "duration-neutral sequence models on IAAFT surrogates?"),
        "cell": CELL,
        "contract": CONTRACT,
        "sizes": {"fit": args.n_fit, "evaluation": args.n_eval,
                  "seeds": args.n_seeds, "bootstrap": args.n_boot},
        "seeds": seeds,
        "by_arm": by_arm,
        "elapsed_s": time.time() - t0,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    _plot(by_arm, output)
    return summary


def _plot(by_arm: dict, output: pathlib.Path) -> None:
    models = ["e1", "hsmm_explicit", "hsmm_geometric", "hsmm_shuffled"]
    fig, axes = plt.subplots(1, len(ARMS), figsize=(11.5, 3.6), sharey=True)
    for ax, arm in zip(axes, ARMS):
        for j, model in enumerate(models):
            values = [r["auc"][model] for r in by_arm[arm]["replicates"]]
            ax.plot(np.full(len(values), j), values, "o", ms=5, alpha=0.7,
                    color=f"C{j}")
            ax.plot([j - 0.22, j + 0.22], [np.mean(values)] * 2, "-",
                    color=f"C{j}", lw=2)
        ax.set_xticks(range(len(models)),
                      [m.replace("hsmm_", "") for m in models], fontsize=8)
        ax.set_title(arm, fontsize=9)
        ax.axhline(0.5, color="0.8", lw=0.8)
    axes[0].axhline(CONTRACT["iaaft_min_auc"], color="C3", ls="--", lw=0.9)
    axes[0].set_ylabel("AUC")
    fig.suptitle("E2 at the locked cell (10 s, 20% share)", fontsize=10)
    fig.tight_layout()
    fig.savefig(output / "e2_models.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--n-fit", type=int, default=120)
    parser.add_argument("--n-eval", type=int, default=120)
    parser.add_argument("--n-boot", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260808)
    parser.add_argument("--output-dir", default=str(_DEFAULT_OUTPUT))
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if args.quick:
        args.n_seeds, args.n_fit, args.n_eval, args.n_boot = 2, 30, 30, 100
        args.output_dir = str(pathlib.Path(args.output_dir) / "quick")

    summary = run(args)
    for arm, block in summary["by_arm"].items():
        s = block["summary"]
        print(f"\n{arm}: " + "  ".join(
            f"{m.replace('hsmm_','')}={s['auc'][m]['mean']:.3f}"
            for m in s["auc"]))
        print(f"   vs E1 {s['improvements']['explicit_minus_e1']['mean']:+.3f}"
              f"  vs geometric "
              f"{s['improvements']['explicit_minus_hsmm_geometric']['mean']:+.3f}"
              f"  vs shuffled "
              f"{s['improvements']['explicit_minus_hsmm_shuffled']['mean']:+.3f}"
              f"  | duration claim: {s['duration_claim_supported']}")
        if "primary_criterion" in s:
            print(f"   PRIMARY: {s['primary_criterion']}")
        if "non_inferior_to_e1" in s:
            print(f"   non-inferior to E1: {s['non_inferior_to_e1']}")
    print(f"\nelapsed {summary['elapsed_s']:.1f}s -> {args.output_dir}")


if __name__ == "__main__":
    main()
