#!/usr/bin/env python3
"""E0/E1 smoke test for duration-aware compute/communication events.

The name follows ``spec.md``'s HSMM research direction, but this first campaign
stops at the prerequisite gates:

* E0: oracle-aligned event-feature information ceiling;
* E1: blind marked-event extraction and a profiled renewal likelihood.

Both gates passed (``notes/results/hsmm-renewal-smoke-findings.md``), but the
result is saturated, so E2 is deferred behind two stress tests -- see
``notes/plans/hsmm-e2-preconditions-plan.md``, ``scripts/hsmm_generalization.py``
and ``scripts/hsmm_meter.py``.

All fitting, calibration, and evaluation populations use disjoint deterministic
RNG streams.  Outputs are isolated under ``results/hsmm_smoke`` and never touch
frozen ST1/ST2 artefacts.
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
from powerladder.st1.pipeline import (  # noqa: E402
    ST1_DETECTORS,
    stage4_semicoherent,
)
from powerladder.typeb.detectors import viterbi_statistic  # noqa: E402
from powerladder.typeb.renewal import (  # noqa: E402
    FEATURE_NAMES,
    fit_likelihood_from_features,
)
from powerladder.typeb.renewal_campaign import (  # noqa: E402
    CONDITIONS,
    FEATURE_SET_REMOVED,
    FEATURE_SETS,
    blind_matrix,
    composite_threshold,
    glue_for,
    make_annotated_population,
    make_null_populations,
    metrics,
    null_rates,
    oracle_matrix,
    profile_scores,
    timing_metrics,
)


_ROOT = pathlib.Path(__file__).resolve().parent.parent
_DEFAULT_OUTPUT = _ROOT / "results" / "hsmm_smoke"
_FAR = 0.05

_local_only = FEATURE_SETS["local_only"]
_shape_duration = FEATURE_SETS["shape_duration"]


def _score_detector(traces, detector, glue) -> np.ndarray:
    return np.asarray([
        detector(trace.t, trace.P_obs, glue.band_lo, glue.band_hi)
        for trace in traces
    ])


def _plot_scores(score_groups: dict[str, np.ndarray], output: pathlib.Path) -> None:
    labels = list(score_groups)
    values = [score_groups[label] for label in labels]
    fig, ax = plt.subplots(figsize=(9.0, 3.5))
    ax.boxplot(values, tick_labels=[label.replace("_", "\n") for label in labels],
               showfliers=False)
    ax.axhline(0.0, color="0.5", lw=0.8, ls="--")
    ax.set_ylabel("Blind renewal log-likelihood score")
    ax.tick_params(axis="x", labelsize=7)
    fig.tight_layout()
    fig.savefig(output / "score_distributions.png", dpi=180)
    plt.close(fig)


def _plot_tpr(condition_metrics: dict[str, dict[str, dict]],
              output: pathlib.Path) -> None:
    detectors = ("renewal", "renewal_local_only", "renewal_shape_duration",
                 "viterbi", "dg_order_full", "dg_order_semicoh")
    labels = list(condition_metrics)
    x = np.arange(len(labels))
    width = 0.13
    fig, ax = plt.subplots(figsize=(8.2, 3.5))
    for j, detector in enumerate(detectors):
        values = [condition_metrics[name][detector]["tpr_at_composite_far"]
                  for name in labels]
        ax.bar(x + (j - 2.5) * width, values, width, label=detector)
    ax.set_xticks(x, [label.replace("_", "\n") for label in labels])
    ax.set_ylim(0.0, 1.04)
    ax.set_ylabel("TPR at composite-null FAR 0.05")
    ax.legend(frameon=False, fontsize=7, ncols=2)
    fig.tight_layout()
    fig.savefig(output / "composite_tpr.png", dpi=180)
    plt.close(fig)


def run(args: argparse.Namespace) -> dict:
    t0 = time.time()
    glue = glue_for(args.duration_s)
    output = pathlib.Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    sizes = {"fit": args.n_fit, "calibration": args.n_cal,
             "evaluation": args.n_eval}
    streams = {"fit": args.seed + 10_000, "calibration": args.seed + 20_000,
               "evaluation": args.seed + 30_000}

    fit_pos: dict[str, list] = {}
    for j, (name, family, level) in enumerate(CONDITIONS):
        fit_pos[name] = make_annotated_population(
            family, level, args.n_fit, glue, streams["fit"] + j,
        )
    fit_null = make_null_populations(args.n_fit, glue, streams["fit"] + 100)
    blind_train = np.concatenate([blind_matrix(group)
                                  for group in fit_pos.values()], axis=0)
    oracle_train = np.concatenate([oracle_matrix(group)
                                   for group in fit_pos.values()], axis=0)
    fit_null_features = {name: blind_matrix(group)
                         for name, group in fit_null.items()}
    blind_model = fit_likelihood_from_features(
        blind_train, fit_null_features.values(),
    )
    local_only_model = fit_likelihood_from_features(
        _local_only(blind_train),
        [_local_only(group) for group in fit_null_features.values()],
    )
    shape_duration_model = fit_likelihood_from_features(
        _shape_duration(blind_train),
        [_shape_duration(group) for group in fit_null_features.values()],
    )
    oracle_model = fit_likelihood_from_features(
        oracle_train, fit_null_features.values(),
    )

    detectors = {
        "viterbi": viterbi_statistic,
        "dg_order_full": ST1_DETECTORS["dg_order_full"],
        "dg_order_semicoh": partial(stage4_semicoherent, params=DEFAULT.st1),
    }

    cal_null = make_null_populations(args.n_cal, glue,
                                     streams["calibration"] + 100)
    eval_null = make_null_populations(args.n_eval, glue,
                                      streams["evaluation"] + 100)
    cal_null_features = {name: blind_matrix(group)
                         for name, group in cal_null.items()}
    eval_null_features = {name: blind_matrix(group)
                          for name, group in eval_null.items()}

    cal_scores: dict[str, dict[str, np.ndarray]] = {
        "renewal": {name: profile_scores(blind_model, matrix)
                    for name, matrix in cal_null_features.items()},
        "renewal_local_only": {
            name: profile_scores(local_only_model, _local_only(matrix))
            for name, matrix in cal_null_features.items()
        },
        "renewal_shape_duration": {
            name: profile_scores(shape_duration_model, _shape_duration(matrix))
            for name, matrix in cal_null_features.items()
        },
        "oracle": {name: profile_scores(oracle_model, matrix)
                   for name, matrix in cal_null_features.items()},
    }
    eval_scores: dict[str, dict[str, np.ndarray]] = {
        "renewal": {name: profile_scores(blind_model, matrix)
                    for name, matrix in eval_null_features.items()},
        "renewal_local_only": {
            name: profile_scores(local_only_model, _local_only(matrix))
            for name, matrix in eval_null_features.items()
        },
        "renewal_shape_duration": {
            name: profile_scores(shape_duration_model, _shape_duration(matrix))
            for name, matrix in eval_null_features.items()
        },
        "oracle": {name: profile_scores(oracle_model, matrix)
                   for name, matrix in eval_null_features.items()},
    }
    for detector_name, detector in detectors.items():
        cal_scores[detector_name] = {
            name: _score_detector(group, detector, glue)
            for name, group in cal_null.items()
        }
        eval_scores[detector_name] = {
            name: _score_detector(group, detector, glue)
            for name, group in eval_null.items()
        }

    thresholds = {name: composite_threshold(groups, _FAR)
                  for name, groups in cal_scores.items()}
    calibration_fpr = {name: null_rates(groups, thresholds[name])
                       for name, groups in cal_scores.items()}
    evaluation_fpr = {name: null_rates(eval_scores[name], thresholds[name])
                      for name in thresholds}

    conditions: dict[str, dict] = {}
    score_plot_groups: dict[str, np.ndarray] = {}
    for j, (name, family, level) in enumerate(CONDITIONS):
        traces = make_annotated_population(
            family, level, args.n_eval, glue, streams["evaluation"] + j,
        )
        blind_features = blind_matrix(traces)
        oracle_features = oracle_matrix(traces)
        pos_scores = {
            "renewal": profile_scores(blind_model, blind_features),
            "renewal_local_only": profile_scores(
                local_only_model, _local_only(blind_features),
            ),
            "renewal_shape_duration": profile_scores(
                shape_duration_model, _shape_duration(blind_features),
            ),
            "oracle": profile_scores(oracle_model, oracle_features),
        }
        for detector_name, detector in detectors.items():
            pos_scores[detector_name] = _score_detector(traces, detector, glue)
        condition_metrics = {
            detector_name: metrics(scores, eval_scores[detector_name],
                                   thresholds[detector_name])
            for detector_name, scores in pos_scores.items()
        }
        oracle_auc = condition_metrics["oracle"]["auc_union"]
        blind_auc = condition_metrics["renewal"]["auc_union"]
        auc_recovery = ((blind_auc - 0.5) / (oracle_auc - 0.5)
                        if oracle_auc > 0.5 else 0.0)
        conditions[name] = {
            "family": family,
            "level": level,
            "timing": timing_metrics(traces),
            "oracle_auc_recovered_by_blind": float(auc_recovery),
            **condition_metrics,
        }
        score_plot_groups[name] = pos_scores["renewal"]

    for name, scores in eval_scores["renewal"].items():
        score_plot_groups[f"null:{name}"] = scores

    hard = ("work_0.7", "drift_0.8", "drift_1.5")
    renewal_worst = min(conditions[name]["renewal"]["tpr_at_composite_far"]
                        for name in hard)
    baseline_best = {
        name: max(conditions[name][detector]["tpr_at_composite_far"]
                  for detector in detectors)
        for name in hard
    }
    baseline_worst = min(baseline_best.values())
    worst_improvement = renewal_worst - baseline_worst
    local_only_worst = min(
        conditions[name]["renewal_local_only"]["tpr_at_composite_far"]
        for name in hard
    )
    shape_duration_worst = min(
        conditions[name]["renewal_shape_duration"]["tpr_at_composite_far"]
        for name in hard
    )
    work = conditions["work_0.7"]
    e0 = work["oracle"]["min_auc"] >= 0.90
    e1 = (work["timing"]["recall_mean"] >= 0.70
          and work["oracle_auc_recovered_by_blind"] >= 0.70)
    fpr_ok = max(evaluation_fpr["renewal"].values()) <= 0.10
    overall = (
        e0 and e1 and fpr_ok
        and conditions["honest"]["renewal"]["tpr_at_composite_far"] >= 0.90
        and worst_improvement >= 0.15
    )

    summary = {
        "status": "smoke-test only; not a frozen paper result",
        "method": "oracle E0 plus blind marked-renewal E1; no HSMM E2 yet",
        "duration_s": args.duration_s,
        "fs": glue.fs,
        "sizes": sizes,
        "seed": args.seed,
        "target_far": _FAR,
        "feature_names": list(FEATURE_NAMES),
        "null_families": list(eval_null),
        "thresholds": thresholds,
        "calibration_fpr": calibration_fpr,
        "evaluation_fpr": evaluation_fpr,
        "conditions": conditions,
        "gate": {
            "E0_oracle_continue": e0,
            "E1_blind_continue": e1,
            "evaluation_fpr_within_smoke_tolerance": fpr_ok,
            "renewal_worst_hard_tpr": renewal_worst,
            "best_existing_worst_hard_tpr": baseline_worst,
            "worst_hard_tpr_improvement": worst_improvement,
            "local_only_worst_hard_tpr": local_only_worst,
            "local_only_removed_features": list(FEATURE_SET_REMOVED["local_only"]),
            "shape_duration_worst_hard_tpr": shape_duration_worst,
            "shape_duration_removed_features": list(
                FEATURE_SET_REMOVED["shape_duration"]
            ),
            "overall_go_to_E2": overall,
        },
        "elapsed_s": time.time() - t0,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    _plot_scores(score_plot_groups, output)
    _plot_tpr(conditions, output)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration-s", type=float, default=90.0)
    parser.add_argument("--n-fit", type=int, default=50)
    parser.add_argument("--n-cal", type=int, default=200)
    parser.add_argument("--n-eval", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--output-dir", default=str(_DEFAULT_OUTPUT))
    parser.add_argument("--quick", action="store_true",
                        help="plumbing run: 10 fit / 20 calibration / 20 eval")
    args = parser.parse_args()
    if args.quick:
        args.n_fit, args.n_cal, args.n_eval = 10, 20, 20
        args.duration_s = min(args.duration_s, 60.0)
        args.output_dir = str(pathlib.Path(args.output_dir) / "quick")

    summary = run(args)
    gate = summary["gate"]
    print(json.dumps(gate, indent=2))
    print(f"elapsed {summary['elapsed_s']:.1f}s -> {args.output_dir}")


if __name__ == "__main__":
    main()
