#!/usr/bin/env python3
"""E0/E1 smoke test for duration-aware compute/communication events.

The name follows ``spec.md``'s HSMM research direction, but this first campaign
stops at the prerequisite gates:

* E0: oracle-aligned event-feature information ceiling;
* E1: blind marked-event extraction and a profiled renewal likelihood.

Only a successful E0/E1 result justifies implementing the explicit-duration
sample-level HSMM (E2).  All fitting, calibration, and evaluation populations
use disjoint deterministic RNG streams.  Outputs are isolated under
``results/hsmm_smoke`` and never touch frozen ST1/ST2 artefacts.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import sys
import time
from dataclasses import dataclass
from functools import partial

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from powerladder.config import DEFAULT  # noqa: E402
from powerladder.forward import TraceSpec, make_time_grid  # noqa: E402
from powerladder.ko_workload import (  # noqa: E402
    TrainingPhaseMetadata,
    training_F_phase_meta,
)
from powerladder.st1.nulls import (  # noqa: E402
    ar1,
    ar2_resonant,
    controller,
    controller_ar1,
)
from powerladder.st1.pipeline import (  # noqa: E402
    ST1_DETECTORS,
    stage4_semicoherent,
)
from powerladder.typeb.detectors import viterbi_statistic  # noqa: E402
from powerladder.typeb.ko_synth import _observe  # noqa: E402
from powerladder.typeb.renewal import (  # noqa: E402
    FEATURE_NAMES,
    blind_trace_features,
    event_timing_diagnostics,
    extract_events,
    fit_likelihood_from_features,
    oracle_trace_features,
)
from powerladder.typeb.roc import auc  # noqa: E402
from powerladder.typeb.rung2_scenarios import make_control_population  # noqa: E402
from powerladder.typeb.st2_attacks import make_negative_population  # noqa: E402


_ROOT = pathlib.Path(__file__).resolve().parent.parent
_DEFAULT_OUTPUT = _ROOT / "results" / "hsmm_smoke"
_FAR = 0.05

_CONDITIONS: tuple[tuple[str, str, float], ...] = (
    ("honest", "drift", 0.0),
    ("work_0.5", "work", 0.5),
    ("work_0.7", "work", 0.7),
    ("drift_0.8", "drift", 0.8),
    ("drift_1.5", "drift", 1.5),
)

_ST1_NULLS = {
    "ar1": ar1,
    "ar2_resonant": ar2_resonant,
    "controller": controller,
    "controller_ar1": controller_ar1,
}

_CADENCE_FEATURES = tuple(
    FEATURE_NAMES.index(name) for name in ("event_rate_hz", "interval_cv")
)
_AMPLITUDE_FEATURES = tuple(
    FEATURE_NAMES.index(name)
    for name in ("amplitude_median_z", "amplitude_cv")
)


@dataclass(frozen=True)
class AnnotatedTrace:
    """Observed positive trace plus evaluation-only latent phase metadata."""

    t: np.ndarray
    P_obs: np.ndarray
    meta: TrainingPhaseMetadata


def _training_kwargs(family: str, level: float) -> dict[str, float]:
    if family == "work":
        return {"work_sigma": level}
    if family == "drift":
        return {"f0_drift_hz": level}
    raise ValueError(f"unsupported positive family {family!r}")


def _make_annotated_population(family: str, level: float, n: int, glue,
                               seed: int) -> list[AnnotatedTrace]:
    rng = np.random.default_rng(seed)
    t = make_time_grid(glue.duration_s, 1.0 / glue.fs)
    f_peak = glue.f_peak_frac * DEFAULT.floor.F_max
    traces: list[AnnotatedTrace] = []
    for _ in range(n):
        f0 = float(rng.uniform(DEFAULT.ko.f0_lo, DEFAULT.ko.f0_hi))
        F, meta = training_F_phase_meta(
            t, DEFAULT.ko, rng, f_peak=f_peak, f0=f0,
            eta_scale=glue.eta_scale, **_training_kwargs(family, level),
        )
        spec = TraceSpec(t=t, F=F, r=np.full_like(t, glue.r),
                         P0=np.full_like(t, glue.P0))
        t_obs, p_obs = _observe(spec, glue, None, rng)
        traces.append(AnnotatedTrace(t_obs, p_obs, meta))
    return traces


def _make_null_populations(n: int, glue, seed: int) -> dict[str, list]:
    groups: dict[str, list] = {}
    groups["inference"] = make_negative_population(
        "work", n, DEFAULT.ko, glue, np.random.default_rng([seed, 0]),
        meter=None,
    )
    n_samples = int(round(glue.duration_s * glue.fs)) + 1
    for j, (name, generator) in enumerate(_ST1_NULLS.items(), start=1):
        rng = np.random.default_rng([seed, j])
        groups[name] = [
            generator(n_samples, glue.fs, DEFAULT.st1_null, rng)
            for _ in range(n)
        ]
    groups["periodic_inference"] = make_control_population(
        "periodic_inference", n, DEFAULT.ko, glue,
        np.random.default_rng([seed, 10]), DEFAULT.rung2, meter=None,
    )
    return groups


def _blind_matrix(traces) -> np.ndarray:
    return np.stack([
        blind_trace_features(trace.t, trace.P_obs) for trace in traces
    ])


def _oracle_matrix(traces: list[AnnotatedTrace]) -> np.ndarray:
    return np.stack([
        oracle_trace_features(
            trace.t, trace.P_obs, trace.meta.communication_starts,
            trace.meta.communication_durations,
        )
        for trace in traces
    ])


def _profile_scores(model, features: np.ndarray) -> np.ndarray:
    return np.asarray([model.score_features(row) for row in features])


def _without_cadence_features(features: np.ndarray) -> np.ndarray:
    """Ablation: remove event rate and interval variability from the score."""
    out = np.asarray(features, dtype=float).copy()
    out[..., list(_CADENCE_FEATURES)] = 0.0
    return out


def _shape_duration_features(features: np.ndarray) -> np.ndarray:
    """Stricter ablation: remove cadence and absolute-amplitude information."""
    out = np.asarray(features, dtype=float).copy()
    out[..., list(_CADENCE_FEATURES + _AMPLITUDE_FEATURES)] = 0.0
    return out


def _threshold_at_far(scores: np.ndarray, far: float) -> float:
    """Most permissive observed-score threshold whose empirical FAR is valid."""
    x = np.asarray(scores, dtype=float)
    unique = np.unique(x)
    # With the >= decision rule, a point mass at one score may be larger than
    # the FAR budget.  Thresholds immediately above each observed value are
    # therefore genuine operating points (excluding the whole tied mass), not
    # merely a numerical trick.  Omitting them can incorrectly leave +inf as
    # the only valid threshold for a perfectly separable discrete score.
    candidates = np.unique(np.concatenate([
        unique,
        np.nextafter(unique, np.inf),
        [np.inf],
    ]))
    valid = candidates[np.asarray([(x >= value).mean() <= far
                                   for value in candidates])]
    return float(np.min(valid))


def _composite_threshold(groups: dict[str, np.ndarray], far: float) -> float:
    return max(_threshold_at_far(scores, far) for scores in groups.values())


def _score_detector(traces, detector, glue) -> np.ndarray:
    return np.asarray([
        detector(trace.t, trace.P_obs, glue.band_lo, glue.band_hi)
        for trace in traces
    ])


def _metrics(pos: np.ndarray, neg_groups: dict[str, np.ndarray],
             threshold: float) -> dict:
    union = np.concatenate(list(neg_groups.values()))
    aucs = {name: auc(pos, scores) for name, scores in neg_groups.items()}
    return {
        "auc_union": auc(pos, union),
        "auc_by_null": aucs,
        "min_auc": min(aucs.values()),
        "tpr_at_composite_far": float((pos >= threshold).mean()),
        "score_median": float(np.median(pos)),
    }


def _timing_metrics(traces: list[AnnotatedTrace]) -> dict:
    rows = []
    for trace in traces:
        events = extract_events(trace.t, trace.P_obs)
        rows.append(event_timing_diagnostics(
            events.down_times, trace.meta.communication_starts,
            tolerance_s=0.1,
        ))
    values = np.asarray(rows)
    finite_error = values[np.isfinite(values[:, 2]), 2]
    return {
        "recall_mean": float(values[:, 0].mean()),
        "recall_median": float(np.median(values[:, 0])),
        "precision_mean": float(values[:, 1].mean()),
        "precision_median": float(np.median(values[:, 1])),
        "timing_error_median_s": (float(np.median(finite_error))
                                  if finite_error.size else None),
    }


def _null_rates(groups: dict[str, np.ndarray], threshold: float) -> dict[str, float]:
    return {name: float((scores >= threshold).mean())
            for name, scores in groups.items()}


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
    glue = dataclasses.replace(DEFAULT.ko_typeb, duration_s=args.duration_s)
    output = pathlib.Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    sizes = {"fit": args.n_fit, "calibration": args.n_cal,
             "evaluation": args.n_eval}
    streams = {"fit": args.seed + 10_000, "calibration": args.seed + 20_000,
               "evaluation": args.seed + 30_000}

    fit_pos: dict[str, list[AnnotatedTrace]] = {}
    for j, (name, family, level) in enumerate(_CONDITIONS):
        fit_pos[name] = _make_annotated_population(
            family, level, args.n_fit, glue, streams["fit"] + j,
        )
    fit_null = _make_null_populations(args.n_fit, glue, streams["fit"] + 100)
    blind_train = np.concatenate([_blind_matrix(group)
                                  for group in fit_pos.values()], axis=0)
    oracle_train = np.concatenate([_oracle_matrix(group)
                                   for group in fit_pos.values()], axis=0)
    fit_null_features = {name: _blind_matrix(group)
                         for name, group in fit_null.items()}
    blind_model = fit_likelihood_from_features(
        blind_train, fit_null_features.values(),
    )
    local_only_model = fit_likelihood_from_features(
        _without_cadence_features(blind_train),
        [_without_cadence_features(group)
         for group in fit_null_features.values()],
    )
    shape_duration_model = fit_likelihood_from_features(
        _shape_duration_features(blind_train),
        [_shape_duration_features(group)
         for group in fit_null_features.values()],
    )
    oracle_model = fit_likelihood_from_features(
        oracle_train, fit_null_features.values(),
    )

    detectors = {
        "viterbi": viterbi_statistic,
        "dg_order_full": ST1_DETECTORS["dg_order_full"],
        "dg_order_semicoh": partial(stage4_semicoherent, params=DEFAULT.st1),
    }

    cal_null = _make_null_populations(args.n_cal, glue,
                                      streams["calibration"] + 100)
    eval_null = _make_null_populations(args.n_eval, glue,
                                       streams["evaluation"] + 100)
    cal_null_features = {name: _blind_matrix(group)
                         for name, group in cal_null.items()}
    eval_null_features = {name: _blind_matrix(group)
                          for name, group in eval_null.items()}

    cal_scores: dict[str, dict[str, np.ndarray]] = {
        "renewal": {name: _profile_scores(blind_model, matrix)
                    for name, matrix in cal_null_features.items()},
        "renewal_local_only": {
            name: _profile_scores(local_only_model,
                                  _without_cadence_features(matrix))
            for name, matrix in cal_null_features.items()
        },
        "renewal_shape_duration": {
            name: _profile_scores(shape_duration_model,
                                  _shape_duration_features(matrix))
            for name, matrix in cal_null_features.items()
        },
        "oracle": {name: _profile_scores(oracle_model, matrix)
                   for name, matrix in cal_null_features.items()},
    }
    eval_scores: dict[str, dict[str, np.ndarray]] = {
        "renewal": {name: _profile_scores(blind_model, matrix)
                    for name, matrix in eval_null_features.items()},
        "renewal_local_only": {
            name: _profile_scores(local_only_model,
                                  _without_cadence_features(matrix))
            for name, matrix in eval_null_features.items()
        },
        "renewal_shape_duration": {
            name: _profile_scores(shape_duration_model,
                                  _shape_duration_features(matrix))
            for name, matrix in eval_null_features.items()
        },
        "oracle": {name: _profile_scores(oracle_model, matrix)
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

    thresholds = {name: _composite_threshold(groups, _FAR)
                  for name, groups in cal_scores.items()}
    calibration_fpr = {name: _null_rates(groups, thresholds[name])
                       for name, groups in cal_scores.items()}
    evaluation_fpr = {name: _null_rates(eval_scores[name], thresholds[name])
                      for name in thresholds}

    conditions: dict[str, dict] = {}
    score_plot_groups: dict[str, np.ndarray] = {}
    for j, (name, family, level) in enumerate(_CONDITIONS):
        traces = _make_annotated_population(
            family, level, args.n_eval, glue, streams["evaluation"] + j,
        )
        blind_features = _blind_matrix(traces)
        oracle_features = _oracle_matrix(traces)
        pos_scores = {
            "renewal": _profile_scores(blind_model, blind_features),
            "renewal_local_only": _profile_scores(
                local_only_model, _without_cadence_features(blind_features),
            ),
            "renewal_shape_duration": _profile_scores(
                shape_duration_model, _shape_duration_features(blind_features),
            ),
            "oracle": _profile_scores(oracle_model, oracle_features),
        }
        for detector_name, detector in detectors.items():
            pos_scores[detector_name] = _score_detector(traces, detector, glue)
        metrics = {
            detector_name: _metrics(scores, eval_scores[detector_name],
                                    thresholds[detector_name])
            for detector_name, scores in pos_scores.items()
        }
        oracle_auc = metrics["oracle"]["auc_union"]
        blind_auc = metrics["renewal"]["auc_union"]
        auc_recovery = ((blind_auc - 0.5) / (oracle_auc - 0.5)
                        if oracle_auc > 0.5 else 0.0)
        conditions[name] = {
            "family": family,
            "level": level,
            "timing": _timing_metrics(traces),
            "oracle_auc_recovered_by_blind": float(auc_recovery),
            **metrics,
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
            "local_only_removed_features": [
                FEATURE_NAMES[index] for index in _CADENCE_FEATURES
            ],
            "shape_duration_worst_hard_tpr": shape_duration_worst,
            "shape_duration_removed_features": [
                FEATURE_NAMES[index]
                for index in (_CADENCE_FEATURES + _AMPLITUDE_FEATURES)
            ],
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
