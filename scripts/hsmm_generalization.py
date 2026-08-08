#!/usr/bin/env python3
"""Stage 1: generalization falsification for the E1 marked-renewal detector.

The E0/E1 smoke fitted the positive profile on the concatenation of all five
positive conditions and then evaluated those same named families, so its perfect
scores cannot distinguish "the features capture an invariant" from "the profile
interpolates among known attacks".  This campaign separates the two before any
work is spent on E2 (``notes/plans/hsmm-e2-preconditions-plan.md``).

Three tests, each run for the full, cadence-free and shape/duration-only feature
sets:

* **1A honest-only fit** -- fit on honest traces alone, evaluate work and drift.
* **1B leave-one-positive-family-out** -- fit on honest+drift and evaluate work;
  fit on honest+work and evaluate drift.
* **1C leave-one-null-out** -- drop one confuser from the composite denominator,
  either retaining it in calibration (representation generalization) or
  excluding it from calibration too (truly unseen-confuser FPR, which carries no
  claim of formal level control).

Every configuration draws the *same* total number of fitting traces (default
250), so a loss of power cannot be a fit-set size artefact.  Trace features are
model-independent, so populations are built and featurised once and each
configuration is a cheap re-fit over the cached matrices.

Outputs land in ``results/hsmm_generalization/`` and never touch frozen ST1/ST2
artefacts or the E0/E1 smoke.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from powerladder.typeb.renewal import (  # noqa: E402
    FEATURE_NAMES,
    fit_likelihood_from_features,
    widen_profile,
)
from powerladder.typeb.renewal_campaign import (  # noqa: E402
    CONDITIONS,
    FEATURE_SET_REMOVED,
    FEATURE_SETS,
    NULL_FAMILIES,
    blind_matrix,
    composite_threshold,
    glue_for,
    make_annotated_population,
    make_null_populations,
    metrics,
    null_rates,
    oracle_matrix,
    profile_scores,
)


_ROOT = pathlib.Path(__file__).resolve().parent.parent
_DEFAULT_OUTPUT = _ROOT / "results" / "hsmm_generalization"
_FAR = 0.05

_CONDITION_NAMES = tuple(name for name, _, _ in CONDITIONS)
_WORK_CONDITIONS = ("work_0.5", "work_0.7")
_DRIFT_CONDITIONS = ("drift_0.8", "drift_1.5")

#: A configuration counts as a material decline against the pooled baseline at
#: either of these margins, on any evaluated condition and feature set.
_TPR_DECLINE = 0.10
_AUC_DECLINE = 0.05
#: Held-out-null false-alarm tolerance, matching the E0/E1 smoke.
_FPR_TOLERANCE = 0.10


def _even_split(total: int, k: int) -> list[int]:
    """Split ``total`` fitting traces as evenly as possible over ``k`` conditions."""
    base, remainder = divmod(total, k)
    return [base + (1 if j < remainder else 0) for j in range(k)]


def _fit_plan(conditions: tuple[str, ...], total: int) -> dict[str, int]:
    return dict(zip(conditions, _even_split(total, len(conditions))))


def _positive_fit_configs(total: int) -> dict[str, dict[str, int]]:
    """1A/1B fitting plans, each drawing the same total number of traces."""
    return {
        "pooled_baseline": _fit_plan(_CONDITION_NAMES, total),
        "honest_only": _fit_plan(("honest",), total),
        "fit_honest_drift": _fit_plan(("honest",) + _DRIFT_CONDITIONS, total),
        "fit_honest_work": _fit_plan(("honest",) + _WORK_CONDITIONS, total),
    }


def _held_out_conditions(plan: dict[str, int]) -> list[str]:
    return [name for name in _CONDITION_NAMES if name not in plan]


def _stack_fit_features(cache: dict[str, np.ndarray],
                        plan: dict[str, int]) -> np.ndarray:
    """Take the first ``plan[name]`` rows of each permitted condition's pool."""
    return np.concatenate(
        [cache[name][:count] for name, count in plan.items() if count > 0],
        axis=0,
    )


def _evaluate(model, feature_set, eval_pos: dict[str, np.ndarray],
              eval_null: dict[str, np.ndarray],
              cal_null: dict[str, np.ndarray],
              calibration_families: tuple[str, ...]) -> dict:
    """Score every population, calibrate on the permitted nulls, and report."""
    transform = FEATURE_SETS[feature_set]
    cal_scores = {name: profile_scores(model, transform(cal_null[name]))
                  for name in calibration_families}
    threshold = composite_threshold(cal_scores, _FAR)
    eval_scores = {name: profile_scores(model, transform(matrix))
                   for name, matrix in eval_null.items()}
    conditions = {}
    for name, matrix in eval_pos.items():
        pos = profile_scores(model, transform(matrix))
        conditions[name] = metrics(pos, eval_scores, threshold)
    return {
        "threshold": threshold,
        "calibration_fpr": null_rates(
            {name: cal_scores[name] for name in calibration_families}, threshold,
        ),
        "evaluation_fpr": null_rates(eval_scores, threshold),
        "conditions": conditions,
    }


def _fit_model(train: np.ndarray, null_features: dict[str, np.ndarray],
               fit_families: tuple[str, ...], feature_set: str,
               shrinkage: float = 0.25):
    transform = FEATURE_SETS[feature_set]
    return fit_likelihood_from_features(
        transform(train),
        [transform(null_features[name]) for name in fit_families],
        shrinkage=shrinkage,
    )


def _profile_width_sensitivity(fit_pos_features, fit_null_features,
                               eval_pos_features, eval_null_features,
                               cal_null_features, total: int) -> dict:
    """Is the honest-only power loss missing information, or a too-tight profile?

    1B shows that fitting on *any* attack family restores full power, which
    suggests the honest-only fit simply produces a narrow Gaussian whose tails
    reject severe attacks.  If widening the covariance recovers the loss, the
    honest-only gap is a regularisation artefact and not an estimation gap an
    explicit-duration model would close.
    """
    plan = _fit_plan(("honest",), total)
    train = _stack_fit_features(fit_pos_features, plan)
    base = _fit_model(train, fit_null_features, NULL_FAMILIES, "full")

    shrinkage: dict[str, dict] = {}
    for value in (0.25, 0.5, 0.75, 0.9, 0.99):
        model = _fit_model(train, fit_null_features, NULL_FAMILIES, "full",
                           shrinkage=value)
        shrinkage[f"{value:.2f}"] = _evaluate(
            model, "full", eval_pos_features, eval_null_features,
            cal_null_features, NULL_FAMILIES,
        )

    # Shrinkage preserves every marginal variance, so it tests correlation
    # structure, not width.  Inflating the training covariance isotropically is
    # the actual width test: it is the cheapest alternative to E2 and must be
    # ruled out before an explicit-duration model is justified.
    width: dict[str, dict] = {}
    for factor in (1.0, 2.0, 5.0, 10.0, 50.0, 200.0):
        model = dataclasses.replace(
            base, training=widen_profile(base.training, factor),
        )
        width[f"{factor:g}"] = _evaluate(
            model, "full", eval_pos_features, eval_null_features,
            cal_null_features, NULL_FAMILIES,
        )
    return {"shrinkage": shrinkage, "training_covariance_inflation": width}


def _plot_tpr(configurations: dict[str, dict], output: pathlib.Path) -> None:
    """Held-out TPR per positive-fit configuration, one panel per feature set."""
    configs = [name for name, block in configurations.items()
               if block["kind"] == "positive_fit"]
    feature_sets = list(FEATURE_SETS)
    fig, axes = plt.subplots(1, len(feature_sets), figsize=(11.0, 3.4),
                             sharey=True)
    x = np.arange(len(_CONDITION_NAMES))
    width = 0.8 / max(len(configs), 1)
    for ax, feature_set in zip(axes, feature_sets):
        for j, config in enumerate(configs):
            block = configurations[config]["feature_sets"][feature_set]
            values = [block["conditions"][name]["tpr_at_composite_far"]
                      for name in _CONDITION_NAMES]
            ax.bar(x + (j - (len(configs) - 1) / 2) * width, values, width,
                   label=config)
        ax.set_xticks(x, [name.replace("_", "\n") for name in _CONDITION_NAMES],
                      fontsize=7)
        ax.set_title(feature_set, fontsize=9)
        ax.set_ylim(0.0, 1.04)
    axes[0].set_ylabel("TPR at composite-null FAR 0.05")
    axes[-1].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(output / "generalization_tpr.png", dpi=180)
    plt.close(fig)


def _plot_null_holdout(configurations: dict[str, dict],
                       output: pathlib.Path) -> None:
    """Held-out confuser FPR, retained-in-calibration vs excluded."""
    families = list(NULL_FAMILIES)
    fig, ax = plt.subplots(figsize=(8.0, 3.4))
    x = np.arange(len(families))
    width = 0.4
    for j, variant in enumerate(("retained", "excluded")):
        values = []
        for family in families:
            block = configurations[f"null_holdout_{variant}:{family}"]
            values.append(
                block["feature_sets"]["full"]["evaluation_fpr"][family]
            )
        ax.bar(x + (j - 0.5) * width, values, width,
               label=f"{variant} in calibration")
    ax.axhline(_FAR, color="0.4", ls="--", lw=0.9, label="target FAR 0.05")
    ax.axhline(_FPR_TOLERANCE, color="0.7", ls=":", lw=0.9,
               label="smoke tolerance 0.10")
    ax.set_xticks(x, [name.replace("_", "\n") for name in families], fontsize=7)
    ax.set_ylabel("Held-out confuser FPR")
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(output / "null_holdout_fpr.png", dpi=180)
    plt.close(fig)


def run(args: argparse.Namespace) -> dict:
    t0 = time.time()
    glue = glue_for(args.duration_s)
    output = pathlib.Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    streams = {"fit": args.seed + 10_000, "calibration": args.seed + 20_000,
               "evaluation": args.seed + 30_000}

    # --- populations, built once -------------------------------------------
    # Each condition contributes a pool of `n_fit_total` traces; a configuration
    # takes a prefix of the pool, so every fit uses the same total count.
    fit_pos_features: dict[str, np.ndarray] = {}
    fit_pos_oracle: dict[str, np.ndarray] = {}
    for j, (name, family, level) in enumerate(CONDITIONS):
        traces = make_annotated_population(
            family, level, args.n_fit_total, glue, streams["fit"] + j,
        )
        fit_pos_features[name] = blind_matrix(traces)
        fit_pos_oracle[name] = oracle_matrix(traces)

    fit_null = make_null_populations(args.n_fit_null, glue, streams["fit"] + 100)
    fit_null_features = {name: blind_matrix(group)
                         for name, group in fit_null.items()}

    cal_null = make_null_populations(args.n_cal, glue,
                                     streams["calibration"] + 100)
    cal_null_features = {name: blind_matrix(group)
                         for name, group in cal_null.items()}

    eval_null = make_null_populations(args.n_eval, glue,
                                      streams["evaluation"] + 100)
    eval_null_features = {name: blind_matrix(group)
                          for name, group in eval_null.items()}

    eval_pos_features: dict[str, np.ndarray] = {}
    eval_pos_oracle: dict[str, np.ndarray] = {}
    for j, (name, family, level) in enumerate(CONDITIONS):
        traces = make_annotated_population(
            family, level, args.n_eval, glue, streams["evaluation"] + j,
        )
        eval_pos_features[name] = blind_matrix(traces)
        eval_pos_oracle[name] = oracle_matrix(traces)

    configurations: dict[str, dict] = {}

    # --- 1A and 1B: positive-fit generalization ----------------------------
    for config, plan in _positive_fit_configs(args.n_fit_total).items():
        train = _stack_fit_features(fit_pos_features, plan)
        block: dict = {
            "kind": "positive_fit",
            "fit_conditions": plan,
            "fit_null_families": list(NULL_FAMILIES),
            "calibration_null_families": list(NULL_FAMILIES),
            "held_out_conditions": _held_out_conditions(plan),
            "held_out_null": None,
            "n_fit_total": int(train.shape[0]),
            "feature_sets": {},
        }
        for feature_set in FEATURE_SETS:
            model = _fit_model(train, fit_null_features, NULL_FAMILIES,
                               feature_set)
            block["feature_sets"][feature_set] = _evaluate(
                model, feature_set, eval_pos_features, eval_null_features,
                cal_null_features, NULL_FAMILIES,
            )
        # The oracle under the same fitting restriction separates a failure of
        # feature invariance from a failure of blind estimation.
        oracle_train = _stack_fit_features(fit_pos_oracle, plan)
        oracle_model = _fit_model(oracle_train, fit_null_features,
                                  NULL_FAMILIES, "full")
        block["oracle"] = _evaluate(
            oracle_model, "full", eval_pos_oracle, eval_null_features,
            cal_null_features, NULL_FAMILIES,
        )
        configurations[config] = block

    # --- 1C: leave-one-null-out --------------------------------------------
    pooled_plan = _positive_fit_configs(args.n_fit_total)["pooled_baseline"]
    pooled_train = _stack_fit_features(fit_pos_features, pooled_plan)
    for family in NULL_FAMILIES:
        kept = tuple(name for name in NULL_FAMILIES if name != family)
        for variant, calibration_families in (("retained", NULL_FAMILIES),
                                              ("excluded", kept)):
            block = {
                "kind": "null_holdout",
                "fit_conditions": pooled_plan,
                "fit_null_families": list(kept),
                "calibration_null_families": list(calibration_families),
                "held_out_conditions": [],
                "held_out_null": family,
                "held_out_null_in_calibration": variant == "retained",
                "n_fit_total": int(pooled_train.shape[0]),
                "feature_sets": {},
            }
            for feature_set in FEATURE_SETS:
                model = _fit_model(pooled_train, fit_null_features, kept,
                                   feature_set)
                block["feature_sets"][feature_set] = _evaluate(
                    model, feature_set, eval_pos_features, eval_null_features,
                    cal_null_features, calibration_families,
                )
            configurations[f"null_holdout_{variant}:{family}"] = block

    summary = {
        "status": "stage-1 generalization falsification; not a frozen paper result",
        "method": ("blind marked-renewal E1 re-fitted under positive-family and "
                   "null-family holdouts; no HSMM E2"),
        "duration_s": args.duration_s,
        "fs": glue.fs,
        "sizes": {"fit_total_per_config": args.n_fit_total,
                  "fit_pool_per_condition": args.n_fit_total,
                  "fit_per_null_family": args.n_fit_null,
                  "calibration": args.n_cal, "evaluation": args.n_eval},
        "seed": args.seed,
        "target_far": _FAR,
        "feature_names": list(FEATURE_NAMES),
        "feature_set_removed": {name: list(removed)
                                for name, removed in FEATURE_SET_REMOVED.items()},
        "null_families": list(NULL_FAMILIES),
        "configurations": configurations,
    }
    summary["profile_width_sensitivity"] = _profile_width_sensitivity(
        fit_pos_features, fit_null_features, eval_pos_features,
        eval_null_features, cal_null_features, args.n_fit_total,
    )
    summary["verdict"] = _verdict(configurations)
    summary["elapsed_s"] = time.time() - t0
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    _plot_tpr(configurations, output)
    _plot_null_holdout(configurations, output)
    return summary


def _verdict(configurations: dict[str, dict]) -> dict:
    """Locate any configuration where E1 materially weakens."""
    baseline = configurations["pooled_baseline"]["feature_sets"]
    declines: list[dict] = []
    for config, block in configurations.items():
        if config == "pooled_baseline":
            continue
        for feature_set, result in block["feature_sets"].items():
            reference = baseline[feature_set]["conditions"]
            for name, values in result["conditions"].items():
                tpr_drop = (reference[name]["tpr_at_composite_far"]
                            - values["tpr_at_composite_far"])
                auc_drop = reference[name]["auc_union"] - values["auc_union"]
                if tpr_drop >= _TPR_DECLINE or auc_drop >= _AUC_DECLINE:
                    declines.append({
                        "configuration": config,
                        "feature_set": feature_set,
                        "condition": name,
                        "held_out": name in block["held_out_conditions"],
                        "tpr": values["tpr_at_composite_far"],
                        "tpr_drop": float(tpr_drop),
                        "auc_union": values["auc_union"],
                        "auc_drop": float(auc_drop),
                    })

    holdout_breaches = []
    for config, block in configurations.items():
        if block["kind"] != "null_holdout":
            continue
        family = block["held_out_null"]
        for feature_set, result in block["feature_sets"].items():
            fpr = result["evaluation_fpr"][family]
            if fpr > _FPR_TOLERANCE:
                holdout_breaches.append({
                    "configuration": config,
                    "feature_set": feature_set,
                    "held_out_null": family,
                    "evaluation_fpr": fpr,
                })

    held_out_tprs = [
        block["feature_sets"][feature_set]["conditions"][name][
            "tpr_at_composite_far"]
        for block in configurations.values()
        for feature_set in block["feature_sets"]
        for name in block["held_out_conditions"]
    ]
    return {
        "tpr_decline_margin": _TPR_DECLINE,
        "auc_decline_margin": _AUC_DECLINE,
        "held_out_null_fpr_tolerance": _FPR_TOLERANCE,
        "n_material_declines": len(declines),
        "material_declines": declines,
        "n_held_out_null_fpr_breaches": len(holdout_breaches),
        "held_out_null_fpr_breaches": holdout_breaches,
        "worst_held_out_condition_tpr": (min(held_out_tprs)
                                         if held_out_tprs else None),
        "generalizes_without_material_decline": (
            not declines and not holdout_breaches
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration-s", type=float, default=90.0)
    parser.add_argument("--n-fit-total", type=int, default=250,
                        help="fitting traces per configuration (held constant)")
    parser.add_argument("--n-fit-null", type=int, default=50,
                        help="fitting traces per null family")
    parser.add_argument("--n-cal", type=int, default=200)
    parser.add_argument("--n-eval", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--output-dir", default=str(_DEFAULT_OUTPUT))
    parser.add_argument("--quick", action="store_true",
                        help="plumbing run: 25 fit / 10 null fit / 20 cal / 20 eval")
    args = parser.parse_args()
    if args.quick:
        args.n_fit_total, args.n_fit_null = 25, 10
        args.n_cal, args.n_eval = 20, 20
        args.duration_s = min(args.duration_s, 60.0)
        args.output_dir = str(pathlib.Path(args.output_dir) / "quick")

    summary = run(args)
    print(json.dumps(summary["verdict"], indent=2)[:4000])
    print(f"elapsed {summary['elapsed_s']:.1f}s -> {args.output_dir}")


if __name__ == "__main__":
    main()
