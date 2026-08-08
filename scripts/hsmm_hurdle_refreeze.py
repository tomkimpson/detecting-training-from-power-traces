#!/usr/bin/env python3
"""Re-freeze the locked cell under the hurdle likelihood, and audit the old ones.

Step 1-3 of the agreed order (2026-08-08):

1. the hurdle / missing-feature representation is implemented in
   ``powerladder.typeb.renewal_hurdle`` and validated by ``tests/test_renewal_hurdle.py``;
2. **this script** re-runs the locked cell -- no new search -- and re-freezes the
   corrected E1 baseline;
3. it re-runs the composite arm with **all** null families retained, which the
   hurdle model makes legitimate: a higher no-event rate among physical nulls is
   now explicit, auditable evidence rather than eleven zeros under a Gaussian.

Two extras the fix makes possible:

* a **count-only ablation** (``p(N=0)`` and the truncated count term, no marks),
  so that if composite separation comes mostly from the no-event rate we say so;
* an **audit** (``--audit``) of the earlier campaigns' operating points by
  zero-event rate per class and per null family.  Only cells with material or
  asymmetric rates need re-running.

The pre-fix confirmation stays frozen in
``notes/results/hsmm-cell-confirmation-and-e2-precommitment.md``.  Its constants
are **not** mixed with the corrected ones; instead the absolute AUC targets it
implied are carried forward so the E2 success criterion cannot drift:

    AAFT  0.6477 + 0.5 * 0.3177 = 0.8066
    IAAFT 0.6437 + 0.5 * 0.3313 = 0.8094
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

from powerladder.typeb.renewal_campaign import (  # noqa: E402
    NULL_FAMILIES,
    alignment_pool,
    glue_for,
    make_annotated_population,
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
_DEFAULT_OUTPUT = _ROOT / "results" / "hsmm_hurdle"

#: The locked cell.  Frozen -- this script confirms, it does not search.
CELL = {"duration_s": 10.0, "share": 0.2}
NULL_KINDS = ("aaft", "iaaft", "composite")

#: Absolute AUC targets implied by the pre-fix confirmation at R = 0.50.
#: Carried forward so the E2 bar cannot move when the baseline is corrected.
PREFIX_ABSOLUTE_TARGETS = {"aaft": 0.8066, "iaaft": 0.8094}
PREFIX_BASELINE = {
    "aaft": {"e1_auc": 0.6477, "e0_true_auc": 0.9653, "gap": 0.3177},
    "iaaft": {"e1_auc": 0.6437, "e0_true_auc": 0.9750, "gap": 0.3313},
}


def _positives(n: int, glue, seed: int):
    return make_diluted_population(CELL["share"], n, glue, seed)


def _nulls(kind: str, glue, n: int, seed: int, source) -> dict:
    if kind == "composite":
        return make_null_populations(n, glue, seed)
    return {kind: matched_surrogate_population(source, seed, kind=kind)}


def _observations(traces) -> list:
    return [observe(trace.t, trace.P_obs) for trace in traces]


def _replicate(kind: str, seed: int, args) -> dict:
    glue = glue_for(CELL["duration_s"])
    fit_pos = _positives(args.n_fit, glue, seed + 1)
    eval_pos = _positives(args.n_eval, glue, seed + 2)
    source_fit = _positives(args.n_fit, glue, seed + 3)
    source_eval = _positives(args.n_eval, glue, seed + 4)

    n_null = max(args.n_eval // len(NULL_FAMILIES), 20) \
        if kind == "composite" else args.n_eval
    fit_nulls = _nulls(kind, glue, n_null, seed + 5, source_fit)
    eval_nulls = _nulls(kind, glue, n_null, seed + 6, source_eval)

    fit_pos_obs = _observations(fit_pos)
    eval_pos_obs = _observations(eval_pos)
    fit_null_obs = {name: _observations(group)
                    for name, group in fit_nulls.items()}
    eval_null_obs = {name: _observations(group)
                     for name, group in eval_nulls.items()}

    model = fit_hurdle_likelihood(fit_pos_obs, list(fit_null_obs.values()))

    def score(observations, counts_only):
        return np.asarray([model.score_observation(o, counts_only=counts_only)
                           for o in observations])

    # --- E0 under the same likelihood: true-aligned positives against
    # matched random-aligned nulls, so the corrected gap is comparable. -------
    counts, durations = alignment_pool(fit_pos)

    def true_aligned(traces):
        return [aligned_observation(tr.t, tr.P_obs,
                                    tr.meta.communication_starts,
                                    tr.meta.communication_durations)
                for tr in traces]

    fit_null_traces = [tr for group in fit_nulls.values() for tr in group]
    eval_null_traces = [tr for group in eval_nulls.values() for tr in group]
    e0_model = fit_hurdle_likelihood(
        true_aligned(fit_pos),
        [random_aligned_observations(fit_null_traces, counts, durations,
                                     seed + 101)],
    )
    e0_null_eval = random_aligned_observations(eval_null_traces, counts,
                                               durations, seed + 202)
    e0_pos = np.asarray([e0_model.score_observation(o)
                         for o in true_aligned(eval_pos)])
    e0_neg = np.asarray([e0_model.score_observation(o) for o in e0_null_eval])
    e0_true_auc = rank_auc(e0_pos, e0_neg)

    eval_null_flat = [o for group in eval_null_obs.values() for o in group]
    full = {
        "positive": score(eval_pos_obs, False),
        "null": score(eval_null_flat, False),
    }
    counts_only = {
        "positive": score(eval_pos_obs, True),
        "null": score(eval_null_flat, True),
    }
    return {
        "seed": seed,
        "e1_auc": rank_auc(full["positive"], full["null"]),
        "e0_true_auc": e0_true_auc,
        "e0_minus_e1": float(e0_true_auc - rank_auc(full["positive"],
                                                    full["null"])),
        "e1_auc_counts_only": rank_auc(counts_only["positive"],
                                       counts_only["null"]),
        "zero_event_rate_positive": zero_event_rate(eval_pos_obs),
        "zero_event_rate_by_null": {name: zero_event_rate(group)
                                    for name, group in eval_null_obs.items()},
        "events_per_trace_positive": float(np.median(
            [o.n_events for o in eval_pos_obs])),
        "n_positive": len(eval_pos_obs),
        "n_null": len(eval_null_flat),
    }


def _summarise(replicates: list[dict], kind: str) -> dict:
    def stats(key):
        values = np.asarray([r[key] for r in replicates], dtype=float)
        return {"mean": float(values.mean()),
                "sd": float(values.std(ddof=1)) if values.size > 1 else 0.0,
                "min": float(values.min()), "max": float(values.max())}

    families = sorted({name for r in replicates
                       for name in r["zero_event_rate_by_null"]})
    out = {
        "e1_auc": stats("e1_auc"),
        "e0_true_auc": stats("e0_true_auc"),
        "e0_minus_e1": stats("e0_minus_e1"),
        "e1_auc_counts_only": stats("e1_auc_counts_only"),
        "zero_event_rate_positive": stats("zero_event_rate_positive"),
        "events_per_trace_positive": stats("events_per_trace_positive"),
        "zero_event_rate_by_null": {
            name: float(np.mean([r["zero_event_rate_by_null"][name]
                                 for r in replicates if name in
                                 r["zero_event_rate_by_null"]]))
            for name in families
        },
    }
    counts_share = (out["e1_auc_counts_only"]["mean"] - 0.5)
    full_share = (out["e1_auc"]["mean"] - 0.5)
    out["counts_only_share_of_excess_auc"] = (
        float(counts_share / full_share) if full_share > 1e-9 else float("nan")
    )
    if kind in PREFIX_ABSOLUTE_TARGETS:
        out["prefix_absolute_target"] = PREFIX_ABSOLUTE_TARGETS[kind]
        out["prefix_baseline"] = PREFIX_BASELINE[kind]
    # Corrected normalisation, reported alongside the preserved absolute target
    # so the E2 bar cannot drift when the baseline moves.
    out["corrected_normalisation"] = {
        "e1_auc": out["e1_auc"]["mean"],
        "e0_true_auc": out["e0_true_auc"]["mean"],
        "gap": out["e0_true_auc"]["mean"] - out["e1_auc"]["mean"],
        "auc_at_R_half": (out["e1_auc"]["mean"]
                          + 0.5 * (out["e0_true_auc"]["mean"]
                                   - out["e1_auc"]["mean"])),
    }
    return out


def _audit(args) -> dict:
    """Zero-event rate per class and null family at earlier campaign settings."""
    settings = [
        ("smoke_90s", {"duration_s": 90.0, "share": None}),
        ("regime_90s_share0.2", {"duration_s": 90.0, "share": 0.2}),
        ("regime_20s", {"duration_s": 20.0, "share": None}),
        ("regime_10s", {"duration_s": 10.0, "share": None}),
        ("locked_10s_share0.2", {"duration_s": 10.0, "share": 0.2}),
        ("regime_5s", {"duration_s": 5.0, "share": None}),
        ("regime_3s", {"duration_s": 3.0, "share": None}),
    ]
    out = {}
    for name, spec in settings:
        glue = glue_for(spec["duration_s"])
        if spec["share"] is None:
            positives = make_annotated_population("drift", 0.0, args.n_audit,
                                                  glue, 4242)
        else:
            positives = make_diluted_population(spec["share"], args.n_audit,
                                                glue, 4242)
        nulls = make_null_populations(max(args.n_audit // 3, 15), glue, 8484)
        positive_rate = zero_event_rate(_observations(positives))
        null_rates = {family: zero_event_rate(_observations(group))
                      for family, group in nulls.items()}
        worst = max(null_rates.values())
        out[name] = {
            **spec,
            "zero_event_rate_positive": positive_rate,
            "zero_event_rate_by_null": null_rates,
            "worst_null_rate": worst,
            # Material if either side is non-trivial; asymmetric if the classes
            # differ enough that the old all-zero vector could separate them.
            "material": bool(max(positive_rate, worst) > 0.05),
            "asymmetric": bool(abs(worst - positive_rate) > 0.05),
        }
    return out


def run(args: argparse.Namespace) -> dict:
    t0 = time.time()
    output = pathlib.Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    seeds = [args.seed + 7919 * k for k in range(args.n_seeds)]

    by_kind: dict[str, dict] = {}
    for kind in NULL_KINDS:
        replicates = []
        for seed in seeds:
            replicates.append(_replicate(kind, seed, args))
            r = replicates[-1]
            print(f"  {kind:<10s} seed={seed:<10d} "
                  f"E1={r['e1_auc']:.3f} counts-only={r['e1_auc_counts_only']:.3f} "
                  f"zero(pos)={r['zero_event_rate_positive']:.3f} "
                  f"zero(null max)={max(r['zero_event_rate_by_null'].values()):.3f}",
                  flush=True)
        by_kind[kind] = {"replicates": replicates,
                         "summary": _summarise(replicates, kind)}

    summary = {
        "status": ("corrected E1 baseline under the hurdle likelihood; "
                   "supersedes the pre-fix confirmation's E1 constants"),
        "cell": CELL,
        "sizes": {"fit": args.n_fit, "evaluation": args.n_eval,
                  "seeds": args.n_seeds},
        "seeds": seeds,
        "prefix_absolute_targets": PREFIX_ABSOLUTE_TARGETS,
        "prefix_baseline": PREFIX_BASELINE,
        "by_null_kind": by_kind,
        "elapsed_s": time.time() - t0,
    }
    if args.audit:
        summary["earlier_campaign_audit"] = _audit(args)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    _plot(by_kind, output)
    return summary


def _plot(by_kind: dict, output: pathlib.Path) -> None:
    kinds = list(by_kind)
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.6))
    for j, kind in enumerate(kinds):
        replicates = by_kind[kind]["replicates"]
        axes[0].plot(np.full(len(replicates), j),
                     [r["e1_auc"] for r in replicates], "o", ms=5, color=f"C{j}",
                     alpha=0.75)
        axes[0].plot(np.full(len(replicates), j + 0.25),
                     [r["e1_auc_counts_only"] for r in replicates], "s", ms=4,
                     color=f"C{j}", alpha=0.45)
        rates = by_kind[kind]["summary"]["zero_event_rate_by_null"]
        axes[1].plot(np.full(len(rates), j), list(rates.values()), "o", ms=5,
                     color=f"C{j}", alpha=0.75)
        axes[1].plot(j, by_kind[kind]["summary"]
                     ["zero_event_rate_positive"]["mean"], "*", ms=13,
                     color=f"C{j}")
    axes[0].set_xticks(range(len(kinds)), kinds)
    axes[0].set_ylabel("E1 AUC  (circles full, squares counts-only)")
    axes[0].axhline(0.5, color="0.7", lw=0.8)
    axes[1].set_xticks(range(len(kinds)), kinds)
    axes[1].set_ylabel("zero-event rate (star = positives)")
    fig.suptitle("Hurdle likelihood — locked cell, all null families retained",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(output / "hurdle_refreeze.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-seeds", type=int, default=8)
    parser.add_argument("--n-fit", type=int, default=150)
    parser.add_argument("--n-eval", type=int, default=150)
    parser.add_argument("--n-audit", type=int, default=90)
    parser.add_argument("--seed", type=int, default=20260808)
    parser.add_argument("--output-dir", default=str(_DEFAULT_OUTPUT))
    parser.add_argument("--audit", action="store_true",
                        help="also audit earlier campaign cells by zero-event rate")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if args.quick:
        args.n_seeds, args.n_fit, args.n_eval, args.n_audit = 3, 40, 40, 30
        args.output_dir = str(pathlib.Path(args.output_dir) / "quick")

    summary = run(args)
    for kind, block in summary["by_null_kind"].items():
        s = block["summary"]
        print(f"\n{kind}: E1={s['e1_auc']['mean']:.3f}+-{s['e1_auc']['sd']:.3f}  "
              f"counts-only={s['e1_auc_counts_only']['mean']:.3f}  "
              f"(counts carry {s['counts_only_share_of_excess_auc']:.0%} of excess AUC)  "
              f"zero(pos)={s['zero_event_rate_positive']['mean']:.3f}")
        c = s["corrected_normalisation"]
        print(f"   E0={s['e0_true_auc']['mean']:.3f}  corrected gap={c['gap']:+.3f}  "
              f"AUC at R=0.5: corrected {c['auc_at_R_half']:.4f}"
              + (f" | preserved target {s['prefix_absolute_target']:.4f}"
                 if 'prefix_absolute_target' in s else ""))
        print(f"   zero-event rate by null: "
              + ", ".join(f"{k}={v:.3f}"
                          for k, v in s["zero_event_rate_by_null"].items()))
    print(f"\nelapsed {summary['elapsed_s']:.1f}s -> {args.output_dir}")


if __name__ == "__main__":
    main()
