#!/usr/bin/env python3
"""Locked confirmation of the single selected stage-3 cell.

**This is confirmation, not regime search.**  Exactly one operating point is
measured -- 10 s records, training at a 20% share of a Ko eq-11 aggregate,
against surrogate nulls -- selected in
``notes/results/hsmm-regime-search-findings.md``.  No axis is swept and no
alternative cell is considered here; adding one would reintroduce the selection
problem this run exists to control for.

Two things the stage-3 interval could not do, and this run does:

1. **Fresh seeds.**  The stage-3 bootstrap resampled the evaluation draw while
   holding the fitted profiles fixed, so it excluded fitting-set variability.
   Each replicate here redraws the fitting *and* evaluation populations, so the
   spread across replicates is the honest one.  It also answers the pointwise
   -after-searching-40-cells problem: a quantity that survives independent
   redraws was not a selection artefact.
2. **Both surrogate constructions.**  One-pass AAFT restores the marginal exactly
   but perturbs the spectrum it has just imposed; IAAFT iterates the two
   constraints to convergence.  The measured spectral residual of each is
   reported, so the null can be described accurately rather than assumed to be
   matched on both.

Outputs land in ``results/hsmm_confirm_cell/``.
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
    glue_for,
    make_diluted_population,
    make_null_populations,
    matched_surrogate_population,
    surrogate_spectral_error,
)
from powerladder.typeb.renewal_regime import (  # noqa: E402
    ADMISSION,
    admits_e2,
    measure_point,
)


_ROOT = pathlib.Path(__file__).resolve().parent.parent
_DEFAULT_OUTPUT = _ROOT / "results" / "hsmm_confirm_cell"

#: The selected cell.  Frozen -- do not parameterise.
CELL = {"duration_s": 10.0, "share": 0.2, "f_peak_frac": None}

#: Surrogate constructions the cell is confirmed under.  ``composite`` is the
#: physically-motivated null, reported alongside so the AAFT-only result is never
#: mistaken for an operational training-detection claim.
NULL_KINDS = ("aaft", "iaaft", "composite")

_METRICS = ("e1_auc", "e0_true_auc", "e0_random_auc", "align_plus",
            "e0_minus_e1", "event_recall", "events_per_trace",
            "degenerate_fraction_positive", "degenerate_fraction_null")


def _replicate(kind: str, seed: int, args) -> dict:
    glue = glue_for(CELL["duration_s"], f_peak_frac=CELL["f_peak_frac"])
    fit_pos = make_diluted_population(CELL["share"], args.n_fit, glue, seed + 1)
    eval_pos = make_diluted_population(CELL["share"], args.n_eval, glue, seed + 2)

    if kind == "composite":
        n_null = max(args.n_eval // 6, 20)
        fit_nulls = make_null_populations(n_null, glue, seed + 5)
        eval_nulls = make_null_populations(n_null, glue, seed + 6)
        spectral_error = None
    else:
        # Surrogates are built from independent positive draws, so a positive is
        # never scored against a surrogate of itself.
        source_fit = make_diluted_population(CELL["share"], args.n_fit, glue,
                                             seed + 3)
        source_eval = make_diluted_population(CELL["share"], args.n_eval, glue,
                                              seed + 4)
        fit_nulls = {kind: matched_surrogate_population(source_fit, seed + 5,
                                                        kind=kind)}
        eval_nulls = {kind: matched_surrogate_population(source_eval, seed + 6,
                                                         kind=kind)}
        spectral_error = float(np.median([
            surrogate_spectral_error(original.P_obs, surrogate.P_obs)
            for original, surrogate in zip(source_eval, eval_nulls[kind])
        ]))

    result = measure_point(fit_pos, eval_pos, fit_nulls, eval_nulls, seed=seed,
                           n_random=args.n_random, n_boot=args.n_boot)
    return {
        "seed": seed,
        "surrogate_spectral_error": spectral_error,
        **result.as_dict(),
        "verdict": admits_e2(result),
    }


def _across_seeds(replicates: list[dict]) -> dict:
    """Mean and spread of each metric across independent redraws."""
    out = {}
    for metric in _METRICS:
        values = np.asarray([r[metric] for r in replicates], dtype=float)
        out[metric] = {
            "mean": float(values.mean()),
            "sd": float(values.std(ddof=1)) if values.size > 1 else 0.0,
            "min": float(values.min()),
            "max": float(values.max()),
        }
    admitted = [r["verdict"]["admitted"] for r in replicates]
    clause_names = [k for k in replicates[0]["verdict"]
                    if k not in {"admitted", "diagnostic_e0_random_auc"}]
    return {
        "metrics": out,
        "n_replicates": len(replicates),
        "n_admitted": int(sum(admitted)),
        "admitted_in_every_replicate": all(admitted),
        "clause_pass_counts": {
            name: int(sum(r["verdict"][name] for r in replicates))
            for name in clause_names
        },
    }


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
            print(f"  {kind:<10s} seed={seed:<10d} E1={r['e1_auc']:.3f} "
                  f"[{r['e1_auc_lo']:.3f},{r['e1_auc_hi']:.3f}] "
                  f"E0={r['e0_true_auc']:.3f} align+={r['align_plus']:+.3f} "
                  f"gap={r['e0_minus_e1']:+.3f} "
                  f"[{r['e0_minus_e1_lo']:+.3f},{r['e0_minus_e1_hi']:+.3f}]"
                  f"{'  ADMITTED' if r['verdict']['admitted'] else ''}",
                  flush=True)
        by_kind[kind] = {"replicates": replicates,
                         "summary": _across_seeds(replicates)}

    aaft = by_kind["aaft"]["summary"]["metrics"]
    normalisation = {
        "note": ("fixed E2 normalisation: R = (AUC_E2 - E1) / (E0_true - E1), "
                 "'benchmark-gap recovery'. E0 is specific to this feature "
                 "family, so R may legitimately exceed 1."),
        "e1_auc": aaft["e1_auc"]["mean"],
        "e0_true_auc": aaft["e0_true_auc"]["mean"],
        "gap": aaft["e0_true_auc"]["mean"] - aaft["e1_auc"]["mean"],
        "prototype_success_threshold_R": 0.50,
    }

    summary = {
        "status": "locked confirmation of one selected cell; not a paper result",
        "cell": CELL,
        "null_kinds": list(NULL_KINDS),
        "admission_rule": ADMISSION,
        "sizes": {"fit": args.n_fit, "evaluation": args.n_eval,
                  "random_alignments": args.n_random, "bootstrap": args.n_boot,
                  "seeds": args.n_seeds},
        "seeds": seeds,
        "by_null_kind": by_kind,
        "e2_normalisation": normalisation,
        "elapsed_s": time.time() - t0,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    _plot(by_kind, output)
    return summary


def _plot(by_kind: dict, output: pathlib.Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))
    kinds = list(by_kind)
    x = np.arange(len(kinds))
    for ax, (metric, label) in zip(axes, [
        ("e0_minus_e1", "estimation gap, E0 - E1"),
        ("align_plus", "alignment benefit, align+"),
    ]):
        for j, kind in enumerate(kinds):
            values = [r[metric] for r in by_kind[kind]["replicates"]]
            ax.plot(np.full(len(values), j), values, "o", ms=5, alpha=0.7,
                    color=f"C{j}")
            ax.plot([j - 0.2, j + 0.2], [np.mean(values)] * 2, "-",
                    color=f"C{j}", lw=2)
        ax.axhline(0.10, color="0.4", ls="--", lw=0.9, label="admission 0.10")
        ax.axhline(0.0, color="0.7", lw=0.8)
        ax.set_xticks(x, kinds)
        ax.set_ylabel(label)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Confirmation across fresh seeds — T10, share 0.2", fontsize=10)
    fig.tight_layout()
    fig.savefig(output / "confirmation.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-seeds", type=int, default=8)
    parser.add_argument("--n-fit", type=int, default=150)
    parser.add_argument("--n-eval", type=int, default=150)
    parser.add_argument("--n-random", type=int, default=8)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260808)
    parser.add_argument("--output-dir", default=str(_DEFAULT_OUTPUT))
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if args.quick:
        args.n_seeds, args.n_fit, args.n_eval = 3, 40, 40
        args.n_random, args.n_boot = 3, 200
        args.output_dir = str(pathlib.Path(args.output_dir) / "quick")

    summary = run(args)
    for kind, block in summary["by_null_kind"].items():
        s = block["summary"]
        m = s["metrics"]
        print(f"\n{kind}: admitted {s['n_admitted']}/{s['n_replicates']}  "
              f"E1={m['e1_auc']['mean']:.3f}+-{m['e1_auc']['sd']:.3f}  "
              f"E0={m['e0_true_auc']['mean']:.3f}+-{m['e0_true_auc']['sd']:.3f}  "
              f"gap={m['e0_minus_e1']['mean']:+.3f}+-{m['e0_minus_e1']['sd']:.3f}")
    print(f"\nelapsed {summary['elapsed_s']:.1f}s -> {args.output_dir}")


if __name__ == "__main__":
    main()
