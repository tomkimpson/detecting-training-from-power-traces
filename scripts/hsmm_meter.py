#!/usr/bin/env python3
"""Stage 2: the observation-channel boundary for the E1 renewal detector.

Sweeps meter sampling rate against integration window and asks, at every cell,
which of three things has happened (``notes/plans/hsmm-e2-preconditions-plan.md``):

| Outcome | Interpretation |
|---|---|
| frozen fails, refit succeeds | domain/calibration shift; information remains |
| frozen and refit fail, oracle succeeds | blind estimation gap -- an E2 target |
| frozen, refit and oracle all fail | the meter erased the information |
| everything succeeds | no E2 headroom here |

Two regimes are therefore run at every cell:

1. **frozen-profile transport** -- profiles fitted on the pristine reference
   channel and applied unchanged;
2. **per-cell refit** -- profiles refitted on that cell's own fitting split.

Both recalibrate the decision threshold on the cell's own calibration nulls, so
every reported TPR sits at composite-null FAR 0.05 and is comparable across
cells.  The pristine threshold is *also* transported without recalibration and
reported separately, to size the calibration drift by itself.

**E0 is recomputed at every cell**, refitted per cell: a surviving oracle is what
separates an estimation problem from an information problem, and "E1 broke" alone
never justifies an HSMM.

**Channel symmetry** is enforced by construction: every positive and every null
passes through the identical ``MeterParams`` (see
``powerladder.typeb.renewal_campaign``).

Note on grid resolution: the device is simulated on the existing 20 Hz Type-B
grid, matching the frozen smoke, so integration windows are quantised to
multiples of 50 ms and 20 Hz sampling is a no-op decimation.  Continuity with the
frozen ST1/ST2 null parameterisation was preferred over a finer device grid.

Outputs land in ``results/hsmm_meter/`` and never touch frozen ST1/ST2 artefacts.
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

from powerladder.config import MeterParams  # noqa: E402
from powerladder.typeb.renewal import (  # noqa: E402
    FEATURE_NAMES,
    fit_likelihood_from_features,
)
from powerladder.typeb.renewal_campaign import (  # noqa: E402
    CONDITIONS,
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
    timing_metrics,
)


_ROOT = pathlib.Path(__file__).resolve().parent.parent
_DEFAULT_OUTPUT = _ROOT / "results" / "hsmm_meter"
_FAR = 0.05

_CONDITION_NAMES = tuple(name for name, _, _ in CONDITIONS)
_SAMPLE_HZ = (20.0, 10.0, 5.0, 2.0)
_INTEG_S = (0.0, 0.1, 0.25, 0.5)
#: ``spec.md`` names this an expected information-erasure point, not a required
#: success.  It is swept so the erasure is measured rather than assumed.
_EXTRA_CELLS = ((1.0, 1.0),)
_REFERENCE_CELL = (20.0, 0.0)

_TPR_DECLINE = 0.10
_AUC_DECLINE = 0.05
_ORACLE_FLOOR = 0.90
#: Conditions the per-cell event-timing diagnostic is run on (the anchor and
#: the primary event-model target); running it on all five doubles extraction.
_TIMING_CONDITIONS = ("honest", "work_0.7")
_GAP_MIN = 0.10


def _cell_name(sample_hz: float, integ_window_s: float) -> str:
    return f"{sample_hz:g}Hz_{integ_window_s:g}s"


def _grid() -> list[tuple[float, float]]:
    cells = [(fs, w) for fs in _SAMPLE_HZ for w in _INTEG_S]
    return cells + [cell for cell in _EXTRA_CELLS if cell not in cells]


def _meter(sample_hz: float, integ_window_s: float, sigma_eta: float) -> MeterParams:
    return MeterParams(integ_window_s=integ_window_s, sample_hz=sample_hz,
                       sigma_eta=sigma_eta)


def _fit_model(train, null_features, feature_set):
    transform = FEATURE_SETS[feature_set]
    return fit_likelihood_from_features(
        transform(train),
        [transform(null_features[name]) for name in NULL_FAMILIES],
    )


def _evaluate(model, feature_set, eval_pos, eval_null, cal_null,
              frozen_threshold: float | None = None) -> dict:
    transform = FEATURE_SETS[feature_set]
    cal_scores = {name: profile_scores(model, transform(matrix))
                  for name, matrix in cal_null.items()}
    threshold = composite_threshold(cal_scores, _FAR)
    eval_scores = {name: profile_scores(model, transform(matrix))
                   for name, matrix in eval_null.items()}
    out = {
        "threshold": threshold,
        "evaluation_fpr": null_rates(eval_scores, threshold),
        "conditions": {
            name: metrics(profile_scores(model, transform(matrix)),
                          eval_scores, threshold)
            for name, matrix in eval_pos.items()
        },
    }
    if frozen_threshold is not None:
        # The pristine threshold carried over without recalibration: this is the
        # calibration drift on its own, separated from profile transport.
        out["transported_threshold"] = frozen_threshold
        out["transported_threshold_fpr"] = null_rates(eval_scores,
                                                      frozen_threshold)
        out["transported_threshold_tpr"] = {
            name: float((profile_scores(model, transform(matrix))
                         >= frozen_threshold).mean())
            for name, matrix in eval_pos.items()
        }
    return out


class _Cell:
    """Featurised populations for one observation channel."""

    def __init__(self, sample_hz, integ_window_s, glue, streams, args):
        meter = _meter(sample_hz, integ_window_s, glue.sigma_eta)
        self.sample_hz = sample_hz
        self.integ_window_s = integ_window_s

        self.fit_pos, self.fit_oracle = {}, {}
        for j, (name, family, level) in enumerate(CONDITIONS):
            traces = make_annotated_population(
                family, level, args.n_fit, glue, streams["fit"] + j, meter=meter,
            )
            self.fit_pos[name] = blind_matrix(traces)
            self.fit_oracle[name] = oracle_matrix(traces)
        self.fit_null = {
            name: blind_matrix(group) for name, group in
            make_null_populations(args.n_fit_null, glue, streams["fit"] + 100,
                                  meter=meter).items()
        }
        self.cal_null = {
            name: blind_matrix(group) for name, group in
            make_null_populations(args.n_cal, glue,
                                  streams["calibration"] + 100,
                                  meter=meter).items()
        }
        self.eval_null = {
            name: blind_matrix(group) for name, group in
            make_null_populations(args.n_eval, glue, streams["evaluation"] + 100,
                                  meter=meter).items()
        }
        self.eval_pos, self.eval_oracle = {}, {}
        self.timing: dict[str, dict] = {}
        for j, (name, family, level) in enumerate(CONDITIONS):
            traces = make_annotated_population(
                family, level, args.n_eval, glue, streams["evaluation"] + j,
                meter=meter,
            )
            self.eval_pos[name] = blind_matrix(traces)
            self.eval_oracle[name] = oracle_matrix(traces)
            if name in _TIMING_CONDITIONS:
                # Does the score still rest on *resolved* events at this
                # channel?  A high AUC with collapsed recall means the statistic
                # has stopped being duration-aware and is separating the
                # populations on residual summary structure instead.
                self.timing[name] = {
                    "fixed_100ms": timing_metrics(traces, tolerance_s=0.1),
                    "one_sample_period": timing_metrics(
                        traces, tolerance_s=max(0.1, 1.0 / sample_hz),
                    ),
                }

    @property
    def train(self) -> np.ndarray:
        return np.concatenate(list(self.fit_pos.values()), axis=0)

    @property
    def oracle_train(self) -> np.ndarray:
        return np.concatenate(list(self.fit_oracle.values()), axis=0)


def run(args: argparse.Namespace) -> dict:
    t0 = time.time()
    glue = glue_for(args.duration_s)
    output = pathlib.Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    streams = {"fit": args.seed + 10_000, "calibration": args.seed + 20_000,
               "evaluation": args.seed + 30_000}

    reference = _Cell(*_REFERENCE_CELL, glue, streams, args)
    frozen_models = {feature_set: _fit_model(reference.train, reference.fit_null,
                                             feature_set)
                     for feature_set in FEATURE_SETS}
    frozen_thresholds = {
        feature_set: composite_threshold(
            {name: profile_scores(model, FEATURE_SETS[feature_set](matrix))
             for name, matrix in reference.cal_null.items()}, _FAR,
        )
        for feature_set, model in frozen_models.items()
    }

    cells: dict[str, dict] = {}
    for sample_hz, integ_window_s in _grid():
        name = _cell_name(sample_hz, integ_window_s)
        cell = (reference if (sample_hz, integ_window_s) == _REFERENCE_CELL
                else _Cell(sample_hz, integ_window_s, glue, streams, args))
        block: dict = {
            "sample_hz": sample_hz,
            "integ_window_s": integ_window_s,
            "is_reference": (sample_hz, integ_window_s) == _REFERENCE_CELL,
            "n_traces": int(cell.eval_pos["honest"].shape[0]),
            "timing": cell.timing,
            "frozen": {},
            "refit": {},
        }
        for feature_set in FEATURE_SETS:
            block["frozen"][feature_set] = _evaluate(
                frozen_models[feature_set], feature_set, cell.eval_pos,
                cell.eval_null, cell.cal_null,
                frozen_threshold=frozen_thresholds[feature_set],
            )
            block["refit"][feature_set] = _evaluate(
                _fit_model(cell.train, cell.fit_null, feature_set),
                feature_set, cell.eval_pos, cell.eval_null, cell.cal_null,
            )
        # E0 is refitted per cell: the ceiling is what this channel carries, not
        # what survives transport from another channel.
        block["oracle"] = _evaluate(
            _fit_model(cell.oracle_train, cell.fit_null, "full"), "full",
            cell.eval_oracle, cell.eval_null, cell.cal_null,
        )
        cells[name] = block
        print(f"  {name:>12s}  "
              f"refit full AUC={block['refit']['full']['conditions']['work_0.7']['auc_union']:.3f}"
              f"  oracle AUC={block['oracle']['conditions']['work_0.7']['auc_union']:.3f}",
              flush=True)

    summary = {
        "status": "stage-2 observation-channel sweep; not a frozen paper result",
        "method": ("blind marked-renewal E1 under frozen-profile transport and "
                   "per-cell refit, with E0 recomputed per cell; no HSMM E2"),
        "duration_s": args.duration_s,
        "device_fs": glue.fs,
        "reference_cell": _cell_name(*_REFERENCE_CELL),
        "sizes": {"fit_per_condition": args.n_fit,
                  "fit_per_null_family": args.n_fit_null,
                  "calibration": args.n_cal, "evaluation": args.n_eval},
        "seed": args.seed,
        "target_far": _FAR,
        "feature_names": list(FEATURE_NAMES),
        "null_families": list(NULL_FAMILIES),
        "declared_erasure_cells": [_cell_name(*cell) for cell in _EXTRA_CELLS],
        "cells": cells,
    }
    summary["e2_candidates"] = _e2_candidates(cells)
    summary["elapsed_s"] = time.time() - t0
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    _plot_grid(cells, output)
    return summary


def _e2_candidates(cells: dict[str, dict]) -> dict:
    """Cells with a genuine estimation gap: oracle survives, blind E1 does not."""
    reference = cells[_cell_name(*_REFERENCE_CELL)]
    declared_erasure = {_cell_name(*cell) for cell in _EXTRA_CELLS}
    candidates, erasure, transport_only = [], [], []
    for name, block in cells.items():
        if block["is_reference"]:
            continue
        for feature_set in FEATURE_SETS:
            for condition in _CONDITION_NAMES:
                oracle = block["oracle"]["conditions"][condition]
                refit = block["refit"][feature_set]["conditions"][condition]
                frozen = block["frozen"][feature_set]["conditions"][condition]
                base = reference["refit"][feature_set]["conditions"][condition]
                declined = (
                    base["tpr_at_composite_far"]
                    - refit["tpr_at_composite_far"] >= _TPR_DECLINE
                    or base["auc_union"] - refit["auc_union"] >= _AUC_DECLINE
                )
                gap_auc = oracle["auc_union"] - refit["auc_union"]
                gap_tpr = (oracle["tpr_at_composite_far"]
                           - refit["tpr_at_composite_far"])
                record = {
                    "cell": name, "feature_set": feature_set,
                    "condition": condition,
                    "oracle_auc": oracle["auc_union"],
                    "refit_auc": refit["auc_union"],
                    "frozen_auc": frozen["auc_union"],
                    "refit_tpr": refit["tpr_at_composite_far"],
                    "frozen_tpr": frozen["tpr_at_composite_far"],
                    "oracle_tpr": oracle["tpr_at_composite_far"],
                    "auc_gap": float(gap_auc), "tpr_gap": float(gap_tpr),
                }
                if not declined:
                    if (frozen["tpr_at_composite_far"]
                            < refit["tpr_at_composite_far"] - _TPR_DECLINE):
                        transport_only.append(record)
                    continue
                if oracle["auc_union"] < _ORACLE_FLOOR:
                    erasure.append(record)
                elif (max(gap_auc, gap_tpr) >= _GAP_MIN
                      and name not in declared_erasure):
                    candidates.append(record)
    candidates.sort(key=lambda r: -max(r["auc_gap"], r["tpr_gap"]))
    return {
        "criterion": {
            "oracle_auc_floor": _ORACLE_FLOOR,
            "min_oracle_minus_e1_gap": _GAP_MIN,
            "tpr_decline_margin": _TPR_DECLINE,
            "auc_decline_margin": _AUC_DECLINE,
            "excluded_cells": sorted(declared_erasure),
        },
        "n_candidates": len(candidates),
        "candidates": candidates[:40],
        "n_information_erasure": len(erasure),
        "information_erasure": erasure[:40],
        "n_transport_only": len(transport_only),
        "transport_only": transport_only[:20],
        "build_e2": bool(candidates),
    }


def _plot_grid(cells: dict[str, dict], output: pathlib.Path) -> None:
    """Oracle / refit / frozen AUC on work_0.7 over the sampling x integration grid."""
    panels = [
        ("oracle (E0)", lambda b: b["oracle"]["conditions"]["work_0.7"]["auc_union"]),
        ("refit (E1)",
         lambda b: b["refit"]["full"]["conditions"]["work_0.7"]["auc_union"]),
        ("frozen (E1)",
         lambda b: b["frozen"]["full"]["conditions"]["work_0.7"]["auc_union"]),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(11.0, 3.2))
    for ax, (title, getter) in zip(axes, panels):
        grid = np.full((len(_SAMPLE_HZ), len(_INTEG_S)), np.nan)
        for r, fs in enumerate(_SAMPLE_HZ):
            for c, w in enumerate(_INTEG_S):
                name = _cell_name(fs, w)
                if name in cells:
                    grid[r, c] = getter(cells[name])
        im = ax.imshow(grid, vmin=0.5, vmax=1.0, cmap="viridis")
        ax.set_xticks(range(len(_INTEG_S)), [f"{w:g}" for w in _INTEG_S])
        ax.set_yticks(range(len(_SAMPLE_HZ)), [f"{fs:g}" for fs in _SAMPLE_HZ])
        ax.set_xlabel("integration window [s]")
        ax.set_title(title, fontsize=9)
        for r in range(grid.shape[0]):
            for c in range(grid.shape[1]):
                if np.isfinite(grid[r, c]):
                    ax.text(c, r, f"{grid[r, c]:.2f}", ha="center", va="center",
                            fontsize=7,
                            color="w" if grid[r, c] < 0.85 else "k")
    axes[0].set_ylabel("sample rate [Hz]")
    fig.colorbar(im, ax=axes, shrink=0.85, label="AUC on work_0.7")
    fig.savefig(output / "meter_grid.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration-s", type=float, default=90.0)
    parser.add_argument("--n-fit", type=int, default=50)
    parser.add_argument("--n-fit-null", type=int, default=50)
    parser.add_argument("--n-cal", type=int, default=200)
    parser.add_argument("--n-eval", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--output-dir", default=str(_DEFAULT_OUTPUT))
    parser.add_argument("--quick", action="store_true",
                        help="plumbing run: 10 fit / 20 calibration / 20 eval")
    args = parser.parse_args()
    if args.quick:
        args.n_fit, args.n_fit_null, args.n_cal, args.n_eval = 10, 10, 20, 20
        args.duration_s = min(args.duration_s, 60.0)
        args.output_dir = str(pathlib.Path(args.output_dir) / "quick")

    summary = run(args)
    print(json.dumps({k: v for k, v in summary["e2_candidates"].items()
                      if not isinstance(v, list)}, indent=2))
    print(f"elapsed {summary['elapsed_s']:.1f}s -> {args.output_dir}")


if __name__ == "__main__":
    main()
