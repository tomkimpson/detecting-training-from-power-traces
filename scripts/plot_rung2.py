"""Plot the Rung-2 training-vs-inference classification sweep (Phase 2, plan §3.3).

Reads results/rung2/rung2_summary.json and renders:

  figures/rung2_stated_transfer.{pdf,png}   AUC per decision rule across the
      stated population and each transfer / domain shift — how well each rule
      separates training from the inference null, and how it degrades off the
      stated population.
  figures/rung2_controls.{pdf,png}          per-rule fraction scored AS training
      for each semantic falsification control, at the primary operating FAR.
      The central-question figure: bars high where a NON-training load scores as
      training (the meter certifies physics, not semantics) and low where
      genuine training is missed (async / scoped-out). x-labels are annotated
      with the semantic ground truth; markers show the pre-registered
      expectation.

Compute-free: run scripts/rung2_eval.py first.

Usage:
    python scripts/plot_rung2.py
    python scripts/plot_rung2.py --summary results/rung2/rung2_smoke_summary.json
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

_ROOT = pathlib.Path(__file__).resolve().parent.parent
_SUMMARY = _ROOT / "results" / "rung2" / "rung2_summary.json"

_RULE_STYLE = {
    "physics_score": (C["blue"], "physics score (prespecified)"),
    "fitted_discriminant": (C["orange"], "fitted discriminant"),
    "learned_reference": (C["green"], "RF learned reference"),
}


def _load(path: pathlib.Path) -> dict:
    if not path.exists():
        raise SystemExit(f"{path} not found — run scripts/rung2_eval.py first "
                         "(or pass --summary <smoke summary>)")
    return json.loads(path.read_text())


def _far_key(summary: dict) -> str:
    return f"{summary['target_fars'][0]:g}"


def _cells_by_kind(summary: dict, kind: str) -> dict:
    return {c["condition"]: c for c in summary["cells"] if c["kind"] == kind}


def _grouped_bars(ax, groups, rules, value_of, *, ylabel, title):
    """Grouped bar chart: one group per condition, one bar per rule."""
    n = len(rules)
    width = 0.8 / n
    x = np.arange(len(groups))
    for k, rule in enumerate(rules):
        color, label = _RULE_STYLE[rule]
        vals = [value_of(g, rule) for g in groups]
        ax.bar(x + (k - (n - 1) / 2) * width, vals, width,
               color=color, label=label)
    ax.set_xticks(x)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0.0, 1.02)
    ax.set_title(title)
    return x


def plot_stated_transfer(summary: dict) -> None:
    rules = summary["rules"]
    stated = _cells_by_kind(summary, "stated")
    transfer = _cells_by_kind(summary, "transfer")
    order = (["stated"] if "stated" in stated else []) + \
        [n for n in summary["transfer_shifts"] if n in transfer]
    cells = {**stated, **transfer}

    fig, ax = plt.subplots(figsize=(WIDTH_WIDE, 3.1))
    x = _grouped_bars(
        ax, order, rules,
        lambda g, r: cells[g]["rules"][r]["auc"],
        ylabel="AUC (train vs inference null)",
        title="Rung 2: stated population and transfer")
    ax.axhline(0.5, color=C["grey"], lw=0.8, ls=":", zorder=0)
    ax.set_xticklabels([g.replace("transfer__", "").replace("_", "\n", 1)
                        if g != "stated" else "stated" for g in order],
                       rotation=0, fontsize=6)
    ax.legend(loc="lower left", fontsize=6, ncol=1, framealpha=0.9)
    save(fig, "rung2_stated_transfer")
    plt.close(fig)


def plot_controls(summary: dict) -> None:
    rules = summary["rules"]
    far = _far_key(summary)
    controls = _cells_by_kind(summary, "control")
    order = [n for n in summary["controls"] if n in controls]

    fig, ax = plt.subplots(figsize=(WIDTH_WIDE, 3.6))
    fig.subplots_adjust(bottom=0.30, top=0.84)
    x = _grouped_bars(
        ax, order, rules,
        lambda g, r: controls[g]["rules"][r]["frac_training"][far],
        ylabel=f"fraction scored as training (FAR={far})",
        title="")
    ax.set_ylim(0.0, 1.10)

    # expectation marker just above each group (below the title band).
    for xi, name in zip(x, order):
        c = controls[name]
        ax.annotate("▲" if c["expect"] == "train" else "▽",
                    (xi, 1.02), ha="center", va="bottom", fontsize=7,
                    color=C["black"], annotation_clip=False)
    ax.set_xticklabels(
        [f"{n.replace('_', ' ')}\n({'training' if controls[n]['is_training'] else 'NOT training'})"
         for n in order], rotation=30, ha="right", fontsize=5.5)
    for lbl, name in zip(ax.get_xticklabels(), order):
        lbl.set_color(C["vermillion"] if controls[name]["is_training"] else C["black"])
    ax.legend(loc="center right", fontsize=6, framealpha=0.9)
    fig.suptitle("Rung 2 semantic falsification controls", y=0.97, fontsize=9)
    fig.text(0.5, 0.90,
             "▲ expected to score as training   ▽ expected not   ·   "
             "red label = genuine training",
             ha="center", fontsize=5.5, color=C["grey"])
    save(fig, "rung2_controls")
    plt.close(fig)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--summary", type=pathlib.Path, default=_SUMMARY,
                    help="path to the rung2 summary JSON")
    args = ap.parse_args(argv)

    apply_house_style()
    summary = _load(args.summary)
    plot_stated_transfer(summary)
    plot_controls(summary)
    print(f"wrote figures/rung2_stated_transfer.* and figures/rung2_controls.* "
          f"from {args.summary}")


if __name__ == "__main__":
    main()
