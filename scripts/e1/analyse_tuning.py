"""Analyse E1b (exploratory): can a tuned attacker avoid the learning cost?

The honest arm and the lognormal s = 1.0 attacker are each run under the same
recipe changes (learning rate, clipping off, beta2), tagged in the run
directory name. The pilot's recipe is config "base" (untagged directories).
For every attacker config this reports the honest-equivalent cost
(``analyse.py``'s equiv_cost) against:

    base honest   the pilot's honest recipe
    best honest   the honest recipe with the lowest seed-0 final loss
    same recipe   the honest run with the attacker's own recipe and seed

The attacker's tuned cost is the minimum over its configs, against the best
honest recipe. Where gradient norms were logged it also reports, per range of G,
the median pre-clip norm and the fraction of steps clipped at 1.0.

    python scripts/e1/analyse_tuning.py   -> results/e1/tuning_summary.json
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from analyse import _read, tokens_to_target

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results" / "e1"
M = 16
ATTACKER = "lognorm1.0"
G_BINS = ((1, 4), (5, 12), (13, 40), (41, 160))
CLIP = 1.0


def _runs(arm: str) -> dict[tuple[str, int], Path]:
    """(config, seed) -> run dir. Untagged = "base"; skips smoke/bnoise runs."""
    pat = re.compile(rf"^{re.escape(arm)}_m{M}_s(\d+)(?:_(.+))?$")
    out = {}
    for d in RESULTS.iterdir():
        mt = pat.match(d.name)
        if not mt or not (d / "log.jsonl").exists():
            continue
        tag = mt.group(2) or "base"
        if tag in ("smoke", "bnoise") or tag.endswith("_smoke"):
            continue
        out[(tag, int(mt.group(1)))] = d
    return out


def _equiv(honest: dict, arm: dict) -> float | None:
    t = tokens_to_target(honest["tokens"], honest["loss"], float(arm["loss"][-1]))
    return float(arm["tokens"][-1] / t - 1) if t else None


def _mechanism(path: Path) -> dict | None:
    steps = [json.loads(line) for line in path.open()]
    steps = [r for r in steps if r["kind"] == "step" and "gnorm" in r]
    if not steps:
        return None
    G = np.array([r["G"] for r in steps])
    gn = np.array([r["gnorm"] for r in steps])
    out = {"all": dict(n=int(G.size), median_gnorm=float(np.median(gn)),
                       frac_clipped=float((gn > CLIP).mean()))}
    for lo, hi in G_BINS:
        sel = (G >= lo) & (G <= hi)
        if sel.any():
            out[f"G{lo}-{hi}"] = dict(n=int(sel.sum()), median_gnorm=float(np.median(gn[sel])),
                                      frac_clipped=float((gn[sel] > CLIP).mean()))
    return out


def main() -> None:
    hon = {k: _read(p / "log.jsonl") for k, p in _runs("honest").items()}
    att = {k: _read(p / "log.jsonl") for k, p in _runs(ATTACKER).items()}
    att_paths = _runs(ATTACKER)
    hon_paths = _runs("honest")

    def finals(runs):
        cfgs = sorted({c for c, _ in runs})
        return {c: {s: float(runs[(c, s)]["loss"][-1]) for (cc, s) in runs if cc == c}
                for c in cfgs}

    hon_final, att_final = finals(hon), finals(att)
    best_honest = min((c for c in hon_final if 0 in hon_final[c]),
                      key=lambda c: hon_final[c][0])

    def ref(cfg: str, seed: int) -> dict:
        return hon.get((cfg, seed)) or hon[(cfg, 0)]

    costs = {}
    for (c, s), r in sorted(att.items()):
        costs.setdefault(c, {})[s] = dict(
            vs_base_honest=_equiv(ref("base", s), r),
            vs_best_honest=_equiv(ref(best_honest, s), r),
            vs_same_recipe=_equiv(hon[(c, s)], r) if (c, s) in hon else None,
            same_recipe_gap=(float(r["loss"][-1] - hon[(c, s)]["loss"][-1])
                             if (c, s) in hon else None),
            final_loss=float(r["loss"][-1]))

    def mean_cost(c, key):
        v = [x[key] for x in costs[c].values() if x[key] is not None]
        return (float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else None, len(v))

    per_config = {c: dict(vs_base_honest=mean_cost(c, "vs_base_honest"),
                          vs_best_honest=mean_cost(c, "vs_best_honest"),
                          vs_same_recipe=mean_cost(c, "vs_same_recipe"),
                          same_recipe_gap=mean_cost(c, "same_recipe_gap"))
                  for c in costs}
    best_attacker = min((c for c in costs if 0 in costs[c]),
                        key=lambda c: costs[c][0]["vs_best_honest"])

    summary = dict(
        micro_batch=M, attacker=ATTACKER,
        honest_final_loss=hon_final, attacker_final_loss=att_final,
        best_honest_config=best_honest, best_attacker_config=best_attacker,
        tuned_cost=per_config[best_attacker]["vs_best_honest"],
        per_config_cost=per_config, per_run_cost=costs,
        mechanism={**{f"honest/{c}_s{s}": _mechanism(p / "log.jsonl")
                      for (c, s), p in sorted(hon_paths.items())},
                   **{f"{ATTACKER}/{c}_s{s}": _mechanism(p / "log.jsonl")
                      for (c, s), p in sorted(att_paths.items())}},
    )
    (RESULTS / "tuning_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"best honest recipe: {best_honest}   best attacker recipe: {best_attacker}")
    print(f"{'config':16s} {'honest final':>22s} {'attacker final':>22s} "
          f"{'cost vs base':>16s} {'cost vs best':>16s} {'same recipe':>16s} {'gap':>14s}")
    for c in sorted(set(hon_final) | set(att_final)):
        hf = ", ".join(f"{v:.3f}" for _, v in sorted(hon_final.get(c, {}).items()))
        af = ", ".join(f"{v:.3f}" for _, v in sorted(att_final.get(c, {}).items()))
        cb = per_config.get(c, {}).get("vs_base_honest")
        cs = per_config.get(c, {}).get("vs_best_honest")
        cr = per_config.get(c, {}).get("vs_same_recipe")
        gp = per_config.get(c, {}).get("same_recipe_gap")
        fmt = lambda x: "-" if not x else f"{100*x[0]:.0f}%" + (f"±{100*x[1]:.0f}" if x[1] else "") + f" (n={x[2]})"  # noqa: E731
        gtxt = "-" if not gp else f"{gp[0]:.3f}" + (f"±{gp[1]:.3f}" if gp[1] else "")
        print(f"{c:16s} {hf:>22s} {af:>22s} {fmt(cb):>16s} {fmt(cs):>16s} "
              f"{fmt(cr):>16s} {gtxt:>14s}")
    for k, v in summary["mechanism"].items():
        if v:
            print(k, {b: (round(x["median_gnorm"], 2), round(x["frac_clipped"], 2))
                      for b, x in v.items()})


if __name__ == "__main__":
    main()
