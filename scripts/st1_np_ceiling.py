"""NP-optimal LRT ceiling for the ST1 detector bake-off (Phase 2, optional).

Computes the Whittle spectral likelihood-ratio ceiling
(:mod:`powerladder.typeb.np_ceiling`) — the Neyman–Pearson optimal test between the
training and null generators under the stationary-Gaussian model — and reports each
corpus-free bake-off detector as a FRACTION of it: "X% of NP-optimal power at Y% of
the information cost" (``notes/discussion/method-soundness-and-prior-art.md`` §2.2).

The ceiling is fit ONCE on a Monte-Carlo corpus drawn from an rng stream disjoint from
the bake-off's (``corpus_seed_offset``), then frozen and applied to the EXACT bake-off
eval populations — same ``seed``/``drifts``/``n_each`` as ``scripts/plot_st1_bakeoff.py``,
reproduced here by replaying its RNG sequence and reusing its population builders — so
"fraction of optimal" is apples-to-apples. A parity guard re-scores the corpus-free
detectors and checks their TPRs against the frozen ``results/st1/bakeoff_summary.json``.

Ceilings are fit for three negatives: the inference null and the structural mix (the two
bake-off comparison columns) and the controller-only null (the hard case at 0.4 Hz). The
training template bank (the H1 model) is shared across all three.

CAVEAT (recorded in the summary ``notes``): NP-optimal under the WHITTLE model only — a
2nd-order/periodogram statistic that discards harmonic-phase coherence and the null's
non-Gaussian structure. Where a tracking detector approaches or beats it at high drift,
that is a finding about wandering-line structure, not a bug.

The full number-freeze is a single Slurm task (~10–15 min; the shared bank dominates, so
splitting by negative class would rebuild it) — repo policy forbids expensive runs on the
dev node, so local use is ``--smoke`` only.

Usage:
    python scripts/st1_np_ceiling.py                 # full run (Slurm; ~10-15 min)
    python scripts/st1_np_ceiling.py --smoke         # tiny end-to-end (-> *_smoke_*)
    python scripts/st1_np_ceiling.py --no-parity     # skip the bake-off TPR cross-check
Outputs: results/st1/np_ceiling_summary.json  (or np_ceiling_smoke_summary.json)
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))  # sibling scripts

import plot_st1_bakeoff as bakeoff  # noqa: E402  (reuse the frozen eval builders)
from powerladder.config import DEFAULT  # noqa: E402
from powerladder.typeb import np_ceiling as npc  # noqa: E402
from powerladder.typeb.ko_synth import ko_make_trace  # noqa: E402
from powerladder.typeb.roc import auc  # noqa: E402

_ROOT = pathlib.Path(__file__).resolve().parent.parent
_RESULTS = _ROOT / "results" / "st1"

# The three negative classes: (label, builder(n, rng)). inference/structural are the
# two bake-off columns; controller_only backs the hard case at HARD_CASE_DRIFT_HZ.
NEG_BUILDERS = {
    "inference": lambda n, rng: bakeoff._make_inference_negatives(n, rng),
    "structural": lambda n, rng: bakeoff._make_structural_negatives(n, rng),
    "controller_only": lambda n, rng: bakeoff._make_structural_negatives(
        n, rng, names=("controller",)),
}


def _structural_mix_sampler(r):
    """One structural-null trace, null type drawn UNIFORMLY from the mix.

    ``_make_structural_negatives(1, r)`` would return an ar1 trace every time — its
    round-robin allocator puts the single trace in the first class — so an MC S_neg
    built from it would see ar1 only. Draw the null name uniformly instead, so the
    estimated PSD reflects the true equal-parts ar1/ar1_t/controller mixture that
    the eval ``structural`` population (and the bake-off column) actually use.
    """
    name = bakeoff.STRUCTURAL_MIX[int(r.integers(len(bakeoff.STRUCTURAL_MIX)))]
    return _tp(bakeoff._make_structural_negatives(1, r, names=(name,))[0])


# MC samplers for the negative-class PSDs (drawn from the disjoint corpus stream).
NEG_SAMPLERS = {
    "inference": lambda r: _tp(ko_make_trace("infer", DEFAULT.ko, DEFAULT.ko_typeb, r)),
    "structural": _structural_mix_sampler,
    "controller_only": lambda r: _tp(
        bakeoff._make_structural_negatives(1, r, names=("controller",))[0]),
}


def _tp(tr):
    return tr.t, tr.P_obs


def build_eval_populations(n_each, drifts, eval_seed):
    """Reproduce the bake-off eval populations by replaying its RNG sequence.

    Order matches ``plot_st1_bakeoff.main``: the three negative classes (inference,
    structural, controller_only), then the per-class oracle-f0 draws (consumed in
    insertion order so the positives that follow are byte-aligned; kept, since they are
    the dg_fixed_oracle null alphas), then the positives per drift level.
    """
    rng = np.random.default_rng(eval_seed)
    negatives = {
        "inference": bakeoff._make_inference_negatives(n_each, rng),
        "structural": bakeoff._make_structural_negatives(n_each, rng),
        "controller_only": bakeoff._make_structural_negatives(
            n_each, rng, names=("controller",)),
    }
    # dg_fixed_oracle null alphas — same draws as bakeoff.oracle_null_alphas, in order.
    oracle_alphas = {
        cls: rng.uniform(DEFAULT.ko.f0_lo, DEFAULT.ko.f0_hi, size=len(tr))
        for cls, tr in negatives.items()
    }
    positives = {d: bakeoff._make_positives(n_each, rng, d) for d in drifts}
    return negatives, oracle_alphas, positives


def _det_metrics(pos_scores, neg_scores, fars):
    """Per-detector {auc, tpr:{far:val}} from precomputed score dicts."""
    out = {}
    for name in pos_scores:
        tpr = {f"{far:g}": bakeoff._tpr_at_far(pos_scores[name], neg_scores[name], far)
               for far in fars}
        out[name] = {"auc": auc(pos_scores[name], neg_scores[name]), "tpr": tpr}
    return out


def _ceiling_metrics(ceiling, pos, neg, fars):
    ps = npc.score_ceiling(ceiling, pos)
    ns = npc.score_ceiling(ceiling, neg)
    tpr = {f"{far:g}": bakeoff._tpr_at_far(ps, ns, far) for far in fars}
    return {"auc": auc(ps, ns), "tpr": tpr}


def _fractions(det_metrics, ceil_metrics, fars, tpr_floor=0.05):
    """rho_auc = (auc_det-.5)/(auc_ceil-.5); rho_tpr = tpr_det/tpr_ceil (None if the
    ceiling's own tpr is below tpr_floor — the ratio is meaningless there)."""
    denom_auc = ceil_metrics["auc"] - 0.5
    rho_auc, rho_tpr = {}, {f"{far:g}": {} for far in fars}
    for name, m in det_metrics.items():
        rho_auc[name] = ((m["auc"] - 0.5) / denom_auc) if abs(denom_auc) > 1e-9 else None
        for far in fars:
            k = f"{far:g}"
            tc = ceil_metrics["tpr"][k]
            rho_tpr[k][name] = (m["tpr"][k] / tc) if tc >= tpr_floor else None
    return {"rho_auc": rho_auc, "rho_tpr": rho_tpr}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    p = DEFAULT.np_ceiling
    ap.add_argument("--n-mc", type=int, default=p.n_mc)
    ap.add_argument("--f0-n", type=int, default=p.f0_n)
    ap.add_argument("--n-each", type=int, default=p.n_each)
    ap.add_argument("--smoke", action="store_true",
                    help="tiny sizes + disjoint *_smoke_* output (plumbing check)")
    ap.add_argument("--no-parity", action="store_true",
                    help="skip the bake-off TPR cross-check")
    args = ap.parse_args()

    lo, hi = DEFAULT.st1.band_lo, DEFAULT.st1.band_hi
    drifts = list(bakeoff.DRIFTS_HZ)
    fars = list(bakeoff.FARS)

    n_mc, f0_n, n_each = args.n_mc, args.f0_n, args.n_each
    stem = "np_ceiling_summary.json"
    if args.smoke:
        n_mc, f0_n, n_each = 40, 9, 40
        stem = "np_ceiling_smoke_summary.json"
    out_path = _RESULTS / stem

    f0_grid = np.linspace(DEFAULT.ko.f0_lo, DEFAULT.ko.f0_hi, f0_n)
    corpus_rng = np.random.default_rng(p.eval_seed + p.corpus_seed_offset)

    t0 = time.time()
    print(f"building training bank: {f0_n} templates x n_mc={n_mc} ...", flush=True)
    S_bank, bank_freqs = npc.build_training_bank(
        DEFAULT.ko, DEFAULT.ko_typeb, f0_grid=f0_grid, drift_grid=p.drift_grid,
        n_mc=n_mc, rng=corpus_rng, band_lo=lo, band_hi=hi)
    print(f"  bank done in {time.time()-t0:.0f} s", flush=True)

    ceilings = {}
    for label, sampler in NEG_SAMPLERS.items():
        t1 = time.time()
        ceilings[label] = npc.fit_ceiling(
            sampler, label, S_bank=S_bank, f0_grid=f0_grid, bank_freqs=bank_freqs,
            n_mc=n_mc, rng=corpus_rng, band_lo=lo, band_hi=hi, psd_floor=p.psd_floor)
        print(f"  S_neg[{label}] in {time.time()-t1:.0f} s", flush=True)

    # ---- eval populations (byte-reproduced from the bake-off) --------------------
    negatives, oracle_alphas, positives = build_eval_populations(
        n_each, drifts, p.eval_seed)
    detectors = bakeoff._detectors(include_semicoh=False)

    neg_scores = {cls: bakeoff._score(tr, detectors, oracle_alphas[cls])
                  for cls, tr in negatives.items()}
    pos_scores = {d: bakeoff._score(pos, detectors, np.array([tr.f0 for tr in pos]))
                  for d, pos in positives.items()}

    # ---- assemble the two bake-off columns + the hard case -----------------------
    def column(neg_label):
        col = {"ceiling": {"auc": [], "tpr": {f"{f:g}": [] for f in fars}},
               "detectors": {}, "rho_auc": {}, "rho_tpr": {f"{f:g}": {} for f in fars}}
        det_names = list(pos_scores[drifts[0]].keys())
        for name in det_names:
            col["detectors"][name] = {"auc": [], "tpr": {f"{f:g}": [] for f in fars}}
            col["rho_auc"][name] = []
            for far in fars:
                col["rho_tpr"][f"{far:g}"][name] = []
        for d in drifts:
            dm = _det_metrics(pos_scores[d], neg_scores[neg_label], fars)
            cm = _ceiling_metrics(ceilings[neg_label], positives[d],
                                  negatives[neg_label], fars)
            fr = _fractions(dm, cm, fars)
            col["ceiling"]["auc"].append(cm["auc"])
            for far in fars:
                col["ceiling"]["tpr"][f"{far:g}"].append(cm["tpr"][f"{far:g}"])
            for name in det_names:
                col["detectors"][name]["auc"].append(dm[name]["auc"])
                col["rho_auc"][name].append(fr["rho_auc"][name])
                for far in fars:
                    k = f"{far:g}"
                    col["detectors"][name]["tpr"][k].append(dm[name]["tpr"][k])
                    col["rho_tpr"][k][name].append(fr["rho_tpr"][k][name])
        return col

    by_negative = {"inference": column("inference"), "structural": column("structural")}

    hd = bakeoff.HARD_CASE_DRIFT_HZ
    dm = _det_metrics(pos_scores[hd], neg_scores["controller_only"], fars)
    cm = _ceiling_metrics(ceilings["controller_only"], positives[hd],
                          negatives["controller_only"], fars)
    hard_case = {"drift_hz": hd, "negatives": "controller_only",
                 "ceiling": cm, "detectors": dm, **_fractions(dm, cm, fars)}

    # ---- parity guard: our reproduced detector TPRs vs the frozen bake-off -------
    parity = {"checked": False}
    frozen = _RESULTS / "bakeoff_summary.json"
    if not args.no_parity and not args.smoke and frozen.exists():
        ref = json.loads(frozen.read_text())
        deltas = []
        for cls in ("inference", "structural"):
            for far in fars:
                k = f"{far:g}"
                for name, ours in by_negative[cls]["detectors"].items():
                    if name in ref["tpr"][cls][k]:
                        deltas += [abs(a - b) for a, b in
                                   zip(ours["tpr"][k], ref["tpr"][cls][k][name])]
        parity = {"checked": True, "max_tpr_delta_vs_bakeoff": max(deltas) if deltas else None}
        print(f"parity: max |TPR delta vs bakeoff| = {parity['max_tpr_delta_vs_bakeoff']:.4g}",
              flush=True)

    summary = {
        "n_each": n_each, "eval_seed": p.eval_seed, "drifts_hz": drifts, "fars": fars,
        "n_mc": n_mc, "f0_n": f0_n, "f0_grid_hz": [float(x) for x in f0_grid],
        "drift_grid_hz": list(p.drift_grid), "band": [lo, hi],
        "corpus_seed_offset": p.corpus_seed_offset,
        "detectors": list(pos_scores[drifts[0]].keys()),
        "by_negative": by_negative,
        "hard_case": hard_case,
        "parity_check": parity,
        "provenance": {
            "method": "Whittle spectral LRT, f0-marginalised (logsumexp over an "
                      "MC per-f0 training template bank), drift-pooled",
            "corpus_rng": "default_rng(eval_seed + corpus_seed_offset), disjoint from eval",
            "eval_rng": "default_rng(eval_seed) replaying plot_st1_bakeoff.main order",
            "n_train_templates": f0_n, "n_mc_per_template": n_mc,
        },
        "notes": (
            "Ceiling = NP-optimal LRT UNDER THE WHITTLE (stationary-Gaussian) MODEL: a "
            "2nd-order periodogram statistic that discards harmonic-phase coherence and "
            "the null's non-Gaussian OU/burst structure, so the true optimum can exceed "
            "it. A tracking detector approaching/beating it at high drift is a finding "
            "about wandering-line structure, not a bug. rho_auc=(auc-0.5)/(auc_ceil-0.5); "
            "rho_tpr=tpr/tpr_ceil (null where tpr_ceil<0.05). Information cost is "
            "categorical: ceiling needs white-box generative access to both samplers + a "
            "large MC corpus; the corpus-free detectors need one trace and the band."
        ),
    }
    _RESULTS.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(summary, indent=2) + "\n")
    tmp.replace(out_path)
    print(f"-> {out_path}  ({time.time()-t0:.0f} s total)")


if __name__ == "__main__":
    main()
