# Rung 2 — training-vs-inference classification: findings

**Date:** 2026-07-24 · **Status:** numbers frozen (n=200, slurm array 5612 on MATS
`compute`) · **Branch:** `feat/phase2-rung2` · **Phase:** 2 (plan §3.3)

Machine outputs: `results/rung2/rung2_summary.json`, `figures/rung2_stated_transfer.*`,
`figures/rung2_controls.*`. Harness: `powerladder/typeb/rung2{,_features,_scenarios}.py`,
`scripts/rung2_eval.py`. Regenerate: `sbatch --array=0-13 scripts/slurm/rung2_eval.sbatch`
then `python scripts/plot_rung2.py`.

## What Rung 2 asks

Rung 1 asks *is there iteration-structured cyclicity?* Rung 2 converts that structure
into a decision — *is this trace more training-like than the stated inference null?* —
at the cost of an explicit inference null (`ko_workload.inference_F`). Three decision
rules share one interpretable, **estimated-from-trace** physics feature vector (8 dims:
mtf, mtf-comb, spectral, viterbi, dg_fixed, dg_order_full, + estimated path-frequency
stability and phase-fold repeatability from the Viterbi path):

1. **physics score** — prespecified, no label fitting (null-standardised sum; the
   fixed-cadence comparator `dg_fixed` is pre-registered *out*, since the declared
   cadence is drawn from a band);
2. **fitted discriminant** — a regularised logistic model fitted to the generator (OOF);
3. **learned reference** — the Rahman RF baseline (shape features), reused as-is.

## Frozen results (n_each=200, 300 s, band 0.3–1.7 Hz, FAR=0.05)

**Stated population.** All three rules separate training from the hard inference null
perfectly: AUC 1.00, FNR 0 at FPR 0.05.

**Transfer / domain shift.** AUC stays 1.00 across every shift tested — f_peak 0.70/0.95,
150 s duration, cadence band shifted to 0.8–1.8 Hz, the hostile controller-limit-cycle
meter, and heavy coloured meter noise. The classifier is robust off the stated population.

**Semantic falsification controls** (fraction scored as training, FAR=0.05):

| control | truth | physics | fitted | RF |
|---|---|---|---|---|
| discarded-update decoy | not training | 1.00 | 1.00 | 1.00 |
| gradient-only | not training | 1.00 | 1.00 | 1.00 |
| non-ML kernel loop | not training | 1.00 | 1.00 | 1.00 |
| controller cycle | not training | 1.00 | 1.00 | 1.00 |
| periodic inference | not training | 1.00 | 1.00 | 1.00 |
| **async training** | **training** | **0.72** | **0.55** | **1.00** |
| co-resident mixture | training | 1.00 | 1.00 | 1.00 |

The headline: the five **non-training** loads all score *as* training. The meter
certifies the **physical schedule**, not training semantics — a discarded-update decoy,
a gradient-only pass, a training-shaped non-ML kernel loop, a power-controller limit
cycle, and a periodic request generator are indistinguishable at the schedule level from
genuine training. This is the honest scope of what a passive power meter can certify at
Rung 2, and it is a *feature of the claim*, not a failure: Rung 2 is a physical-likeness
test, and causal attribution to training requires the stated nuisance model.

## The one divergence: async training was *not* missed

The plan (§3.3) anticipated **async / de-periodicised genuine training** as the
scoped-out miss — "missed asynchronous training ⇒ the result is scoped to the efficient
iteration-structured family." I pre-registered `expect="not_train"` for it.

At the tiny smoke size (n=6) this held: the physics rule flagged only ~0–20% of async
traces. **At the frozen n=200 it does not.** The tracked physics rule catches async
training most of the time — AUC 0.89, 72% flagged; the RF catches all of it (100%). The
weakest rule is the *fitted linear* discriminant (AUC 0.75, 55%), not the physics tracker.

**Why the smoke was misleading.** The operating threshold is the 95th percentile of the
held-out null. At n=6 that quantile is estimated from six points — noisy and biased high,
so it flagged almost nothing (of anything). At n=200 the threshold is well-estimated and
the real separation shows through. *Lesson: frac-at-a-fixed-FAR is unreliable at tiny n;
the AUC (0.89) was the trustworthy signal and it was already telling the true story.*

**Why the physics answer is the right one.** The async control de-periodicises via OU
centre-frequency drift (1.2 Hz) plus Brownian phase slip (σ=0.15). This is exactly the
regime the Viterbi / order-tracking pipeline was built to survive — ST1 showed tracking
holds a wandering line to ≈0.8 Hz drift on 300 s, and ST2 showed the tracker holds
(TPR 1.0) where fixed tests collapse. So a drift-plus-slip "async" trace is *still a
trackable line*, and the tracker follows it. The genuine evasion is the ≈zero-cost
**work-variation** attack characterised on the ST2 frontier (work_sigma → 0.5), which
moves the cadence *without* leaving a coherent line — that is a de-periodicisation
*attack*, not what this async-training control models.

So the divergence is not a bug: it is a consistent, cross-rung robustness result. It does,
however, mean the async control as built does not exhibit the scope boundary the plan
wanted it to.

## Open decision (Phase-4 write-up)

Two honest framings, to settle when writing the §6 Rung-2 subsection:

- **(a) Report as robustness.** State plainly that de-periodicisation by drift + phase
  slip does *not* evade the tracked physics rule (async caught at AUC 0.89), and that the
  scope limit lives in the work-variation regime already quantified on the ST2 frontier.
  Note the rule contrast: the fitted linear discriminant is the one that degrades on
  async, which is itself informative about where each rule's power comes from.
- **(b) Strengthen the control.** Add a work-variation leg to the async control (reuse the
  ST2 `work` knob) so a Rung-2 control directly exhibits the missed-training boundary,
  rather than pointing at the ST2 frontier for it.

Recommendation: **(a)** — it is the truthful reading of the data and it strengthens the
method claim; cite the ST2 frontier for the genuine boundary. Keep (b) as an optional
extra control if a reviewer wants the miss shown inside Rung 2 itself.

## Cross-links

- Method / claim ladder: `notes/plans/plan-for-paper-2.md` §3.3.
- Tracker robustness this rests on: `notes/results/st1-findings.md` (wander boundary),
  ST2 frontier verdict `results/st2/frontier_summary.json` (work-variation evasion).
- Status: `tasks.md` Phase-2 Rung-2 entry.
