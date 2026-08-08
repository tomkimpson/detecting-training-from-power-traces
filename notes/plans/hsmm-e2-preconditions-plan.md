# Plan — stress-test E1 before building E2

**Date:** 2026-08-07
**Status:** agreed (supersedes the "next step is E2" line in
`notes/results/hsmm-renewal-smoke-findings.md` §"Boundaries and next test")
**Branch:** `feat/hsmm-renewal-smoke`

## Why the order changed

`spec.md`'s provisional extension lists E2 (the explicit-duration HSMM) as the step
after E0/E1 pass. Both passed, so E2 is formally next. We are deferring it anyway, for
a measurement reason rather than a scope reason.

The E1 result is saturated: AUC 1.000 and TPR 1.000 at composite-null FAR 0.05 in all
five positive cells. A more expressive model cannot be *measured* against that — its
best available outcome is a tie, which would teach us nothing about why the method
works. Building E2 now risks an expensive tie.

One qualification, which strengthens rather than weakens the argument. E1 is not
saturated in *every* sense: held-out inference FPR rose from the calibrated 0.05 to
0.08, so calibration stability is genuinely open, and an HSMM could in principle
improve it. But that is an argument for first finding out where the current detector
actually breaks, not for building the HSMM now.

So: falsify first, and build E2 only against a *measured* gap.

## Stage 1 — generalization falsification

The branch's central weakness is that the positive profile is fitted on the
concatenation of all five positive conditions (`smoke_hsmm.run`, `blind_train`) and
then evaluated on those same named families. Three tests separate "the features capture
an invariant" from "the profile interpolates among known attacks".

### 1A. Honest-only positive fit

Fit the training profile on honest traces only; evaluate every work and drift cell.
This is the cleanest invariance test.

### 1B. Leave-one-positive-family-out

- fit on honest + drift, evaluate all work cells;
- fit on honest + work, evaluate all drift cells.

### 1C. Leave-one-null-out

Exclude one confuser from profile fitting, in two variants per family:

- **retained in calibration** — tests representation generalization (the threshold must
  still reject a family the composite denominator never saw);
- **excluded from fitting and calibration** — tests truly unseen-confuser FPR. No claim
  of formal level control is made for this variant.

### Protocol common to stage 1

- Primary outputs: composite-threshold TPR and per-null FPR. AUC is secondary.
- Every configuration is run for three feature sets: **full**, **cadence-free**, and
  **shape/duration-only**. The last is expected to be the most diagnostic — it is the
  only ablation with existing headroom (TPR 0.84 at drift 1.5).
- **Fit-set size is held constant across configurations.** The pooled baseline fit uses
  5 conditions x 50 traces = 250; the honest-only and leave-one-family-out fits draw
  the same 250 total from their permitted conditions. Otherwise a drop in power could
  be a sample-size artefact rather than a generalization failure.
- Fitting, calibration, and evaluation keep disjoint deterministic RNG streams.

## Stage 2 — meter boundary

Run before E2, in two regimes that fail differently:

1. **Frozen-profile transport** — fit at pristine 20 Hz, apply the unchanged model and
   preprocessing to degraded channels.
2. **Per-cell refit** — refit and recalibrate at each meter condition.

| Outcome | Interpretation |
|---|---|
| Frozen fails, refit succeeds | Domain/calibration shift; information remains |
| Frozen and refit fail, oracle succeeds | Blind estimation gap — a good E2 target |
| Frozen, refit and oracle all fail | The meter erased the information; an HSMM cannot help |
| Everything succeeds | Still no E2 headroom |

Two requirements:

- **Channel symmetry.** Every positive and every null passes through the identical
  integration/sampling transformation. A comparison across an asymmetric channel is
  meaningless.
- **Recompute E0 at every meter cell.** "E1 broke" alone does not justify an HSMM; only
  a surviving oracle distinguishes an estimation problem from an information problem.

## E2 cell-selection criterion

Build E2 only for cells satisfying all of:

- oracle AUC >= 0.90;
- E1 AUC or TPR materially declines;
- oracle-minus-E1 gap >= 0.10-0.15;
- the channel is not an obvious erasure point (e.g. the integrating 1 Hz meter).

That set identifies an *estimation* problem rather than an *information* problem.

## E2 success criterion

Not "does E2 also reach 1.00 on the easy population" but the fraction of the newly
measured oracle gap that it closes:

```
fraction of oracle gap recovered = (perf_E2 - perf_E1) / (perf_E0 - perf_E1)
```

## The admissible null result

If stage 1 and stage 2 reveal no cell where E0 stays strong while E1 weakens, E2 is not
built, and the finding is reported as:

> Within the modelled observable region, a comparatively simple marked-renewal feature
> model already extracts essentially all available local-event information; failures
> occur only when the observation channel erases that information.

This is a legitimate and arguably more interesting outcome than adding an HSMM because
it was next in the original plan.

## Implementation

- `powerladder/typeb/renewal_campaign.py` — shared population construction, feature
  caching, threshold/FAR machinery, metrics. Extracted from `scripts/smoke_hsmm.py`,
  which is refactored onto it and must reproduce `results/hsmm_smoke/summary.json`
  unchanged.
- `scripts/hsmm_generalization.py` -> `results/hsmm_generalization/`
- `scripts/hsmm_meter.py` -> `results/hsmm_meter/`
- Frozen ST1/ST2 artefacts are never touched.

Feature extraction is model-independent, so each stage extracts features once and every
leave-out configuration is a cheap re-fit of Gaussian profiles on cached matrices.
