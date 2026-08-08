# Stage 3 — is there a regime where knowing the true event locations helps?

**Date:** 2026-08-08
**Branch:** `feat/hsmm-renewal-smoke`
**Status:** complete. **Answer: yes, marginally.** One near-qualifying regime
exists; nothing passes the guarded admission rule outright.

Design: `notes/plans/hsmm-e2-preconditions-plan.md`.
Prerequisite: `notes/results/hsmm-e0-defect-and-fair-ceiling.md` (the corrected,
metadata-free alignment benchmark this campaign depends on).

## Question

Stages 1 and 2 left the extension without a target: with a valid benchmark there
was no measured gap anywhere for an explicit-duration model to close, and every
population saturated at AUC ~1.000. So the question stopped being *"can an HSMM
beat E1"* and became:

> Can we construct a non-saturated regime in which knowledge of the true event
> locations measurably helps?

## Method

Four desaturation axes, swept as a coordinate search from the 90 s baseline and
then combined in a targeted grid — 40 operating points, 264 s, CPU-only:

1. **record length** — 3/5/10/20/45/90 s;
2. **dilution** — the training line at a share of a Ko eq-11 aggregate
   (`aggregate_F_phase_meta`, a new metadata-carrying sibling of `aggregate_F`);
3. **event contrast** — `f_peak_frac`, the workload's peak share of `F_max`,
   against fixed meter noise;
4. **harder nulls** — phase-randomised surrogates of the positives, matched on
   mean, variance and power spectrum (FT) or additionally on the full marginal
   (AAFT).

Three quantities per point (`powerladder/typeb/renewal_regime.py`):

- **E1** — blind discrimination, both sides through the same procedure;
- **align+** — true minus random alignment *within the same positive traces*,
  with a paired bootstrap interval (1000 replicates, positives and nulls
  resampled once per replicate so the pairing survives);
- **E0 − E1** — the estimation gap against the alignment-informed benchmark.

```bash
MPLCONFIGDIR=/tmp/mplconfig-hsmm python scripts/hsmm_regime_search.py
```

## Result 1: the ceiling confound was real, and record length lifts it

At 90 s, `align+` is **+0.001** [+0.000, +0.003] — knowing every true event
location is worth nothing, because random windows already reach 0.999. Shorten
the record and the confound lifts:

| Record | E1 | E0 true | E0 random | align+ [95% CI] |
|---:|---:|---:|---:|---|
| 3 s | 0.966 | 1.000 | 0.653 | +0.346 [+0.284, +0.411] |
| 5 s | 0.972 | 1.000 | 0.901 | +0.099 [+0.068, +0.135] |
| 10 s | 0.989 | 1.000 | 0.945 | +0.055 [+0.029, +0.082] |
| 45 s | 1.000 | 1.000 | 0.998 | +0.002 [+0.000, +0.005] |
| 90 s | 1.000 | 1.000 | 0.999 | +0.001 [+0.000, +0.003] |

**Local event information is real and can be large.** The stage-2 reading — that
alignment adds nothing — was an artefact of measuring only at 90 s. That said, on
its own a short record does not produce an E2 target: the blind detector tracks
the ceiling closely (E1 0.966 against E0 1.000 at 3 s), so `E0 − E1` stays ~0.03.

## Result 2: what each axis does

- **Record length** is the strongest desaturation axis, but shortens toward
  degeneracy (see below).
- **Dilution** is the most operationally meaningful: it collapses event recall
  (0.83 undiluted → 0.19 at 20% share) while barely touching E1 at 90 s. It is
  the axis that separates *recall* from *discrimination*.
- **Event contrast** is nearly inert. At `f_peak_frac` 0.053 — 16x below default
  — E1 is still 1.000 at 90 s. The detector is remarkably robust to amplitude.
- **Nulls**: the **FT** surrogate is useless (align+ exactly 0.000; it preserves
  the spectrum but not the marginal, so amplitude features separate it
  trivially). The **AAFT** surrogate bites: it matches the marginal exactly and
  raises align+ to +0.221 at 10 s where the composite null gives +0.055.

## Result 3: the strict rule admitted an artefact

Under the rule exactly as agreed, one point qualified: `T3_share0.35`
(E1 0.726, E0 0.927, align+ **+0.580**, gap +0.201). It should not have.

At 3 s with a 35% share there is a median of **2 true events per trace**, and
**35%** of feature vectors are the all-zero fallback that `summarize_events`
returns when no events are extracted. The separation is substantially the
degenerate-versus-non-degenerate distinction, and no duration law is estimable
from two events.

Two guards were therefore added to `admits_e2` (both reported alongside the
unmodified strict verdict, never silently replacing it):

- `enough_events_per_trace` (>= 5);
- `not_degenerate` (all-zero feature-vector fraction <= 0.05).

**Under the guarded rule nothing is admitted.**

## Result 4: one near-qualifying regime

| Point | E1 | E0 true | E0 rand | align+ [CI] | E0−E1 | events | degen | recall |
|---|---:|---:|---:|---|---:|---:|---:|---:|
| **`T10_share0.2_surrogate_aaft`** | 0.661 | 0.993 | 0.717 | **+0.276** [+0.218, +0.334] | **+0.331** | 9 | 0.01 | 0.19 |
| `T10_share0.2` | 0.856 | 0.985 | 0.678 | +0.307 [+0.246, +0.375] | +0.128 | 10 | 0.09 | 0.19 |
| `T5_surrogate_aaft` | 0.839 | 1.000 | 0.389 | +0.610 [+0.544, +0.675] | +0.161 | 4 | 0.03 | 0.80 |

`T10_share0.2_surrogate_aaft` — 10 s records, training at 20% of an aggregate,
against marginal-and-spectrum-matched surrogate nulls — satisfies **every clause
but one**:

- benchmark not saturated (E0 random 0.717);
- align+ +0.276, interval excluding zero;
- **the largest estimation gap in the campaign, +0.331**;
- 9 events per trace, 1% degenerate — a duration law is estimable;
- event recall 0.19, so poor event recovery plausibly explains the gap;
- nominal 20 Hz channel.

It fails only `e1_in_measurable_band`: E1 = 0.661 against a floor of 0.70.

## Two threshold judgements this exposes

Both are thresholds chosen when the rule was written, and both now bind:

1. **The `E0_true` saturation clause misfires.** Its purpose was to guarantee
   headroom for measuring `align+`. That headroom lives in `E0_random`, not
   `E0_true` — a high true-aligned benchmark with a lower blind score *is* the
   estimation gap being sought, so gating on `E0_true` excludes the target
   configuration. It blocked 6 points, including three with gaps above 0.15.
   The amended rule uses `e0_random_not_saturated`.
2. **The E1 band floor of 0.70 is doing unintended work.** E1 = 0.661 is not "too
   weak to measure" — it is well above chance with a benchmark at 0.993. The
   floor was meant to exclude regimes where nothing is learnable; it is instead
   excluding the regime where blind estimation struggles most.

Relaxing the floor to ~0.65 admits `T10_share0.2_surrogate_aaft`. That is a
judgement call, not a result, and is flagged rather than taken.

## Verdict

The stage-3 question has a **qualified yes**. There is a regime — short
observation, diluted workload, spectrum-and-marginal-matched nulls — where true
alignment measurably beats random alignment (+0.276, interval excluding zero) and
the blind detector falls well short of the alignment-informed benchmark (+0.331).

So the resolved-event framing is **not** unsupported. It is unsupported *at the
operating point the paper currently uses* (90 s, undiluted, composite null),
where align+ is +0.001, and supported in a corner that is operationally
meaningful but narrow.

## Caveats that bound the claim

- **The AAFT surrogate is a statistical null, not a workload.** It is the right
  falsification instrument for locality but it is not hard inference or a
  controller, so a claim denominated against it means something different from
  the Rung-2 claim.
- **Only the drift/honest positive family was swept.** Work-variation cells were
  not re-measured under the new axes.
- **`E0` remains an alignment-informed benchmark for this feature family**, not a
  universal ceiling. A sequence model could exploit ordering these trace-level
  summaries discard — which is an argument for E2 having *more* headroom than
  +0.331, not less.
- The single-A100 transport limitation from
  `hsmm-renewal-smoke-findings.md` is unchanged: none of this validates that an
  external meter sees a distributed all-reduce phase.

## What this makes the next step

Not E2 development, and not more searching. The decision is whether
`T10_share0.2_surrogate_aaft` is accepted as the target, which turns on the two
threshold judgements above. If accepted, E2 is built and scored on exactly one
question — what fraction of the measured +0.331 gap an explicit-duration forward
likelihood recovers — with the physically-motivated composite null re-introduced
alongside the surrogate before any claim is made.
