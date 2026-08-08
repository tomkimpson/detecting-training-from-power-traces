# E2 — explicit-duration HSMM at the locked cell

**Date:** 2026-08-08
**Branch:** `feat/hsmm-renewal-smoke`
**Status:** complete. **The primary criterion fails on both clauses, and the
duration claim is unsupported on every arm.** One unanticipated positive result,
which the duration-neutral control attributes to something other than durations.

Contract: `notes/results/hsmm-cell-confirmation-and-e2-precommitment.md` as
amended 2026-08-08 (absolute pre-fix targets retired as gates; raw model
comparisons primary; IAAFT the primary arm).

## The precommitted question

> At the fixed 10 s, 20%-share operating point, does explicit-duration
> marginalisation separate training from an IAAFT phase-randomised surrogate
> materially better than E1 and otherwise identical duration-neutral sequence
> models?

**No.**

## Run

```bash
MPLCONFIGDIR=/tmp/mplconfig-hsmm python scripts/hsmm_e2.py
```

5 seeds x 3 arms x 4 models, 120 fit / 120 evaluation traces, 500 bootstrap
replicates, 695 s CPU-only. Every model scores from `(t, P_obs)` alone. The three
HSMM variants share one code path and a shared duration support, differing only
in the duration law — asserted by test, so no margin can come from an incidental
difference.

## Result: AUC by arm and model

| Arm | E1 (hurdle) | HSMM explicit | HSMM geometric | HSMM shuffled |
|---|---:|---:|---:|---:|
| **IAAFT** (primary) | 0.647 ± 0.039 | **0.652 ± 0.018** | 0.641 ± 0.024 | 0.597 ± 0.043 |
| AAFT (robustness) | 0.632 ± 0.039 | 0.658 ± 0.031 | 0.647 ± 0.035 | 0.607 ± 0.057 |
| composite (safeguard) | 0.888 ± 0.022 | **0.959 ± 0.013** | 0.949 ± 0.013 | 0.935 ± 0.028 |

### Primary criterion — fails on both clauses

| Clause | Required | Observed | |
|---|---|---:|---|
| IAAFT HSMM AUC | >= 0.75 | **0.652** | fail |
| improvement over E1 | >= +0.10 | **+0.006 ± 0.027** | fail |

The improvement's paired interval excludes zero in **0 of 5** replicates. Against
the phase-matched surrogate, the explicit-duration HSMM and the marked-renewal
hurdle model are indistinguishable.

### The duration claim — unsupported on every arm

| Arm | vs geometric | vs shuffled | intervals excluding zero |
|---|---:|---:|---|
| IAAFT | +0.011 ± 0.008 | +0.055 ± 0.032 | 2/5 and 3/5 |
| AAFT | +0.011 ± 0.005 | +0.051 ± 0.052 | 3/5 and 2/5 |
| composite | +0.010 ± 0.003 | +0.024 ± 0.016 | 4/5 and 3/5 |

The margin over the geometric-duration control is **+0.010 to +0.011 everywhere**,
an order of magnitude below the +0.05 requirement. The shuffled control is beaten
by more, but never with intervals excluding zero in every replicate.

**A geometric-dwell HMM recovers essentially all of whatever the HSMM gains.**
Explicit duration modelling is not what is doing the work.

## The unanticipated positive: the operational safeguard

Against the **physical composite null**, the HSMM clearly beats E1:

- AUC +0.070 ± 0.021, paired interval excluding zero in **5 of 5** replicates;
- TPR at composite-null FAR 0.05 rises from **0.338** (E1) to **0.472**;
- per-family FPR at that threshold <= 0.05 for every family
  (inference 0.05, AR(1) 0.04, resonant AR(2) 0.02, controllers 0.00,
  periodic inference 0.00).

Non-inferiority was the bar; the result is superiority. But the geometric control
reaches 0.949 against explicit's 0.959, so **this gain is duration-neutral**. It
comes from the sequence model and from marginalising over segmentations instead
of committing to the extractor's hard segmentation — precisely the alternative
explanation the control was precommitted to separate. It is a real improvement in
blind detection and it is *not* evidence for duration-aware structure.

## The gap the HSMM did not touch

On IAAFT the corrected alignment-informed benchmark sits at **E0 = 0.899** with
E1 at 0.647 — an estimation gap of **+0.252** — and the HSMM closes
**aggregate R = 0.023**, about 2% of it. Corrected `align+` is
**+0.387 ± 0.177**, recomputed inside this harness under the hurdle
representation, so the alignment information is real and substantial.

So the picture is coherent and negative: **knowing where the events are is worth
a great deal; an explicit-duration sequence model recovers almost none of it.**

The composite arm's aggregate R is reported as not meaningful: its gap is
**−0.018** (E0 0.871 below E1 0.888), i.e. blind scoring already exceeds the
alignment-informed benchmark there, consistent with the +0.026 gap having been
noise relative to seed variation.

### Retired pre-fix bars, reported for continuity

Neither cleared: IAAFT 0.8094 (observed 0.652), AAFT 0.8066 (observed 0.658).
These were retired as gates because they were derived from a biased E0; they are
recorded only so the historical claim can be traced.

## What this establishes

1. **The duration-aware hypothesis is not supported at its own best-identified
   operating point.** This is not a null from a saturated experiment — the cell
   was selected, confirmed over fresh seeds, and carries a large measured
   alignment benefit. The model built to exploit that benefit does not.
2. **Marginalising over segmentations does help against physical nulls**
   (+0.070 AUC, +0.134 TPR at FAR 0.05), which is a usable improvement to the
   Rung-1 detector bank and is independent of any duration claim.
3. **`spec.md`'s stated NO-GO conditions are met** for the extension as framed:
   the score does not require latent boundaries, but neither does it demonstrate
   recovered compute-to-communication structure.

## Honest limits

- Five seeds at 120/120. The margins over the geometric control are so small
  (+0.010) that more seeds would sharpen the interval, not the conclusion; the
  primary criterion misses by 0.098 AUC, which no plausible sample size closes.
- Fitting is **symmetric and blind** for both classes. `spec.md` permits
  profiling the training model from latent labels as a "best plausible chance";
  that was declined because it reintroduces the asymmetry that invalidated E0.
  `latent_segmentation` is implemented and tested, so this remains checkable if
  the negative is to be pressed further.
- Emissions are diagonal Gaussians on normalised power and slope. A richer
  emission model is the most plausible place a stronger HSMM could hide.
- Only the IAAFT/AAFT/composite nulls at one cell. Nothing here bears on real
  distributed hardware.
