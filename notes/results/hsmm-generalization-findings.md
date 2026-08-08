# Stage 1 — generalization falsification of the E1 renewal detector

**Date:** 2026-08-07
**Branch:** `feat/hsmm-renewal-smoke`
**Status:** complete. **The single E2 candidate this stage produced was killed by a
one-line regularisation change.** One serious negative result stands.

> **CORRECTION (2026-08-07).** Every **oracle** column in this note is invalid —
> see [`hsmm-e0-defect-and-fair-ceiling.md`](hsmm-e0-defect-and-fair-ceiling.md).
> The E1 results (1A, 1B, 1C and the profile-width sweep) are unaffected, because
> the blind path scores positives and nulls with the same procedure. The 1A
> conclusion is *strengthened*: with a valid ceiling there is no estimation gap to
> attribute the honest-only loss to in the first place.

Design and rationale: `notes/plans/hsmm-e2-preconditions-plan.md`.
Predecessor: `notes/results/hsmm-renewal-smoke-findings.md`.

## Question

The E0/E1 smoke fitted the positive profile on the concatenation of all five
positive conditions and then evaluated those same named families. Its perfect
scores therefore could not distinguish *the features capture an invariant* from
*the profile interpolates among known attacks*.

## Run

```bash
MPLCONFIGDIR=/tmp/mplconfig-hsmm python scripts/hsmm_generalization.py
```

90 s records at 20 Hz; 250 fitting traces per configuration (held constant across
configurations, so no result is a fit-size artefact); 50 fitting traces per null
family; 200 calibration and 200 held-out evaluation traces per family; disjoint
deterministic RNG streams; composite threshold at FAR 0.05. 34 s, CPU-only.
Outputs: `results/hsmm_generalization/summary.json` plus two PNGs.

**Harness validation.** The `pooled_baseline` configuration reproduces the frozen
smoke exactly in all three feature sets (full and cadence-free 1.000 everywhere;
shape/duration 1.000, 1.000, 0.990, 0.970, 0.840), because a 250-trace pool and a
50-trace draw share the same RNG prefix.

## 1A — honest-only positive fit

Fitting on honest traces alone and evaluating every attack cell (TPR at composite
FAR 0.05, full features; the oracle is fitted under the same restriction):

| Condition | E1 TPR | E0 oracle TPR | E1 AUC | E0 AUC |
|---|---:|---:|---:|---:|
| honest | 1.000 | 1.000 | 1.0000 | 1.0000 |
| work 0.5 | 0.990 | 1.000 | 0.9901 | 1.0000 |
| work 0.7 | 0.940 | 1.000 | 0.9405 | 1.0000 |
| drift 0.8 | 0.900 | 1.000 | 0.9007 | 1.0000 |
| drift 1.5 | **0.705** | 1.000 | 0.7116 | 0.9998 |

Power degrades monotonically with attack severity while the oracle stays
saturated. On its face this is exactly the decision table's *blind estimation
gap*: oracle AUC 0.9998 >= 0.90, E1 declines materially, oracle-minus-E1 gap 0.29
in TPR and 0.29 in AUC, on the pristine channel. It was the only E2 candidate
stage 1 produced.

## 1B — leave-one-positive-family-out

Fitting on honest plus *one* attack family and evaluating the other:

| Fit | work 0.5 | work 0.7 | drift 0.8 | drift 1.5 |
|---|---:|---:|---:|---:|
| honest + drift (work held out) | 1.000 | 1.000 | 1.000 | 1.000 |
| honest + work (drift held out) | 1.000 | 1.000 | 1.000 | 1.000 |

No degradation at all under the full and cadence-free feature sets. Under the
strictest shape/duration-only set, holding out drift costs a little
(0.950 / 0.780 against the pooled baseline's 0.970 / 0.840) and holding out work
costs nothing.

**The detector does not need to have seen the attack family it is judging.** What
1A and 1B jointly show is that it needs the fit to contain *some* off-honest
variation, regardless of which kind — which points at the width of the fitted
profile rather than at any missing family-specific structure.

## The 1A gap is a profile-width artefact, not missing information

Two sensitivity tests on the honest-only fit, full features:

**Covariance shrinkage** (0.25 -> 0.99) moves drift 1.5 from 0.705 to 0.710 —
essentially flat. This tests the profile's *correlation structure*: shrinkage
blends the covariance towards its diagonal and preserves every marginal variance.

**Isotropic training-covariance inflation** is the actual width test:

| Inflation factor | honest | work 0.5 | work 0.7 | drift 0.8 | drift 1.5 | max null FPR |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1.000 | 0.990 | 0.940 | 0.900 | 0.705 | 0.060 |
| 2 | 1.000 | 0.990 | 0.940 | 0.900 | 0.710 | 0.060 |
| **5** | 1.000 | **1.000** | **1.000** | **1.000** | **1.000** | **0.050** |
| 10-200 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.040-0.050 |

Inflating the honest-only training covariance by 5x recovers the full oracle
performance in every cell **at no cost in false alarms** — the maximum null FPR
falls from 0.060 to 0.050.

So the honest-only deficit is not an information gap and not a blind-estimation
gap. It is a too-narrow Gaussian whose tails reject severe-but-genuine attacks,
and it is closed by a regularisation constant. An explicit-duration HSMM is not
the indicated fix, and building one against this gap would have bought a result
that one line of covariance scaling already delivers.

**Honesty note.** The factor 5 was read off the evaluation split, so it is a
*diagnostic* demonstrating that the loss is not information-limited. It is not a
tuned detector setting; adopting it would require its own held-out selection.

## 1C — leave-one-null-out: the serious negative result

Held-out confuser FPR at the composite threshold (full features; the other two
feature sets behave the same):

| Held-out null | Retained in calibration | Excluded from calibration |
|---|---:|---:|
| Hard inference | 0.055 | **1.000** |
| AR(1) | 0.020 | **0.985** |
| Resonant AR(2) | 0.050 | **1.000** |
| Controller | 0.015 | 0.015 |
| Controller + AR(1) | 0.020 | 0.020 |
| Periodic inference | 0.030 | **1.000** |

**Representation generalization is fine.** Dropping a family from the composite
denominator while it still participates in calibration costs almost nothing —
every retained-variant FPR stays at or below 0.090 across all three feature sets.

**Unseen-confuser robustness is absent.** When a family is absent from both the
fitted denominator and the calibration set, four of six families false-alarm at
0.985-1.000. This is structural, not incidental: the score is
`log p(x | train) - max_j log p(x | null_j)` over a *finite* fitted null set, so a
confuser with no denominator term has nothing to be compared against and lands
above the threshold the other families set.

The two controller families are the exception (0.015, 0.020) — they are rejected
even when wholly unseen, presumably because they sit far from the training cloud
in a direction other retained nulls already cover.

The governance reading is blunt: **the composite null must be exhaustive.** This
detector cannot be described as rejecting arbitrary structured non-training
computation; it rejects the confusers it was shown. Any claim built on it is
conditional on the stated null library, in a stronger sense than the Rung-2
conditionality already stated in `spec.md`.

## Verdict

- 1A: a real power loss, **explained** as profile width and closed by a 5x
  covariance inflation at no FPR cost. **Not an E2 target.**
- 1B: **passes outright.** No cross-family interpolation dependence.
- 1C retained-in-calibration: **passes.**
- 1C excluded-from-calibration: **fails hard** for inference, AR(1), resonant
  AR(2) and periodic inference. Not a formal level-control claim, but it bounds
  what may be claimed.

Stage 1 therefore produced **no surviving E2 cell**. The next test is the
observation-channel sweep (stage 2), which is where an information-erasure or a
genuine estimation gap is most likely to appear.

## Follow-up this raises

1. **Profile width deserves a principled setting**, chosen on a fitting split
   rather than read off evaluation — or a heavier-tailed profile in place of the
   Gaussian.
2. **Unseen-confuser behaviour is now a first-class limitation** and belongs in
   any write-up of this direction, not only in a smoke appendix.
3. Whether an explicit-duration model has *structurally* better unseen-confuser
   behaviour is a different and more interesting question than whether it wins on
   the hard positive cells. If E2 is ever built, this is the question it should
   be aimed at.
