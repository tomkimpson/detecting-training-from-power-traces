# The E0 oracle ceiling was invalid — defect, fix, and what the fair ceiling says

**Date:** 2026-08-07
**Branch:** `feat/hsmm-renewal-smoke`
**Status:** defect confirmed and fixed. **Supersedes the E0 result in
`notes/results/hsmm-renewal-smoke-findings.md` and every oracle column in
`notes/results/hsmm-generalization-findings.md` and `results/hsmm_meter/`.**

## The defect

`scripts/smoke_hsmm.py` scored E0 by comparing `oracle_trace_features` on
positives against `blind_trace_features` on nulls. Those are **two different
measurement procedures**, and several oracle features are read straight off the
generator's latent schedule rather than off the trace:

- `paired_fraction` is the literal constant `1.0` for any non-empty oracle window
  set (`oracle_trace_features` passes `paired_fraction=1.0 if events else 0.0`);
- `duration_median_s` and `duration_cv` are the generator's own communication
  durations, filtered only by which windows pass the amplitude check — so they
  report the latent schedule rather than anything measured from the trace. (They
  are not bit-identical between a real and a signal-free trace, because a
  different subset of windows survives; the values always lie in the support of
  the true durations.)

Blind extraction on a null can never produce `paired_fraction = 1.0` (measured
medians: 0.72 on hard inference, 0.53 on AR(1)), so that one feature separates
the populations perfectly regardless of what the trace contains.

**Demonstration.** Replace every positive trace with white Gaussian noise of the
same mean and standard deviation, keep the true metadata, and re-score:

| Feature | oracle, real trace | oracle, **noise** trace | blind, inference null | blind, AR(1) null |
|---|---:|---:|---:|---:|
| `paired_fraction` | 1.0000 | **1.0000** | 0.7164 | 0.5272 |
| `duration_median_s` | 0.3000 | **0.3000** | 0.3500 | 0.5500 |
| `duration_cv` | 0.2572 | **0.2583** | 0.6391 | 0.4931 |
| `amplitude_median_z` | 2.6819 | 0.3305 | 1.9112 | 1.3620 |
| `plateau_noise` | 0.0884 | 0.7265 | 0.2133 | 0.2173 |

(Population medians. Per trace the two duration columns differ slightly, because a
different subset of windows survives the amplitude check; `paired_fraction` is
exactly equal trace by trace.)

The oracle achieves **AUC 1.000 on pure noise** at every channel tested
(20 Hz pristine, 2 Hz, and the 1 Hz / 1 s meter), under both the full and the
shape/duration-only feature sets. The waveform-derived features do degrade
correctly on noise; it makes no difference, because one leaking feature
saturates the AUC.

So the E0 gate never measured whether local event information survives the
observation map. It measured whether the generator's latent schedule differs from
a blind detector's output — which is true by construction.

## What this invalidates

- **The E0 gate itself**: "oracle AUC is 1.00 against the union and against every
  individual null for every positive condition" and `E0_oracle_continue: true` in
  `results/hsmm_smoke/summary.json`.
- **Every oracle-minus-E1 gap** computed in stage 1 and stage 2, therefore the
  whole E2 cell-selection criterion as first implemented.
- `oracle_auc_recovered_by_blind`, the E1 continue criterion. Note the direction:
  the bogus denominator is pinned at its maximum, so the ratio is *conservative* —
  E1's pass is not at risk, but the number does not mean what it says.

**E1 itself is untouched.** The blind path scores positives and nulls with the
same procedure, so all of stage 1 (1A/1B/1C) and the stage-2 frozen/refit results
stand as reported.

## The fix

`spec.md` already specified the right control — *"compare with randomly aligned
or best-matched windows from each null"*. The implementation had not done it.

`powerladder.typeb.renewal_campaign.oracle_null_matrix` now windows each null
trace at uniformly random start times, with the event count and durations
resampled from the positive population. Every metadata-derived feature is then
matched by construction, and only genuine local waveform structure can separate
the two sides.

**Validation:** with the fair null side, the noise control falls to **AUC 0.530**
(honest) and **0.467** (work 0.7) at the pristine channel — chance, as it must be.

## What the fair ceiling says

Two quantities, 100 traces per population, 90 s records, full feature set.
`align+` is the fair E0 minus the same measurement with *random* alignment on the
**same positive traces** — i.e. what knowing the true event locations actually buys:

| Cell | Condition | E0 (true align) | E0 (random align) | **align+** | E1 blind | E0 − E1 |
|---|---|---:|---:|---:|---:|---:|
| 20 Hz, 0 s | honest | 1.000 | 0.998 | **+0.002** | 1.000 | 0.000 |
| 5 Hz, 0.25 s | honest | 1.000 | 0.945 | +0.055 | 0.995 | 0.005 |
| 2 Hz, 0 s | honest | 0.999 | 0.973 | +0.027 | 0.931 | 0.069 |
| 2 Hz, 0.5 s | honest | 0.974 | 0.835 | +0.139 | 0.794 | 0.180 |
| 1 Hz, 1 s | honest | 0.854 | 0.596 | +0.258 | 0.672 | 0.182 |
| 20 Hz, 0 s | work 0.7 | 1.000 | 0.978 | +0.022 | 1.000 | 0.000 |
| 5 Hz, 0.25 s | work 0.7 | 1.000 | 0.998 | +0.002 | 1.000 | 0.000 |
| 2 Hz, 0 s | work 0.7 | 1.000 | 0.988 | +0.012 | 1.000 | 0.000 |
| 2 Hz, 0.5 s | work 0.7 | 0.999 | 0.979 | +0.019 | 1.000 | −0.001 |
| 1 Hz, 1 s | work 0.7 | 0.988 | 0.892 | +0.095 | 0.991 | −0.004 |

Three readings:

1. **No estimation gap on `work=0.7` at any channel.** The fair oracle−E1 gap is
   0.000 to −0.004 throughout. The primary event-model target has no E2 headroom.
2. **The only surviving candidate is the honest population at near-erasure
   channels** (2 Hz / 0.5 s: E0 0.974, E1 0.794, gap 0.180). That is a strange and
   weak E2 target: the no-regression anchor, at a channel the paper would not
   claim.
3. **Where the detector works well, true alignment adds ~nothing** (+0.002 to
   +0.055). Alignment only starts to matter once the whole measurement comes off
   its ceiling, at 2 Hz / 0.5 s and below.

## Corroboration from the stage-2 timing diagnostic

Event recall against ground truth, versus trace-level AUC, on `work=0.7`:

| Cell | Recall @100 ms | Recall @1 sample | Refit E1 AUC |
|---|---:|---:|---:|
| 20 Hz, 0 s | 0.991 | 0.991 | 1.000 |
| 20 Hz, 0.25 s | 0.091 | 0.091 | 1.000 |
| 20 Hz, 0.5 s | 0.042 | 0.042 | 1.000 |
| 10 Hz, 0.5 s | 0.044 | 0.044 | 1.000 |
| 2 Hz, 0 s | 0.164 | 0.609 | 1.000 |
| 1 Hz, 1 s | 0.029 | 0.317 | 0.998 |

At 20 Hz with 0.5 s integration the detector recovers **4.2%** of true
communication events and still separates training from all six nulls at **AUC
1.000**. Integration destroys events faster than decimation does, and neither
destroys the decision.

## Interpretation, stated carefully

The `align+` figures at the pristine channel are subject to a **ceiling
confound**: random-aligned windows already reach 0.978-0.998 there, so the
measurement has no room to show an alignment effect even if one exists. The
honest claim is therefore *not* "local event structure carries no information".
It is:

> At every channel where this detector performs well, the experiment cannot
> demonstrate that its power comes from resolved compute-to-communication events.
> Event recall and trace-level discrimination are decoupled — 4% recall still
> gives AUC 1.000 — and correct alignment adds under 0.06 AUC. Where saturation
> lifts, alignment does carry real information (+0.14 to +0.26).

That decoupling is what `spec.md` names as a NO-GO condition for the extension:
the score must not merely rediscover a global property, and there must be
oracle-aligned information at the hard cells. Under the fair ceiling, the hard
cells have no oracle-aligned *increment* to recover.

## What this implies for the next experiment

The blocking problem is **saturation**, not the model class. Every population in
this campaign separates at AUC ~1.000 on the pristine channel, so no comparison —
E0 vs E1, E1 vs E2, feature set vs feature set — has room to resolve anything.

The experiment that would actually test `spec.md`'s hypothesis is one tuned to a
**mid-range operating point**, where `align+` can be measured without a ceiling:
shorter records, lower event SNR, harder or better-matched nulls, or the
dilution/superposition families. If `align+` is large there, a duration-aware
model has something to recover and E2 is justified against a measured target. If
it stays near zero, the marked-renewal features are doing global work and the
duration-aware framing should be dropped rather than escalated.
