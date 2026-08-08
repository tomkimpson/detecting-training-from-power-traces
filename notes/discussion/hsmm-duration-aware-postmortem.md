# Postmortem — duration-aware event detection (HSMM). Closed, do not retry as framed.

**Date:** 2026-08-08
**Branch:** `feat/hsmm-renewal-smoke` (kept, unmerged; PR #19 closed without merging)
**Verdict:** **killed.** The hypothesis failed at the operating point built
specifically to give it its best chance.

This is the index. If you are considering this idea again, read this file first.

## What was proposed

`spec.md`'s provisional extension: under work variation and cadence drift,
training traces should retain locally repeatable compute-to-communication events
whose *shape and duration* let an explicit-duration model (HSMM) distinguish
training from inference and controller-like nulls — where the existing frequency
-tracking detectors lose power. Three gates: E0 (oracle information ceiling),
E1 (blind marked-event detector), E2 (the HSMM).

## What was actually done

Full record in `notes/results/`, in order:

| Note | What it establishes |
|---|---|
| `hsmm-renewal-smoke-findings.md` | the original E0/E1 smoke — **its E0 gate is invalid**, see below |
| `hsmm-generalization-findings.md` | stage 1: cross-family transfer fine; unseen confusers fail hard |
| `hsmm-e0-defect-and-fair-ceiling.md` | **the E0 defect**, the fix, and the corrected ceiling |
| `hsmm-regime-search-findings.md` | stage 3: the search for a non-saturated regime |
| `hsmm-cell-confirmation-and-e2-precommitment.md` | the confirmed target cell + the E2 contract |
| `hsmm-e2-findings.md` | **E2 itself: the negative result** |

Plan and rationale: `notes/plans/hsmm-e2-preconditions-plan.md`.
Code: `powerladder/typeb/{renewal_campaign,renewal_regime,renewal_hurdle,hsmm}.py`,
`scripts/hsmm_*.py`, `tests/test_{renewal_campaign,renewal_regime,renewal_hurdle,hsmm}.py`.

## Why it was killed

The kill is *not* "we ran an HSMM and it was no better." The target was
constructed to be as favourable as possible:

1. At the paper's 90 s operating point everything saturates at AUC ~1.000, so
   nothing is measurable. Shortening the record lifts the ceiling.
2. A regime search over record length, dilution, event contrast and null
   hardness found one cell — **10 s records, training at 20% of an aggregate,
   against phase-randomised surrogate nulls** — that is off the ceiling and where
   knowing the true event locations is worth a lot (`align+` = **+0.387**).
3. That cell was confirmed **8/8 across fresh seeds** and carries an estimation
   gap of **+0.252** between blind scoring and the alignment-informed benchmark.
4. The explicit-duration HSMM closes **2%** of that gap (aggregate R = 0.023),
   misses the primary AUC bar by 0.098, and beats an otherwise identical
   **geometric-dwell HMM by +0.011** — an order of magnitude under the +0.05
   required to attribute anything to durations.

So: the local event information the hypothesis predicted **is real and is
substantial**, and an explicit-duration sequence model does not recover it. The
duration-neutral control is what makes this conclusive rather than suggestive.

## The three things worth keeping

1. **A duration-neutral detector improvement.** Marginalising over segmentations
   rather than committing to a hard event extraction gains **+0.070 AUC and
   +0.134 TPR at composite FAR 0.05** against the physical nulls (5/5 seeds,
   per-family FPR <= 0.05). This has nothing to do with durations and is the only
   operational improvement the whole line produced. If any of this is revisited,
   revisit *this*, not the HSMM.
2. **The E0 defect, as a methodological lesson.** The original oracle gate
   compared oracle-windowed positives against *blind*-extracted nulls, and
   `paired_fraction` is a hardcoded 1.0 for any oracle window set. It scored
   **AUC 1.000 on pure white noise carrying only the true metadata**, at every
   channel. It passed review and licensed a GO to E2. The control that would have
   caught it ("randomly aligned or best-matched windows from each null") was
   written in `spec.md` and never implemented.
3. **Unseen confusers are not rejected.** With a null family absent from both the
   fitted denominator and calibration, four of six false-alarm at 0.985-1.000.
   Generic to empirically-calibrated composite thresholds; it tightens the
   existing Rung-2 conditionality rather than contradicting the paper.

## Two artefact classes to watch for anywhere in this codebase

Both produced *better* numbers, which is the direction that does not prompt
suspicion:

- **Comparing two different measurement procedures.** Scoring one class with
  ground truth and the other with an estimator manufactures separation.
- **Sentinel values for undefined quantities.** `summarize_events` returns eleven
  zeros when no events are extracted, and reports *perfect* shape repeatability
  from a single event. A population where that is common "separates" on
  degeneracy alone. `powerladder/typeb/renewal_hurdle.py` fixes this properly
  (point mass for N=0, marginalisation over undefined marks) — **the legacy
  feature path in `renewal.py` still has it**, deliberately, so the pre-fix
  results stay auditable.

## If you are tempted to try again

Do not re-run the E0/E1/E2 ladder as specified — it is done, and the answer is
no. The only conditions under which this becomes live again:

- **real distributed hardware.** Everything here is synthetic and
  literature-parameterised. The decisive experiment remains simultaneous external
  measurement of multi-GPU/multi-node training with a genuine all-reduce phase.
  The single-A100 data is an expected negative transport control.
- **a richer emission model.** Emissions here are diagonal Gaussians on
  normalised power and slope. That is the most plausible place a stronger HSMM
  could still hide, and it is the one stone left unturned.

One loose end, deliberately not closed: fitting was **symmetric and blind** for
both classes. `spec.md` permits profiling the training model from latent labels
as a "best plausible chance"; that was declined because it reintroduces exactly
the asymmetry that invalidated E0. `latent_segmentation` is implemented and
tested, so that diagnostic is ~10 minutes if anyone wants to press the negative
further.

## Repo state

- `spec.md`'s "Provisional extension: duration-aware event detection" section
  **still describes this as an approved prototype with the original (invalid) E0
  design.** It needs amending to point here. Not done — `spec.md` requires
  explicit approval to edit.
- The manuscript is untouched by this work. Nothing here is a paper claim.
- Frozen ST1/ST2 artefacts, figures and `paper/` are untouched, verified.
- 332 tests pass; `scripts/smoke_hsmm.py` still reproduces its frozen
  `summary.json` byte-for-byte.
