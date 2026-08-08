# Locked confirmation of the E2 target cell, and the E2 precommitment

**Date:** 2026-08-08
**Branch:** `feat/hsmm-renewal-smoke`
**Status:** confirmation **passed** (8/8 replicates under both surrogate
constructions). Conditional GO to E2, scoped to this one cell. One blocker found
for the third precommitted comparison.

Selection: `notes/results/hsmm-regime-search-findings.md`.
Benchmark validity: `notes/results/hsmm-e0-defect-and-fair-ceiling.md`.

## The cell

10 s records at 20 Hz; the training workload at a **20% share** of a Ko eq-11
aggregate (`aggregate_F_phase_meta`); nulls that are phase-randomised surrogates
of independent positive draws. Nothing else is swept — this run exists to confirm
one point, not to search.

## Why a fresh-seed replication was needed

The stage-3 interval was pointwise after searching 40 cells, and its bootstrap
resampled the evaluation draw while holding the **fitted profiles fixed**, so it
excluded fitting-set variability. Each replicate here redraws the fitting *and*
evaluation populations from an independent seed. A quantity that survives eight
such redraws was not a selection artefact.

```bash
MPLCONFIGDIR=/tmp/mplconfig-hsmm python scripts/hsmm_confirm_cell.py
```

8 seeds x 3 null constructions, 150 fit / 150 eval traces, 8 random alignments,
1000 bootstrap replicates. 63 s, CPU-only.

## Result: confirmed under both surrogate constructions

Mean ± SD across eight independent redraws:

| Null | Admitted | E1 | E0 true | align+ | **E0 − E1** | spectral residual |
|---|---:|---:|---:|---:|---:|---:|
| AAFT | **8/8** | 0.648 ± 0.038 | 0.965 ± 0.017 | +0.390 ± 0.139 | **+0.318 ± 0.034** | 6.3e-3 |
| IAAFT | **8/8** | 0.644 ± 0.033 | 0.975 ± 0.016 | +0.431 ± 0.187 | **+0.331 ± 0.038** | 1.0e-3 |
| composite | 0/8 | 0.867 ± 0.016 | 0.980 ± 0.007 | +0.345 ± 0.131 | +0.113 ± 0.019 | — |

Every admission clause passes in every replicate for both surrogate arms. The
searched estimate (+0.331) reproduces at +0.318 (AAFT) and +0.331 (IAAFT).

**The result is not an artefact of the loose surrogate.** IAAFT reduces the
spectral residual 6x and *increases* the gap slightly, so the finding does not
depend on the spectral slack that one-pass AAFT leaves.

### The AAFT wording, now quantified

`fourier_phase_surrogate` (FT) matches the amplitude spectrum to 1.7e-16 but does
not match the marginal. One-pass AAFT restores the marginal exactly and leaves a
median relative spectral residual of **5-6e-3**; IAAFT iterates both constraints
to **1.0e-3**. So AAFT is *marginal-matched and approximately spectrum-matched*,
and is described that way in `matched_surrogate_population`'s docstring. Two
tests pin it: the AAFT spectral error must be non-zero, and IAAFT's must be
strictly smaller.

## The blocker: the composite null is degenerate at this record length

The composite arm fails on two clauses, and only one of them is about power:

- `estimation_gap_large_enough` passes 5/8 (gap 0.113 ± 0.019, straddling 0.10);
- `nulls_not_degenerate` passes **0/8** — **18.6% ± 2.3%** of composite-null
  traces yield no extractable events at all, against a 5% guard.

At 10 s the coloured and controller null families frequently produce no
paired down/up transition, so their feature vector is the all-zero fallback.
Any comparison denominated against the composite null *at this cell* is
therefore partly a comparison of empty vectors, exactly the artefact that killed
`T3_share0.35`.

This does not affect the surrogate arms (null degeneracy 2.0-2.2%), which is
where the E2 target lives. It does mean the third precommitted comparison below
cannot be run at 10 s as-is.

## E2 precommitment

Registered before any E2 code exists.

### Fixed normalisation

```
R = (AUC_E2 - E1) / (E0_true - E1)          "benchmark-gap recovery"
```

Frozen on the **confirmation** means, not the searched point (the searched
0.661/0.993 were pointwise-after-search; the confirmed equivalents are the
honest ones):

| Arm | E1 | E0 true | gap |
|---|---:|---:|---:|
| AAFT | 0.6477 | 0.9653 | **0.3177** |
| IAAFT | 0.6437 | 0.9750 | **0.3313** |

`R` is *not* a fraction of an oracle ceiling: `E0` is specific to this
trace-level feature family, and a sequence model could legitimately exceed it, so
`R > 1` is admissible and is not evidence of a bug.

### The three comparisons

1. **E1 vs the full explicit-duration HSMM**, on the surrogate null. Establishes
   whether the blind estimation gap is recoverable at all.
2. **Full HSMM vs a duration-neutral ablation** — an otherwise identical
   geometric-duration HMM, plus a duration-shuffled control. Without this, an
   improvement cannot be attributed to *explicit durations* rather than to the
   sequence model or to extra capacity.
3. **Both models against the physical composite null.** Improvement only against
   the surrogate supports a phase-coupling claim, not an operational
   training-detection claim.

   **Blocked as specified** by the degeneracy above. Options, to decide before
   running: lengthen the record for this comparison only (accepting that E1 rises
   and the gap shrinks), restrict the composite null to the families that remain
   non-degenerate at 10 s, or give `summarize_events` a principled
   no-events representation instead of the all-zero fallback. The third is the
   real fix and would also retire a latent artefact in every earlier campaign.

### Success criterion

Roughly **R >= 0.50** together with a clear win over the duration-neutral
ablation. Absent the second, a high `R` is not evidence for explicit durations.

## The corrected conclusion

> Alignment is unidentifiable at the paper's saturated 90-second operating point,
> but substantial local event information exists in a short, diluted,
> statistically hard regime. That regime provides one legitimate, narrowly scoped
> E2 target.

Enough to build E2. **Not** enough to broaden the paper's operational claim: the
target is denominated against a statistical surrogate null, at 10 s records, with
event recall of 0.195, and the physical-null comparison that would license an
operational reading is currently blocked.
