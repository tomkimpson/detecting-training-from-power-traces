# Duration-aware event detector — E0/E1 smoke findings

**Date:** 2026-08-07  
**Branch:** `feat/hsmm-renewal-smoke`  
**Status:** **GO to E2 — WITHDRAWN 2026-08-07.** Viability smoke only; not a frozen
paper result.

> **CORRECTION (2026-08-07).** The E0 gate below is **invalid**. It compared
> oracle-windowed positives against *blind*-extracted nulls, and several oracle
> features (`paired_fraction`, hardcoded to 1.0; `duration_median_s`;
> `duration_cv`) are read off the generator's latent schedule — so the oracle
> scores AUC 1.000 on a pure-noise trace. Every "oracle AUC 1.00" claim in this
> note measures metadata, not channel information. See
> [`hsmm-e0-defect-and-fair-ceiling.md`](hsmm-e0-defect-and-fair-ceiling.md) for
> the defect, the fix, and the fair ceiling. The blind E1 results in this note are
> unaffected; `oracle_auc_recovered_by_blind` is conservative but misnamed.

## Question

When work variation or fast drift destroys a coherent global cadence, does the
observed trace retain local compute-to-communication events that a duration-aware
model can recover without knowing the iteration boundaries?

The predeclared design and kill criteria are in the provisional extension to
`spec.md`. This campaign implements:

- **E0:** an oracle-aligned information ceiling using evaluation-only phase metadata;
- **E1:** blind down-transition/recovery extraction plus a profiled marked-renewal
  likelihood; and
- a shared composite-null recalibration of Viterbi and the two tracked-order
  baselines.

The explicit-duration sample-level HSMM (**E2**) is not implemented yet. E0/E1 exist
to decide whether it is worth implementing.

## Run

- Records: 90 s at 20 Hz.
- Fitting: 50 traces per positive condition and per null family.
- Calibration: 200 traces per null family.
- Evaluation: 200 traces per positive condition and per null family.
- RNG streams: deterministic and disjoint for fitting, calibration, and evaluation.
- Operating point: the most conservative per-family empirical threshold at composite
  FAR 0.05.
- Positives: honest, work 0.5, work 0.7, drift 0.8 Hz, drift 1.5 Hz.
- Nulls: hard inference, AR(1), resonant AR(2), controller, controller + AR(1), and
  periodic inference.

Reproduce:

```bash
MPLCONFIGDIR=/tmp/mplconfig-hsmm python scripts/smoke_hsmm.py
```

Outputs: `results/hsmm_smoke/summary.json`, `score_distributions.png`, and
`composite_tpr.png`.

## Result: both prerequisite gates pass

E0 is saturated: oracle AUC is 1.00 against the union and against every individual
null for every positive condition. In particular, `work=0.7` clears the predeclared
minimum AUC 0.90.

The blind event extractor also passes. Communication-event recall and trace-level
power are:

| Positive | Event recall | Blind AUC, null union | Blind TPR @ composite FAR 0.05 |
|---|---:|---:|---:|
| Honest | 0.855 | 1.000 | 1.000 |
| Work 0.5 | 0.994 | 1.000 | 1.000 |
| Work 0.7 | 0.992 | 1.000 | 1.000 |
| Drift 0.8 Hz | 0.801 | 1.000 | 1.000 |
| Drift 1.5 Hz | 0.727 | 1.000 | 1.000 |

Median recovered transition timing error is 23--25 ms in every cell. On `work=0.7`
the blind detector recovers all of the oracle's excess AUC over chance, clearing the
predeclared 70% criterion.

## Shared-calibration comparison

At the same composite-null threshold:

| Positive | Renewal | Viterbi | DG order full | DG order semi-coherent |
|---|---:|---:|---:|---:|
| Honest | 1.000 | 0.000 | 0.445 | 0.685 |
| Work 0.5 | 1.000 | 0.000 | 0.000 | 0.000 |
| Work 0.7 | 1.000 | 0.000 | 0.000 | 0.000 |
| Drift 0.8 Hz | 1.000 | 0.000 | 0.000 | 0.000 |
| Drift 1.5 Hz | 1.000 | 0.000 | 0.000 | 0.000 |

The existing baselines' zeros should not be confused with their previously reported
inference-only power. The composite threshold must also reject the controller and
periodic-inference nulls: the controller sets a high threshold for raw Viterbi, while
periodic inference sets a high threshold for the tracked-order scores. This is exactly
the joint-decision gap the experiment was built to expose.

The renewal detector's worst TPR over `work=0.7`, drift 0.8, and drift 1.5 is 1.00;
the best existing detector chosen separately in each cell has worst TPR 0.00. The
predeclared improvement requirement of 0.15 is therefore passed.

## False alarms: promising, not calibrated

Held-out evaluation false-alarm rates for the blind renewal score are:

| Null | Evaluation FPR |
|---|---:|
| Hard inference | 0.080 |
| AR(1) | 0.010 |
| Resonant AR(2) | 0.000 |
| Controller | 0.015 |
| Controller + AR(1) | 0.020 |
| Periodic inference | 0.010 |

Inference rises from 0.05 on the calibration split to 0.08 on evaluation. This is
within the predeclared smoke tolerance of 0.10, but it prevents any claim that the
profiled score is already level-exact at 0.05. The appropriate conclusion is method
viability and a GO to E2, not calibrated deployment.

## Does it merely reconstruct cadence or exploit amplitude?

Two prespecified-after-inspection diagnostic ablations address the most obvious
shortcut explanations.

1. **Local-only:** zero `event_rate_hz` and `interval_cv`. AUC remains 0.9997--1.00
   and TPR remains 1.00 in every cell. The result is not carried by the mean cadence
   or interval regularity.
2. **Shape/duration only:** additionally zero the median event amplitude and amplitude
   CV. TPR is 1.00, 1.00, 0.99, 0.97, and 0.84 across the five positive conditions;
   AUC is 0.994, 0.994, 0.994, 0.989, and 0.983. Controller FPR remains 0.015.

The stricter ablation is therefore still useful at the far drift edge. Separation is
present in dwell duration, paired-transition balance, plateau slope/noise, and
within-trace shape repeatability rather than only in rate or absolute dip size.

## Verdict

The saved gate record is:

- E0 oracle continue: **true**;
- E1 blind continue: **true**;
- evaluation FPR within smoke tolerance: **true**;
- worst-hard-cell renewal TPR: **1.00**;
- worst-hard-cell improvement: **1.00**;
- cadence-free local-only worst-hard-cell TPR: **1.00**;
- cadence-and-amplitude-free shape/duration worst-hard-cell TPR: **0.84**; and
- overall recommendation: **GO to E2**.

## Boundaries and next test

This is intentionally an optimistic, in-distribution synthetic smoke:

- the positive profile was fitted on the same named attack families later evaluated,
  although on disjoint traces;
- the model is trained on the scenario generator and is not corpus-free;
- the 90 s / 20 Hz channel is easier than the meter boundary;
- perfect AUC reflects how separable this generator is, not real distributed
  hardware; and
- a full Gaussian feature profile is not itself an HSMM or a calibration theorem.

The next justified step is E2: a two-state explicit-duration forward likelihood whose
score marginalises over segmentations. Its first falsification should be
leave-one-attack-family-out evaluation, followed by the full ST2 attack grid and the
meter sampling/integration boundary. Only after that should this direction be tested
on external measurements of real distributed all-reduce; the existing single-A100
trace remains an expected negative transport control.
