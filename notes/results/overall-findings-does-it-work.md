# Does it work? — overall findings synthesis

**Date:** 2026-07-24 · **Status:** synthesis of the frozen Phase-0→2 results (the
NP-optimal ceiling leg is freezing on slurm at time of writing; its qualitative
direction is included, its numbers are not yet frozen). **Scope of truth:** `spec.md`;
per-result detail in the notes cross-linked below. This is a state-of-play record, not a
tracker — status lives in `tasks.md`.

## The question

If a verifier has **only an external power measurement** of an accelerator, can it detect
that the device is **training** (vs running inference) — and does that hold **against an
adversary** who wants to hide the training? This synthesises what we have found on
**synthetic data** (the Ko et al. workload model plus our inference null and attack
knobs; CPU-only; single-GPU cost anchors are the only measured input).

## Bottom line

Power-only detection of a training **schedule** works well — and, per the ceiling leg,
close to the theoretical optimum — and it survives a *mildly* adaptive adversary. It is
**not** an adversarially-robust *identification* of training. Read it as **strong physical
evidence and necessary conditions, plus an upper bound on hideability — not proof.** The
work is deliberately a **claim ladder**: the answer is different at each rung, and each
rung's ceiling motivates the next.

## What works

1. **Detecting the honest training line — solid (ST1).** The tracked-cyclostationary
   pipeline (Viterbi line-tracker + Dandawaté–Giannakis order statistic) is level-safe on
   every stationary null tested (white, AR1, controller limit cycles, heavy-tailed, …) and
   separates a training iteration-line from structured confusers. Only the DG *order*
   family separates a wandering line from a controller limit cycle. Honest power boundary:
   a line wandering faster than ≈0.8 Hz over 300 s. (`st1-findings.md`.)

2. **Training-vs-inference classification — near-perfect on the stated problem (Rung 2).**
   All three decision rules reach AUC 1.00 / FNR 0 at FPR 0.05 against the hard inference
   null, and it **transfers**: different power scale, shorter traces, shifted cadence band,
   a hostile controller-cycle meter, heavy coloured noise — AUC stays 1.00.
   (`rung2-findings.md`.)

3. **Robust to mild de-periodicisation — the key "survives obfuscation" result.** An
   adversary who *smears* the line (frequency drift, Brownian phase slip) does **not**
   evade detection: async/de-periodicised genuine training is still caught (AUC 0.89),
   because the tracker was built to follow a wandering line. The NP-optimal **Whittle
   ceiling** (freezing now) makes this quantitative: the *achievable* detection power stays
   ≈1.0 across the whole drift axis; a fixed matched filter throws almost all of it away
   under drift, while the tracker recovers essentially all of it. So the method is not just
   working — it is close to optimal on the honest-schedule problem.
   (`st1-np-ceiling-findings.md`.)

## The three boundaries (what it can't do / can't claim)

1. **A near-free work-variation attack defeats schedule detection (ST2 frontier — the
   definitive result).** Varying the *real work per iteration* (`work_σ → 0.5`) moves the
   cadence **without leaving a coherent line to track**, and collapses *every* detector —
   fixed and tracking — at **≈zero measured throughput cost**. This is the genuine scope
   limit. The counterweight is a **theoretical** floor (the lower-bound spike): hiding the
   synchronous all-reduce is communication-bound, so reaching true statistical covertness
   must cost **throughput or learning efficiency** — the free escape is bounded by physics.
   But the learning-efficiency leg is **argued, not measured** (GPU campaign descoped), so
   "the adversary can't hide for free" is currently a proposition, not a closed experiment.
   (`st2-frontier-freeze-findings.md`; lower bound: `../discussion/lower-bound-feasibility-spike.md`.)

2. **It needs a fast-enough meter (minimum meter specification).** "Just power" is not
   sufficient at any bandwidth. A 1 Hz integrating sampler erases all detectors — a channel
   requirement, not an attack. The spec: sample rate ≥ ~2 Hz **and** integration window
   ≲ 0.5 s **and** no deep in-band notch at the cadence. Below that, nothing works (Nyquist
   crosses the cadence band). (`st2-meter-boundary-findings.md`.)

3. **It certifies the physical schedule, not training semantics (Rung 2 controls).** Five
   *non-training* loads — a discarded-update decoy, a gradient-only forward/backward, a
   training-shaped non-ML kernel loop, a power-controller limit cycle, a periodic inference
   request generator — **all score as training**. From power alone you can say "there is a
   training-shaped periodic compute schedule," not "this is gradient-descent training rather
   than a mimic." Causal attribution needs the challenge-synchronous protocol (Rungs 3–4),
   which is **specified but not empirically run**. (`rung2-findings.md`.)

## Summary table

| Question | Answer |
|---|---|
| Detect an honest training schedule from power? | **Yes** — reliably, near-optimally |
| Survive mild obfuscation (jitter / drift / phase-slip)? | **Yes** — tracking holds |
| Survive a determined adversary (work-variation)? | **No** at the schedule level — but hiding provably costs throughput/learning (floor argued, not measured) |
| Work with any power meter? | **No** — needs ≥ ~2 Hz sampling, integration ≲ 0.5 s |
| Prove it is *actually training* (not a mimic)? | **No** at Rungs 1–2 — certifies the schedule, not semantics; needs the (unrun) challenge protocol |

## The claim ladder (how the paper frames all of the above)

- **Rung 1** — *evidence of iteration-structured cyclicity* (not identification). Input:
  external time-resolved trace only. Ceiling: a controller can mimic the structure.
- **Rung 2** — *more-training-like than a stated inference null* (a specified statistical
  task). Input: a representative/modelled inference null. Ceiling: certifies the schedule,
  not the semantics (the five controls pass).
- **Rungs 3–4** — *challenge-synchronous attribution* (conditional protocol). Input: prover
  participation + a challenge-bound schedule. Stated as ceilings; not empirically run.

Each rung answers a strictly more governance-relevant question at a stated cost in verifier
information; each rung's hard ceiling motivates the next. (`../plans/plan-for-paper-2.md`.)

## Standing caveats

All results are **synthetic**; the only measured input is the single-A100 systems-cost
anchors (`data/measured_cost_anchors/`). Learning-efficiency is **stated, not measured**
(the transport trap was deliberately avoided). Rungs 3–4 are a conditional-protocol
section, not results. The NP-optimal ceiling is optimal **under the Whittle
(stationary-Gaussian) model** only. The honest overall framing is *evidence + necessary
conditions + an upper bound on hideability* — not identification.

## Cross-links

- ST1 detector null-validity + power boundary: `st1-findings.md`
- ST2 de-periodicisation frontier (the work-variation boundary): `st2-frontier-freeze-findings.md`
- Minimum meter specification: `st2-meter-boundary-findings.md`
- Rung 2 classification + semantic controls: `rung2-findings.md`
- NP-optimal Whittle ceiling: `st1-np-ceiling-findings.md`
- Lower-bound / hideability floor: `../discussion/lower-bound-feasibility-spike.md`
- Plan + claim ladder: `../plans/plan-for-paper-2.md`, `../plans/plan-for-paper-2-review.md`
