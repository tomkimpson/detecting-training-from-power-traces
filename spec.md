# spec.md — detecting-training-from-power-traces

**Source of truth for project scope.** Do not modify without explicit approval. The
detailed realisation (methods, staged plan, wording discipline) lives in
[`notes/plans/plan-for-paper-2.md`](notes/plans/plan-for-paper-2.md) and its
[review](notes/plans/plan-for-paper-2-review.md); this file fixes *what the project is*.

## Thesis (the sentence the paper defends)

> A passive external power trace can provide calibrated structural evidence and
> conditional evidence of efficient iteration-structured training, but not universal
> semantic identification. An adaptive prover can attack the exposed cadence; the
> empirical question is the detection–utility frontier required to erase it. Stronger
> work verification is possible only conditionally on additional challenge and
> transcript/binding primitives.

## The question

For AI-compute governance, what can a verifier certify about **training** from a
**time-resolved external power trace** of untrusted hardware — with what methods, at
what cost in verifier information, and against an adversary who knows the detector?

## Scope discipline

- **Time-resolved power trace only.** The integrated-energy "Amount" channel (how much
  compute, energy-per-token pricing) is **excluded by decision**.
- **Theory and methods.** Evidence is on a **literature-parameterized scenario model**,
  not calibrated to measured hardware. A single GPU cannot test the distributed
  question; real hardware enters only as a **negative transport case** and as measured
  cost anchors. Distributed synchronization is a **falsifiable prediction** (it should
  make the cadence deep, coherent, legible), not a result claimed here.
- **Target population:** efficient, synchronous, iteration-structured training that
  exposes a recoverable cadence. De-periodicising that cadence is an **adversarial
  action with a measurable budget**, not a counterexample to the method.

## The claim ladder (the spine)

Each rung answers a strictly more governance-relevant question at a stated cost in
verifier information; each rung's hard ceiling motivates the next. Within each rung,
two adversary settings: **benign/non-adaptive** and **adaptive/hiding**. This is
*analogous in spirit* to an attestation ladder but is **not** the `β(I)` object of the
Amount problem — it is a ladder of claims and assumptions.

| Rung | Claim | Additional requirement | Hard ceiling |
|---|---|---|---|
| **1. Structural evidence** | Order-coherent power variation exists under a specified structural null | External time-resolved trace only | Periodic physical confounds; structural-null misspecification; superposition/dilution; active power shaping; cadence moved outside the searched family; die-to-meter attenuation |
| **2. Conditional classification** | The trace is more training-like than the stated inference population | A representative or explicitly modelled inference null | Domain shift; the semantic decoy (training-shaped execution without retained training) |
| **3. Active physical authentication** | A load behind the meter responded to an unpredictable verifier challenge | Prover participation; a challenge-bound schedule | A challenge-aware dummy load; no semantic binding by power alone |
| **4. Work verification** | The physical response is tied to declared training state transitions | A sound, fresh transcript/proof + explicit binding assumption | The remote-proof / co-location gap, unless the primitive closes it |

**Scope decision:** Rungs 1–2 are the empirical spine; Rungs 3–4 are a concise
conditional-protocol section (assumed cryptographic primitive stated explicitly).

## The observation channel

```
P_meter(t)  = (h * P_device)(t) + P_other(t) + η(t)
P_device(t) = r(t) F(t) + P₀(t)
```

`h` collects power-delivery dynamics, meter integration, reporting filters, and
sampling; the LTI form is an explicit first approximation. The synthetic evaluation
sweeps bandwidth/integration, sample cadence/aliasing, transfer-function notches,
coloured noise, controller interference, superposition, and time-varying baselines.

## Scenario-model provenance (four layers, stated explicitly)

1. **Ko & Zhu (2025)** — training/fine-tuning waveforms and their aggregate superposition.
2. **Our inference null** — literature-motivated continuous-batching decode structure,
   slow MoE envelope, aperiodic prefill bursts (not Ko's). Rung-2 conclusions are
   conditional on this stated null.
3. **Our additions** — slow frequency wander and all adversary transformations.
4. **Our observation map** — workload waveform → power/meter observation model.

The generator is **literature-parameterized, not calibrated** (a ~75× gap separates
the Ko-scale modulation from the measured single-GPU cadence).

## Methods (at a glance)

- **Rung 1 benign:** multitaper harmonic F-test (approximately pivotal under a
  stationary, locally smooth spectral null; needs a search correction).
- **Rung 1 adaptive:** tracker (`f̂₀(t)`) → phase resampling (tacholess order tracking)
  → Dandawaté–Giannakis cyclostationary `Q_α` on the de-warped signal. Null validity of
  the *complete* pipeline was the ST1 gate question; **outcome: no analytic CFAR** — the
  pipeline is an **effective score with per-trace surrogate calibration** (level-exact
  at the tested operating points on stationary nulls). "Calibrated" in the thesis means
  this surrogate calibration.
- **Rung 2:** fixed interpretable feature vector; three decision rules compared
  (prespecified physics score; model-based discriminant; flexible learned reference);
  semantic falsification controls evaluated.
- **Central result:** the **de-periodicisation frontier** — how far a prover must
  de-periodicise useful training before the meter loses it, at what measured systems
  cost. The plausible learning-efficiency penalty is stated as an open empirical
  question (future work), not measured here.
- **Channel requirement:** the observation channel is a first-class condition on every
  rung — a 1 s-integrating, 1 Hz-reporting meter defeats all detectors on an
  *unmodified* honest workload (order methods to chance). The paper reports the
  minimum meter specification (sampling, integration window, in-band transfer
  function) under which Rungs 1–2 are available at all.

## Explicitly out of scope / deferred

- Anytime-valid / sequential monitoring (fixed-horizon results only; no claim the χ²
  statistic repairs the e-process).
- The Amount / energy-accounting channel.
- Designing the cryptographic binding primitive (assumed as an ideal functionality at
  Rung 4, not built).
- Real distributed hardware (a falsifiable prediction + the single-GPU negative
  transport case only).
- Measured learning-efficiency penalties of de-periodicisation (GPU campaign) —
  deferred to future work (decision 2026-07-22).

## Status

Phase 0 de-risking gates are **complete** (ST0 desk work; ST1 adaptive-detector null
validity = **GO, honest reframe**; ST2 de-periodicisation frontier = **GO,
definitive**). See `tasks.md` for the phased tracker and `notes/results/st1-findings.md` for the
ST1 gate record. Phase 1 (Rung 1 full) is next.
