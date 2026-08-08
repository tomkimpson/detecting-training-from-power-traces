# spec.md — certifying-training-from-power

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
definitive**). See `handoff.md` for current status and `notes/results/st1-findings.md` for the
ST1 gate record. Phases 1–4 are complete; the manuscript is drafted in full.

---

## Provisional extension: duration-aware event detection

**Status:** approved prototype / smoke test, not yet part of the paper's claimed
method or evidence.

**Branch:** `feat/hsmm-renewal-smoke`.

### Motivation

The current detector family has complementary boundaries. The raw Viterbi ridge
score retains power when cadence wanders rapidly or real work varies, but a strong
controller limit cycle can score more highly than training. The tracked
cyclostationary/order statistics reject that structural confuser at modest wander,
but lose power at the fast-wander end of the frontier. We have not demonstrated one
jointly calibrated decision rule that inherits both strengths across the full
frontier.

The proposed extension tests a different physical invariant. Under the admissible
training family, an iteration contains a compute phase followed by a synchronous
communication/optimizer phase. Varying useful real work changes the compute-phase
duration, while the communication/down-phase duration retains its honest
distribution. The resulting global cadence may be incoherent even though a marked
sequence of local compute-to-communication events remains observable.

### Hypothesis

> Under work variation and cadence drift, externally observed training traces retain
> locally repeatable compute-to-communication events whose shape and duration allow a
> duration-aware event model to distinguish training-shaped multi-phase execution
> from hard inference and controller-like structural nulls without access to the
> generator's phase boundaries at decision time.

The intended detector is a marked-renewal model or an explicit-duration hidden
semi-Markov model (HSMM), not a frequency tracker. Its evidence should arise from
local state transitions, dwell durations, event marks, and their ordered recurrence,
not from a single coherent spectral ridge.

### Claim and information cost

If successful, this extension strengthens Rung 1 only:

> The trace contains recurrent, ordered, multi-phase physical execution consistent
> with compute/communication alternation under a specified composite structural null.

It does not identify training semantics. Discarded-update training, a
training-shaped non-ML loop, and any sufficiently faithful semantic decoy remain
expected positives. The detector may use profiled distributions learned from the
scenario model; that additional modelling information must be stated and must not be
described as corpus-free.

### Load-bearing physical prediction

The prototype exists first to test one prediction, before a general HSMM is built:

1. work variation broadens compute/up-phase durations;
2. communication/down-phase durations retain a distinguishable distribution;
3. the external observation retains a repeatable local dip/transition around the
   communication phase; and
4. that local evidence survives at the `work=0.7` and fast-drift cells where the
   existing detector families cease to provide one jointly strong rule.

If ground-truth-aligned communication events are not separable at those cells, the
HSMM direction is stopped: a blind state estimator cannot recover information that is
absent even under oracle alignment.

### Three-gate prototype

#### Gate E0: oracle-aligned information ceiling

Expose evaluation-only phase metadata from the scenario generator:

- iteration/compute-phase start;
- communication/down-phase start;
- compute duration;
- communication duration; and
- their locations after the observation map is applied.

This metadata may be used to evaluate and profile an oracle statistic, but must never
be passed to a deployable detector score.

For each true communication start, extract a fixed physical-time window from the
observed power trace. Robustly remove its local baseline and scale, then estimate on a
disjoint fitting split:

- a mean communication-event template;
- the distribution of transition amplitude and slope;
- the distribution of low-state dwell time; and
- within-trace event repeatability.

Score held-out traces by their aligned template likelihood/repeatability and compare
with randomly aligned or best-matched windows from each null. E0 asks only whether
the proposed local information is present in the observed channel.

**E0 continue criterion:** oracle AUC at least `0.90` for `work=0.7` against both the
hard inference null and the controller-like composite null on held-out traces.

**E0 kill criterion:** oracle AUC below `0.80` in either comparison. Values in
`[0.80, 0.90)` are an honest marginal result and require inspecting event SNR and
meter sensitivity before E1.

#### Gate E1: blind marked-event detector

Build the smallest detector that does not use phase metadata or a nominal cadence:

1. robustly detrend and scale the trace;
2. smooth at one fixed, physically expressed bandwidth;
3. obtain candidate downward and upward transitions from matched edge/dip filters;
4. pair a downward transition with a subsequent recovery under broad physical
   duration limits; and
5. attach event marks: dip amplitude, dwell duration, downward/upward slope, local
   template similarity, local variance change, and time since the previous event.

The first E1 score may be a fixed interpretable combination of:

- credible paired-event rate;
- median held-out template similarity;
- communication-duration consistency;
- ordered down/up pairing fraction; and
- a penalty for unmatched transitions.

A marked-renewal likelihood is preferred once the plumbing works:

```
S_event(x) = log p({event times, durations, marks} | train-event model)
             - max_j log p({event times, durations, marks} | null model j)
```

Ground-truth metadata is used only for diagnostics: event recall, precision, and
transition timing error. It is forbidden from the scoring interface and this
separation must be covered by tests.

**E1 continue criterion:** on `work=0.7`, recover at least `70%` of true
communication events within `100 ms` at the nominal 20 Hz channel, and retain at
least `70%` of the oracle's excess AUC over chance:

```
(AUC_blind - 0.5) / (AUC_oracle - 0.5) >= 0.70.
```

Failure of the timing diagnostic alone is not decisive if the trace-level blind score
passes: multiple equivalent segmentations may support the correct structural
decision.

#### Gate E2: explicit-duration HSMM

Build E2 only if E0 passes and E1 is non-trivial. The minimum alternative model has
two mandatory-alternating states:

```
COMPUTE -> COMMUNICATION -> COMPUTE -> ...
```

It uses robust emissions for normalized power and local slope, a broad compute-state
duration distribution, and a separate communication-state duration distribution.
The decision statistic is the forward marginal likelihood over all admissible
segmentations, not the maximum-likelihood state path.

The composite-null denominator must include, at minimum:

- hard continuous-batching inference;
- stationary coloured and resonant processes;
- a controller limit cycle;
- controller plus coloured noise; and
- periodic inference.

The provisional score is

```
S_HSMM(x) = log p(x | M_train) - max_j log p(x | M_null,j).
```

For the first smoke test, emissions and duration laws may be profiled from a disjoint
scenario-model fitting split using latent labels. This deliberately gives the model
its best plausible chance. A later experiment must distinguish this profiled detector
from a self-fitted or corpus-light structural test.

### Smoke populations

The minimum hard-cell grid is:

| Role | Population | Reason |
|---|---|---|
| Positive | honest training | no-regression anchor |
| Positive | `work=0.5` | confirmed approximately zero-cost work variation |
| Positive | `work=0.7` | Viterbi bend / primary event-model target |
| Positive | `drift=0.8` | tracked-order fast-wander boundary |
| Positive | `drift=1.5` | extreme fast-wander stress |
| Null | hard inference | stated Rung-2 null |
| Null | AR(1) / resonant AR(2) | coloured structural nulls |
| Null | controller | Viterbi's principal confuser |
| Null | controller + AR(1) | confuser plus coloured background |
| Null | periodic inference | periodic non-training computation |

If this grid passes, expand to phase slip, harmonic smoothing, dilution, amplitude
shaping, and the meter boundary. `shape_fill_frac=1` and the one-second-integrating
1 Hz meter are expected information-erasure points, not required successes.

Semantic controls are reported separately. The discarded-update decoy and
training-shaped non-ML loop are expected positives and are not part of the
composite structural null.

### Data separation and calibration

Use deterministic, disjoint RNG streams for:

1. model/template fitting;
2. null calibration; and
3. untouched evaluation.

The local plumbing smoke may use short records and tiny populations, but cannot freeze
scientific numbers. The viability run uses:

- 60--90 s records initially, followed by the canonical 300 s record if the method
  passes;
- at least 50 traces per fitted population;
- 200 traces per null family for calibration; and
- 200 held-out traces per evaluation population.

Only FAR `0.05` is interpreted in the viability smoke. FAR `0.01` has inadequate
tail resolution until a larger calibration campaign is run.

The event/HSMM score, Viterbi, and tracked-order baselines must all be calibrated and
evaluated on the same populations. For a finite union of null families, the operative
threshold is the most conservative per-family empirical threshold. No previously
reported detector rate obtained under an inference-only or controller-only threshold
is used as a direct comparison.

### Overall go/no-go decision

The extension is a **GO** if, at composite-null FAR `0.05` on untouched evaluation
traces:

- honest-training TPR is at least `0.90`;
- no individual calibrated null family exceeds the allowed false-alarm rate beyond
  finite-sample uncertainty;
- `work=0.7` or the fast-drift cells improve worst-cell TPR by at least `0.15` over
  the better of Viterbi and tracked-order baselines under the same composite
  calibration, or otherwise produce a material improvement in worst-cell power;
- controller rejection is retained rather than traded for work-variation power; and
- the score consumes only `(t, P_obs)` at decision time.

The extension is a **NO-GO** if it merely rediscovers global cadence, cannot reject a
controller, requires latent boundaries at decision time, or has no oracle-aligned
information at the hard frontier cells.

### Observation-channel check

If the nominal test passes, repeat E1/E2 at 20, 10, 5, and 2 Hz and with integration
windows of 0, 0.1, 0.25, and 0.5 s. The method's meter requirement is reported even if
it is stricter than the existing tracking detector's requirement. A method that works
only on the pristine 20 Hz channel is scientifically interesting but operationally
weak.

### Implementation and outputs

The prototype should add:

- a non-breaking phase-metadata interface beside `training_F_meta`;
- `powerladder/typeb/renewal.py` for preprocessing, event extraction, marks, and the
  initial trace score;
- `powerladder/typeb/hsmm.py` only after the E0/E1 gates justify it;
- `scripts/smoke_hsmm.py` for the disjoint smoke campaign;
- deterministic unit tests that prove detector functions accept no latent metadata;
  and
- results under `results/hsmm_smoke/`, never overwriting frozen ST1/ST2 artifacts.

The first report contains:

1. oracle and blind score distributions by population;
2. event recall and timing error versus work variation;
3. TPR versus work level at a shared composite-null FAR `0.05`;
4. the same operating point for Viterbi and tracked-order baselines; and
5. an explicit E0/E1/E2 go/no-go verdict.

### Interpretation limit

Synthetic success would establish only that a duration-aware detector extracts the
local event information assumed by the scenario model. It would not validate the
physical prediction that an external meter sees a distributed all-reduce phase. The
decisive transport experiment remains simultaneous external measurement of real
multi-GPU or multi-node distributed training. The existing single-A100 data is an
expected negative transport control because it lacks a substantial collective
communication phase.
