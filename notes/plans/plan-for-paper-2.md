# Plan for Paper 2: What a passive power meter can certify about training

**Status:** planning, 2026-07-21 (revised same day per
[`plan-for-paper-2-review.md`](plan-for-paper-2-review.md)). Supersedes the
single-method framing of `paper-verification/main.tex` as the direction for the
*next* paper.

**Source of truth:** `spec.md` (the project scope — identifiability, the floor `β`,
the attestation ladder, Amount vs Type). This plan is a realisation of that spec,
not a replacement for it. Rationale and the full menu of options live in
`notes/discussion/power-verification-paths-forward.md`; **status and the concrete plan live
here**.

**Thesis (the sentence the paper defends):**

> A passive external power trace can provide calibrated structural evidence and
> conditional evidence of efficient iteration-structured training, but not universal
> semantic identification. An adaptive prover can attack the exposed cadence; the
> empirical question is the detection–utility frontier required to erase it. Stronger
> work verification is possible only conditionally on additional challenge and
> transcript/binding primitives.

---

## 1. Big picture

### 1.1 Why theory/methods, not hardware

We want to detect training from an off-chip power trace, for AI-governance
verification. The honest constraint is that **we do not have hardware that can
test the real question**:

- We have a single GPU, not a cluster. The iteration cadence *exists* on a single
  card — the measurements find a faint corresponding ripple (~0.4% of board power;
  issue #54) — but on one saturated card there is no communication phase, so the
  modulation is shallow and easily washed out (this is why the measured Viterbi
  headline was retired). **Distributed synchronization is expected to make the
  cadence signature deep, coherent, and legible** — not to create it from nothing.
- A single-node rig matches neither a real datacenter nor the aggregate regime of
  the workload model we adopt. Trying to carry the paper on it would repeat the #54
  mistake.

So this paper is deliberately a **theory-and-methods** contribution: what can (and
cannot) be certified in principle from time-resolved power, with what methods, and
against an adversary who knows the detector. All evidence is on a
**literature-parameterized scenario model** (we do not claim "calibration" — the
~75× gap between the Ko-scale modulation and the measured single-GPU cadence in
`../results/issue54-investigation.md` shows the generator is *not* calibrated
to the measured channel). Real hardware enters as a **negative transport case**
(§4.8), not a pillar.

**The scenario model and its provenance.** Four layers, each with its own owner,
stated explicitly in the paper:

1. **Ko & Zhu (2025, `ko2025widearea`)**: the training and fine-tuning waveforms and
   their aggregate superposition (built for a grid-stability question; we adopt the
   forward power model to ask a verification question). Superposition is native to
   it — many jobs at a shared feed — which is one of our adversary axes.
2. **Our inference null**: the continuous-batching decode structure, slow MoE
   envelope, and aperiodic prefill bursts are **our literature-motivated
   construction, not Ko's** (stated in `code/ko_workload.py`). The
   inference-dominant aggregate that substitutes our inference process into Ko's
   dominant slot is likewise ours.
3. **Our additions**: slow frequency wander and all adversary transformations.
4. **Our observation map**: workload waveform → power/meter observation model (§2).

Suggested paper wording: *"We adopt Ko and Zhu's training, fine-tuning, and
aggregate workload model. We construct a literature-motivated continuous-batching
inference null in the same power and superposition framework. Level 2 conclusions
are conditional on that stated null."*

**Scope discipline (decided).** The paper **uses the time-resolved power trace and
excludes the integrated-energy Amount channel** — no "how much compute" bounding, no
energy-per-token pricing as a verification instrument. (Not "everything lives in the
spectrum": cyclostationary structure is expressed spectrally, but challenge
correlation is a time-domain operation; the underlying observable is the trace.)
Adversary cost is characterised quantitatively where we can measure it (§5) and
qualitatively otherwise.

### 1.2 The target population: efficient iteration-structured training

The paper does not claim a recoverable line is universally necessary for all
training — some conceivable execution can be made fully aperiodic, and organising
the paper around that pedantry would be unhelpful. Instead:

> We target efficient, synchronous, iteration-structured training that exposes a
> recoverable cadence — an expected physical signature of that regime, especially
> when distributed synchronization makes the modulation deep and coherent. An
> adaptive prover may de-periodicise its schedule; we **measure the
> detection–utility frontier** as that randomisation increases (§5).

Destroying the cadence is thus an *adversarial action with a measurable budget*,
not a counterexample to the method.

### 1.3 The spine: the claim ladder × two adversary settings

The paper is organised as a **claim ladder** (equivalently, verification ladder): a
ladder of claims, each answering a strictly more governance-relevant question at a
stated cost in verifier information. It is **analogous in spirit** to `spec.md`'s
Amount attestation ladder `β(I)` — more verifier information permits a stronger
claim at greater institutional cost — but it is *not* the same mathematical object
(`β(I)` is a quantitative bound on unidentified covert compute; this is a ladder of
claims and assumptions), so we do not call it `β(I)`.

Within each rung we run two adversary settings: **benign / non-adaptive** and
**adaptive / hiding**. (Not "Q1/Q2" — `spec.md` uses Q1/Q2 for Amount/Type and the
collision would confuse repository readers.)

| Rung | Claim | Additional requirement | Hard ceiling |
|---|---|---|---|
| **1. Structural evidence** | Order-coherent power variation exists under a specified structural null | External time-resolved trace only | Periodic physical confounds (controllers, cooling, mains, co-resident loads); structural-null misspecification / generic nonstationarity; strong superposition or dilution; exact active power shaping; cadence moved outside the searched family; attenuation or a notch in the die-to-meter transfer function |
| **2. Conditional classification** | The trace is more training-like than the stated inference population | A representative or explicitly modelled inference null | Domain shift; the **semantic decoy** (training-shaped execution without retained training) |
| **3. Active physical authentication** | A load behind the meter responded to an unpredictable verifier challenge | Prover participation; a challenge-bound schedule | A challenge-aware dummy load; no semantic binding by power alone |
| **4. Work verification** | The physical response is tied to declared training state transitions | A sound, fresh transcript/proof and an explicit binding assumption | The remote-proof / co-location gap, unless the assumed primitive closes it |

**Scope decision: Rungs 1–2 are the empirical spine of the paper. Rungs 3–4 are a
concise conditional-protocol section** (with an assumed cryptographic primitive
stated explicitly, §3.4) unless the optional ST3 pilot (§6) earns Rung 3 a fuller
treatment.

Reading **down** is benign → adaptive. Reading **across** is the ladder: each
rung's price and ceiling stated.

### 1.4 What each rung buys (and what Rung 1 actually establishes)

- **Rung 1 on its own is a genuine contribution — worded carefully.** What a line
  or cyclostationarity test *directly* establishes is **evidence of coherent,
  repeated power structure within the specified time-scale and meter model**, under
  a stated structural null. The *interpretation* as repeated multi-phase
  computation additionally assumes competing periodic physical sources have been
  excluded or bounded — an explicit nuisance model, not a logical consequence of
  rejecting stationarity. The measured A100 controller oscillation is the motivating
  counterexample: a structural detector may correctly detect cyclic power while the
  causal interpretation is wrong. Rung 1's hard-boundary row above is therefore
  non-empty by design.
- **But governance asks about *training*, not cyclicity.** Rung 1's honest answer to
  "is this machine training?" is "there is order-coherent power structure — an
  expected signature of efficient iteration-structured training — but I cannot yet
  distinguish it from another periodic workload." So Rung 1 is evidence, not
  identification.
- **Rung 2 converts structure into training-likeness**, at the cost of a null: "more
  training-like than the stated inference population." Ceiling: domain shift and the
  semantic decoy.
- **Rung 3 converts classification into physical authentication**, at the cost of
  prover cooperation: binding the physical response to an unpredictable challenge
  escapes both the null *and* the decoy. Ceiling: the challenge-aware dummy load.
- **Rung 4 converts authentication into work verification** only conditionally on a
  transcript/binding primitive (§3.4). Ceiling: the co-location gap.

The through-line: **each rung is the principled response to the previous rung's
ceiling**, and the paper's value is mapping the ladder honestly — including where
each rung stops.

---

## 2. The observation channel

`P = rF + P₀` describes the device, not the meter. For a paper framed around an
external sensor the minimum model is:

```text
P_meter(t)  = (h * P_device)(t) + P_other(t) + η(t)
P_device(t) = r(t) F(t) + P₀(t)
```

where `h` collects power-delivery dynamics, meter integration, reporting filters,
and sampling. `h` may be operating-point dependent; the LTI form is an explicit
first approximation, stated as such.

The synthetic evaluation sweeps, at minimum:

- low-pass bandwidth and integration window;
- sample cadence and aliasing;
- stable gains and transfer-function notches near the cadence;
- additive coloured noise;
- periodic controller interference;
- multiple independent workloads behind the meter (superposition);
- time-varying baselines and operating points.

Until simultaneous on-device + external-meter measurement is performed, claims about
what reaches an off-chip sensor are **predictions**. The single-A100 result is the
**negative transport case**: the adopted aggregate signature becomes shallow on that
channel and is overtaken by a controller feature — more than a closing nod (§4.8).

---

## 3. Methods

### 3.1 Rung 1, benign setting — the qualified baseline

Non-adaptive prover, line present and near-stationary. A **multitaper harmonic
F-test** (Thomson): *a classical, approximately pivotal line test under a
stationary, locally smooth spectral null* — not an exact false-alarm guarantee
across arbitrary external power traces, and searching an unknown frequency /
harmonic family requires a search correction. A spectral matched filter / peak-PSD
is the simpler comparator. Message: the benign case is easy, so the paper is not
about the benign case.

### 3.2 Adaptive setting — tracked cyclostationary detection

**The methodological heart, and the highest-risk component. Its null validity is
the main Phase 0 research question — it is *not* an established result** (§6, ST1).

Three components, each with one job:

1. **Tracker (Viterbi / ridge / synchrosqueezing)** — estimates the instantaneous
   cycle frequency `f̂₀(t)` as the line wanders. An estimator of a nuisance
   parameter, not the detector.
2. **Phase resampling** — `θ̂(t) = 2π ∫ f̂₀`; resample at equal increments of `θ̂`,
   so a wandering line becomes a constant-order line (tacholess order tracking).
3. **Cyclostationary / harmonic test** — Dandawaté–Giannakis
   `Q_α = N r̂_αᴴ Σ̂_α⁻¹ r̂_α` on the de-warped signal.

**The intended win, stated at the right strength.** For a *fixed* candidate cycle
frequency, `Q_α` is **asymptotically pivotal (→ χ²) under the stated structural
null** given weak dependence, consistent covariance estimation, and regularity —
which would let the threshold come from asymptotics rather than a hardware-matched
serving corpus. But those conditions do **not** automatically cover the complete
adaptive pipeline, and the paper must not say "analytic CFAR" until that is shown.
Open validity issues ST1 must resolve:

- **Path selection**: the effective search is over tracker *paths*, not a few final
  α values; a small-set correction may not capture the full look-elsewhere effect.
- **Estimated time warp**: even a warp learned on other data can turn stationary
  time-domain covariance into phase-dependent covariance in the angle domain; the
  fixed-α null does not automatically survive resampling.
- **Sample splitting**: splitting over tapers is not equivalent to independent
  sample splitting; splitting frames needs stated assumptions about temporal
  dependence, interpolation, and what the tracker may see.
- **Covariance estimation**: the long-run covariance estimator must stay stable
  under the held-out design, finite length, coloured noise, and the lag family.
- **Finite-sample vs asymptotic**: scope every "CFAR" phrase to its assumptions.
- **Surrogates**: Fourier-phase surrogates test a linear stationary-invariance
  null — a useful corpus-free calibrator, not a distribution-free replacement for
  every hardware or serving null.

If calibration cannot be defended, the method is honestly reframed as **an
effective score requiring empirical calibration** — still usable, differently
worded.

**No sequential-repair claim.** An asymptotic *marginal* χ² null per block does not
give the *conditional* e-value property needed to multiply blocks into a
supermartingale (dependence, adaptive nuisance estimates, information reuse).
Anytime-valid monitoring is **deferred**: this paper either restricts to a fixed
horizon or, at most, sketches the non-overlapping-block construction with its
conditional-null requirements as future work. The memo-§6 soundness gap is *not*
described as repaired.

**Rung 2 still needs a null — intrinsically.** Distinguishing training-cyclicity
from inference-cyclicity requires modelling inference; that is our inference null
(§1.1 layer 2), under our control in the synthetic setting.

### 3.3 Rung 2 — a specified statistical task, not a vibe

Decided before Phase 2, in the plan now:

- **Features**: the Rung 1 physics quantities as a fixed interpretable vector —
  harmonic energy, cyclic correlations, phase-folded waveform repeatability,
  tracker stability.
- **Decision rules compared** (all three, so the reader sees where performance
  comes from): (i) a prespecified physics score; (ii) a model-based likelihood /
  discriminant fitted to the scenario generator; (iii) a flexible learned reference
  that exposes how much comes from the chosen null rather than the structural
  method.
- **Reporting, kept separate**: Rung 1 false alarms under the structural null;
  Rung 2 FPR/FNR under the stated workload population; **transfer** — performance
  when hardware, meter, or workload parameters leave that population (domain
  shift evaluated, not just named).

**Semantic falsification controls (evaluated, not merely stated).** These are the
paper's central question — what the meter can and cannot certify:

- periodic inference / a periodic request generator;
- gradient-only computation;
- forward/backward/optimizer computation whose update is discarded;
- a training-shaped non-ML kernel loop (the semantic decoy);
- a controller-cycle or periodic non-compute load;
- asynchronous / de-periodicised genuine training;
- mixtures of the above with unrelated co-resident loads.

Expected interpretations: decoys scoring as training ⇒ Rung 2 certifies *physical
training-likeness*, not semantic training; missed asynchronous training ⇒ the
result is scoped to the efficient iteration-structured family and the measured
frontier (§5); controllers triggering Rung 1 ⇒ the structural test works and causal
attribution needs the stated nuisance model.

### 3.4 Rungs 3–4 — conditional protocol with an explicit assumed primitive

Verifier emits an unpredictable balanced challenge `c_k ∈ {−1,+1}`; the prover runs
a committed, semantically-equivalent schedule variation keyed to `c_k`; the
verifier correlates external power against the challenge over a causal lag window,
lock-in / spread-spectrum style. The null comes from **re-randomising the
challenge** — no corpus, no matched hardware. This escapes the semantic decoy (a
passive decoy is uncorrelated with the challenge); its own ceiling is the
**challenge-aware dummy load**.

**The assumed primitive (Rung 4), stated as an ideal functionality** rather than "a
proof or something": *the verifier has a sound, fresh, challenge-bound transcript or
proof tying each challenge bit to the prescribed semantically-equivalent variation
of declared training state transitions.* It must provide: commitment before the
challenge is revealed; unpredictability at commitment time; sound binding of the
response to valid declared state transitions; freshness against replay; and a
stated position on whether co-location is assumed or open. Conditional on that
primitive, challenge-synchronous power supplies physical evidence for the declared
response; **semantic work binding remains an assumed/open layer**, and the
remote-proof + dummy-load co-location gap stays open unless the primitive rules it
out.

**Randomisation-test discipline**: the test is exact under the sharp null that the
challenge has no effect on the measured process, *conditional on the randomisation
design* — so any maximisation over causal lags, impulse responses, or preprocessing
must be repeated inside every re-randomised replicate; carryover between challenge
periods, adaptive schedule choices, and balance constraints belong in the design.

If Rung 3 is to be a full empirical contribution (only if ST3 runs and passes,
§6), a generic injected modulation is not enough: it needs a concrete pair of
semantically-equivalent schedules, a stated transcript/commitment interface, the
full search inside every replicate, a challenge-aware dummy-load adversary,
modulation-depth / overhead / observation-time curves — and preferably a real
single-GPU pilot, since challenge modulation does not need a cluster.

---

## 4. The adversary: operators and budgets

The paper claims robustness only over **specified attack families at specified
budgets** — with unlimited superposition, dilution, shaping power, time, or freedom
to restructure work, the prover can drive SNR arbitrarily low or synthesise the
null. Each attack is an operator with a reported budget:

- **phase attack**: jitter, drift, or randomised work per iteration;
- **frequency relocation**: move the cadence toward or outside the search family;
- **additive burial**: independent workload superposition / spectral-floor jamming;
- **amplitude shaping**: added load flattening the compute/communication swing;
- **dilution**: reduce the training duty fraction among other work;
- **semantic decoy**: reproduce the physical schedule without retained training;
- **challenge-aware decoy**: a correlated dummy response in the active protocol.

For each: detector power vs physical resource use, throughput, energy, and — where
applicable — learning progress.

---

## 5. The de-periodicisation frontier (a central result, not a caveat)

The strongest response to the hiding adversary is not to assert that randomising
training must be expensive — **our own measurements already refute the naive
version of that claim**:

1. **Idle-insertion jitter** smears timing but *deepens* the power swing: measured
   ~15–680% throughput overhead and made detection *easier*.
2. **Work-varying jitter** (varying real gradient-accumulation work) moves the
   cadence without idle: measured hardware-throughput cost **≈ zero within noise**
   (`results/b2/workjitter_summary.json`), while line-band detection degrades at
   some levels.

So the paper must not say de-periodicisation is necessarily inefficient. What is
plausible but **unmeasured** is a *learning-efficiency* penalty: changing effective
batch sizes, accumulation counts, or update timing may preserve tokens/s while
worsening progress per token or per joule. The experiment therefore reports, per
attack level:

- a physical de-periodicisation measure (cadence CV, phase diffusion);
- detector power for the fixed line test, tracker, order/cyclostationary test, and
  any Rung 2 classifier;
- raw token/sample throughput; energy per processed token; optimizer updates per
  unit time; training progress per token and per joule; ideally time/energy to a
  target validation loss.

**Deliverable: a Pareto frontier** — *how far must a prover de-periodicise useful
training before the meter loses it, and what systems or learning-efficiency cost is
paid to get there?* Attack families on the frontier: per-iteration i.i.d. duration
jitter; slowly drifting cadence; variable real work / gradient accumulation;
asynchronous or overlapped schedules where the model permits; cadence relocation
toward search-band edges; with power shaping and superposition as separate,
clearly-labelled families (§4).

---

## 6. Phase 0 — de-risking gates (go/no-go before writing)

Four gates replace the old two.

### ST0 — estimand, threat model, novelty (paper-viability desk work)

- Fix the four rungs and which are in-paper (decision above: 1–2 spine, 3–4
  conditional).
- Define the admissible efficient-training family and attack budgets (§1.2, §4).
- State the full off-chip observation model (§2).
- Correct Ko/ours provenance everywhere (§1.1).
- Choose the target venue (open question §9.1).
- **New novelty audit** for adaptive order/cyclostationary detection and
  challenge-synchronous power authentication — the existing `novelty-audit.md` is
  scoped to the old Viterbi paper.

### ST1 — adaptive structural detector (the make-or-break gate)

The go/no-go question, restated:

> Can we derive or defensibly calibrate the null distribution of the complete
> tracker–phase-estimation–resampling–cyclostationary pipeline, including its
> selection and covariance-estimation steps?

Staged design, isolating each adaptive ingredient:

1. fixed known cycle frequency (base implementation);
2. known externally-supplied phase path (isolates resampling);
3. phase path estimated on genuinely held-out data;
4. the fully adaptive pipeline;
5. stationary Gaussian and coloured nulls;
6. stationary non-Gaussian nulls;
7. locally-stationary and controller-like nulls (nominal validity *may*
   intentionally fail here — that is information);
8. sweep observation length, lag set, tracker bandwidth, search size;
9. compare against a full-pipeline surrogate / randomisation calibration.

Report realised FAR **with uncertainty at operationally small nominal levels**, not
only 0.05. Include the bake-off here: fixed harmonic F, fixed matched filter,
Viterbi/ridge, order-folded, cyclostationary. **Go** if the method has defensible
calibration **or** is honestly reframed as a score requiring empirical calibration.

### ST2 — de-periodicisation frontier

- Sweep idle, real-work, drift, and phase-diffusion attacks; include band-edge
  relocation and harmonic-suppression attempts.
- Measure detector power, systems cost, and learning-efficiency cost (§5 metrics).
- Establish the regime where tracking/order methods materially beat fixed tests.

### ST3 — optional challenge-synchronous pilot

Run **only if** Rung 3 is to remain a paper contribution: concrete equivalent
schedule pair; full randomisation test (search inside every replicate); transfer
uncertainty; challenge-aware dummy load; and determine whether the result says more
than the expected √N processing gain. If ST3 is omitted or fails, Rungs 3–4 stay a
conditional protocol / future-work section — that is the default scope.

**Fallback honesty.** If ST1 fails, an F-test-only paper on an adopted synthetic
generator is *not automatically publishable*. The fallback would have to be carried
by the identifiability result, the semantic counterexamples (§3.3), the negative
transport case (§4.8 of the spine), and a clearly mapped detection boundary — and
we would assess viability on that basis, not assume it.

---

## 7. Provisional paper outline

1. **Introduction** — what an external power trace can and cannot certify; the
   claim ladder, concisely; scope to the Type / time-resolved channel.
2. **Threat model and identifiability** — full observation channel (§2); admissible
   efficient-training family; adversary operators and budgets; the same-trace
   semantic counterexample.
3. **Scenario models** — Ko training/fine-tuning/superposition; *our* inference
   null; the transfer family; explicit provenance and sensitivity analysis rather
   than "calibration."
4. **Structural evidence (Rung 1)** — multitaper baseline (qualified); tracked
   order/cyclostationary method; null assumptions and their validation.
5. **The de-periodicisation frontier** — detection vs timing and real-work attacks,
   with systems and learning-efficiency costs.
6. **Conditional training classification (Rung 2)** — stated inference population,
   decision rules, calibration, domain shift, semantic controls.
7. **From passive evidence to active authentication (Rungs 3–4)** — short
   conditional protocol; assumed transcript primitive; binding/co-location gap.
   Moves to an appendix if it distracts from Rungs 1–2.
8. **Reality check and transfer limits** — the single-A100 negative transport case;
   external-meter and distributed-scale predictions (falsifiable: distributed scale
   deepens the line).
9. **Discussion and conclusion** — exactly what each rung certifies, its
   information cost, and its hard ceiling.

Appendices: detector bake-off (pre-registered, so "we present one method" is an
*outcome*); identifiability details.

---

## 8. Task phasing (to seed a reconciled `tasks.md`)

- **Phase 0** — ST0 + ST1 + ST2 gates (§6); ST3 only if Rung 3 stays empirical.
- **Phase 1** — Rung 1 full: qualified F-test + tracked-cyclostationary on the
  scenario model, benign and adaptive settings; null-validity validation; bake-off.
- **Phase 2** — Frontier (§5) at full resolution + Rung 2: inference null,
  specified decision rules, semantic controls, transfer.
- **Phase 3** — Rungs 3–4 conditional-protocol section (or ST3-backed demo).
- **Phase 4** — Write-up; negative transport case; identifiability theory section.

`tasks.md` currently tracks the *old* (paper-verification) direction and will need
reconciling once Phase 0 clears.

---

## 9. Explicitly out of scope / deferred

- **Anytime-valid / sequential monitoring**: deferred; no claim that the χ²
  statistic repairs the e-process (see §3.2). Fixed-horizon results only.
- **Tier B unified track-detect estimator** (forward recursion + e-value
  emissions): parked in **#62**.
- **Real distributed hardware**: a falsifiable prediction plus the single-GPU
  negative transport case only.
- **Amount / energy-accounting channel**: excluded by decision; time-resolved
  trace only.
- **Designing the cryptographic binding primitive** (ZK proof-of-training): assumed
  as an ideal functionality at Rung 4, not built.

---

## 10. Open questions still to settle

1. **Target venue** (ST0). A mostly-synthetic theory/methods paper fits a
   governance / security / signal-processing venue better than ICML's empirical
   track; the choice sets how much the identifiability theory must carry.
2. **How rigorous is the Rung 1 identifiability theory?** Real theorems (decoy
   construction, conditions for structural-null validity) → a contribution;
   qualitative → a position paper. Sets effort in §3.
3. **Which tracker** for `f̂₀` (Viterbi vs synchrosqueezing vs Vold–Kalman) — settle
   via ST1 / the bake-off.
4. **Does ST3 run at all?** Default is no (Rungs 3–4 conditional-protocol only);
   revisit after ST1/ST2 land, given a single-GPU pilot is actually feasible for
   the challenge channel.
