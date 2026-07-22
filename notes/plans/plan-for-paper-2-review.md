# Review of the Paper 2 plan

**Date:** 2026-07-21  
**Plan reviewed:** [`plan-for-paper-2.md`](plan-for-paper-2.md)  
**Background memo:** [`power-verification-paths-forward.md`](../discussion/power-verification-paths-forward.md)

## Overall verdict

The proposed direction is strong. The ladder from structural power evidence, to
conditional training-likeness, to challenge-bound verification is much clearer and
more honest than centring the paper on a single Viterbi statistic. It directly exposes
what an external power trace can establish, what additional information each stronger
claim requires, and where each claim stops.

The plan should nevertheless receive one revision before implementation. The main
changes are to:

1. correct the provenance of the synthetic workloads;
2. state the Level 1 estimand and adversary more precisely;
3. treat the null validity of the adaptive tracker--resampler--cyclostationary pipeline
   as the main Phase 0 research question, rather than as an established result;
4. turn de-periodicisation into a measured detection--utility frontier;
5. specify what Level 2 actually computes and test its semantic counterexamples;
6. either demote Level 3 to a conditional protocol/outlook or state its assumed
   cryptographic primitive explicitly; and
7. model the actual off-chip observation channel, at least synthetically.

With those changes, the plan is ready to proceed to Phase 0.

---

## 1. Correct the workload-model attribution

The plan currently describes Ko and Zhu as the training/serving generator and refers
several times to a "Ko inference model." This is inaccurate:

- Ko and Zhu supply the training and fine-tuning waveforms and their aggregate
  superposition.
- They do not supply an inference workload.
- The inference process in this repository---continuous-batching decode structure, a
  slow MoE envelope, and aperiodic prefill bursts---is our literature-motivated
  construction. The code states this explicitly in
  [`code/ko_workload.py`](../../powerladder/ko_workload.py).
- The inference-dominant aggregate, which substitutes our inference process into Ko's
  dominant slot, is also our construction rather than Ko's equation 11 workload.

Suggested wording:

> We adopt Ko and Zhu's training, fine-tuning, and aggregate workload model. We
> construct a literature-motivated continuous-batching inference null in the same
> power and superposition framework. Level 2 conclusions are conditional on that
> stated null.

Likewise, replace **"calibrated synthetic workload model"** with
**"literature-parameterized scenario model"** unless the paper defines the target,
data, and procedure used for calibration. The roughly 75-fold difference between the
Ko-scale modulation and the measured single-GPU cadence in
[`issue54-investigation.md`](../results/issue54-investigation.md) is evidence
that the present generator is not calibrated to the measured channel.

The paper should distinguish four layers of model provenance:

1. Ko and Zhu's training/fine-tuning waveform and superposition;
2. our inference null;
3. our added slow frequency wander and adversary transformations; and
4. our mapping from the workload waveform through the power/meter observation model.

---

## 2. Clean up the ladders and notation

### 2.1 Q1/Q2

`spec.md` uses Q1 for Amount and Q2 for Type, whereas the plan uses Q1 and Q2 for
non-hiding and hiding adversaries. The specification is somewhat outdated, so this is
not a substantive problem, but the collision will still confuse repository readers.
Prefer:

- **benign/non-adaptive setting**; and
- **adaptive/hiding adversary**.

### 2.2 Do not call the new ladder `beta(I)`

This is more than a notation clash. In `spec.md`, `beta(I)` is a quantitative bound on
the unidentified covert-compute fraction in the Amount problem. The new ladder is a
ladder of claims, assumptions, or evidence. It does not return the same mathematical
object.

Call it the **claim ladder**, **evidence ladder**, or **verification ladder**. The paper
can say it is *analogous in spirit* to the Amount attestation ladder: more verifier
information permits a stronger claim, at greater institutional cost.

### 2.3 Recommended conceptual rungs

The cleanest fully expanded ladder is:

| Rung | Claim | Additional requirement | Ceiling |
|---|---|---|---|
| **1. Structural evidence** | Order-coherent power variation exists under a specified structural null | External time-resolved trace | Periodic controllers and other physical confounds; aperiodic training; transfer-function loss |
| **2. Conditional classification** | The trace is more training-like than the stated inference population | A representative or explicitly modelled inference null | Domain shift and a semantic decoy with training-like execution |
| **3. Active physical authentication** | A load behind the meter responded to an unpredictable verifier challenge | Prover participation and a challenge-bound schedule | A challenge-aware dummy load; no semantic binding by power alone |
| **4. Work verification** | The physical response is tied to declared training state transitions | A sound, fresh transcript/proof and an explicit binding assumption | Remote-proof/co-location gap unless the assumed primitive closes it |

For Paper 2, Rungs 1--2 can form the empirical spine. Rungs 3--4 can be compressed into
a conditional protocol section or deferred to future work.

---

## 3. Frame Level 1 around efficient iteration-structured training

It would be unhelpfully pedantic to organise the paper around the observation that some
conceivable training execution can be made completely aperiodic. The programme is
specifically interested in detecting the frequency exposed by efficient, synchronous,
iteration-structured training. The right response is to state that target population
and make destruction of the cadence an adversarial action.

Suggested central statement:

> We target efficient, iteration-structured training that exposes a recoverable
> cadence. An adaptive prover may de-periodicise its schedule; we measure the
> detection--utility frontier as that randomisation increases.

Avoid saying that a recoverable line is universally necessary for all training. A more
precise claim is that it is an expected physical signature of the synchronous training
regime studied here, especially when distributed synchronization makes the modulation
deep and coherent.

Also change the statement that the signature "only exists" in distributed training.
The cadence exists on a single GPU and the measurements find a faint corresponding
ripple. Distributed synchronization is expected to make it **deep, coherent, and
legible**, rather than to create it from nothing.

### 3.1 What Level 1 actually establishes

A line or cyclostationarity test directly establishes order-coherent variation in the
observed power process under a stated structural null. Causal attribution to
"multi-phase computation" additionally assumes that competing periodic physical
sources have been excluded or bounded.

The Level 1 claim should therefore be worded as:

> Evidence of coherent, repeated power structure within the specified time-scale and
> meter model.

The interpretation as repeated computation should then be presented under an explicit
nuisance model, rather than as a logical consequence of rejecting stationarity.

Level 1's currently blank hard boundary should include:

- periodic power-management, cooling, fan, mains, or co-resident-load processes;
- structural-null misspecification and generic nonstationarity;
- sufficiently strong superposition or dilution;
- exact or near-exact active power shaping;
- a cadence moved outside the searched family; and
- attenuation or a zero in the die-to-meter transfer function.

The measured A100 controller oscillation is a useful motivating counterexample: a
structural detector may correctly detect cyclic power while the causal interpretation
is still wrong.

---

## 4. Make de-periodicisation a central detection--utility frontier

The strongest response to a cadence-hiding adversary is not to assume that randomising
training must be expensive, but to measure how much randomisation is required and what
it costs.

Previous work in the repository already separates two mechanisms:

1. **Idle-insertion jitter.** Pauses smear timing but deepen the power swing. The
   measured implementation incurred approximately 15--680% throughput overhead and
   made detection easier.
2. **Work-varying jitter.** Varying real gradient-accumulation work moves the cadence
   without inserting idle. In [`workjitter_summary.json`](../../data/measured_cost_anchors/workjitter_summary.json),
   its measured hardware-throughput cost is approximately zero within noise, while
   line-band detection degrades at some levels.

Consequently the new paper must not state that de-periodicisation is necessarily
inefficient. The existing work shows that some implementations are costly and another
tested implementation is essentially free in raw device throughput.

What remains plausible but unmeasured is a **learning-efficiency** penalty. Changing
effective batch sizes, accumulation counts, or update timing may preserve tokens per
second while worsening progress per token or per joule. The relevant experiment should
therefore report:

- cadence coefficient of variation, phase diffusion, or another physical measure of
  de-periodicisation;
- detector power for the fixed line test, tracker, order/cyclostationary test, and any
  Level 2 classifier;
- raw token or sample throughput;
- energy per processed token;
- optimizer updates per unit time;
- training progress per token or joule; and, ideally,
- time or energy to reach a target validation loss.

The desired result is a Pareto frontier:

> How far must a prover de-periodicise useful training before the meter loses it, and
> what systems or learning-efficiency cost is paid to reach that point?

This frontier should include at least:

- per-iteration i.i.d. duration jitter;
- slowly drifting cadence;
- variable real work/gradient accumulation;
- asynchronous or overlapped schedules where the model permits them;
- cadence relocation toward search-band edges; and
- power shaping or superposition as separate, clearly labelled attack families.

---

## 5. Treat adaptive CFAR validity as the main Phase 0 question

The statistical core of the proposed method is promising but not yet justified by the
standard fixed-frequency cyclostationarity result.

The proposed pipeline is data-adaptive:

1. search for a favourable frequency path;
2. estimate instantaneous phase from that path;
3. resample the same process using the estimated phase; and
4. test for cyclic structure after the resampling.

For a fixed candidate cycle frequency, a Dandawate--Giannakis-style self-normalised
statistic can have an asymptotic chi-squared null under weak-dependence, consistent
covariance estimation, and other regularity conditions. Those conditions do not
automatically establish the null distribution of the complete adaptive pipeline.

Specific issues to resolve:

- **Path selection.** The effective search is over paths, not merely a small number of
  final alpha values. A correction over a few tracker proposals may not capture the
  full look-elsewhere effect.
- **Estimated time warp.** Even when learned on other data, a non-uniform time warp can
  turn stationary time-domain covariance into phase-dependent covariance in the angle
  domain. The fixed-alpha null does not automatically survive resampling.
- **Sample splitting.** Splitting over tapers is not generally equivalent to independent
  sample splitting. Splitting frames requires assumptions about temporal dependence,
  interpolation, and what information the tracker is allowed to use.
- **Covariance estimation.** The long-run covariance estimator must remain stable under
  the held-out design, finite observation length, coloured noise, and the chosen lag
  family.
- **Finite-sample versus asymptotic validity.** The phrase "analytic CFAR" should be
  scoped to its assumptions. Prefer "asymptotically pivotal under the stated structural
  null" until stronger validity is proved.
- **Surrogate calibration.** Fourier-phase surrogates test a linear stationary-process
  invariance null. They are useful, but are not a distribution-free replacement for
  every hardware or serving null.

### 5.1 Revised ST1 go/no-go

ST1 should ask:

> Can we derive or defensibly calibrate the null distribution of the complete
> tracker--phase-estimation--resampling--cyclostationary pipeline, including its
> selection and covariance-estimation steps?

The smoke test should include:

1. a fixed, known cycle frequency, establishing the base implementation;
2. a known externally supplied phase path, isolating the resampling step;
3. a phase path estimated on genuinely held-out data;
4. the fully adaptive pipeline;
5. stationary Gaussian and coloured nulls;
6. stationary non-Gaussian nulls;
7. locally stationary and controller-like nulls, on which nominal structural-null
   validity may intentionally fail;
8. a sweep over observation length, lag set, tracker bandwidth, and search size; and
9. comparison with a full-pipeline surrogate or randomisation calibration.

Report realised false-alarm rates with uncertainty at operationally small nominal
levels, not only at 0.05. A method that achieves power but not calibration should be
presented as an effective score requiring empirical calibration, not as analytic CFAR.

### 5.2 Do not claim that this automatically repairs the e-process

An asymptotic marginal chi-squared null for each block does not imply the conditional
e-value property needed to multiply block e-values into a supermartingale. Dependence,
adaptive nuisance estimates, and reuse of information across blocks remain.

The plan should either:

- defer anytime-valid monitoring to a later paper;
- restrict the result to a fixed horizon;
- use non-overlapping blocks with a fully stated conditional null and predictable
  nuisance estimation; or
- add a separate sequential-validity theorem and validation programme.

Do not describe the sequential gap as repaired until conditional validity is actually
shown.

---

## 6. Qualify the multitaper baseline

The multitaper harmonic F-test is an excellent Level 1 baseline. It removes an unknown
local scale and supplies an interpretable classical line test. Its analytic F reference
still depends on its noise and local-stationarity assumptions, and searching an unknown
frequency or harmonic family requires a search correction.

Use wording such as:

> a classical, approximately pivotal line test under a stationary, locally smooth
> spectral null

rather than implying an exact false-alarm guarantee across arbitrary external power
traces.

If the adaptive cyclostationary method fails, the F-test alone on an adopted synthetic
generator is unlikely to constitute a methods paper. The fallback paper would need to
be carried by the identifiability result, semantic counterexamples, negative external
validation, and a clearly mapped detection boundary.

---

## 7. Specify Level 2 as an actual statistical task

The plan says that Level 2 uses an inference null but does not yet specify:

- the feature or statistic used for classification;
- whether this is a likelihood-ratio test, a calibrated score, or a learned classifier;
- how thresholds are estimated;
- the intended false-positive guarantee;
- what hardware, meter, and workload variables define the deployment population; or
- how domain shift is evaluated.

These choices should be made before Phase 2. One coherent approach is to use Level 1
features---harmonic energy, cyclic correlations, phase-folded waveform repeatability,
and tracker stability---as a fixed interpretable feature vector, then compare:

1. a prespecified physics score;
2. a model-based likelihood or discriminant fitted to the scenario generator; and
3. a flexible learned reference that exposes how much performance comes from the chosen
   null rather than from the structural method.

Report calibration and detection separately:

- Level 1: false alarms under the structural null;
- Level 2: false positives and false negatives under the stated workload population;
- transfer: performance when hardware, meter, or workload parameters leave that
  population.

### 7.1 Restore the semantic falsification controls

The background memo correctly prioritises negative controls, but they largely disappear
from the concrete plan. At minimum include:

- periodic inference or a periodic request generator;
- gradient-only computation;
- forward/backward/optimizer computation whose update is discarded;
- a training-shaped non-ML kernel loop;
- a controller-cycle or periodic non-compute load;
- asynchronous/de-periodicised genuine training; and
- mixtures of the above with unrelated co-resident loads.

These are not merely limitations to state. They should be evaluated because the paper's
central question is what the meter can and cannot certify.

Expected interpretation:

- If training-shaped decoys score as training, Level 2 certifies physical
  training-likeness rather than semantic training.
- If asynchronous genuine training is missed, the result is scoped to the efficient
  iteration-structured family and the measured de-periodicisation frontier.
- If periodic controllers trigger Level 1, the structural test is working but causal
  attribution requires the nuisance assumptions to be stated.

---

## 8. Model the actual off-chip observation channel

The plan's observation equation `P = rF + P0` is insufficient for a paper framed around
an external meter. Use at least:

```text
P_meter(t) = (h * P_device)(t) + P_other(t) + eta(t),
P_device(t) = r(t) F(t) + P0(t),
```

where `h` includes power-delivery dynamics, meter integration, reporting filters, and
sampling. It may be operating-point dependent; the LTI model is an explicit first
approximation rather than a universal truth.

At minimum the synthetic evaluation should sweep:

- low-pass bandwidth and integration window;
- sample cadence and aliasing;
- stable gains and transfer-function notches near the cadence;
- additive coloured noise;
- periodic controller interference;
- multiple independent workloads behind the meter; and
- time-varying baselines and operating points.

Until simultaneous on-device and external-meter measurement is performed, claims about
what reaches an off-chip sensor should be predictions. The existing single-A100 result
should be more than a closing nod: it is a valuable negative transport case showing
that the adopted aggregate signature becomes shallow and is overtaken by a controller
feature on the available channel.

Also change "everything lives in the time-resolved power spectrum" to **"everything
uses the time-resolved power trace rather than the integrated-energy Amount channel."**
Cyclostationary structure may be expressed spectrally, but challenge correlation is a
time-domain operation and the underlying observable is the trace.

---

## 9. Define the hiding adversary and its budgets

The paper should not claim robustness to an unrestricted adversary. With unlimited
superposition, dilution, shaping power, time, or freedom to restructure work, the prover
can drive signal-to-noise arbitrarily low or synthesize the null.

Define each attack as an operator and report its budget:

- **phase attack:** jitter, drift, or randomized work per iteration;
- **frequency relocation:** move the cadence toward or outside the search family;
- **additive burial:** independent workload superposition or spectral-floor jamming;
- **amplitude shaping:** add load to flatten the compute/communication swing;
- **dilution:** reduce the training duty fraction among serving or other work;
- **semantic decoy:** reproduce the physical schedule without retained training; and
- **challenge-aware decoy:** produce a correlated dummy response in the active protocol.

For each attack report detector power against physical resource use, throughput, energy,
and---where applicable---learning progress. "Robust" should always mean robust over a
specified attack family and budget.

---

## 10. Level 3: conditional protocol or future work

There is no conceptual objection to asking what power measurement can add when the
prover supplies a cryptographic transcript or zero-knowledge proof. This paper need not
design that proof system. It should, however, replace vague language such as "a proof or
something" with a small ideal functionality.

Suggested assumption:

> The verifier has a sound, fresh, challenge-bound transcript or proof tying each
> challenge bit to the prescribed semantically equivalent variation of declared
> training state transitions.

The assumed primitive should provide:

- commitment before the relevant challenge is revealed;
- unpredictability of the challenge to the prover at commitment time;
- sound binding of the challenge response to valid declared state transitions;
- freshness or timing sufficient to prevent replay; and
- a stated position on whether co-location is assumed or remains open.

Given this primitive, challenge-synchronous power can establish that
challenge-responsive physical activity occurred behind the meter without an inference
corpus. A separate remote proof and a challenge-correlated dummy load still leave a
co-location gap unless the ideal primitive explicitly rules that attack out.

### 10.1 Recommended scope decision

The preferred Paper 2 scope is:

- Rungs 1--2: main theory and experiments;
- Rung 3: concise conditional protocol and perhaps an analytic/synthetic illustration;
- Rung 4: assumed primitive and explicit open systems problem.

This preserves the passive-paper title and keeps the empirical contribution coherent.

If Level 3 remains a full empirical contribution, a synthetic injected modulation is
not enough. It should use:

- a concrete pair of semantically equivalent schedules;
- a stated transcript/commitment interface;
- the complete causal-lag and transfer-family search inside every randomisation replicate;
- a challenge-aware dummy-load adversary;
- modulation-depth, overhead, and observation-time curves; and
- preferably a real single-GPU pilot, since challenge modulation does not require a
  distributed cluster.

### 10.2 Randomisation-test detail

The randomisation test can be exact under the sharp null that the randomized challenge
has no effect on the measured process, conditional on the chosen randomisation design.
Any maximisation over causal lags, impulse responses, or preprocessing choices must be
repeated for every re-randomized challenge sequence. Carryover between challenge periods,
adaptive schedule choices, and balance constraints must be included in the design.

---

## 11. Revised Phase 0

Before full implementation, use four gates.

### ST0 -- estimand, threat model, and novelty

- Define the four conceptual rungs and which are in the paper.
- Define the admissible efficient-training family and attack budgets.
- State the full off-chip observation model.
- Correct the Ko/our-model provenance.
- Choose the target venue.
- Run a new novelty audit for adaptive order/cyclostationary detection and
  challenge-synchronous power authentication. The existing
  [`novelty-audit.md`](novelty-audit.md) is scoped to the old Viterbi paper.

### ST1 -- adaptive structural detector

- Validate the fixed-frequency cyclostationarity implementation.
- Add phase resampling with a known path.
- Add a path learned on held-out data.
- Test the complete adaptive pipeline.
- Measure false-alarm calibration across the null families listed in Section 5.
- Compare fixed harmonic F, fixed matched filter, Viterbi/ridge, order-folded, and
  cyclostationary methods.
- Go only if the method either has defensible calibration or is honestly reframed as a
  score requiring empirical calibration.

### ST2 -- de-periodicisation frontier

- Sweep idle, real-work, drift, and phase-diffusion attacks.
- Measure detector power, systems cost, and learning-efficiency cost.
- Include band-edge and harmonic-suppression attempts.
- Establish the range over which tracking/order methods materially beat fixed tests.

### ST3 -- optional challenge-synchronous pilot

Run only if Level 3 remains a paper contribution:

- select a concrete equivalent schedule pair;
- implement the full randomisation test;
- include transfer uncertainty and a challenge-aware dummy load; and
- determine whether the result says more than the expected square-root processing gain.

If ST3 is omitted or fails, retain Level 3 as a conditional protocol/future-work section.

---

## 12. Suggested revised paper spine

1. **Introduction.** What an external power trace can and cannot certify; concise claim
   ladder; scope to the Type/time-resolved channel.
2. **Threat model and identifiability.** Full observation channel, admissible efficient
   training family, adversary budgets, same-trace semantic counterexample.
3. **Scenario models.** Ko training/fine-tuning/superposition; our inference null;
   transfer family; explicit provenance and sensitivity rather than "calibration."
4. **Structural evidence.** Multitaper baseline; tracked order/cyclostationary method;
   null assumptions and validation.
5. **The de-periodicisation frontier.** Detection against timing and real-work attacks,
   with systems and learning-efficiency costs.
6. **Conditional training classification.** Stated inference population, decision rule,
   calibration, domain shift, and semantic controls.
7. **From passive evidence to active authentication.** Short conditional challenge
   protocol; assumed transcript primitive; binding/co-location gap. Omit or move to an
   appendix if it distracts from Rungs 1--2.
8. **Reality check and transfer limits.** Single-A100 negative transport case; external
   meter and distributed-scale predictions.
9. **Discussion and conclusion.** Exactly what each rung certifies, its information cost,
   and its hard ceiling.

---

## 13. Concrete wording changes to the current plan

- Replace **"training/serving generator"** with **"Ko training/fine-tuning generator
  plus our inference null."**
- Replace **"calibrated synthetic"** with **"literature-parameterized scenario model"**
  unless calibration is documented.
- Replace **"the signature only exists in distributed training"** with **"distributed
  synchronization is expected to make the cadence signature deep and coherently
  legible."**
- Replace **"everything lives in the time-resolved power spectrum"** with **"the paper
  uses the time-resolved power trace and excludes the integrated-energy Amount
  channel."**
- Replace the Q1/Q2 adversary labels with **benign/non-adaptive** and
  **adaptive/hiding**.
- Replace references to the new ladder as `beta(I)` with **claim ladder** or
  **verification ladder**.
- Replace **"cyclicity is a necessary signature of training"** with the scoped
  efficient-training/de-periodicisation statement in Section 3.
- Replace **"self-calibrated false-alarm rate"** with **"an asymptotically pivotal test
  under the stated structural null"** until the adaptive pipeline is validated.
- Remove the claim that the chi-squared statistic automatically repairs the sequential
  e-process.
- Give Level 1 a non-empty hard-boundary row.
- Replace **"Level 3 binds activity to declared work"** with either:
  - **"conditional on the assumed transcript/binding primitive, challenge-synchronous
    power supplies physical evidence for the declared response"**; or
  - **"active physical authentication; semantic work binding remains an assumed/open
    layer."**
- Replace ST2's generic injected modulation with a concrete equivalent schedule pair if
  Level 3 remains experimental.
- Do not say that an F-test-only fallback is automatically a publishable result; state
  what identifiability and negative-control contributions would carry that fallback.

---

## Final recommendation

Proceed with the paper, centred on the following thesis:

> A passive external power trace can provide calibrated structural evidence and
> conditional evidence of efficient iteration-structured training, but not universal
> semantic identification. An adaptive prover can attack the exposed cadence; the
> empirical question is the detection--utility frontier required to erase it. Stronger
> work verification is possible only conditionally on additional challenge and
> transcript/binding primitives.

That thesis preserves the programme's central bet on detecting the training frequency,
turns the strongest objection into a measurable adversarial frontier, and keeps each
rung's claim commensurate with the information available to the verifier.
