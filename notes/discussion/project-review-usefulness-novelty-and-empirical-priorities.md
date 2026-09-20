# Project review: usefulness, novelty, and empirical priorities

**Date:** 2026-09-11  
**Status:** discussion note (external-style assessment, not a tracker). This note
reviews the project's usefulness to the AI safety and verification community,
positions its novelty against the literature available as of the assessment date,
and addresses two concerns: the absence of direct multi-GPU empirical validation
and the complexity of the null-test presentation. Scope remains defined by
[`../../spec.md`](../../spec.md); this note recommends changes but does not change
that scope.

---

## Executive assessment

This is a useful and intellectually honest AI-verification project, but it is not yet
a convincing empirical detector paper.

Its strongest contribution is not the generic claim that training can be detected
from power. That claim is now supported by adjacent work using on-device telemetry
and high-bandwidth physical probes. The distinctive contribution is narrower and
more valuable:

> A framework for determining when low-rate external power measurements provide
> structural evidence of synchronous training, how that evidence can be evaded, and
> why power alone cannot certify what computation occurred.

That is useful to the AI safety and verification community because it turns an
appealing monitoring idea into a falsifiable measurement problem. It separates the
physical observation, the statistical claim, the adversary's hiding budget, and the
semantic ceiling. It also gives concrete meter requirements rather than treating
"power monitoring" as a binary capability.

The evidence does not yet support a claim that this would work operationally. The
central observable—a coherent cadence from distributed training after aggregation
and external-meter transport—has not been directly measured in this project. The
perfect or near-perfect synthetic classification results therefore demonstrate
properties of the scenario model, not real-world detector performance.

The overall assessment is:

- **Conceptual and policy usefulness:** high.
- **Methodological novelty:** moderate.
- **Empirical maturity:** low.
- **Operational readiness:** very low.
- **Overall novelty:** moderate, provided the low-rate external-meter channel,
  explicit limits, and adversarial boundary remain central.

As written, the work is a strong technical report or verification workshop paper.
Validation on public multi-node traces would materially improve it. An owned
multi-GPU experiment with simultaneous rack- or wall-power measurement would move it
toward a credible systems or security measurement paper.

## 1. Value to AI safety and verification

The project is most useful as **measurement science for one layer of a verification
system**, not as a standalone proof that training occurred.

Its main value is fourfold:

1. It identifies a potentially verifier-controlled channel that does not depend on
   trusting prover-reported GPU telemetry.
2. It turns the proposed channel into testable requirements: sampling rate,
   integration time, in-band transfer, aggregation, background power, and cadence
   coherence.
3. It treats the monitored operator as adaptive and asks what physical or economic
   price must be paid to hide the signal.
4. It states the semantic limitation clearly: a sufficiently capable decoy workload
   can reproduce a power observable without performing retained training.

This last point is especially important for governance. Power could support anomaly
detection, inspection triage, corroboration of declarations, or selection of sites
for stronger scrutiny. It cannot by itself establish model state transitions,
retained learning, or the identity of the computation. The claim ladder is therefore
useful when read as defense in depth; it becomes misleading only if the passive rungs
are presented as certification in the stronger semantic sense.

For operational verification, the paper will eventually need a more deployment-like
error metric. A nominal per-trace false-positive rate of 5% is not sufficient when a
verifier repeatedly scans many sites and overlapping windows. The relevant outcomes
are closer to false alerts per monitored site-day, detection delay, multiple-testing
control, and the cost of adjudicating an alarm. Fixed-horizon calibration is a
reasonable scoped first result, but it should not be mistaken for a monitoring
guarantee.

## 2. The empirical gap

The concern about missing empirical tests is correct, with one qualification. The
project includes real single-A100 observations and measured systems-cost anchors,
but those are a negative transport case and cost inputs. They do not directly test
the central distributed claim. The manuscript is explicit about this distinction in
`sec:reality`, and the scope document calls distributed synchronization a
"falsifiable prediction" rather than a result.

The current generator inherits a large modulation from Ko and Zhu's
literature-parameterized grid model and adds an author-designed Ornstein–Uhlenbeck
frequency drift. The roughly 75-fold difference between the assumed modulation and
the measured single-A100 line is an honest warning, but it also means that synthetic
ROC values are not evidence that a real external meter has adequate signal-to-noise
ratio.

There are two distinct empirical questions:

1. **Workload coherence:** does multi-GPU training generate the predicted coherent
   cadence, with what line depth, width, drift, and scaling?
2. **Channel transport:** how much of that cadence survives node aggregation, power
   delivery, other loads, PDU filtering, meter integration, and facility control?

A telemetry-aggregation dataset can answer the first question. Only simultaneous
device- and external-meter measurements can answer the second.

### 2.1 A public validation opportunity

Vercellino et al. released multi-node H100 traces for training, fine-tuning, and
inference, including Llama-2 runs across 2, 4, 8, and 16 four-GPU nodes. Their study
reports synchronous aggregate fluctuations at each scale, becoming more pronounced
with node count. The [paper](https://arxiv.org/html/2604.07345v1) and
[NLR dataset](https://data.nlr.gov/submissions/312) offer an immediate test of the
project's central workload-level prediction.

The minimum useful reanalysis is:

1. Segment warm-up, steady training, evaluation, and checkpoint periods.
2. Estimate cadence frequency, line width or quality factor, modulation depth, and
   time-varying drift for each run.
3. Measure how absolute and fractional line strength scale from 8 to 64 GPUs.
4. Apply this project's integration and downsampling map, then run the detector
   unchanged.
5. Split train, calibration, and test data by complete run rather than drawing
   windows from the same execution into multiple splits.
6. Compare the measured distribution with the Ko–Zhu parameter assumptions,
   especially the assumed down-phase depth.
7. Evaluate against real inference and other available workloads rather than only
   the synthetic inference generator.

The result should be described as **multi-node workload validation**, not external
meter validation: the published aggregate is constructed from telemetry rather than
measured independently at a rack or facility meter.

### 2.2 Preferred owned experiment

A stronger experiment would record, on the same clock:

- per-GPU power;
- node-level telemetry;
- rack-PDU or wall power;
- training-step, all-reduce, optimizer-step, and checkpoint timestamps.

The workload matrix should include at least:

1. synchronous DDP or FSDP;
2. gradient accumulation or local SGD with periodic synchronization;
3. pipeline parallelism, such as 1F1B or an interleaved schedule;
4. a genuinely asynchronous or stale-synchronous system;
5. power stabilization enabled and disabled, where available.

The primary endpoints should be cross-worker coherence, aggregate line
signal-to-noise ratio, transfer from device aggregate to external meter, detector
ROC or precision-recall under realistic prevalence, and false alerts per site-day.
Five to ten complete replicates of 10–30 minutes per condition would be much more
persuasive than a large number of correlated windows from a few runs.

## 3. Does asynchronous training destroy the frequency line?

Potentially yes, but "asynchronous" covers several materially different systems.

Let the periodic component of worker $j$ have phase $\phi_j$. A useful coherence
order parameter is

$$
R(t)=\left|\frac{1}{N}\sum_{j=1}^{N} e^{i\phi_j(t)}\right|.
$$

The coherent aggregate component is proportional to $N R$. Under synchronous
operation, $R\approx1$, so the line amplitude grows approximately with $N$. If
worker phases are independent, the phasor sum grows only as $\sqrt{N}$, giving
$R\approx1/\sqrt{N}$. Mean power still grows as $N$, so fractional modulation
falls roughly as $1/\sqrt{N}$. Independent frequency drift broadens the remaining
energy and can remove the narrow line entirely.

The consequences differ by training regime:

- **Synchronous DDP/FSDP:** the most favorable case; barriers and collective
  communication can align worker phases.
- **Local SGD or periodic averaging:** may move the strongest global cadence to the
  less-frequent synchronization interval.
- **Pipeline parallelism:** staggered stages can flatten local compute/communication
  swings, but optimizer steps, pipeline flushes, accumulation boundaries, and global
  synchronization may leave slower lines or harmonics.
- **Stale-synchronous training:** retains bounded synchronization and may retain a
  broadened or slower cadence.
- **Fully asynchronous parameter-server or decentralized training:** the strongest
  counterexample; independent workers may eliminate a facility-wide narrow line.

The current "asynchronous training" control in
[`../../powerladder/typeb/rung2_scenarios.py`](../../powerladder/typeb/rung2_scenarios.py)
does not test this mechanism. It applies additional drift and phase slip to one
training waveform. It does not sum independently phased workers, model stragglers or
parameter-server updates, or vary cross-worker coherence. It should be renamed until
that is fixed.

The synthetic model should be extended to a multi-oscillator population and sweep:

- worker count;
- phase dispersion;
- worker-frequency dispersion;
- common versus idiosyncratic drift;
- synchronization interval;
- target job's share of aggregate facility power;
- multiple co-resident training jobs with unrelated cadences.

Asynchrony does not refute the scoped claim about efficient synchronous training. It
does refute any universal claim that training necessarily implies a detectable
facility-wide frequency line. The defensible statement is:

> Synchronous distributed training can produce a remotely detectable aggregate-power
> cadence; detectability depends on cross-worker coherence, meter transport,
> background load, and power stabilization.

## 4. Power stabilization is an additional boundary condition

Power stabilization may be a more immediate challenge than deliberately asynchronous
training. Production operators already have non-adversarial reasons to suppress
synchronized load swings.

Choukse et al. report production-scale synchronized compute- and
communication-phase fluctuations and software, hardware, and infrastructure
mitigations in [*Power Stabilization for AI Training Datacenters*](https://arxiv.org/abs/2508.14318).
Google reports that its [Balance of Power](https://cloud.google.com/blog/topics/systems/mitigating-power-and-thermal-fluctuations-in-ml-infrastructure)
system reduces some fluctuations by about 50% with less than 1% performance impact.

This does not show that complete concealment is cheap: reducing a swing by half is
different from erasing all detector evidence. It does show that a broad inference
from this project's 170–230% shaping costs to real deployment would be unsafe. The
cost may be shifted into scheduling flexibility, unused power headroom, energy
storage, control equipment, or capital expenditure rather than visible as lost
training throughput.

The frontier should therefore include at least one realistic stabilizer model and
distinguish:

- reduction of the observed cadence;
- complete erasure below a specified detector and false-alarm budget;
- throughput cost;
- energy and capital cost;
- effects imposed by ordinary datacentre operation rather than an adversary.

## 5. Novelty against the literature

The novelty is uneven across the project's components.

| Component | Assessment | Reason |
|---|---|---|
| Detecting training from power | Low by itself | Recent work already demonstrates workload classification from power-related channels. |
| Low-rate remote or facility-meter channel | Moderate to high | It creates a different trust boundary and a harder transport problem than on-device telemetry or a probe attached to one GPU. |
| HMM/Viterbi tracking of a drifting line | Low | Track-before-detect and HMM frequency-line tracking are mature signal-processing ideas. |
| Phase tracking followed by cyclostationary testing | Moderate as an application-specific composition | The composition is sensible, but its parts are established. |
| Surrogate calibration of the adaptive pipeline | Moderate and useful | The empirical invalidation of naive analytic calibration is practically important. |
| Detector-aware detection–utility frontier | Moderate | Adversarial workload studies now exist, but this project's external-channel formulation and explicit frontier remain differentiated. |
| Meter-resolution boundary | Moderate and practically useful | This converts an abstract monitoring proposal into a falsifiable sensor specification. |
| Semantic observational-equivalence ceiling | Useful framing, modest mathematical novelty | The construction is important for governance but follows a general identifiability argument. |
| Four-rung verification ladder | Useful synthesis | It organizes claims and assumptions but should not be sold as the principal technical novelty. |

### 5.1 Closest recent work

Gargiulo and Kulp's [*Workload Identification with Physical Side Channels for AI
Governance*](https://arxiv.org/abs/2609.00309) uses a verifier-owned, high-bandwidth
current probe on a single H200 and reports strong training/inference/non-AI
classification together with measured evasion costs. It shares this project's broad
headline but occupies the opposite end of the meter-location and bandwidth axis.
Their result makes "power can reveal training" unavailable as the main novelty, while
supporting the premise that physical workload signatures exist.

Rahman and Tajdari's [1 Hz telemetry study](https://arxiv.org/html/2606.19262v1)
classifies training using rich on-device GPU telemetry and includes multi-GPU and
adversarial configurations. It provides much broader real hardware evidence but
depends on a telemetry path controlled by or secured against the monitored operator.
This project's external channel remains differentiated by its trust boundary.

Industry-scale observations also precede a direct empirical result here. Choukse et
al. and Google's Balance of Power report synchronized power swings in large training
systems. Vercellino et al. provide an especially relevant public dataset for testing
the line-strength and scale prediction.

### 5.2 Signal-processing prior art

The detector components should not be presented as individually novel. Track-before-
detect, Viterbi or forward-algorithm line tracking, order tracking, spectral
surrogates, and cyclostationary tests each have mature literatures. A particularly
close reference is Suvorova et al.,
[*Phase-Continuous Frequency Line Track-Before-Detect of a Tone with Slow Frequency
Variation*](https://researchportalplus.anu.edu.au/en/publications/phase-continuous-frequency-line-track-before-detect-of-a-tone-wit/),
which combines hidden-state frequency tracking, phase continuity, and a
phase-wrapped Ornstein–Uhlenbeck process. The exact tracker–resampling–DG pipeline
may be new in this application, but the novelty claim should be integration and
verification use, not invention of a new general detector family.

### 5.3 The defensible novelty claim

The strongest novelty statement is the integration of:

- an independently observed, low-rate external measurement channel;
- explicit meter and transport requirements;
- a detector-aware evasion analysis;
- coherence and aggregation as training-system boundary conditions;
- statistical separation of structural evidence from conditional attribution;
- a formal semantic ceiling and a route to stronger active verification.

## 6. Simplifying the null-test story

The paper does not have too many true statistical nulls. It uses the word "null" for
several objects with different roles, making the method appear more complicated than
the reasoning requires.

These should be renamed and separated:

1. **Calibration surrogate:** a phase-randomized version of the same trace. It asks
   whether phase-coherent cadence remains after conditioning on the observed
   spectrum.
2. **Operational negative class:** held-out measured inference and non-training
   workloads. It asks whether the trace is more training-like than the declared
   alternative population.
3. **Confounder stress tests:** controller cycles, nonlinear loads, and other
   periodic processes. They ask what benign mechanisms can mimic the feature.
4. **Semantic counterexamples:** training-shaped execution without retained
   training. These establish an identifiability ceiling; they are not statistical
   nulls.

Only the first two need to be called null hypotheses.

The main paper should revolve around two reader-facing questions:

1. **Is there a coherent, drifting cadence?**
2. **Is the cadence more consistent with training than with the stated alternatives?**

For the first question, retain one simple fixed-frequency baseline and one tracked
detector. State the calibration result in one sentence: analytic calibration is
unreliable after adaptive tracking; trace-conditioned phase surrogates calibrate the
tested stationary backgrounds but do not protect against genuine cyclic confounders.

The following material can move to an appendix without weakening the contribution:

- the full four-stage by eight-background calibration matrix;
- most covariance and heavy-tail sensitivity detail;
- the detailed analytic $\chi^2$ failure demonstration;
- most detector-bank variants;
- the Whittle reference-detector grid.

The all-perfect synthetic Rung-2 AUC tables also add little credibility. They mostly
show that classes produced by the project's generator are separable by features
derived from that generator. A concise sensitivity result plus the semantic-decoy
failure would be stronger. The Whittle result should be called a **model-based
reference detector** or **Whittle-model ceiling**, not unqualified Neyman–Pearson
optimality.

The four-rung ladder should remain as a compact governance figure or table, but Rungs
3 and 4 should not compete with the empirical spine because neither protocol is run.
The narrative can be reduced to:

1. visibility of a physical cadence;
2. calibrated detection and conditional attribution;
3. evasion, coherence, and cost;
4. the semantic ceiling and need for active or cryptographic evidence.

## 7. Recommended revision order

The highest-value changes are:

1. **Reanalyse the public 8–64 GPU traces.** This is the fastest route to direct
   evidence about the project's central scale-and-coherence prediction.
2. **Replace the present asynchronous control.** Implement the multi-worker
   coherence model before describing robustness to asynchronous training.
3. **Add production power stabilization to the threat model.** Separate partial
   attenuation from complete hiding and throughput cost from infrastructure cost.
4. **Simplify the statistical story.** Use the four distinct labels above and move
   calibration diagnostics to the appendix.
5. **Compress self-generated classification results.** Lead with measured line
   properties and detector transfer once those are available.
6. **Report monitoring-oriented errors.** Add event-level false-alert and detection-
   delay analysis, or state clearly that it is deferred.
7. **Reframe the novelty.** Lead with the remote low-rate trust boundary, explicit
   channel requirements, adversarial frontier, and semantic limit.

A title such as **"When Can Datacentre Power Reveal Synchronous AI Training?"** or
**"Limits of Detecting Synchronous AI Training from Low-Rate Power"** would foreground
both the contribution and its central condition.

## Conclusion

The project should continue, but with a narrower empirical claim. Its value does not
depend on proving that every training system exposes one frequency line. A useful
verification result can instead identify the class of systems for which the line is
observable, measure how coherence and the observation channel control detectability,
quantify what hiding requires, and state what no passive power trace can establish.

With the current synthetic evidence, the paper is best understood as a rigorous
research agenda and methods framework. Public multi-node validation would make its
central prediction evidence-based. A simultaneous multi-GPU and external-meter
campaign—including asynchronous schedules and existing power stabilization—would
make the contribution substantially stronger and much more useful to verification
practitioners.
