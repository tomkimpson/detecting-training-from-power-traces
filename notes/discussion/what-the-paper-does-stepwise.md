# What the paper does, stepwise

**Date:** 2026-09-09
**Status:** pedagogical walkthrough (rationale record, not a tracker). Written in
response to "I am getting confused exactly what we have done and what we have shown
with all the different techniques, confounds etc." Scope lives in `spec.md`; status in
`handoff.md`; the detailed realisation in [`../plans/plan-for-paper-2.md`](../plans/plan-for-paper-2.md).
This is the document to re-read when the chain from physics to claim stops being
holdable in one head. Companion to
[`north-star-and-positioning.md`](north-star-and-positioning.md), which fixes the
spine and the inclusion test; this one explains the machine.

The framing choice that does the most work below: **the word "null" is doing four
different jobs in this paper**, and §5 separates them.

---

## 1. The problem

A regulator wants to know whether training is happening on hardware it doesn't
control. Every channel that could tell it has a catch: on-chip telemetry is reported
by the prover (forgeable), challenge-response needs the prover to play along. **A
power meter outside the prover's control is the only passive channel left.** So: what
can it actually certify?

## 2. The physical chain

Two equations carry the whole paper:

$$P_{\text{device}} = r(t)F(t) + P_0(t) \qquad P_{\text{meter}} = (h * P_{\text{device}}) + P_{\text{other}} + \eta$$

Computation happens at rate $F(t)$, costs $r$ joules per op, plus overhead. Then the
**channel** mangles it: $h$ is power-delivery dynamics, meter integration, filtering,
sampling. $P_{\text{other}}$ is everything else behind the same meter. $\eta$ is
noise.

The key structural point: **the verifier never sees $F(t)$.** It sees a low-passed,
integrated, decimated, contaminated scalar. Much of the paper is about what survives
that.

## 3. What creates a signal at all

Synchronous distributed training is a **two-phase cycle**: compute at high op-rate,
then an all-reduce communication phase at low op-rate. Repeat. That alternation puts a
**periodic line** in the power spectrum at the iteration cadence $f_0$ — for realistic
setups, ~0.5–1.5 Hz.

That line is the *only* thing you're detecting. Everything downstream is about finding
it, defending it, or hiding it.

**Why it's hard:** the cadence isn't stable. Real runs drift with data, checkpointing,
contention. You model this as a slow Ornstein–Uhlenbeck walk of the centre frequency —
deliberately as a *realism correction, not an attack*. That matters: the line wanders
even against a completely honest prover.

## 4. What you're simulating, and why

No real cluster traces exist, so you build a generator:

| Piece | What it is | Where it comes from |
|---|---|---|
| Training waveform | Two-phase cycle, $f_0 \sim U(0.5,1.5)$ Hz, per-iteration period jitter, phase-ratio draws | **Ko & Zhu's** published model, adapted |
| Inference null | Decode floor (continuous batching), OU envelope (MoE expert imbalance), Poisson prefill bursts | **Yours**, from the LLM-serving literature |
| Attack operators | Work variation, phase slip, harmonic smoothing, shape filling | **Yours**, implemented *inside* the generator |
| Observation map | Filtering, integration, decimation, notch, noise, baseline wander | **Yours** |

Two design choices worth noticing because they do a lot of work:

- **Attacks live inside the generator, not as post-hoc filters.** Work variation is
  implemented as varying the number of gradient-accumulation micro-steps $G_i$ about
  its honest value of 8 — and crucially the communication phase keeps its honest
  duration, because *a synchronous exchange doesn't shrink just because the preceding
  compute phase was shorter*. That physical constraint is why the attack turns out to
  be nearly free, and you'd never discover it by filtering a trace afterwards.
- **The observation map's default is an exact identity** — every axis off returns the
  trace unchanged and consumes zero randomness. That's what makes one-axis-at-a-time
  sweeps interpretable.

## 5. The four things called "null"

They are genuinely different objects with different jobs:

| # | Name | What it is | Question it answers |
|---|---|---|---|
| 1 | **Structural nulls** | Stationary processes: white, AR(1), resonant AR(2), heavy-tailed, skewed | *Does my test fire when nothing periodic is there?* (level safety) |
| 2 | **Controller confound** | A **measured** A100 power-management limit cycle, 0.31–0.45 Hz | *Does my test fire on a real periodic thing that isn't training?* |
| 3 | **Inference null** | Modelled LLM serving | *Is this trace more training-like than serving?* (Rung 2) |
| 4 | **Semantic controls** | Non-training loads with training-shaped schedules (incl. the discarded-update decoy) | *Does "training-shaped" mean "training"?* (the ceiling) |

They escalate in difficulty, and each one kills a different claim. #1 is a calibration
check. #2 is a *physical* confound you measured on real hardware. #3 is the reference
class for classification. #4 is the philosophical limit.

## 6. Building the detector, by necessity

Each step exists because the previous one demonstrably failed:

**Step 1 — Test for a line.** Cadence is periodic, so use the Thomson multitaper
harmonic F-test. Classical, pivotal under a smooth stationary Gaussian null.

*Fails:* the line wanders, smearing across the band. Bake-off detection rate
**0.01–0.02**. Essentially dead.

**Step 2 — Give it the answer.** Try the fixed-$\alpha$ Dandawaté–Giannakis
cyclostationary test, *handed the true $f_0$ as an oracle*.

*Fails:* 0.34 at zero drift, ≤0.1 under any wander. **Even knowing the true frequency
doesn't help**, because a coherent fixed-frequency test needs the line to *stay put*,
not merely be known.

**Step 3 — Track the wander and undo it.** Three components:

1. **Viterbi tracker** estimates the instantaneous frequency path $\hat f_0(t)$
   through the time-frequency plane (borrowed from passive sonar line-tracking and
   continuous-gravitational-wave searches).
2. **Phase resampling**: $\hat\theta(t) = 2\pi\!\int\!\hat f_0$, resample at equal
   increments of $\hat\theta$ — tacholess order tracking. A wandering line becomes a
   *constant-order* line.
3. **DG statistic** on the de-warped trace.

The tracker is a **nuisance-parameter estimator, not the detector.** This is the
single most important architectural point and it's where to focus when re-reading.

**Step 4 — Discover that tracking alone still isn't enough.** Against the structural
mixture (AR(1) + heavy-tailed + controller), the **raw Viterbi score scores 0.00 at
every drift** — same as the fixed tests. A Viterbi tracker locks happily onto a limit
cycle. Only *order-coherent* testing after resampling survives: DG order
(semi-coherent) holds 1.00 → 0.69 out to 0.4 Hz drift.

So the real finding isn't "tracking beats fixed." It's: **phase coherence at the
tracked order is the one signature the confuser can't fake.**

## 7. The calibration problem

You now have a detector that searches over frequency paths, estimates a warp from the
same record it tests, and uses a plug-in covariance estimator. The asymptotic $\chi^2$
tail is worthless here. So you did two things:

**A staged false-alarm campaign** — four stages adding one adaptive ingredient at a
time, $10^4$ null traces per cell, Clopper–Pearson intervals. The results are clean
and slightly surprising:

- No null inflates the rate anywhere. The failure mode you were hunting doesn't occur.
- **Resampling deflates by 3.4×** (0.056 → 0.016): interpolation correlates adjacent
  samples, the Bartlett estimator reads that as dependence, the statistic shrinks.
- Estimating the warp on held-out blocks moves it **nowhere** (0.016 → 0.016).
- Full Viterbi path selection adds **nothing** (0.016 → 0.018). Even on the resonant
  AR(2) that the tracker provably locks onto, it fires **never in 10⁴ traces** —
  locking onto a noise resonance doesn't manufacture phase coherence.

**Surrogate calibration** — for each trace, randomise the Fourier phases while keeping
the modulus, rerun *the entire pipeline* on each surrogate, report the observed
statistic's rank. Running the whole pipeline inside the loop is the point: it makes
the path search part of the null rather than unmodelled freedom.

**The honest catch, stated up front:** surrogates encode the invariance "linear and
stationary with this spectrum." A genuinely cyclostationary null isn't covered — and a
controller limit cycle is exactly that. Which is why the controller false-alarm rate
rises to **0.17–0.30** at nominal 0.05. Not a bug. It's the attribution boundary: the
detector correctly reports coherent cyclic structure, and calling that structure
*training* is a causal step power alone can't take.

**Net result:** you don't claim an analytic CFAR detector. You claim an effective
score with per-trace surrogate calibration, and you say exactly what it's exact
against.

## 8. How good is "good"? — the ceiling

Detector comparisons are meaningless without a scale. Since you own the generators,
you built a **Whittle spectral likelihood-ratio ceiling** with a Monte-Carlo template
bank marginalised over cadence, fitted on a disjoint stream.

The ceiling is **perfect** — AUC 1.0 in every column at every drift. So the problem is
separable in principle from a 300 s trace, and all the difficulty lives in the
detector, not the physics. Against the inference null the Viterbi tracker is
**essentially Neyman–Pearson optimal**.

This converts "our detector worked" into "no detector would have done better," which
is what makes the frontier meaningful.

## 9. Rung 2 — classification, and its ceiling

Reduce each trace to 8 physics features (6 detector scores + 2 read off the estimated
Viterbi path: frequency-path stability, phase-folded waveform repeatability). Three
decision rules of increasing freedom — a *prespecified* physics score, a fitted
logistic, and a random forest included specifically to expose how much separation
comes from the null rather than the method.

**Result: AUC 1.00, stable zero-shot across six domain shifts.**

**Then the ceiling.** Seven loads whose training status you *know*. Five non-training
ones — a training-shaped non-ML kernel loop, a controller limit cycle, a periodic
inference request generator, a gradient-only computation, and a **discarded-update
decoy** that runs the full forward/backward/optimizer step and throws the update away
— **all score as training at fraction 1.00 under every rule.**

Three of those five need no adversary at all; they occur in ordinary operation. So the
meter certifies *a training-shaped periodic compute schedule*, not that the
computation is training. And this cuts both ways: it caps what a positive detection
proves, not just what a fake could fake.

## 10. The adaptive prover — the frontier

Now the prover knows your detector. Six attack families, each stepped through attack
levels, each priced against measured single-A100 throughput anchors. The result is
**asymmetric**, which is the paper's central finding:

- **Attacks that hurt the line are ones nobody would buy.** Idle-insertion jitter:
  15–375% throughput, line still detectable. Drift: ~140%, and it's exactly what the
  order tracker was built for. Power shaping: 170–230%.
- **The one cheap attack: vary real work per iteration.** Measured overhead −0.10% and
  −0.36% — zero within noise. At that price the fixed tests are *gone*: the matched
  filter's AUC collapses to 0.44 and 0.41, **below chance**. And the tracker holds at
  1.0.

Pareto view: the fixed class gives up hiding 0.74 at ≈0% cost; the tracking class
never leaves the floor, 0.16 at 159%.

**Plus the meter itself is an attack surface.** A prover who can't de-periodicise
cheaply might instead argue for a coarse meter. So you swept the instrument: minimum
spec is **≥2 Hz sampling AND ≤0.5 s integration AND no deep in-band notch**. A 1
s-integrating 1 Hz meter defeats everything on an honest workload.

## 11. The theory rung — cost of hiding

Everything above bounds the *wrong direction*: a detection sweep says what can be
hidden from **your** detector, which a better detector moves. Governance needs the
other direction.

So: how far must the prover distort before the training-trace *distribution* is
indistinguishable from the null *distribution*? Distributional distance, Pinsker,
routed through a verifier's **attained** advantage ($J \le \mathrm{TV}$, Youden
index). Gives $\sigma^*(\varepsilon)$ — required distortion for detectability
$\varepsilon$ — priced at tens of percent of throughput for $\varepsilon = 0.5$, order
100%+ for $\varepsilon = 0.2$.

A nice bit of physics fell out: carrying the Lorentzian/bin algebra through properly
gives a **parameter-free** $\kappa = \pi^2 f_0 T = 2961$, matching the numerical fit
to 2%.

**Honest limits, stated:** one attack family; it's read off a fixed-bin test so it's
valid against any class containing that test but **loose** against your own tracker;
and the cost leg is an *assumption*, not a result.

## 12. What's real, and what isn't

Everything above is synthetic. The one place you check against hardware, it fails
instructively, and you report it as a result:

- On a saturated A100 the cadence is present but **shallow** — ~0.4% of board power,
  against Ko & Zhu's assumed ~30%. A **75× gap**.
- Worse, the spectrum carries a 0.31–0.45 Hz line **~10× stronger than the training
  line** — the card's own power-management loop. It says "time-varying load," not
  "training."

One card has no communication barrier, so the compute/communication contrast the whole
model rests on is absent by construction. That's why the distributed deepening is
stated as a **falsifiable prediction**, not a finding.

---

## The one-paragraph version

*Synchronous training puts a wandering periodic line in the power spectrum.
Fixed-frequency tests can't see it even when handed the true frequency; tracking the
wander and testing order-coherence can, and that's provably near-optimal against a
serving null. But the test can't be calibrated analytically, so it's a
surrogate-calibrated effective score, and it fires on power-management limit cycles
too — so it certifies a training-shaped schedule, never training itself. An adaptive
prover can erase the line by varying real work at essentially zero cost, though only
against fixed detectors; erasing it against the tracker costs real throughput, and
there's a lower bound saying roughly how much. All of it is conditional on a meter
that samples fast enough and integrates briefly enough, and all of it is modelled —
the one real card shows the signature 75× shallower than assumed.*
