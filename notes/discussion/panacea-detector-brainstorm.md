# Panacea detectors — brainstorm on a single jointly calibrated decision rule

**Date:** 2026-08-08
**Status:** open brainstorm; no experiments run, nothing here is a claim
**Prompted by:** the research-talk caveat (`feat/research-talk`, `talk/slides.md`):
*"We have not yet demonstrated one jointly calibrated decision rule that inherits
both strengths across the full frontier."*
**Constraint:** every candidate must remain passive — no access to the prover's
hardware; the observable is the external power trace alone.
**Related:** `notes/discussion/hsmm-duration-aware-postmortem.md` (the one prior
attempt at a frequency-free alternative, killed 2026-08-08 on
`feat/hsmm-renewal-smoke`).

---

## 1. The gap, stated precisely

From §5/§6 of the manuscript and the talk's strengths table:

| Method | Follows variable real work | Rejects controller-line confusers | Calibration story |
|---|---|---|---|
| Raw Viterbi score | strong | weak (locks onto any coherent oscillation) | selection over paths ⇒ no defensible analytic null |
| Order-domain DG tests | degrade earlier on the frontier | strong (specific fixed question after de-warp) | level-safe; surrogate-calibratable |

The published frontier is a detector-**class envelope** (best score per cell), not
one deployed decision policy. The gap therefore decomposes into **two different
gaps** that need different medicine:

1. **A statistic-level gap.** No single statistic has both properties. The common
   cause of the Viterbi score's two weaknesses is *maximization*: maximizing over
   frequency paths is what survives variable real work, and is also what locks
   onto the controller and destroys the analytic null (look-elsewhere).
2. **A decision-layer gap.** Even granting both detectors, no combined rule
   (AND/OR/gating/learned combiner) has been exhibited whose false-alarm rate is
   calibrated *including the selection among components*. The ST1 campaign showed
   why this is hard: no single covariance configuration calibrates the whole
   ladder, and the analytic tails are untrustworthy anyway.

The second gap is much lower-risk than the first.

---

## 2. Decision-layer candidates (gap 2)

### A1. E-value merging / universal inference — *shortest path to discharging the caveat*

Convert each detector's evidence into an e-value rather than a p-value. E-values
merge under arbitrary dependence (averaging is always valid; products when
independent), so "run the Viterbi tracker AND the order-domain test AND anything
else, then combine" becomes a **single jointly valid decision rule by
construction** — no per-component calibration compromise, no look-elsewhere
bookkeeping. Wasserman-style **universal inference** (split the record, estimate
the warp/path on one half, evaluate a likelihood ratio on the other) gives
finite-sample-valid tests under composite nulls *with estimated nuisances* —
precisely the estimated-warp selection effect that killed the analytic-CFAR
hope. The ST1 stage-3 held-out-warp design is already halfway there; the
machinery was built but never cashed in as an inference framework. Bonus:
e-processes are anytime-valid, a genuinely good property for a governance
monitor watching a meter continuously with optional stopping.

- **Cheapest kill experiment:** take the existing frozen ST1/ST2 detector
  scores, wrap each in a surrogate-derived e-value, average-merge, re-run the
  FAR grid and the frontier. Pure post-processing of frozen artefacts.
- **Risk:** e-values from surrogate calibration are "effective" e-values, not
  exact ones — this stacks the existing surrogate caveat rather than removing
  it. Merging is conservative; must check the merged rule actually holds the
  tracker's power at the work-variation cells.

### A2. Three-hypothesis classification with abstention ("model the confuser")

Stop treating the controller as a null to be robust against and put it **in the
model**: training vs inference vs controller, decided by posterior odds, with an
abstain / "cannot attribute" output when nothing fits (goodness-of-fit gate or
conformal out-of-model test). Directly addresses HSMM postmortem lesson 3 —
unseen confusers false-alarming at 0.985–1.000 is generic to empirically
calibrated composite thresholds, and abstention is the honest fix. It also
matches the paper's own theory: `prop:noident` says attribution has a ceiling,
so a verifier that *says so per-trace* is more defensible than one that
silently misfires.

---

## 3. Statistic-level candidates (gap 1)

### B1. Line-robust marginal Bayes factor — *transplant from continuous-GW searches; strongest thesis*

The continuous-gravitational-wave community faced literally this problem: a weak
wandering quasi-periodic signal, searched by maximization (the F-statistic;
Viterbi trackers à la Suvorova et al.), which then locks onto **instrumental
lines** — coherent narrowband confusers, the exact analogue of the power
controller. Their solution was a *single statistic*, not a bolted-on veto:
Keitel & Prix's **line-robust statistic**, a Bayes factor whose denominator is a
mixture of the Gaussian null *and* the line model:

    B = P(x | signal) / [ c1 · P(x | noise) + c2 · P(x | line) ]

Transplanted here: numerator = cyclostationary comb with a **marginalized** (not
maximized) phase path; denominator = linear-stationary null **plus** a Van der
Pol-class controller family. One scalar, jointly calibrated by construction; the
controller stops being a false alarm because evidence fitting the limit cycle
*raises the denominator*.

The other CW lesson comes free: marginalizing over the frequency path (the
B-statistic) dominates maximizing over it (the F-statistic) — and the repo
already holds independent evidence this principle works here: the one survivor
of the HSMM campaign was exactly "marginalize over segmentations, don't commit"
(+0.070 AUC, +0.134 TPR at composite FAR 0.05, 5/5 seeds). That result and the
CW literature point at the same move. **Thesis in one line: maximization is the
common cause of both weaknesses; marginalization over an explicit
confuser-inclusive model removes both at once.**

- **Cheapest kill experiment:** Rao-Blackwellised particle filter or coarse
  HMM-grid marginal likelihood over phase paths for three generative families
  already in `powerladder` (Ko–Zhu training, inference null, VdP controller),
  scored as a Bayes factor, surrogate-calibrated in the house style, run on the
  existing frontier cells. CPU-only.
- **Risk:** likelihood misspecification — the statistic is only as attributive
  as the controller family is representative. Mitigation: treat it as an
  effective score with per-trace surrogate calibration (the paper's existing
  epistemics) and keep A2's abstention gate.

### B2. Wavelet scattering — warp-stability without estimating the warp

The mathematical property required is *Lipschitz stability to
time-diffeomorphisms*, which is the defining theorem of Mallat's scattering
transform. Variable real work is a time-warp of the schedule; scattering
coefficients move little under warps, while second-order coefficients retain
exactly the waveform information that separates a phase-locked harmonic comb
with a communication dip from a smooth limit cycle. Crucially it is a **fixed,
non-adaptive representation** — no path search, no estimated warp, no selection
effect — so it calibrates as cleanly as fixed-α DG did (the one stage of the
ST1 ladder that was near-nominal). A genuine candidate for inheriting both
strengths in one object rather than gluing two objects together.

- **Cheapest kill experiment:** `kymatio` scattering features on the existing
  ST2 synthesis harness, a simple one-class or two-sample test on top, swept
  over the work-variation axis and the controller null. Hours of CPU; the RF
  bake-off baseline provides the comparison scaffold.
- **Risk:** warp stability is proven for smooth, small deformations;
  per-iteration renewal jumps at high work-variation levels may exceed the
  stability regime. Measurable — even the failure point would be a publishable
  boundary.

### B3. Phase-dynamics attribution: renewal versus diffusion

The deepest physical difference between the two "wandering oscillators" is not
the spectrum but the *law of the phase*:

- **Training:** cadence wander is a **renewal process** — i.i.d.-ish iteration
  durations ⇒ jumpy, piecewise-constant instantaneous frequency. Power level
  and iteration duration are *independent knobs* (schedule sets frequency;
  utilization sets amplitude).
- **Controller:** a limit cycle wanders **diffusively** (continuous OU-like
  phase noise), and a nonlinear oscillator has *deterministic
  amplitude–frequency coupling* (the Van der Pol constraint) that a schedule
  does not.

So: keep the Viterbi tracker, but make the **statistics of its estimated path**
the test — increment roughness / quadratic variation, jump-vs-diffusion
discrimination, instantaneous amplitude–frequency mutual information —
surrogate-calibrated. This converts the tracker from unattributed power into
attributed evidence: it fires on the controller's line, and the path dynamics
then say "that is an oscillator, not a schedule."

The HSMM postmortem does **not** block this: that campaign tested explicit
*duration* modelling of extracted events against a geometric-dwell control at
10 s records. This is a different question — the dynamics of the phase path
over the full record — and it composes with B1 (renewal phase prior in the
numerator, diffusion phase prior in the denominator makes B1 and B3 the same
statistic).

### B4. Maximal invariants / self-clocked statistics

The most radical framing: attacks are a group action (time reparameterisations,
dilution, shaping), so seek a **maximal invariant** under the warp group and
test on that — immunity to the cheap attack *by construction*, and invariants
tend to have cleaner nulls (classical UMPI theory). One concrete instance: use
the trace's own cumulative energy as the clock (**self-clocking**) — resample in
energy-time rather than estimated-frequency-time, no tracker, no path search,
then run the fixed DG test there. If work-per-iteration variation preserves the
waveform in energy-time (plausible: same power level, duration stretches), the
cheap attack is undone with zero adaptive estimation. Unknown whether it works —
which makes it a good afternoon experiment.

### B5. Warp-invariant kernel two-sample tests

Soft-DTW / alignment kernels; MMD against a null bank, permutation-calibrated.
Listed for completeness — clean calibration, but the physics gets buried in a
kernel choice. Ranked below B1–B4.

---

## 4. What NOT to revisit

- **Duration-aware event models (HSMM).** Killed properly with a
  duration-neutral control; the postmortem's own revival conditions (real
  distributed hardware, richer emission models) have not changed.
- **Generic learned classifiers without an abstention mechanism.** HSMM lesson 3
  will recur: anything empirically calibrated against an enumerated null suite
  false-alarms on the confuser that was not enumerated.

---

## 5. Recommended starting points

- **To discharge the talk's caveat:** A1. It is a calibration claim, it reuses
  frozen artefacts, and "we exhibit one anytime-valid merged decision rule over
  both classes across the full frontier" is exactly the missing experiment as
  the slide's speaker notes describe it.
- **For a successor paper's method:** B1 with B3's renewal-vs-diffusion priors
  inside it. It has a real thesis (maximization is the common cause of both
  weaknesses; marginalization over a confuser-inclusive model removes both at
  once), independent in-repo supporting evidence (+0.070 AUC from
  segmentation marginalisation), and a mature precedent literature (CW
  line-robust statistics; the Viterbi tracker already came from that world).

## 6. Open questions

1. **Which gap is the target** — a calibrated combination of the existing
   detectors (enough to delete the slide's caveat), or a genuinely new single
   statistic (a new results section / next paper)?
2. **How much likelihood modelling to buy?** B1 is the strongest candidate but
   commits to generative families for the confusers; B2/B4 stay nonparametric
   and closer to the paper's current epistemics.
3. **Does the CW analogy hold up on inspection?** Whether the
   line-robust-statistic transfer is as clean as it looks, and whether citing
   that lineage strengthens a follow-up's framing.
