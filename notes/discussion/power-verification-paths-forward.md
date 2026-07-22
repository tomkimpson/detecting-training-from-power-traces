# Detecting Training from External Power: Paths Forward

**Status:** research-direction memo, 2026-07-21

**Manuscript considered:** `paper-verification/main.tex`

**Scope:** physics-, signal-analysis-, and engineering-based approaches to distinguishing training from inference when the verifier has only a noisy external power meter and may additionally have cryptographically verifiable declarations of work.

## Executive recommendation

Keep the Viterbi tracker, but demote it from the conceptual centre of the project to one component in a broader verification stack. Viterbi solves cadence wander, which is a real but secondary nuisance. It does not solve the harder questions:

1. whether semantic training is identifiable from a scalar power trace;
2. how to control false alarms without a representative hardware- and site-specific null;
3. whether the on-die signature reaches an external meter; or
4. how to bind observed physical activity to declared computational work.

The most promising **passive** technical direction is a self-normalised cyclostationarity and harmonic-order detector. It should test for repeated multi-phase computation rather than for one favourable spectral ridge. Viterbi, or another ridge estimator, can remain as the mechanism for estimating instantaneous cycle phase when the cadence moves.

The strongest direction **when verifiable declarations are available** is challenge-synchronous detection: bind an unpredictable verifier challenge to declared work, then recover its physical response with lock-in/spread-spectrum correlation. In this formulation the null is supplied by the verifier's randomisation, not by a collection of traces from matching hardware.

The paper should distinguish three levels of claim:

- **physical detection:** a cyclic or otherwise structured computation is present;
- **conditional classification:** under a stated hardware, meter, and workload population, the trace resembles training more than the specified null;
- **work verification:** the measured physical activity is bound to declared training work.

Power alone can establish the first. The second needs a representative null or explicit structural assumptions. The third needs a protocol that combines physical measurement with commitments, proofs, or challenges.

## 1. The fundamental identifiability limit

The observation model is many-to-one. Let $w$ denote a computational execution, $h$ the hardware and power-delivery path, and $K_h(w)$ the probability law of the observed power trace. A passive detector $\phi(P_{\mathrm{obs}})$ cannot distinguish two executions whenever

\[
K_{h_0}(w_0)=K_{h_1}(w_1).
\]

In that case its probability of declaring training is identical under the two executions. Hardware uncertainty enlarges the set of observationally equivalent pairs.

There is a nearly constructive same-hardware counterexample. A nominally non-training program can execute the same forward, backward, and optimiser kernels as training and discard each updated state. Its physical execution can be made arbitrarily close to genuine training while its semantic classification differs. Conversely, genuine training can be scheduled asynchronously, diluted among other work, or actively power-shaped.

This means that no passive scalar-power detector can offer a universal semantic guarantee against an unrestricted adversary. A theorem or operational guarantee must restrict at least one of:

- the admissible non-training workload class;
- the adversary's ability to add or restructure computation;
- the hardware and power-transfer family;
- the physical property being detected; or
- the protocol, by adding a trusted challenge or verifiable declaration.

This is not a reason to abandon power verification. It is a reason to state the estimand correctly. A line, harmonic comb, or cyclostationary feature certifies **recurrent physical activity**, not the intention to update a model.

The manuscript currently makes the stronger reduction that a training iteration has a clock, inference does not, and therefore the type question reduces to detecting a persistent line. That is useful as a model for the selected continuous-batching null, but it is not universal:

- fixed-batch or benchmarked inference can be periodic;
- token decoding, request generators, schedulers, cache maintenance, and control loops can introduce cadences;
- gradient calculation without retained updates is physically training-like;
- asynchronous or pipeline-parallel training need not expose one narrow stationary cadence; and
- the measured A100 full-band separator is already a controller oscillation that means "time-varying load," not "training."

The appropriate passive hypothesis is therefore closer to:

> Is there evidence of coherent, repeated, multi-phase computation within a specified time-scale family?

Training is then an important cause of that property, not a logically equivalent name for it.

## 2. What Viterbi does and does not buy

The current construction is sensible as a **track-before-detect** heuristic:

1. make a short-time power spectrum;
2. normalise each frequency column by its local median;
3. reward a high-contrast path;
4. penalise rapid frequency movement; and
5. take the best path score.

This is well matched to a narrow line whose frequency wanders slowly. The sonar and continuous-gravitational-wave lineage is legitimate, and it is useful insurance against a cadence-smearing adversary.

However, the tracker is solving the wrong layer of the hardest problem:

- It estimates one favourable path but does not show that the path is training-specific.
- Its median normalisation removes an unknown broadband scale but does not produce a hardware-independent false-alarm threshold.
- The maximum over paths creates a look-elsewhere effect whose distribution depends on the number of bins and frames, spectrogram correlations, coloured background, and transition penalty.
- A fixed value $\lambda=1$ is tied to the chosen frequency grid and frame duration. A physical prior should instead be expressed in quantities such as Hz/s.
- The statistic uses only ridge amplitude and continuity. It discards cycle phase, harmonic relationships, and repeatable within-iteration waveform shape.
- On measured honest hardware, the iteration line is stationary and a fixed-bin integrator ties the tracker. Its distinctive measured role appears only under an implemented smearing adversary; the clean natural-wander result remains model-based.

There is also a modelling-language issue. The recursion has the algebraic shape of Viterbi, but the emissions are median-normalised log periodogram values rather than derived log likelihoods, and the movement penalty is not a calibrated log transition probability. It is therefore better described as an HMM-inspired penalised ridge score than literally as a MAP path under a specified generative model.

The resulting conclusion is not that Viterbi is a bad method. It is that it should be one candidate phase/ridge estimator inside a detector that uses more of the mechanical signature.

## 3. Passive paths that reduce reliance on a hardware null

No passive method is genuinely null-free. What is possible is to replace a semantic null such as "all inference on this hardware" with a narrower structural null whose nuisance parameters can be estimated from the trace itself.

### 3.1 Multitaper harmonic testing

The cheapest immediate baseline is Thomson's multitaper harmonic F-test. Orthogonal discrete-prolate tapers provide several approximately independent views of the same data. A coherent sinusoid contributes consistently across tapers; the local stochastic background appears in the residual. The resulting F-statistic tests a line while eliminating the unknown local noise scale under a stationary, locally smooth spectral model.

This is stronger than the current peak Welch-PSD comparator because it:

- reduces leakage from large low-frequency power variation;
- distinguishes a coherent line from a locally elevated continuum;
- has an explicit statistical line test;
- provides a better emission statistic for a moving-line tracker; and
- can be applied to a harmonic family rather than only one bin.

It will not be robust to arbitrary nonstationary clutter, and searching an unknown frequency still needs a multiple-search correction. Nevertheless, it is the first detector to implement because it is classical, interpretable, inexpensive, and directly testable on the existing traces.

### 3.2 Cyclostationary detection

The physical schedule of an iteration can modulate more than the mean power. It can cause the variance, autocorrelation, and spectral content to vary periodically. This is a second-order cyclostationary signature even when the first-order spectral line is shallow.

For a candidate cycle frequency $\alpha$, estimate cyclic correlations over several lags,

\[
\widehat R_x^\alpha(\tau)
=\frac{1}{N}\sum_n x[n]x[n-\tau]e^{-2\pi i\alpha n}.
\]

Stack their real and imaginary components into $\widehat r_\alpha$, estimate their long-run covariance $\widehat\Sigma_\alpha$ from the same record, and form a self-normalised quadratic statistic of the form

\[
Q_\alpha=N\widehat r_\alpha^\top
\widehat\Sigma_\alpha^{-1}\widehat r_\alpha.
\]

Dandawaté and Giannakis derive asymptotic chi-squared, constant-false-alarm tests of this kind for cyclic moments and cumulants without specifying a particular data distribution. This does not give finite-sample distribution-free validity: consistency, asymptotic normality, and weak-dependence conditions still matter. It does, however, address the nuisance that currently forces calibration on A100 serving traces.

Cyclostationarity is especially attractive for the external-meter setting:

- stationary additive noise has no cyclic spectrum at nonzero cycle frequency;
- an unknown linear time-invariant transfer function transforms the cyclic spectrum but generally does not erase it except at transfer-function zeros;
- second-order structure can remain when the mean line is small or amplitudes vary between iterations; and
- multiple cyclic frequencies and lags can express a repeated multi-stage schedule.

Time-varying or nonlinear power-control loops remain a serious nuisance. Their presence is not a reason to reject the method; it is precisely why the null must be phrased as a structural claim and stress-tested against controller cycles.

### 3.3 Harmonic and phase-coupling tests

An iteration is a repeated waveform rather than a sinusoid. A detector should exploit the family

\[
f_0,\;2f_0,\;3f_0,\ldots
\]

and, where measurable, the phase relationships among those components. Candidate statistics include:

- a multitaper harmonic-comb F-statistic;
- cyclic spectral coherence at several cycle frequencies;
- bicoherence or related polyspectral measures of quadratic phase coupling;
- the proportion of total cyclic energy explained by one fundamental and its orders; and
- repeatability of a phase-folded cycle waveform.

This is a plausible way to distinguish a multi-phase iteration from the A100's nearly single-tone control oscillation. It still cannot distinguish training from an adversarial periodic decoy, because any such decoy can copy the harmonic structure at a cost. That cost should be measured rather than assumed.

### 3.4 Tacholess order tracking

Rotating-machinery diagnostics solve a closely analogous problem: a machine creates harmonically related vibration features, its speed changes, and a tachometer may be unavailable. Tacholess order tracking estimates instantaneous rotational frequency from the vibration itself, recovers phase, resamples into the angle domain, and then measures order-coherent structure.

Applied here:

1. estimate $\widehat f_0(t)$ from a time-frequency ridge;
2. form phase $\widehat\theta(t)=2\pi\int^t\widehat f_0(u)\,du$;
3. resample $P_{\mathrm{obs}}(t)$ at equal increments of $\widehat\theta$;
4. stack the reconstructed iteration cycles; and
5. score the phase-dependent mean, variance, harmonics, and residual repeatability.

The Viterbi tracker can supply step 1. Chirplet ridge estimation, synchrosqueezing, or a Vold–Kalman filter are alternatives. The important change is that the final evidence comes from a de-warped repeated waveform, not merely from the ridge used to estimate its phase.

Because the phase is estimated from the same signal, a naive repeatability test will overfit noise. Defensible implementations should use sample splitting, for example estimating phase from the fundamental on one set of tapers and testing higher harmonics or held-out frames, or should include the phase-estimation operation inside the null/randomisation procedure.

### 3.5 Within-trace surrogate and randomisation tests

Some nuisance calibration can be obtained from transformations of the observed record:

- locally rotate or permute frequency bins to destroy cross-frame ridge coherence while preserving each column's marginal spectrum;
- scramble iteration phase while preserving the ordinary PSD;
- use Fourier-phase surrogates to preserve the power spectrum while destroying higher-order phase coupling; or
- permute independent blocks to destroy a drifting path's continuity.

These tests can be exact under the corresponding invariance assumption, but their null is the invariance itself, not "inference." Frequency-bin permutation, for example, requires local exchangeability after whitening; Fourier-phase surrogates test a linear stationary-process null. They are useful sensitivity analyses and possible per-trace calibrators, provided the paper states what each transformation assumes.

## 4. A better tracking detector architecture

If a wandering-line detector remains central, replace the current arbitrary contrast score with a likelihood- or e-value-based architecture.

### 4.1 Calibrated emissions

For each non-overlapping frame and candidate frequency, compute an emission with a known or self-normalised null:

- a multitaper harmonic F-statistic;
- a local CFAR statistic with guard cells and a robust order-statistic background estimate; or
- a cyclostationary significance statistic.

Express transitions as a probability law over physical frequency slew rather than as a penalty in bin units.

### 4.2 Marginalise over paths for detection

Viterbi finds the single best state path. For detecting whether a member of a composite wandering-line family is present, the HMM forward recursion is more natural:

\[
L_{1:J}=\sum_{k_{1:J}}\pi(k_{1:J})
\prod_{j=1}^J L_j(k_j).
\]

This integrates evidence over possible paths instead of selecting the most favourable one. With valid conditional likelihood ratios or e-values as emissions, the mixture can automatically account for the path search and may itself support an anytime-valid test. The exact guarantee depends on conditional validity and temporal dependence; overlapping STFT frames should not simply be treated as independent observations.

The useful comparison is therefore not only "Welch peak versus Viterbi." It is:

1. multitaper fixed-line test;
2. maximum-path tracker;
3. forward/marginal track detector;
4. cyclostationary test; and
5. order-coherent, phase-folded detector.

## 5. The strongest route with declared work: challenge-synchronous measurement

If the prover can produce commitments or zero-knowledge proofs about its work, the verifier can create a reference signal. This changes the problem from blind classification to system authentication.

### 5.1 Basic protocol sketch

1. The prover commits to a job, initial model state, and permissible training program.
2. The verifier emits an unpredictable, balanced challenge sequence $c_k\in\{-1,+1\}$.
3. The program incorporates a small, harmless, provably equivalent schedule variation determined by $c_k$. Possibilities include two equivalent microbatch partitions, randomised but semantics-preserving barrier placement, or committed marker operations.
4. A proof or verifiable transcript binds model updates to the challenges.
5. The verifier measures external power and evaluates its cross-correlation with the challenge over an allowed causal lag window:

\[
C(\tau)=\sum_k c_k\bigl(P_{k+\tau}-\widehat P_{\mathrm{local},k+\tau}\bigr).
\]

Under the null of no challenge-bound physical response, the verifier's random challenge is independent of the power process. Randomisation over $c_k$ can therefore calibrate the test without a hardware-matched inference corpus. This is the measurement analogue of a lock-in amplifier or direct-sequence spread-spectrum receiver: unrelated loads average away, while challenge-coherent response accumulates approximately with $\sqrt N$ processing gain.

An unknown stable transfer function can be handled by searching a bounded causal impulse-response family or using challenge-to-power coherence, with the search included in the randomisation test.

### 5.2 What this would and would not prove

The construction establishes that challenge-responsive physical activity occurred behind the meter. The cryptographic component establishes that a declared computational transcript is internally valid. Binding the two remains the crucial systems problem.

A prover might generate the proof on remote hardware while driving a dummy challenge-correlated load at the metered site. Mitigations could include:

- low-latency unpredictable nonces;
- challenge-dependent data or model commitments;
- timestamped intermediate-state commitments;
- random audits of selected update steps;
- limits on the response lag consistent with local execution; and
- multiple independently placed meters or network observations.

Without such binding, the combined evidence is "a valid proof exists and a responsive load exists," not necessarily "the proved work consumed this measured power." This limitation should be treated explicitly.

### 5.3 A simpler near-term version

Committed or verified step timestamps would already be valuable. They act as a tachometer:

- fold the power trace at the declared step phase;
- coherently average many iterations;
- test the phase-resolved waveform against randomly shifted timestamps; and
- compare the observed phase response with declared batch-size or schedule changes.

The timestamps must be committed before the verifier observes or reveals the relevant meter data; otherwise the prover can select timestamps post hoc to fit chance fluctuations.

## 6. The sequential guarantee needs repair

The present sequential procedure ranks each block score against one fixed finite calibration set, applies the p-to-e transform $e=\kappa p^{\kappa-1}$, and multiplies the results. A conformal rank p-value is marginally superuniform for one exchangeable test point. Reusing the same realised calibration sample does not automatically imply

\[
\mathbb E[e_j\mid\mathcal F_{j-1}]\leq 1,
\]

which is the conditional property required for the product to be a nonnegative supermartingale. Shared calibration also induces dependence among the p-values. Empirical false-alarm performance on bootstrap chains is useful evidence but does not prove Ville-style anytime control.

Possible repairs include:

- a genuine online conformal test martingale with randomised sequential ranks;
- predictable per-block e-values whose nuisance estimates use only past data;
- a sequential two-sample e-process comparing a trusted reference stream with the monitored stream;
- an HMM forward likelihood/e-value ratio with a fully stated structural null; or
- limiting the result to a fixed horizon and using a valid finite-sample randomisation test.

There is no purely statistical repair for a misspecified semantic null. A mathematically valid exchangeability test detects departure from exchangeability or from a reference distribution; it does not turn that departure into proof of training.

## 7. Meter and transfer-function issues

The die-to-meter path is part of the hypothesis, not a final implementation detail. The external observation is better written as

\[
P_{\mathrm{meter}}(t)
= (h_{\mathrm{PSU}}*P_{\mathrm{die}})(t)
+P_{\mathrm{other}}(t)+\eta(t),
\]

where $h_{\mathrm{PSU}}$ includes voltage regulation, energy storage, power-factor correction, meter integration, and reporting filters. It may be operating-point dependent and therefore not strictly linear or time invariant.

The current 1 Hz conclusion is an empirical result about the captured NVML-derived population, not a general instrument requirement. A one-second boxcar integrator has response

\[
H(f)=e^{-i\pi f\Delta}\operatorname{sinc}(\pi f\Delta),
\qquad \Delta=1\ \mathrm{s},
\]

with zeros at integer hertz, while 1 Hz sampling has a 0.5 Hz Nyquist frequency. It cannot uniformly preserve or identify an arbitrary line in a 0.3--1.7 Hz band. Aliased detection may work for particular populations, but a line near a boxcar zero can disappear and out-of-band clutter can alias into the detection region.

The correct near-term experiment is simultaneous NVML and external voltage/current measurement while commanding:

- sine or chirp duty modulation;
- pseudorandom binary load modulation;
- honest training and serving;
- a controller-cycle excitation; and
- active shaping.

This estimates the end-to-end transfer, coherence, noise floor, nonlinearities, and operating-point dependence. The experiment should precede claims about the minimum meter cadence.

## 8. Energy is complementary, but it is not a positive type detector

If $\eta_{\max}$ is a defensible upper bound on achievable compute per joule, metered energy gives

\[
C\leq \eta_{\max}E.
\]

It can therefore rule out a compute amount when insufficient energy crossed the boundary. High energy cannot prove that a large amount of training occurred: the energy may belong to inference, cooling, padding, or unrelated co-resident work.

Similarly, "energy per trained token" is not directly available to a verifier unless training progress or the token/update count is independently proved. In the experiments it is known from ground truth, which is appropriate for measuring an attack's cost, but a deployment claim should separate:

- **economic deterrence:** evasion consumes more energy and loses throughput;
- **amount exclusion:** low total energy rules out high compute under an efficiency bound; and
- **work-linked accounting:** energy divided by cryptographically verified progress.

The third is the strongest combination of the power and proof channels.

## 9. Highest-value falsification experiments

The immediate aim should be to discover what the detector actually identifies, not to accumulate more positive demonstrations.

### 9.1 Semantic-identifiability controls

1. **Discarded-update decoy:** execute forward, backward, and optimiser kernels but discard each update.
2. **Gradient-only workload:** calculate gradients for evaluation or analysis without training a retained model.
3. **Periodic inference:** fixed batches or a periodic request generator tuned to the training cadence.
4. **Training-shaped non-ML loop:** reproduce the same compute/memory/communication phase sequence with unrelated kernels.
5. **Asynchronous training:** deliberately remove the global step clock without inserting idle gaps.

If the detector calls the first four training, it is a detector of training-like physical work. That may still be governance-relevant, but it changes the claim. If it misses the fifth, the line is not necessary for training.

### 9.2 Hardware and nuisance controls

1. Repeat across power caps, clock locks, temperatures, drivers, and GPU generations.
2. Excite controller cycles with non-training loads.
3. Add periodic CPU, network, storage, fan, and co-resident GPU loads.
4. Test aggregate feeds with several independent workloads.
5. Compare on-die telemetry, cable current, PSU input, PDU, and wall measurements simultaneously.

### 9.3 Detector comparison on existing data

At the same operating point and with untouched test populations, compare:

1. Thomson multitaper harmonic F-test;
2. the current Welch peak;
3. current Viterbi score;
4. forward/marginal track score;
5. second-order cyclostationarity test;
6. harmonic-comb/cyclic-coherence score; and
7. tacholess phase-folded repeatability.

Report performance separately for:

- stationary honest lines;
- naturally or deliberately wandering lines;
- the A100 controller line;
- periodic non-training decoys;
- dilution and power shaping; and
- external-meter transformations.

The most important metric is not mean ROC area on the current null. It is the false-positive/detection trade-off on the semantic counterexamples and across hardware transfer.

## 10. Suggested staged research programme

### Stage A: reanalyse without new hardware

1. Implement the multitaper harmonic F-test.
2. Implement the Dandawaté--Giannakis second-order cyclic-correlation test over a small lag set.
3. Use the existing Viterbi ridge to phase-fold traces and calculate held-out harmonic repeatability.
4. audit the e-process proof and replace the fixed-calibration product or narrow its claim;
5. run surrogate-data tests that preserve the PSD but destroy phase coupling.

This stage tests whether the current conclusion depends on the chosen detector and whether richer physics exists in the recorded channel.

### Stage B: broaden the negative class

Run the semantic-identifiability controls above, especially discarded-update compute and periodic inference. This is likely more informative than collecting another ordinary serving workload.

### Stage C: measure the actual channel

Acquire simultaneous NVML and external power data, identify the transfer function with chirp/PRBS loads, and repeat the most informative positive and negative controls. Until this is done, external-meter claims should remain predictions.

### Stage D: challenge-bound pilot

Use a simple committed schedule rather than a full proof-of-training system. Generate a random marker sequence, execute two semantically equivalent schedules, and measure challenge-to-power coherence. Quantify required modulation depth, observation time, and the cost of dummy-load mimicry.

### Stage E: distributed-scale validation

Only after the external transfer is understood, test whether collective communication creates a deeper harmonic/cyclostationary signature and whether cadence de-warping or Viterbi tracking helps on real multi-GPU runs.

## 11. Recommended manuscript framing

A stronger paper would ask:

> What can a verifier certify from an untrusted computer's external power trace?

The answer would be layered:

1. **Passive structural evidence:** self-normalised cyclostationary and order-coherence tests detect repeated computation under explicit nuisance assumptions.
2. **Conditional classification:** calibration against a broad, stated null supports training-versus-inference conclusions for that deployment domain.
3. **Active physical authentication:** unpredictable challenge modulation and synchronous detection avoid a hardware-specific empirical null.
4. **Cryptographic binding:** commitments or proofs connect declared updates to the challenge transcript.
5. **Energy accounting:** integrated power constrains the possible amount of computation and prices masking, especially when progress is independently verified.

Within that framing, Viterbi has a clean and defensible role: it is one way to recover a moving cycle clock so that cyclic evidence can be integrated coherently. It need not carry the semantic or statistical burden of the entire verification claim.

Claims in the current draft that should eventually be narrowed or qualified include:

- "the type question therefore reduces" to persistent-line detection;
- "no training data" without immediately adding "but null calibration is required";
- the literal MAP interpretation of the current Viterbi emissions;
- anytime-valid false-alarm control from the reused fixed calibration set;
- a general 1 Hz external-meter specification;
- treating the A100 power-management oscillation as training evidence; and
- treating integrated energy by itself as a positive detector of training amount.

## 12. Selected technical sources

- D. J. Thomson, ["Spectrum Estimation and Harmonic Analysis," *Proceedings of the IEEE* 70(9), 1982](https://doi.org/10.1109/PROC.1982.12433). Multitaper spectrum estimation and an analysis-of-variance test for line components.
- W. A. Gardner, ["The Spectral Correlation Theory of Cyclostationary Time-Series," *Signal Processing* 11(1), 1986](https://doi.org/10.1016/0165-1684(86)90092-7). Spectral correlation, filtering relations, and cyclostationary characterisation.
- A. V. Dandawaté and G. B. Giannakis, ["Statistical Tests for Presence of Cyclostationarity," *IEEE Transactions on Signal Processing* 42(9), 1994](https://doi.org/10.1109/78.317857). Self-normalised asymptotic CFAR tests for cyclic moments and polyspectra.
- H. Rohling, ["Radar CFAR Thresholding in Clutter and Multiple Target Situations," *IEEE Transactions on Aerospace and Electronic Systems* AES-19(4), 1983](https://doi.org/10.1109/TAES.1983.309350). Order-statistic CFAR in unknown and non-homogeneous clutter.
- M. Zhao et al., ["A Tacho-Less Order Tracking Technique for Large Speed Variations," *Mechanical Systems and Signal Processing* 40(1), 2013](https://doi.org/10.1016/j.ymssp.2013.03.024). Ridge/instantaneous-frequency estimation, Vold--Kalman extraction, phase recovery, and order tracking without a tachometer.
- B. Satchidanandan and P. R. Kumar, ["Dynamic Watermarking: Active Defense of Networked Cyber-Physical Systems," 2016](https://arxiv.org/abs/1606.08741). Private physical excitation for authenticating observed system response.
- K. Abbaszadeh et al., ["Zero-Knowledge Proofs of Training for Deep Neural Networks," CCS 2024](https://eprint.iacr.org/2024/162.pdf). Incrementally verifiable proofs for gradient-descent training iterations.
- V. Vovk et al., ["Retrain or Not Retrain: Conformal Test Martingales for Change-Point Detection," PMLR 152, 2021](https://proceedings.mlr.press/v152/vovk21b.html). Sequential conformal martingales under exchangeability.
