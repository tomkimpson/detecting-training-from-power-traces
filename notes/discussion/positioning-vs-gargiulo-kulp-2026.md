# Positioning against Gargiulo & Kulp (arXiv 2609.00309)

**Date:** 2026-09-03
**Status:** discussion note (rationale record, not a tracker). Written the week the
paper appeared (submitted 31 Aug 2026). Companion to
[`rahman-critique-long-term-verification.md`](rahman-critique-long-term-verification.md)
(the telemetry-side comparison) and
[`north-star-and-positioning.md`](north-star-and-positioning.md) (our spine).
Intended uses: the related-work and channel-requirement edits to `paper/main.tex`,
referee-response material, and the standing answer to "didn't someone already do
this with real hardware?"

**Verdict up front.** We are not sunk. The two papers share a headline (power draw
reveals training; a governance verifier can use it; hiding costs throughput) and then
sit at opposite ends of a single axis, *where the verifier's meter is*. Their result
is the real-hardware instance of the setting our introduction explicitly says we are
not in, and read along the channel axis it is evidence *for* our central channel
claim rather than against it. The overlap is large enough that we must cite them
prominently and position against them in three places, and honest enough that we
should adopt one of their findings as support.

---

## 1. What they did

Gargiulo, S. & Kulp, G., *Workload Identification with Physical Side Channels for AI
Governance*, arXiv 2609.00309 (cs.CR; cs.AI, cs.CY, cs.LG), 10 pp., 2 figures.

- **Channel.** A verifier-owned Rogowski current probe (Micsig RCP-XS series,
  RCP120; specified bandwidth 34 Hz to 30 MHz, 10 mV/A) clamped around all positive
  conductors of the **PCIe auxiliary power supply** of a single **NVIDIA H200 NVL**,
  hosted by Amodo Design. Digitised on a PicoScope at a nominal **10 MHz**, in
  **5-second** windows. AC-coupled: the probe sees ripple, not the DC level.
- **Corpus.** 930 honest traces: 17 open LLM families (4B to 21B parameters, dense
  and MoE), each run as inference, full fine-tuning, LoRA and gradient accumulation,
  10 captures per label from a fresh model load; plus 25 non-AI CUDA workloads
  (matmul, FFT, convolution, sort, gather, ...). 680 adversarial traces: 4 evasion
  strategies x 17 families x 10 captures. All workloads run by the authors on their
  own GPU. Dataset released on Hugging Face.
- **Detector.** Two-stage random forest on 30 amplitude-independent features, 22 of
  them spectral (normalised band power in 14 log-spaced bands from 20 Hz to 8 MHz,
  centroid, spread, skewness, entropy, roll-off percentiles) and 8 temporal (envelope
  CV, crest factor, repetition frequency, autocorrelation decay, idle fraction, duty
  cycle, transition rate). Five-fold cross-validation grouped by model family.
- **Headline.** Training vs inference vs non-AI on unseen model families: **97%
  accuracy, macro-F1 0.955**. The discriminative content sits **below about
  20 kHz** and is attributed to the **memory-bound optimizer-update phase**. Accuracy
  stays at or above 94.6% after 256-fold decimation, i.e. down to about
  **38 kS/s**; they do not go lower.
- **Evasion.** Four strategies, with measured throughput penalties: chunked
  optimizer (28 to 56%), dilution with decoy inference (48 to 56%), throttling (39 to
  42%), diluted LoRA (34 to 69%). A classifier hardened on three strategies detects
  the held-out fourth at 99% or better for the first three; diluted LoRA falls to
  48 to 88%, recovered to 98% by a hand-written "rescue rule" (a spectral dip gated
  on more than half the power sitting below 10 kHz, targeting the optimizer-update
  event).
- **Framing.** "Unlike on-chip NVML telemetry, which can be spoofed or replayed, such
  a physical channel can in principle be observed independently of operator
  cooperation." Evasion "requires restructuring the computation itself." They do
  *not* use the word unspoofable; an earlier summary of ours did, and the manuscript
  should not attribute it to them.
- **Stated limits.** Single device ("generalization across devices and hardware
  architectures remains to be established"); a limited evasion menu; deployment
  "requires physical access to the monitored hardware and likely continuous on-site
  monitoring", plus "regulation of its own, together with tamper-resistant mechanisms
  that guarantee the fidelity of its readout".
- **Not addressed.** Distributed or multi-GPU synchronisation; meter noise, filtering
  or integration; how a verifier obtains labelled traces for a device it does not
  own; any challenge protocol or cryptographic binding; Ko & Zhu (2025) is not cited.

---

## 2. Side by side

| Axis | Gargiulo & Kulp | This paper |
|---|---|---|
| Question | How accurately can a classifier tell training from inference and non-AI? | What can a passive meter *certify*, with what methods, at what information cost, with what ceiling? |
| Meter | Verifier-owned probe on the GPU's own power conductors; 10 MHz; on-site | Remote, filtered, low-rate meter outside the prover's control; the minimum meter spec is a result |
| Signature | Spectral shape of the optimizer-update phase; kHz band; single-card observable | Iteration cadence as a wandering second-order (cyclostationary) line; shallow on one card, predicted deep under distributed sync |
| Evidence | Measured H200 corpus, public | Literature-parameterised scenario model (Ko & Zhu training waveform, our inference null, our observation map); single-A100 as negative transport case and cost anchors |
| Detector | Learned (random forest) on a labelled corpus | Physics-derived, corpus-free, per-trace surrogate calibration; a learned RF is one baseline in the bank |
| Adversary | Fixed menu of four evasions; classifier hardened on three, tested on the fourth; post-hoc rescue rule | Prover who knows the detector; attack families priced on a detection-utility frontier against measured throughput anchors |
| Cheapest attack found | Diluted LoRA, 34 to 69% throughput | Variable real work per iteration, about 0% throughput, defeats every fixed-frequency test |
| Semantics | Training pooled over full FT, LoRA, grad-accum; non-AI = CUDA kernels; no decoy | Five non-training loads, including a discarded-update decoy, score as training under every rule: the semantic ceiling |
| Beyond passive | Not addressed | Rungs 3 and 4: challenge protocol specified (not run), binding primitive assumed |
| Distributed | Not addressed | Stated as a falsifiable prediction, not a result |

---

## 3. The hardware gap, carefully

This is the axis on which a reviewer will hit us first, so it deserves the most
careful statement.

### 3.1 Our gap

The manuscript is explicit (intro, `sec:scenario`, `sec:reality`, Limitations): the
generator is *literature-parameterised, not calibrated*. Ko & Zhu assume an
iteration-cadence modulation of roughly 30% of power; we measure about 0.40% of board
power at the corresponding frequency on one saturated A100, a factor of some 75. One
card has no communication barrier, so the deep compute/communication contrast the
scenario model assumes is absent. Real hardware enters as a negative transport case
(the shallow cadence plus the 0.31 to 0.45 Hz power-management line that says
"time-varying load", not "training") and as three measured throughput anchors on the
frontier. Every quantitative result is a sensitivity analysis over a stated parameter
family.

### 3.2 Their gap, and why it looks like none

They have no simulation-to-hardware gap for the question they ask: they own the H200,
they run the workloads, they label the traces, and every number is measured. That is
a genuine advantage and we should say so plainly.

But the absence of a gap is bought by the threat model. The hardware in the
experiment is the hardware in the threat model: a verifier who has clamped a
Rogowski coil around the GPU's PCIe power conductors, digitising at 10 MHz, on site,
continuously. Three things follow.

**(a) The gap has moved, not vanished.** For us the open question is "does the
signature survive the transport from the scenario model to a real meter?" For them
it is "does the measurement point survive the transport from a lab bench to an
adversary's datacentre?" Their own discussion names the cost: physical access,
continuous on-site monitoring, regulation of the probe, tamper-resistant readout. A
probe on the board is a prover-hosted sensor with a verifier's label on it, and
the trust-anchor problem we raised against NVML in
[`rahman-critique-long-term-verification.md`](rahman-critique-long-term-verification.md)
returns in physical form: who guarantees the coil is on the right conductors, that
the PCIe rail is the only rail (the H200 NVL also draws through the slot), that the
board under the coil is the board doing the work? None of this is fatal to their
paper, and they say it. It does mean their "no hardware gap" and our "remote meter"
are the same fact viewed from two sides: the closer the meter, the smaller the
transport gap and the larger the deployment gap.

**(b) They share our distributed gap; it just does not bite their observable.**
Their corpus is one card, as is our A100 data. Neither paper has seen the
synchronous compute/communication modulation of a distributed run. For them this
does not matter, because the optimizer-update phase is a single-card observable and
their probe has the bandwidth to see it. For us it matters, because the cadence is
what we detect, it is shallow on one card, and its deepening under distributed
synchronisation is the prediction we state as falsifiable. Same gap, different
consequence, and we should say that rather than let a reviewer imply only we have
it.

**(c) They need a labelled corpus per device; we need a model.** Their classifier
is trained and tested on one H200 and cross-device generalisation is left open. A
verifier deploying on device X therefore needs labelled traces on X, which means
owning X or having the operator run labelled workloads for them, a cooperative step
inside a passive channel. Our detectors are corpus-free (physics-derived statistics
with per-trace surrogate calibration of the null) and carry no labelled data across
devices; the price is that we rely on a model of the signature that is, by our own
admission, 75x off the one card we measured. This is the same trade as against
Rahman & Tajdari: they buy realism with a corpus, we buy corpus-freedom with a
model. The caveat from that note still stands and must not be dropped: our own RF
baseline was at ceiling on measured single-card data, so the argument is never
"learned detects worse". It is that a learned detector cannot say why it will keep
working on hardware it has not seen, and cannot be calibrated without a corpus.

### 3.3 The measurement-point axis reconciles the two

Put the two papers on one axis, from the die outward:

| Point | Bandwidth | What is visible | Who has it |
|---|---|---|---|
| On-die telemetry (NVML) | ~1 Hz, prover-reported | Everything, forgeable | Rahman & Tajdari |
| Board power conductors | 10 MHz, verifier probe | Optimizer-update phase, sub-20 kHz; survives to 38 kS/s | Gargiulo & Kulp |
| PDU / rack / facility meter | 1 to 20 Hz, 0.05 to 1 s integration | Iteration cadence, if synchronised and deep enough; dead at 1 Hz with 1 s integration | This paper |

Their decimation study stops four orders of magnitude above where a facility meter
reads. Their informative band (below 20 kHz) is therefore direct real-hardware
support for our finding that a one-second-integrating, one-hertz meter defeats every
detector on an honest workload: the single-card signature they exploit is gone long
before the meter, and what a facility meter can hope to see is a different, slower
observable that exists only as a collective effect. One precision point: our minimum
meter specification (sampling at or above 2 Hz and integration at or below 0.5 s) is
for the cadence on the Ko-scale generator, not for their observable, so the two
results are consistent and complementary, not the same result.

---

## 4. Adversary and cost of hiding

Their evasion study is the part closest to our central result and the part we most
want to cite as corroboration.

- **Costs are the same order.** Their four strategies cost 28 to 69% throughput.
  Our idle-insertion jitter costs 15 to 375%, and our anchored attack against the
  tracking class buys little hiding for 159%. Both papers find that the obvious ways
  to disguise training are expensive. Cite this.
- **They stopped before our cheap attack.** Their strategies all insert something
  (decoy inference, throttling, chunked or shrunk updates) and pay for it. Our
  variable-real-work attack changes the useful computation per iteration, inserts
  nothing, is measured at zero throughput cost within noise, and defeats every
  fixed-frequency test on our channel. Whether it defeats *their* detector is an
  open question we cannot settle: their features are spectral-shape band powers,
  which are indifferent to cadence wander, and varying accumulation steps per
  iteration leaves the optimizer-update phase in place. At 10 MHz they would very
  likely still see it. The right statement is that our zero-cost attack is against
  cadence detectors on a low-rate channel, and that their observable is not
  available on that channel. Do not claim more.
- **The hardening loop is the same red-queen loop.** Hardened on three, tested on
  the fourth, then a hand-written rescue rule after seeing the fourth fail. This is
  the inductive-robustness point from the Rahman note and it applies verbatim. Their
  own future-work sentence concedes it: find "evasion strategies that retain the
  optimization flexibility of full fine-tuning while still reducing detectability".
- **Their adversary is not detector-aware.** The strategies are chosen a priori; no
  white-box round. Ours is defined as knowing the detector. This is a difference in
  question, not a flaw in theirs.

---

## 5. The semantic ceiling applies to them too

Their three-class problem pools full fine-tuning, LoRA and gradient accumulation as
"training" and uses CUDA kernels as the non-AI class. There is no decoy. Their
mechanism, the memory-bound optimizer update, is exactly the phase our
discarded-update decoy reproduces: run forward, backward and update, discard the
state. A load that does that produces their sub-20 kHz signature and is not
training. Their 97% is a Rung-2 number on our ladder, conditional on their null
population and ceiling-bound by the same argument we make in
`subsec:identifiability`: the meter, at any bandwidth, certifies a physical schedule
and not the retention of a training state. Their channel statement ("observed
independently of operator cooperation") is about the channel's trust boundary, not
about semantics, and is correct as stated. The manuscript should draw the contrast
without putting a claim in their mouth.

---

## 6. What they have that we do not

Say these plainly in the paper and in any response; a reviewer who has read both
will otherwise say them for us.

1. **Real data, released.** 1,610 measured traces on current hardware, public.
2. **Breadth.** 17 model families, dense and MoE, 25 non-AI kernels. We have one
   training waveform model and one inference null.
3. **A named mechanism.** The memory-bound optimizer-update phase as the physical
   origin of the signature is a genuine physics insight, and it is a single-card
   effect, which is why they can see it and we cannot.
4. **A measured bandwidth study** (10 MHz down to 38 kS/s) on real hardware.
5. **Measured evasion costs on the hardware doing the evading**, for four
   strategies, where we have three anchors from one A100.

---

## 7. What we have that they do not

1. The **question**: certification with ceilings, not classification accuracy.
2. The **channel a verifier can actually read today** without a probe on the board,
   and the minimum meter specification under which it is usable at all.
3. A **detector-aware adversary** and a priced **detection-utility frontier**,
   including the zero-cost attack they did not reach.
4. **Corpus-free detection** with per-trace null calibration, hence no labelled
   traces per device.
5. The **semantic ceiling** made concrete with a decoy, which bounds their result as
   well as ours.
6. **Rungs 3 and 4**: what it takes to get past that ceiling.
7. The **distributed prediction**, stated as falsifiable.

---

## 8. Review risks and the answer to each

| Risk | Answer |
|---|---|
| "They did it on real hardware at 97%; you simulated." | Different channel and question. Their measurement point is the assumption; a 5-second trace on the board's conductors at 10 MHz cannot test a remote low-rate meter or a distributed run. We cite them as the proximate-probe instance and use their sub-20 kHz result as support for the channel claim. |
| "Their evasion costs contradict your zero-cost attack." | No: their strategies insert or shrink work; ours varies real work. Their costs corroborate the expensive half of our frontier. Whether our attack beats their observable at 10 MHz is open and stated as such. |
| "Your semantic-decoy ceiling is a synthetic artefact." | Their mechanism is the optimizer-update phase; a discarded-update load produces it on real hardware by construction. The ceiling is about what the observable is, not about the generator. |
| "Why not use their public dataset?" | 5-second windows at the board, AC-coupled: no cadence at meter rates in 5 s, no distributed regime, no `P_other`. Usable for a bandwidth/integration sweep from 38 kS/s downward (see section 10), not for the cadence question. |
| "You have the same single-GPU limitation." | Yes, and we say so; the difference is that their observable is single-card and ours is collective, so the gap bites us and not them, which is why our real-hardware section is a negative transport case rather than a pillar. |

---

## 9. Manuscript actions

All additive; no frozen number changes.

1. **`references.bib`**: add

   ```bibtex
   @misc{gargiulo2026workload,
     author        = {Gargiulo, Simone and Kulp, Gabriel},
     title         = {Workload Identification with Physical Side Channels for {AI} Governance},
     year          = {2026},
     eprint        = {2609.00309},
     archivePrefix = {arXiv},
     primaryClass  = {cs.CR},
     url           = {https://arxiv.org/abs/2609.00309}
   }
   ```

2. **Intro, trust-anchor sentence** (currently cites `rahman2026telemetry,
   ogara2025hardware`): add a clause that a verifier-owned probe on the board's power
   conductors removes the telemetry trust anchor and separates training from
   inference at 97% on a single H200, at the price of physical access and on-site
   monitoring, citing `gargiulo2026workload`.

3. **Intro, "what changes here" paragraph**: the sentence "side-channel analysis
   assumes a proximate, high-bandwidth probe" now has a governance-facing instance;
   cite it there so the reader sees we know the proximate case has been done.

4. **`subsec:meter_spec` (minimum meter specification)**: one sentence noting that
   on real hardware the single-card training signature is concentrated below about
   20 kHz and remains classifiable to 38 kS/s [gargiulo2026workload], four orders of
   magnitude above the meter rates studied here, consistent with the collapse we
   find at 1 Hz.

5. **`subsec:identifiability` or the Rung-2 decoy paragraph**: one sentence that a
   classifier keyed to the optimizer-update phase [gargiulo2026workload] is subject
   to the same ceiling, since the discarded-update decoy reproduces that phase.

6. **`sec:frontier`**: cite their 28 to 69% measured penalties beside our idle-jitter
   range as independent real-hardware evidence that insertion-type disguises are
   expensive, and note their menu stops short of the variable-real-work family.

7. **`sec:reality` or Limitations**: one clause acknowledging the single-card gap is
   shared and explaining why it bites the cadence observable and not theirs.

Then rebuild `main.pdf` with the pinned-epoch invocation and run `check-refs`.

---

## 10. Optional experiment with their public dataset

Cheap, and it would turn part of our channel claim into a measured statement on an
H200. Pull the Hugging Face traces, treat each as `P_device`, push it through
`powerladder/observation.py` (`apply_meter`: bandwidth, boxcar integration,
zero-order-hold decimation, notch), and sweep from their 38 kS/s floor down toward
1 Hz, reporting where their own 30-feature RF and our bank lose the training class.
Limits to state: 5-second windows mean nothing below a few hertz is meaningful; the
probe is AC-coupled so the DC baseline and `P_other` superposition are absent; it is
one card, so the cadence question is untouched. This would be a real-hardware
extension of `subsec:meter_spec`, not a validation of the cadence detectors.

---

## 11. Open questions

- Does the variable-real-work attack move their spectral-shape features at all? Only
  answerable with an H200 and their harness; note as future work, do not speculate
  in the manuscript.
- Is the optimizer-update phase visible in our own single-A100 traces at the sample
  rates we recorded? If so, it is a second single-card observable for
  `sec:reality`; if the rate is too low, that is itself the point.
- Their probe is on the PCIe auxiliary rail only. How much of H200 NVL board power
  flows through the slot, and does the split vary with workload phase? Relevant to
  the trust-anchor argument in section 3.2(a).
