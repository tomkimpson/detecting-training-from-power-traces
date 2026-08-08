---
marp: true
title: What a Passive Power Meter Can Certify About AI Training
description: Research talk on the claim ladder, adversarial frontier, and limits of training verification from power traces
author: Tom Kimpson, Mauricio Baker, Emlyn Graham, Anjay Friedman, Coby Joseph
theme: default
size: 16:9
paginate: true
math: katex
style: |
  :root {
    --ink: #132238;
    --muted: #526174;
    --paper: #f7f5ef;
    --white: #ffffff;
    --navy: #0c1c33;
    --teal: #008b8b;
    --teal-light: #d9f0ee;
    --orange: #e7792f;
    --orange-light: #fbe7d6;
    --red: #b94848;
    --red-light: #f5dddd;
    --blue: #3976a8;
    --line: #cbd2d9;
  }

  section {
    background: var(--paper);
    color: var(--ink);
    font-family: "Avenir Next", "Helvetica Neue", Arial, sans-serif;
    font-size: 27px;
    line-height: 1.25;
    padding: 54px 70px 48px;
  }

  h1 {
    color: var(--navy);
    font-size: 45px;
    font-weight: 680;
    letter-spacing: -0.025em;
    margin: 0 0 24px;
  }

  h2 {
    color: var(--teal);
    font-size: 27px;
    font-weight: 650;
    margin: 0 0 14px;
  }

  p { margin: 0.4em 0; }
  ul, ol { margin: 0.35em 0 0.2em 1.05em; }
  li { margin: 0.3em 0; }
  strong { color: var(--navy); }

  section::after {
    color: #7f8994;
    font-size: 15px;
    right: 34px;
    bottom: 24px;
  }

  section.title {
    background: var(--navy);
    color: var(--white);
    padding: 72px 82px;
  }

  section.title h1 {
    color: var(--white);
    font-size: 64px;
    line-height: 1.03;
    max-width: 1010px;
    margin-top: 102px;
  }

  section.title h2 {
    color: #74d5cd;
    font-size: 30px;
    font-weight: 520;
    max-width: 900px;
  }

  section.title .authors {
    color: #d7e2ef;
    font-size: 20px;
    margin-top: 72px;
  }

  section.title::after { color: transparent; }

  section.dark {
    background: var(--navy);
    color: var(--white);
  }

  section.dark h1, section.dark strong { color: var(--white); }
  section.dark h2 { color: #74d5cd; }
  section.dark::after { color: #9eb0c5; }

  .section-kicker {
    color: var(--teal);
    font-size: 18px;
    font-weight: 700;
    letter-spacing: 0.12em;
    margin-bottom: 10px;
    text-transform: uppercase;
  }

  section.dark .section-kicker { color: #74d5cd; }

  .question {
    font-size: 45px;
    font-weight: 650;
    line-height: 1.13;
    letter-spacing: -0.02em;
    max-width: 1080px;
  }

  .big-number {
    color: var(--teal);
    font-size: 78px;
    font-weight: 760;
    line-height: 0.95;
  }

  .big-claim {
    color: var(--navy);
    font-size: 38px;
    font-weight: 650;
    line-height: 1.16;
  }

  section.dark .big-claim { color: var(--white); }

  .muted { color: var(--muted); }
  section.dark .muted { color: #b6c3d1; }
  .small { font-size: 18px; }
  .tiny { font-size: 14px; }
  .center { text-align: center; }

  .scope {
    background: #e8edf2;
    border-radius: 4px;
    color: #566473;
    display: inline-block;
    font-size: 13px;
    font-weight: 750;
    letter-spacing: 0.07em;
    padding: 5px 9px 4px;
    text-transform: uppercase;
  }

  .scope.orange { background: var(--orange-light); color: #a04a16; }
  .scope.teal { background: var(--teal-light); color: #046b69; }
  section.dark .scope { background: #203751; color: #c7d6e6; }

  .two-col {
    align-items: center;
    display: grid;
    gap: 46px;
    grid-template-columns: 1fr 1fr;
  }

  .two-col.wide-right { grid-template-columns: 0.78fr 1.22fr; }
  .two-col.wide-left { grid-template-columns: 1.2fr 0.8fr; }
  .two-col.top { align-items: start; }

  .three-col {
    align-items: stretch;
    display: grid;
    gap: 25px;
    grid-template-columns: repeat(3, 1fr);
  }

  .card {
    background: var(--white);
    border: 1px solid var(--line);
    border-radius: 10px;
    box-sizing: border-box;
    padding: 22px 24px;
  }

  .card h2 { margin-bottom: 8px; }
  .card p { font-size: 21px; }
  .card.teal { border-top: 7px solid var(--teal); }
  .card.orange { border-top: 7px solid var(--orange); }
  .card.red { border-top: 7px solid var(--red); }
  .card.navy { border-top: 7px solid var(--navy); }

  .callout {
    background: var(--teal-light);
    border-left: 7px solid var(--teal);
    border-radius: 4px;
    font-size: 25px;
    margin-top: 22px;
    padding: 14px 20px;
  }

  .callout.orange {
    background: var(--orange-light);
    border-left-color: var(--orange);
  }

  .callout.red {
    background: var(--red-light);
    border-left-color: var(--red);
  }

  .flow {
    align-items: stretch;
    display: grid;
    gap: 12px;
    grid-template-columns: 1fr 38px 1fr 38px 1fr;
    margin-top: 25px;
  }

  .flow.four { grid-template-columns: 1fr 32px 1fr 32px 1fr 32px 1fr; }

  .flow-box {
    align-items: center;
    background: var(--white);
    border: 2px solid #b7c2cd;
    border-radius: 9px;
    display: flex;
    flex-direction: column;
    font-size: 21px;
    justify-content: center;
    min-height: 108px;
    padding: 12px;
    text-align: center;
  }

  .flow-box strong { display: block; margin-bottom: 5px; }
  .arrow { align-items: center; color: var(--orange); display: flex; font-size: 40px; justify-content: center; }

  .trace {
    background: var(--white);
    border: 1px solid var(--line);
    border-radius: 10px;
    padding: 18px;
  }

  .signal {
    align-items: center;
    display: grid;
    gap: 24px;
    grid-template-columns: 1fr 50px 1fr 50px 1fr;
    margin-top: 38px;
  }

  .signal-box {
    background: var(--white);
    border: 2px solid #aebbc8;
    border-radius: 12px;
    min-height: 175px;
    padding: 22px;
    text-align: center;
  }

  .signal-box .icon {
    color: var(--teal);
    font-size: 50px;
    font-weight: 760;
    line-height: 1;
    margin-bottom: 12px;
  }

  .ladder {
    display: grid;
    gap: 11px;
    margin: 18px auto 0;
    max-width: 1110px;
  }

  .rung {
    align-items: center;
    background: var(--white);
    border: 1px solid var(--line);
    border-left: 10px solid var(--teal);
    border-radius: 7px;
    display: grid;
    gap: 18px;
    grid-template-columns: 64px 265px 1fr 1fr;
    min-height: 76px;
    padding: 9px 18px;
  }

  .rung:nth-child(2) { margin-left: 45px; border-left-color: #287ba0; }
  .rung:nth-child(3) { margin-left: 90px; border-left-color: var(--orange); }
  .rung:nth-child(4) { margin-left: 135px; border-left-color: var(--red); }
  .rung .n { font-size: 38px; font-weight: 780; text-align: center; }
  .rung .name { font-size: 23px; font-weight: 720; }
  .rung .info, .rung .ceiling { color: var(--muted); font-size: 17px; }
  .rung .label { color: #8994a0; font-size: 11px; font-weight: 750; letter-spacing: 0.08em; text-transform: uppercase; }

  .matrix {
    border-collapse: collapse;
    font-size: 21px;
    margin: 24px auto 0;
    width: 94%;
  }

  .matrix th, .matrix td {
    border-bottom: 1px solid var(--line);
    padding: 13px 17px;
    text-align: center;
  }

  .matrix th { color: var(--muted); font-size: 16px; font-weight: 700; }
  .matrix th:first-child, .matrix td:first-child { text-align: left; }
  .yes { color: #087b69; font-weight: 760; }
  .no { color: var(--red); font-weight: 760; }
  .partial { color: #a55d13; font-weight: 760; }

  .equation {
    background: var(--white);
    border: 1px solid var(--line);
    border-radius: 9px;
    font-family: "Avenir Next", "Helvetica Neue", Arial, sans-serif;
    font-size: 29px;
    margin: 22px 0;
    padding: 22px;
    text-align: center;
  }

  section.figure img {
    display: block;
    margin: 0 auto;
    max-height: 535px;
    max-width: 100%;
    object-fit: contain;
  }

  section.figure-tight img { max-height: 495px; }

  .caption {
    color: var(--muted);
    font-size: 15px;
    line-height: 1.25;
    margin-top: 5px;
    text-align: center;
  }

  .bar {
    background: #dfe5ea;
    border-radius: 20px;
    height: 15px;
    overflow: hidden;
  }

  .bar > span { background: var(--teal); display: block; height: 100%; }

  section.takeaway {
    background: var(--navy);
    color: var(--white);
  }

  section.takeaway h1 { color: var(--white); }
  section.takeaway strong { color: var(--white); }
  section.takeaway::after { color: #9eb0c5; }

  .takeaway-row {
    align-items: start;
    border-top: 1px solid #3b4b60;
    display: grid;
    gap: 24px;
    grid-template-columns: 75px 1fr;
    padding: 20px 0;
  }

  .takeaway-row .n { color: #74d5cd; font-size: 43px; font-weight: 760; }
  .takeaway-row p { font-size: 27px; margin: 5px 0 0; }

  section.appendix {
    background: #eef1f4;
  }

  section.appendix h1::before {
    color: var(--orange);
    content: "APPENDIX  ";
    font-size: 14px;
    letter-spacing: 0.11em;
    vertical-align: middle;
  }
---

<!-- _class: title -->

# What a passive power meter can certify about AI training

## A claim ladder for adversarial verification

<p class="authors">Tom Kimpson · Mauricio Baker · Emlyn Graham · Anjay Friedman · Coby Joseph</p>

<!--
Opening (20 seconds): Start with the governance question, not the detector. The talk is
about the limits of inference from one externally measured physical channel.
-->

---

<div class="section-kicker">The policy question</div>

# A regulator sees one scalar trace

<div class="signal">
  <div class="signal-box">
    <div class="icon">?</div>
    <strong>Inside the facility</strong><br>
    Training, inference, control loops, cooling, and unrelated tenants
  </div>
  <div class="arrow">→</div>
  <div class="signal-box">
    <div class="icon">y(t)</div>
    <strong>Outside the prover</strong><br>
    A time-resolved power trace
  </div>
  <div class="arrow">→</div>
  <div class="signal-box">
    <div class="icon">✓?</div>
    <strong>Governance decision</strong><br>
    Did a restricted training run occur?
  </div>
</div>

<div class="callout"><strong>Question:</strong> what conclusions does the trace actually license?</div>

<!--
The attraction is that the meter can sit outside the prover's software and control. But
the verifier is solving an inverse problem: many internal causes have been collapsed to
one signal.
-->

---

# Power is trusted as a channel—not automatically as evidence

<div class="two-col top">
  <div>
    <h2>Why it is attractive</h2>
    <ul>
      <li>Measured outside the prover's software stack</li>
      <li>Continuous and comparatively cheap</li>
      <li>Physical work must consume energy</li>
    </ul>
  </div>
  <div>
    <h2>Why it is dangerous</h2>
    <ul>
      <li>Thousands of devices collapse into one number</li>
      <li>The meter filters, integrates, and samples</li>
      <li>Different programs can create the same physical schedule</li>
    </ul>
  </div>
</div>

<div class="equation">observed trace = workload physics + nuisance loads + meter transfer function</div>

<p class="big-claim center">The meter may be trustworthy while the interpretation is not.</p>

---

<div class="section-kicker">The physical hypothesis</div>

# Efficient synchronous training should leave a rhythm

<div class="flow four">
  <div class="flow-box"><strong>Forward</strong>compute</div>
  <div class="arrow">→</div>
  <div class="flow-box"><strong>Backward</strong>compute</div>
  <div class="arrow">→</div>
  <div class="flow-box"><strong>All-reduce</strong>communication barrier</div>
  <div class="arrow">→</div>
  <div class="flow-box"><strong>Update</strong>then repeat</div>
</div>

<div class="callout">
  Repeated compute–communication phases can create a <strong>cadence line</strong> whose frequency wanders as iteration time changes.
</div>

<p class="small muted center">The target is not a perfectly fixed frequency. It is order-coherent variation that remains repeatable after following the cadence.</p>

<!--
The physical claim is intentionally narrow. We are not asserting that every training run
has a line. We target efficient, synchronous, iteration-structured training.
-->

---

<!-- _class: figure -->

# The scenario model makes that hypothesis visible

![w:1120](../figures/scenario_traces.png)

<p class="caption"><span class="scope orange">synthetic scenario</span> Literature-parameterized workload and off-chip observation model. Time-domain traces above; spectra below.</p>

<!--
Point out the contrast between the training-like cadence and the inference/null traces.
Do not imply that these plots are measured datacenter traces.
-->

---

<div class="section-kicker">The first trap</div>

# A tracker can lock onto a real line—and still track the wrong thing

<div class="three-col">
  <div class="card teal">
    <h2>Target A</h2>
    <p>True iteration cadence on one measured A100</p>
    <p><strong>≈ 0.4% of board power</strong></p>
  </div>
  <div class="card orange">
    <h2>Confounder B</h2>
    <p>A power-management limit cycle under time-varying load</p>
    <p><strong>≈ 10× stronger than A</strong></p>
  </div>
  <div class="card red">
    <h2>Viterbi result</h2>
    <p>Correctly follows the strongest plausible path</p>
    <p><strong>But that path is B</strong></p>
  </div>
</div>

<div class="callout red"><strong>Tracking solves frequency wander. It does not solve causal attribution.</strong></div>

<!--
This is the historical failure that motivates the present framing. A fixed frequency bin
was not the only problem. Even a good adaptive tracker can return a clean answer to the
wrong physical question.
-->

---

<div class="section-kicker">The deeper trap</div>

# Even the right cadence does not prove training

<div class="two-col">
  <div class="card teal center">
    <h2>Genuine training</h2>
    <p>forward → gradients → update</p>
    <p class="big-number">y(t)</p>
  </div>
  <div class="card orange center">
    <h2>Semantic decoy</h2>
    <p>forward → gradients → discard</p>
    <p class="big-number">y(t)</p>
  </div>
</div>

<div class="equation">If two behaviours induce the same trace distribution, every passive detector has the same decision distribution.</div>

<p class="big-claim center">No better algorithm can recover information that never reaches the meter.</p>

<!--
This is an identifiability statement, not an engineering complaint. The decoy can perform
the same physical operations and omit only the semantically important state transition.
-->

---

<div class="section-kicker">The organizing idea</div>

# Every stronger claim must buy more information

<div class="ladder">
  <div class="rung">
    <div class="n">1</div><div class="name">Structural evidence</div>
    <div class="info"><span class="label">information</span><br>One trace</div>
    <div class="ceiling"><span class="label">cannot exclude</span><br>Confounders, shaping, superposition</div>
  </div>
  <div class="rung">
    <div class="n">2</div><div class="name">Conditional classification</div>
    <div class="info"><span class="label">information</span><br>Trace + stated null population</div>
    <div class="ceiling"><span class="label">cannot exclude</span><br>Domain shift, semantic decoys</div>
  </div>
  <div class="rung">
    <div class="n">3</div><div class="name">Active authentication</div>
    <div class="info"><span class="label">information</span><br>Unpredictable challenge</div>
    <div class="ceiling"><span class="label">cannot exclude</span><br>Challenge-aware dummy work</div>
  </div>
  <div class="rung">
    <div class="n">4</div><div class="name">Work verification</div>
    <div class="info"><span class="label">information</span><br>Challenge + bound transcript</div>
    <div class="ceiling"><span class="label">cannot exclude</span><br>Unbound co-located responder</div>
  </div>
</div>

<!--
This is the spine of both the talk and the paper. Each rung answers a different question.
The point is not that the lower rungs are useless; it is that their conclusions must not
be silently upgraded.
-->

---

<div class="section-kicker">Rung 1 · structural evidence</div>

# Test for order-coherent structure—not for “training”

<div class="flow four">
  <div class="flow-box"><strong>1 · Trace</strong>remove slow trend and nuisance scale</div>
  <div class="arrow">→</div>
  <div class="flow-box"><strong>2 · Track</strong>follow a candidate wandering cadence</div>
  <div class="arrow">→</div>
  <div class="flow-box"><strong>3 · Dewarp</strong>express time in estimated cycle order</div>
  <div class="arrow">→</div>
  <div class="flow-box"><strong>4 · Test</strong>ask whether the waveform repeats by order</div>
</div>

<div class="two-col top" style="margin-top: 26px;">
  <div class="card teal">
    <h2>Evidence for</h2>
    <p>A repeatable physical schedule whose cadence may drift.</p>
  </div>
  <div class="card red">
    <h2>Not evidence for</h2>
    <p>The semantic identity of the program or a model update.</p>
  </div>
</div>

<p class="small muted"><span class="scope teal">corpus-free calibration</span> A single trace can be compared with Fourier surrogates under a restricted linear-stationary structural null.</p>

---

<!-- _class: figure -->

# Finding a line and attributing structure are different jobs

![w:1120](../figures/st1_bakeoff.png)

<p class="caption"><span class="scope orange">synthetic bake-off</span> Detection rate at nominal false-alarm rate 0.05. Viterbi dominates against the inference null (left); order-based statistics reject controller-line confusers (middle and right).</p>

<div class="callout"><strong>Lesson:</strong> the detector that best follows a wandering line need not be the detector that best distinguishes its cause.</div>

<!--
Read the panels from left to right. The Viterbi score is excellent when the negative has
no competing line. Against structural/controller confusers it is not an attribution test.
The order-domain statistics ask a different, more specific question.
-->

---

# There are two different “null” problems

<table class="matrix">
  <thead>
    <tr><th></th><th>Question</th><th>Where the comparison comes from</th><th>What calibration means</th></tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>Rung 1</strong></td>
      <td>Is this more structured than a linear-stationary trace?</td>
      <td>Fourier surrogates made from the trace itself</td>
      <td>Valid only for that structural null</td>
    </tr>
    <tr>
      <td><strong>Rung 2</strong></td>
      <td>Is this more training-like than stated inference?</td>
      <td>An explicit reference population</td>
      <td>Conditional on that population and transfer</td>
    </tr>
  </tbody>
</table>

<div class="callout orange">
  Adaptive search changes the score distribution. Calibration must rerun the <strong>entire</strong> search on every surrogate or null trace.
</div>

<p class="small muted">The surrogate method does not make controller cycles part of the null, and the classifier cannot avoid stating what “non-training” means.</p>

<!--
This addresses the concern that verification requires an inaccessible set of null traces.
Rung 1 can be corpus-free, but only by asking the weaker structural question. Rung 2
genuinely requires a population assumption. Also emphasize that the analytic chi-square
calibration failed after adaptation; rerunning the full pipeline is the honest repair.
-->

---

<div class="section-kicker">The adaptive prover</div>

# Robustness is not binary; it is a cost frontier

<div class="two-col wide-left">
  <div>
    <p class="question">How much useful work must the prover give up to make the trace look benign?</p>
    <div class="callout"><strong>Unlimited adversary:</strong> passive semantic verification is impossible.</div>
    <div class="callout orange"><strong>Operational question:</strong> which evasions are cheap, and which are costly?</div>
  </div>
  <div class="card navy">
    <h2>Attack families</h2>
    <p>idle jitter · real-work variation · drift · phase diffusion · harmonic relocation · shaping · dilution · meter degradation</p>
  </div>
</div>

<p class="small muted"><span class="scope orange">50 synthetic cells</span> <span class="scope teal">16 measured cost anchors</span> The attacker knows the detector family; unpriced cells remain visibly unpriced.</p>

---

<!-- _class: figure figure-tight -->

# The cost frontier is sharply asymmetric

![w:760](../figures/st2_cost_pareto.png)

<p class="caption"><span class="scope orange">synthetic detection</span> <span class="scope teal">measured throughput anchors</span> Fixed-frequency tests lose 0.74 detection power at ≈0% overhead; the tracking envelope loses only 0.16 at 159% overhead. Hiding = 1 − best detection rate in class, at nominal false-alarm rate 0.05.</p>

<!--
This is the central empirical figure. Explain the asymmetry, then immediately state its
scope: only 16/50 cells are priced, and the class curve is the best score per cell rather
than a deployed joint policy.
-->

---

<div class="section-kicker">A qualification that matters operationally</div>

# The frontier is a detector-class envelope—not yet one deployed verifier

<table class="matrix">
  <thead>
    <tr><th>Method</th><th>Follows variable real work</th><th>Rejects controller-line confusers</th><th>Role in current evidence</th></tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>Raw Viterbi score</strong></td>
      <td class="yes">Strong</td>
      <td class="no">Weak</td>
      <td>Robust path finder</td>
    </tr>
    <tr>
      <td><strong>Order-domain tests</strong></td>
      <td class="partial">Degrade earlier</td>
      <td class="yes">Strong</td>
      <td>Structural attribution</td>
    </tr>
  </tbody>
</table>

<div class="callout red">
  We have not yet demonstrated one jointly calibrated decision rule that inherits both strengths across the full frontier.
</div>

<p class="small muted">A real verifier may run every algorithm it wants—but it must calibrate the final combined decision policy, including selection among them.</p>

<!--
This is not the claim that the verifier must choose one algorithm in advance. It can run
many. The missing experiment is a calibrated joint rule—AND, OR, gating, or a learned
combiner—evaluated against both controller confusers and the work-variation frontier.
-->

---

<!-- _class: figure -->

# The meter is part of the theorem

![w:1120](../figures/st2_meter_boundary.png)

<div class="three-col" style="margin-top: 8px;">
  <div class="card teal"><p><strong>Sample</strong><br>at least ≈ 2 Hz</p></div>
  <div class="card teal"><p><strong>Integrate</strong><br>no longer than ≈ 0.5 s</p></div>
  <div class="card orange"><p><strong>Avoid</strong><br>a deep in-band notch</p></div>
</div>

<p class="caption"><span class="scope orange">scenario-specific boundary</span> A 1 s boxcar nulls the 1 Hz band centre; a 1 Hz sampler crosses the cadence-band Nyquist edge.</p>

<!--
Do not state these numbers as universal hardware requirements. They are the minimum in the
scenario's 0.3–1.7 Hz cadence band. The transferable point is that observation-channel
specification is logically prior to detector claims.
-->

---

<div class="section-kicker">Rung 2 · conditional classification</div>

# Against the stated inference population, the classifiers look perfect

<div class="three-col" style="margin-top: 32px;">
  <div class="card teal center">
    <p class="big-number">1.00</p>
    <h2>Physics score AUC</h2>
  </div>
  <div class="card teal center">
    <p class="big-number">1.00</p>
    <h2>Fitted linear AUC</h2>
  </div>
  <div class="card teal center">
    <p class="big-number">1.00</p>
    <h2>Random forest AUC</h2>
  </div>
</div>

<p class="center" style="margin-top: 28px;">The result persists across the tested duration, frequency-band, controller, and coloured-noise shifts.</p>

<div class="callout orange center"><strong>But accuracy against a declared null is not the same thing as semantic validity.</strong></div>

<p class="small muted center"><span class="scope orange">synthetic populations · n = 200 per class</span></p>

---

<!-- _class: figure figure-tight -->

# The semantic controls overturn the naive conclusion

<div class="two-col wide-right" style="gap: 20px;">
  <div>
    <p class="big-claim">Five non-training workloads are called “training” by every rule.</p>
    <ul class="small">
      <li>discarded update</li>
      <li>gradient only</li>
      <li>non-ML kernel loop</li>
      <li>controller cycle</li>
      <li>periodic inference</li>
    </ul>
    <div class="callout red"><strong>The meter certifies a physical schedule, not training semantics.</strong></div>
  </div>
  <div>
    <img src="../figures/rung2_controls.png" style="max-height: 525px; max-width: 100%; object-fit: contain;">
  </div>
</div>

<p class="caption"><span class="scope orange">synthetic falsification controls</span> Fraction classified as training at false-positive rate 0.05 on the stated null.</p>

<!--
This is the payoff of the ladder. The classifiers are not broken: they correctly detect
the physics they were given. The problem is silently relabelling physical likeness as
semantic identity.
-->

---

<div class="section-kicker">Rungs 3 and 4</div>

# Stronger claims require changing the protocol

<div class="flow four">
  <div class="flow-box"><strong>Unpredictable challenge</strong>Verifier changes timing or workload conditions</div>
  <div class="arrow">→</div>
  <div class="flow-box"><strong>Physical response</strong>Power changes at the right time and shape</div>
  <div class="arrow">→</div>
  <div class="flow-box"><strong>Bound transcript</strong>Response commits to fresh state transitions</div>
  <div class="arrow">→</div>
  <div class="flow-box"><strong>Stronger claim</strong>Authenticated physical work</div>
</div>

<div class="two-col top" style="margin-top: 30px;">
  <div class="card orange">
    <h2>Rung 3 says</h2>
    <p>Something at the meter answered my fresh challenge.</p>
  </div>
  <div class="card red">
    <h2>Rung 4 still needs</h2>
    <p>Cryptographic or systems binding between that response and the claimed model update.</p>
  </div>
</div>

<p class="small muted"><span class="scope">protocol specified, not run</span> A co-located dummy responder remains the key gap without binding.</p>

---

<div class="section-kicker">Reality check</div>

# The one measured transport case fails the naive story

<div class="three-col" style="margin-top: 25px;">
  <div class="card orange center">
    <p class="big-number">0.4%</p>
    <p>iteration-cadence modulation of ≈370 W board power</p>
  </div>
  <div class="card red center">
    <p class="big-number">10×</p>
    <p>stronger power-controller line in the relevant band</p>
  </div>
  <div class="card navy center">
    <p class="big-number">75×</p>
    <p>gap from the scenario model's modulation depth</p>
  </div>
</div>

<div class="callout red">
  Every positive result in this talk is synthetic or literature-parameterized. The A100 trace is a <strong>negative transport case</strong>, not validation.
</div>

<p class="small muted">Falsifiable next step: synchronized multi-GPU communication should deepen and cohere the cadence at an external rack or wall meter.</p>

---

# What the present work establishes—and what remains open

<div class="two-col top">
  <div class="card teal">
    <h2>Established in scope</h2>
    <ul class="small">
      <li>A claim ladder with explicit information costs and ceilings</li>
      <li>Corpus-free calibration for a restricted structural null</li>
      <li>A synthetic de-periodicisation frontier with measured cost anchors</li>
      <li>A scenario-specific minimum meter boundary</li>
      <li>A constructive semantic impossibility result</li>
    </ul>
  </div>
  <div class="card orange">
    <h2>Not yet established</h2>
    <ul class="small">
      <li>External-meter detection on a real synchronized cluster</li>
      <li>A joint rule robust to both controller confusers and variable work</li>
      <li>The learning-efficiency cost where the tracker finally fails</li>
      <li>Passive proof that model state was updated</li>
      <li>An implemented active challenge and binding protocol</li>
    </ul>
  </div>
</div>

<div class="callout"><strong>This is a theory-and-methods result:</strong> a map of defensible claims, useful tests, attack costs, and hard limits.</div>

<!--
This slide is the contract with the audience. It also defines the paper revision: keep
the ladder and frontier central, and make the joint-rule gap and real cluster validation
explicit priorities rather than burying them.
-->

---

<!-- _class: takeaway -->

# Three things to remember

<div class="takeaway-row">
  <div class="n">1</div>
  <p>Passive power can provide calibrated evidence about a <strong>physical schedule</strong>.</p>
</div>
<div class="takeaway-row">
  <div class="n">2</div>
  <p>Against an adversary, the meaningful result is a <strong>detection–utility frontier</strong>, not “robust” or “not robust.”</p>
</div>
<div class="takeaway-row">
  <div class="n">3</div>
  <p>Claims about <strong>training semantics</strong> require extra information: a stated null, an active challenge, and ultimately binding to state transitions.</p>
</div>

<p style="color: #74d5cd; font-size: 25px; margin-top: 22px;">The meter is useful precisely when we are exact about what it cannot say.</p>

---

<!-- _class: appendix -->

# The paper spine implied by the talk

<div class="flow four">
  <div class="flow-box"><strong>1 · Identifiability</strong>What can a scalar trace possibly distinguish?</div>
  <div class="arrow">→</div>
  <div class="flow-box"><strong>2 · Claim ladder</strong>What extra information buys each stronger claim?</div>
  <div class="arrow">→</div>
  <div class="flow-box"><strong>3 · Frontier</strong>How cheaply can an adaptive prover hide?</div>
  <div class="arrow">→</div>
  <div class="flow-box"><strong>4 · Ceilings</strong>What remains impossible or unvalidated?</div>
</div>

<div class="callout" style="margin-top: 45px;">Detector construction supports the story; it should not become the story.</div>

<div class="three-col" style="margin-top: 28px;">
  <div class="card teal"><p><strong>Method role</strong><br>Operationalize the weakest claim.</p></div>
  <div class="card orange"><p><strong>Empirical role</strong><br>Price adaptive evasion.</p></div>
  <div class="card red"><p><strong>Control role</strong><br>Falsify semantic overclaiming.</p></div>
</div>

---

<!-- _class: appendix figure -->

# Full de-periodicisation frontier

![w:1100](../figures/st2_frontier.png)

<p class="caption">Cadence variation and phase diffusion versus detector power across all attack families. Solid: best tracking-class score per cell. Dashed: best fixed-class score per cell.</p>

---

<!-- _class: appendix figure -->

# A limited lower-bound direction

![w:700](../figures/lower_bound_spike.png)

<div class="callout orange"><strong>If an exhibited test still has advantage J, then the attacked and benign trace distributions are at least J apart in total variation.</strong></div>

<p class="caption">The converse does not follow: failure of our detector is not proof of covertness. The cost bridge shown here is specific to i.i.d. timing jitter and a fixed-bin verifier.</p>

---

<!-- _class: appendix figure -->

# Why analytic calibration was rejected

![w:1120](../figures/st1_calibration_far.png)

<p class="caption">The adaptive pipeline does not preserve the simple analytic reference law. Per-trace surrogates rerun the complete search; empirical false-alarm control is reported only over the tested structural null family.</p>

---

<!-- _class: appendix figure -->

# Rung 2 transfer tests

![w:930](../figures/rung2_stated_transfer.png)

<p class="caption">All three rules separate the scenario training population from the stated inference population across the tested shifts. This is conditional discrimination, not semantic identification.</p>

---

<!-- _class: appendix -->

# Evidence map

<table class="matrix">
  <thead>
    <tr><th>Claim</th><th>Evidence used here</th><th>Main missing validation</th></tr>
  </thead>
  <tbody>
    <tr><td><strong>Structural detector</strong></td><td>Synthetic bake-off + surrogate calibration</td><td>External aggregate traces</td></tr>
    <tr><td><strong>Adversarial frontier</strong></td><td>Synthetic attacks + 16 measured cost anchors</td><td>Prices for 34 cells; learning cost</td></tr>
    <tr><td><strong>Meter boundary</strong></td><td>Scenario observation sweep</td><td>Real meter transfer functions</td></tr>
    <tr><td><strong>Conditional classifier</strong></td><td>Synthetic stated null + transfer shifts</td><td>Representative deployment null</td></tr>
    <tr><td><strong>Semantic ceiling</strong></td><td>Constructive decoys + identifiability argument</td><td>Cannot be repaired passively</td></tr>
    <tr><td><strong>Active verification</strong></td><td>Protocol specification</td><td>Implementation and binding</td></tr>
  </tbody>
</table>
