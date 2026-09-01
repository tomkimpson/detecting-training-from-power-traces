# Methods critique: do the detectors work, and what to fix before submission

**Date:** 2026-09-01 · **Status:** critique memo with recommended fixes. Rationale record,
not a tracker — status lives in `handoff.md`, scope in `spec.md`. Reviewed against
`paper/main.tex` (full), `results/st2/meter_boundary_summary.json`,
`results/rung2/rung2_summary.json`, `powerladder/config.py`.

---

## 0. Verdict

The **methods are correct; the evidence base is the weak link, and the most valuable
results are the negative ones.** The detector stack (Viterbi tracker → tacholess order
resampling → Dandawaté–Giannakis) is the canonical tool for a frequency-wandering
second-order signature — no referee will call it the wrong instrument. `subsec:st1` (staged
adaptivity isolation, 10⁴ nulls/cell, the swept covariance estimator, the refusal to claim
CFAR) is the strongest methodological asset in the paper and is above the norm for this
literature.

What is weak is what the numbers rest on: a generator we built, at a modulation depth we
cannot verify, against a null designed to be line-free. Three specific items below; a
fourth (all-synthetic evidence) is deferred to the hardware follow-up (§5).

**Novelty read.** What lands with a verification-literature reader: the claim ladder as a
vocabulary for *what a physical channel certifies at what information cost*; the adversary-
with-a-priced-budget frontier (NILM has no adversary, spectrum sensing has no cost model);
and the negative results. What does not land as novel: cyclostationary detection of periodic
compute, and "training has an iteration cadence" — both are prior art and the paper says so.

---

## 1. The 1 Hz limit: what a standard meter can actually constrain

### The finding we are currently under-reading

From `meter_boundary_summary.json`, at the cadence band 0.5–1.5 Hz:

| cell | Viterbi AUC | Viterbi TPR@0.05 |
|---|---|---|
| `fs=20, iw=0` (reference) | 1.00 | 1.00 |
| `fs=20, iw=0.5` | 1.00 | 0.99 |
| `fs=20, iw=1.0` | 0.35 | 0.25 |
| `fs=2, iw=0` | 1.00 | 1.00 |
| `fs=1, iw=0` | **0.75** | **0.36** |
| `fs=1, iw=1.0` | 0.66 | 0.30 |
| `fs=0.5, any` | 0.50 | 0.00 |

Two things follow that the manuscript does not currently say.

**(a) The binding axis is aperture, not sample rate.** A 2 Hz meter with a short aperture
(AUC 1.00) beats a 20 Hz meter that averages over 1 s (AUC 0.35). The spec as written
("≥2 Hz *and* ≤0.5 s") is correct but presents the two as co-equal; they are not, and the
distinction is exactly the one that separates instrument classes in the field. Commodity
revenue/PDU telemetry is *interval-averaging* — the failure mode is the aperture, not the
reporting rate.

**(b) 1 Hz is not dead; it is aliased, and we searched the wrong band.** At `fs=1, iw=0`
the tracker still holds AUC 0.75. That is not chance. The observation map decimates
deliberately *without* an anti-alias filter (`subsec:obsmap`), so a cadence at
f₀ ∈ [0.5, 1.5] Hz folds to |f₀ − 1| ∈ [0, 0.5] Hz — while the detectors go on searching
0.3–1.7 Hz, most of which lies above the 0.5 Hz Nyquist edge. **We have never run an
alias-aware detector at 1 Hz.** The paper reports "1 Hz collapses it to 0.36 as the Nyquist
edge crosses the band" as though that were a channel limit; part of it is a detector
mismatch we introduced.

### Recommended experiment (cheap, CPU-only, no new machinery)

Add a band-folded search: at sample rate f_s, map each candidate cadence f₀ to its alias
|f₀ − n·f_s| and run the existing tracker on the folded grid. Detection does not need f₀ to
be identifiable — and it is not, since 0.8 Hz and 1.2 Hz alias together — only coherent
structure. The null aliases too and raises the floor, so this is a genuine experiment, not
a free win. Expected outcome: 1 Hz *instantaneous* sampling recovers materially more than
0.36, while 1 Hz *integrating* stays dead (the 1 s boxcar sinc-nulls the band centre
irrespective of aliasing).

### How to frame the recommendation

Yes — frame it as a procurement/retrofit specification, but restructure it:

1. **Aperture first.** Integration window ≤ 0.5 s. This is the requirement commodity
   metering fails, and it fails it by design (interval energy averaging).
2. **Then rate.** ≥ 2 Hz for direct observation; **1 Hz instantaneous sampling may be
   sufficient with an alias-aware detector** (pending the experiment above) — a much cheaper
   retrofit ask, and worth far more politically than "2 Hz or nothing".
3. **No deep in-band notch** at the cadence.

And state the spec **as a function of modulation depth and record length**, not as a fixed
box. Every number above is at Ko–Zhu's assumed depth (`mu_delta_tr = 0.3`); at the measured
single-card 0.4% no setting in the swept grid helps. The spec is contingent on the
distributed-depth prediction and must say so on its face.

### What a 1 Hz integrating meter *can* still constrain

Worth one honest paragraph, because "nothing" is wrong and a regulator will ask. Minutes-scale
structure survives any aperture: job start/stop edges, sustained-load duty cycle, checkpoint
and evaluation periodicity, power-envelope steps, diurnal pattern. These support claims well
below Rung 1 — "a large sustained load ran here, with slow periodic structure consistent with
checkpointing" — not "iteration-structured training". If we want to say it, it is a **Rung 0**
and should be labelled as one rather than smuggled into the meter discussion.

---

## 2. Why every AUC is 1.00

It is not a bug, not overfitting, and not a leak. It is the benchmark being saturated, and
there are four compounding reasons:

1. **The null is line-free by construction.** `subsec:null`: "the design intent is that none
   of the three carries a line." We then report that line detectors separate it perfectly.
   Rung 2 is close to circular as posed.
2. **The SNR is enormous.** `mu_delta_tr = 0.3` — a 30% coherent modulation, integrated over
   ~300 cycles in a 300 s record at 20 Hz. The deflection is huge; saturation is the expected
   outcome, not a surprising one.
3. **The Whittle NP ceiling confirms it.** AUC = TPR = 1.0 in *every* column at *every* drift.
   The two generators are perfectly separable in principle. The paper's own gloss — "all the
   difficulty lives in the detector, not the problem" — is the correct reading and should be
   the framing everywhere 1.00 appears.
4. **The transfer shifts do not touch the deciding variable.** Headroom, duration, cadence
   band, meter noise, controller: none of them changes *line present vs line absent*. AUC
   1.00 across six shifts demonstrates that the shifts are orthogonal to the discriminating
   feature — not that the method is robust.

### The fix: add the missing axis

**Sweep modulation depth.** `mu_delta_tr` is a single config parameter; the sweep is
CPU-only and cheap. Report AUC and TPR@FAR as a function of depth from 30% down through
1% to the measured 0.4%, at fixed record length. This one axis does four jobs at once:

- Replaces the uninformative 1.00 with an operating curve a reader can interrogate.
- Converts the 75× gap from a standing caveat into a **quantitative sensitivity result** —
  "here is the depth at which each rung dies" — which is what `subsec:sensitivity` promises
  but does not yet deliver.
- Makes the meter specification a surface in (depth, aperture, rate) rather than a box
  measured at one privileged depth.
- **Prices the hardware campaign.** It states, in advance, the modulation depth the
  distributed measurement has to find for the paper's claims to hold — i.e. it turns
  `sec:reality`'s falsifiable prediction into a number a collaborator can go and test.

Interim wording change regardless of whether the sweep lands: stop leading with 1.00. Report
it as "the stated task is saturated; the informative experiments are the semantic controls
and the frontier."

---

## 3. The false-alarm regime — the critical fix

### The gap

Calibration exists at 0.05 and 10⁻²; the surrogate null uses S = 999 (finest resolvable
p ≈ 10⁻³) and the bake-off's 10⁻² threshold rests on the second-highest of n = 200 negatives.

Governance arithmetic: continuous monitoring on 300 s records is ~1.05 × 10⁵ tests per site
per year. At FAR 10⁻² that is ~1,000 false alarms per site per year. For ≤ 1 per site per
year you need FAR ≲ 10⁻⁵; across a 100-site fleet, ≲ 10⁻⁷. **We are three to five orders of
magnitude away from the operative regime, and nothing in the campaign design can reach it.**

Compounding it: surrogate calibration encodes *linear stationarity*, and the one real-hardware
null we have — the A100 controller limit cycle — already breaches it 3–6× at nominal 0.05
(realised 0.17–0.30). So the calibration is both too coarse and, on real traces, wrong in the
inflating direction.

### Recommended fixes, in order

1. **Reframe the deployment claim (do this regardless, costs nothing).** A per-window FAR is
   the wrong operating concept for continuous monitoring. Either move to a **sequential /
   anytime-valid** treatment (e-values, test martingales) — currently deferred in `Scope`, and
   it should be promoted, because it *is* the deployment regime — or state the detector as a
   **screening statistic that triages for follow-up**, whose operative quantity is positive
   predictive value under a stated base rate, not FAR. The screening framing is honest, is
   what a regulator would actually deploy, and does not require 10⁻⁶.
2. **Extend the calibration by tail extrapolation.** Fit a generalised Pareto (peaks-over-
   threshold) to the upper tail of the surrogate null; that reaches p ≈ 10⁻⁶–10⁻⁷ from ~10³–10⁴
   surrogates. Standard practice in gravitational-wave searches and genomics multiple testing.
   Must be validated against one brute-force ~10⁶ null cell before it is trusted.
3. **Accept that the binding constraint is the null, not the arithmetic.** Extrapolating a
   *wrong* null to 10⁻⁶ is worse than useless — a 3–6× breach at 0.05 becomes an unbounded one
   in the far tail. Reaching the governance FAR regime honestly needs a **measured null corpus**
   from real hardware: idle, inference, mixed, controller-active, thermally cycling. That is a
   deliverable of the hardware follow-up (§5), and it should be named as such rather than left
   as a limitation.

For this paper, (1) is mandatory and (2) is a strong addition. (3) is scoped out with an
explicit statement of what it would take.

---

## 4. One further item: the straggler argument (highest-value single addition)

Not on the original list, but it is the cheapest material strengthening available and it
patches the weakest joint in the paper — `ass:cost` / `rem:costgap`.

In **synchronous data-parallel** training the variable-real-work attack has only two
realisations:

- Vary micro-steps **identically across ranks** → the schedule stays globally coherent, and
  the tracker sees exactly the structure it was built for.
- Vary them **per rank** → the all-reduce barrier waits for the slowest, and the prover pays
  the straggler tail directly in throughput.

That is a *mechanism*, not an assumption, and it converts the weakest leg of the central claim
from "argued" to "physically forced". It also repairs a mismatch we do not currently flag: we
price the work-variation attack at ≈0% overhead using **single-A100 anchors from a card with
no communication phase** — the one regime in which that attack really is free.

Second, smaller: sharpen the semantic decoy's governance reading. The decoy is statistically
free but costs the prover *a full training run's worth of compute*. Under a FLOP-threshold
regime, an actor who must burn threshold-scale compute to manufacture deniability has already
conceded the thing the regime measures. The current framing is more pessimistic than the
economics require.

---

## 5. Deferred to the hardware follow-up

All-synthetic evidence is the fourth major criticism and is **out of scope for this paper** —
delegated to a follow-up with hardware access (contacts at Lucid Computing and Amodo). To keep
that follow-up well-posed, this paper should state the measurements it needs, and §2's depth
sweep should state them *quantitatively*:

1. **Modulation depth at cluster scale** vs the single-card 0.40% — the load-bearing prediction
   of `sec:reality`, and the axis §2 turns into a threshold.
2. **Simultaneous on-device and external rack/wall metering** — the first real measurement of
   the transfer function `h`, currently an LTI first approximation.
3. **A measured null corpus** — the prerequisite for any honest claim below FAR 10⁻³ (§3.3).
4. **Real meter aperture and rate** as deployed, to test the §1 spec against instruments that
   exist rather than against a swept grid.

---

## 6. Recommended action list

| # | Action | Cost | Section |
|---|---|---|---|
| 1 | Reframe FAR as screening PPV or anytime-valid; stop implying a deployable per-window FAR | wording | §3.1 |
| 2 | Modulation-depth sweep; replace AUC 1.00 headlines with the operating curve | one config axis, CPU | §2 |
| 3 | Alias-aware band-folded detector at 1 Hz | small, CPU | §1 |
| 4 | Restate the meter spec: aperture first, rate second, as a function of depth | rewrite + §2 output | §1 |
| 5 | Add the synchronous-straggler mechanism to `ass:cost` | prose | §4 |
| 6 | GPD tail extrapolation of the surrogate null, validated at one cell | moderate, slurm | §3.2 |
| 7 | Name the hardware follow-up's four measurements explicitly | prose | §5 |

Items 1, 4, 5 and 7 are prose-only and should land regardless. Items 2 and 3 are the two
experiments that most change what the paper can claim.
