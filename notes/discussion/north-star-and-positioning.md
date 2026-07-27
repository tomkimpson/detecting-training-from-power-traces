# North star & positioning — detecting-training-from-power-traces

**Date:** 2026-07-23
**Status:** strategic memo (rationale record, not a tracker). Captures a sense-making
discussion re-anchoring the project after the worry that it had "run off too fast in
too complicated a direction." Scope still lives in `spec.md`; status in `tasks.md`;
the detailed realisation in `notes/plans/plan-for-paper-2.md`. This memo is the thing
to re-read when the project feels sprawling.

---

## 0. Why this document exists

The paper had accreted a lot of moving parts and it had become hard to hold the whole
thing in one head. This memo fixes the **spine** (what the paper is), the **lineage**
(why each big choice was made), the **alpha** (why the paper is worth writing at all),
and — most usefully — a single **inclusion test** that decides what stays in the spine
and what drops to a satellite. If a future session is unsure whether a section earns
its place, apply the test in §6 and stop deliberating.

---

## 1. The spine — what the paper defends

### One sentence

> From a passive external power trace you can obtain *calibrated structural evidence*
> of iteration-structured training (Rung 1) and *conditional evidence* that a trace is
> more training-like than inference (Rung 2) — but not semantic proof that training
> occurred. An adaptive prover can erase the cadence, and the central result measures
> **how much it costs them to erase it** (the de-periodicisation frontier). Anything
> stronger requires active challenges plus a binding/transcript primitive (Rungs 3–4,
> sketched, not built).

### One paragraph

The paper is a **theory-and-methods, governance-facing** contribution: what a verifier
can and cannot certify about *training* from a time-resolved external power trace,
with what methods, at what cost in verifier information, and **against an adversary who
knows the detector**. All evidence is on a literature-parameterized scenario model
(Ko & Zhu 2025 for training/superposition; our own inference null); the model is
explicitly *not* calibrated to measured hardware (a ~75× gap separates the Ko-scale
modulation from the measured single-GPU cadence). Real hardware enters only as a
**negative transport case** and as measured cost anchors. The organising device is a
**claim ladder** whose central, defensible result is the **detection–utility
frontier** of de-periodicisation.

If you can restate the one-paragraph version from memory, you are not lost — the
project has a spine.

---

## 2. Lineage — why this is a coherent evolution, not a random complication

The predecessor (`paper-viterbi`, retired) was clean but died of two specific defects.
The two biggest design choices in this project are the **principled fixes** for those
two defects — that is the whole reason the surface area grew.

| Viterbi's fatal flaw | The design choice that answers it |
|---|---|
| The tracker locked onto **the wrong frequency** — it was picking up physical cadences that were *not* the training frequency, so the hardware experiments weren't "real". | **Abandon empirical NVML measurement; work on a synthetic scenario model** where the ground-truth cadence is known by construction. "Did we track the real line?" is no longer a question that can be answered wrong. |
| A null detection statistic required running on **the prover's own hardware**, which defeats the point of external verification. | The ST1 gate landed on **per-trace surrogate calibration** — corpus-free, needs no matched hardware. This is precisely the "we shouldn't need the prover's box" fix. |

**Takeaway:** the pivot is sound. Each major move is the correct response to a named
Viterbi failure. The structure is not the problem; the *breadth* is.

---

## 3. The honest cost of the pivot (the complexity diagnosis)

Moving to a synthetic model gave up the one thing that could have carried the paper on
its own — real measurements. A paper whose entire evidence base is an admittedly
uncalibrated generator (our own ~75× gap) cannot stand on results, so it stands on
**framing** — and framing accreted. As of this memo the paper carries roughly seven
distinct would-be contributions competing for the reader's attention:

1. the 4-rung claim ladder × 2 adversary settings;
2. the tracked cyclostationary pipeline (tracker → phase resampling →
   Dandawaté–Giannakis `Q_α`);
3. the de-periodicisation frontier;
4. the newly-promoted "minimum meter specification" contribution;
5. three competing decision rules for Rung 2;
6. the semantic-decoy falsification suite;
7. the observation-channel model + single-A100 negative transport case.

Each is individually justified in the plan. Collectively they are seven contributions
in a trench coat, and that is the source of the "too complicated / I've lost the
thread" feeling. The plan already names the underlying risk in its *fallback honesty*
note: **an F-test-only paper on an adopted synthetic generator is not automatically
publishable.** That sentence is the real worry underneath the complexity worry.

---

## 4. The alpha — why the paper is worth writing anyway

Two claimed edges. They are **not equal**, and being precise about which is load-bearing
is what unlocks the simplification in §5–6.

### (b) The adversarial frontier — the *real* alpha, insulated from the synthetic-model weakness

- "We detected training" is an **absolute** claim. On a generator we built it is partly
  circular — we recover the structure we injected.
- "To erase the cadence the adversary must pay at least X in throughput / learning
  efficiency" is a **conditional, relative** claim: *given* a workload with this
  structure, here is the achievable detection-vs-cost tradeoff.
- The conditional survives the ~75× calibration gap. We are not claiming the absolute
  signal depth is right; we are claiming the **shape of the game** is right. A frontier
  is a statement about the adversary's options, and those options are governed by the
  physics of *useful* training (you cannot fake synchronous gradient exchange for
  free), not by whether the generator is calibrated to a particular meter.
- **This is the part that cannot be dismissed as "your detector works on your own
  simulator."** Everything that makes the paper *about the frontier* strengthens the
  defensible core.

### (a) Governance framing — true, but it *re-points* the burden rather than removing it

- Governance / technical-AI-governance venues do reward "a method that works" over
  methodological novelty. Correct — and it means we get to **skip the novelty burden**:
  use standard tracked cyclostationary detection off the shelf, cite Thomson and
  Dandawaté–Giannakis, and stop agonising over "is this a new detector / can we claim
  analytic CFAR." That removes the expensive half of the methods work.
- **But** "works" now means "works on something a governance reader believes stands in
  for a real datacenter." So (a) does not lower the bar — it *moves* it from *is the
  method novel?* (skip) to *is it credible this transfers?* (own it). The realism /
  provenance burden goes **up**, not down.

**Net:** (b) is the load-bearing alpha; (a) is a licence to simplify the *methods*, not
a licence to skip *realism*.

---

## 5. The synthesis — the alpha argues for *less* machinery, not more

Both edges, followed honestly, push toward simplification:

- **(a) says: stop defending the method as novel.** Cite the standard detectors, use
  them, move on. The entire ST1 "can we claim analytic CFAR" anguish becomes a single
  honest paragraph: no analytic CFAR; per-trace surrogate calibration; here is its
  validity envelope. Done.
- **(b) gives a sharp inclusion test for everything else** (see §6). The tracked /
  cyclostationary pipeline earns its place *only* if it forces the adversary to pay
  materially more than a naïve F-test would. If it does, **that is the result** — state
  it as "the naïve detector is cheap to beat; tracking raises the adversary's price to
  X." If it does not move the frontier, it is dead weight.

The "seven contributions in a trench coat" problem dissolves: the frontier is the
paper, and the adversarial lens tells you which of the other six pieces are actually
holding it up.

---

## 6. The inclusion test (apply this whenever the paper feels sprawling)

> **A section stays in the spine if and only if it changes what the paper certifies or
> moves the adversary's cost frontier. Otherwise it is a satellite: appendix, one
> sentence, or future work.**

Corollary test for figures: **every figure must map to a clause of the one-sentence
thesis (§1).** If a figure does not defend a clause, it is a satellite.

### Triage as of this memo

**Spine (this is the paper):**

1. **Rung 1 — structural evidence**, stated at honest strength: evidence of coherent,
   repeated power structure under a stated structural null — *not* identification of
   training. Ceiling stated explicitly (periodic confounds, superposition, shaping,
   die-to-meter attenuation).
2. **The de-periodicisation frontier — the money figure and the genuine novel result.**
   The candidate lead is the counterintuitive finding in hand: work-varying jitter
   *appears* to move the cadence at ≈ zero throughput cost (within noise), whereas
   idle-insertion jitter costs 15–680% throughput and makes detection *easier*. Treat
   that asymmetry as **provisional** — the work-jitter line-band numbers ride the `B`
   component and must be requalified on the slurm number-freeze pass
   (`notes/results/issue54-investigation.md`). Stay agnostic about which finding leads
   until the frozen results are in; whatever survives requalification is the lead.
3. **Rung 2 — conditional classification**, with the **semantic decoy** as the honest
   limit (training-shaped execution without retained training scores as training ⇒ the
   meter certifies physical training-*likeness*, not semantic training).

**Satellites (demote / appendix / future work / one sentence):**

- **Rungs 3–4** (active challenge + binding primitive): keep as one short conditional
  section with the primitive stated as an ideal functionality. Resist any pull to
  build them.
- **Minimum meter specification** (the `integrating_1hz` finding): genuinely
  interesting, but it is a *second paper's* headline and currently competes with the
  frontier for attention. Demote to a subsection. (Finding: a 1 s trailing boxcar has a
  sinc null at the 1 Hz band centre and 1 Hz ZOH sampling pushes the whole
  f0 ~ U(0.5, 1.5) Hz band past the 0.5 Hz Nyquist ⇒ TPR@0.05 ≤ 0.32 for all detectors,
  order methods to chance. Honest caveat: Viterbi retains AUC 0.68 via band-edge
  aliasing, so "defeats at the stated operating point," not absolute erasure.)
- **The three Rung-2 decision rules:** report the one that best exposes how much
  performance comes from the chosen null vs the structural method; appendix the rest.

---

## 7. Figure-to-clause map (fill in as Phase 1 lands)

| Thesis clause | Figure(s) that defend it |
|---|---|
| "calibrated structural evidence … Rung 1" | ST1 bake-off + calibration/FAR figures |
| "how much it costs them to erase it — the frontier" | ST2 frontier (the money figure) + attack sweeps |
| "more training-like than inference … Rung 2" | Rung-2 classifier ROC + semantic-control panel |
| "not semantic proof" | semantic-decoy result (decoy scores as training) |
| "the channel can defeat all detectors" | meter-spec subsection (demoted) |
| "anything stronger needs challenges + crypto" | Rungs 3–4 conditional protocol (short) |

A figure with no clause is a candidate for the cut.

---

## 8. The kill / continue decision (confront before committing Phase 1 effort)

The upstream question is **not** "is the method good?" It is: **is the
synthetic-model evidence base strong enough to carry a paper, or is the machinery
compensating for thin ground?** The honest answer shapes everything.

The frontier (alpha b) is the answer to *most* of this — a conditional/relative result
does not need calibration. But the strong version of (b) needs **one** of the
following, or it is only "robust to the attacks we tried," not robust:

1. **An argument the attack families are canonical** — that jitter / drift /
   work-variation / relocation / superposition span what a hiding adversary can do, so
   the measured frontier *is* the frontier, not a sample of it; **or**
2. **A lower-bound argument** — "any schedule that hides the cadence to detectability ε
   must degrade learning-efficiency by ≥ f(ε)." This is the identifiability-theory
   question flagged in `plan-for-paper-2.md` §10.Q2, and it is the line between a
   *position paper* and a *contribution*. Even a weak, physics-level version ("hiding
   requires breaking synchronous gradient exchange, which is communication-bound and
   therefore costly") is worth more than one more attack in the sweep.

**Recommended next strategic step:** resolve *this specific question* (is the
synthetic evidence base enough, and can we get any lower bound?) before pouring Phase 1
writing effort in — via the ST0 prior-art audit first, then a time-boxed lower-bound
feasibility spike (§8.2). That is the real kill/continue gate.

---

## 9. The synthetic → real bridge (an asset we already own)

Pre-empt the "only works in your simulator" objection *without* the descoped GPU
campaign, using the issue-#54 result:

> The cadence provably exists on real silicon — a ~0.4%-of-board-power ripple on a
> single A100 — but is shallow there because one saturated card has no communication
> phase to modulate. Distributed synchronization is *predicted* to make the cadence
> deep, coherent, and legible. This is a falsifiable prediction, not a result claimed
> here.

This turns the negative transport case from an embarrassment into an honest, testable
link between the synthetic model and reality.

---

## 10. Do / don't (the discipline this memo encodes)

**Do**
- Lead with the frontier; make the paper *about the adversary's cost*.
- Use standard detectors off the shelf; cite, don't defend as novel.
- State every Rung-1 claim as *structural evidence*, never as *training identified*.
- Keep Rungs 3–4 to one conditional section.
- Apply the §6 inclusion test before adding anything.

**Don't**
- Claim methodological novelty for the tracker / cyclostationary machinery.
- Claim "analytic CFAR" — it is surrogate calibration, say so.
- Claim de-periodicisation is *necessarily* inefficient (our own work-jitter result
  refutes the naïve version).
- Let the meter-spec finding or the decision-rule bake-off crowd out the frontier.
- Overclaim transfer from the uncalibrated generator — the claims are conditional and
  relative, and should read that way.

---

## 11. Open questions carried forward

1. Can we state a **lower bound** on hiding cost (§8.2), even a weak physics-level one?
   This is the highest-value piece of new intellectual work available.
2. Are the attack families **canonical / spanning** (§8.1), and how do we argue it?
3. Which tracker (Viterbi / synchrosqueezing / Vold–Kalman) — settle via the bake-off,
   but do **not** treat the choice as a novelty claim.
4. Venue (governance / security / signal-processing vs empirical ML) — sets how much
   the identifiability theory must carry. arXiv-first for now (decided 2026-07-22).
