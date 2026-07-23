# Method soundness & prior-art positioning — a zoom-out

**Date:** 2026-07-23
**Status:** strategic memo (rationale record, not a tracker). A big-picture sanity
check: is the proposed detection pipeline the right way to attack the problem, or is
there an obvious solution / well-known technique / established problem framing we have
overlooked? Companion to [`north-star-and-positioning.md`](north-star-and-positioning.md).
Feeds the still-unowned **ST0 novelty/prior-art audit** (see `handoff.md`).

---

## 0. The setup

We have a single noisy external measurement `P_obs(t)` — power vs time, filtered
through the meter channel `h`. **That is all we have; no access to the prover's
hardware.** Two things we want from it:

- **(a) Detection:** are they training?
- **(b) Cost-of-hiding:** if they are training but we cannot see it, they must be
  paying a cost of at least `X` (throughput / learning-efficiency / energy).

## 1. The key reframing: (a) and (b) are different problems wanting different toolkits

Much of the project's apparent complexity comes from running **one** pipeline at both
questions. They are not the same problem:

| | Native question | Native tools |
|---|---|---|
| **(a) Detection** | Is this trace training vs the null? | Likelihood-ratio test, GLRT, matched filter, cyclostationary detection |
| **(b) Cost-of-hiding** | How much must behaviour be distorted so the training-trace *distribution* becomes indistinguishable from the null *distribution*? | Total-variation / KL bounds between distributions, Le Cam's method, **covert-communication theory** |

(b) is **not a detection problem at all** — it is a distinguishability / covertness
problem. Keeping them separate is what exposes the trap in §5.

---

## 2. On (a): the detection method is sound — but name it and benchmark it

The training signature is a quasi-periodic, frequency-wandering, second-order
structure buried in noise. **Cyclostationary detection with order tracking is the
canonical tool for exactly that** — this is not an exotic choice, and the pipeline is
defensible as-is. Two moves turn "is this the right way?" from an assertion into a
demonstrated fact:

### 2.1 Call it what it is — an approximate GLRT

The tracker → phase-resample → Dandawaté–Giannakis stack is, structurally, a
**generalized likelihood-ratio test**: the tracker performs the nuisance-parameter
(frequency-path) maximization; DG is the test statistic on the de-warped signal.
Naming it that (i) gives the pipeline a principled pedigree, and (ii) makes the
"path-selection look-elsewhere" worry a *known* GLRT phenomenon with standard
treatment, not a bespoke concern.

### 2.2 The move we may have overlooked — compute the optimal detector as a ceiling

We have a generative model, so we can compute (or closely approximate) the
**Neyman–Pearson optimal likelihood-ratio detector** — the clairvoyant best-possible
test between the training generator and the null generator. Report the practical,
corpus-free detector as **a fraction of that ceiling**:

> "the corpus-free surrogate detector achieves X% of NP-optimal power at Y% of the
> information cost."

This converts "is our method good?" from a vibe into a number. The Rung-2
"model-based discriminant" is halfway there — but it must be framed explicitly as the
**optimality ceiling**, not merely a third baseline. Without it we can never state how
much power we are leaving on the table, and therefore can never answer the user's
question.

### 2.3 The fields we must be positioned against (the "known technique" check)

Three mature literatures own pieces of this problem. The ST0 audit must place us
against each (exact references to be pinned by the audit; these are the fields, not
verified citations):

- **Non-intrusive load monitoring (NILM)** — inferring device/activity from a single
  aggregate power meter (Hart's problem). This is *literally our problem* at datacenter
  scale — the smart-meter / dishwasher analogy — with 30 years of technique. Absence
  from related-work is a reviewer red flag.
- **Spectrum sensing / LPI–LPD detection** (cognitive radio, electronic warfare) —
  detecting whether a source is active, and what it is doing, from noisy observations
  *against an adversary trying to stay below the detection floor*. The
  energy-detector vs matched-filter vs cyclostationary-detector taxonomy and its known
  SNR/robustness tradeoffs are exactly our Rung-1 bake-off. This is the closest
  structural twin to the whole enterprise.
- **Power side-channel analysis** (DPA / CPA / template attacks) — the hardware-security
  field on "infer the computation from the power trace." Different regime
  (high-bandwidth, proximate) but it is the prior art for the core premise that
  computation is legible in power.

None of these threaten novelty — we apply them to a new question (compute governance)
with a new adversary model. But "we adapt cyclostationary spectrum-sensing to the
compute-governance setting" is both **more credible and less work** than implying we
invented the detector (the north-star's "licence to simplify" in action).

---

## 3. On (b): this is the alpha, and it has a name — covert communication

The de-periodicisation frontier is, formally, the **warden-vs-covert-transmitter
game**: the prover wants to "train covertly" under a distortion budget; the verifier
is the warden trying to detect. **Covert-communication theory** (the "square-root law"
line — Bash / Goeckel / Towsley and descendants) studies exactly the tradeoff between
how constrained a hidden activity is and how detectable it remains, with
distinguishability measured by KL / TV between the two observation distributions.
Steganography's warden model is the same skeleton.

This is the literature that turns our **empirical** frontier into a **theorem**, and it
is the tool for the north-star kill/continue question ("can we state a lower bound on
hiding cost?"). The recipe:

> Bound the total-variation distance between the training-trace distribution and the
> null-trace distribution as a function of the adversary's schedule distortion. Any
> detector's power is capped by that TV distance (a standard hypothesis-testing
> inequality). So "TV small ⇒ undetectable" yields "to be undetectable, distortion
> must be ≥ f(ε)" — a statement about **every** verifier, not just ours.

---

## 4. The sharpest observation — the trap

**The current frontier measures the wrong direction of bound for the governance claim.**

- "Our detector's ROC as the adversary attacks harder" = **how much can be hidden from
  *us*.** That is an **upper** bound on hideability — a smarter adversary or a better
  detector moves it.
- The governance claim ("they must pay ≥ X") needs a **lower** bound on the cost of
  hiding *from an optimal verifier*.

An empirical sweep of *our* detector against *our* attack list cannot deliver the
second. It can only say "hiding from the specific detector we built, against the
specific attacks we imagined, costs X." To claim "hiding from anyone costs X" we need
either:

1. the **NP-optimal detector** from our generator as the strongest verifier
   (best-case detection, §2.2); or
2. the **TV/KL distinguishability bound** from covert-comms (best-case, detector-free,
   §3).

This is precisely why the north-star flagged the lower-bound question as the
position-paper-vs-contribution line. **The empirical frontier is necessary but not
sufficient** — it is evidence, and it upper-bounds hideability; the theorem is what
makes the governance claim.

---

## 5. Concrete recommendations

1. **Do the ST0 prior-art audit now, before Phase 1**, scoped to exactly four fields:
   NILM, spectrum-sensing / LPI–LPD, power side-channel, covert-communication /
   steganography. Cheap, overdue, and it either hands us tools or surfaces the paper
   that already did this.
2. **Add the NP-optimal LRT ceiling to the bake-off.** We have the generator; it costs
   little and it is the only honest answer to "is our detector good?"
3. **Frame (b) as a covertness lower bound** and test whether even a weak TV/KL bound
   is derivable. If it is, that is the headline theorem and the project graduates from
   position paper to contribution.

## 6. Open questions carried forward

1. Is a TV/KL lower bound on hiding cost derivable for even one attack family
   (e.g. i.i.d. phase jitter)? — the highest-value new theory.
2. How close is the corpus-free surrogate detector to the NP-optimal LRT ceiling, as a
   fraction of power? — quantifies §2.2.
3. Does the ST0 audit surface an existing method (in NILM or spectrum sensing) that
   beats or subsumes the tracked cyclostationary pipeline? — would force a redesign.
4. Which distinguishability metric (TV, KL, Hellinger) gives the cleanest
   distortion→detectability bound for our observation model?
