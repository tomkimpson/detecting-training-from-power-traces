# Detection vs. conviction: why the semantic ceiling does not make the meter useless

**Date:** 2026-08-21
**Status:** discussion memo — rationale for the paper's governance framing, written
after a conversation probing the "so is the method useless?" objection. Candidate
material for the introduction and the Rung 3 transition
(`paper/main.tex`, `\cref{sec:active}`).

## The objection chain

The objection arrives in three escalating steps:

1. *The semantic decoy seems self-undermining.* The paper claims the detector works,
   then shows a discarded-update decoy — full forward/backward/optimizer step, update
   thrown away — scores as training under every rule. Does the method work or not?
2. *Granted the rung structure resolves that — but isn't the ceiling moot?* In the
   governance scenario the prover WANTS to train and is not allowed. A violator will
   not fabricate evidence that incriminates them, so nobody runs a decoy.
3. *If a positive detection can be explained away ("that was the rack controller"),
   is the method useless for the rules we actually want, like "this cluster is
   inference-only"?*

## Resolution of step 1: two different questions

The decoy is not the detector failing; it is the detector doing exactly what it
certifies. Power depends on which computation ran, not on what was done with the
result — storing versus discarding a weight update is power-invisible, so the two
laws coincide and no test (ours or any future one) can separate them. The meter
certifies a *training-shaped physical schedule*; the decoy runs one. The thesis
sentence scopes to this from the start ("structural evidence … not universal
semantic identification"), and the ceiling is what makes Rungs 3–4 necessary rather
than optional.

**Correction to this memo's original framing (2026-09-02).** As first written, this
section left the impression that the decoy is an attack — and the manuscript made
that error explicitly, listing "semantic decoy" as a prover attack operator in
§4's adversary list beside jitter, burial and dilution. It is not one. Under the
paper's only threat model the prover conceals training, and manufacturing evidence
of the activity you are hiding buys nothing; the code had it right all along, where
the decoy lives solely in `rung2_scenarios.py` as a *control* with no attack budget
and no frontier cell. What is load-bearing is the non-identifiability result, whose
governance content is the innocent mimics and the semantic-rule/structural-evidence
gap, not a motivated decoy-runner. See "Manuscript actions" below.

## Resolution of step 2: the ceiling returns as the false-accusation problem

Correct that in the covert-training direction the binding limitation is *hiding*
(the de-periodicisation frontier), not the decoy — a violator will not mount it.
But the decoy is one member of an equivalence class that contains **innocent**
loads: the Rung-2 control table shows a non-ML kernel loop, a controller limit
cycle, and a periodic inference request generator all scoring 1.00 as training
under every rule. Flip the same information-theoretic fact to the accused's side:
*because* mimics are indistinguishable from training, a positive detection cannot
by itself prove training. The decoy's mootness as an attack is exactly what makes
it non-moot as a limit on evidentiary weight. It also bites directly in two other
governance settings: permits rather than bans (power cannot distinguish the
licensed run A from the forbidden run B), and positive attestation (treaty-style
"prove you ran the declared training", where a prover does want to fake training
and the decoy defeats any passive verifier).

## Resolution of step 3: detection instruments do not need to convict

**A detection instrument is not useless because it can't convict; almost no
verification instrument in any governance regime can convict on its own.** Four
arguments, in decreasing order of importance:

### 1. Verification regimes are layered; passive monitoring is the cheap outer layer

The working analogy is CTBT seismic monitoring: a seismometer network cannot
*prove* a signal was a nuclear test rather than an earthquake — the signal classes
overlap, and states have claimed "earthquake". Nobody concludes seismic monitoring
is useless. Its job is to be cheap, continuous, and non-intrusive, and to **cue
escalation**: an anomaly triggers on-site inspection, which the treaty machinery
provides for. IAEA safeguards work the same way — cameras and seals do not prove
diversion; they flag inconsistencies that trigger special inspections. The power
meter is that instrument: passive, external, requiring no prover cooperation,
always on. Rung 3 (challenge protocols) and plain physical audit are the
escalation layer, and the claim ladder is built in exactly that shape.

### 2. "That was the rack controller" is an excuse that makes falsifiable predictions

Benign confounds are specific physical processes with characterizable signatures,
not ghosts. The controller line was found empirically on real hardware: it lives
at 0.31–0.45 Hz, appears only under time-varying load, and has its own coherence
structure — and Rung 1's structural null and the Rung-2 controller control exist
precisely to exclude it. In deployment the verifier baselines the site's benign
lines at commissioning; after that, "it was the controller" is checkable — does
the detected line sit in the controller band, carry the controller's harmonic
structure, persist for weeks with the coherence of a synchronous run? An excuse
that must be sustained against a calibrated instrument under threat of inspection
is very different from a free pass.

### 3. The compliance rule can close the semantic gap by fiat

An "inference-only" agreement need not say "no training"; it can say "no
unexplained training-shaped schedules". Under that rule the prover has no
legitimate reason to run a periodic kernel loop that mimics training, and the
burden of explaining a detected cadence sits with them by construction. This is
standard verification design: the rule is written so that ambiguity itself is a
violation or a trigger, rather than asking the instrument to resolve ambiguity.
The meter's job then reduces to the thing it *can* certify — the schedule — which
is exactly what the paper says it certifies.

### 4. The cheater's fork is genuinely bad (deterrence value)

A prover actually training on an inference-only cluster faces a fork the paper
prices: either the cadence shows up — and they must survive escalation while the
forbidden run is physically present on the machines — or they de-periodicise and
pay the measured systems cost, forever, on their production workload. The meter
converts "train quietly and nobody knows" into "pay a running tax or accept
escalation risk". Deterrence does not require conviction-grade evidence; it
requires making cheating costly and detection plausible.

### The counterfactual

Without the meter, the prover does not even need the rack-controller excuse —
nothing fires at all. A tripwire that forces the adversary to either pay a hiding
cost or manufacture and sustain explanations under audit is doing real governance
work. The paper's contribution is to say *quantitatively* what that tripwire can
certify, at what meter specification, against what adversary budget. "Here is
exactly how much this instrument is worth" is more useful to a governance designer
than an overclaimed method — regimes built on overclaimed instruments fail in the
field, not in review.

## Manuscript actions

**Done 2026-09-02** (branch `fix/decoy-attack-framing`) — nine edits to
`paper/main.tex` demoting the decoy from attack to proof device, and promoting the
innocent mimics to the practically binding consequence:

- **§4 adversary list:** semantic decoy removed from the attack operators; the
  prover's concealment objective stated explicitly; a following paragraph says why
  the decoy is *not* on the list, and notes that Rung 3's incentive flip is what
  makes the challenge-aware decoy a genuine attack.
- **§4 after `prop:noident`:** new paragraph — the proposition binds with no
  adversary present, most of the equivalence class is benign, and the limit is
  two-sided (it caps positive evidentiary weight as much as it admits counterfeits).
- **§6 "semantic controls set the ceiling":** control list reordered to put the
  benign loads first and name them the practically binding ones; the decoy demoted
  to extremal member that cannot be engineered away.
- **§7 Rung 2→3 transition, §7 challenge-aware ceiling, governance reading,
  contributions bullet, both ladder tables, intro ladder preview:** "the semantic
  decoy" → "semantic non-identifiability" as the name of the Rung-2 ceiling, with
  the two-sidedness carried through.

Build verified clean: 0 errors, 0 overfull boxes, 0 undefined references, 28 pages.

**Still not done** — the screening-instrument framing from step 3 above (CTBT/IAEA
tradition: passive wide-area cueing → targeted escalation) is not yet in the
manuscript. One or two sentences in the introduction or the Rung 3 transition would
preempt the "so it's useless" reading from a policy-minded reviewer. Deliberately
left out of the present pass, which was scoped to the decoy-as-attack category
error.
