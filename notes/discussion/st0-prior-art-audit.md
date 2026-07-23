# ST0 prior-art audit — four fields

**Date:** 2026-07-23
**Status:** prior-art / novelty audit (rationale record, not a tracker). Closes the
"novelty audit skipped" hole left by the Phase-0 ST0 desk work (`tasks.md`, Phase 0 →
ST0). Point-in-time record; status lives in `tasks.md`. Companion to
[`method-soundness-and-prior-art.md`](method-soundness-and-prior-art.md) (which named
these four fields but flagged them as "the fields, not verified citations") and
[`north-star-and-positioning.md`](north-star-and-positioning.md) (the positioning
stance: *cite standard detectors, don't defend them as novel*). Detector-method lineage
that is already verified lives in
[`power-verification-paths-forward.md`](power-verification-paths-forward.md) §12 and is
cross-linked here rather than re-cited.

---

## 0. Purpose & scope

The project positions a passive-power-meter training detector, yet the original ST0
desk work closed with the novelty audit skipped — so the work is currently positioned
against *none* of the four mature literatures that own pieces of the problem. This audit
closes that hole. It has **two jobs**:

1. **§1 related-work prerequisite.** The paper's `sec:intro` related-work checklist is
   blocked on it, and `references.bib` has no entries for these fields. This note pins
   *verified* citations and drafts the one-line positioning per field so the §1 write-up
   (and the follow-up `references.bib` additions) can proceed.
2. **Gate on the lower-bound feasibility spike** (`tasks.md`, next Strategic gate). The
   covert-communication literature is where any TV/KL lower bound on hiding cost would
   come from. The audit either hands over that machinery or surfaces a paper that already
   derived it — a kill-check for the spike.

The four fields split cleanly across the two questions the project keeps separate
(`method-soundness-and-prior-art.md` §1):

| Field | Feeds |
|---|---|
| NILM | (a) detection |
| Spectrum sensing / LPI–LPD | (a) detection |
| Power side-channel analysis | (a) detection — the *premise* that computation is legible in power |
| Covert communication / steganography | (b) cost-of-hiding — the distinguishability/covertness bound |

**Scope of this note:** the audit record only. Adding the citations to
`references.bib` and writing the §1 prose are left as follow-ups (§7 lists exactly what
they should contain). All citations below were verified against publisher / DBLP / IACR
records on 2026-07-23; they still need a `/check-refs` pass before any arXiv push.

---

## 1. Non-intrusive load monitoring (NILM)

**What the field owns.** Inferring which devices/activities are running from a *single
aggregate* power meter — Hart's original "non-intrusive appliance load monitor" problem.
This is *literally our problem* (one aggregate meter → what is the load doing) at
datacenter scale rather than household scale; the smart-meter / dishwasher analogy is
exact. Thirty-plus years of disaggregation technique (edge/event detection, steady-state
signatures, HMM/factorial-HMM state models, and latterly deep sequence models).

**Closest prior work (verified).**
- **Hart, G. W. (1992). "Nonintrusive Appliance Load Monitoring." *Proceedings of the
  IEEE* 80(12): 1870–1891.** The foundational paper; establishes single-meter
  disaggregation.
- **Zoha, A., Gluhak, A., Imran, M. A., & Rajasegarar, S. (2012). "Non-Intrusive Load
  Monitoring Approaches for Disaggregated Energy Sensing: A Survey." *Sensors* 12(12):
  16838–16866. DOI 10.3390/s121216838.** A modern survey for "thirty years of
  technique."

**How we position.** We are non-intrusive load monitoring lifted to the compute-governance
setting: the "appliances" are training vs. non-training workloads, the meter is off-chip
and untrusted, and — the load-bearing difference — the load *actively hides* from the
monitor. Classical NILM assumes cooperative or at least indifferent loads; it has no
adversary who knows the disaggregator and reshapes its power signature to evade it.

**Does it subsume us?** No. NILM's device signatures are steady-state or transient-edge
features under a benign load; it does not target the quasi-periodic, frequency-*wandering*
second-order structure our tracker exploits, and it has no knows-the-detector adversary
model. It is the closest *problem* analogue and a mandatory citation, not a method that
beats our pipeline.

## 2. Spectrum sensing / LPI–LPD detection (cognitive radio, EW)

**What the field owns.** Deciding whether a source is active — and what it is doing —
from noisy observations, explicitly *against a transmitter that may be trying to stay
below the detection floor* (low-probability-of-intercept / low-probability-of-detection).
This is the closest **structural** twin to the whole enterprise. Its canonical detector
taxonomy — **energy detector vs. matched filter vs. cyclostationary feature detector** —
*is* our Rung-1 bake-off, and its known SNR/robustness tradeoffs (cyclostationary
detection is robust to noise-power uncertainty where the energy detector fails at low
SNR; the matched filter is optimal but needs a known template) are exactly the tradeoffs
we report.

**Closest prior work (verified).**
- **Yücek, T., & Arslan, H. (2009). "A Survey of Spectrum Sensing Algorithms for
  Cognitive Radio Applications." *IEEE Communications Surveys & Tutorials* 11(1):
  116–130.** The canonical survey laying out the energy/matched-filter/cyclostationary
  taxonomy and its tradeoffs.
- The cyclostationary-detection lineage itself (Gardner's spectral-correlation theory;
  Dandawaté–Giannakis time-domain cyclostationarity test) is already verified in
  `power-verification-paths-forward.md` §12 and cited in `references.bib` as our method
  lineage — cross-link, do not duplicate.

**How we position.** We adapt cognitive-radio spectrum sensing to compute governance:
the "primary user" is a training job, the observation is an off-chip power trace rather
than an RF band, and the adversary's evasion is schedule de-periodicisation rather than
LPI waveform design. Framing the bake-off as "the standard spectrum-sensing detector
taxonomy, evaluated on power traces against a de-periodicising adversary" is both more
credible and less work than implying we invented the detectors (the north-star's
"licence to simplify").

**Does it subsume us?** No — but it is the field most likely to contain a method we
should adopt rather than reinvent. The taxonomy and the noise-uncertainty robustness
results transfer directly; what does *not* exist there is (i) the frequency-*wandering*
line + order-tracking front end (our tracker → phase-resample → DG stack, which is
structurally a GLRT with the tracker doing nuisance-parameter maximisation —
`method-soundness-and-prior-art.md` §2.1), and (ii) our specific adversary (a training
schedule that can be smeared in time at a measurable throughput/efficiency cost). No
spectrum-sensing result subsumes the pipeline; several strengthen its pedigree.

## 3. Power side-channel analysis (DPA / CPA / template attacks)

**What the field owns.** The hardware-security field on "infer the computation (or the
secret it processes) from the power trace." This is the prior art for our **core
premise** — that computation is legible in power — established long before us, just in a
different regime (high-bandwidth, physically proximate, often keyed to a known
instruction/data model).

**Closest prior work (verified).**
- **Kocher, P., Jaffe, J., & Jun, B. (1999). "Differential Power Analysis." *CRYPTO
  1999*, LNCS 1666: 388–397.** Introduced DPA; the origin of "computation is legible in
  power."
- **Brier, E., Clavier, C., & Olivier, F. (2004). "Correlation Power Analysis with a
  Leakage Model." *CHES 2004*, LNCS 3156: 16–29.** CPA — the leakage-model / correlation
  refinement.
- **Chari, S., Rao, J. R., & Rohatgi, P. (2002). "Template Attacks." *CHES 2002*, LNCS
  2523: 13–28.** The information-theoretically strongest side-channel attack; the
  profiled/generative-model analogue of our NP-optimal-detector ceiling idea
  (`method-soundness-and-prior-art.md` §2.2).

**How we position.** We inherit the premise but invert the regime. Side-channel analysis
assumes a *proximate, high-bandwidth* probe and typically a *cooperative* target running
a known algorithm; it extracts fine-grained secrets (keys). We assume a *remote,
low-bandwidth, filtered* meter and an *adversarial* target, and we extract only a coarse
structural fact ("is this efficient iteration-structured training?"). The template-attack
idea — build a generative model of the trace and test against it — is the honest
precedent for reporting our corpus-free detector as a fraction of a generator-based
optimal ceiling.

**Does it subsume us?** No. Different channel bandwidth, different threat model
(cooperative vs. adversarial), different estimand (secret extraction vs. coarse workload
certification). It is a premise citation, not a competing method. (Recent *GPU* power
side-channel extraction work already lives in the paper's "extraction corpus" citations;
this audit adds the classical DPA/CPA/template roots behind it.)

## 4. Covert communication / steganography — the warden game

**What the field owns.** The formal **warden-vs-covert-transmitter game**: a transmitter
wants to convey/act under a distortion or power budget while a warden tries to detect any
deviation from the "innocent" distribution, with detectability measured by a
distributional distance (KL / total variation) between the two observation
distributions. This is the field that turns our **empirical** de-periodicisation frontier
into a **theorem** — and the one that matters for question (b), cost-of-hiding.

**Closest prior work (verified).**
- **Bash, B. A., Goeckel, D., & Towsley, D. (2013). "Limits of Reliable Communication
  with Low Probability of Detection on AWGN Channels." *IEEE Journal on Selected Areas
  in Communications* 31(9): 1921–1930. DOI 10.1109/JSAC.2013.130923.** The **square-root
  law**: only O(√n) covert bits over n channel uses; more forces either warden detection
  (prob → 1) or decoding failure. Conference version: *ISIT 2012*, pp. 448–452. This is
  the direct machinery for "to stay hidden you must pay ≥ f(ε)."
- **Cachin, C. (2004). "An Information-Theoretic Model for Steganography." *Information
  and Computation* 192(1): 41–56. DOI 10.1016/j.ic.2004.02.003** (preliminary version:
  *2nd Information Hiding Workshop*, LNCS 1525, 1998). Casts the warden's task as a
  hypothesis test and quantifies steganographic security by the **relative entropy (KL)**
  between cover and stego distributions — i.e. the passive-warden bound, the exact
  skeleton of our verifier's problem.

**How we position.** Our de-periodicisation frontier *is* a covert-communication problem:
the prover "trains covertly" under a schedule-distortion budget; the verifier is the
warden. The recipe (`method-soundness-and-prior-art.md` §3): bound TV between the
training-trace and null-trace distributions as a function of the adversary's distortion;
every detector's power is capped by that TV (standard hypothesis-testing inequality); so
"TV small ⇒ undetectable" gives "to be undetectable, distortion ≥ f(ε)" — a statement
about **every** verifier, not just ours.

**Does it subsume us?** No — and this is the important negative result of the audit. The
covert-comms square-root law is proved for a transmitter that *chooses* an
information-bearing codebook over an AWGN channel with a per-use power constraint. Our
setting is different in three ways that block a drop-in application: (i) the "signal" is a
training *schedule* with physics-imposed structure (synchronous all-reduce is
communication-bound), not a free codebook; (ii) the constraint is a
throughput/learning-efficiency cost, not a per-symbol power budget; (iii) our observation
channel is a filtered, low-rate meter, not AWGN. **No existing covert-comms paper derives
the hiding-cost bound for our object** — which is good news for novelty and is exactly
the gap the lower-bound spike must fill (§6).

---

## 5. Subsumption verdict (source-memo open question 3)

**Does any of the four fields already beat or subsume the tracked-cyclostationary
pipeline? No.** Reasoned verdict:

- **NILM** — same *problem* (single-meter disaggregation), no adversary, no
  wandering-line second-order structure. Problem citation, not a competing method.
- **Spectrum sensing** — same *detector taxonomy* (our bake-off is theirs), and the most
  likely source of a reusable technique, but it lacks the wandering-line order-tracking
  front end and our schedule-distortion adversary. Strengthens pedigree; does not subsume.
- **Power side-channel** — same *premise* (computation legible in power), different
  channel bandwidth / threat model / estimand. Premise citation.
- **Covert comms** — same *game skeleton* (warden vs. hider), but no existing result
  covers a physics-constrained training schedule over a filtered low-rate meter.

The novelty claim the positioning memos recommend holds: **we apply mature detectors to a
new question (compute governance) with a new adversary model (a hider who knows the
detector), over a new channel (an untrusted off-chip meter).** No redesign is forced. The
one field to keep actively scanning during Phase 1/2 is spectrum sensing, since it is the
likeliest to yield a method worth adopting rather than reinventing.

## 6. Lower-bound-spike readiness (kill-check for the next gate)

**Does covert-comms hand over the tools the spike needs? Partially — the machinery yes,
a ready-made theorem no.** Concretely:

- **Hands over:** the framing (warden game), the metric (KL/TV between trace
  distributions), and the capping inequality (detector power ≤ TV). Cachin's relative-
  entropy security definition and the Bash–Goeckel–Towsley square-root law are the
  templates for "distortion ≥ f(ε)."
- **Does *not* hand over:** a bound for *our* object. The square-root law's codebook /
  AWGN / per-use-power assumptions do not match a physics-constrained training schedule
  over a filtered meter (§4). So the spike is **not pre-empted** by an existing paper —
  the "did someone already do this?" kill-check comes back *no*.

**Implication for the spike:** proceed, but scope it to deriving a TV/KL bound for one
concrete attack family (the memo suggests i.i.d. phase jitter) under our observation
model, using the covert-comms inequalities as scaffolding. The honest anchor is physical:
hiding synchronous all-reduce is communication-bound and therefore costly — the TV/KL
wrapper formalises what the physics already suggests. The fork stands
(`tasks.md`): a derivable bound ⇒ contribution ("must pay ≥ X"); no bound ⇒ position
paper (evidence + upper bound on hideability, necessary conditions only).

## 7. What flows into the paper (follow-up scope)

**Per-field one-line positioning sentences** (ready to lift into §1):
- *NILM:* "Certifying training from one aggregate meter is non-intrusive load monitoring
  [Hart 1992; Zoha 2012] lifted to compute governance, with a load that actively hides
  from the monitor."
- *Spectrum sensing:* "Our Rung-1 detector bake-off is the cognitive-radio spectrum-
  sensing taxonomy — energy vs. matched-filter vs. cyclostationary [Yücek & Arslan 2009]
  — evaluated on power traces against a de-periodicising adversary."
- *Side-channel:* "That computation is legible in power is the founding premise of power
  side-channel analysis [Kocher 1999; Brier 2004; Chari 2002]; we invert the regime to a
  remote, low-bandwidth, adversarial meter and a coarse structural estimand."
- *Covert comms:* "The de-periodicisation frontier is a covert-communication /
  steganographic warden game [Bash–Goeckel–Towsley 2013; Cachin 2004]; a KL/TV
  distinguishability bound is what would turn the empirical frontier into a
  cost-of-hiding theorem."

**Placement recommendation:** a short **own `\section{Related work}`** (or a dense single
subsection) organised by these four fields, *plus* the existing inline governance /
telemetry-competitor citations kept in §1. Four distinct literatures with a one-paragraph
positioning each read better as a labelled section than woven inline; it also gives the
reviewer the "we know NILM / spectrum sensing exist" signal in one glance (the reviewer
red-flag the memo warns about). Resolve the §1 checklist's "own section vs woven" question
this way.

**`references.bib` additions (follow-up; each still needs `/check-refs`):**
`hart1992nilm`, `zoha2012nilm`, `yucek2009spectrum`, `kocher1999dpa`, `brier2004cpa`,
`chari2002template`, `bash2013covert`, `cachin2004steganography`. Do **not** re-add the
Gardner / Dandawaté–Giannakis / Thomson method-lineage entries — already in
`references.bib`.

## 8. Open questions carried forward

1. Is a TV/KL lower bound on hiding cost derivable for one attack family (i.i.d. phase
   jitter) under our filtered-meter observation model? — handed to the lower-bound spike
   (§6); the audit confirms no existing paper pre-empts it.
2. Does a deeper scan of the spectrum-sensing literature (beyond the Yücek–Arslan survey)
   surface a wandering-line / order-tracking detector we should adopt rather than
   reinvent? — the one live "could it subsume us" thread (§5).
3. Which distinguishability metric (TV, KL, Hellinger) gives the cleanest
   distortion→detectability bound for our observation model? — inherited from
   `method-soundness-and-prior-art.md` §6 Q4; a spike sub-question.
