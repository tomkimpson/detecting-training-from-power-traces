# tasks.md — certifying-training-from-power

Phased task tracker for Paper 2. Phasing follows
[`notes/plans/plan-for-paper-2.md`](notes/plans/plan-for-paper-2.md) §8. Mark tasks in-progress /
done and record a one-line **Result** under each as it completes. Rationale lives in
the plan docs; status lives here.

Legend: `[x]` done · `[~]` in progress · `[ ]` todo.

---

## Phase 0 — de-risking gates (COMPLETE)

Ported from the source monorepo (branch `feat/paper2-phase0`, merged). All
pre-registered criteria closed. Full record: `notes/results/st1-findings.md`, `handoff.md`.

- [x] **ST0 — estimand, threat model, scope (desk work).**
  **Result:** Four rungs fixed (1–2 spine, 3–4 conditional); admissible efficient-
  training family + attack budgets defined; full off-chip observation model stated;
  Ko/ours provenance corrected. Novelty audit skipped; ST3 not run; venue deferred.
- [x] **ST1 — adaptive structural detector null validity (make-or-break gate).**
  **Result: GO (honest reframe).** The complete tracker→resample→DG pipeline is
  level-safe on every stationary null; no covariance config calibrates the ladder
  end-to-end, so "analytic CFAR" is not claimable — reframed as an **effective score
  with per-trace surrogate calibration** (exact at 0.05/1e-2 on white/AR1 incl. fully-
  adaptive stage). Only the DG order family separates a wandering line from structural
  confusers; honest power boundary ≈ drift 0.8 Hz on 300 s.
- [x] **ST2 — de-periodicisation frontier (synthetic).**
  **Result: GO, definitive** (`results/st2/frontier_summary.json`, provisional=false).
  The ≈zero-cost work-variation attack (work=0.35/0.5) collapses every fixed test
  (≤0.27) while tracking holds 1.0. The 1 Hz integrating sampler erases all detectors
  (a channel requirement, not an attack cost).
- [x] **ST3 — optional challenge-synchronous pilot.** Not run (default scope: Rungs 3–4
  stay a conditional-protocol section).

### Phase 0 follow-ups (carried, not blocking)
- [x] **Slurm surrogate confirmation before any paper number freezes:** S=999,
  incl. stage 3 and stage-6 nulls (ar1_t, lognormal, sq_gauss). Slurm array over cells
  (`st1_far.py --calibs surrogate`, one cell per task) on MATS (preferred) or OzSTAR.
  **Result (2026-07-24): DONE — GO stands (confirmed-with-caveat).** MATS `compute`
  slurm array 5567 (24 cells, stages 1/3/4 × 8 nulls) at the surrogate sizing
  **M=2000 × S=999** (the `n_null_surrogate_cells` cap; the task's "M=10⁴" is the
  analytic-cell figure). Stationary Gaussian nulls exact at **all three stages**
  incl. fully-adaptive (white 0.051/0.009 fixed-α, 0.052/0.013 adaptive; ar1,
  ar2_resonant likewise; KS not rejected); linear non-Gaussian (ar1_t, lognormal)
  conservative. **One stage-6 exception:** `sq_gauss`/stage4 over-rejects at
  0.087/0.018 (≈1.7×, below the 3× NO-GO) — nonlinear null vs linear-stationary
  surrogate, the mild sibling of the controller inflation. Level-safety claim scoped
  to linear-stationary nulls; §4 footnote reworded (provisional footnote removed).
  Full record: `notes/results/number-freeze-2026-07-24.md`, `st1-findings.md`
  (2026-07-24 section). Harness: `scripts/slurm/st1_far_surrogate.sbatch`,
  `st1_far.py --list/--array-id`.
- [x] **Slurm re-verification of the ST2 sweeps in this repo** (same number-freeze
  pass): regenerate `results/st2/*_summary.json` from scratch, esp. the meter family
  behind the "integrating_1hz" claim **and the work-jitter line-band numbers, which
  ride the `B` component and are flagged for requalification** (`notes/results/issue54-investigation.md`).
  Which finding leads the frontier is left open until these freeze (positioning memo,
  2026-07-23). Do NOT run locally on the dev node (decision 2026-07-22) — slurm only;
  environment setup on the cluster is part of this task.
  **Result (2026-07-24): DONE — verdict reproduces exactly.** MATS `compute` slurm
  arrays 5565 (9 per-family sweeps, one/task via `scripts/slurm/st2_sweeps.sbatch`)
  + 5566 (frontier assembly via `scripts/slurm/st2_frontier.sbatch`, `--b2-dir
  data/measured_cost_anchors`). `frontier_summary.json`: `provisional=false`, **GO**,
  supporting `work=0.35/0.5`; `meter/integrating_1hz`=0.320 and `work` collapse
  identical to the committed claim (seed=0 deterministic; per-family diffs are
  last-digit AUC float churn). The `integrating_1hz` decomposition was already frozen
  by the meter-boundary sweep (job 5498, not re-run). The **B / work-jitter line-band**
  requalification lives in the measured b2 campaign (not this repo) — the synthetic
  `work` family reads only the clean `throughput_overhead` cost anchor, not the
  B-contaminated line-band detection arrays, so the frontier evidence is re-frozen; the
  B caveat is recorded, not resolved. **Which finding leads §6 now resolved
  (2026-07-24, branch `wire-in-phase2-results`): the ≈zero-cost work-variation
  attack leads the frontier; the meter-erasure point becomes the minimum-meter-spec
  subsection.** Harness: `scripts/slurm/st2_sweeps.sbatch`,
  `st2_frontier.sbatch`, `plot_st2_sweeps.py --list/--array-id`. Record:
  `notes/results/number-freeze-2026-07-24.md`.
- [x] **Settle the venue** (plan §10 Q1).
  **Result (2026-07-22): arXiv-first.** Write the strongest self-contained preprint;
  choose the submission venue after results freeze. Identifiability rigor at
  proposition level unless the eventual venue demands more.

### Strategic gates (resolve before Phase 1 numbers freeze)

From the 2026-07-23 positioning memos (`notes/discussion/north-star-and-positioning.md`,
`method-soundness-and-prior-art.md`). Sequenced: audit first, then the spike, then decide.
**Audit done (2026-07-23, `notes/discussion/st0-prior-art-audit.md`) — the spike is next.**

- [x] **ST0 prior-art audit (four fields).** Position the work against the four mature
  literatures the ST0 desk work skipped: **NILM** (non-intrusive load monitoring),
  **spectrum-sensing / LPI–LPD detection**, **power side-channel analysis** (DPA/CPA),
  and **covert-communication / steganography** (the warden game). Cheap desk work; it
  either hands us tools or surfaces the paper that already did this. This is the canonical
  home for the still-unowned "novelty audit" referenced in Phase 1; it is a §1 related-work
  prerequisite and gates the lower-bound spike below.
  **Result:** done (`notes/discussion/st0-prior-art-audit.md`). All four fields covered with
  verified citations (Hart 1992 + Zoha 2012 NILM; Yücek–Arslan 2009 spectrum sensing;
  Kocher 1999 / Brier 2004 / Chari 2002 side-channel; Bash–Goeckel–Towsley 2013 + Cachin
  2004 covert-comms — all need `check-refs` before arXiv). **Subsumption verdict: no field
  beats or subsumes the tracked-cyclostationary pipeline** — the novelty claim holds
  (mature detectors applied to a new question, new adversary, new channel); spectrum sensing
  is the one field to keep scanning for a reusable order-tracking method. **Lower-bound-spike
  readiness: proceed** — covert-comms hands over the KL/TV machinery (Cachin relative-entropy
  security; B–G–T square-root law) but *no* existing paper derives the hiding-cost bound for
  a physics-constrained training schedule over a filtered low-rate meter, so the spike is not
  pre-empted. Recommends an own `\section{Related work}` organised by the four fields and
  lists 8 bib keys for the §1 / `references.bib` follow-up.
- [x] **Lower-bound feasibility spike (the kill/continue gate).** Time-boxed: is a
  TV/KL (covertness) lower bound on hiding cost derivable for even one attack family
  (e.g. i.i.d. phase jitter)? Anchor it in the physics (hiding synchronous all-reduce is
  communication-bound and costly), with the TV/KL machinery as the formal wrapper — a
  purely generator-internal bound is still conditional on the generator. **Outcome is a
  hard fork on claim strength:** bound exists ⇒ contribution, frame as "must pay ≥ X";
  no bound ⇒ position paper, frame as "evidence + upper bound on hideability, necessary
  conditions only." Feeds the Phase 4 identifiability section (plan §10 Q2).
  **Result: bound derivable ⇒ CONTRIBUTION side of the fork**
  (`notes/discussion/lower-bound-feasibility-spike.md`). Derivation for i.i.d. phase
  jitter: coherent line power rolls off as `1/(1+κσ²)`, `κ≈πf₀T` (phase diffusion
  `D≈(2π)²σ²f₀`); `detector power ≤ TV`, Pinsker → a finite covertness threshold
  `σ*(ε)`; the physics anchor (synchronous comm barrier `T_down` fixed under work
  variation) makes reaching `σ*(ε)` cost **throughput OR learning efficiency** — the
  free-work-jitter escape does not evade it. Sanity check `scripts/lower_bound_spike.py`
  → `figures/lower_bound_spike.*`, `results/spike/lower_bound_spike.json`: coherent-power
  rolloff form R²=0.97 (fitted κ≈2950, within ~3× of πf₀T≈940); work-jitter escape
  measurably MORE detectable at matched σ (confirms the comm-barrier residual). Caveats
  set claim wording: bound proved vs a FIXED verifier (tracking/optimal is the Phase-4
  analytic step); "cost" is throughput-or-learning, learning leg stated not measured;
  generator-internal. Frame Phase-4 identifiability (plan §10 Q2) as a proposition-level
  covertness cost bound. Open: metric choice (TV/KL/Hellinger) for the tightest form.

---

## Phase 1 — Rung 1 full (IN PROGRESS — prose written; number-freeze pending slurm)

- [x] Qualified multitaper F-test (benign) + tracked-cyclostationary (adaptive) on the
  scenario model, both adversary settings.
  **Result:** `paper/main.tex` §4 written — `subsec:benign` (Thomson multitaper F-test,
  approx-pivotal caveat, search correction, analytic Bonferroni anti-conservative ~3.6–6×
  ⇒ ranking-only, matched-filter comparator, foreshadows mtf's zero wander-power) and
  `subsec:tracked` (three-component pipeline; fixed-α χ²₂ₗ scoped; open validity issues;
  explicit non-CFAR disclaimer). §4 intro states the Rung-1 wording discipline
  (evidence-not-identification; controller counterexample).
- [x] Null-validity validation section wording from `notes/results/st1-findings.md`.
  **Result:** `subsec:st1` written — staged design, level-safety everywhere (no stage-5/6
  inflation), the three deltas (resampling ~3.4× deflation / estimated-warp nil /
  path-selection no inflation), covariance decision axis (b=54 repairs stage 4 but breaks
  stage-1 anchor ⇒ no end-to-end analytic calibration), surrogate calibration exact on
  tested nulls, controller as attribution boundary, honest reframe. Surrogate numbers
  carry a `\footnote` marking them provisional (partial M=1000/S=199 grid) pending the
  slurm S=999/M=10⁴ confirmation.
- [x] Pre-registered detector bake-off → paper §4 / Appendix A.
  **Result:** `app:bakeoff` written with `tab:bakeoff` (TPR@1e-2 by detector × drift vs
  inference and mixed-structural negatives, from `bakeoff_summary.json`) + hard-case and
  power-boundary prose; the three findings (tracking necessary; only DG order family
  separates the confuser; semi-coherent dominates).
- [x] Wire the ST1 figures (calibration, bake-off, power) into `paper/main.tex`.
  **Result:** `st1_calibration_far` + `st1_calibration_cov` wired into `subsec:st1`
  (`fig:st1_far`, `fig:st1_cov`); `st1_bakeoff` into Appendix A (`fig:st1_bakeoff`).
  Section-4 + App-A checklists removed for the written items. 3 references added to
  `references.bib` (Thomson 1982, Percival & Walden 1993, Dandawaté–Giannakis 1994,
  Bonnardot 2005 tacholess) — flagged for `check-refs` before arXiv push.

**Still open before Phase 1 numbers freeze / merge:** ~~(i) slurm S=999 surrogate
confirmation~~ **DONE 2026-07-24** (Phase-0 follow-up above; provisional footnote removed,
level-safety claim scoped to linear-stationary nulls after the `sq_gauss`/stage4 sub-3×
exception surfaced); ~~(iii) the ST0 prior-art audit (four fields)~~ **DONE 2026-07-23**
(Strategic gates above); (ii) `check-refs` / `check-arxiv-llm-compliance` over the new
prose + the now **12** new bib entries (4 Rung-1 method + 8 ST0 related-work) remains.
PDF not built locally
(no TeX toolchain on the dev node) — verified statically (figures on disk, cite keys
resolve, labels/refs consistent, environments balanced).

- [x] **ST0 related-work section written** (follow-up to the prior-art audit).
  **Result:** `\section{Related work}` (`sec:related`) added to `paper/main.tex` between
  §1 and the threat-model section — four `\paragraph`s (NILM; spectrum sensing / LPI–LPD;
  power side-channel; covert-comms / steganography) framing the contribution as *the
  question, the adversary, and the channel, not the detectors*, and cross-linking the
  Rung-1 bake-off (`sec:structural`/`app:bakeoff`), frontier (`sec:frontier`), and
  identifiability (`sec:threat`). 8 verified bib entries added to `references.bib`
  (`hart1992nilm`, `zoha2012nilm`, `yucek2009spectrum`, `kocher1999dpa`, `brier2004cpa`,
  `chari2002template`, `bash2013covert`, `cachin2004steganography`) — flagged for
  `check-refs` before arXiv. §1 Related-work checklist item removed. Verified statically
  (all cite keys + crefs resolve, checklist envs balanced 9/9); PDF not built (no TeX
  toolchain on the dev node).

## Phase 2 — Frontier (full) + Rung 2

- [x] Frontier at full resolution with the systems-cost anchors. **Learning-efficiency
  metric descoped (decision 2026-07-22):** no GPU measurement campaign — keeps the
  paper theory/methods and avoids the single-GPU/NVML transport trap. The frontier's
  utility axis is the measured systems-cost anchors only; the learning-efficiency
  penalty is stated in the paper as the open empirical question / future work
  (paper §5 checklist already words it this way).
  **Result: frozen on slurm** (`results/st2/frontier_summary.json`, provisional=false,
  n_each=200, 50 cells, full 5-detector set; slurm array **5635** on MATS `compute`,
  all 9 families COMPLETED; full record `notes/results/st2-frontier-freeze-findings.md`).
  Built the missing slurm harness — `plot_st2_sweeps.py --array-id` (one family per
  task, disjoint per-family summaries, no race) via `scripts/slurm/st2_sweeps.sbatch`
  (array) + `scripts/slurm/st2_frontier.sbatch` (assembly)
  + smoke-path split so `--smoke` can't clobber the tracked full-res files. **Verdict
  GO (definitive):** the ≈zero-cost real-work-variation attack `work=0.35/0.5` collapses
  every fixed test (≤0.5) while tracking holds (≥0.8) — the genuine scope boundary.
  New numbers are **byte-identical to the prior local run** (max |AUC/TPR Δ| = 0.0000;
  only `phase_diffusion_D` wobbles at the last ULP), confirming determinism. **Cost-anchor
  provenance fixed** (`data/measured_cost_anchors/` everywhere; `results/b2/` never
  existed). This also closes the **ST2 leg** of the Phase-0 re-verification follow-up
  above — the FAR/ST1-surrogate leg stays open (Phase 1). issue-54: the frontier reads
  only the clean `throughput_overhead` anchor field, not the B-contaminated line-band
  detection arrays, so it is unaffected. **WIRED into `paper/main.tex` §6
  (`sec:frontier`; `fig:frontier`+`fig:st2_work`) on branch `wire-in-phase2-results`
  (2026-07-24): body led by the work-variation attack; issue-54 non-impact noted.**
- [x] **Meter-requirement boundary sweep** (supports promoting the ST2 channel-
  requirement finding to a named "minimum meter specification" contribution —
  pending spec.md approval): grid over `sample_hz` × `integ_window_s` between the
  nominal 20 Hz channel and the 1 Hz integrating sampler, plus notch depth/Q, honest
  training only; locate where each detector class dies. CPU-only synthetic; run as a
  slurm array.
  **Result: done** (`results/st2/meter_boundary_summary.json`, 48 cells, n_each=200;
  slurm array 5498 on MATS `compute`, all cells COMPLETED; full record
  `notes/results/st2-meter-boundary-findings.md`). **Minimum meter specification:**
  the tracking class (Viterbi, spectral) holds TPR@0.05 ≳ 0.9 iff sample rate
  ≥ ~2 Hz **and** integration window ≲ 0.5 s **and** no deep in-band notch at the
  cadence centre. fs=1 Hz collapses it (Nyquist crosses the band); fs=0.5 Hz is
  totally dead; a 1 s boxcar sinc-nulls the 1 Hz band centre even at 20 Hz sampling.
  Decomposes the committed `integrating_1hz` two-point result: sampling **or**
  integration alone already kills tracking (over-determined). Figures:
  `figures/st2_meter_boundary.*` (per-detector heatmaps + death contour) +
  `figures/st2_meter_boundary_notch.*`. Harness: `powerladder/typeb/meter_boundary.py`,
  `scripts/st2_meter_boundary.py`, `scripts/slurm/st2_meter_boundary.sbatch`.
  **WIRED into `paper/main.tex` §6 subsection `subsec:meter_spec`
  (`fig:meter_boundary`+`fig:meter_boundary_notch`) on branch `wire-in-phase2-results`
  (2026-07-24): framed as a minimum meter specification / channel requirement
  (§6 results subsection, NOT a named contribution — spec.md untouched per decision).**
- [x] **(Optional, non-blocking) NP-optimal LRT ceiling in the bake-off.** We have the
  generator, so compute/approximate the Neyman–Pearson optimal detector between the
  training and null generators and report the corpus-free surrogate detector as a
  fraction of it ("X% of NP-optimal power at Y% of the information cost"). Quantifies
  "is our detector good" — not a pre-gate (decision 2026-07-23).
  **Result (frozen on slurm):** method = **Whittle spectral LRT** (user decision
  2026-07-24 — the light option; no closed-form likelihood exists).
  `powerladder/typeb/np_ceiling.py` (f₀-marginalised Bayes–Whittle LRT: per-f₀ MC training
  template bank, drift-pooled, `logsumexp_k ℓ_tr(x|f0_k) − ℓ_neg(x)`). Driver
  `scripts/st1_np_ceiling.py` reproduces the **exact** bake-off eval populations (parity
  guard **max |TPR Δ vs bakeoff| = 0**), fits ceilings for the inference / structural /
  controller-only nulls, scores ceiling + all corpus-free detectors, reports
  `rho_auc`/`rho_tpr`. Plot `scripts/plot_st1_np_ceiling.py`, sbatch (single task ~9 min),
  tests `tests/test_np_ceiling.py` (5, pass; suite 5 BLAS-baseline fail / 186 pass).
  Config `NpCeilingParams`. **Frozen (slurm job 5646, MATS `compute`, n_each=200, f0_n=61,
  n_mc=2000; `results/st1/np_ceiling_summary.json`, `figures/st1_np_ceiling.*`):** the
  Whittle ceiling is **AUC=TPR=1.0 in every column at every drift** (the two generators
  are perfectly separable in principle), so all difficulty is in the detector. **vs the
  inference null the Viterbi tracker is essentially NP-optimal** (`rho_auc=1.00` across the
  whole wander axis); the fixed matched filter collapses (ρ 1.0→0.78→0.38→−0.1). **vs
  structural confusers / the hard controller case only the DG order family approaches the
  ceiling** (dg_order_full ρ 0.93→…→0.10; Viterbi and spectral score AUC 0.0 on the hard
  case), with a widening gap at high drift = the honest headroom above today's detectors.
  **Caveat:** NP-optimal under the Whittle model only (a lower bound on the true optimum),
  so reporting fractions against it is conservative. Full record:
  `notes/results/st1-np-ceiling-findings.md`. **WIRED into `paper/main.tex`
  App A `app:bakeoff` (new paragraph + `fig:np_ceiling`) + the Related-work
  optimality-ceiling hook (`\cref{app:bakeoff}`) + a §6 sentence, on branch
  `wire-in-phase2-results` (2026-07-24).**
- [x] Rung 2: stated inference null; three decision rules; semantic falsification
  controls (periodic inference, gradient-only, discarded-update decoy, training-shaped
  non-ML loop, controller cycle, async training, co-resident mixtures); transfer / domain
  shift evaluated.
  **Result (harness built + n=200 numbers frozen on slurm):** full pipeline on
  `feat/phase2-rung2`. `powerladder/typeb/rung2_features.py` (8-feature physics
  vector, **estimated-from-trace only** — mtf/comb/spectral/viterbi/dg_fixed/
  dg_order_full + estimated path-stability & phase-fold repeatability from the
  Viterbi path); `rung2.py` (three rules — prespecified physics score with the
  fixed-cadence comparator pre-registered out, fitted logistic discriminant OOF,
  RF learned reference; AUC + FPR/FNR; transfer via fit-nominal/zero-shot-predict);
  `rung2_scenarios.py` (seven controls, `is_training`/`expect` annotations). Driver
  `scripts/rung2_eval.py` (crc-seeded 14-cell grid = stated ∪ 6 transfer ∪ 7
  controls, flock-merged `results/rung2/rung2_summary.json`), slurm
  `scripts/slurm/rung2_eval.sbatch`, plotter `scripts/plot_rung2.py`
  (`figures/rung2_stated_transfer`, `rung2_controls`). Tests `tests/test_rung2.py`
  (10, pass; suite 5 fail/181 pass = known BLAS baseline + 10).
  **Frozen findings (n_each=200, 300 s, band 0.3–1.7, FAR 0.05; slurm array 5612
  on MATS `compute`, all 14 cells COMPLETED ~100 s each):**
  (1) STATED — all three rules separate training from the hard inference null
  perfectly (AUC 1.00, FNR 0 at FPR 0.05). (2) TRANSFER — AUC stays 1.00 across
  every shift (f_peak lo/hi, short duration, band shift, hostile controller meter,
  heavy coloured noise): robust off the stated population. (3) CONTROLS — the five
  NON-training loads (discarded-update decoy, gradient-only, non-ML loop,
  controller cycle, periodic inference) all score AS training (frac 1.00 all
  rules): **the meter certifies the physical schedule, not training semantics.**
  Co-resident (diluted genuine) training caught (1.00). **Divergence from the
  pre-registration:** async (de-periodicised) genuine training was expected to be
  MISSED, but the tracked physics rule still catches it (AUC 0.89, frac 0.72; RF
  1.00) — the fitted LINEAR rule is the weakest (0.75/0.55). Consistent with the
  ST1/ST2 tracker-robustness story: drift+phase-slip de-periodicisation does NOT
  evade tracking at these levels; the genuine scope limit is the ≈zero-cost
  work-variation attack characterised on the ST2 frontier, not this async control.
  **Async decision RESOLVED (2026-07-24): report as a robustness result** (tracked
  rule catches it; genuine boundary cited from the §6 work-variation frontier; no
  new sweep). **WIRED into `paper/main.tex` §7 `sec:classification`
  (`fig:rung2_stated`+`fig:rung2_controls`+`tab:rung2_controls`) on branch
  `wire-in-phase2-results`.**

## Phase 3 — Rungs 3–4 conditional protocol

- [x] Concise challenge-synchronous protocol; assumed transcript primitive as an ideal
  functionality; randomisation-test discipline; ceilings stated. (Full empirical
  treatment only if ST3 is later run and passes.)
  **Result:** prose-only scope honoured (ST3 stays unrun). `paper/main.tex` §7
  (`sec:active`, "From passive evidence to active authentication") written — 5
  paragraphs: (1) why go active (Rung-2 semantic-decoy ceiling, `\cref{sec:classification}`);
  (2) the Rung-3 challenge-synchronous lock-in protocol (balanced $c_k$, committed
  semantically-equivalent schedule variation, causal-lag correlation, re-randomisation
  null, challenge-aware-dummy ceiling); (3) randomisation-test discipline (exact
  conditional on the design; search repeated in every replicate; carryover/adaptivity/
  balance in the design; the $\sqrt N$-processing-gain caveat); (4) the Rung-4 ideal
  functionality (commitment/unpredictability/sound binding/freshness/co-location; not
  built); (5) the two ceilings + co-location gap + future-work fold of the "if ST3 runs"
  list, cross-linked to `sec:threat`/`sec:frontier`/`sec:discussion`. Checklist + `\todo`
  removed (convention). **Novelty gap closed:** new `\paragraph{Challenge-response
  authentication}` added to `sec:related` (distance-bounding, PUFs, active telemetry).
  **5 new bib entries** (`brands1994distance`, `pappu2002puf`, `canetti2001uc`,
  `jia2021proofoflearning`, `fisher1935design`) + reuse of `monfared2026timing` — all
  flagged for `check-refs` before arXiv. Verified statically (checklist envs 8/8,
  `\todo` 8, all cite keys + crefs resolve); PDF not built (no TeX toolchain on the dev
  node).

## Phase 4 — Write-up

- [x] Negative transport case (single-A100) as a result; identifiability theory section
  (rigor level per plan §10 Q2); discussion closing out each rung's ceiling.
  **Done in two passes.** (a) 2026-07-24, branch `wire-in-phase2-results`: §9
  `sec:reality` (negative transport case + falsifiable predictions, from the frozen ~0.4%
  ripple / 0.31–0.45 Hz limit-cycle / ~75× gap numbers; no measured figure — captures not
  in this repo) and §10 `sec:discussion` (ladder close-out `tab:closeout` +
  scope/limits/governance). (b) 2026-07-24, branch `phase-4`: the **identifiability
  theory** at proposition level (plan §10 Q2) — new §3 `subsec:identifiability` (2
  definitions + `def:covert`, `prop:noident` no-passive-semantic-identification from the
  `K_h(w)` observational-equivalence formalism, `prop:rolloff`, `prop:sigmastar`,
  `prop:cost`, + `rem:direction`/`rem:boundscope`) with proofs, the numerical check
  (`fig:lower_bound`, previously orphaned) and the σ*→cost bridge in App B
  `app:identifiability`. §3 checklist + `\todo` removed; `tab:closeout` gained a
  cost-of-hiding row; the three "we leave the bound open" passages (§2/§8/§10) reworded.
  **Two defects in the spike's derivation found and fixed** (recorded as a dated
  correction in `notes/discussion/lower-bound-feasibility-spike.md` §7): (i) the Pinsker
  chain proved the *converse* (sufficient, not necessary, distortion) — the bound now runs
  through an explicit verifier's achieved advantage; (ii) `2·AUC−1` is a Gini index that
  can exceed the best threshold test's advantage, so `scripts/lower_bound_spike.py` now
  computes **Youden J** and inverts *that* rolloff (κ_J=34, R²=0.95; σ*(0.5)=0.17,
  σ*(0.2)=0.34 both now inside the swept range; AUC/Gini retained in the summary, every
  previously committed number unchanged). Also corrected: σ* ∝ ε^(−1/2), not 1/ε.
  2 new bib entries (`tsybakov2009nonparametric`, `cover2006elements`) — flagged for
  `check-refs`.
- [x] **Cost-vs-hiding Pareto re-plot (cheap, CPU-only, no new sweeps).** Render the
  already-frozen ST2 frontier as an explicit **adversary-cost vs hiding** Pareto curve —
  the plan's stated deliverable (`notes/plans/plan-for-paper-2.md:370`,
  `plan-for-paper-2-review.md:198`). The (hiding, systems-cost) pairs already exist in
  `results/st2/frontier_summary.json`: per cell, detection loss (best-tracking &
  best-fixed TPR@0.05, or 1−TPR) on one axis and the measured `cost_overhead_pct` on the
  other, for the four anchored families only (jitter, drift, work, shape; others carry
  `null`). A new plot script reading the tracked summary (no re-run) — one Pareto panel
  per detector class so the "how much systems cost buys how much hiding from the tracker
  vs the fixed detectors" gap is visible, with the ≈zero-cost work-variation point called
  out. **Out of scope / still future work:** the learning-efficiency cost leg (descoped
  2026-07-22, needs a GPU campaign — the transport trap) stays the stated open empirical
  question; this item plots only the measured systems-cost axis.
  **Result (2026-07-24, branch `phase-4`): done.** `scripts/plot_st2_pareto.py` — a
  **pure reader** of the tracked `frontier_summary.json` (never writes it; no `--smoke`
  path that could corrupt it), reusing `plotstyle` + the summary's own `detector_classes`
  field cross-checked against `powerladder/typeb/meter_boundary.TRACKING/FIXED`.
  `figures/st2_cost_pareto.*`: two panels (hiding = 1−TPR@0.05 vs measured overhead, one
  per detector class) with the Pareto staircase. **The asymmetry is the headline:** against
  the fixed class the staircase reaches hiding **0.74 at ≈0% cost** (work=0.35/0.5) and
  barely improves thereafter; against the tracking class it never leaves the floor,
  peaking at **0.16 at 159%** (jitter=0.35). Only **16 of 50** cells carry a measured
  anchor (jitter 5, work 5, shape 5, drift 1); the 34 unpriced — incl. **work=0.7, where
  the tracker itself bends** — are drawn in a hatched "cost not measured" strip and counted
  on stdout rather than dropped. Tests `tests/test_st2_pareto.py` (7, pass; suite 5 known
  BLAS-digest fail / 194 pass). **WIRED into §6 `sec:frontier`** as `fig:st2_cost_pareto`
  + a "The trade-off itself" paragraph; README repro line added.
- [ ] **Finish the remaining manuscript scaffolds** (the last write-up block; surfaced
  2026-07-24 when the two items above closed). After this session only **2 of the
  original 6** section checklists remain, but they are load-bearing:
  - **Abstract** — still placeholder prose + `\todo` (`main.tex:109`).
  - **§1 `sec:intro`** — has the governance-motivation paragraph and a drafted
    `tab:ladder`, but no thesis paragraph, no **contributions list**, no scope /
    target-population / why-theory-not-hardware paragraphs (`main.tex:129`, `:177`).
    **Decision needed:** whether the covertness cost bound
    (`subsec:identifiability`) is named as a contribution here. It is currently a §3/App-B
    result only; promoting it would be a **scope change requiring `spec.md` approval**,
    whose "Methods at a glance" names the frontier as the central result and lists no
    identifiability theory (same precedent as the minimum meter spec, 2026-07-24).
  - **§4 `sec:scenario`** — a **pure scaffold, zero body prose** (`main.tex:513`, `:543`):
    four-layer Ko/ours provenance, the inference-null spec, the transfer family, and
    **two figures that do not exist yet** (example generator traces + spectra; an
    observation-map sensitivity sweep) ⇒ two new plot scripts. Cheapest remaining
    experimental work in the repo, CPU-only. Note §3's `def:family` and
    `subsec:identifiability` already forward-`\cref` this section.
- [ ] **`check-refs` + `check-arxiv-llm-compliance` over the whole manuscript** (owed and
  accumulating): 12 Rung-1/related-work entries, 5 Rung-3/4 entries, and 2 new
  identifiability entries (`tsybakov2009nonparametric`, `cover2006elements`) have never
  been verified; the compliance pass is owed over all prose written 2026-07-24 (§6/§7/§9/§10
  wiring, §3, App B). Both are pre-arXiv blockers. **No PDF has ever been built** — there
  is no TeX toolchain on the dev node, so every section has been verified statically only
  (env balance, `\cref`/cite-key resolution, figure stems on disk). A real `latexmk` build
  is required before submission.

---

## Housekeeping / tech-debt (low priority)

- [ ] Trim `powerladder/config.py` to paper-2 params (`B2Params`/`BenchConfig`/beta
  params are dead weight carried over from the monorepo).
- [ ] Optionally make `powerladder` pip-installable (`[project]` + setuptools) and drop
  the `sys.path.insert` shims in `scripts/`.
