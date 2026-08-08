# Handoff — 2026-08-07 (research talk drafted and rendered)

## Current status

- Branch: `feat/research-talk`.
- Added a 21-slide main research talk plus 6 appendix slides in
  `talk/slides.md`, with presenter notes embedded as Markdown comments.
- Rendered deliverables: `talk/slides.pdf` (27 pages, 16:9, 3.2 MB) and
  `talk/slides.html`; reproduction commands are in `talk/README.md` and pin Marp CLI
  4.5.0.
- Visually inspected all 27 pages as rasterized contact sheets, then checked the dense
  Pareto, meter-boundary, semantic-control, and lower-bound slides at higher resolution.
  No content is clipped in the final render.
- No Python or paper source changed, so the test suite and paper build were not rerun.

## Talk spine and decisions

The talk is deliberately not a paper-section summary. Its sequence is:

1. governance inverse problem — one external scalar trace;
2. the synchronous-training cadence hypothesis;
3. the historical failure mode — Viterbi can follow a real but wrong controller line;
4. the deeper identifiability limit — even the right schedule does not establish
   training semantics;
5. the four-rung claim ladder;
6. the Rung-1 order-tracking pipeline and the two distinct null/calibration problems;
7. the de-periodicisation cost frontier as the central empirical result;
8. the meter boundary and Rung-2 semantic falsification controls;
9. why stronger Rung-3/4 claims need an active challenge and transcript binding;
10. the single-A100 negative transport case and an explicit established/open ledger.

One qualification is now in the main talk rather than hidden in notes: the ST2
“tracking class” curve is a diagnostic per-cell envelope. Raw Viterbi is robust to
variable-real-work attacks but fails against controller-line structural confusers; the
order-domain rules reject those confusers but degrade earlier under variable work. The
project has not yet demonstrated one jointly calibrated deployed decision policy that
inherits both strengths across the full frontier. A verifier may run any algorithms it
wants, but the final combined policy (including algorithm selection) must be calibrated
and evaluated as a whole.

## Next steps

1. Rehearse once and trim to the actual time slot; the current main deck is suited to
   roughly 20 minutes if the technical figures are narrated selectively.
2. Use Appendix slide 22 (“The paper spine implied by the talk”) when restructuring the
   manuscript: identifiability → claim ladder → frontier → ceilings, with detector
   construction supporting rather than driving the narrative.
3. Decide whether to add the missing joint-rule experiment before tightening the paper,
   or scope the frontier language consistently as a detector-class diagnostic.
4. Highest-value empirical follow-up remains an external meter on synchronized
   multi-GPU training; the measured single-A100 trace is a negative transport case.

## Key files and gotchas

- Source: `talk/slides.md`; rendered deck: `talk/slides.pdf`; browser version:
  `talk/slides.html`; build instructions: `talk/README.md`.
- The deck links directly to generated assets under `figures/`; do not copy or hand-edit
  them under `talk/`.
- Rendering needs `--html --allow-local-files` because the deck uses HTML layout and
  repository-local figures.
- All positive detection results are labelled synthetic/scenario-based. Keep those
  labels and the 75× modulation-gap reality-check slide when shortening the talk.

---

# Previous handoff — 2026-07-24 (Phase 4 closed, then hardened by /check-PR)

## What happened this session

Closed **both remaining Phase-4 items** on branch **`phase-4`** (pushed; PR #16 open).
Note the request's framing was slightly off: the negative transport case (§9) and the
per-rung ceiling discussion (§10) were already written and merged on 2026-07-24 (`aff8f7b`).
What actually remained was the identifiability theory section, plus the Pareto re-plot.

### 1. Identifiability theory — proposition level (plan §10 Q2)

New §3 `subsec:identifiability` + a fully written App B `app:identifiability`. The
theorem environments had been sitting unused in the preamble since the scaffold; they are
now used. Contents: `def:obseq` (observational equivalence $K_{h_0}(w_0)=K_{h_1}(w_1)$),
`def:family`, `def:covert`, `prop:noident` (no passive semantic identification — the formal
counterpart of the Rung-2 decoy result), `prop:rolloff`, `prop:sigmastar`, `ass:cost`,
plus `rem:direction`, `rem:costgap` and `rem:boundscope`. Proofs, the numerical check, the σ*→cost bridge
and the coverage caveat are in App B.

**Two real defects in the spike's derivation were found and fixed** (dated correction
appended as `notes/discussion/lower-bound-feasibility-spike.md` §7 — the note's §3 is left
as the historical record):

- **The Pinsker chain proved the converse.** `TV ≤ √(KL/2)` bounds TV from *above*, so
  driving it below ε shows some distortion **suffices** for covertness, not that any is
  **necessary** — the opposite of what a cost-of-hiding claim needs. The bound now runs
  through an **explicit** verifier: a single-trace test's achieved advantage `A ≤ TV`, so
  `A(σ) > ε ⟹ not ε-covert`. Pinsker is now a remark on what is *not* proved.
- **`2·AUC−1` is not an achieved single-trace advantage.** It is a Mann–Whitney/Gini index
  and can exceed the best threshold test's advantage. `scripts/lower_bound_spike.py` now
  also computes **Youden J** (`max(TPR−FPR)`, the *one-sided* two-sample KS distance) and
  inverts *that* rolloff. Re-run locally — legitimate, it is the designated lightweight CPU
  check (seeded; ~100 s at the final n=600), not one of the slurm-only grids.
- **Also corrected: σ\* ∝ ε^(−1/2), not 1/ε** (the note's 1/ε followed from its alternative
  formula with ε² under the root). Confirmed against the frozen thresholds: a 10× drop in ε
  moves σ\* by 3.32× ≈ √10.

**The numbers from this first pass are SUPERSEDED** by the n=600 re-derivation forced by
the review — see the next section for the figures now in the paper. `figures/lower_bound_spike.*`
— tracked but cited nowhere before — is now `fig:lower_bound` in App B, and
`covertness_thresholds_gini` retains the old Gini-based thresholds for continuity with the
deliberation note.

### 2. Cost-vs-hiding Pareto plot

`scripts/plot_st2_pareto.py` → `figures/st2_cost_pareto.*`, wired into §6 as
`fig:st2_cost_pareto` with a "The trade-off itself" paragraph. A **pure reader** of the
tracked `frontier_summary.json`; no `--smoke` path exists, so there is nothing that can
corrupt the frozen artefact (the failure mode the meter-boundary PR review caught).
Two panels, hiding = 1−TPR@0.05 vs measured overhead, one per detector class, with the
Pareto staircase.

**The asymmetry is the paper's central claim in one figure:** against the *fixed* class the
staircase reaches hiding **0.74 at ≈0% cost** and barely improves however much the prover
spends; against the *tracking* class it never leaves the floor, peaking at **0.16 at 159%**.
Only **16 of 50** cells have a measured anchor; the 34 unpriced — including **work=0.7,
where the tracker itself bends** — are drawn in a hatched "cost not measured" strip and
counted on stdout rather than dropped.

### 3. Ripple edits and housekeeping

- The **three "we leave the bound open" passages** (§2 related work, §8, §10 limitations)
  reworded to "derived for one family against a fixed verifier; general case open" —
  they directly contradicted the new propositions.
- `tab:closeout` gained a **cost-of-hiding row**; Rung 2's ceiling now cites `prop:noident`.
- §6 governance reading and the §9 falsifiable-predictions cref retargeted at `ass:cost`.
- 2 new bib entries, **DOIs verified against the resolver** (`tsybakov2009nonparametric`
  10.1007/b13794; `cover2006elements` 10.1002/047174882X — the resolver reports 2005 for
  the online-first record, the entry keeps 2006 for the print 2nd edition).
- README: repro lines for `plot_st2_pareto.py` and `lower_bound_spike.py` (the latter was
  missing entirely).
- Deleted the stray root-level `PR_REVIEW_feat-phase2-meter-boundary_2026-07-23.md`.

## Then: /check-PR, and the fixes it forced

Ran the pre-merge review over the branch (6 review agents + 2 adversarial verifiers).
Verdict was **DO NOT MERGE**, and it was right. All findings are now fixed; the report is
was untracked and `PR_REVIEW_*.md` is now gitignored, so the record lives in this file,
`tasks.md` and `log.md` rather than in a report committed to the repo.

**Two blocking findings — both internal contradictions in claims stated as proved:**

1. `prop:cost`'s mechanism was refuted by our own §6. It asserted a residual marker
   "survives every reallocation of work", but §6 measures the fixed tests at AUC 0.44 —
   *below chance* — under variable real work, so the residual is recoverable only by the
   tracking class, which `rem:direction` explicitly excludes from the bound. Its
   "period jitter has only two realisations" premise also fails against our own
   additive-burial and dilution families. It is now **`ass:cost`**, an assumption, with
   **`rem:costgap`** stating both gaps plainly.
2. `tab:closeout`'s "no passive detector can do better" contradicted App A, which shows
   `dg_order_full` at 0.93 of the NP ceiling on the controller case and calls the gap
   "honest headroom above today's detectors, not a limit of the channel". Now scoped to
   the discarded-update decoy, the one control for which `prop:noident` actually argues.

**A factor-of-π error that made the paper stronger.** Carrying the Lorentzian/bin algebra
through gives the parameter-free **κ = π²f₀T = 2961**, which the fit matches to **4%**
(3068; 2812, i.e. 5%, restricted to the faithful σ ≤ 0.1 regime).
The paper had claimed only "within a factor of about three" of πf₀T = 942 and called the
prefactor irreducible. The derivation pins it; `eq:arctan`/`eq:kappa` now show the work.

**σ\* re-derived at n=600** (was 80, ~30 s). At n=80 the H0 floor of J is 0.25 (against
0.093 at n=600; ~0.09 is the *mean* of the n=80 null, not the floor), so a
*null* point (AUC 0.4967 — the script's own mask already excluded it from the Gini fit)
carried 78% of the fit leverage; κ_J varied 4× across seeds and P(σ\*(0.2) > 0.35) = 0.61,
i.e. the published price bracket was more likely wrong than right. Now masked on the
n-dependent floor with bootstrap CIs: **κ_J = 32 [24, 40]**, **σ\*(0.5) = 0.18
[0.16, 0.20]**, **σ\*(0.2) = 0.36 [0.32, 0.41]**. σ\*(0.2) moved up a cost bracket, so the
paper now states the price as an order of magnitude — "tens of percent", "of order 100% or
more" — rather than two-anchor brackets. The artefact computes `sigma_faithful_max` and
flags thresholds beyond it, and reports a faithful-only refit (κ = 22) as the sensitivity.

**Also fixed:** `rem:direction` had it backwards — since J ≤ TV the bound is valid against
any class *containing* the exhibited test and merely *loose* against the tracker, so the
old wording undersold it (§2/§6/§10 all corrected); ε^(−1/2) moved out of the proposition
(it is fit extrapolation); one-sided vs two-sided KS wording; R² now computed over fitted
points only; the "work is more detectable at matched σ" claim now holds at every σ ≥ 0.05
(see the 2026-07-27 arms fix below); the §6 staircase maximum (0.82) stated;
`edition = {Second}`.

**Engineering:** `_youden` had zero test coverage while supplying every published
threshold — new `tests/test_lower_bound_spike.py`, 14 tests including a scipy KS oracle and
a polarity guard. Figure PDFs were carrying a wall-clock `/CreationDate`, so
"every figure is regenerable" could not be checked by regenerate-and-diff —
`plotstyle.save` now suppresses it and figures are byte-reproducible. README's ST2 block
read as three safe local commands, two of which overwrite the slurm freeze. Agent E's
simplifications applied (dead `--stem`, loop-carried `n_unpriced`, dead `label=`, hoisted
set, `_invert` → module-level `_covertness_thresholds`, shared `FAMILY_COLOR` in
`plotstyle`, vacuous test assertion). `log.md` back-filled.

## 2026-07-27: second /check-PR pass over `b6d9eb8`, and the arms fix

Re-reviewed the fix commit itself (6 agents + 10 adversarial verifiers). **No CRITICAL
survived**: the one raised (`prop:sigmastar` monotonicity) was downgraded because the
commit had *replaced a bogus derivation with an honest disclaimer*, and three separate
verifiers found b6d9eb8 removed defective reasoning rather than adding it. Three MAJORs
were refuted outright (bootstrap-untested; σ\*-bias-direction; bootstrap-CI-mixture).

**The one real scientific defect — the "matched distortion" comparison was not matched.**
Found independently by two agents. `_train_scores(work=True)` passed `KO` unmodified, so
`KoWorkloadParams.sigma_jitter` kept its **0.1** default *and* added `work_sigma`, while
the jitter arm set `sigma_jitter=σ` with no work variation. Signature: J_work(σ=0) = 0.862
≈ J_jitter(σ=0.1) = 0.885. This is what produced the sub-0.1 "work is less detectable"
reversals — a parameterisation artefact reported as a physical finding.

**Fixed and re-run** (work arm now zeroes `sigma_jitter` before applying `work_sigma`).
The result is *cleaner*: work ≥ jitter at every σ ≥ 0.05, an exact tie at σ = 0, and the
sub-0.05 gaps ≤ 0.005 (inside sampling error). Superseding numbers, all inside the old
CIs — **κ_J = 32 [24, 40]** (was 28 [23, 37]), R² 0.90; **σ\*(0.5) = 0.18 [0.16, 0.20]**,
**σ\*(0.2) = 0.36 [0.32, 0.41]**, σ\*(0.1) = 0.53, σ\*(0.05) = 0.78; faithful-only refit
κ_J = 22. Coherent κ = 3068 (4% off analytic; 2812, i.e. 5%, over σ ≤ 0.1).
**The cost brackets are unchanged** — 32–66% and 159–375% — so every pricing conclusion
and `tab:closeout` row stands.

**Also fixed:** `rem:direction`'s quantifier (J ≤ TV licenses "any class *containing* the
exhibited test", not "every verifier class" — the tracking class is disjoint from the
fixed class, not a superset) at 4 sites; the "better than 1%" Lorentzian-vs-arctan claim
(actually 22% at σ=0.02); README's fictitious "5 failed / 208 passed" baseline (it is
**213/0** in the pinned venv) and its unconditional bit-for-bit claim; the 0.09-at-n=80
floor error in 4 files (the n=80 floor is **0.25**; 0.09 is the n=600 value).

**`plotstyle` now pins DejaVu Sans first** (was Helvetica-first). Every tracked figure was
rendered with DejaVu; Helvetica resolves only on some machines and the silent fallback
changes text metrics and therefore figure bytes. This is what makes regenerate-and-diff
usable.

**Reproducibility, measured properly this time.** In a venv built from `requirements.txt`
verbatim the spike is byte-identical run-to-run. *Across* machines it is not, and cannot
be: pinning numpy does not pin the BLAS (ULP drift, worst 5e-14, no published value moves),
PNG bytes embed the matplotlib version and a zlib-build-dependent stream, and PDF metrics
follow the system fonts. README now says this.

## Current status

- Branch `phase-4`, pushed; **PR #16 open** against `main`. Second review verdict:
  **MERGE AFTER FIXES**, and the fixes are applied (2026-07-27, uncommitted at time of
  writing).
- **Verification done (2026-07-27, in a venv built from `requirements.txt` verbatim):**
  full suite **213 passed / 0 failed**; `git diff results/st2/` empty (the frontier
  freeze is only ever read); the spike regenerates byte-identically run-to-run in a
  fixed environment; and a full re-audit of every quoted number against the regenerated
  artefacts after the arms fix.
- **The PDF now builds** — this laptop has a TeX toolchain, unlike the dev node.
  `latexmk`: **24 pages, 0 errors, 0 undefined refs, 0 undefined citations, 0 overfull
  boxes**, all 12 graphics resolving. First time the manuscript has ever been compiled.
  (One first-pass `natbib` notice from `\usepackage{natbib}` + `\bibliographystyle{unsrt}`
  resolves on the bibtex pass — pre-existing config, harmless.)

## Next steps

1. **`check-refs` + `check-arxiv-llm-compliance` over the whole manuscript** — now tracked
   as an explicit `tasks.md` item. 19 bib entries have never been verified (12 Rung-1/
   related-work, 5 Rung-3/4, 2 new); the compliance pass is owed over all prose written
   today. Both are pre-arXiv blockers.
2. **The last manuscript scaffolds** (`tasks.md`, new item): the abstract; §1 `sec:intro`
   (thesis paragraph, contributions list, scope, target population, why-theory-not-
   hardware); and §4 `sec:scenario`, which is a **pure scaffold with zero body prose** and
   needs **two figures that do not exist yet** (example generator traces + spectra;
   observation-map sensitivity sweep) ⇒ two new plot scripts, CPU-only.
3. **Decision waiting in §1:** whether the covertness cost bound is named as a
   *contribution*. It is currently a §3/App-B result only. Promoting it is a **scope change
   needing `spec.md` approval** — `spec.md`'s "Methods at a glance" names the frontier as
   the central result and lists no identifiability theory. Same precedent as the
   minimum meter spec.
4. Build the PDF somewhere with LaTeX before any submission.

## Key file locations

- New/changed prose: `paper/main.tex` §3 `subsec:identifiability` (~line 364), §6
  `fig:st2_cost_pareto` (~line 860), App B `app:identifiability` (~line 1440+).
- New code: `scripts/plot_st2_pareto.py`, `tests/test_st2_pareto.py`; modified
  `scripts/lower_bound_spike.py` (Youden J).
- Artefacts: `results/spike/lower_bound_spike.json`, `figures/{lower_bound_spike,
  st2_cost_pareto}.{pdf,png}`.
- Derivation record + its corrections: `notes/discussion/lower-bound-feasibility-spike.md`
  (§3 original, **§7 corrections**).

## Gotchas

- **`scripts/plot_st2_frontier.py` WRITES `frontier_summary.json`** (it assembles it from
  the per-family sweeps plus the cost anchors). Do not run it to "refresh" a figure — the
  summary is a slurm freeze. `plot_st2_pareto.py` is the read-only one.
- `scripts/lower_bound_spike.py` is the one script that is *fine* to run locally
  (~17 s, seeded). Everything ST1/ST2 still goes to slurm.
- The `covertness_thresholds` in the spike JSON now come from **Youden J**, not
  `2·AUC−1`. If you quote σ\* anywhere, use that key; `covertness_thresholds_gini` exists
  only for continuity with the note's §4 narrative and does **not** carry the bound.
- §3/App B deliberately state what is **not** proved (the tracking-class case). Don't
  "tidy" those hedges away — they are the difference between the claim being true and false.
- Idle-jitter cost range: the manuscript says **15–375%** (the frozen number). The spike
  note and plan §5 say 15–680%; prefer the frozen figure in prose.

## Backlog (low priority)

Carried over from `tasks.md`, which was retired on 2026-07-27 once its Phase 0–4
tracker was fully closed out. Neither item blocks anything.

- **Trim `powerladder/config.py` to paper-2 parameters.** `B2Params`, `BenchConfig`
  and the beta params are dead weight carried over from the
  `analogue-sensors-for-ai-verification` monorepo. Grep for uses before deleting —
  `DEFAULT` is constructed from all of them.
- **Optionally make `powerladder` pip-installable** (`[project]` + setuptools) and
  drop the `sys.path.insert` shims at the top of every script in `scripts/`. Note
  that `tests/test_scenario_figures.py`, `test_st2_pareto.py`, `test_st2_sweeps.py`
  and `test_lower_bound_spike.py` load scripts by path via `importlib` precisely
  because `scripts/` is not a package; that idiom would still work, but the shims
  inside the scripts could go.

## Pre-submission blocker

- **Emlyn Graham's affiliation is unset.** `paper/main.tex:81` renders as
  "Affiliation TBD" in the author block. `check-arxiv-llm-compliance` flags this as
  Category B placeholder text, and it is the sole reason that check returns FAIL —
  everything else is clean. Fix before any arXiv submission.
