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
- [ ] **Slurm surrogate confirmation before any paper number freezes:** S=999, M=10⁴,
  incl. stage 3 and stage-6 nulls (ar1_t, lognormal, sq_gauss). Slurm array over cells
  (`st1_far.py --calibs surrogate`, one cell per task) on MATS (preferred) or OzSTAR.
  The ST1 verdict's surrogate leg currently rests on a partial (M=1000, S=199) grid.
- [ ] **Slurm re-verification of the ST2 sweeps in this repo** (same number-freeze
  pass): regenerate `results/st2/*_summary.json` from scratch, esp. the meter family
  behind the "integrating_1hz" claim **and the work-jitter line-band numbers, which
  ride the `B` component and are flagged for requalification** (`notes/results/issue54-investigation.md`).
  Which finding leads the frontier is left open until these freeze (positioning memo,
  2026-07-23). Do NOT run locally on the dev node (decision 2026-07-22) — slurm only;
  environment setup on the cluster is part of this task.
- [x] **Settle the venue** (plan §10 Q1).
  **Result (2026-07-22): arXiv-first.** Write the strongest self-contained preprint;
  choose the submission venue after results freeze. Identifiability rigor at
  proposition level unless the eventual venue demands more.

### Strategic gates (resolve before Phase 1 numbers freeze)

From the 2026-07-23 positioning memos (`notes/discussion/north-star-and-positioning.md`,
`method-soundness-and-prior-art.md`). Sequenced: audit first, then the spike, then decide.

- [ ] **ST0 prior-art audit (four fields).** Position the work against the four mature
  literatures the ST0 desk work skipped: **NILM** (non-intrusive load monitoring),
  **spectrum-sensing / LPI–LPD detection**, **power side-channel analysis** (DPA/CPA),
  and **covert-communication / steganography** (the warden game). Cheap desk work; it
  either hands us tools or surfaces the paper that already did this. This is the canonical
  home for the still-unowned "novelty audit" referenced in Phase 1; it is a §1 related-work
  prerequisite and gates the lower-bound spike below.
- [ ] **Lower-bound feasibility spike (the kill/continue gate).** Time-boxed: is a
  TV/KL (covertness) lower bound on hiding cost derivable for even one attack family
  (e.g. i.i.d. phase jitter)? Anchor it in the physics (hiding synchronous all-reduce is
  communication-bound and costly), with the TV/KL machinery as the formal wrapper — a
  purely generator-internal bound is still conditional on the generator. **Outcome is a
  hard fork on claim strength:** bound exists ⇒ contribution, frame as "must pay ≥ X";
  no bound ⇒ position paper, frame as "evidence + upper bound on hideability, necessary
  conditions only." Feeds the Phase 4 identifiability section (plan §10 Q2).

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

**Still open before Phase 1 numbers freeze / merge:** (i) slurm S=999/M=10⁴ surrogate
confirmation incl. stage-6 non-Gaussian nulls + stage 3 (removes the provisional footnote);
(ii) `check-refs` / `check-arxiv-llm-compliance` over the new prose + 4 new bib entries;
(iii) the ST0 prior-art audit (four fields — see Strategic gates above; §1 related-work
prerequisite). PDF not built locally
(no TeX toolchain on the dev node) — verified statically (figures on disk, cite keys
resolve, labels/refs consistent, environments balanced).

## Phase 2 — Frontier (full) + Rung 2

- [ ] Frontier at full resolution with the systems-cost anchors. **Learning-efficiency
  metric descoped (decision 2026-07-22):** no GPU measurement campaign — keeps the
  paper theory/methods and avoids the single-GPU/NVML transport trap. The frontier's
  utility axis is the measured systems-cost anchors only; the learning-efficiency
  penalty is stated in the paper as the open empirical question / future work
  (paper §5 checklist already words it this way).
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
  **Not yet wired into `paper/main.tex` (§6/new subsection) — Phase 4 write-up.**
- [ ] **(Optional, non-blocking) NP-optimal LRT ceiling in the bake-off.** We have the
  generator, so compute/approximate the Neyman–Pearson optimal detector between the
  training and null generators and report the corpus-free surrogate detector as a
  fraction of it ("X% of NP-optimal power at Y% of the information cost"). Quantifies
  "is our detector good" — not a pre-gate (decision 2026-07-23).
- [ ] Rung 2: stated inference null; three decision rules; semantic falsification
  controls (periodic inference, gradient-only, discarded-update decoy, training-shaped
  non-ML loop, controller cycle, async training, co-resident mixtures); transfer / domain
  shift evaluated.

## Phase 3 — Rungs 3–4 conditional protocol

- [ ] Concise challenge-synchronous protocol; assumed transcript primitive as an ideal
  functionality; randomisation-test discipline; ceilings stated. (Full empirical
  treatment only if ST3 is later run and passes.)

## Phase 4 — Write-up

- [ ] Negative transport case (single-A100) as a result; identifiability theory section
  (rigor level per plan §10 Q2); discussion closing out each rung's ceiling.

---

## Housekeeping / tech-debt (low priority)

- [ ] Trim `powerladder/config.py` to paper-2 params (`B2Params`/`BenchConfig`/beta
  params are dead weight carried over from the monorepo).
- [ ] Optionally make `powerladder` pip-installable (`[project]` + setuptools) and drop
  the `sys.path.insert` shims in `scripts/`.
