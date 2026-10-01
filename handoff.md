# Handoff — 2026-10-01 (§6 on the aggregate at FAR 1e-2; moving to OzSTAR to price missing cost cells)

## Goal
Price the frontier cells that have no measured throughput cost, starting with
**work 0.7**, where the tracker bends and §6.3 calls it "unpriced". This needs a
top-up of the monorepo's B2 A100 campaign on **OzSTAR**.

## Status
- §6 is restructured and on the aggregate at FAR 1e-2. Merged into the feature
  branch `feat/frontier-aggregate` (branched off `restructure`), **not yet merged
  to `restructure`**. Paper builds clean (37 pp); 251 tests pass.
- GPU pricing hasn't started. Tom is resuming on OzSTAR with a fresh Claude
  instance (see `machine-handoff.md`).

## Next steps
1. **On OzSTAR, in the monorepo** (`analogue-sensors-for-ai-verification`, not
   this repo), check the B2 campaign still runs:
   - `python scripts/run_b2_capture.py --phase smoke` and
     `pytest -m gpu tests/test_b2_gpu.py`;
   - run them on `milan-gpu` (A100, account `oz022`), conda base at
     `/fred/oz022/tkimpson/miniconda3`.
2. **Add a top-up phase** (new phase name, e.g. `workjitter_hi` / `spoof_hi`) for
   the 7 off-grid levels:
   - work 0.7 (real-work realisation; most important);
   - jitter 0.7 (idle-pad);
   - drift 0.1, 0.4, 0.8, 1.5 Hz (idle-pad).

   **Don't** edit the existing level tuples in `B2Params`: `plan_phase` draws all
   seeds from one RNG over the full plan, so changing the tuples reshuffles every
   existing trace. Each trace carries its own honest baseline
   (`code/b2/analysis.py::throughput_overhead`), so no new honest population is
   needed. Budget: 8 traces per level at about 300 s, i.e. about 5 h on one A100.
3. Analyse the new traces into overhead (mean, std), then append the new levels
   to the anchor JSONs in this repo's `data/measured_cost_anchors/`. Keep the
   existing keys and add a provenance note in its README. Re-run
   `scripts/plot_st2_frontier.py --st2-dir results/st2/aggregate --fig-name
   st2_frontier_aggregate --b2-dir data/measured_cost_anchors` (it only joins
   costs, so detection numbers don't change), then the two Pareto plots at
   `--far 0.01`.
4. Price **drift 0 at 0%**: it's the honest schedule. This is a frontier-script
   change, not a measurement.
5. Rewrite §6.1 and §6.3 for whatever work 0.7 costs. If it's near zero, a cheap
   attack beats the tracker too, and the §6 story changes; flag that to Tom. If
   it's costly, the tracker's boundary is priced. Recheck "16 of 50 priced cells".
6. Merge `feat/frontier-aggregate` → `restructure` once Tom is happy
   (`/check-PR` first).

## Open questions
- **Figure 7 tidy-ups (Tom hasn't chosen):**
  - drop the two meter cells from the unpriced strip (they belong to §5.2);
  - rename the strip "no hardware cost measurement";
  - explain why shape fill 0 sits at 189%: its anchor was taken on a jitter-0.35
    base, so it is an upper bound.
- **Phase slip / relocation / harmonic smoothing** have no hardware schedules.
  Their cost may be learning efficiency rather than throughput. Leave them
  unpriced for now?
- **Monorepo hardware dilution** (`results/b2/dilute_summary.json`: overhead
  ≈ fraction of time not training) measures single-card time-sharing, not the
  synthetic facility-share dilution. Map it, or leave dilution unpriced?
- **Tom's draft opener paragraph in §6** ("We construct the Pareto curve …") is
  half-written. It was committed by mistake in `a3591f1`; Tom is still editing it.
- Carried from 2026-09-30:
  - §5.2's "holds full power (≳0.9) if and only if …" is contradicted by the
    full-depth notch at 0.8 Hz (0.80 at 1e-2).
  - The §5 intro oversells the background as a hard case (~37 dB down).

## Non-obvious context
- **Aggregate frontier artefacts:** `results/st2/aggregate/*.json`,
  `figures/*_aggregate.*`, and §6/App. D now cite these. The single-workload
  `results/st2/*_summary.json` freeze is kept but no longer cited.
- **New flags:**
  - `scripts/plot_st2_sweeps.py --scenario aggregate`;
  - `SCENARIO=aggregate` for `st2_sweeps.sbatch` / `st2_frontier.sbatch`;
  - `--far` / `--out` on `plot_st2_pareto.py` and `plot_money_pareto.py`;
  - `--frontier hull` on `plot_money_pareto.py` (Tom chose the **staircase**).
- **Verdict FAR:** the pre-registered GO/NO-GO verdict in `frontier_summary.json`
  stays at FAR 0.05; only the figures and text use 1e-2.
- **MATS:**
  - The `~/certifying-training-from-power` checkout is on `feat/frontier-aggregate`
    (not `phase-4`).
  - Seven byte-identical job-14048 outputs were moved to
    `~/mats-stash-job14048-20261001`.
  - A torch venv `~/venvs/b2-torch` exists there but went unused: elastic A100s
    are paid, and the account isn't in `elastic-fellows`.
- **Tom's untracked WIP:** `scripts/plot_scenario_aggregate.py`,
  `figures/scenario_aggregate.*`.

---

(Previous handoff follows.)

# Handoff — 2026-09-30 (later: §5 finished; §6 next)

## Goal
Get §5 (Detection results) and its supporting appendices into final shape, so
the next session can start on §6 (the de-periodicisation frontier).

## Status
§5 is **done and approved by Tom**. The work is on branch `restructure`: flow
pass, NP-ceiling appendix, bake-off appendix dissolved, and §5.2 moved to FAR
1e-2. The paper builds clean at 37 pp, with 0 errors and 0 undefined references.
See `log.md` (2026-09-30 entry) for what changed and why.

## Next steps
1. **§6 flow pass**, same treatment as §5:
   - "In this section we …" opener plus an organisation paragraph;
   - trace every appendix item (App. D "Full attack sweep") to a main-text
     citation and cut leftovers;
   - plain-language pass.
2. **Move §6 to FAR 1e-2.** It is still at 0.05 (`sec:frontier` text,
   `tab:attacks` notes, frontier figures, App. D). Every
   `results/st2/*_summary.json` already stores `tpr_at_far` at `"0.01"`, so no
   rerun is needed. It needs `--far` flags on the ST2 frontier/pareto plot
   scripts, then renumbering. Attack costs will move: recheck every §6 number
   against the JSONs.
3. Resolve the open §5.2 question below.

## Open questions
- **§5.2 "holds full power (≳0.9) if and only if …"** is contradicted by the
  full-depth notch at 0.8 Hz (tracker 0.80 at 1e-2; see
  `fig:meter_boundary_notch`). Options: loosen the threshold to ≳0.8, or define
  "at the cadence" as 0.8–1.2 Hz. Tom has not chosen yet.
- The §5 intro still says the tracker "has to find the dominant line among" the
  background's weak lines. That oversells the background as a hard case, given
  it sits ~37 dB down. Carried from the earlier handoff.

## Non-obvious context
- **Frozen ST2 summaries hold both FARs.** Switching operating point is a replot,
  not a slurm job. `plot_st2_meter_boundary.py` and `plot_drift_ceiling.py` take
  `--far` (default 0.01). `_far_key` refuses a FAR the summary doesn't store.
- **Labels renamed:**
  - `app:bakeoff` → `app:ceiling` ("Optimality ceiling");
  - the controller-confound result is now `subsec:confound` / `tab:confound`
    inside `app:attribution`;
  - `tab:bakeoff`, `fig:st1_bakeoff`, `fig:np_ceiling` and `fig:meter_boundary`
    no longer exist.
  - App. E is "Meter specification details" (`app:meter`).
- **Orphaned figures, left on disk:** `figures/st1_bakeoff*`,
  `figures/st1_np_ceiling*` and `figures/st2_meter_boundary_aggregate.*`
  (per-detector grid). Single-workload `figures/st2_meter_boundary*` are still
  at 0.05 and unused.
- **Semi-coherent DG numbers have no committed results file.** The `--semicoh`
  bake-off run shared the output path `results/st1/bakeoff_summary.json` with
  the default run. Don't cite them without a rerun to a distinct file.
- **Earlier TODOs 6–7 (below) now refer to `subsec:confound`.** These are the
  known-line veto and the surrogate-calibrated Viterbi; they previously referred
  to "the bake-off".
- **Tom's WIP, still uncommitted:** `scripts/plot_scenario_aggregate.py` and
  `figures/scenario_aggregate.*`.

(Previous handoff follows.)

---

# Handoff — 2026-09-30 (section 5 moved onto the multi-workload aggregate)

## What happened this session

Branch **`feat/aggregate-headline`** (off `restructure`, merged back into it).
Resolved the two red TODOs (§4.2 threshold population, head of §5). §5 now tests
the §2.4 aggregate: dominant training over 4 small trainings + 4 fine-tunings,
against the null aggregate (inference in the dominant slot), shares 9:0.5:0.5.

- **Answer: the method still works.** Viterbi on the aggregate at FAR 1e-2 gets
  1.00/1.00/1.00/1.00/0.99/0.96 over drift 0–1.5 Hz. The Whittle NP ceiling is 1.0
  everywhere, and its parity vs the aggregate bake-off is exact (0).
  Single-workload vs aggregate: Viterbi moves ≤0.02; the largest move for any
  detector is 0.12 (DG-full at 0.1 Hz).
- **Why:** each background member has 1.25% of peak vs 90%, so its line is
  ~37 dB down and buried. The background only matters at low dominant share,
  which is the dilution attack (§6). The existing `dilute_summary.json` already
  shows Viterbi 1.00 down to share 0.2 (no drift).
- **Code:** `--scenario {single,aggregate}` on `plot_st1_bakeoff.py`,
  `st1_np_ceiling.py`, `st2_meter_boundary.py`, `plot_drift_ceiling.py`,
  `plot_st2_meter_boundary.py`. Library: `train_obs_at_f0`/`build_training_bank`
  (`aggregate=`) and `run_meter_cell(aggregate=)`. Single defaults keep the frozen
  names and numbers. New tests: `tests/test_aggregate_scenario.py`.
- **Frozen on MATS:** job 14048 (aggregate bake-off + ceiling, ~40 min;
  `scripts/slurm/st1_aggregate_headline.sbatch`) and array 14049 (aggregate meter
  boundary; `sbatch --export=ALL,SCENARIO=aggregate scripts/slurm/st2_meter_boundary.sbatch`).
  The branch was pushed straight into the MATS checkout over ssh (not GitHub);
  that checkout was switched back to `phase-4`.
- **Paper:** main-text figures are now `drift_ceiling_aggregate` and
  `st2_meter_boundary_aggregate_viterbi`, and the appendix meter figures are aggregate too.
  `tab:bakeoff` gains an aggregate block; the rest of the bake-off appendix
  (structural confusers, semicoh) stays single-workload, as does the frontier's
  meter-erasure decomposition (labelled). Builds clean (39 pp, 0 errors/overfull/undefined).

## Open items from this pass

1. **Tom's WIP left untracked:** `scripts/plot_scenario_aggregate.py` +
   `figures/scenario_aggregate.*` (shares 0.9 / 0.5 / null). A natural §2.4 or
   appendix figure to back the "37 dB down" paragraph in §5.1; not committed.
2. The committed `figures/st2_meter_boundary{,_notch}.*` (single) are stale vs
   their script: regenerating them changes the bytes even on the base commit.
   Not touched.
3. The §5 intro still says "the tracker has to find the dominant line"; consider
   whether §2.4's "weak lines inside the search band" framing oversells the
   background as a hard case, given the ~37 dB result.
4. The frontier (§6) and rung-2 are still single-workload apart from dilution.

---

# Previous handoff — 2026-09-23 (introduction restructured as a methods paper)

## What happened this session

Ran `/write-introduction` in critique-and-revise mode on `paper/main.tex` and
applied the result on branch **`restructure`**:

- **Introduction rewritten** (1693 words / 11 paragraphs / 16.5% of body →
  ~1000 words / 7 paragraphs / 10.5%). Role structure is marked with
  `% ---------- [n] ... ----------` comments: [1] hook, [2a–c] literature
  (physical signature; signal-processing lineages; covert comms), [3] one-sentence
  gap + stakes, [4] aim as five doing-verb claims with `\cref` pointers, [5]
  roadmap. Framed as a **methods paper**: the gap is "no method turns a low-rate
  external trace into a calibrated test, prices its evasion, and states what the
  channel cannot certify."
- **Cut:** the red developer's note and its preamble macros (`\devnote`,
  `\devpath`; `xcolor` kept); the results-preview paragraph ("Our answer is…");
  the two rung-by-rung ladder paragraphs; the "Three restrictions fix the scope"
  paragraphs.
- **Moved:** `tab:ladder` + a lead-in paragraph (ladder description, benign vs
  adaptive prover) now open `sec:threat`. The three scope restrictions now form
  the `\paragraph{Scope.}` of `sec:discussion`. Distance-bounding / PUF citations
  re-homed to `app:active`.
- **New intro citations:** `patel2023polca`, `streit1990frequency`,
  `suvorova2016hmm`, `bayley2019soap` (the sonar/CW-GW lineage the 2026-08-31
  handoff flagged as uncited is now cited in the intro too).
- **Now uncited anywhere:** `brier2004cpa`, `maia2022magnetic`,
  `naghibijouybari2018rendered`, `hu2020deepsniffer`, `liu2016codeexecution`
  (the GPU side-channel corpus). Bib entries left in place; re-home or drop.
- **Proposed citation left as a LaTeX comment** in intro §[2a]: Vercellino et al.
  2026, arXiv:2604.07345 (multi-node H100 traces). Verify with `/check-refs`
  before adding to `references.bib`.
- PDF rebuilt with the pinned-epoch command: **28 pp**, 0 errors / overfull /
  undefined. Ladder table floats to p. 4.

## Open items from this pass

1. **Conclusion opening no longer answers the gap.** `sec:discussion` still opens
   "We set out to ask what a passive power meter can certify…"; rewrite to answer
   the methods gap (build a calibrated test, price evasion, state the ceiling).
2. **Abstract** leads with the ladder, not the method; revisit with `/write-abstract`
   after the rest of the restructure.
3. Two intro sentences are my gloss and should be checked against the body:
   "both do so for *every* verifier at once" (steganography/covert comms) and
   "tests for cyclostationary coherence at the tracked order" (matches the
   stepwise memo's phrasing; confirm against `sec:methods`).
4. Retitle? Review memo suggests foregrounding the low-rate condition.
5. The review memo's remaining surgery (four "null" jobs in `sec:results`;
   compress synthetic classification; async control; power stabilisation in the
   threat model) is untouched.

## Experiment TODOs from the 2026-09-23 restructure (Viterbi-led main text)

The paper was restructured around the Viterbi line tracker as *the* method
(plan: `~/.claude/plans/consider-the-manuscript-paper-main-tex-rosy-star.md`).
Two experiments are needed to back that framing; neither is run yet.

6. **Known-line veto.** Viterbi alone scores 0.00 vs the mixed structural
   nulls because it locks onto the controller limit cycle. Borrow the CW-GW
   known-lines veto: extend Viterbi to multi-candidate extraction
   (peel-and-rerun via `viterbi_best_path`, `powerladder/typeb/detectors.py:266`,
   or sub-banding), veto candidates near catalogued instrument lines (A100
   controller 0.31-0.45 Hz), re-score the bake-off vs mixed structural nulls,
   and add the veto band's relocation blind spot to the frontier
   (frequency-relocation family). Caveats to state: the adversary knows the
   veto list; lines need per-facility characterisation; load-coupled
   controller lines are not handled by a frequency veto.
7. **Surrogate-calibrated Viterbi.** Viterbi thresholds are Monte Carlo on the
   modelled null; the surrogate campaign only ever calibrated DG's Q_alpha.
   Run the Viterbi score through the Fourier-phase surrogate + FAR harness in
   `powerladder/st1/` over the null suite, to give the headline detector a
   deployable per-trace threshold.

(Previous handoff follows.)

---

# Handoff — 2026-09-11 (external-style project review)

## What happened this session

Recorded a self-contained assessment of the project's usefulness to AI safety and
verification, novelty against the 2026 literature, empirical gap, asynchronous-
training boundary, production power-stabilization challenge, and null-test
simplification:

- **`notes/discussion/project-review-usefulness-novelty-and-empirical-priorities.md`**
  — dated discussion memo with an executive verdict, multi-worker coherence analysis,
  public H100-dataset reanalysis plan, preferred owned experiment, novelty table,
  four-way cleanup of the overloaded "null" terminology, and a prioritized revision
  sequence.
- **`notes/discussion/project-review-usefulness-novelty-and-empirical-priorities.pdf`**
  — six-page A4 rendering of the memo, generated with Pandoc and XeLaTeX; hyperlinks
  and displayed mathematics are preserved. The Markdown uses Pandoc-compatible
  dollar-delimited math (`$...$` / `$$...$$`), after correcting an initial render
  that printed single-backslash math delimiters literally.

No code, frozen result, manuscript text, or `spec.md` scope changed. Existing untracked
`results/hsmm_smoke/` was left untouched. Current branch remains
**`docs/cite-gargiulo-kulp`**.

## Main conclusions captured in the note

1. The strongest contribution is measurement science for a low-rate, independently
   observed verification channel—not the generic claim that power can reveal
   training.
2. The current "asynchronous training" control is one oscillator with additional
   drift/phase slip, not a true multi-worker asynchronous model. A coherence sweep is
   required before claiming robustness.
3. The public Vercellino et al. 8–64 GPU H100 dataset is the fastest route to testing
   the central scale-and-coherence prediction, though it does not validate external
   meter transport.
4. Production power stabilization may challenge the present evasion-cost story and
   should be added to the threat model.
5. The main-text statistics can be simplified by separating calibration surrogates,
   operational negatives, confounder stress tests, and semantic counterexamples.

## Possible next steps

1. Decide whether to turn the review's recommended revision sequence into an
   implementation plan.
2. If yes, start with the public H100 trace reanalysis and the multi-oscillator
   coherence model before restructuring manuscript prose.
3. Carry forward the existing Gargiulo & Kulp manuscript edits and third-author
   affiliation blocker below.

(Previous handoff follows.)

---

# Handoff — 2026-09-03 (positioning vs Gargiulo & Kulp, arXiv 2609.00309)

## What happened this session

A new paper appeared (Gargiulo & Kulp, *Workload Identification with Physical Side
Channels for AI Governance*, arXiv 2609.00309, 31 Aug 2026): 97% training / inference
/ non-AI classification from a verifier-owned 10 MHz Rogowski probe on a single
H200's PCIe power conductors, with four measured evasions costing 28–69% throughput.
Assessed the overlap and wrote the positioning record on branch
**`docs/position-vs-gargiulo-kulp`**:

- **`notes/discussion/positioning-vs-gargiulo-kulp-2026.md`** — verdict (not sunk:
  same headline, opposite end of the where-is-the-meter axis), side-by-side table,
  a careful treatment of the hardware gap (their "no gap" is bought by the threat
  model; the single-card gap is shared but bites only our collective observable;
  corpus-per-device vs model), adversary/cost comparison, the semantic ceiling
  applied to their observable, review risks with answers, and a numbered list of
  additive manuscript edits (§9) plus an optional experiment on their public
  dataset (§10).

## Next steps

1. Apply the §9 manuscript edits (bib entry `gargiulo2026workload`; intro trust-anchor
   sentence; "what changes here" paragraph; `subsec:meter_spec`; decoy paragraph;
   `sec:frontier`; Limitations). Additive only, no frozen numbers. Rebuild `main.pdf`
   with the pinned-epoch invocation; run `check-refs`.
2. Decide whether to run the §10 bandwidth/integration sweep on their Hugging Face
   traces through `apply_meter`.
3. Precision: they do **not** say "unspoofable"; quote "observed independently of
   operator cooperation" and "requires restructuring the computation itself".

(Previous handoff follows.)

---

# Handoff — 2026-08-31 (track-before-detect cast; forward-statistic prototype)

## What happened this session

Answered the question "is there a technique other than Viterbi we should try — can
this be cast as a known signal-analysis problem?", then wrote it up and prototyped
the answer. Branch **`claude/viterbi-alternative-techniques-ns1lpx`**, pushed.

### 1. The cast (write-up)

**`notes/discussion/track-before-detect-forward-statistic.md`** — the full record.
Short version: the Viterbi detector is an instance of **track-before-detect**
(radar/sonar; narrowband case = passive-sonar lofargram line tracking). Under the
HMM the tracker already assumes, Viterbi is the MAP-path *approximation*; the
Neyman–Pearson statistic is the **marginal likelihood — the forward algorithm, sum
over all paths** — at identical cost (`max` → `logsumexp`, penalty → normalised
kernel). They agree when one path dominates and diverge at low per-frame SNR under
heavy wander — the adversarial jitter regime.

Also found: `references.bib` carries the sonar→CW-GW lineage block
(`streit1990frequency`, `suvorova2016hmm`, `bayley2019soap`, `djurovic2011viterbi`)
but **none of the four keys is cited in `main.tex`** — cheap related-work fix.

### 2. The prototype

- `powerladder/typeb/detectors.py`: **`forward_statistic`** + sequential
  **`forward_path_scores`** + `_wander_log_kernel`, mirroring the Viterbi pair
  (same emission map, defaults, detector contract). Additive only.
- `tests/test_typeb.py`: 5 new tests + `forward_statistic` added to the ranking
  parametrisation (§ "forward (sum-over-paths) statistic"). **40/40 pass.**
- Full suite: **229 passed / 5 failed** — the 5 are the known cross-machine BLAS
  byte-identity digest tests (`test_ko_workload` ×3, `test_deperiod`,
  `test_ko_synth_meter`), verified identical on a clean checkout of `main` in this
  container. Not caused by this change; would be 0 in the pinned reference venv.

### 3. Smoke result (scratchpad, NOT frozen)

Amplitude × wander sweep on the B0-local synth generator, n_each=100: forward ≥
Viterbi in **every** cell; gap opens exactly at weak-line + heavy-wander (amp 8,
wander 0.6: AUC 0.94 vs 0.86, TPR@0.05 0.79 vs 0.65; amp 6: AUC 0.82 vs 0.72).
Seed-robust (seeds 1–3: AUC gap 0.06–0.13). Full table in the note §4.

## Current status

- Branch pushed; no PR opened (not requested). **No frozen number changes**: ST1/ST2
  freezes, figures, manuscript, `spec.md` all untouched.
- The forward statistic is a **prototype/bank candidate**, cited in no result. Any
  promotion is a new bank member with its own surrogate calibration + FAR run.

## Next steps

1. **Real evaluation of the forward statistic on the headroom axes** (slurm-scale,
   bake-off harness): structural confusers at high drift and the tracking-class
   ceiling for the σ* covertness bound (`rem:direction` is loose against exactly
   this class). Do **not** evaluate on the plain inference null — Viterbi is
   already at ρ_auc = 1.00 there (np-ceiling note).
2. **Harmonic-comb emissions** (pitch-tracking cast; note §6) — the follow-on that
   targets the controller AUC-0.0 failure; composes with the forward recursion.
3. ~~Cite the lineage block~~ — **DONE (2026-08-31, same branch/PR #20):** the four
   keys are now cited in `subsec:tracked` (tracker lineage sentence) and
   `app:bakeoff` (headroom passage names the forward statistic as future work, no
   result claimed). `main.pdf` rebuilt with the pinned-epoch invocation: 28 pp,
   0 errors / undefined / overfull — but in this session's **container TeX Live,
   not the usual laptop toolchain**, so byte provenance differs (the known
   cross-machine font/metric caveat); rebuild on the laptop if byte-continuity
   with earlier PDFs matters.
4. (Carried) **Pre-submission blocker:** third author's affiliation renders as
   "Affiliation TBD" (`paper/main.tex` author block) — sole `check-arxiv-llm-compliance`
   FAIL.

## Key file locations

- Note: `notes/discussion/track-before-detect-forward-statistic.md`.
- Code: `powerladder/typeb/detectors.py` (forward pair at the end of the
  Viterbi section); tests in `tests/test_typeb.py`.
- Smoke script (scratchpad only, not committed): amplitude × wander AUC/TPR sweep;
  numbers reproduced in the note §4 with seeds.

## Gotchas (carried forward)

- **`scripts/plot_st2_frontier.py` WRITES `frontier_summary.json`** (slurm freeze).
  `plot_st2_pareto.py` is the read-only one. Everything ST1/ST2 runs on slurm;
  `scripts/lower_bound_spike.py` is the one locally-runnable script (~17 s, seeded).
- Spike JSON: quote σ* from `covertness_thresholds` (Youden J), never the
  `_gini` key.
- §3/App B deliberately state what is **not** proved (tracking-class case) — don't
  tidy those hedges away.
- Idle-jitter cost range in prose: **15–375%** (frozen), not 15–680%.
- `forward_statistic` values are **not numerically comparable** to
  `viterbi_statistic` (normalised vs unnormalised wander prior); ROC/surrogate
  calibration only.

## Backlog (low priority, carried forward)

- Trim `powerladder/config.py` of monorepo dead weight (`B2Params`, `BenchConfig`,
  beta params) — grep first; `DEFAULT` is constructed from all of them.
- Optionally make `powerladder` pip-installable and drop the `sys.path.insert`
  shims in `scripts/` (tests load scripts by path via `importlib`; that keeps
  working).
- `jump_penalty` sensitivity sweep for the forward statistic (default 1.0
  inherited, meaning differs under the normalised kernel) — fold into any real
  evaluation (note §9).
