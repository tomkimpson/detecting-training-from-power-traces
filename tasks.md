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
  behind the "integrating_1hz" claim. Do NOT run locally on the dev node (decision
  2026-07-22) — slurm only; environment setup on the cluster is part of this task.
- [x] **Settle the venue** (plan §10 Q1).
  **Result (2026-07-22): arXiv-first.** Write the strongest self-contained preprint;
  choose the submission venue after results freeze. Identifiability rigor at
  proposition level unless the eventual venue demands more.

---

## Phase 1 — Rung 1 full (NEXT)

- [ ] Qualified multitaper F-test (benign) + tracked-cyclostationary (adaptive) on the
  scenario model, both adversary settings.
- [ ] Null-validity validation section wording from `notes/results/st1-findings.md`.
- [ ] Pre-registered detector bake-off → paper §4 / Appendix A.
- [ ] Wire the ST1 figures (calibration, bake-off, power) into `paper/main.tex`.

## Phase 2 — Frontier (full) + Rung 2

- [ ] Frontier at full resolution with the systems-cost anchors. **Learning-efficiency
  metric descoped (decision 2026-07-22):** no GPU measurement campaign — keeps the
  paper theory/methods and avoids the single-GPU/NVML transport trap. The frontier's
  utility axis is the measured systems-cost anchors only; the learning-efficiency
  penalty is stated in the paper as the open empirical question / future work
  (paper §5 checklist already words it this way).
- [ ] **Meter-requirement boundary sweep** (supports promoting the ST2 channel-
  requirement finding to a named "minimum meter specification" contribution —
  pending spec.md approval): grid over `sample_hz` × `integ_window_s` between the
  nominal 20 Hz channel and the 1 Hz integrating sampler, plus notch depth/Q, honest
  training only; locate where each detector class dies. CPU-only synthetic; run as a
  slurm array.
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
