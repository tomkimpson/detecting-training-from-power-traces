# Handoff — 2026-07-22 (plan iteration: spec/tasks decisions locked)

## What happened this session
Plan-iteration session (no experiments): reviewed `spec.md`/`tasks.md` against the
paper outline and `notes/plans/plan-for-paper-2.md` + review. The pasted outline == plan §7
== `paper/main.tex` structure, so no structural change was needed. Three scope
decisions were made and recorded in `spec.md` (user-approved) and `tasks.md`:

1. **Learning-efficiency axis descoped.** No GPU measurement campaign — keeps the
   paper theory/methods and avoids the single-GPU/NVML transport trap. The frontier's
   utility axis is the measured systems-cost anchors only; the learning-efficiency
   penalty is stated as the open empirical question / future work.
2. **Venue: arXiv-first.** Write the strongest self-contained preprint; pick the venue
   after results freeze. Identifiability rigor at proposition level for now.
3. **Meter channel-requirement promoted to a named contribution.** The ST2
   `integrating_1hz` finding (1 s trailing boxcar + 1 Hz ZOH sampling defeats all
   detectors on an *unmodified* honest workload) is now a spec.md deliverable:
   "minimum meter specification." Mechanism verified from code + committed JSON
   (sinc null at the 1 Hz band centre + whole-band aliasing past the 0.5 Hz Nyquist);
   honest wording: TPR@0.05 ≤ 0.32 for every detector, order methods to chance —
   but Viterbi retains AUC 0.68 via band-edge aliasing, so say "defeats at the stated
   operating point," not absolute "erases."

Also: spec.md's Rung-1-adaptive methods bullet now records the ST1 outcome (no
analytic CFAR; effective score with per-trace surrogate calibration — that is what
"calibrated" in the thesis means).

## Current status
- Branch `feat/plan-iteration` (spec/tasks/handoff edits only; no code changes).
- No local Python env on this MATS dev node — deliberately not created. Expensive
  sweeps run via slurm ONLY (user decision this session; see memory note).

## Next steps
1. **Merge `feat/plan-iteration` → main** once pushed/reviewed.
2. **Phase 1 (Rung 1 full)** — see `tasks.md`: F-test + tracked-cyclostationary
   sections, bake-off, wire ST1 figures into `paper/main.tex`. Mostly turning
   `notes/results/st1-findings.md` into prose; figures already exist.
3. **Slurm number-freeze pass** (before freezing any paper numbers): ST1 surrogate
   grid S=999/M=10⁴ AND from-scratch ST2 sweep re-verification (esp. meter family).
   MATS preferred (`/mats-cluster` skill); env setup on cluster is part of the task.
4. **Novelty audit** for adaptive order/cyclostationary detection +
   challenge-synchronous power authentication — still unowned (ST0 skipped it; the
   §1 checklist requires it before related-work prose). Machinery-diagnostics
   order-tracking literature is the exposure to check.
5. **Meter-requirement boundary sweep** (Phase 2 task): `sample_hz` ×
   `integ_window_s` (+ notch) grid between 20 Hz and 1 Hz channels — turns the
   two-point contrast into an actual minimum specification.

## Gotchas
- `pytest -p no:debugging -m "not gpu"`; 4 sklearn test failures are environmental.
- Do not run ST2/FAR sweeps on the dev node — slurm only.
- spec.md edits always need explicit user approval (done for this round).
