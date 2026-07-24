# Handoff — 2026-07-24 (Phase 2: ST2 frontier frozen at full resolution on slurm)

## What happened this session
Froze the **ST2 de-periodicisation frontier at full resolution on slurm** — the
Phase-2 task "Frontier at full resolution with the systems-cost anchors". Branch
`feat/phase2-frontier-freeze` (not yet merged / pushed).

The frontier was already computed at full resolution but only **locally**, which
the project rules forbid for a number-freeze (ST2 grids run on slurm only). Built
the missing harness and re-froze from scratch on the canonical cluster build.

- **Harness:** `plot_st2_sweeps.py` gained a Slurm-array mode (`--array-id N` /
  `SLURM_ARRAY_TASK_ID`, one family per task; `--list`), and `--smoke` now writes
  to a disjoint `frontier_smoke/` tree so a plumbing check can't clobber the
  tracked full-res summaries. New sbatch `scripts/slurm/st2_frontier.sbatch`
  (compute partition, `--array=0-8`, single-threaded OpenBLAS). Each family writes
  a disjoint `<family>_summary.json` → no race, no flock.
- **Run:** slurm array **5635** on MATS `compute`, all 9 families COMPLETED
  (~5.5 min wall, serialized behind a concurrent st1surr array). Assembly via
  `plot_st2_frontier.py` on the login node.
- **Frozen result:** `results/st2/frontier_summary.json` — provisional=false,
  n_each=200, 50 cells, full 5-detector set, **verdict GO (definitive)** on the
  ≈zero-cost real-work-variation attack `work=0.35/0.5`.
- **Byte-identical to the prior local run** (max |AUC/TPR Δ| = 0.0000 over all 50
  cells × 5 detectors; only `phase_diffusion_D` moves at the last ULP) — confirms
  determinism and that the local numbers were already correct.
- **Cost-anchor provenance fixed:** default `--b2-dir` and every recorded
  `cost_anchor`/`cost_anchor_source`/`cost_anchor_mapping` label (in
  `plot_st2_frontier.py` AND `powerladder/typeb/st2_attacks.py`) now point at the
  real committed location `data/measured_cost_anchors/` (`results/b2/` never
  existed here). Two tests updated to the new path (`test_st2.py`,
  `test_st2_sweeps.py`).
- Full record: `notes/results/st2-frontier-freeze-findings.md`.

## Current status
- Branch `feat/phase2-frontier-freeze` — **not yet merged / pushed.** Contains:
  3 script/lib code files, 2 test fixes, regenerated `results/st2/*` +
  `figures/st2_*`, the findings note, `tasks.md` + this handoff.
- `tasks.md`: the Phase-2 frontier task is ticked `[x]` with a Result; the
  Phase-0 "Slurm re-verification of the ST2 sweeps" follow-up is also ticked
  (its ST2-synthetic leg is what this freeze closes).
- `pytest -p no:debugging -m "not gpu"` → **5 failed, 181 passed** = the known
  baseline (BLAS byte-identity digests; env, not a bug) + the merged rung2 tests.
  No new failures.
- **issue-54 clarification (important, carried to Phase 4):** the frontier's cost
  axis reads only the clean measured `throughput_overhead` field; the
  B-contaminated `tpr_at_far_line_band` detection arrays in the same anchor files
  are NOT consumed. The issue-54 requalification is a measured-data prose concern
  for the paper §6 body, not a frontier concern.

## Next steps
1. **`check-PR` + merge `feat/phase2-frontier-freeze` → main.** Self-contained;
   no paper prose or bib entries touched, so `check-refs`/`check-arxiv` not needed.
2. **Phase 2 remainder:** the two remaining Phase-2 items are the (optional,
   non-blocking) NP-optimal LRT ceiling, and paper wiring — both deferrable.
   Rung 2 is already merged (PR #10). Meter-boundary merged (PR #5).
3. **Phase 4 write-up** now has all Phase-2 numbers frozen:
   - `paper/main.tex` §6 `sec:frontier` (still a checklist scaffold) — wire the
     frozen `frontier_summary.json` + `figures/st2_frontier.*`; reconcile the
     checklist's stale `results/b2/…` path (line ~633) to
     `data/measured_cost_anchors/…`; carry the issue-54 requalification into the
     measured-data prose; state learning-efficiency as future work.
   - Rung-2 subsection + minimum-meter-spec subsection (both frozen, unwired).
4. **Still-open Phase 1 number-freeze (separate, blocking a paper freeze):** the
   ST1 surrogate S=999/M=10⁴ slurm leg — an `st1surr` array (job 5567) was
   observed RUNNING on `compute` this session; check whether it is the S=999/M=10⁴
   confirmation and whether it completed. Do NOT run sweeps on the dev node.

## Key file locations
- Harness: `scripts/plot_st2_sweeps.py` (`--array-id`/`--list`/smoke split),
  `scripts/slurm/st2_frontier.sbatch`, `scripts/plot_st2_frontier.py` (assembler +
  anchor fix), `powerladder/typeb/st2_attacks.py` (anchor labels).
- Frozen data: `results/st2/frontier_summary.json` (+ per-family
  `<family>_summary.json`), `figures/st2_frontier.*`, `figures/st2_<family>.*`.
- Anchors (static input): `data/measured_cost_anchors/{spoof,workjitter,shaped}_summary.json`.
- Findings: `notes/results/st2-frontier-freeze-findings.md`.
- Reused unchanged: `powerladder/typeb/{st2.py,deperiod.py}`, `powerladder/config.py`
  (`St2Params`, `DEFAULT.st2`).

## Gotchas
- **slurm only** for ST2/FAR grids — never on the dev node (auto-memory
  `expensive-sweeps-via-slurm`).
- `sacct` accounting DB was down this session (connection refused) — verify slurm
  jobs via the per-task `.out` files under `results/st2/slurm/` instead.
- The frontier assembler always writes the figure to the tracked
  `figures/st2_frontier.*` regardless of `--st2-dir`; run it only against the real
  full-res summaries (a smoke assembly clobbers the tracked figure — restore from
  git if that happens).
- `venv/` in this worktree is functional (numpy 2.2.6 etc.); the sbatch activates it.
