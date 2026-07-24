# ST2 de-periodicisation frontier — full-resolution slurm freeze

**Date:** 2026-07-24 · **Branch:** `feat/phase2-frontier-freeze` · **Status:** frozen.

Phase-2 task "Frontier at full resolution with the systems-cost anchors"
(`tasks.md`). Also closes the **ST2 leg** of the carried Phase-0 re-verification
follow-up (`tasks.md` "Slurm re-verification of the ST2 sweeps") — the FAR/ST1
surrogate leg remains open and separate (Phase 1).

## What was done

The frontier was already computed at full resolution but **locally** — which the
project rules forbid for a number-freeze (ST2 grids run on slurm only). This
session built the missing slurm harness and re-froze the numbers from scratch on
the canonical cluster environment.

- **Harness:** `scripts/plot_st2_sweeps.py` gained a Slurm-array mode
  (`--array-id N` / `SLURM_ARRAY_TASK_ID`, one family per task; `--list`), and
  `--smoke` now writes to a disjoint `frontier_smoke/` tree so a plumbing check
  can never clobber the tracked full-res summaries. New sbatch
  `scripts/slurm/st2_frontier.sbatch` (compute partition, `--array=0-8`,
  single-threaded OpenBLAS = the canonical build). Each family writes a disjoint
  `results/st2/<family>_summary.json`, so array tasks never race (no flock,
  unlike the meter-boundary sweep). Assembly (`scripts/plot_st2_frontier.py`) is
  cheap JSON aggregation, run on the login node.
- **Run:** slurm array **5635** on MATS `compute`, all 9 families COMPLETED
  (~30–65 s each; serialized on one node behind the concurrent st1surr array,
  ~5.5 min wall). Per-family seeding is deterministic off `p.seed` + family index
  (`powerladder/typeb/st2.py`), so the array reproduces the prior local
  `multiprocessing` run exactly.
- **Cost-anchor provenance fix:** the assembler's default `--b2-dir` and every
  recorded `cost_anchor` / `cost_anchor_source` / `cost_anchor_mapping` label
  (across `plot_st2_frontier.py` and `powerladder/typeb/st2_attacks.py`) now
  point at the real, committed anchor location `data/measured_cost_anchors/`
  (`results/b2/` never existed in this repo). No logic change.

## Frozen numbers (`results/st2/frontier_summary.json`)

- `provisional: false`, `n_each: 200`, `target_fars: [0.05, 0.01]`, `seed: 0`,
  **50 cells** (9 families × their level grids), full 5-detector set
  (tracking = viterbi, dg_order_full, dg_order_semicoh; fixed = spectral, mtf).
- **Verdict: GO (definitive).** Criterion fired on the contiguous ~zero-cost run
  `work=0.35, work=0.5`: best tracking TPR@0.05 ≥ 0.8 while every fixed test
  ≤ 0.5. This is the ≈zero-cost real-work-variation attack — the genuine scope
  boundary (measured throughput overhead ≈ 0 within noise, `work` anchor).
- Systems-cost anchors attached on `jitter` (15–375%), `drift` (0.2→142%, the
  sole shared measured level), `work` (≈0% across the grid), `shape` (172–228%);
  analytic-only families (phase/relocate/harmonic/dilute/meter) carry `null`.

## New-vs-old comparison (the freeze reproduces the local run exactly)

Diff of the slurm-frozen `frontier_summary.json` against the previously committed
(local) one, over all 50 cells × 5 detectors:

- **max |AUC delta| = 0.0000, max |TPR delta| = 0.0000** — detection metrics
  identical to all printed digits. Verdict and supporting cells unchanged
  (GO; work=0.35/0.5).
- The only numeric movement is in the derived physical x-axis `phase_diffusion_D`,
  at the last ULP (~1e-15 relative; e.g. `…284704` → `…284725`) — floating-point
  summation-order noise in the diffusion regression, not a result.

So the local numbers were already correct; the freeze makes them provably the
output of the committed code on the canonical cluster build.

## issue #54 clarification (why the frontier is unaffected)

`notes/results/issue54-investigation.md` flags the **measured** work-jitter and
shaped *line-band detection* numbers as riding the card-fixed power-management
side channel (signature B), needing requalification. Those are the
`tpr_at_far_line_band` fields in the anchor files. **The frontier does not consume
them:** its cost axis reads only the clean `throughput_overhead` field (a hardware
throughput measurement, unaffected by B), and its detection numbers come from the
synthetic `results/st2/*_summary.json`, which contain no B. The issue-54
requalification is a measured-data prose concern (paper §5 body, Phase 4), not a
frontier-freeze concern.

## Reproduce

```
# full-resolution numbers (slurm only — NOT locally):
python scripts/plot_st2_sweeps.py --list                 # -> "9 families"
sbatch --array=0-8 scripts/slurm/st2_frontier.sbatch     # MATS `compute`
# then assemble on the login node (cheap):
python scripts/plot_st2_frontier.py
# -> results/st2/{<family>,frontier}_summary.json ; figures/st2_{<family>,frontier}.*

# local plumbing check only (writes to frontier_smoke/, never the tracked files):
python scripts/plot_st2_sweeps.py --smoke
```

## Not done here (Phase 4 write-up)

Wiring the frontier into `paper/main.tex` §5 (`sec:frontier`, still a checklist
scaffold) — including reconciling the checklist's old `results/b2/…` path
reference to `data/measured_cost_anchors/…`, and carrying the issue-54
requalification into the measured-data prose. The learning-efficiency penalty
stays stated as the open empirical question / future work (descoped 2026-07-22).
