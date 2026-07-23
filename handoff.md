# Handoff — 2026-07-23 (Phase 2: meter-requirement boundary sweep done)

## What happened this session
Built and ran the **meter-requirement boundary sweep** (Phase 2 task; spec.md
"minimum meter specification"). Branch `feat/phase2-meter-boundary`, two commits:
1. `86c8974` — the harness (config + observation + library + scripts + tests);
2. `72a03ea` — the results, figures, findings note, and docs.

The sweep generalises the ST2 `meter` family from four named variants to a dense
`sample_hz × integ_window_s` grid (+ a notch depth/Q sub-sweep), honest training
only, reusing every scoring primitive (`make_positive/negative_population`,
`gate.score_population`, `roc`). 48 cells, `n_each=200`, ran as a **Slurm array
on MATS `compute`** (job 5498, one cell/task, crc-seeded, flock-merged summary —
all cells COMPLETED, no errors).

**Result (`notes/results/st2-meter-boundary-findings.md`):** the tracking class
(Viterbi, spectral) holds TPR@0.05 ≳ 0.9 iff sample rate ≥ ~2 Hz **and**
integration window ≲ 0.5 s **and** no deep in-band notch at the cadence centre.
fs=1 Hz collapses it (band crosses Nyquist); fs=0.5 Hz totally dead; a 1 s boxcar
sinc-nulls the 1 Hz band centre even at 20 Hz sampling. This **decomposes the
committed `integrating_1hz` two-point result**: sampling *or* integration alone
already kills tracking (over-determined). DG-order methods are the fragile class;
the fixed multitaper F is ~0 everywhere (known zero wander-power), not a boundary.

## Current status
- Branch `feat/phase2-meter-boundary` — **not yet merged / pushed**. Ready for a
  `check-PR` pass, then merge to `main`.
- **A CPU venv now EXISTS at `.venv/`** (gitignored) on the MATS dev node —
  numpy 2.2.6 / scipy 1.15.3 / matplotlib / sklearn from `requirements.txt`. This
  reverses the prior "no local env" state; `.venv` is fine for light CPU work +
  `--smoke`, but the standing rule holds: **full sweeps go to slurm** (job 5498
  was an array, not a dev-node run).
- Artifacts: `results/st2/meter_boundary_summary.json` (tracked),
  `figures/st2_meter_boundary*.{pdf,png}` (tracked); per-cell raws + slurm logs
  gitignored (`results/st2/meter_boundary_raw/`, `results/st2/slurm/`).

## Gotcha discovered — byte-identity fixtures are BLAS-sensitive (NOT my bug)
`pytest -m "not gpu"` → **5 failures, 171 passed**. The 5 are pre-existing and
environmental (verified: they fail identically on base commit `f43aed2` in this
same venv): `test_ko_workload` prechange-digest / byte-identity guards and their
`test_deperiod` / `test_ko_synth_meter` cousins. The stored digests were made in
the source monorepo's environment; this freshly-built OpenBLAS produces
bit-different floats → digest mismatch. The old "4 sklearn failures" gotcha is
gone (sklearn now installed). **Before the number-freeze pass, regenerate these
digest fixtures in the canonical environment (or pin the exact BLAS).** My 15 new
tests pass; all 171 non-digest tests pass.

## Next steps
1. **`check-PR` + merge `feat/phase2-meter-boundary` → main.** Self-contained;
   no paper prose touched.
2. **Wire the boundary into `paper/main.tex`** (§6 frontier / a new "minimum
   meter specification" subsection) — deferred to Phase 4 write-up; the figures
   (`st2_meter_boundary*`) and the findings note are ready.
3. **Remaining Phase 2 sub-tasks** (this session did only the meter sweep):
   - **Frontier at full resolution** with the systems-cost anchors (denser
     St2Params level grids + larger n_each; existing harness; slurm). Subsumes the
     standing ST2 number-freeze re-verification.
   - **Rung 2** — the big one, currently pure scaffold: physics feature vector,
     three decision rules, semantic-decoy generators (periodic inference,
     gradient-only, discarded-update, non-ML kernel loop, controller-cycle,
     async, co-resident), transfer matrix, §5 prose. Inference null + two physics
     detectors already exist to build on.
4. **Slurm number-freeze pass** (still open): ST1 surrogate S=999/M=10⁴ +
   from-scratch ST2 re-verification. Regenerate the byte-identity digests here too.
5. **Novelty audit** (still unowned; §1 related-work prerequisite).

## Key file locations
- Harness: `powerladder/typeb/meter_boundary.py`, `scripts/st2_meter_boundary.py`,
  `scripts/plot_st2_meter_boundary.py`, `scripts/slurm/st2_meter_boundary.sbatch`.
- Config: `MeterParams.notch_depth` + `St2MeterBoundaryParams` in
  `powerladder/config.py`; notch blend in `powerladder/observation.py`.
- Reproduce: README §"ST2 — meter-requirement boundary sweep".
- MATS: `compute` partition (252 CPU, free); `/mats-cluster` skill for slurm.
