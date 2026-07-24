# Number-freeze pass — ST1 surrogate + ST2 re-verification

**Date:** 2026-07-24 · **Status:** DONE. ST1 surrogate leg frozen at full sizing;
ST2 frontier re-verified from scratch. Both Phase-0 follow-ups in `tasks.md`
closed. One caveat surfaced (`sq_gauss`/stage4, see below); GO verdict stands
(confirmed-with-caveat, author decision 2026-07-24).

This is the slurm number-freeze pass the ST1 gate verdict
(`notes/results/st1-findings.md`, 2026-07-22 GATE VERDICT) and the ST2 frontier
required before any paper number is frozen. All runs on the **MATS `compute`
partition**, single-threaded OpenBLAS, crash-safe per-cell persistence.

## What ran

| job | array | what | sizing |
|---|---|---|---|
| 5567 | 0–23 | `st1_far.py --calibs surrogate` stages 1/3/4 × 8 nulls | M=2000 × S=999 |
| 5565 | 0–8  | `plot_st2_sweeps.py` one family/task (9 families) | n_each=200, seed 0 |
| 5566 | —    | `plot_st2_frontier.py --b2-dir data/measured_cost_anchors` | assembly |

All tasks COMPLETED. Wall: the fully-adaptive stage-4 surrogate cells dominate
(stage4/white ≈ 6.6 h for M=2000×S=999); ST2 families ≈ minutes each. Harness:
`--list`/`--array-id` added to both driver scripts (mirroring
`scripts/st2_meter_boundary.py`); sbatch drivers `scripts/slurm/st1_far_surrogate.sbatch`,
`st2_sweeps.sbatch`, `st2_frontier.sbatch`. CPU venv created at `.venv/`
(reconciled the `venv`→`.venv` mismatch across the sbatch/README).

## ST1 — frozen surrogate FAR (realised at nominal 0.05 / 1e-2)

| null (stationary) | stage1 | stage3 (held-out warp) | stage4 (fully adaptive) |
|---|---|---|---|
| white           | 0.051 / 0.009 (KSp 0.20) | 0.050 / 0.013 (0.26) | 0.052 / 0.013 (0.79) |
| ar1             | 0.050 / 0.013 (0.57) | 0.046 / 0.009 (0.97) | 0.054 / 0.010 (0.75) |
| ar2_resonant    | 0.044 / 0.010 (0.72) | 0.051 / 0.007 (0.91) | 0.046 / 0.011 (0.53) |
| ar1_t           | 0.039 / 0.005 | 0.049 / 0.005 | 0.037 / 0.005 |
| lognormal       | 0.004 / 0.001 | 0.011 / 0.001 | 0.035 / 0.006 |
| **sq_gauss**    | 0.014 / 0.001 | 0.013 / 0.001 | **0.087 / 0.018** (KSp 1.9e-38) |
| tvar (bdy)      | 0.032 / 0.007 | 0.008 / 0.002 | 0.021 / 0.002 |
| controller (bdy)| 0.172 / 0.060 | 0.011 / 0.001 | 0.297 / 0.100 |

Source: `results/st1/far_summary.json` (surrogate cells, all M=2000).

### Reading

- **Stationary Gaussian nulls (white, ar1, ar2_resonant) are exact at all three
  stages** including the fully-adaptive stage 4 (realised ≈ nominal, KS-uniformity
  not rejected). This is the core of the GO reframe's leg (b) and it now holds on
  a full grid, not the partial M=1000/S=199 one.
- **Heavy-tailed but *linear* non-Gaussian nulls (ar1_t, lognormal) stay
  level-safe (conservative)** at every stage — realised FAR at or below nominal.
- **`sq_gauss` at stage 4 is anti-conservative: 0.087 / 0.018 ≈ 1.7× nominal**
  (~7σ over 0.05; KS p=1.9e-38 — real, not MC noise). It is fine (conservative)
  at stages 1 and 3. Strictly this bears on the **GO-honest-reframe requirement**
  (surrogate calibration holds at 0.05/1e-2 on the stage-5/6 nulls), which
  `sq_gauss`/stage4 fails — not on the pre-registered **NO-GO**, whose 3× clause is
  scoped to stationary *Gaussian/coloured* nulls and so does not bind a nonlinear
  non-Gaussian null. Even so, applying the stricter NO-GO bar is conservative and it
  does not fire: the inflation is **below 3×**, and NO-GO additionally requires
  inflation *under both calibrations*, which the analytic dg_bartlett arm never shows.
  We therefore narrow the level-safety claim rather than revisit the verdict.
  - **Mechanism:** the Fourier-phase surrogate encodes a *linear*-stationary null.
    A squared-Gaussian process is *nonlinear* (quadratic phase coupling / non-zero
    bispectrum); the surrogate cannot reproduce that structure and so under-spreads
    the null. Same failure mode as the cyclostationary `controller` (0.17–0.30, the
    attribution boundary), just far milder.
  - **Resolution (author decision 2026-07-24): GO stands, confirmed-with-caveat.**
    The level-safety claim is scoped to **linear-stationary** nulls; the paper §4
    footnote now states the `sq_gauss`/stage-4 sub-3× exception and its cause.
    Candidate follow-up if ever needed: amplitude-preserving (IAAFT) surrogates.
- **tvar / controller (stage-7, locally non-stationary) inflate as expected** —
  the attribution boundary, unchanged from the partial grid and consistent with
  the analytic characterisation.

## ST2 — frontier re-verification

Regenerated all nine per-family summaries + `frontier_summary.json` from scratch.
**Verdict reproduces exactly at the decision level:** `provisional: false`,
`GO`, supporting run `work=0.35 / 0.5` (best tracking TPR@0.05 ≥ 0.8 while every
fixed test ≤ 0.5); detector classes unchanged (tracking = viterbi / dg_order_full
/ dg_order_semicoh; fixed = spectral / mtf). Spot values identical to the
committed frontier: `work` viterbi TPR@0.05 = 1.0 through level 0.5;
`meter/integrating_1hz` = 0.320, `meter/notch_at_cadence` = 0.640. Per-family
JSON/figure byte-diffs are last-digit AUC float churn (BLAS build), not decision
changes. The `meter_boundary` sweep (prior Phase-2 deliverable, job 5498) was
**not** re-run.

**B / work-jitter line-band caveat (out of scope here):** the `B`
power-management limit-cycle and the work-jitter *line-band detection* numbers
flagged in `issue54-investigation.md` live in the **measured b2 campaign**, not
this repo. The synthetic `work` family reads only the measured *cost anchor*, so
this pass re-freezes the synthetic frontier evidence; the B-contamination caveat
is recorded, not resolved. Which finding leads the frontier remains the
positioning-memo open question.

## Environment / reproducibility

- `pytest -m "not gpu"` baseline on the cluster `.venv`: **5 failed, 171 passed** —
  the known BLAS-build-sensitive byte-identity digest tests (env, not a bug). The
  frozen numbers here are FAR/AUC *level statistics*, not byte-identity digests,
  so the BLAS caveat does not invalidate them.
- Reproduce: `sbatch --array=0-23 scripts/slurm/st1_far_surrogate.sbatch`;
  `sw=$(sbatch --parsable --array=0-8 scripts/slurm/st2_sweeps.sbatch); sbatch --dependency=afterok:$sw scripts/slurm/st2_frontier.sbatch`.
