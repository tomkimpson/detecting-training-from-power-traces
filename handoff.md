# Handoff — 2026-07-24 (Phase 2: NP-optimal LRT ceiling harness built)

## What happened this session
Built the **NP-optimal LRT ceiling for the ST1 bake-off** — the last open Phase-2
item (optional, non-blocking; `tasks.md` §Phase 2). Branch
**`feat/phase2-np-ceiling`** (not merged / pushed).

Method chosen with the user: the **Whittle spectral LRT** (the light option; there is
no closed-form likelihood — both generators are black-box samplers). The ceiling is the
f₀-marginalised Bayes–Whittle LRT between the training and null generators: MC-estimated
class PSDs, a per-f₀ training template bank (drift-pooled, drift-agnostic), score
`logsumexp_k ℓ_tr(x|f0_k) − ℓ_neg(x)`. Each corpus-free detector is reported as a
fraction of it (`rho_auc`, `rho_tpr`).

- **New code:** `powerladder/typeb/np_ceiling.py` (library), `scripts/st1_np_ceiling.py`
  (driver — reproduces the exact bake-off eval pops by replaying
  `plot_st1_bakeoff.main`'s RNG order and reusing its builders; fits ceilings for the
  inference / structural / controller-only nulls; scores ceiling + all detectors),
  `scripts/plot_st1_np_ceiling.py` (figure), `scripts/slurm/st1_np_ceiling.sbatch`
  (single task, ~10–15 min), `tests/test_np_ceiling.py` (5, pass). `NpCeilingParams`
  added to `powerladder/config.py`. `.gitignore` + `results/st1/slurm/.gitkeep` added.
- **Validation (local, reduced corpus — NOT frozen numbers):** parity guard
  **max |TPR Δ vs bakeoff| = 0** (eval pops byte-identical to
  `results/st1/bakeoff_summary.json`); qualitatively the ceiling holds AUC≈1.0 vs the
  inference null across drift, Viterbi tracks it, the fixed matched filter collapses.
- **Tests:** `pytest -p no:debugging -m "not gpu"` → **5 failed, 186 passed** = the known
  BLAS byte-identity baseline (`byte-identity-fixtures-blas-sensitive`) + 5 new; no new
  failures (no generator code was touched).
- Full record: `notes/results/st1-np-ceiling-findings.md`.

## Current status
- Branch `feat/phase2-np-ceiling` — **uncommitted at time of writing / then committed
  locally; not merged / pushed.** Contains: 1 config edit, 1 new lib, 3 new scripts, 1
  sbatch, 1 test, 1 findings note, `.gitignore`, `tasks.md` + this handoff.
- `tasks.md`: the NP-ceiling item is `[~]` (harness built; freeze pending). All other
  Phase-2 items are `[x]`.
- **No tracked results/figure yet** — `results/st1/np_ceiling_summary.json` and
  `figures/st1_np_ceiling.*` are produced only by the slurm freeze (repo policy: no
  expensive runs on the dev node; the reduced-corpus local runs were deleted).

## Next steps
1. **Freeze on Slurm:** `sbatch scripts/slurm/st1_np_ceiling.sbatch` on MATS `compute`
   (`n_mc=2000`, `f0_n=61`). Verify via `results/st1/slurm/*.out` (sacct may be down —
   gotcha carried from last session). It runs the driver **and** the plotter, so it
   writes both `results/st1/np_ceiling_summary.json` and `figures/st1_np_ceiling.*`.
2. **Record frozen numbers** in `notes/results/st1-np-ceiling-findings.md` (the ρ values;
   the drift where tracking detectors approach/beat the Whittle ceiling) and tick
   `tasks.md` `[x]`.
3. **`check-PR` + merge `feat/phase2-np-ceiling` → main.** No paper prose / bib touched,
   so `check-refs`/`check-arxiv` not needed for this branch.
4. **Phase 4 write-up** now has every Phase-2 number frozen (frontier, meter-boundary,
   Rung 2) plus this ceiling: wire `paper/main.tex` §6 + the Related-Work "optimality
   ceiling" hook (`paper/main.tex:271-273`) to the NP-ceiling result; Rung-2 subsection;
   minimum-meter-spec subsection; frontier §6.
5. **Still-open Phase 1 number-freeze (separate, blocking a paper freeze):** the ST1
   surrogate S=999/M=10⁴ slurm leg (check job status; do NOT run sweeps on the dev node).

## Key file locations
- Library: `powerladder/typeb/np_ceiling.py` (`band_periodogram`, `mean_periodogram`,
  `build_training_bank`, `fit_ceiling`, `WhittleCeiling.score`, `score_ceiling`).
- Driver/plot/sbatch: `scripts/st1_np_ceiling.py`, `scripts/plot_st1_np_ceiling.py`,
  `scripts/slurm/st1_np_ceiling.sbatch`.
- Reused unchanged: `scripts/plot_st1_bakeoff.py` (eval builders + `_score`/`_tpr_at_far`,
  imported for parity), `powerladder/typeb/roc.py`, `powerladder/typeb/ko_synth.py`.
- Config: `NpCeilingParams` in `powerladder/config.py` (`DEFAULT.np_ceiling`).
- Findings: `notes/results/st1-np-ceiling-findings.md`.

## Gotchas
- **slurm only** for the freeze — never the dev node (auto-memory
  `expensive-sweeps-via-slurm`); the ~10–15 min bank build counts as expensive.
- The driver **imports `plot_st1_bakeoff`** (adds `scripts/` to `sys.path`); importing
  it runs `apply_house_style()` under Agg — harmless. Parity depends on the bake-off's
  RNG order being unchanged; if `plot_st1_bakeoff.main` is ever edited, re-check the
  parity guard (`max |TPR Δ| = 0`).
- Local `--smoke` writes only to `*_smoke_*` (gitignored) and a `st1_np_ceiling_smoke`
  figure — never the tracked files. A non-`--smoke` local run WOULD write the tracked
  `np_ceiling_summary.json`; that's why the freeze belongs on slurm.
- The 5 BLAS failures are the pre-existing environment baseline, not this branch.
