# Handoff — 2026-07-22 (repo spun out from the monorepo; clean start)

## What happened this session
Paper 2 was **spun out** of the `analogue-sensors-for-ai-verification` monorepo into
this fresh, standalone repo. The monorepo hosted five manuscripts on one shared `code/`
package, which made clean development hard. This repo carries only the paper-2 assets.

- Package `code/` → **`powerladder/`** (renamed to kill the stdlib-`code` shadow).
  Brought: `st1/` + `typeb/` (whole), `ko_workload.py`, `observation.py`, plus shared
  primitives `config/forward/noise/plotstyle`. Left behind: the GPU bench harness
  (`bench/`, `b2/`), the β/energy-accounting code (`floor*`, `inversion`, `measured`,
  `scenarios`), and the Kalman `tracker.py` — none are imported by paper-2 code.
- Import rewrite `from code.` → `from powerladder.` applied to package/scripts/tests.
- Manuscript `paper-ladder/` → `paper/`; `\graphicspath` fixed to `{{../figures/}}`.
- Measured GPU cost anchors moved to `data/measured_cost_anchors/` (+ provenance
  README); consumed by the frontier via its existing `--b2-dir` flag (no code edit).
- Fresh scaffolding written: `CLAUDE.md`, `spec.md`, `tasks.md`, `log.md`, `README.md`,
  `requirements.txt` (trimmed — no torch/nvidia-ml-py), `pyproject.toml`, `.gitignore`.

## Current status — VERIFIED GREEN
- `pytest -p no:debugging -m "not gpu"`: **165 passed, 4 failed**. The 4 failures are all
  `test_rf_baseline.py` → `ModuleNotFoundError: sklearn` (this local base env is Python
  3.13 without sklearn). This is the **exact** pre-existing environmental gap the
  monorepo documented — not a spin-out breakage. `pip install scikit-learn` (it's in
  `requirements.txt`) makes them pass.
- Import closure clean (`python -c "import powerladder.st1.pipeline, ..."`).
- ST2 sweeps + frontier regenerate from scratch → **verdict GO, provisional=false**,
  supporting cells `work=0.35/0.5` (matches the monorepo's definitive result).
- `plot_st1_bakeoff.py` regenerates.
- `plot_st1_calibration.py` needs `results/st1/raw/*.npz` produced by running
  `st1_far.py` first — same as the monorepo (raw npz are intermediates, never committed;
  the committed `far_summary.json` + copied calibration figures are present).

## Next steps
1. **Create the GitHub remote and push** (not yet done this session — see below).
2. Start **Phase 1** (Rung 1 full): qualified F-test + tracked-cyclostationary on the
   scenario model, both adversary settings, pre-registered bake-off → paper §4. See
   `tasks.md`.
3. Carry over the **OzSTAR surrogate confirmation** follow-up before freezing any paper
   numbers (`tasks.md` Phase 0 follow-ups).

## Git status
Repo initialised locally with a single clean initial commit on `main`. GitHub remote
`tomkimpson/certifying-training-from-power` (private) + push: confirm done in the next
session if this session ended before it completed.

## Gotchas / technical notes
- Run pytest with `-p no:debugging` (retained from the source project's invocation).
- The full surrogate FAR grid is the only expensive step; run it as a slurm array on
  OzSTAR for the definitive S=999, M=10⁴ numbers.
- `config.py` still carries dead monorepo params (`B2Params`, `BenchConfig`, β params) —
  harmless; trimming is a low-priority `tasks.md` item.
- The monorepo is **untouched** (copy-only). Its history remains the provenance record
  for everything here.
