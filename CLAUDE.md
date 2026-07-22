# CLAUDE.md — certifying-training-from-power

Paper 2 of the AI-compute-verification programme: *"What a Passive Power Meter Can
Certify About AI Training."* Spun out from the `analogue-sensors-for-ai-verification`
monorepo on 2026-07-22 as a clean, single-paper repo.

## Session Startup (NON-NEGOTIABLE)

1. **Read `handoff.md` first.** Current status, open questions, blockers, next steps.
2. **Read `tasks.md`** — the phased task tracker with a one-line **Result** under each
   task as it completes.
3. **Skim the top entry of `log.md`** — dated narrative of past sessions (newest at top).
4. **Read `spec.md`** — the project scope and source of truth.
5. **Check current branch:** `git branch --show-current`. If on `main`, create a
   feature branch before making any changes.

## Source of Truth

- **`spec.md`** completely defines the project scope. Refer to it when uncertain.
  Never modify it without explicit user approval.
- **`notes/plan-for-paper-2.md`** + **`plan-for-paper-2-review.md`** are the detailed
  realisation of the spec (thesis, claim ladder, Phase 0–4 plan). `spec.md` cross-links
  them; rationale lives there, scope lives in `spec.md`, status lives in `tasks.md`.
- **`handoff.md`** tracks session-to-session context. Read at start, update at end.
- **`tasks.md`** is the phased task tracker. Keep it current as you work.

## Version Control

- **Commit early and often**, small commits, clear messages. Use `/commit`.
- **Never work directly on `main`.** Feature branches: `feat/…`, `fix/…`, `refactor/…`.
- **Merge to `main`** only when a feature is complete and tested.

## Project Structure

```
certifying-training-from-power/
├── powerladder/    # Importable library (the paper-2 detection/scenario code)
├── scripts/        # Standalone entry points (run directly)
├── data/           # Static input data (e.g. measured cost anchors)
├── notes/          # Plan docs, findings, research notes
├── paper/          # LaTeX manuscript (arxiv preprint style)
├── figures/        # Generated figures (st1_*, st2_*)
├── results/        # Experiment outputs (results/st1, results/st2)
├── tests/          # pytest suite
├── spec.md handoff.md tasks.md log.md CLAUDE.md README.md
```

### The `powerladder/` package
- `ko_workload.py` — Ko & Zhu workload scenario model (+ our inference null and attack knobs).
- `observation.py` — the off-chip observation/meter map (`MeterParams`, `apply_meter`).
- `config.py` — central parameter dataclasses (`DEFAULT`, `St1*`, `TypeBParams`, `Ko*`, `St2Params`, …).
- `forward.py`, `noise.py` — simulation and noise primitives.
- `st1/` — the ST1 adaptive structural detector: multitaper F-test, Dandawaté–Giannakis
  cyclostationary statistic, phase resampling, EST/TEST splitting, surrogates, FAR harness.
- `typeb/` — detector bank + the ST2 de-periodicisation frontier (`st2`, `st2_attacks`,
  `deperiod`), plus scenario synthesis, gate/ROC, and the RF bake-off baseline.
- `plotstyle.py` — house SciencePlots figure style.

### scripts/ vs powerladder/
- **`scripts/`** = standalone entry points you run directly (`st1_far.py`,
  `st1_sweeps.py`, `plot_st1_*`, `plot_st2_*`).
- **`powerladder/`** = reusable library imported by scripts and tests.

## Reproducibility (FIRST-CLASS CONCERN)

1. **Every figure regenerable** from a script in `scripts/`. No hand-made figures.
2. **Every dataset has provenance.** `data/measured_cost_anchors/` are measured GPU
   throughput anchors carried from the source monorepo's B2 single-A100 campaign —
   **not regenerable here** (see `data/measured_cost_anchors/README.md`); committed as
   static input.
3. **No magic numbers.** Parameters live in `powerladder/config.py` or at the top of scripts.
4. **Pin dependencies** — `requirements.txt`.
5. **Document reproduction commands** in `README.md`.

## Environment / compute

- The whole paper-2 pipeline is **CPU-only and synthetic**. Install with
  `pip install -r requirements.txt`; run `pytest -p no:debugging -m "not gpu"`.
- `-p no:debugging` is retained from the source project's pytest invocation.
- Any GPU work (learning-efficiency measurements, ST3 challenge pilot) runs on
  **OzSTAR via slurm** — this laptop has no GPU.

## Paper

- `paper/main.tex`, arXiv preprint style (`arxiv.sty`). Figures via `\graphicspath{{../figures/}}`.
- Build: `cd paper && latexmk -pdf main.tex`. Bibliography: `paper/references.bib`.
- The manuscript is currently a **scaffold**: each section opens with a red TODO
  checklist and prose is filled in as Phase 1–4 land. Delete the `checklist`/`todo`
  environments before submission.

## Session Handoff (NON-NEGOTIABLE)

Before ending any session, update `handoff.md`: what was accomplished, current status
and blockers, what to do next, key file locations, gotchas. The next session depends
on it.
