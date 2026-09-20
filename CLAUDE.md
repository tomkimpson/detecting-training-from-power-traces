# CLAUDE.md — certifying-training-from-power

Paper 2 of the AI-compute-verification programme: *"What a Passive Power Meter Can
Certify About AI Training."* Spun out from the `analogue-sensors-for-ai-verification`
monorepo on 2026-07-22 as a clean, single-paper repo.

## Session Startup (NON-NEGOTIABLE)

1. **Read `handoff.md` first.** Current status, open questions, blockers, next steps,
   and the low-priority backlog.
2. **Skim the top entry of `log.md`** — dated narrative of past sessions (newest at top).
3. **Read `spec.md`** — the project scope and source of truth.
4. **Check current branch:** `git branch --show-current`. If on `main`, create a
   feature branch before making any changes.

## Source of Truth

- **`spec.md`** completely defines the project scope. Refer to it when uncertain.
  Never modify it without explicit user approval.
- **`notes/plans/plan-for-paper-2.md`** + **`plan-for-paper-2-review.md`** are the detailed
  realisation of the spec (thesis, claim ladder, Phase 0–4 plan). `spec.md` cross-links
  them; rationale lives there, scope lives in `spec.md`, status lives in `handoff.md`.
- **`handoff.md`** tracks session-to-session context and the open backlog. Read at
  start, update at end. (It replaced `tasks.md`, retired 2026-07-27 once the Phase 0–4
  tracker was fully closed out; the narrative record is in `log.md`.)

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
├── notes/          # plans/ + results/ + discussion/ — every note in exactly one (rules: notes/CLAUDE.md)
├── paper/          # LaTeX manuscript (ICML 2026 style, [preprint] mode)
├── figures/        # Generated figures (st1_*, st2_*)
├── results/        # Experiment outputs (results/st1, results/st2)
├── tests/          # pytest suite
├── spec.md handoff.md log.md CLAUDE.md README.md
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
  **slurm** — this laptop has no GPU. Two clusters are available:
  - **MATS (preferred)** — use this by default; the OzSTAR queue can get busy.
    On the MATS cluster the **`/mats-cluster`** skill is available for launching and
    managing GPU/slurm jobs (sbatch/srun/squeue, partitions, elastic GPUs, storage) —
    use it for cluster work.
  - **OzSTAR** — fallback; most of our work so far has run here.

## Paper

- `paper/main.tex`, ICML 2026 two-column style in `[preprint]` mode (matches the sibling
  paper `power-to-flops`). `icml2026.{sty,bst}`, `fancyhdr.sty` and `algorithm{,ic}.sty`
  are vendored in `paper/`. Figures via `\graphicspath{{../figures/}}`; body figures are
  `figure*` because they are generated at 5.1 in and would be illegible in a 3.25 in column.
- Build: `cd paper && SOURCE_DATE_EPOCH=1785110400 FORCE_SOURCE_DATE=1 latexmk -f -pdf main.tex`.
  The env vars pin the embedded `/CreationDate` so the build is byte-reproducible.
  Use a real fixed date, **not 0** — the `[preprint]` footer prints `\today`, so epoch 0
  renders "Preprint. January 1, 1970."
  `paper/main.pdf` is **tracked**, so rebuild and commit it whenever `main.tex` changes.
  Bibliography: `paper/references.bib`, author–year via `icml2026.bst`.
- The manuscript is **fully drafted** (29 pp, 0 errors/overfull/undefined). All section
  scaffolds and the `checklist`/`todo` machinery were removed on 2026-07-27. `check-refs`
  and `check-arxiv-llm-compliance` both pass except for one known blocker: the third
  author's affiliation still renders as "Affiliation TBD".

## Session Handoff (NON-NEGOTIABLE)

Before ending any session, update `handoff.md`: what was accomplished, current status
and blockers, what to do next, key file locations, gotchas. The next session depends
on it.
