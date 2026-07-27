# detecting-training-from-power-traces

*What a passive power meter can certify about AI training.*

A theory-and-methods study of what a verifier can (and cannot) certify about AI
**training** from a **time-resolved external power trace**, for compute governance.
The work is organised as a **claim ladder** — a sequence of claims, each answering a
strictly more governance-relevant question at a stated cost in verifier information —
with the **de-periodicisation detection–utility frontier** as its central result.

The paper (`paper/main.tex`) is *"What a Passive Power Meter Can Certify About AI
Training."* Its scope is fixed by [`spec.md`](spec.md); the detailed plan lives in
[`notes/plans/plan-for-paper-2.md`](notes/plans/plan-for-paper-2.md) and its
[review](notes/plans/plan-for-paper-2-review.md).

This repo was spun out of the `analogue-sensors-for-ai-verification` monorepo (Paper 2
of that programme). All evidence here is on a **literature-parameterized scenario
model** and is **CPU-only and synthetic**; real hardware enters only as a negative
transport case and a set of measured cost anchors (see `data/`).

## Install

```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.10 recommended (versions pinned in `requirements.txt`).

## Test

```
pytest -p no:debugging -m "not gpu"
```

(`-p no:debugging` is retained from the source project. `scikit-learn` is required for
the RF bake-off baseline tests; without it those 4 tests error out and the rest pass.)

## Reproduce the results and figures

Everything under `results/` and `figures/` is regenerable. From the repo root:

**ST2 — de-periodicisation frontier (fast, ~2 min):**
```
python scripts/plot_st2_sweeps.py                                    # -> results/st2/*_summary.json, figures/st2_*
python scripts/plot_st2_frontier.py --b2-dir data/measured_cost_anchors
    # -> results/st2/frontier_summary.json (verdict: GO, provisional=false), figures/st2_frontier.*
```

**ST2 — meter-requirement boundary sweep ("minimum meter specification", slurm):**
```
python scripts/st2_meter_boundary.py --list          # cell count for the array range
sbatch --array=0-<N-1> scripts/slurm/st2_meter_boundary.sbatch   # compute partition
    # -> results/st2/meter_boundary_raw/*.json + meter_boundary_summary.json
python scripts/plot_st2_meter_boundary.py            # -> figures/st2_meter_boundary*.pdf
```
Single-node fallback: `python scripts/st2_meter_boundary.py --jobs 8`. This is an
expensive sweep — run it on slurm, not the dev node (see `scripts/slurm/`).

**ST1 — adaptive detector bake-off & calibration:**
```
python scripts/plot_st1_bakeoff.py        # -> figures/st1_bakeoff.*, results/st1/bakeoff_summary.json
python scripts/st1_far.py                 # runs the FAR harness -> results/st1/raw/*.npz + far_summary.json
python scripts/plot_st1_calibration.py    # consumes results/st1/raw/*.npz -> figures/st1_calibration_*
python scripts/st1_sweeps.py              # -> results/st1/sweeps/*.json
```

`plot_st1_calibration.py` needs the per-cell `results/st1/raw/*.npz` that `st1_far.py`
produces, so run `st1_far.py` first. The full surrogate FAR grid is the only expensive
step (minutes locally). The frozen surrogate confirmation ran as a slurm array on the
MATS `compute` partition at the surrogate sizing M=2000 × S=999 (the analytic /
level-safety / bake-off cells are at M=10⁴); see `notes/results/number-freeze-2026-07-24.md`:
```
python scripts/st1_far.py --calibs surrogate --stages stage1,stage3,stage4 \
    --nulls white,ar1,ar2_resonant,ar1_t,lognormal,sq_gauss,tvar,controller --list  # -> "24 cells"
sbatch --array=0-23 scripts/slurm/st1_far_surrogate.sbatch
python scripts/plot_st2_sweeps.py --list                  # -> "9 families"
sw=$(sbatch --parsable --array=0-8 scripts/slurm/st2_sweeps.sbatch)
sbatch --dependency=afterok:$sw scripts/slurm/st2_frontier.sbatch
```

**ST1 — NP-optimal Whittle-LRT ceiling (Phase 2, optional; slurm):**
```
sbatch scripts/slurm/st1_np_ceiling.sbatch   # -> results/st1/np_ceiling_summary.json + figures/st1_np_ceiling.*
python scripts/plot_st1_np_ceiling.py         # re-draw the figure from the summary
```
The single ~10–15 min task builds the shared Whittle template bank, fits the three
ceilings, and reports each bake-off detector as a fraction of the NP-optimal power. Local
use is `python scripts/st1_np_ceiling.py --smoke` only (repo policy: freezes on slurm).

## Build the paper

```
cd paper && latexmk -pdf main.tex
```

The manuscript is a scaffold in active development: each section opens with a red TODO
checklist that is deleted before submission.

## Layout

- `powerladder/` — library: `st1/` (adaptive structural detector), `typeb/` (detector
  bank + ST2 frontier), `ko_workload.py`, `observation.py`, `config.py`.
- `scripts/` — the six figure/experiment entry points above.
- `results/`, `figures/` — regenerable outputs.
- `data/measured_cost_anchors/` — measured GPU throughput anchors (static; see its README).
- `notes/` — plan, review, ST1 findings, background memo.
- `paper/` — the LaTeX manuscript.

See [`CLAUDE.md`](CLAUDE.md) for development conventions.
