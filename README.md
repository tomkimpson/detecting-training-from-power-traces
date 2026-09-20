# certifying-training-from-power

*Detecting AI training in datacentre power.*

A theory-and-methods study of what a verifier can (and cannot) certify about AI
**training** from a **time-resolved external power trace**, for compute governance.
The work is organised as a **claim ladder** — a sequence of claims, each answering a
strictly more governance-relevant question at a stated cost in verifier information —
with the **de-periodicisation detection–utility frontier** as its central result.

The paper (`paper/main.tex`) is *"Detecting AI Training in Datacentre Power."*
Its scope is fixed by [`spec.md`](spec.md); the detailed plan lives in
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

**Known baseline:** in the pinned environment (`requirements.txt`) a clean clone reports
**228 passed / 0 failed** — verified 2026-07-27 in a fresh venv built from the pins and in
a conda env with numpy 2.4.4 / scipy 1.17.1. Treat any failure as real.

The exception to watch for is the `ko_workload` / `deperiod` byte-identity and digest
fixtures: their reference hashes are sensitive to the BLAS build, so on an unpinned or
differently-built numpy they can fail without anything having regressed. If you see
failures, check them against the pins before treating them as regressions — but do not
assume a fixed number of them is expected.

## Reproduce the results and figures

Everything under `results/` and `figures/` is regenerable. From the repo root:

**ST2 — de-periodicisation frontier.** The committed numbers are a slurm freeze
(`notes/results/st2-frontier-freeze-findings.md`). The first two commands below
**rebuild** `results/st2/*_summary.json` and overwrite that freeze, so run them on slurm
(see the array recipe further down), not on the dev node:
```
python scripts/plot_st2_sweeps.py                                    # -> results/st2/*_summary.json, figures/st2_*
python scripts/plot_st2_frontier.py --b2-dir data/measured_cost_anchors
    # -> results/st2/frontier_summary.json (verdict: GO, provisional=false), figures/st2_frontier.*
```
Only the Pareto re-plot is safe to run locally — it reads `frontier_summary.json` and
writes no results artefact (~1 s):
```
python scripts/plot_st2_pareto.py    # -> figures/st2_cost_pareto.*
```

**Scenario models — example traces and observation-map sensitivity (local, CPU-only,
~5 s each):**
```
python scripts/plot_scenario_traces.py   # -> figures/scenario_traces.*
python scripts/plot_scenario_meter.py    # -> figures/scenario_meter.*
```
The two paper §4 figures. Both generate their own data and write no results artefact,
so they are safe to run locally. `plot_scenario_traces.py` renders the four workload
classes (training, fine-tuning, inference null, aggregate) in time and frequency;
`plot_scenario_meter.py` sweeps each `MeterParams` axis on one fixed honest training
workload and reports how much of the cadence survives the channel.

**Identifiability — covertness cost bound (lightweight CPU check, ~30 s):**
```
python scripts/lower_bound_spike.py
    # -> results/spike/lower_bound_spike.json, figures/lower_bound_spike.*
```
Sanity-checks paper §3 / App. B: the parameter-free coherent-power rolloff
(`kappa = pi^2 f0 T`), the attained advantage `J = max(TPR-FPR) <= TV`, and the
covertness thresholds `sigma*(eps)` read off it with bootstrap intervals.

Seeded and deterministic **within a fixed environment**: run it twice on one machine and
the JSON and both figures come back byte-identical. Across machines they will not, and
that is expected — `requirements.txt` pins numpy but not the BLAS (last-ULP drift in the
JSON floats), the PDF's text metrics depend on which fonts the system resolves for the
SciencePlots `nature` style, and PNG bytes embed the matplotlib version and a
zlib-build-dependent image stream. With fonts matched the PNGs are pixel-identical. So
regenerate-and-diff is a valid check on one machine; across machines compare the
*values*, not the bytes.

n=600 per class: at smaller sizings the H0 floor of `J` (`_ks_null_floor`, mean + 3 sd of
the one-sided KS null) is on the same scale as the epsilons being inverted — 0.25 at
n=80 against 0.093 at n=600 — which lets null points into the fit and destabilises it.

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

`paper/main.pdf` is **tracked**, so you can read the compiled manuscript without a TeX
toolchain. When you change `main.tex`, rebuild and commit the PDF alongside the source:

```
cd paper && SOURCE_DATE_EPOCH=1785110400 FORCE_SOURCE_DATE=1 latexmk -f -pdf main.tex
```

The two environment variables pin the PDF's embedded `/CreationDate`, which makes the
build byte-reproducible — without them every rebuild rewrites ~700 kB of binary with
identical content and the tracked PDF churns on every commit. The epoch is a fixed real
date (2026-07-27) rather than 0, because the ICML `[preprint]` footer prints `\today`:
at epoch 0 it reads "Preprint. January 1, 1970." Bump it when you want the printed date
to move. `-f` carries the build past a first-pass `natbib` notice that resolves once the
bibtex pass has run.

The manuscript uses the ICML 2026 two-column style in `[preprint]` mode; `icml2026.sty`,
`icml2026.bst`, `fancyhdr.sty` and `algorithm{,ic}.sty` are vendored in `paper/`.

Current state: **29 pages, 0 errors, 0 overfull boxes, 0 undefined references or
citations.** All section scaffolds are written; the one item outstanding before
submission is the third author's affiliation, which still renders as "Affiliation TBD".

## Layout

- `powerladder/` — library: `st1/` (adaptive structural detector), `typeb/` (detector
  bank + ST2 frontier), `ko_workload.py`, `observation.py`, `config.py`.
- `scripts/` — the figure/experiment entry points above, plus the slurm harnesses
  in `scripts/slurm/`.
- `results/`, `figures/` — regenerable outputs.
- `data/measured_cost_anchors/` — measured GPU throughput anchors (static; see its README).
- `notes/` — plan, review, ST1 findings, background memo.
- `paper/` — the LaTeX manuscript.

See [`CLAUDE.md`](CLAUDE.md) for development conventions.
