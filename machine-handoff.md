# Machine handoff

_Packed 2026-10-01 on Tom's MacBook. To resume elsewhere: `/switch-machine resume`._

**Destination: OzSTAR** (`nt.swin.edu.au`, user `tkimpson`), to run a B2 A100 cost-anchor top-up.
The GPU work happens in the **monorepo** `analogue-sensors-for-ai-verification`, not in
this repo. This repo receives the resulting anchors. See `handoff.md` → Next steps 1–5.

## Environment
- **This repo** (CPU-only): specs `requirements.txt`, `pyproject.toml`, both committed ✓.
  - setup: `pip install -r requirements.txt`; test with `pytest -p no:debugging -m "not gpu"`
  - Not strictly needed on OzSTAR unless you re-run the frontier assembly there.
- **Monorepo** (GPU): `requirements.txt` committed ✓; torch is NOT pinned there.
  - torch comes from the OzSTAR conda env: torch 2.5.1+cu124, Python 3.10,
    `source /fred/oz022/tkimpson/miniconda3/etc/profile.d/conda.sh && conda activate base`.
  - On OzSTAR, the monorepo checkout should be fast-forwarded to `origin/main` (f611730).
    Its B2 code (`code/b2/`, `scripts/run_b2_capture.py`) is unchanged since the original campaign.

## Data
- [x] `results/st2/aggregate/*.json`, `figures/*_aggregate.*` — tracked, travels in git
- [x] `data/measured_cost_anchors/*.json` — tracked, travels in git (these get extended by the top-up)
- [ ] Monorepo raw B2 traces `results/b2/traces/*.npz` — gitignored. **Cluster-resident on
  OzSTAR**, under the monorepo checkout used for the July campaign (likely `/fred/oz022/tkimpson/...`;
  locate with `find /fred/oz022/tkimpson -maxdepth 3 -name analogue-sensors-for-ai-verification`).
  The top-up doesn't need the old traces; each new trace calibrates its own honest baseline.
- [ ] `figures/money_pareto_aggregate_hull.*` — untracked comparison figure (Tom kept the staircase).
  Regenerable, won't travel.
- [ ] Tom's WIP `scripts/plot_scenario_aggregate.py`, `figures/scenario_aggregate.*` — untracked,
  **won't travel** (deliberately not committed).

## Cluster access
- OzSTAR: `ssh nt.swin.edu.au`. Key auth from the laptop is rejected, so log in interactively.
  - GPU partition `milan-gpu`, `--gres=gpu:a100:1`, `--account=oz022`
    (template: monorepo `scripts/run_b2_capture.slurm`).
- MATS (not for this task): `ssh 185.141.218.207`. Elastic GPUs are paid and need Tom's approval.

## Regeneration
- Aggregate frontier after new anchors land (cost join only, seconds):
  `python scripts/plot_st2_frontier.py --st2-dir results/st2/aggregate --fig-name st2_frontier_aggregate --b2-dir data/measured_cost_anchors`
  then `python scripts/plot_money_pareto.py --summary results/st2/aggregate/frontier_summary.json --far 0.01 --out money_pareto_aggregate`
  and `python scripts/plot_st2_pareto.py --summary results/st2/aggregate/frontier_summary.json --far 0.01 --out st2_cost_pareto_aggregate`.

## To resume
Run `/switch-machine resume` on OzSTAR, in a clone of this repo
(`gh repo clone tomkimpson/detecting-training-from-power-traces`, then
`git checkout feat/frontier-aggregate`).
