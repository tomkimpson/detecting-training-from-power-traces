# Machine handoff

_Packed 2026-10-01 on OzSTAR (farnarkle2). To resume elsewhere: `/switch-machine resume`._

**Destination: Tom's MacBook.** The GPU top-up is done; the remaining work (figure choice, text review, merges) is local.

## Environment
- This repo: `requirements.txt`, `pyproject.toml` committed ✓. setup: `pip install -r requirements.txt`; test: `pytest -p no:debugging -m "not gpu"`.
- Paper: rebuild `paper/main.pdf` with the laptop TeX Live (the committed PDF was built on OzSTAR):
  `cd paper && SOURCE_DATE_EPOCH=1785110400 FORCE_SOURCE_DATE=1 latexmk -f -pdf main.tex`.

## Data
- [x] `data/measured_cost_anchors/*.json` (extended), `results/st2/aggregate/frontier_summary.json`, `figures/money_pareto_aggregate{,_symlog,_xmax}.*` are tracked and travel in git.
- [ ] The 48 new raw B2 traces (`results/b2/traces/*.npz`, gitignored) are **cluster-resident** on OzSTAR. They aren't needed locally: the summaries are committed in the monorepo (`86b412a`).
- [ ] Tom's WIP `scripts/plot_scenario_aggregate.py` and `figures/scenario_aggregate.*` are still on the MacBook only.

## Cluster access
- OzSTAR: `ssh nt.swin.edu.au` (interactive login). Monorepo traces are at
  `/fred/oz022/tkimpson/analogue-sensors-for-ai-verification/results/b2/traces/`.
  The OzSTAR monorepo checkout is on `feat/b2-offgrid-topup`.

## To resume
`git pull` on `feat/frontier-aggregate`, then `/switch-machine resume`.
