# Handoff — 2026-07-24 (Phase 4: frozen Phase-2 results wired into paper/main.tex)

## What happened this session
Wired **all four frozen Phase-2 results** into `paper/main.tex` on branch
**`wire-in-phase2-results`** (not committed / pushed yet). This converts the §6/§7/§9/§10
red-checklist scaffolds into written prose and lands the frozen numbers + existing figures.
No new experiments, no new plots — every figure was already on disk (both `.pdf`/`.png`).

Sections written (checklists + `\todo`s deleted per convention):
- **§6 `sec:frontier`** — body led by the ≈zero-cost variable-real-work attack (resolves the
  old "which finding leads the frontier" open question). `work=0.35/0.5` at −0.10%/−0.36%
  cost: Viterbi TPR@0.05 = 1.0 while fixed spectral collapses to 0.26/0.27 (AUC 0.44/0.41);
  Viterbi bends only at work=0.7 (0.59, unmeasured cost). Wired `fig:frontier` + `fig:st2_work`.
  issue-54 non-impact noted (clean `throughput_overhead` anchor only).
- **§6 new subsection `subsec:meter_spec`** — minimum meter specification (fs ≥ ~2 Hz AND
  integ ≲ 0.5 s AND no deep in-band notch), decomposes the over-determined `integrating_1hz`
  point. Wired `fig:meter_boundary` + `fig:meter_boundary_notch`. Framed as a channel
  requirement / results subsection — **spec.md and the intro contributions list untouched**
  (per user decision; NOT promoted to a named contribution).
- **§7 `sec:classification`** — 8-feature vector; three rules; STATED/TRANSFER AUC 1.00;
  five non-training controls score as training (semantic-decoy ceiling); async reported as a
  **robustness result** (physics AUC 0.89, 72%) with the genuine boundary cited from §6.
  Wired `fig:rung2_stated` + `fig:rung2_controls` + `tab:rung2_controls`.
- **App A `app:bakeoff`** — NP-optimal ceiling paragraph + `fig:np_ceiling`: perfect Whittle
  ceiling → all difficulty in the detector; Viterbi ≈ NP-optimal vs inference; only the DG
  order family approaches the ceiling on confusers; conservative-lower-bound caveat. Plus a
  forward `\cref{app:bakeoff}` from the Related-work optimality-ceiling hook and a §6 sentence.
- **§9 `sec:reality`** — negative transport case as a result (~0.4% ripple, 0.31–0.45 Hz
  controller limit cycle, ~75× modulation gap) + falsifiable distributed-scale predictions.
  **No figure** — the measured A100 captures are not in this repo; not fabricated.
- **§10 `sec:discussion`** — ladder close-out table `tab:closeout` (evidence per rung) + scope
  + limitations + governance reading.

All numbers spot-checked against the JSONs (work family, meter-boundary decomposition,
np-ceiling ρ, rung2 async) — match exactly (one rounding fixed: 0.295 → 0.30).

## Current status
- Branch `wire-in-phase2-results` — **edits made, NOT committed / pushed.** Changed files:
  `paper/main.tex` (7 edits), `tasks.md` (Phase-2 markers flipped to WIRED; Phase-4 first item
  → `[~]`), this `handoff.md`. Untracked pre-existing file `PR_REVIEW_feat-phase2-meter-boundary_2026-07-23.md`.
- Static verification PASSED: checklist envs 4/4 balanced (remaining 4 are the intentionally
  out-of-scope §1/§3/§4/App B scaffolds); all figure/table/tabular/equation envs balanced;
  every `\includegraphics` stem exists on disk; all 30 `\cref`/`\ref` targets resolve; 0 stale
  `results/b2` refs. **PDF not built** (no TeX toolchain on the dev node).

## Next steps
1. **Commit** the branch, then `check-PR` before merge. **`check-refs` /
   `check-arxiv-llm-compliance` still owed** over all prose written this session (no NEW bib
   entries were added — all cite keys were already present).
2. **Remaining Phase-4 write-up:** identifiability theory section (App B `app:identifiability`
   still a scaffold, rigor per plan §10 Q2); the **cost-vs-hiding Pareto re-plot** (new plot
   script over the frozen frontier — `tasks.md` Phase 4); and the still-scaffolded §1 intro,
   abstract, §3 threat-model, §4 scenario-models (with their two example-trace / sensitivity
   figures still unmade).
3. **Still-open Phase 1 number-freeze:** the ST1 surrogate S=999/M=10⁴ leg was flagged DONE
   (array 5567) in the prior handoff — confirm the §4 footnote wording matches before a paper freeze.

## Key file locations
- Manuscript: `paper/main.tex` (§6=647+, §7=~800, App A bake-off + ceiling near end).
- Frozen results: `results/st2/frontier_summary.json`, `.../meter_boundary_summary.json`,
  `results/st1/np_ceiling_summary.json`, `results/rung2/rung2_summary.json`.
- Findings narratives: `notes/results/st2-frontier-freeze-findings.md`,
  `st2-meter-boundary-findings.md`, `st1-np-ceiling-findings.md`, `rung2-findings.md`,
  `overall-findings-does-it-work.md`.
- Figures wired: `figures/{st2_frontier,st2_work,st2_meter_boundary,st2_meter_boundary_notch,
  rung2_stated_transfer,rung2_controls,st1_np_ceiling}.{pdf,png}`.

## Gotchas
- Scope was deliberately bounded: §1/§3/§4/App B scaffolds + abstract were left as-is (broader
  Phase-4 write-up). Do not read their remaining checklists as regressions.
- §9 has no figure on purpose — the measured single-A100 captures live in the source
  campaign, not this repo. Don't wire a fabricated one.
- The minimum-meter-spec is a §6 results subsection, NOT a named contribution — promoting it
  would need spec.md approval (see `tasks.md` meter-boundary item).
