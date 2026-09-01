# Handoff — 2026-08-31 (track-before-detect cast; forward-statistic prototype)

## What happened this session

Answered the question "is there a technique other than Viterbi we should try — can
this be cast as a known signal-analysis problem?", then wrote it up and prototyped
the answer. Branch **`claude/viterbi-alternative-techniques-ns1lpx`**, pushed.

### 1. The cast (write-up)

**`notes/discussion/track-before-detect-forward-statistic.md`** — the full record.
Short version: the Viterbi detector is an instance of **track-before-detect**
(radar/sonar; narrowband case = passive-sonar lofargram line tracking). Under the
HMM the tracker already assumes, Viterbi is the MAP-path *approximation*; the
Neyman–Pearson statistic is the **marginal likelihood — the forward algorithm, sum
over all paths** — at identical cost (`max` → `logsumexp`, penalty → normalised
kernel). They agree when one path dominates and diverge at low per-frame SNR under
heavy wander — the adversarial jitter regime.

Also found: `references.bib` carries the sonar→CW-GW lineage block
(`streit1990frequency`, `suvorova2016hmm`, `bayley2019soap`, `djurovic2011viterbi`)
but **none of the four keys is cited in `main.tex`** — cheap related-work fix.

### 2. The prototype

- `powerladder/typeb/detectors.py`: **`forward_statistic`** + sequential
  **`forward_path_scores`** + `_wander_log_kernel`, mirroring the Viterbi pair
  (same emission map, defaults, detector contract). Additive only.
- `tests/test_typeb.py`: 5 new tests + `forward_statistic` added to the ranking
  parametrisation (§ "forward (sum-over-paths) statistic"). **40/40 pass.**
- Full suite: **229 passed / 5 failed** — the 5 are the known cross-machine BLAS
  byte-identity digest tests (`test_ko_workload` ×3, `test_deperiod`,
  `test_ko_synth_meter`), verified identical on a clean checkout of `main` in this
  container. Not caused by this change; would be 0 in the pinned reference venv.

### 3. Smoke result (scratchpad, NOT frozen)

Amplitude × wander sweep on the B0-local synth generator, n_each=100: forward ≥
Viterbi in **every** cell; gap opens exactly at weak-line + heavy-wander (amp 8,
wander 0.6: AUC 0.94 vs 0.86, TPR@0.05 0.79 vs 0.65; amp 6: AUC 0.82 vs 0.72).
Seed-robust (seeds 1–3: AUC gap 0.06–0.13). Full table in the note §4.

## Current status

- Branch pushed; no PR opened (not requested). **No frozen number changes**: ST1/ST2
  freezes, figures, manuscript, `spec.md` all untouched.
- The forward statistic is a **prototype/bank candidate**, cited in no result. Any
  promotion is a new bank member with its own surrogate calibration + FAR run.

## Next steps

1. **Real evaluation of the forward statistic on the headroom axes** (slurm-scale,
   bake-off harness): structural confusers at high drift and the tracking-class
   ceiling for the σ* covertness bound (`rem:direction` is loose against exactly
   this class). Do **not** evaluate on the plain inference null — Viterbi is
   already at ρ_auc = 1.00 there (np-ceiling note).
2. **Harmonic-comb emissions** (pitch-tracking cast; note §6) — the follow-on that
   targets the controller AUC-0.0 failure; composes with the forward recursion.
3. ~~Cite the lineage block~~ — **DONE (2026-08-31, same branch/PR #20):** the four
   keys are now cited in `subsec:tracked` (tracker lineage sentence) and
   `app:bakeoff` (headroom passage names the forward statistic as future work, no
   result claimed). `main.pdf` rebuilt with the pinned-epoch invocation: 28 pp,
   0 errors / undefined / overfull — but in this session's **container TeX Live,
   not the usual laptop toolchain**, so byte provenance differs (the known
   cross-machine font/metric caveat); rebuild on the laptop if byte-continuity
   with earlier PDFs matters.
4. (Carried) **Pre-submission blocker:** third author's affiliation renders as
   "Affiliation TBD" (`paper/main.tex` author block) — sole `check-arxiv-llm-compliance`
   FAIL.

## Key file locations

- Note: `notes/discussion/track-before-detect-forward-statistic.md`.
- Code: `powerladder/typeb/detectors.py` (forward pair at the end of the
  Viterbi section); tests in `tests/test_typeb.py`.
- Smoke script (scratchpad only, not committed): amplitude × wander AUC/TPR sweep;
  numbers reproduced in the note §4 with seeds.

## Gotchas (carried forward)

- **`scripts/plot_st2_frontier.py` WRITES `frontier_summary.json`** (slurm freeze).
  `plot_st2_pareto.py` is the read-only one. Everything ST1/ST2 runs on slurm;
  `scripts/lower_bound_spike.py` is the one locally-runnable script (~17 s, seeded).
- Spike JSON: quote σ* from `covertness_thresholds` (Youden J), never the
  `_gini` key.
- §3/App B deliberately state what is **not** proved (tracking-class case) — don't
  tidy those hedges away.
- Idle-jitter cost range in prose: **15–375%** (frozen), not 15–680%.
- `forward_statistic` values are **not numerically comparable** to
  `viterbi_statistic` (normalised vs unnormalised wander prior); ROC/surrogate
  calibration only.

## Backlog (low priority, carried forward)

- Trim `powerladder/config.py` of monorepo dead weight (`B2Params`, `BenchConfig`,
  beta params) — grep first; `DEFAULT` is constructed from all of them.
- Optionally make `powerladder` pip-installable and drop the `sys.path.insert`
  shims in `scripts/` (tests load scripts by path via `importlib`; that keeps
  working).
- `jump_penalty` sensitivity sweep for the forward statistic (default 1.0
  inherited, meaning differs under the normalised kernel) — fold into any real
  evaluation (note §9).
