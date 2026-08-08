# Handoff — 2026-08-08 (duration-aware extension: BUILT, TESTED, KILLED)

Branch **`feat/hsmm-renewal-smoke`** kept, **unmerged**; PR **#19 closed without
merging**. Full suite: **332 passed / 0 failed** (`pytest -p no:debugging -m "not gpu"`).

> **The duration-aware extension is closed.** Read
> **[`notes/discussion/hsmm-duration-aware-postmortem.md`](notes/discussion/hsmm-duration-aware-postmortem.md)**
> first — it is the index to the whole line and says why it will not be retried.
> E2 was built, run against precommitted criteria, and failed: it closes 2% of a
> confirmed +0.252 estimation gap and beats a geometric-dwell HMM by +0.011.

> Supersedes the 2026-07-27 Phase-4 handoff. That work (PR #16, identifiability
> theory, the Pareto plot) is merged and its record is in `log.md`; nothing below
> revisits it. The paper itself is **untouched** by this branch — the
> duration-aware extension is still a `spec.md` provisional extension, not part
> of the manuscript's claimed method or evidence.

## Where the extension stands

The branch arrived with E0/E1 passing and a recorded **GO to E2**. E2 was then
built and run, and the extension is **closed**. The path, with the findings note
for each stage:

| Stage | Question | Outcome | Note |
|---|---|---|---|
| E0/E1 smoke | is the local information there? | **E0 gate invalid** (see below) | `notes/results/hsmm-renewal-smoke-findings.md` |
| Stage 1 | does E1 generalize? | 1B passes; 1A explained; **1C fails hard** | `notes/results/hsmm-generalization-findings.md` |
| Stage 2 | where is the meter boundary? | recall and decision are decoupled | (in the E0 note) |
| **Defect** | is the oracle a real ceiling? | **no — it read the metadata** | `notes/results/hsmm-e0-defect-and-fair-ceiling.md` |
| Stage 3 | is any regime identifiable? | **yes, one** | `notes/results/hsmm-regime-search-findings.md` |
| Confirmation | does it survive fresh seeds? | **8/8, both surrogates** | `notes/results/hsmm-cell-confirmation-and-e2-precommitment.md` |
| Representation | is "no events" encoded honestly? | no — fixed by a hurdle likelihood | (in the E2 note) |
| **E2** | do explicit durations help? | **no — killed** | `notes/results/hsmm-e2-findings.md` |
| Closure | should it be retried? | **no** | `notes/discussion/hsmm-duration-aware-postmortem.md` |

Design rationale for the whole re-ordering: `notes/plans/hsmm-e2-preconditions-plan.md`.

## The three things that matter most

1. **The E0 gate was measuring metadata, not the power channel.** It compared
   oracle-windowed positives against *blind*-extracted nulls, and
   `paired_fraction` is the hardcoded constant `1.0` for any oracle window set.
   Substituting white noise for every positive trace, keeping the true metadata,
   still gives **AUC 1.000**. Fixed by `oracle_null_matrix`, which windows nulls
   at random locations with counts and durations resampled from the positives —
   the control `spec.md` had specified and the implementation had skipped. With
   it, the noise control drops to 0.53 (chance). **E1 results are unaffected**
   throughout: the blind path always scored both sides identically.

2. **Unseen confusers are not rejected.** With a null family absent from both the
   fitted denominator and calibration, four of six families false-alarm at
   0.985-1.000. Structural, not incidental: the score is a ratio against a finite
   *fitted* null set. Any claim built on this detector is conditional on the
   stated null library, more strongly than `spec.md`'s existing Rung-2
   conditionality admits.

3. **The target was real and E2 still failed.** 10 s records, training at a 20%
   share of a Ko eq-11 aggregate, against phase-randomised surrogate nulls:
   confirmed 8/8 on fresh seeds, with a corrected estimation gap of **+0.252**
   and a large alignment benefit (`align+` **+0.387**). The explicit-duration
   HSMM closes **2%** of that gap. The information is there; the model does not
   recover it.

## Outcome

E2 was built (`powerladder/typeb/hsmm.py`) and run against the precommitted
contract (`scripts/hsmm_e2.py`, `notes/results/hsmm-e2-findings.md`):

- **primary criterion fails on both clauses** — IAAFT AUC 0.652 against a 0.75
  bar; improvement over E1 +0.006 ± 0.027, interval excluding zero in 0/5 seeds;
- **duration claim unsupported on every arm** — margin over an otherwise
  identical geometric-dwell HMM is +0.010 to +0.011, against a +0.05 requirement;
- the composite-null blocker was fixed properly (the hurdle likelihood in
  `renewal_hurdle.py`: point mass for N=0, marginalisation over undefined marks),
  not engineered away.

**The one thing worth carrying forward** is duration-neutral: marginalising over
segmentations instead of committing to a hard extraction gains **+0.070 AUC and
+0.134 TPR at composite FAR 0.05** against the physical nulls (5/5 seeds, every
per-family FPR ≤ 0.05). That is a Rung-1 detector-bank improvement independent of
the dead hypothesis, and is the only operational gain the line produced.

## Also outstanding

- **`spec.md`'s provisional-extension section still advertises this as an
  approved prototype**, with the original E0 design that turned out to be
  invalid. A future session reading `spec.md` alone would restart the line. It
  needs amending to point at the postmortem — **`spec.md` requires explicit
  approval to edit, so this is Tom's call.**
- **The postmortem lives only on this unmerged branch.** To do its job it needs
  to be reachable from `main`; otherwise a session on `main` will not see it.
- `/check-PR` was never run on #19. Moot now the PR is closed, but the branch
  carries ~4k lines that no adversarial pass has seen.
- The meter-degraded stage-2 cells were never audited for zero-event rate; those
  numbers should not be reused until they are.

## Code map

| Path | What it is |
|---|---|
| `powerladder/typeb/renewal.py` | E0/E1 detector (pre-existing; gained `widen_profile`, a `shrinkage` passthrough) |
| `powerladder/typeb/renewal_campaign.py` | **new** — populations, features, thresholds, surrogates, dilution, the fair oracle null |
| `powerladder/typeb/renewal_regime.py` | **new** — E1 / align+ / estimation gap with paired bootstrap; the admission rule |
| `powerladder/ko_workload.py` | gained `aggregate_F_phase_meta` (metadata-carrying sibling of `aggregate_F`) |
| `scripts/smoke_hsmm.py` | refactored onto the library; **reproduces its frozen `summary.json` exactly** |
| `scripts/hsmm_generalization.py` | stage 1 |
| `scripts/hsmm_meter.py` | stage 2 |
| `scripts/hsmm_regime_search.py` | stage 3 |
| `scripts/hsmm_confirm_cell.py` | the locked one-cell confirmation |
| `powerladder/typeb/renewal_hurdle.py` | **new** — hurdle likelihood: point mass for N=0, marginalisation over undefined marks |
| `powerladder/typeb/hsmm.py` | **new** — E2: explicit-duration HSMM + geometric and shuffled controls |
| `scripts/hsmm_hurdle_refreeze.py` | corrected E1 baseline + zero-event audit |
| `scripts/hsmm_e2.py` | E2 against the precommitted contract |
| `tests/test_{renewal_campaign,renewal_regime,renewal_hurdle,hsmm}.py` | **new**, 99 tests |

Results under `results/hsmm_{smoke,generalization,meter,regime_search,confirm_cell,hurdle,e2}/`.
Frozen ST1/ST2 artefacts are untouched — verified by `git status`.

## Gotchas

- **`results/hsmm_smoke/summary.json` is a reproduction check.** Re-running
  `smoke_hsmm.py` reproduces it byte-for-byte apart from `elapsed_s`. If a future
  change breaks that, the library refactor has drifted.
- **Never score E0 nulls with `blind_matrix`.** That is the defect. Use
  `oracle_null_matrix`. `tests/test_renewal_campaign.py` guards it.
- **AAFT is marginal-matched and only *approximately* spectrum-matched**
  (residual 5-6e-3; IAAFT 1.0e-3). Do not describe it as matched on both.
- **The all-zero feature vector is still live in the legacy path.**
  `summarize_events` returns zeros when no events are extracted and reports
  *perfect* repeatability from a single event; a population where that is common
  can "separate" on degeneracy alone. `renewal_hurdle.py` fixes it properly;
  `renewal.py` deliberately still has it so the pre-fix results stay auditable.
- The single-A100 transport limitation is unchanged: none of this bears on
  whether an external meter sees a distributed all-reduce phase.
