# NP-optimal LRT ceiling in the ST1 bake-off — findings

**Date:** 2026-07-24 · **Status:** harness built + validated locally; **full
number-freeze pending Slurm** (single task, `scripts/slurm/st1_np_ceiling.sbatch`).
**Scope:** the optional/non-blocking Phase-2 item (`tasks.md` §Phase 2). **Origin:**
`notes/discussion/method-soundness-and-prior-art.md` §2.2. **Regenerate:**
`scripts/st1_np_ceiling.py` → `results/st1/np_ceiling_summary.json` →
`scripts/plot_st1_np_ceiling.py` → `figures/st1_np_ceiling.*`.

## What this is

We own the generators, so we can approximate the Neyman–Pearson optimal
likelihood-ratio detector between the training and inference-null generators and
report each corpus-free bake-off detector as a **fraction** of it — the memo's
"X% of NP-optimal power at Y% of the information cost", which turns *"is our detector
good?"* into a number. Neither generator has a closed-form likelihood, so the ceiling
is approximated (user decision 2026-07-24) by a **Whittle spectral LRT**.

## Method

Work on the detrended in-band periodogram `I(f)` (band 0.3–1.7 Hz). Under the Whittle
(stationary-Gaussian) approximation the ordinates are ~independent `Exp(mean = S(f))`,
giving `ℓ_H(x) = −Σ_f [log S_H(f) + I(f)/S_H(f)]`. Class PSDs are Monte-Carlo estimates
(mean periodogram): one `S_neg` per negative class, and a **per-f₀ training template
bank** `S_tr(f; f0_k)` over a grid across the Ko band [0.5, 1.5] Hz. The line frequency
`f0` is the known generator nuisance, so it is **marginalised**:

    LR(x) = logsumexp_k ℓ_tr(x | f0_k)  −  ℓ_neg(x)     (uniform f₀ prior).

The template bank is built **drift-agnostic** — each template pools training traces
whose drift is drawn from the bake-off drift grid — so the ceiling knows no more about
the adversary's drift than a deployable detector does. The MC corpus is drawn from an
rng stream disjoint from the bake-off's (`corpus_seed_offset`); the ceiling is fit once,
then frozen and applied to the **exact** bake-off eval populations (same
`seed=20260721`, drifts, `n_each=200`, reproduced by replaying
`plot_st1_bakeoff.main`'s RNG order and reusing its population builders). Ceilings are
fit for the inference null, the mixed structural null, and the controller-only null
(the hard case at 0.4 Hz). Detectors and ceiling are scored on the same populations via
the same `roc` / conservative `_tpr_at_far` machinery, so the fraction is
apples-to-apples.

**Fractions:** `rho_auc = (AUC_det − 0.5)/(AUC_ceiling − 0.5)` (threshold-free) and
`rho_tpr = TPR_det/TPR_ceiling` at fixed FAR (null where the ceiling's own TPR < 0.05).
**Information cost** is reported categorically (the memo admits it is rhetorical): the
ceiling needs white-box generative access to both samplers plus a large MC corpus; the
corpus-free detectors need one trace and the band.

## The caveat (carried into any report of the number)

This is NP-optimal **under the Whittle model only** — a 2nd-order/periodogram statistic
that discards harmonic-phase coherence and the null's non-Gaussian OU/Poisson-burst
structure, so the true optimum can exceed it. Where a *tracking* detector (Viterbi /
DG-order) approaches or beats the ceiling at high drift, that reflects wandering-line
structure a stationary spectral template cannot see — a finding, not a bug. A heavier
learned/HGB Bayes-optimal ceiling with a saturation study is the noted future
strengthening if a reviewer disputes the Whittle model.

## Validation this session (local, reduced corpus — NOT the frozen numbers)

- **Parity guard: `max |TPR delta vs bakeoff| = 0`** across both negative columns, all
  detectors, all drifts, both FARs — the reproduced eval populations are byte-identical
  to the frozen `results/st1/bakeoff_summary.json`. So the "fraction of optimal" divides
  the *same* populations the bake-off already reports.
- **Qualitative pattern** (reduced-corpus smoke; direction only, magnitudes not frozen):
  against the inference null the Whittle ceiling holds AUC ≈ TPR@0.05 ≈ 1.0 across the
  whole drift axis — the optimal spectral test separates the training comb from the
  power-matched-but-smooth null regardless of drift. The **Viterbi tracker tracks the
  ceiling** (ρ_auc ≈ 1); the **fixed matched filter collapses** under drift (AUC falls
  below chance as the wandering line smears, so ρ_auc goes negative). This is the same
  "tracking is necessary" story as the bake-off, now quantified against the optimum: the
  achievable power stays high across drift, and only a tracking detector realises it.
- Unit tests (`tests/test_np_ceiling.py`, 5, pass): periodogram contract; Whittle-LR
  ordering on a controlled tone; corpus/eval seed disjointness; eval-pop determinism;
  ceiling ≥ spectral at drift 0. Full suite: 5 failed / 186 passed = the known BLAS
  byte-identity baseline (`byte-identity-fixtures-blas-sensitive`) + these 5; no new
  failures (no generator code was touched).

## Next step

Freeze on Slurm (`sbatch scripts/slurm/st1_np_ceiling.sbatch`, MATS `compute`,
`n_mc=2000`, `f0_n=61`; ~10–15 min single-threaded). Then record the frozen ρ numbers
and the drift where tracking detectors approach/beat the Whittle ceiling here, and tick
`tasks.md`. Paper wiring (`tab:bakeoff` / `app:bakeoff` / the Related-Work "optimality
ceiling" hook at `paper/main.tex:271-273`) is Phase 4.
