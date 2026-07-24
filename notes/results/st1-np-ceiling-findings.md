# NP-optimal LRT ceiling in the ST1 bake-off — findings

**Date:** 2026-07-24 · **Status:** **frozen** (slurm job **5646**, MATS `compute`,
single task ~9 min: bank 392 s + negs 86 s + eval/scoring). **Scope:** the
optional/non-blocking Phase-2 item (`tasks.md` §Phase 2). **Origin:**
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

## Frozen results (n_each=200, f0_n=61, n_mc=2000, drifts 0→1.5 Hz)

**Parity guard: `max |TPR delta vs bakeoff| = 0`** — the reproduced eval populations are
byte-identical to `results/st1/bakeoff_summary.json`, so "fraction of optimal" divides the
*same* populations the bake-off reports.

**The ceiling is AUC = TPR@0.05 = TPR@0.01 = 1.0 in every column and at every drift**
(inference null, structural mix, and the controller-only hard case). Under the Whittle
model the training comb and the (line-free) nulls are *perfectly separable* given a
300 s / 20 Hz trace, at every drift — so the intrinsic information is always present; the
only question is whether a deployable detector extracts it. (With a perfect ceiling
`rho_tpr = tpr_det`, redundant with the bake-off table, so **`rho_auc` is the informative
fraction**; `rho_auc = (auc_det − 0.5)/(auc_ceil − 0.5)`, negative when a detector is
below chance.)

**vs the inference null — the Viterbi tracker is essentially NP-optimal.**
`rho_auc[viterbi] = 1.00` at *every* drift (AUC 1.00 throughout). Everything else leaves
power on the table and collapses under drift: spectral matched filter `1.00 → 0.78 → 0.38
→ −0.10` (drift 0/0.1/0.2/0.4), dg_order_full `0.83 → 0.68 → 0.58 → 0.28`, mtf and
dg_fixed_oracle near/below chance beyond drift 0. So against the stated null the tracker
recovers 100% of the achievable power across the whole wander axis.

**vs structural confusers (ar1 / ar1_t / controller) — the tracker is NOT enough; the DG
order family carries it, with a real gap at high drift.** Here `rho_auc[viterbi]` is only
0.34 at low drift and goes negative by drift 0.4. The best deployable detector is
**dg_order_full**: `rho_auc = 0.93 → 0.85 → 0.79 → 0.67 → 0.30 → 0.10` over the six drifts
— it approaches the ceiling at low drift but still surrenders most of the power by 0.8 Hz.

**Hard case (drift 0.4 vs controller-only) — only the DG order family separates the
wandering line from a limit cycle.** Viterbi and the spectral filter score **AUC 0.0**
(actively fooled — the controller cycle out-lines the smeared training line);
dg_order_full reaches `rho_auc = 0.96` (AUC 0.978) and dg_order_split 0.92. The f₀-oracle
manages only 0.36 — confirming the missing ingredient is *order-tracking structure*, not
knowledge of the line frequency.

**Reading.** The ceiling turns the three qualitative bake-off findings into numbers: (i)
tracking is necessary (the fixed matched filter falls below chance under drift, ρ<0); (ii)
against confusers only the DG order family approaches the optimum; (iii) even the best
deployable detector leaves a widening fraction of the achievable power unclaimed as drift
grows — the honest headroom above today's detectors. Against the plain inference null,
though, there is essentially no headroom left: the tracker is optimal.

Unit tests (`tests/test_np_ceiling.py`, 5, pass); full suite 5 failed / 186 passed = the
known BLAS byte-identity baseline (`byte-identity-fixtures-blas-sensitive`) + these 5, no
new failures.

## Caveat on the perfect ceiling

AUC/TPR = 1.0 everywhere is a property of the Whittle model, not a bug: the class-mean
periodograms (comb vs smooth/broadband) are linearly separable by the log-LR given this
trace length, so the rank separation is perfect. It is a *lower bound on the true
optimum* (the true NP test can only do at least as well), so reporting deployable
detectors as a fraction of it is conservative in the right direction. The value is
diagnostic: it localises all remaining difficulty in the detector, and — via the
confuser/hard-case gaps — shows exactly where headroom remains.

## Addendum 2026-07-24 — structural negative-PSD sampler fix (numerically inert)

Pre-merge review caught that the `structural` ceiling's Monte-Carlo negative PSD was
sampled via `_make_structural_negatives(1, r)`, whose round-robin allocator puts the
single trace in the first class — so `S_neg["structural"]` was estimated from **ar1
traces only**, not the ar1/ar1_t/controller mixture the eval population and bake-off
column use. Fixed (`scripts/st1_np_ceiling.py::_structural_mix_sampler`): the null type
is now drawn uniformly from the mix per MC draw, with a regression test
(`test_structural_neg_sampler_covers_the_full_mix`).

This does **not** change any frozen number above: the ceiling saturates at
AUC = TPR = 1.0 in every column and drift (the comb template separates the training
class from *any* smooth null, whatever feeds `S_neg`), so every ceiling entry stays 1.0
and every `rho = det/ceiling` is unchanged; the detector metrics derive from the
independent eval seed and never touched the fixed sampler. The frozen
`np_ceiling_summary.json` therefore remains valid as published; the fix matters only if
the ceiling is ever run in a non-saturated regime (shorter traces / harder null / tighter
band), where the structural fractions would otherwise divide by an ar1-only ceiling.

## Next step (Phase 4)

Paper wiring only: add the ceiling row/fraction to `tab:bakeoff`, extend `app:bakeoff`
prose, and connect the Related-Work "optimality ceiling" hook (`paper/main.tex:271-273`).
Optional future strengthening if a reviewer disputes the Whittle model: a learned/HGB
Bayes-optimal ceiling with a saturation study.
