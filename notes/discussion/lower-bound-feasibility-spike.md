# Lower-bound feasibility spike — is a covertness cost bound derivable?

**Date:** 2026-07-24 · **Status:** done — kill/continue gate resolved.
**Verdict:** a bound is derivable (physics-anchored, conditional) → the project
clears the "position paper vs contribution" gate on the *contribution* side, with
the honest caveats recorded below.

This note is the time-boxed feasibility spike called for in `tasks.md` (Strategic
gates). It is deliberation + a derivation attempt, not a finished theorem section
(that is Phase 4). The numerical sanity check is
`scripts/lower_bound_spike.py` → `figures/lower_bound_spike.*`,
`results/spike/lower_bound_spike.json`.

---

## 1. The question

Can we state a **lower bound on the cost of hiding** the training cadence — "to
drive detectability down to ε the adversary must pay ≥ f(ε)" — for even one
attack family (i.i.d. phase jitter), under our filtered low-rate meter?

Why it is the gate (`notes/discussion/method-soundness-and-prior-art.md` §4, "the
trap"): our empirical frontier measures how much can be hidden *from us* — an
**upper** bound on hideability that a better detector moves. The governance claim
needs the opposite direction: a **lower** bound on the cost of hiding *from any
verifier*. The prior-art audit (`notes/discussion/st0-prior-art-audit.md` §6)
confirmed covert-communication theory hands over the machinery — the warden game,
the metric (KL/TV), the cap `detector power ≤ TV`, Cachin relative-entropy
security, the Bash–Goeckel–Towsley square-root law — but **no ready-made theorem**
for our object, so the spike is not pre-empted. This resolves `plan-for-paper-2.md`
§10 Q2 (how rigorous the identifiability theory can be).

---

## 2. Setup and observation model

- **Attack.** i.i.d. fractional period jitter: iteration `i` has period
  `T_i = 1/(f₀(1+ξ_i))`, `ξ_i ~ N(0, σ²)` (`powerladder/ko_workload.py:170,176`).
  σ is the distortion knob; the ST2 `jitter` family.
- **Signal.** The synchronous training iteration lays a spectral line at the
  cadence `f₀`. i.i.d. period jitter *accumulates*: the boundary phase performs a
  random walk with diffusion `D ≈ (2π)²σ²f₀` and the cadence has `CV ≈ σ`
  (`powerladder/typeb/deperiod.py:17–18`, tested).
- **Channel.** Device map `P = rF + P₀` (`powerladder/forward.py`) then the meter
  (`powerladder/observation.py`): LTI filter, integrate-and-sample at `f_s`, AR(1)
  floor `N₀`. For the in-band cadence (`f₀ < f_s/2`, the nominal 20 Hz channel) the
  line passes with gain `|H(f₀)|²`.
- **Verifier.** The passive warden. The clean, provable object is the **fixed**
  verifier that knows the nominal `f₀` and tests that bin; the tracking/optimal
  verifier is the harder ceiling, addressed in §5.

---

## 3. The derivation

**Step 1 — line attenuation under phase diffusion.** For an accumulating-phase
process the periodic component's autocorrelation decays as `exp(−D|τ|/2)`, so the
line is a Lorentzian of half-width `∝ D`. Measured over a finite window `T`, the
coherent power that stays inside one DFT bin (width `1/T`) is

```
    P_c(σ) / P_c(0)  ≈  1 / (1 + κ σ²),     κ ≈ π f₀ T          (accumulating jitter)
```

with `κ` carrying an O(1) Lorentzian-width prefactor. This is the core physical
claim; §4 confirms it numerically (form R² = 0.97; fitted κ ≈ 2950 within a factor
~3 of the parameter-free π f₀ T ≈ 940).

**Step 2 — detectability cap.** Any verifier's detection advantage is bounded by
the total variation between the training-trace and null-trace laws,
`power ≤ TV(P_train, P_null)`; Pinsker gives `TV ≤ √(KL/2)`, and in the Gaussian
line-vs-floor regime `KL ≈ SNR_c(σ) = P_c(σ)|H(f₀)|²T / N₀`. So detectability
inherits the σ-rolloff of `P_c`.

**Step 3 — covertness threshold.** Requiring ε-covertness (`TV ≤ ε`) inverts the
rolloff into a **minimum distortion**

```
    σ*(ε)  =  (1/√κ) · √( SNR₀/(2ε²) − 1 )        →  grows like 1/ε as ε → 0.
```

A finite, positive floor on σ: hiding is not free of distortion. (§4 measures this
threshold on the fixed-bin verifier.)

**Step 4 — the physics anchor (this is what makes it a *lower* bound, not just an
upper bound on our own detector).** Reaching `σ*(ε)` costs, because the synchronous
all-reduce is communication-bound:

- *Idle-insertion jitter* (pad the period, real work fixed) must inflate the mean
  period to keep every period ≥ the compute time → throughput cost that grows with
  σ. The repo's measured idle-pad anchors are 15–680%
  (`plan-for-paper-2.md` §5).
- *Work-varying jitter* (`work_sigma`, the measured ≈zero-throughput-cost escape,
  `ko_workload.py:106–124`) is bounded: the micro-step count `G_i` is
  integer-quantised and the comm/optimizer down-phase `T_down` is **fixed per
  iteration** regardless of `G_i`. So a fixed periodic marker at the comm cadence
  survives, and work-variation cannot drive the line to zero without either idling
  (throughput) or desynchronising the all-reduce (a correctness / learning cost).

**Step 5 — the bound.** Combining: to hide the cadence to detectability ε via
i.i.d. jitter the adversary must reach σ ≥ σ*(ε), and under the synchronous-barrier
constraint reaching σ*(ε) costs **≥ (throughput overhead OR learning-efficiency
degradation)** — the two cannot both be zero, because the fixed comm barrier leaves
a residual line that only idle-insertion or desynchronisation removes. That is a
lower bound on hiding cost for this family. Its strength is *conditional* (see §5).

---

## 4. Numerical sanity check

`scripts/lower_bound_spike.py` builds training vs inference-null populations through
the exact `powerladder/typeb/ko_synth` path (generator → forward → meter), scores a
fixed-bin statistic at the known `f₀`, and reads the population AUC
(`n_each=80`, `f₀=1 Hz`, `T=300 s`, seeded). It confirms every load-bearing step
(`figures/lower_bound_spike.*`):

- **Coherent line power follows the phase-diffusion rolloff** (panel A). The
  measured `P_c(σ)/P_c(0)` tracks `1/(1+κσ²)` at **R² = 0.97**; the fitted κ ≈ 2950
  sits within a factor ~3 of the parameter-free analytic π f₀ T ≈ 940 — the width
  convention only fixes an O(1) prefactor, so this is order-of-magnitude
  confirmation of Step 1.
- **Detectability decays and is TV-capped** (panel B). `TV ≈ 2·AUC−1` falls
  monotonically from 1.0 to ≈0 over σ ∈ [0, 0.5]. Its effective rolloff constant
  (κ ≈ 23) is far smaller than the coherent-power κ — AUC saturates while *any* f₀
  excess stays separable from the line-free null — so the covertness thresholds are
  read off this (conservative) detectability rolloff, not the power rolloff.
- **Covertness thresholds are finite and sizeable.** σ*(0.5) ≈ 0.21, σ*(0.2) ≈ 0.41
  (within the sweep); σ*(≤0.1) ≥ 0.62 (extrapolated). Driving a *fixed* verifier
  below ε = 0.1 needs σ ≳ 0.6 — a large distortion (CV ≳ 0.6).
- **The de-periodicisation anchors hold** where the thresholds live: CV ≈ σ to
  σ ≈ 0.2 and D matches (2π)²σ²f₀; both linear anchors break down by σ ≈ 0.35
  (period clamp), i.e. outside the covertness-relevant regime.
- **The work-jitter escape is *more* detectable at matched σ** (panel B: TV_work >
  TV_jitter for σ ≥ 0.1). Direct support for Step 4 — the fixed comm-barrier
  residual keeps the cheap-throughput attack visible, so it does not beat the bound
  "for free"; it trades throughput cost for detectability.

---

## 5. Honest scope of the bound (the caveats that set claim wording)

1. **Fixed vs optimal verifier.** The clean, numerically-demonstrated bound is
   against a fixed verifier at the known cadence. Against a *tracking* verifier
   (Viterbi/DG order family, ST1) the relevant quantity is the phase-diffusion D
   itself: a line can be tracked until it diffuses beyond coherence over the window.
   The argument extends (large D ⇒ untrackable), but the fixed-bin check does not by
   itself prove the bound against a tracker — that is the analytic step to firm up
   in Phase 4.
2. **Conditional on the realisation.** "Cost" is throughput *or* learning-efficiency;
   the learning-efficiency leg is stated, not measured (the GPU campaign is descoped,
   `spec.md`). The bound is "must pay in at least one of these", not a single
   dollar figure.
3. **Generator-internal.** TV is between *our* generator's train/null laws, so the
   bound is conditional on the generator being a faithful stand-in — the standing
   caveat for a synthetic paper (`north-star-and-positioning.md` §4).

None of these sink the bound; they set its wording: **"to hide to ε an i.i.d.-jitter
adversary must pay ≥ f(ε) in throughput or learning efficiency; the free-work-jitter
escape does not evade this because the synchronous comm barrier leaves a residual
line"** — a necessary-cost statement, not "de-periodicisation is necessarily
inefficient" (which our own work-jitter result refutes,
`north-star-and-positioning.md` §10).

---

## 6. Verdict on the fork, and what feeds Phase 4

**Bound exists ⇒ contribution.** Frame the identifiability section as a covertness
cost bound: state Step 1–5 as a proposition for the i.i.d.-jitter family, with the
fixed-verifier case as the proved core and the tracking/optimal case as the
physics-anchored extension. Keep the empirical frontier as the complementary
*upper* bound on hideability (evidence), and the bound as what turns it into a
governance statement.

Feeds Phase 4 (`plan-for-paper-2.md` §7 "Threat model and identifiability", §10 Q2):
promote §3's derivation to a proposition, firm up the fixed→tracking step, and cite
`figures/lower_bound_spike.*` as the sanity check. Rigor target: proposition-level
(the venue decision is arXiv-first, `tasks.md`), which this supports.

**Open sub-question carried forward** (all four fields flagged it,
`st0-prior-art-audit.md` §8 Q3 / `method-soundness-and-prior-art.md` §6.4): which
metric — TV, KL, or Hellinger — gives the cleanest distortion→detectability bound.
This spike used TV via `2·AUC−1` and KL via Pinsker in the Gaussian regime; the
tensorisation properties of KL/Hellinger over the many-iteration product law may
give a tighter closed form for the Phase 4 proposition.

---

## 7. Corrections applied when this was promoted to the paper (2026-07-24)

Writing §3/App B (Phase 4) surfaced two mathematical defects in §3 above. The
verdict — bound derivable, contribution side of the fork — **stands**; the
derivation as written did not. Both are fixed in the manuscript
(`paper/main.tex` `subsec:identifiability`, `app:identifiability`); §3 above is
left as the historical record.

1. **Step 2 proved the converse of what is needed.** `power ≤ TV ≤ √(KL/2)`
   (Pinsker) bounds TV from *above*, so driving that bound below ε shows some
   distortion **suffices** for covertness — an upper bound on the σ the adversary
   needs, not the lower bound the governance claim requires. The direction that
   works runs through an **explicit** verifier: a single-trace test with achieved
   advantage `A(σ)` has `A ≤ TV`, so `A(σ) > ε ⟹ not ε-covert`, and since `A`
   decays in σ, ε-covertness *requires* `σ ≥ σ*(ε)`. Pinsker is now stated as the
   complementary converse (a remark on what is *not* proved), not the
   load-bearing step.
2. **`2·AUC−1` is not an achieved single-trace advantage.** It is the
   Mann–Whitney/Gini index and can *exceed* the best threshold test's advantage
   (Youden `J = max_thr(TPR−FPR)`, the two-sample KS distance), so it cannot
   certify that smaller distortions fail. `scripts/lower_bound_spike.py` now also
   computes **J** per σ and inverts the **J** rolloff for the thresholds; AUC and
   the Gini-based thresholds are retained in the summary
   (`detectability_tv`, `covertness_thresholds_gini`) for continuity with §4
   above. Every previously committed number is unchanged — only
   `covertness_thresholds` now reads off J.

**Consequences for the numbers.** J-rolloff `κ_J = 34` (R² = 0.95, J₀ = 1.0)
against the Gini `κ = 23`. Thresholds move *down*, which is the conservative
direction for a necessary condition: **σ\*(0.5) = 0.17, σ\*(0.2) = 0.34** (both
now *inside* the swept range, and below the σ ≈ 0.35 anchor-breakdown point where
§4's caveat bites — an improvement on the old 0.21/0.41), σ\*(0.1) = 0.51 and
σ\*(0.05) = 0.74 extrapolated. The work-jitter escape is still more detectable at
matched σ under J (every σ ≥ 0.1), so Step 4 is unaffected.

**Also corrected: the small-ε scaling.** §3 Step 3 says σ\* "grows like 1/ε".
That follows from its formula `σ* = (1/√κ)·√(SNR₀/(2ε²) − 1)`, with ε² inside the
root. The inversion actually implemented and measured is
`σ* = √((y₀/ε − 1)/κ)`, giving **σ\* ∝ ε^(−1/2)** — confirmed by the frozen
thresholds (ε from 0.1 to 0.01 is a 10× drop and moves σ\* by 3.32×, i.e. √10,
not 10×). The manuscript states ε^(−1/2).

---

## 8. Re-derivation at n=600 after the pre-merge review (2026-07-24, same day)

`/check-PR` over the Phase-4 branch showed the §7 numbers, though derived from the
corrected estimator, were **not stable**, and one further analytic error surfaced.
The §7 figures are therefore superseded; the derivation's structure is unchanged.

**The n=80 sizing was too small for the quantity being inverted.** The H0 floor of
the plug-in Youden statistic (`_ks_null_floor`, mean + 3 sd of the one-sided KS
null) is **0.25 at n=80**, against 0.093 at n=600 — at n=80 it exceeds every ε
being inverted. (≈0.09 is the *mean* of the n=80 null, a different statistic; the
floor is what the mask uses.) Consequences, all measured: the σ=0.5 row (J=0.100, AUC
**0.4967**) was statistically indistinguishable from no separation yet carried
**78%** of the fit leverage, because the fit linearises as `J0/J − 1` and diverges
as J→0; κ_J varied **13–53** across 12 repeats; the bootstrap CI was κ_J [14, 86]
and σ\*(0.2) [0.22, 0.53]; and **P(σ\*(0.2) > 0.35) = 0.61**, so the priced bracket
was more likely wrong than right. The tell was internal: the *same* point fails the
script's mask on the Gini series (2·AUC−1 = −0.007 < 0) but passes it on J, purely
because J is bounded below by zero.

**Fixes in the script:** `N_EACH = 600`; the fit mask now excludes points below an
n-dependent H0 floor (`_ks_null_floor`, mean + 3 sd of the one-sided KS null,
0.093 at n=600) instead of a fixed 0.02·y₀; bootstrap CIs on κ_J and every σ\*;
R² over the fitted points only; `sigma_faithful_max` computed from the measured
CV so thresholds beyond the realised-σ region are flagged in the artefact; and a
faithful-only refit reported as the sensitivity. Runtime ~100 s, still CPU-local.

**Superseding numbers:** κ_J = **28** [23, 37], R² 0.92, fitted on σ ∈ [0.05, 0.35];
**σ\*(0.5) = 0.19** [0.17, 0.21] (inside the realised-σ region);
**σ\*(0.2) = 0.38** [0.33, 0.42] (**beyond** it — flagged); σ\*(0.1) = 0.57,
σ\*(0.05) = 0.82 extrapolated. Faithful-only refit κ = 19, i.e. σ\* = 0.23/0.46 —
the ~20% systematic the paper now quotes. σ\*(0.2) moved up a cost bracket
(66–159% → 159–375%), so the manuscript states the price as an order of magnitude
rather than a bracket. The work-jitter comparison is unchanged (work more
detectable at every σ ≥ 0.1, less below).

**And a factor of π in Step 1.** `κ ≈ π f₀ T` is wrong. Carrying the algebra
through — `D = 4π²σ²f₀` ⇒ HWHM `γ = D/4π = πσ²f₀`, bin half-width `B = 1/2T`,
in-bin mass `(2/π)·arctan(B/γ)` with small-argument form `1/(π²f₀Tσ²)` — gives
**κ = π²f₀T = 2961**, which the measured fit (3020) matches to **2%**. So the
"O(1) width-convention prefactor" §3 leaves open is not open at all: the
derivation is parameter-free and predictive. This strengthens Step 1 from
order-of-magnitude agreement to a 2% test.

---

## §9. Correction, 2026-07-27 — the work/jitter comparison was not at matched distortion

Found in the second `/check-PR` pass over `b6d9eb8`, independently by two agents.
**The numbers in §8 above are superseded**; the derivation's structure is unchanged.

**The defect.** `_train_scores(work=True)` passed the module-level `KO` unmodified,
so `KoWorkloadParams.sigma_jitter` kept its **0.1** default *and* `work_sigma=σ` was
applied on top. The jitter arm, by contrast, set `sigma_jitter=σ` with no work
variation. The work arm therefore always carried 0.1 of period jitter that the
jitter arm did not — the two series were never at matched total distortion, despite
three places in the manuscript saying they were.

The signature is unambiguous: if the work arm carries a 0.1 baseline, J_work(σ=0)
should equal J_jitter(σ=0.1). Measured on the pre-fix artefact: **0.8617 vs 0.8850**,
agreeing to ~2 sd. This fully accounts for the sub-0.1 reversals — the region where
§8 reported work as *less* detectable was reporting a parameterisation artefact as a
physical finding, and it is also why the work curve was non-monotonic (0.862 at σ=0,
rising to 0.983 at σ=0.02, falling to 0.318 at σ=0.5).

**The fix.** The work arm now zeroes `sigma_jitter` before applying `work_sigma`, so
both arms set the jitter scale explicitly. Re-run in a venv built from
`requirements.txt` verbatim (numpy 2.2.6 / scipy 1.15.3 / matplotlib 3.10.5).

**The comparison is now clean, and stronger.** Work ≥ jitter at **every σ ≥ 0.05**;
an exact tie (1.000 vs 1.000) at σ = 0; and the three sub-0.05 gaps are ≤ 0.005,
inside the ~0.004 sampling sd at J ≈ 0.99. The curve is monotone. So the comparative
claim `ass:cost` leans on holds across the sweep instead of only above σ = 0.1.

**Post-fix numbers:** κ_J = **32** [24, 40], R² 0.90, fitted on σ ∈ [0.05, 0.35];
**σ\*(0.5) = 0.18** [0.16, 0.20]; **σ\*(0.2) = 0.36** [0.32, 0.41]; σ\*(0.1) = 0.53,
σ\*(0.05) = 0.78 extrapolated; faithful-only refit κ_J = **22** (σ\* = 0.22/0.43).
Coherent-power fit κ = **3068**, 4% above the parameter-free π²f₀T = 2961 — and 2812,
i.e. 5% below, when restricted to the faithful σ ≤ 0.1 regime. Every value sits inside
the pre-fix confidence intervals, and **both cost brackets are unchanged** (32–66% at
ε = 0.5, 159–375% at ε = 0.2), so no pricing conclusion moves.

**Two things this exposed.** The realised-cadence diagnostics are heavy-tailed at high
σ — CV at nominal σ = 0.35 read 0.50 pre-fix and 1.17 post-fix, from an RNG stream
shift alone — because the period is a reciprocal Gaussian with a clamp, so rare
near-zero denominators dominate. Treat `cadence_cv` and `phase_diffusion_D` above
σ ≈ 0.2 as order-of-magnitude only. And the earlier claim that the π²f₀T agreement is
a "2% test" was over-precise: it is a several-percent test, 81% of whose leverage sits
on the largest σ swept, where the mean period has drifted well off f₀.
