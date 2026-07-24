# ST1 gate — pre-registered go/no-go criteria and findings

**Status:** criteria pre-registered 2026-07-21, BEFORE any calibration numbers
exist. The code that produces the realised-FAR tables (tasks 19.2–19.4) had not
been run — or, for the later stages, written — when this section was committed.
The Results section below is deliberately empty; it is filled in only as the
harness produces numbers, and the criteria above it are not edited afterwards.

**Question (plan-for-paper-2 §6 ST1, the make-or-break gate):** can we derive
or defensibly calibrate the null distribution of the complete
tracker–phase-estimation–resampling–cyclostationary pipeline, including its
selection and covariance-estimation steps?

## Pre-registered decision criteria

- **GO (analytic):** stages 1–3 (ideally 4) on all stage-5/6 nulls: realised
  FAR within the Clopper–Pearson CI of nominal at 0.05 and 1e-2, within
  [⅓×, 3×] at 1e-3; KS-uniformity of the null p-values not rejected at 0.01.
  Stage-7 behaviour characterised — stage-7 inflation does **not** block GO.
- **GO (honest reframe):** asymptotics fail somewhere in stages 3–6 **but**
  full-pipeline surrogate calibration holds at 0.05/1e-2 on stage-5/6 nulls
  **and** the order-folded statistic beats fixed multitaper-F on TPR@1e-2 in
  the wander regime (drift ≳ 0.2 Hz) by a stated margin → the method is
  reported as "an effective score with per-trace surrogate calibration",
  worded per plan-for-paper-2 §3.2.
- **NO-GO:** both calibrations miss >3× at 1e-2 on stationary
  Gaussian/coloured nulls, **or** the calibrated adaptive pipeline shows no
  power advantage over fixed multitaper-F anywhere on the wander sweep →
  plan-§6 fallback-honesty assessment.

**Three named deltas** to report, answering the plan's open validity issues:

1. **path-selection inflation** (stage 3 → 4): what the tracker's effective
   search over paths adds beyond the held-out-phase design;
2. **estimated-warp inflation** (stage 2 → 3): what estimating the phase path
   (rather than being handed it) adds;
3. **resampling-alone effect** (stage 1 → 2): what the angle-domain resampling
   operation itself does to the fixed-α null.

**Sizing arithmetic:** M = 10⁴ null traces per (stage × null × calibration)
cell resolves a 3× inflation at the 1e-3 nominal level — at k ≈ 10 expected
false alarms, the Clopper–Pearson 95% CI is ≈ [0.5, 1.8]×10⁻³, cleanly
separated from 3×10⁻³. Surrogate cells run M = 2,000 × S = 999 and are scoped
to 0.05/1e-2 only (the minimum surrogate p-value 1/(S+1) = 1e-3 is degenerate
at the 1e-3 level — a stated scope limit).

**Interpretation contract (stage-7 nulls):** tvar / am_walk / mean_drift /
controller / controller_ar1 deliberately violate the weak-dependence
stationarity assumptions. FAR inflation there is *information* — it maps the
causal-attribution boundary for paper §1.4 — not a defect and not a gate
failure. Only stage-5/6 miscalibration counts against GO.

## Results

*(empty at pre-registration, 2026-07-21 — to be filled as the FAR harness
produces calibration tables, newest at the bottom; the criteria above are
frozen.)*

### 2026-07-21 — stage 1 × stage-5 nulls: first calibration table (task 19.4)

M = 10,000/cell, seed 0; dg_fixed at the nominal Ko cadence α = 1.0 Hz, lag
set (0,1,2,3,5,8), shrinkage 0.05; mtf as the comparison row. Full numbers in
`results/st1/far_summary.json`; realised FAR [CP 95% CI]:

| cell | 0.05 | 1e-2 | 1e-3 | KS p |
|---|---|---|---|---|
| white / dg_bartlett | 0.0557 [.0513, .0604] | 0.0117 [.0097, .0140] | 0.0013 [.0007, .0022] | 2.1e-5 |
| white / dg_batch | 0.1304 | 0.0481 | 0.0095 | ~0 |
| white / mtf | 0.1846 | 0.0467 | 0.0062 | ~0 |
| ar1 / dg_bartlett | 0.0004 | 0 [0, 3.7e-4] | 0 | ~0 |
| ar1 / dg_batch | 0.0022 | 0.0004 | 0 | ~0 |
| ar1 / mtf | 0.1825 | 0.0413 | 0.0057 | ~0 |
| ar2_resonant / dg_bartlett | 0 | 0 | 0 | ~0 |
| ar2_resonant / dg_batch | 0.0003 | 0 | 0 | ~0 |
| ar2_resonant / mtf | 0.1799 | 0.0442 | 0.0043 | ~0 |

Zero covariance-conditioning failures in 90,000 traces.

**Reading (stage 1 only; criteria unchanged):**

- **dg_bartlett / white:** near-nominal — mildly anti-conservative at 0.05
  (CI excludes nominal by a hair), within CI at 1e-2, and 1.3× at 1e-3 (well
  inside the pre-registered [⅓×, 3×]); KS rejected at 0.01 but mildly. The
  earliest kill-signal is NOT triggered: the fixed-α χ² tail is usable on the
  white null.
- **dg_bartlett / coloured (ar1, ar2_resonant):** strongly CONSERVATIVE
  (realised FAR ≈ 0 at every level). Level-safe but power-costing, and the
  strict two-sided GO-analytic coverage is NOT met on coloured nulls with
  this configuration. Diagnostic probe: shrinkage 0.05 → 0 moves ar1 FAR@0.05
  from 0.000 to 0.005 — shrinkage explains part, the rest is intrinsic to the
  Bartlett bandwidth b = ⌊N^⅓⌋ on the demodulated product series (the
  autocovariance oscillates at α; a triangular truncation at b = 18 ≈ one α
  period over-estimates the long-run variance). This is exactly the
  pre-registered stage-8 covariance-estimator/bandwidth sweep axis.
- **dg_batch:** anti-conservative everywhere on white (2.6× at 0.05 up to
  9.5× at 1e-3). Consistent with Wishart noise: a 12-dim covariance from
  ~77 batch means without a Hotelling-type F correction. As configured it is
  not a usable analytic calibration.
- **mtf:** the analytic Bonferroni-over-Rayleigh correction is
  ANTI-conservative ~3.6–6×, uniformly across nulls — opposite to the
  a-priori "over-corrects" expectation. The padded-grid max-F behaves like
  ~5× more effective independent tests than the Rayleigh count (the F ratio
  fluctuates below the Rayleigh scale). The mtf analytic tail is a ranking
  statistic only; its operative calibration is the harness's Monte-Carlo one.

**Interim verdict signal:** no NO-GO indication (NO-GO requires >3×
*inflation* at 1e-2 on stationary Gaussian/coloured for both calibrations —
we see deflation on coloured, and near-nominal white behaviour). Strict
GO-analytic already looks unlikely to hold two-sided on coloured nulls
without re-tuning the covariance estimator (an explicitly open, swept
choice); the trajectory points at GO-analytic-after-sweep or GO-honest-
reframe. Decision remains open until stages 2–4 and the stage-6/7 nulls land.

### 2026-07-21 — stage ladder 2/3/4 × stage-5 nulls: the three deltas (tasks 19.5–19.7)

M = 10,000/cell, seed 0, dg/bartlett arm only (per this chunk's scope; the
batch-means and surrogate arms run with the stage-6/7 grids). Stage 2 pairs
each null trace with a fresh INDEPENDENT known warp (OU wander, 0.2 Hz, the
pre-registered wander scale); stage 3 estimates the warp on held-out 30 s EST
blocks with 2 s guards and scores TEST-mapped angle samples only; stage 4 is
the fully adaptive full-trace Viterbi pipeline. All paths sub-bin refined
(parabolic; see implementation note below). Realised FAR [CP 95% CI]; the
stage-1 rows repeat the 19.4 table for the ladder read:

| cell (dg_bartlett) | 0.05 | 1e-2 | 1e-3 | KS p |
|---|---|---|---|---|
| stage1 / white | 0.0557 [.0513, .0604] | 0.0117 [.0097, .0140] | 0.0013 [.0007, .0022] | 2.1e-5 |
| stage2 / white | 0.0164 [.0140, .0190] | 0.0027 [.0018, .0039] | 0.0003 [.0001, .0009] | ~0 |
| stage3 / white | 0.0162 [.0138, .0189] | 0.0030 [.0020, .0043] | 0.0001 [.0000, .0006] | ~0 |
| stage4 / white | 0.0175 [.0150, .0203] | 0.0022 [.0014, .0033] | 0.0005 [.0002, .0012] | ~0 |
| stage1 / ar1 | 0.0004 | 0 [0, 3.7e-4] | 0 | ~0 |
| stage2 / ar1 | 0.0002 | 0 | 0 | ~0 |
| stage3 / ar1 | 0.0003 | 0 | 0 | ~0 |
| stage4 / ar1 | 0.0007 [.0003, .0014] | 0 | 0 | ~0 |
| stage1 / ar2_resonant | 0 | 0 | 0 | ~0 |
| stage2 / ar2_resonant | 0.0001 | 0 | 0 | ~0 |
| stage3 / ar2_resonant | 0 | 0 | 0 | ~0 |
| stage4 / ar2_resonant | 0 | 0 | 0 | ~0 |

Zero covariance-conditioning failures in the 90,000 new traces.

**The three pre-registered deltas (at 0.05 / 1e-2 on white, the only null
where stage 1 was near-nominal):**

1. **Resampling-alone (1→2): a ~3.4× DEFLATION** (0.0557 → 0.0164 at 0.05;
   0.0117 → 0.0027 at 1e-2; CIs cleanly separated). Angle-domain resampling
   under a known independent warp makes the fixed-order DG null strictly
   MORE conservative, not less. Mechanism (diagnosed, not assumed): the
   cubic interpolation onto the angle grid (~32 samples per nominal cycle vs
   20 Hz sampling, i.e. ~1.6× oversampling at f0 = 1 Hz, more when the warp
   dips low) correlates adjacent angle samples; the Bartlett HAC picks that
   extra short-range correlation up and inflates Σ̂, shrinking Q. Same
   direction as the coloured-null conservatism of stage 1 — interpolation
   COLOURS the series.
2. **Estimated-warp (2→3): nil** (0.0164 → 0.0162 at 0.05, 0.0027 → 0.0030
   at 1e-2; every CI pair overlaps). Estimating the warp on held-out blocks
   rather than being handed it moves the null NOWHERE on any stage-5 null —
   the held-out design does what it promises under the null.
3. **Path-selection (3→4): no significant inflation** (0.0162 → 0.0175 at
   0.05; 0.0001 → 0.0005 at 1e-3; 0.0003 → 0.0007 on ar1 at 0.05 — every CI
   pair overlaps; largest point ratio ~2.3× on ar1 at 0.05 where both rates
   are O(10⁻⁴) and deeply conservative). The tracker's free search over
   paths, scored on the same data, does NOT measurably re-inflate the null
   past the resampling deflation. Even on ar2_resonant — an in-band spectral
   peak the Viterbi provably locks onto — stage 4 fires never in 10,000
   traces at any level: locking onto a broad noise resonance does not
   manufacture the phase-coherent cyclic structure the DG statistic needs.

**Reading (criteria unchanged):** the adaptive pipeline's null SURVIVES in
the level-safety sense at every stage of the ladder: realised FAR ≤ nominal
everywhere, at all three levels, on all stage-5 nulls, including the fully
adaptive stage 4. The NO-GO trigger (>3× inflation at 1e-2 on stationary
Gaussian/coloured) is nowhere in sight — the failure mode the gate was built
to catch (adaptivity silently inflating the false-alarm rate) did not
materialise. What DOES fail is strict two-sided GO-analytic coverage: after
resampling the pipeline is uniformly conservative (~3–6× deflation at
0.05/1e-2 on white; effectively 0 on coloured), and KS-uniformity is
rejected everywhere. The asymptotic χ² tail is a level-safe but power-
costing bound, not a calibration.

**Implementation finding (affects power, not level):** the pooled coherent
DG sum needs the warp accurate to ~1/(2πT) Hz — a few mHz at 300 s — while
the spectrogram bins are 62.5 mHz. Bin-quantised Viterbi paths make stages
3–4 powerless at ANY SNR; parabolic sub-bin refinement (added to both
stages) restores stage-4 power in the slow-wander regime (OU correlation
~200 s) but stage 3's EST-anchored warp decoheres over 300 s for faster
wander. The honest power boundary — and the case for a semi-coherent
per-block variant — is quantified in the 19.10 bake-off; for the wander
regime of the GO-reframe criterion (drift ≳ 0.2 Hz at correlation ~50 s)
the pooled statistic currently has little power at either stage 3 or 4.

**Interim verdict signal (updated):** still no NO-GO. GO-analytic in the
strict two-sided sense is dead on this configuration (conservatism, not
inflation, is the miss). The live decision is between GO-analytic-after-
stage-8-sweep (re-tune covariance bandwidth/shrinkage to close the
deflation while keeping level safety) and GO-honest-reframe (surrogate
calibration + demonstrated power advantage in the wander regime — where
the power finding above is the risk). Stages 6–7, the sweeps, and the
bake-off (chunk 3) decide.

### 2026-07-21 — stage-6/7 nulls × full stage ladder: the two-sided GO test and the attribution boundary (task 19.9a)

M = 10,000/cell, seed 0. dg/bartlett across stages 1–4; dg_batch at stage 1
(documenting its miscalibration across the suite). Implementation note: the
controller null's RK4 inner loop was rewritten on scalar floats
(bit-identical output, verified on multiple seeds; ~16× faster) to make
these cells tractable. Zero covariance-conditioning failures in all 400,000
new traces.

**Stage 6 (stationary non-Gaussian), dg_bartlett — the two-sided GO-analytic
test:**

| cell (dg_bartlett) | 0.05 | 1e-2 | 1e-3 | KS p |
|---|---|---|---|---|
| stage1 / ar1_t | 0.0001 [2.5e-06, 0.00056] | 0 [0, 0.00037] | 0 [0, 0.00037] | ~0 |
| stage1 / lognormal | 0 [0, 0.00037] | 0 [0, 0.00037] | 0 [0, 0.00037] | ~0 |
| stage1 / sq_gauss | 0 [0, 0.00037] | 0 [0, 0.00037] | 0 [0, 0.00037] | ~0 |
| stage2 / ar1_t | 0 [0, 0.00037] | 0 [0, 0.00037] | 0 [0, 0.00037] | ~0 |
| stage2 / lognormal | 0 [0, 0.00037] | 0 [0, 0.00037] | 0 [0, 0.00037] | ~0 |
| stage2 / sq_gauss | 0 [0, 0.00037] | 0 [0, 0.00037] | 0 [0, 0.00037] | ~0 |
| stage3 / ar1_t | 0.0003 [6.2e-05, 0.00088] | 0 [0, 0.00037] | 0 [0, 0.00037] | ~0 |
| stage3 / lognormal | 0 [0, 0.00037] | 0 [0, 0.00037] | 0 [0, 0.00037] | ~0 |
| stage3 / sq_gauss | 0.0002 [2.4e-05, 0.00072] | 0 [0, 0.00037] | 0 [0, 0.00037] | ~0 |
| stage4 / ar1_t | 0.0003 [6.2e-05, 0.00088] | 0.0001 [2.5e-06, 0.00056] | 0 [0, 0.00037] | ~0 |
| stage4 / lognormal | 0.0002 [2.4e-05, 0.00072] | 0 [0, 0.00037] | 0 [0, 0.00037] | ~0 |
| stage4 / sq_gauss | 0.0026 [0.0017, 0.0038] | 0.0002 [2.4e-05, 0.00072] | 0 [0, 0.00037] | ~0 |

**NO stage-6 null inflates the analytic FAR** — every cell sits far BELOW
nominal at every level and every stage (heavy tails, skew, and χ²-marginals
do not break the χ² tail upward; they inherit the coloured-null
conservatism, as all three stage-6 nulls are AR(1)-coloured). The two-sided
GO-analytic criterion fails on stage 6 exactly as on the coloured stage-5
nulls — by deflation, never inflation.

**Stage 7 (locally stationary / controller), dg_bartlett — the attribution
boundary:**

| cell (dg_bartlett) | 0.05 | 1e-2 | 1e-3 | KS p |
|---|---|---|---|---|
| stage1 / tvar | 0.0007 [0.00028, 0.0014] | 0 [0, 0.00037] | 0 [0, 0.00037] | ~0 |
| stage1 / am_walk | 0.0003 [6.2e-05, 0.00088] | 0 [0, 0.00037] | 0 [0, 0.00037] | ~0 |
| stage1 / mean_drift | 0.0157 [0.013, 0.018] | 0.0028 [0.0019, 0.004] | 0.0001 [2.5e-06, 0.00056] | ~0 |
| stage1 / controller | 0 [0, 0.00037] | 0 [0, 0.00037] | 0 [0, 0.00037] | ~0 |
| stage1 / controller_ar1 | 0 [0, 0.00037] | 0 [0, 0.00037] | 0 [0, 0.00037] | ~0 |
| stage2 / tvar | 0.0002 [2.4e-05, 0.00072] | 0 [0, 0.00037] | 0 [0, 0.00037] | ~0 |
| stage2 / am_walk | 0.0001 [2.5e-06, 0.00056] | 0.0001 [2.5e-06, 0.00056] | 0 [0, 0.00037] | ~0 |
| stage2 / mean_drift | 0.0005 [0.00016, 0.0012] | 0.0001 [2.5e-06, 0.00056] | 0 [0, 0.00037] | ~0 |
| stage2 / controller | 0.0008 [0.00035, 0.0016] | 0.0003 [6.2e-05, 0.00088] | 0.0001 [2.5e-06, 0.00056] | ~0 |
| stage2 / controller_ar1 | 0.0008 [0.00035, 0.0016] | 0.0001 [2.5e-06, 0.00056] | 0 [0, 0.00037] | ~0 |
| stage3 / tvar | 0 [0, 0.00037] | 0 [0, 0.00037] | 0 [0, 0.00037] | ~0 |
| stage3 / am_walk | 0.0003 [6.2e-05, 0.00088] | 0 [0, 0.00037] | 0 [0, 0.00037] | ~0 |
| stage3 / mean_drift | 0.001 [0.00048, 0.0018] | 0.0003 [6.2e-05, 0.00088] | 0 [0, 0.00037] | ~0 |
| stage3 / controller | 0 [0, 0.00037] | 0 [0, 0.00037] | 0 [0, 0.00037] | ~0 |
| stage3 / controller_ar1 | 0 [0, 0.00037] | 0 [0, 0.00037] | 0 [0, 0.00037] | ~0 |
| stage4 / tvar | 0.0003 [6.2e-05, 0.00088] | 0.0001 [2.5e-06, 0.00056] | 0 [0, 0.00037] | ~0 |
| stage4 / am_walk | 0.0002 [2.4e-05, 0.00072] | 0 [0, 0.00037] | 0 [0, 0.00037] | ~0 |
| stage4 / mean_drift | 0.0241 [0.021, 0.027] | 0.0063 [0.0048, 0.0081] | 0.0013 [0.00069, 0.0022] | ~0 |
| stage4 / controller | 0 [0, 0.00037] | 0 [0, 0.00037] | 0 [0, 0.00037] | ~0 |
| stage4 / controller_ar1 | 0 [0, 0.00037] | 0 [0, 0.00037] | 0 [0, 0.00037] | ~0 |

**Which stage-7 nulls fire?** Under the analytic dg/bartlett calibration:
NONE. The a-priori expectation — the controller limit cycle fires the
structural test — did NOT materialise: controller / controller_ar1 sit at
realised FAR 0 in 10,000 traces at every stage and level (0.31–0.45 Hz
relaxation cycle at ~5σ amplitude inflates the long-run covariance by
orders of magnitude, and its per-cycle phase re-jitter decoheres the fixed-
order cyclic moments — the conservatism mechanism swallows the confuser
whole). The only null that moves the analytic FAR upward is **mean_drift**
(first-order nonstationarity: ramp + level steps), peaking at stage 4 with
0.0241 at nominal 0.05 (½× nominal), 0.0063 at 1e-2, and 0.0013 at 1e-3
(1.3×, CP CI straddles nominal). Even the "worst" stage-7 cell therefore
stays at or below nominal — under THIS calibration the causal-attribution
boundary of paper §1.4 is bounded by first-order drift, not by periodic
confusers, and stage-7 behaviour is characterised as the criteria require
(no gate implication either way).

**dg_batch at stage 1 across the new nulls** (the documented-miscalibration
arm; stage-5 rows in the 19.4 table):

| cell (dg_batch) | 0.05 | 1e-2 | 1e-3 | KS p |
|---|---|---|---|---|
| stage1 / ar1_t | 0.0015 [0.00084, 0.0025] | 0.0008 [0.00035, 0.0016] | 0.0006 [0.00022, 0.0013] | ~0 |
| stage1 / lognormal | 0.0046 [0.0034, 0.0061] | 0.0033 [0.0023, 0.0046] | 0.0024 [0.0015, 0.0036] | ~0 |
| stage1 / sq_gauss | 0.0013 [0.00069, 0.0022] | 0.0002 [2.4e-05, 0.00072] | 0 [0, 0.00037] | ~0 |
| stage1 / tvar | 0.0038 [0.0027, 0.0052] | 0.0005 [0.00016, 0.0012] | 0 [0, 0.00037] | ~0 |
| stage1 / am_walk | 0.0009 [0.00041, 0.0017] | 0 [0, 0.00037] | 0 [0, 0.00037] | ~0 |
| stage1 / mean_drift | 0.0526 [0.048, 0.057] | 0.0147 [0.012, 0.017] | 0.0027 [0.0018, 0.0039] | ~0 |
| stage1 / controller | 0.0001 [2.5e-06, 0.00056] | 0 [0, 0.00037] | 0 [0, 0.00037] | ~0 |
| stage1 / controller_ar1 | 0 [0, 0.00037] | 0 [0, 0.00037] | 0 [0, 0.00037] | ~0 |

dg_batch is conservative on the coloured/non-Gaussian suite (its white-null
anti-conservatism from 19.4 does not generalise into coloured inflation) but
INFLATES on mean_drift: 0.0147 at 1e-2 (1.5×, CI excludes nominal) and
0.0027 at 1e-3 (2.7×, CI excludes nominal) — a stage-7 cell, so information
rather than a gate failure, but it reinforces 19.4: batch means is not a
usable analytic calibration (heavy tails also leak into its deep tail:
lognormal realised 2.4e-3 at nominal 1e-3).

**Reading against the criteria (criteria unchanged):** the NO-GO trigger
(>3× inflation at 1e-2 on stationary Gaussian/coloured nulls for both
calibrations) remains absent — there is NO inflation on any stage-5/6 null
for either estimator. Strict GO-analytic coverage fails on stage-6 exactly
as on stage-5-coloured: by conservatism. The decision therefore hinges on
the pre-registered escape routes measured next: (i) the stage-8 covariance
sweep (is the deflation a fixable bandwidth/shrinkage artefact?) and
(ii) the surrogate arm + bake-off (GO-honest-reframe).

### 2026-07-21 — stage-8 sweeps: the swept open choices and the GO-analytic verdict (task 19.9b)

M = 5,000/point, seed 0, one-at-a-time around two reference cells — stage4 ×
ar1 (the coloured-conservatism cell, the fix-or-reframe axis) and stage1 ×
white (the clean anchor whose calibration must not be broken); the cov axis
adds stage4 × white (where the resampling deflation lives). Full numbers in
`results/st1/sweeps/{duration,lags,cov,nperseg,band,interp}.json`; figures
`figures/st1_calibration_{far,qq,cov}`. Zero covariance-conditioning
failures at every point.

**Selection rule (recorded before reading the tables, also in
`scripts/st1_sweeps.py` and each sweep JSON):** any configuration promoted
from these sweeps is chosen on NULL-CALIBRATION QUALITY ACROSS THE SUITE
ONLY — including the stage-1 anchor — never on signal power. Power enters
only afterwards, in the 19.10 bake-off, at whatever configuration the
calibration already fixed.

**Non-covariance axes: the conservatism is not their artefact.**

- *duration* {75, 150, 300, 600 s}: stage4/ar1 stays ≤ 1e-3 realised at
  nominal 0.05 at every duration — the deflation is not small-T asymptotia
  that longer records repair.
- *lags*: no rescue on ar1 at any lag set (dense 0..12 is the most
  conservative). Honest bright spot: stage1/white with the SHORT set (0,1,2)
  is the best-calibrated cell in the entire campaign — 0.0482 / 0.0124 /
  0.0010 with KS p = 0.88 (uniformity NOT rejected) — the fixed-α χ² theory
  is essentially exact when the covariance dimension is small (2L = 6);
  miscalibration grows with L and after resampling.
- *nperseg* {0.5×, 1×, 2×} and *band* {half, nominal, double}: FAR moves by
  O(1e-3) at most on stage4/ar1 — the tracker's search geometry is not the
  lever.
- *interp* {linear, cubic, sinc} on stage2/white and stage4/white: all
  remain ~3–6× deflated at 0.05 (linear 0.0086 → sinc 0.0152 at stage 2);
  ordering matches the interpolation-colouring mechanism (sinc correlates
  neighbours least) but no scheme closes the gap — the interpolation CHOICE
  is not the fix either.

**The covariance axis (THE decision axis).** Bartlett bandwidth b ∈
{6, 18 ≈ default, 54} × shrinkage λ ∈ {0, 0.01, 0.05}, plus batch means ×
λ. Realised FAR at nominal 1e-2 [CP 95% CI], the pivotal cells:

| config | stage4/ar1 | stage4/white | stage1/white |
|---|---|---|---|
| b=18, λ=0.05 (default) | 0.0001 | 0.0016 | 0.0108 [.0081, .014] |
| b=18, λ=0 | 0.0012 [.0004, .0026] | 0.0026 | 0.0132 [.010, .017] |
| b=54, λ=0 | **0.0118 [.009, .015]** | **0.0100 [.0074, .013]** | **0.0188 [.015, .023]** |
| batch, λ=0 | 0.0472 (4.7×) | 0.0282 | 0.0532 |

- Shrinkage is the smaller lever, but even λ = 0.01 re-kills the coloured
  cell at b = 18 (0.0012 → 0); bandwidth is the live knob.
- **b = 54, λ = 0 repairs stage 4 two-sided**: ar1 0.0504/0.0118/0.0016 and
  white 0.0406/0.0100/0.0012 across the three levels — nominal at 1e-2
  (CIs cover), within [⅓×, 3×] at 1e-3. This mechanistically CONFIRMS the
  chunk-1/2 diagnosis: the deflation is an under-bandwidth artefact — the
  Bartlett window must span several α periods of the oscillating
  autocovariance of the (interpolation-coloured) demodulated product
  series, not the ⌊N^⅓⌋ ≈ one period the default gives.
- **But the SAME absolute b breaks the stage-1 anchor**: stage1/white at
  b = 54 inflates to 0.0726 [.066, .080] at 0.05 and 0.0188 [.015, .023]
  at 1e-2 — CIs exclude nominal — the classic over-bandwidth HAC failure
  (Σ̂ variance leaks into the χ² tail). No single (estimator, b, λ) in the
  swept family calibrates the ladder end-to-end.
- **KS-uniformity is rejected at every stage-4 sweep point** (best
  3.3e-6, at the b = 54 repair point itself) — even where the three tail
  rates sit at nominal, the analytic p-values are not uniform.

**GO-analytic verdict from the sweep (criteria unchanged): NOT ACHIEVED —
no candidate configuration found, so no confirming full-grid re-run is
triggered.** The pre-registered GO-analytic bar is stages 1–3 (ideally 4)
covered two-sided on ALL stage-5/6 nulls at 0.05/1e-2 with KS-uniformity
not rejected. b = 54/λ = 0 passes the tail rates on the stage-4 cells it
was swept on but (i) inflates the stage-1 anchor with CIs excluding
nominal — failing the across-the-suite selection rule inside the sweep
itself — and (ii) fails KS-uniformity everywhere adaptive. A
stage-dependent bandwidth (default b before resampling, ~3× after) would
fix the point rates but is post-hoc per-stage tuning outside the
pre-registered criteria, and still fails (ii). The analytic χ² tail
therefore remains what chunk 2 called it: a level-safe, power-costing
bound. The gate decision now rests on the GO-honest-reframe legs: the
surrogate arm and the bake-off.

### 2026-07-21 — detector bake-off on the wander axis (task 19.10)

n = 200/class, seed 20260721; signal = Ko training traces over f0_drift_hz ∈
{0, 0.1, 0.2, 0.4, 0.8, 1.5} Hz; negatives scored separately as (a) the Ko
INFERENCE null (the b1 comparison) and (b) a MIXED STRUCTURAL population
(equal parts ar1 / ar1_t / controller). TPR at empirical-FAR thresholds from
each negative class's own scores (with n = 200 the 1e-2 threshold rests on
the 2nd-highest negative score — granularity stated, not hidden). Full
numbers `results/st1/bakeoff_summary.json`; figure `figures/st1_bakeoff`.

**Semi-coherent variant admitted after its pre-registered level spot-check**
(stage4_semicoh × {white, ar1}, M = 5,000, analytic Fisher-χ² tail, in
`far_summary.json`): white 0.0084 / 0.0016 / 0.0002 at 0.05 / 1e-2 / 1e-3,
ar1 0 everywhere — conservative like the rest of the family, NO inflation
from the shared-tracker coupling of its "independent"-blocks Fisher
combination. Level-safe → included.

**TPR @ empirical FAR 1e-2** (rows = detector; per drift 0 / 0.1 / 0.2 /
0.4 / 0.8 / 1.5 Hz):

vs inference null:

| detector | 0 | 0.1 | 0.2 | 0.4 | 0.8 | 1.5 |
|---|---|---|---|---|---|---|
| mtf (fixed multitaper F) | 0.02 | 0.01 | 0.01 | 0.01 | 0.02 | 0.00 |
| spectral matched filter | 0.94 | 0.53 | 0.32 | 0.20 | 0.31 | 0.33 |
| viterbi tracker | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.98 |
| dg_order_split (st.3) | 0.28 | 0.17 | 0.04 | 0.01 | 0.01 | 0.01 |
| dg_order_full (st.4) | 0.73 | 0.62 | 0.55 | 0.23 | 0.06 | 0.01 |
| dg_order_semicoh | 1.00 | 1.00 | 0.97 | 0.56 | 0.07 | 0.01 |
| dg_fixed_oracle (true f0) | 0.34 | 0.07 | 0.01 | 0.01 | 0.00 | 0.00 |

vs mixed structural nulls:

| detector | 0 | 0.1 | 0.2 | 0.4 | 0.8 | 1.5 |
|---|---|---|---|---|---|---|
| mtf | 0.00 | 0.01 | 0.00 | 0.00 | 0.01 | 0.00 |
| spectral matched filter | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| viterbi tracker | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| dg_order_split | 0.30 | 0.17 | 0.04 | 0.01 | 0.01 | 0.01 |
| dg_order_full | 0.80 | 0.70 | 0.61 | 0.31 | 0.10 | 0.01 |
| dg_order_semicoh | 1.00 | 1.00 | 0.99 | 0.69 | 0.14 | 0.03 |
| dg_fixed_oracle | 0.36 | 0.10 | 0.03 | 0.01 | 0.01 | 0.00 |

**Hard case — drift 0.4 Hz vs CONTROLLER-ONLY negatives, TPR @ 0.05 / 1e-2:**
dg_order_semicoh 1.00 / 1.00; dg_order_full 0.94 / 0.905; dg_order_split
0.875 / 0.805; mtf 0.035 / 0.015; spectral 0 / 0; viterbi 0 / 0;
dg_fixed_oracle 0.065 / 0.

**Reading:**

- **The GO-reframe power leg is satisfied with room to spare.** The
  pre-registered margin question — order-folded DG beats fixed
  multitaper-F on TPR@1e-2 in the wander regime (drift ≳ 0.2 Hz) — reads:
  at drift 0.2 / 0.4 Hz, dg_order_full 0.55 / 0.23 vs mtf 0.01 / 0.01
  (inference negatives) and 0.61 / 0.31 vs 0.00 / 0.00 (structural);
  dg_order_semicoh 0.97 / 0.56 and 0.99 / 0.69. Margin ≥ +0.22 TPR
  everywhere in the regime, ≥ +0.5 for the semi-coherent variant. (Stated
  fairly: mtf has essentially NO usable power against either negative
  class at empirical thresholds — both negative classes carry enough
  spectral structure to push its threshold above the signal scores.)
- **Only the DG order family separates the wandering line from structural
  confusers.** Against the mixed structural negatives, spectral and
  viterbi have TPR = 0 at every drift (the controller third of the mix
  fires them harder than the signal does: the Viterbi score happily locks
  onto a limit cycle). The hard case makes it explicit: at drift 0.4 vs
  controller-only nulls the order pipeline keeps 0.8–1.0 TPR at 1e-2
  while every spectral-shape detector — including the ORACLE fixed-α DG —
  drops to ~0. Phase-coherent cyclic structure at the tracked order is
  the only signature the confuser cannot fake; this is the
  causal-attribution boundary of paper §1.4 drawn with power, not FAR.
- **The semi-coherent variant strictly dominates the pooled statistic**
  at every drift on both negative classes (e.g. 0.97 vs 0.55 at 0.2 Hz,
  0.56 vs 0.23 at 0.4 Hz), confirming the chunk-2 decoherence diagnosis
  and largely filling the flagged fast-wander power hole up to ~0.4 Hz.
  Its Fisher combination stays level-safe (spot-check above).
- **The honest power boundary:** at drift ≥ 0.8 Hz every calibratable
  detector fades (semicoh 0.07–0.14 at 1e-2); only the raw Viterbi score
  retains power vs the inference null there, and it cannot tell signal
  from a controller. 300 s records do not resolve fast wander with an
  honest null — a scope statement for the paper, not a repairable defect.
- dg_fixed_oracle collapsing from 0.34 (drift 0) to ≤ 0.1 (any wander)
  quantifies WHY tracking is needed at all: even the true f0 is useless
  to a coherent fixed-α statistic once the line wanders.

### 2026-07-22 — surrogate arm (stage-9 comparison), rescoped after a crash (task 19.9/19.10)

**Provenance and sizing.** The originally scoped surrogate grid (15 cells:
stages 1/3/4 × {white, ar1, ar1_t, tvar, controller}, M = 2,000, S = 199)
crashed after ~3.5 h with only its five stage-1 cells computed and — because
the harness then wrote results only at end-of-grid — nothing persisted. The
harness now persists every cell as it completes (commit "persist each FAR
cell as it completes"), and the arm was re-run at the decision-relevant
minimum: **stages 1 and 4 × {white, ar1, tvar, controller}, M = 1,000,
S = 199**, scoped to nominal 0.05 and 1e-2. Stage 3 and the stage-6
non-Gaussian nulls are not covered by the surrogate arm at this sizing; a
confirming run at S = 999, M = 10,000 including them is earmarked for an
OzSTAR slurm array before any paper number is frozen. CIs below are
Clopper–Pearson 95%.

**Realised FAR under full-pipeline Fourier-phase surrogate calibration:**

| cell | nominal 0.05 | nominal 1e-2 | KS p |
|---|---|---|---|
| stage1/white | 0.047 [.035, .062] | 0.009 [.004, .017] | 0.093 |
| stage1/ar1 | 0.057 [.044, .073] | 0.014 [.008, .023] | 0.50 |
| stage4/white | 0.050 [.037, .065] | 0.006 [.002, .013] | 0.41 |
| stage4/ar1 | 0.049 [.037, .064] | 0.006 [.002, .013] | 0.048 |
| stage1/tvar | 0.037 [.026, .051] | 0.005 [.002, .012] | 8e-11 |
| stage4/tvar | 0.021 [.013, .032] | 0.002 [.000, .007] | 4e-29 |
| stage1/controller | 0.168 [.145, .193] | 0.054 [.041, .070] | 8e-69 |
| stage4/controller | 0.286 [.258, .315] | 0.094 [.077, .114] | 2e-149 |

**Reading.**

- **On the tested stage-5 nulls the surrogate calibration is exact where the
  asymptotic tail was 3–6× conservative**: all four white/ar1 cells cover
  nominal at both levels at BOTH the fixed-α stage and the fully adaptive
  stage 4, and KS-uniformity is not rejected at 0.01 anywhere among them —
  the property the analytic p-values never achieved at any swept
  configuration. Running the tracker inside every surrogate does what it is
  supposed to do.
- **tvar (stage 7)** stays mildly conservative — locally stationary
  amplitude structure widens the surrogate null's spread more than the
  observed statistic. Level-safe.
- **controller (stage 7) inflates, and must**: a limit cycle is itself
  cyclostationary, so it violates the linear-stationary invariance that
  Fourier-phase surrogates encode. The detector correctly reports coherent
  cyclic structure in a controller trace; calling that structure "training"
  is the causal-attribution step that needs the §1.4 nuisance model. This is
  the interpretation contract playing out with numbers (fire rate 0.17–0.29
  at nominal 0.05), consistent with the analytic stage-7 characterisation
  and with the bake-off's structural-negative results.

### 2026-07-22 — GATE VERDICT

**GO (honest reframe)** — per the pre-registered criteria, with one scope
qualification stated below.

- **GO-analytic: failed** (19.9b). No swept (estimator, bandwidth,
  shrinkage) configuration calibrates the ladder end-to-end two-sided, and
  KS-uniformity of the analytic p-values is rejected at every adaptive
  sweep point. The analytic χ² tail survives only as what chunk 2 called
  it: a level-safe, power-costing conservative bound (no inflation on any
  stage-5/6 null anywhere in the campaign: the two-sided failure is
  deflation, never anti-conservatism).
- **GO-reframe leg (a) — asymptotics fail somewhere in stages 3–6:**
  satisfied (the stage-2/3/4 deflation; 19.9b).
- **GO-reframe leg (b) — full-pipeline surrogate calibration holds at
  0.05/1e-2 on stage-5/6 nulls:** satisfied on the nulls tested (white,
  ar1 — including the fully adaptive stage 4), with KS-uniformity not
  rejected. **Scope qualification:** the crash-rescoped arm did not cover
  the stage-6 non-Gaussian nulls or stage 3; the pre-registered wording
  says "stage-5/6 nulls", so this leg is satisfied on partial evidence.
  The S = 999 / M = 10,000 OzSTAR confirmation (including ar1_t,
  lognormal, sq_gauss, stage 3) is a REQUIRED follow-up before the paper
  freezes any surrogate-calibrated number; if a stage-6 cell fails there,
  this verdict is to be revisited.
- **GO-reframe leg (c) — order-folded beats fixed multitaper-F on TPR@1e-2
  at drift ≳ 0.2 Hz by a stated margin:** satisfied with room to spare
  (19.10): dg_order_full +0.22…+0.61 TPR over mtf across the regime and
  both negative classes; dg_order_semicoh +0.5 or more; and only the DG
  order family retains any power against the mixed-structural and
  controller-only negatives (viterbi and the matched filter: TPR = 0
  there).
- **NO-GO triggers:** neither fires (no >3× miscalibration at 1e-2 under
  both calibrations on stationary nulls; the adaptive pipeline's power
  advantage over fixed multitaper-F is large and reproducible).

**What the paper may now claim (§3.2 wording).** The tracked
order/cyclostationary pipeline is reported as **an effective score with
per-trace surrogate calibration**; the surrogate null is the
linear-stationary invariance and is stated as such. The asymptotic χ²
threshold may be quoted only as a conservative bound. "Analytic CFAR" is
not claimable for the adaptive pipeline — the pre-registered phrase-level
discipline stands. Rung 1's structural claim is supported with power, not
just level: phase-coherent cyclic structure at the tracked order is the
one signature the measured-controller confuser class cannot fake (hard
case: 1.00/1.00 vs 0/0 for every spectral-shape detector), while fast
wander (≥ 0.8 Hz on 300 s records) is an honest, stated power boundary.

**Follow-ups carried out of the gate** (tracked in tasks.md): (1) the
OzSTAR S = 999/M = 10⁴ surrogate confirmation incl. stage 6 and stage 3;
(2) register {mtf, dg_order_full, dg_order_semicoh} into the ST2 sweep
(task 20.9) to definitise the frontier verdict; (3) the semi-coherent
variant's block length and the stage-dependent-bandwidth observation are
stage-8 axes to revisit in Phase 1, not silent defaults.

### 2026-07-24 — surrogate arm frozen at full sizing (follow-up (1) closed)

The S = 999 surrogate confirmation ran as a MATS `compute` slurm array (job
5567, 24 cells: stages 1/3/4 × {white, ar1, ar2_resonant, ar1_t, lognormal,
sq_gauss, tvar, controller}) at the surrogate sizing **M = 2000 × S = 999**
(the config `n_null_surrogate_cells` cap; the analytic/level-safety/bake-off
numbers stay at M = 10⁴). Full frozen table + provenance:
`notes/results/number-freeze-2026-07-24.md`.

**Leg (b) resolution.** Confirmed on the stationary Gaussian nulls at **all
three stages** including the held-out-warp stage and the fully-adaptive stage
(white 0.051/0.009 fixed-α, 0.052/0.013 adaptive; ar1, ar2_resonant likewise;
KS-uniformity not rejected). The heavy-tailed *linear* non-Gaussian nulls
(ar1_t, lognormal) stay level-safe (conservative). **One stage-6 cell fails:**
`sq_gauss` at the fully-adaptive stage over-rejects at **0.087/0.018 (≈1.7×)** —
below the pre-registered 3× NO-GO threshold, which additionally requires
inflation under *both* calibrations (the analytic arm never inflates). Cause is
the linear-stationarity assumption the surrogate encodes: a squared-Gaussian is
*nonlinear*, so Fourier-phase surrogates under-spread it (the mild sibling of the
`controller` cyclostationary inflation). **GO stands (confirmed-with-caveat,
2026-07-24):** the level-safety claim is scoped to **linear-stationary** nulls,
and the §4 footnote now states the exception. Candidate follow-up if ever needed:
amplitude-preserving (IAAFT) surrogates.
