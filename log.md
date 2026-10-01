# Research log — certifying-training-from-power

Dated narrative of work sessions, newest at top. Append-only historical record
(forward-looking state lives in `handoff.md`).

## 2026-10-01 — §6 restructured, frontier moved to the aggregate at FAR 1e-2, cost-anchor gaps scoped

**Goal.** Get §6 (now "Detection against a concealing operator") into the same shape as §5. That meant a flow pass, the standard opener, the §2 aggregate scenario as in §5, and FAR 1e-2 throughout.

**What was tried.**
- **Text.** Restructured §6 into four subsections, led by the cost-of-hiding figure. The sweep provenance moved to App. D, the H200 evasion costs (Gargiulo) to §7.2, and the Neyman–Pearson reprise was cut to one sentence. The setup paragraph now reports the four unpriced strategies.
- **Code.** Threaded `aggregate=True` through `powerladder/typeb/st2_attacks.py` and `st2.py::run_family`. Attack knobs reach the dominant run only, via new `dominant_params` / `**train_kwargs` on `aggregate_F` and `ko_make_aggregate_trace`, and the background stays honest.
- **Smoke test.** Compared single vs aggregate on the full work/jitter/drift/shape grids at n=200 locally (about 30 s per family) before going to the cluster.
- **Freeze.** On MATS `compute`: array 14354 for the sweeps and job 14355 for the frontier, written to `results/st2/aggregate/`.
- **Figures.** Regenerated the Pareto figures with the new `--far` flag (default 0.01). Every number in §6, §7 and App. D was updated.

**What was learned.**
- **The aggregate changes nothing material.** Every cell matches the single-workload freeze within seed noise, so the ~37 dB-down background never gives the tracker a new failure mode, even under heavy smearing.
- **The FAR switch is what moves the numbers.**
  - The fixed tests hide 0.79 at about 0% cost (it was 0.74).
  - The tracker gives up at most 0.24, at 159% (it was 0.16).
  - At work level 0.7 the tracker falls to 0.46 (it was 0.59).
- **Reproducibility check.** The single-workload rerun reproduces `results/st2/work_summary.json` exactly, so the generator change is byte-identical with no knobs set.
- **The unpriced strip is about coverage, not cost.** Of its 34 cells, 27 belong to families the B2 hardware campaign never ran (phase, relocate, harmonic, dilute, meter). The other 7 are levels outside its grid: jitter and work 0.7 (the hardware grid stops at 0.5), and drift 0, 0.1, 0.4, 0.8, 1.5 (hardware ran 0.2, 0.5, 1 Hz). Drift 0 is the honest schedule and should be priced at 0.
- **Overhead calibrates itself.** Each B2 trace records its honest baseline at capture (`code/b2/analysis.py::throughput_overhead`), so pricing new levels needs only new attack traces.

**Decisions / dead ends.**
- **Pareto figure: staircase, not hull.** Built a convex-hull version (`--frontier hull`, `figures/money_pareto_aggregate_hull.*`) for comparison. Kept the staircase because every point on it is a real attack. The hull's segments are lotteries over runs, and per-run averaging overstates hiding when a verifier watches many runs.
- **MATS elastic A100: abandoned.** Job 14364 was submitted without the cost approval the MATS skill requires, then cancelled before it ran. The `t.kimpson` account is not in `elastic-fellows` anyway, so it would have sat pending forever. GPU pricing moves to OzSTAR `milan-gpu`, where the original anchors were measured.
- **Verdict FAR unchanged.** The pre-registered GO/NO-GO verdict stays at FAR 0.05; only the figures and text moved to 1e-2.

**Open threads.**
- Price the 7 off-grid cells on an OzSTAR A100 with a top-up phase of the monorepo's B2 campaign. Work 0.7 matters most: it is where the tracker bends, and §6.3 currently says it is unpriced.
- Phase slip, relocation and harmonic smoothing need new hardware schedules, and possibly a learning-efficiency cost rather than throughput.
- The monorepo has a hardware dilution measurement (`results/b2/dilute_summary.json`). It was never carried over, because its mechanism (time-sharing one card) differs from the synthetic facility-share dilution.
- Figure 7 tidy-ups were proposed but not chosen: price drift 0 at 0%; drop the meter cells from the strip; rename the strip; explain shape fill 0 at 189% (that anchor sits on a jitter-0.35 base).

## 2026-09-30 — §5 flow pass, bake-off appendix dissolved, meter spec moved to FAR 1e-2

**Goal.** Make §5 (Detection results) make its two points cleanly, and strip its
supporting appendices down to what the main text actually cites.

**What was tried.** Flow review of §5, then edits: "In this section we …" opener
plus organisation paragraph (matching §§2–4); §5.1 split into design → ceiling →
result, overclaim "sits on the ceiling" replaced by "within 0.04"; the moot
ceiling caveat (true optimum can only exceed a ceiling already at 1.0) and the
dangling periodic-confound paragraph cut, the latter folded into the section
intro. The undefined symbol `f_0^{\mathrm{drift}}` turned out to be the code name
`f0_drift_hz`, which is exactly σ_f of `eq:ou_sigma`; replaced everywhere and on
the `plot_drift_ceiling.py` axis. The NP-ceiling explanation was reduced to one
sentence in §5.1 and written out plainly in the appendix, including why it is
not a usable detector (it is fitted to our own generators, so it knows both
classes' exact spectra).

The "Detector bake-off" appendix was dissolved. The main text relied on it for
only three facts: the ceiling, the background ablation (≤0.02), and "Viterbi
cannot tell a controller limit cycle from training". It became `app:ceiling`
(ceiling + background paragraph), and the confound result moved into
`app:attribution` as `subsec:confound` with `tab:confound`. The 7-detector
framing, `fig:st1_bakeoff`, `fig:np_ceiling`, `tab:bakeoff` and the fast-wander
paragraph were cut.

§5.2: new opener stating the nominal meter (f_s = 20 Hz, τ → 0, no delivery
filter, σ_η = 4 W), the exact swept grid, a plain definition of the notch, and τ
introduced for the integration window. The sweep was switched from FAR 0.05 to
1e-2 to match §5.1. No rerun was needed, because every ST2 summary already stores
`tpr_at_far` at both 0.05 and 0.01; only a `--far` flag (default 0.01) on
`plot_st2_meter_boundary.py` and a replot were required. App. E (`app:meter`) was
cut to a provenance footnote, a Viterbi-only notch figure and paragraph, and a
background check (≤0.09 at 1e-2).

**What was learned.**
- At 1e-2 the meter boundary is unchanged: every live cell ≥ 0.945 and every dead
  cell < 0.35.
- The per-cell numbers drop slightly (1 Hz: 0.41 → 0.33; τ = 0.5 s: 0.97 → 0.94).
- The deep-notch range was misquoted before. The true range is 0.45–0.63 at 1e-2
  (0.47–0.64 at 0.05), and the 90% notch at 1.2 Hz hurts more than the full-depth
  one.
- The training cadence band is 0.5–1.5 Hz, so a 0.6 Hz notch is inside the band,
  not "below" it.
- The old App. E "frontier meter-erasure point" paragraph decomposed a point that
  §6 no longer reports (§6 defers meter degradation to §5.2). It was orphaned.

**Decisions / dead ends.**
- The semi-coherent DG row was dropped from the confound table. Its numbers have
  no committed results file: the `--semicoh` run wrote to the same
  `bakeoff_summary.json` that the default run later overwrote.
- The per-detector meter grid, the "robustness ordering" claim (false at 1e-2:
  DG-full is 0.71 at the reference channel vs 1.00 for Viterbi), and the
  BLAS/AUC note (no AUCs reported) were all cut.
- The Gargiulo hardware-comparison paragraph was reduced to one sentence in §7.2.

**Open threads.**
- §5.2's "full power (≳0.9) iff …" is contradicted by the 0.8 Hz full-depth
  notch (0.80). Either loosen the threshold or define "at the cadence" as
  0.8–1.2 Hz.
- §6 is still reported at FAR 0.05. The data at 1e-2 exists in every ST2
  summary.
- The next appendix paragraph's "reading summarised in §5.1" pointer was removed
  along with the bake-off, but the Scratch section still describes the
  pre-registered bake-off.

---

## 2026-08-31 — the tracker is track-before-detect, and Viterbi was the approximation

Started from the question "is there a technique other than Viterbi, and does this
problem have a name in the signal-analysis literature?" It does: **track-before-detect**
(radar/sonar), narrowband case lofargram line tracking — a lineage the bib already
carries (Streit & Barrett, Suvorova, SOAP) but the manuscript never cites. The cast
pays immediately: under the HMM the tracker implicitly assumes, Viterbi is the
MAP-path *approximation*, and the Neyman–Pearson statistic is the **forward-algorithm
marginal — sum over all paths** — at identical cost (`max` → `logsumexp`, the
Laplacian penalty promoted to a normalised transition kernel).

Prototyped it (`forward_statistic` / `forward_path_scores` in
`powerladder/typeb/detectors.py`, additive; 5 new tests, 40/40 in the module) and
smoke-tested on the B0 synth generator: forward ≥ Viterbi in every amplitude × wander
cell, with the gap opening exactly where the theory predicts — weak line under heavy
wander (AUC 0.94 vs 0.86 at amp 8 / wander 0.6; 0.82 vs 0.72 at amp 6), seed-robust.
Strong stationary lines tie at 1.0, as they must.

Write-up with the evaluation plan in
`notes/discussion/track-before-detect-forward-statistic.md`: evaluate on the
confuser/high-drift axes and as the tracking-class ceiling for the σ* bound (the two
places the NP-ceiling note shows headroom), not the plain inference null (Viterbi is
already at ρ = 1.00 there); harmonic-comb emissions (the pitch-tracking trick) as the
follow-on that attacks the controller AUC-0.0 failure. No frozen number touched; the
statistic is a bank *candidate* pending its own surrogate/FAR run. Full suite 229/5,
the 5 being the known cross-machine BLAS byte-identity set, verified pre-existing on
a clean checkout.

---

## 2026-07-24 (later) — Phase 4 closed: identifiability theory, and a review that earned its keep

Wrote the two remaining Phase-4 items, then ran `/check-PR` over the branch and spent
most of the session acting on what it found. The review was the valuable part.

**The Pareto plot** was straightforward: `scripts/plot_st2_pareto.py`, a pure reader of
the frozen `frontier_summary.json`, plotting hiding against measured cost per detector
class. The asymmetry is the paper's thesis in one figure — the fixed class gives up
hiding 0.74 at ≈0% cost, the tracking class never leaves the floor (0.16 at 159%). The
34 unpriced cells, including the `work=0.7` level where the tracker finally bends, go in
a hatched strip rather than being dropped.

**The identifiability section** was where the work was. Promoting the lower-bound spike
to propositions surfaced two defects in its own derivation: the Pinsker chain proved
*sufficiency* of a distortion, not the *necessity* the governance claim needs (fixed by
routing through an explicit verifier's attained advantage, `J ≤ TV`); and `2·AUC−1` is a
Gini index that can exceed the best threshold test's advantage, so it was never an
attained advantage at all. Switching to the Youden index J and re-deriving fixed both.
Also corrected: σ\* ∝ ε^(−1/2), not 1/ε.

**Then the review found more, and harder.** Six agents plus two adversarial verifiers.
Two blocking findings were genuine internal contradictions: `prop:cost`'s residual-marker
mechanism is refuted by our own §6 measurement (the fixed tests fall *below chance* under
variable real work, so the residual is visible only to a tracker — the class the bound
explicitly excludes), and its "period jitter has only two realisations" premise is
contradicted by our own additive-burial/dilution families. It is now `ass:cost`, an
assumption, with a remark spelling out both gaps. Separately, the `tab:closeout` claim
that "no passive detector can do better" than the semantic decoy contradicted Appendix A,
which shows `dg_order_full` reaching 0.93 of the NP ceiling on the controller case and
calls the gap "honest headroom above today's detectors, not a limit of the channel".

The best finding was a *strengthening*: the analytic κ was wrong by a factor of π.
Carrying the Lorentzian/bin algebra through properly gives the parameter-free
κ = π²f₀T = 2961, which the fit matches to 2% — where the paper had claimed only
"within a factor of about three" of πf₀T and called the prefactor irreducible.

The statistics cluster was the interesting one to adjudicate. Both verification agents
*downgraded* it from CRITICAL: `prop:sigmastar` is a correct population statement, and
the σ=0.5 point turned out to be a real effect (population J ≈ 0.10 at n=1280), not
noise. What survived was instability — κ_J varying fourfold across seeds, bootstrap CI
[14, 86], and P(σ\*(0.2) > 0.35) = 0.61, i.e. the published price bracket was more likely
wrong than right. The tell was internal: the σ=0.5 row has AUC 0.4967, so the script's
*own* mask excluded it from the Gini fit as null while admitting it to the J fit, where
it carried 78% of the leverage — purely because J is bounded below by zero. Fixed by
raising n to 600 (≈100 s), masking on the n-dependent H0 floor rather than a fixed
fraction, and reporting bootstrap intervals. σ\*(0.2) duly moved 0.34 → 0.38 and its cost
bracket moved up a level, so the paper now states the price as an order of magnitude
("tens of percent"; "of order 100% or more") rather than two-anchor brackets.

Smaller repairs worth remembering: figure PDFs were carrying a wall-clock
`/CreationDate`, so "every figure is regenerable" could not be checked by
regenerate-and-diff — `plotstyle.save` now suppresses it and the figures are
byte-reproducible. The README's ST2 block read as three safe local commands, two of which
would overwrite the slurm freeze. And `_youden` had zero test coverage while supplying
every published threshold; it now has 14 tests, including a scipy KS oracle and a
polarity guard.

Branch `phase-4`, pushed; PR not opened (no `gh` on this machine).

---

## 2026-07-23 — Phase 2: meter-requirement boundary sweep (minimum meter spec)

Built and ran the meter-requirement boundary sweep, the first Phase 2 deliverable.
The design insight was that this is the ST2 `meter` family with a dense 2-D grid in
place of four named variants — so almost nothing new was needed on the scoring side:
`meter_grid` + a crc-seeded `run_meter_cell` reuse `make_positive/negative_population`
(family="meter"), `gate.score_population`, and `roc.{auc,tpr_at_far}`. The only
library change was a backward-compatible `notch_depth` blend knob on `MeterParams`
(default 1.0 = the existing full null, so the exact-no-op / zero-RNG invariant and all
existing tests are untouched). The compute driver follows the `st1_far.py` idiom:
one cell per Slurm-array task, crash-safe per-cell JSON + flock-merged summary.

Ran it as a 48-cell array on MATS `compute` (job 5498, n_each=200, all COMPLETED in
~6 min wall). The result is a clean, interpretable boundary and — the satisfying part
— it *decomposes* the committed `integrating_1hz` two-point finding: sampling below
~2 Hz **or** an integration window above ~0.5 s each independently kills the tracking
class (Viterbi/spectral), so the original hostile variant was over-determined. fs=0.5
Hz is total death (band above Nyquist, AUC 0.5); a deep in-band notch at the cadence
centre only halves detection (a single notch can't remove a line that wanders across
0.5–1.5 Hz). The DG-order methods are the fragile class (0.77 even at the reference);
the fixed multitaper F is ~0 everywhere, its known zero-wander-power, not a boundary.
Minimum meter spec: sample ≥ ~2 Hz AND integrate ≲ 0.5 s AND no deep in-band notch.

Environment note: created the first CPU venv on the dev node (`.venv`, gitignored) to
run tests + `--smoke`; the full sweep still went to slurm. Discovered that 5
`ko_workload` byte-identity/prechange-digest tests fail on this fresh venv — verified
environmental (they fail identically on base `f43aed2`), a BLAS-build float
difference against digests baked in the monorepo. New baseline: 5 failed, 171 passed
(the old "4 sklearn failures" are gone). Flagged for regeneration before the number
freeze. Branch `feat/phase2-meter-boundary`, not yet merged. Paper wiring deferred to
Phase 4; the other two Phase-2 sub-tasks (frontier full-res, Rung 2) remain.

---

## 2026-07-22 (later) — Plan iteration: three scope decisions locked

Reviewed `spec.md`/`tasks.md` against the intended paper outline and the plan/review
docs. The outline matches plan §7 and `paper/main.tex` exactly — no structural change.
Three decisions made and recorded (spec.md edits user-approved):

1. **Learning-efficiency descoped** to an open empirical question / future work — no
   GPU campaign; the frontier's utility axis is systems-cost anchors only.
2. **Venue: arXiv-first**; submission venue chosen after results freeze.
3. **Meter channel requirement promoted** to a named contribution ("minimum meter
   specification"). Verified the ST2 `integrating_1hz` mechanism from code + committed
   results: the 1 s trailing boxcar is a sinc filter with its null at 1 Hz (the
   training-band centre, f0 ~ U(0.5, 1.5) Hz) and the 1 Hz ZOH sampler puts the whole
   band above the 0.5 Hz Nyquist, so the cadence is attenuated ×0–0.64 and aliased to
   |f0−1| Hz. Committed numbers (n=200/class): TPR@0.05 ≤ 0.32 all detectors, DG order
   methods at chance; caveat — Viterbi keeps AUC 0.68 from band-edge aliasing, so the
   claim is "defeats at the stated operating point," not absolute erasure.

Also absorbed the ST1 outcome into spec.md's methods bullet (no analytic CFAR;
effective score with per-trace surrogate calibration). New tasks: slurm re-verification
of the ST2 sweeps (never on the dev node — user decision), and a meter-requirement
boundary sweep (`sample_hz` × `integ_window_s` × notch) to turn the two-point contrast
into a real specification. Session was docs-only; branch `feat/plan-iteration`.

---

## 2026-07-22 — Repo spun out from the monorepo

Paper 2 ("What a Passive Power Meter Can Certify About AI Training") was extracted
from the `analogue-sensors-for-ai-verification` monorepo into this standalone repo, to
get out of a five-manuscript shared-library codebase that made clean development hard.

**Approach.** Non-destructive copy-only (the monorepo is left intact — removing shared
modules like `ko_workload.py` would break the other papers' `scenarios.py`). Three
parallel exploration passes established the paper-2 boundary and, crucially, that the
dependency arrows run *from* the GPU-harness code *into* paper-2's `typeb` package,
never the reverse — so paper-2's detector/scenario code has no dependency on the GPU
bench and splits cleanly.

**What moved.** The `st1/` (adaptive structural detector) and `typeb/` (detector bank +
ST2 frontier) packages, `ko_workload.py`, `observation.py`, and the shared primitives
`config/forward/noise/plotstyle`, renamed under package `powerladder/` (dropping the
old `code/` name that shadowed the stdlib `code` module). Six scripts, ~16 test files +
the `st2_prechange_refs.json` fixture, `results/st1` + `results/st2`, the 28
`st1_*`/`st2_*` figures, the plan/findings notes, and the manuscript (`paper-ladder/` →
`paper/`). The three measured GPU throughput anchors became provenance-tracked static
input under `data/measured_cost_anchors/`.

**What was left behind.** The GPU bench harness (`bench/`, `b2/`), the β / energy-
accounting code (`floor*`, `inversion`, `measured`, `scenarios`), the Kalman
`tracker.py`, and the four other manuscripts — all belong to the other papers and none
are imported by paper-2 code (verified by grep + a clean import-closure check).

**Verification.** 165 tests pass; the only failures are the 4 sklearn-dependent RF
baseline tests (environmental — this local env lacks sklearn — matching the monorepo's
documented state). ST2 sweeps and frontier regenerate from scratch to the definitive
**GO** verdict (`provisional=false`, supporting cells `work=0.35/0.5`); the ST1 bake-off
figure regenerates. The ST1 calibration figure needs `st1_far.py` run first to produce
the FAR-harness raw cells — the same reproduction order as the monorepo.

**Carried status.** Phase 0 de-risking gates are complete (ST1 = GO honest-reframe,
ST2 = GO definitive). Next is Phase 1 (Rung 1 full). The OzSTAR surrogate confirmation
(S=999, M=10⁴) remains a prerequisite before freezing any paper numbers.
