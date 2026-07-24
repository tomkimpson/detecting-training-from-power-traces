# Research log — certifying-training-from-power

Dated narrative of work sessions, newest at top. Append-only historical record
(forward-looking state lives in `handoff.md`).

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
