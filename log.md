# Research log — certifying-training-from-power

Dated narrative of work sessions, newest at top. Append-only historical record
(forward-looking state lives in `handoff.md`).

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
