# ST2 meter-requirement boundary sweep — findings

**Date:** 2026-07-23 · **Status:** complete (definitive; slurm array 5498, MATS
`compute` partition). **Source of truth for scope:** `spec.md` ("minimum meter
specification"). **Regenerate:** `scripts/st2_meter_boundary.py` →
`results/st2/meter_boundary_summary.json` → `scripts/plot_st2_meter_boundary.py`.

## What this is

Phase-2 task (tasks.md §Phase 2). Phase 0 (ST2) had a *two-point* meter result:
the `integrating_1hz` channel (1 s trailing boxcar + 1 Hz ZOH sampler) defeats
every detector on **unmodified honest training**. This sweep generalises it to a
dense grid of observation channels (`sample_hz × integ_window_s`, plus a notch
depth/Q sub-sweep at 20 Hz), honest training only, and locates where each
detector class dies — turning the two-point contrast into a specification.

48 cells, `n_each = 200` per class, 300 s traces, 5-detector "full" set. Scored
exactly as the frontier (`gate.score_population` + `roc`); the `(20 Hz, 0 s,
no-notch)` corner is the honest reference. Ran as a Slurm array (one cell per
task, crash-safe crc-seeded); all 48 cells COMPLETED, no errors.

## Headline: the minimum meter specification

To keep the Rung-1/2 tracking class (Viterbi, spectral matched filter) at full
power (TPR@0.05 ≳ 0.9 vs the inference null), the meter needs **all three** of:

1. **sample rate ≥ ~2 Hz.** fs ∈ {20, 10, 5, 2} hold TPR ≈ 1.0; **fs = 1 Hz
   collapses** (Viterbi 0.36, spectral 0.42) as the cadence band (search
   0.3–1.7 Hz, f0 ~ U(0.5,1.5)) crosses the 0.5 Hz Nyquist; **fs = 0.5 Hz is
   totally dead** (all detectors 0.00 — the band lies entirely above the 0.25 Hz
   Nyquist, AUC = 0.5).
2. **integration window ≲ 0.5 s.** At 20 Hz, integ ∈ {0, 0.1, 0.25} hold ≈ 1.0
   and 0.5 s still holds (Viterbi 0.99, spectral 0.96), but **1 s collapses**
   (Viterbi 0.25, spectral 0.27): the 1 s trailing boxcar is a sinc with its
   null at 1 Hz — the cadence-band centre.
3. **no deep in-band notch at the cadence centre.** An off-band notch (0.6 Hz)
   or a shallow one is tolerable (Viterbi ≈ 1.0); a full-depth notch at
   1.0–1.2 Hz roughly halves tracking power (Viterbi → 0.48–0.55) but does **not**
   erase it — a single notch removes only part of a line that wanders across
   0.5–1.5 Hz.

## The two-point anchor, decomposed

`integrating_1hz` combined a 1 s boxcar and 1 Hz sampling. The grid shows either
one **alone** already kills the tracking class:

| cell | Viterbi | spectral | dg_order_full |
|---|---|---|---|
| reference `fs20 iw0` | 1.00 | 1.00 | 0.77 |
| integration only `fs20 iw1` | 0.25 | 0.27 | 0.23 |
| sampling only `fs1 iw0` | 0.36 | 0.42 | 0.12 |
| both `fs1 iw1` (= integrating_1hz) | 0.29 | 0.18 | 0.05 |

This reproduces the committed finding *inside* the grid (Viterbi retains ~0.29
at the 1 Hz sampler via band-edge aliasing while the DG order methods go to
chance) and shows it was over-determined — two independent channel defects, not
one.

## Detector-class ordering (a robustness ranking, not new to this sweep)

- **Viterbi ≥ spectral > DG-order** in channel robustness. Viterbi is the last
  to die (retains partial power at fs = 1 via aliasing). The DG order /
  cyclostationary methods are the sensitive-but-fragile class: dg_order_full is
  already only 0.77 at the reference and dies first as the channel degrades
  (0.26 at fs = 2, iw = 0). Consistent with the ST1 characterisation.
- **mtf (fixed multitaper F) ≈ 0 everywhere, including the reference.** This is
  the already-established zero-wander-power property of the fixed harmonic test
  (paper §4 `subsec:benign`), not a channel effect — it is not a functioning
  detector against a wandering cadence and should be read as the fixed-test
  floor, not a boundary.

## Caveats

- A benign `RuntimeWarning` (mean of empty slice) fires on the fs = 0.5 Hz cells:
  the detector band is entirely above Nyquist so in-band bins are empty → NaN
  scores → AUC 0.5 (chance). Correct behaviour (total death), not a bug.
- Honest training only; this is a **channel-requirement** boundary, not an
  attack budget. It states what the verifier must be able to observe, orthogonal
  to the de-periodicisation frontier (which holds the channel fixed and moves the
  workload).
- **Canonical environment / cross-BLAS reproducibility.** The committed summary
  was generated on the MATS `compute` partition with single-threaded OpenBLAS
  (slurm array 5498; recorded in the summary's `environment` field). Regenerating
  at the identical `seed = 0`, `n_each = 200` on a *different* OpenBLAS build can
  shift some cells by up to ~0.3 AUC — but these swings hit only the dead /
  marginal cells (mtf everywhere ≈ 0; dg_order_full at iw = 2). Every one of the
  16 live tracking cells in the `fs ≥ 2 ∧ iw ≤ 0.5` box stays ≥ 0.9, so the
  minimum-meter *specification* is environment-invariant even though individual
  marginal/dead-cell AUCs are environment-conditional.
