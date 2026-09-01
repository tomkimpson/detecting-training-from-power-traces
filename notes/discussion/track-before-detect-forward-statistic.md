# Track-before-detect, and the forward statistic the Viterbi tracker is missing

**Date:** 2026-08-31 · **Status:** prototype landed (`powerladder/typeb/detectors.py`
`forward_statistic` / `forward_path_scores` + 5 tests); **not in any frozen result**.
Origin: the question "is there a technique other than Viterbi we should try — can this
be cast as a known problem from the signal-analysis literature?"

## 1. The cast

What `viterbi_statistic` does — integrate energy along candidate frequency
trajectories through a spectrogram *before* thresholding, because no single frame has
enough SNR to detect on — is the **track-before-detect (TBD)** problem from the
radar/sonar literature (dynamic-programming TBD: Barniv 1985), whose narrowband
special case is **passive-sonar lofargram frequency-line tracking**. The repo half
knows this: `paper/references.bib` carries a "Line-tracking method lineage (sonar →
CW gravitational waves → here)" block — `streit1990frequency`, `suvorova2016hmm`,
`bayley2019soap`, `djurovic2011viterbi` — but **none of the four keys is cited in
`main.tex`**. Wiring that lineage into related work is a cheap, independent paper
improvement.

Casting the problem as TBD makes Viterbi's status precise, and that is where the
missing technique falls out.

## 2. The missing statistic: sum over paths, not the best path

The tracker already implicitly assumes a generative model: hidden frequency bin
evolving as a Markov chain (the Laplacian `jump_penalty` wander prior), per-frame
spectrogram emissions. Under that model:

- **Viterbi** computes the **MAP single path** — an approximation.
- The **Neyman–Pearson statistic** is the **marginal likelihood ratio**
  P(spectrogram | wandering line) / P(spectrogram | noise), whose numerator **sums
  over all paths**. That is the HMM **forward algorithm** (Streit & Barrett 1990),
  at the same cost as Viterbi: the identical recursion with `max` replaced by
  `logsumexp`, and the penalty promoted to a normalised transition kernel.

Max ≈ sum when one path dominates (strong line, slow wander). They diverge at **low
per-frame SNR under heavy wander**, where many near-optimal paths carry comparable
mass that the MAP path undersells — which is exactly the adversarial jitter regime
the paper cares about.

## 3. Prototype

`powerladder/typeb/detectors.py`: `forward_statistic` and its sequential view
`forward_path_scores`, mirroring the Viterbi pair (same `_spectrogram_band` emission
map, same defaults, same detector contract). Design decisions:

- **Normalised Laplacian transition kernel** (`_wander_log_kernel`): summing over
  paths must marginalise against a distribution or the statistic inflates with path
  count. Edge bins concentrate slightly more mass (fewer neighbours); negligible at
  ~20-bin bands.
- **Uniform prior over starting bins**; final statistic is the log-marginal averaged
  over frames, comparable across trace lengths like the Viterbi statistic.
- **Not numerically comparable to `viterbi_statistic`** (whose penalty is
  unnormalised); each statistic meets a threshold only through its own ROC /
  surrogate calibration.

Tests: `tests/test_typeb.py` (§ "forward (sum-over-paths) statistic") — contract
pins (sequential-vs-scalar agreement, empty band, strong-line agreement with
Viterbi, ranking) plus the regime claim (`test_forward_gains_over_viterbi_at_low_snr_heavy_wander`).

## 4. Smoke result (B0-local synth generator; scratchpad, not frozen)

Amplitude × wander sweep, `n_each=100`, seed 20260831, AUC / TPR@FAR=0.05
(default `amp_train` is 30; lower = lower per-frame SNR):

| amp | wander | spectral | viterbi | forward |
|----:|-------:|---------:|--------:|--------:|
| 30 | 0.0 | 1.000 / 1.00 | 1.000 / 1.00 | 1.000 / 1.00 |
| 30 | 0.6 | 0.147 / 0.08 | 0.999 / 0.99 | 1.000 / 1.00 |
| 12 | 0.6 | 0.000 / 0.00 | 0.968 / 0.89 | 0.989 / 0.97 |
|  8 | 0.6 | 0.000 / 0.00 | 0.864 / 0.65 | 0.944 / 0.79 |
|  6 | 0.6 | 0.000 / 0.00 | 0.718 / 0.38 | 0.824 / 0.48 |

Forward ≥ Viterbi in **every** cell; the gap opens exactly where the theory says
(weak line + heavy wander) and is seed-robust: over seeds 1–3 at (amp 8, wander 0.6)
the AUC gap is 0.06–0.11 and TPR@0.05 gap 0.16–0.17; at (amp 6, wander 0.6) AUC gap
0.08–0.13, TPR gap 0.10–0.17. At strong line / stationary the two agree at 1.0, as
they must.

## 5. Where the real evaluation lives (and where not to bother)

The NP-ceiling record (`notes/results/st1-np-ceiling-findings.md`) localises the
headroom, so a real evaluation of the forward statistic should target:

1. **Structural confusers at high drift** — the best deployable detector
   (`dg_order_full`) falls to 0.30 → 0.10 of the ceiling by drift 0.8–1.5 Hz.
2. **The controller hard case** — Viterbi scores AUC 0.0 (actively fooled by the
   limit cycle). The forward statistic alone will NOT fix this (same emission map,
   same lock-on failure); see §6.
3. **The tracking-class ceiling for the covertness bound.** The manuscript concedes
   "the tracker's own boundary is unmeasured" and `rem:direction` admits the σ*
   bound is loose against the tracking class. The forward statistic is the natural
   computable proxy for the NP-optimal detector *within* that class — usable both
   as a bank member and as a better-motivated ceiling than the (stationary) Whittle
   LRT, which by construction cannot see wandering-line structure.

**Not worth effort:** the plain inference null — Viterbi is already at
ρ_auc = 1.00 at every drift there; no headroom exists.

These are slurm-scale runs (bake-off harness + surrogate calibration); the smoke
table above is only the existence proof that the statistic behaves as the theory
predicts.

## 6. Follow-on from the same cast: harmonic-comb emissions

Pitch tracking (speech f₀ estimation: harmonic-sum spectra, YIN/PYIN) solved the
same TBD problem with the emission summed over the harmonic comb,
`Σ_k logP(k·f̂₀)`. The NP-ceiling note's diagnosis of the controller hard case —
"the missing ingredient is order-tracking structure, not knowledge of the line
frequency" — is precisely a comb argument: a near-sinusoidal control limit cycle has
weak harmonics, the duty-cycle iteration waveform has a strong comb. Comb emissions
compose with the forward recursion unchanged (richer emission, same kernel) and
attack the one axis where the tracker is actively fooled. Not prototyped here;
natural next experiment.

## 7. ST1 warp estimation: forward–backward, not forward

In ST1 the Viterbi tracker also supplies the `f̂₀(t)` path for phase resampling
(`powerladder/st1/`). The forward statistic yields no single path; the
forward–backward recursion, however, gives per-frame posterior marginals, and the
posterior-mean frequency path is a smoother warp estimate than the MAP path (no
bin-quantised jumps). Separate question from detection; noted for completeness.

## 8. Alternatives considered and not prioritised

- **Synchrosqueezing / reassignment ridge extraction, Vold–Kalman filtering** —
  already flagged as tracker alternatives (plan §10 Q3) and settled by the bake-off;
  they change warp quality, not detection power.
- **Particle-filter TBD** — continuous-state version of the forward algorithm; more
  machinery, same statistic at this problem size.
- **Hough/Radon line integration** (FrequencyHough in the GW literature) — coarser
  than the HMM, largely redundant with it.
- Cyclostationarity, order tracking, sequential (e-process/SPRT) methods — already
  in the bank.

## 9. Caveats / scope discipline

- **No frozen number changes.** The prototype touches `powerladder/typeb/detectors.py`
  (additive) and tests only; ST1/ST2 freezes, figures, and the manuscript are
  untouched. `spec.md` unmodified.
- A promotion to the detector bank is a **new bank member**: its own surrogate
  calibration, FAR harness run, and bake-off column — not a swap for Viterbi.
- The `jump_penalty` default (1.0) is inherited, not tuned; under the normalised
  kernel its effective meaning differs from Viterbi's. A small sensitivity sweep
  belongs to any real evaluation.
