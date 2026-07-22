# Issue #54 investigation — the line-only band reads signature B, not the f_step line

**Date:** 2026-07-17 · **Branch:** `investigate/issue54-line-band` · **Status:** evidence complete, prose HELD pending sign-off on direction.

**Harness:** `scripts/investigate_issue54.py` → `results/b2/issue54_diagnostics.json`, `figures/issue54_{bandedge_sweep,Bfreq_vs_f0}.png`. Re-analysis only (no new capture). Reuses the unchanged detectors/loaders; the `[0.55,1.7]` scores match `results/b2/gate_summary.json` exactly (spectral 0.71/AUC 0.845, viterbi 1.00/AUC 1.00), and the confound-free anchor reproduces the 2026-07-17 log (`[0.90,1.7]` high-`f0` collapse: train_min 0.59 < serving_max 0.75).

## Bottom line

The measured line-only-band story as written is **not supported**. Three of its load-bearing claims fail on the evidence, and the failure is not marginal:

1. **"The tracker follows the naturally wandering measured line."** The tracked feature does **not** wander (within-trace path std = **0.029 Hz**, sub-bin) and is **not** the iteration line — it is signature **B**, a card-fixed power-management limit cycle at ~0.3–0.6 Hz whose tail sits just above the 0.55 cut.
2. **"Only the matched filter, reading the wandering line, falls to 0.71."** The 0.71 is a whole-trace-Welch artifact over the wide `[0.55,1.7]` band, **not** a wander effect: pointed straight at B (`[0.30,0.70]`) the *same* matched filter scores **1.00**. A fair fixed-frequency integrator on the tracker's own map also scores 1.00 on the line band — **the Viterbi tracker earns nothing on measured data.**
3. **"The iteration line itself is detectable in the line-only band."** Even a per-trace window centred on the *known* commanded cadence (ground-truth-gated, the most generous possible test) gives TPR **0.47**, AUC 0.68, not separable. **Signature A is undetectable on these traces**, even cheating with the answer.

What *is* true and defensible on measured data: **training separates perfectly from serving via the power-management side channel B** — but B is fixed-frequency (needs no tracker), independent of the cadence, and a property of *this* A100's controller (portability unknown).

## Evidence (from `issue54_diagnostics.json`)

### Q1 — contamination band-edge sweep (Viterbi TPR at FAR=0.05, raising the lower edge excludes B)
| population | 0.55 | 0.70 | 0.90 | 1.00 | reading |
|---|---|---|---|---|---|
| signatures (honest) | 1.00 | 0.83 | 0.83 | 0.79 | perfection needs B; some residual above 0.55 remains but is not the clean line |
| spoof (mean) | 1.00 | 0.99 | 0.94 | 0.93 | **robust** — not B-dependent |
| lora (mean) | 1.00 | 0.96 | 0.96 | 0.88 | **robust** — not B-dependent |
| shaped (mean) | 0.99 | 0.94 | 0.79 | 0.71 | partly rides B |
| workjitter (mean) | 0.92 | 0.78 | 0.58 | 0.44 | **collapses — rides B heavily** |
| dilute (mean) | 0.53 | 0.50 | 0.53 | 0.50 | ~chance on the line band regardless (its story is the distribution match, unaffected) |

Contamination is **selective**: the spoof and LoRA robustness claims survive without B; the **work-jitter** and (partly) **shaped** line-band claims substantially ride B and must be requalified; dilute's line-band arm is ~chance either way.

### Q2 — B is a distinct, card-fixed oscillation (not a re-manifestation of the cadence)
- B frequency vs commanded `f0`: slope ≈ 0 (−0.044 / +0.049 Hz/Hz), Pearson |r| ≈ 0.22–0.30, n=24 → **no relationship**.
- `cadence_drive*` controls: hunting peak sits at ~0.31–0.61 Hz **independent of the drive rate** (drive 0.1→5.0 Hz); `cadence_idle` B-power ≈ 0.02 (nothing). B-power is large only for **slow** coherent drives (0.1–0.5 Hz → 1300–15000) and tiny for fast ones (0.8–5.0 → 24–250).
- ⇒ B is a hardware limit cycle at a card-fixed frequency, **excited by slow coherent (periodic) load** — training triggers it because training is periodic, not because B encodes the cadence. **Portability risk**: on other silicon B may sit elsewhere, vanish, or land inside the iteration band.

### Q3 — the tracker does not earn its win on measured data
| band | viterbi | fixed-bin integrator | Welch matched filter |
|---|---|---|---|
| line `[0.55,1.7]` | 1.00 / 1.000 | **1.00 / 1.000** | 0.71 / 0.845 |
| B `[0.30,0.70]` | 1.00 / 1.000 | 1.00 / 1.000 | **1.00 / 1.000** |

Within-trace path wander: median std **0.029 Hz** (line band), 0.023 Hz (B band) — sub-bin, i.e. a fixed line. The fixed-bin integrator (tracker's map, path pinned to one bin) matches Viterbi exactly ⇒ letting the path move buys nothing. The MF's line-band 0.71 is Welch peak/median dilution over the wide band, cured by pointing it at B.

### Q4 — signature A cannot be rescued
Per-trace window `[f0−0.15, f0+0.15]` on the **known** cadence (n=15 usable): train_min 0.27, train_median 0.65, serving_max 0.72 → **not separable**, TPR 0.47, AUC 0.68. Ground truth cannot find a line that is not there. (Raising the band edge, Q1, likewise never recovers a clean line; a B-notch was not prototyped because Q4 already shows there is nothing above B to recover.)

### Q5 — synthetic vs measured (unchanged, for reconciliation)
Synthetic B0/B1 inject a clean `f_step` line that genuinely wanders and carries no limit cycle, so there the tracker really does track A and beat the MF **because of wander**. That result is intact. The divergence is specific to *measured* data: the real A100 has no detectable wandering iteration line, only the fixed side-channel B.

**External anchor for this divergence:** our synthetic model *is* Ko & Zhu's grid-stability training-oscillation model (arXiv:2508.16457; `KoWorkloadParams` = their Table I / eqs 1–5). Their model assumes a ~30 % iteration-cadence modulation at `f₀ ~ U(0.5,1.5)` Hz — signature A. We measure ~0.40 % of board power at A on a single A100 (≈75× shallower), because their deep modulation is an *aggregate/datacenter* effect (distributed all-reduce sync) and NVML is a ~10 Hz smoothed channel; the band is instead dominated by the PM limit cycle B, which Ko & Zhu explicitly exclude. So the measured shortfall is a scale + instrument limit, not a modelling error. Full analysis: [`ko2508.16457-measured-vs-model.md`](../ko2508.16457-measured-vs-model.md).

**How to think about it — signature A is a collective, synchronization-driven effect, not a single-GPU one.** The right mental model (sharper than "the line doesn't exist on one GPU"):
- **The cadence exists at every scale; its detectable *power signature* does not.** Each step is fwd→bwd→optimizer, so the periodicity at `f_step` is real on one GPU. What's missing is *amplitude*: ~1.5 W RMS = 0.40 % of the 370 W draw. Present, but far too faint to pull out of NVML — "faint," not "absent."
- **Multi-GPU synchronization amplifies *and* coheres it; it does not create it.** Ko & Zhu's deep "down" phase is the gradient **all-reduce** — a network-bound window where every GPU's compute idles *simultaneously*. On one GPU that window barely exists (tiny optimizer/kernel gap, no inter-node comm), so the up/down contrast is small. Across a synchronized cluster the dip becomes both **deep** (compute idle, network-bound) and **coherent** (dips sum instead of averaging out) → the ~30 %, ~1 Hz aggregate oscillation grid monitors report. Frequency = a property of the loop at any scale; depth+coherence that make it detectable = the datacenter phenomenon.
- **Two caveats so we don't over-attribute to scale.** (i) We have only measured single-GPU; the multi-GPU amplification is Ko & Zhu's model + our inference — a **falsifiable prediction** (a DDP / all-reduce run should raise A toward ~30 %), not a confirmed result. (ii) Scale is not the *only* reason A is invisible here: NVML's ~10 Hz ZOH low-passes even the ripple that is there, we deliberately train on *on-GPU random tokens with no dataloader/disk/network* (which keeps the line clean but also strips the I/O/comm stalls that would deepen the down phase), and B dominates the band regardless.

One-liner for the paper: *the iteration cadence is real at every scale, but its power signature is a collective, synchronization-driven effect — deep and detectable in a synchronized datacenter, marginal-to-invisible on a single A100 through NVML — which is why measured detection rides the PM limit cycle (B) rather than the iteration line (A).* This converts issue #54 from "our measured claim was wrong" into "the adopted aggregate-scale signature is shown, honestly, at the limit of what a single node can expose."

## Recommendation — direction R3 (with an honest-R2 option), for sign-off

**R3 — correct the measured claim; keep the method contribution on synthetic.** The measured data supports *"a single power trace separates training from serving perfectly via the power-management side channel"* but **not** *"physics-informed Viterbi tracking of a wandering iteration line beats matched filtering on real hardware."* The latter thesis stands on synthetic only. Concretely, for the follow-up (prose) session:
- Retire or demote the measured **MF-vs-Viterbi 0.71-vs-1.00** comparison to a Welch-robustness footnote — it is not a wander result.
- Reframe §5.2 / abstract / contributions / `fig:b2_roc`b / RF / discussion / summary: measured detection is a **coherent-load side-channel** result (signature B), not an iteration-line result; drop "naturally wandering measured line."
- Requalify the **work-jitter** and **shaped** line-band robustness numbers (they ride B); the **spoof** and **LoRA** robustness claims and the **dilute** distribution-match story survive as-is.
- State the **portability** limitation for B explicitly (card-fixed controller feature).
- Keep all **synthetic** B0/B1 wander/tracker claims unchanged.

**Honest-R2 alternative (not recommended as headline):** lead with B as "training exposed through the power-management side channel — a lower, louder feature than the cadence line." Attractive, but it *removes* the method's own justification (B needs no tracker), so it weakens the paper's central "you need a Viterbi tracker" thesis on measured data. Better kept as a framing of the side result than promoted to the headline.

**Not R1** (rescue A): ruled out — Q4 shows A is undetectable even ground-truth-gated.

### The wander premise is ours, not Ko's — reposition Viterbi as adversarial insurance

A deeper point that hardens R3: the "you need a tracker because the line wanders" thesis rests on a wander premise **we introduced**, not one from the source model. Ko & Zhu's line is **stationary** — `f₀ ~ U(0.5,1.5)` is drawn once and fixed per run, and their per-iteration i.i.d. `ξ` (σ=0.1) *broadens* the line but does not drift it. We add the wander in two non-Ko places: `TypeBParams.wander_hz` (the synthetic B0 detector-selection knob, Fig. 14 / `fig:b0_roc`, Appendix A — the appendix calls it "a method decision on a placeholder line-plus-null generator") and `f0_drift_hz` (the B1 OU walk, `code/config.py:193`, default 0, justified by our own async/pipeline/grad-accum/overlap argument). Full trace: [`ko2508.16457-measured-vs-model.md`](../ko2508.16457-measured-vs-model.md) §"Does Ko & Zhu's line wander?".

Consequently the tracker's justification splits by strength:
- **Adversarial wander — solid, keep it.** An adversary who deliberately jitters/drifts the cadence (`σ_φ²`) is Ko's `ξ` weaponized; a fixed-bin filter genuinely collapses there and Viterbi holds.
- **Honest wander — ours and unconfirmed.** Not in Ko, and not seen in measurement (the measured line is buried under a *fixed* B, path std 0.029 Hz). At the stationary end even `fig:b0_roc` shows the matched filter and Viterbi *tie*.

**Framing to carry into the reframe:** present the Viterbi tracker as **insurance against an adversary who smears the line, not a requirement for honest detection.** In the honest/stationary case a matched (or fixed-bin) filter is near-optimal and Viterbi merely ties it — exactly what Q3 shows on measured data (fixed-bin = Viterbi = 1.00). Viterbi still *works* honestly and carries no penalty, but its distinctive value appears only where the line moves — and on our data that is only when an adversary makes it move. This is a cleaner, more honest contribution statement than "physics-informed tracking beats matched filtering," and it survives issue #54 intact.

## Reproduce
```
python scripts/investigate_issue54.py
# -> results/b2/issue54_diagnostics.json ; figures/issue54_*.png
```
Runs on OzSTAR login (pure NumPy/SciPy, 24+24 signatures traces + controls + 5 adversary phases; ~seconds). No `.tex` edits made — prose held pending sign-off on R3 vs honest-R2.
