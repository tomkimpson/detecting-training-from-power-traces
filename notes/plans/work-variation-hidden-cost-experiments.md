# The hidden cost of work variation: model estimate and experiments

**Date:** 2026-10-01
**Status:** plan (forward-looking; nothing here has been run). Goal: test whether the
work-variation attack at σ_G = 0.7, which is free in throughput and halves the
tracker's detection rate (§6.3 of `paper/main.tex`), has a real cost in learning
efficiency. If it does, the method is rescued: no attack would weaken the tracker for
free. The mechanism and the schedule statistics are explained in
[`../discussion/work-variation-attack-explained.md`](../discussion/work-variation-attack-explained.md).

## 1. The claim to test

At saturated compute, iteration length is set by tokens per optimiser step (the
explainer, §2). So:

> **Period spread *is* batch-size spread, and batch-size spread costs learning
> efficiency.**

The attack is free in tokens per second but probably not in **loss reduction per
token**. If true, this also replaces the assumption in §6.4 of the paper ("at fixed
useful work, jitter costs throughput or learning efficiency") with an argument.

## 2. A first estimate: the gradient-noise-scale model

McCandlish et al. (2018), *An Empirical Model of Large-Batch Training*, model the
best achievable loss improvement in one step with batch size `B` as

```
ΔL(B) ∝ h(B) = B / (B + B_noise)
```

where `B_noise` is the gradient noise scale. Two consequences:

- **Below `B_noise`**, `h` is roughly linear: doubling the batch roughly doubles the
  progress per step, so no tokens are wasted.
- **Above `B_noise`**, `h` saturates: extra tokens in a step buy almost nothing.

`h` is concave, so by Jensen's inequality any spread in batch size at a fixed token
budget gives less progress than a constant batch. The work-variation schedule also
inflates the *mean* batch (from 8 to 25 at σ_G = 0.7), which costs on top of that.

Progress per token is `E[h(G)] / E[G]` for the attacked schedule and `h(8)/8` for the
honest one. Their ratio, minus one, is the extra tokens needed to reach the same loss.
Taking `G` from the sampled schedule (`B` in micro-batches):

| σ_G | B_noise = 2 | 8 | 32 | 128 |
|---|---|---|---|---|
| 0.35 | 26% | 23% | 15% | 7% |
| 0.5 | 92% | 82% | 56% | 28% |
| **0.7** | **209%** | **184%** | **120%** | **53%** |

```python
import numpy as np
rng = np.random.default_rng(0)
for s in [0.35, 0.5, 0.7]:
    xi = rng.normal(0, s, 10**6)
    g = 8 / np.maximum(1 + xi, 0.05)
    lo = np.floor(g)
    G = np.maximum(lo + (rng.random(g.size) < g - lo), 1)
    for Bn in [2, 8, 32, 128]:
        attacked = np.mean(G / (G + Bn)) / G.mean()
        honest = (8 / (8 + Bn)) / 8
        print(s, Bn, honest / attacked - 1)
```

**Caveats.** This is a model, not a measurement:

- It assumes the learning rate is tuned at every step. Ours is fixed, and Adam
  normalises step size, so a real run may differ in either direction.
- `B_noise` for any given model is unknown here; the table brackets it.
- It holds the honest batch at 8 micro-batches. A real operator picks its honest batch
  relative to its own critical batch size.

Even so, in its most favourable column the free attack costs about 50% more compute
to reach the same loss. That is the same order as the costly attacks it supposedly
beats (idle-pad jitter at σ = 0.35 costs 159% and hides only 0.24).

## 3. Experiments

### E0. Can the attacker avoid the cost and still hide? (CPU, synthetic, hours)

About two thirds of the estimated cost comes from the mean batch inflating from 8 to
25. A referee will ask why the attacker doesn't keep the mean at 8.

- **Do:** sweep `G` distributions with the mean fixed at 8: lognormal, two-point, and a
  heavy tail with a cap. For each, record tracker hiding, Var(G), and the noise-scale
  penalty of §2 over a range of `B_noise`.
- **Output:** a "hiding vs predicted learning cost" frontier for work variation.
- **Decision:**
  - If every distribution that hides needs a large spread in `G`, the rescue holds
    and E1 becomes a clean test.
  - If some distribution hides cheaply, we need to know now, before claiming anything.
- **Cost:** runs on the existing ST2 harness (`powerladder/typeb/st2_attacks.py`). The
  generator will need a way to take a custom `G` distribution; today it only has the
  `8/(1+ξ)` form.

### E1. Measure tokens to a target loss (GPU, the decisive test)

- **Setup:** a small LM (nanoGPT, about 50–125M parameters) on real data
  (FineWeb-edu or OpenWebText), not random tokens.
- **Comparison:** honest `G = 8` against σ_G ∈ {0.35, 0.5, 0.7}, at least 3 seeds each.
- **Metric:** tokens, and therefore GPU-seconds, to reach a fixed validation loss.
- **Attacker arms:**
  - (a) naive, as on the hardware campaign: fixed learning rate, loss divided by `G_i`;
  - (b) best response: learning rate scaled with `G_i` (square-root rule for Adam), and
    the learning-rate schedule indexed by tokens rather than steps;
  - (c) the cheapest hiding distribution found in E0.
- **Output:** **compute-to-target overhead**. For the idle-pad attacks, learning per
  token is unchanged, so this equals their throughput overhead. Every point on the
  frontier can then be re-plotted on one axis.
- **Where:** MATS is preferred (CLAUDE.md); OzSTAR `milan-gpu` (A100, account `oz022`)
  also works.

### E2. Measure `B_noise` for the E1 model (cheap, same jobs)

- **Do:** estimate the gradient noise scale with the simple two-batch-size estimator
  of McCandlish et al. (2018), using gradient norms at two batch sizes.
- **Check:** does the measured penalty in E1 match the §2 formula at the measured
  `B_noise`?
- **Why it matters:** if the formula holds, it plus published critical-batch-size
  scaling (Kaplan et al. 2020) can argue how the cost behaves at frontier scale. A
  100M-model result alone won't convince a governance reader.

### E3. A cost lower bound (theory, no compute)

Chain three steps:

1. Hiding level `H` needs period spread at least `c(H)`. This curve is empirical, from E0.
2. Period spread equals batch-size spread at saturated compute. This is an identity.
3. Batch-size spread costs at least the Jensen gap of `h(B)`.

Together these give a minimum learning-efficiency cost for hiding level `H`. It would
sit in §6.4 of the paper alongside the existing total-variation bound and remove that
section's stated assumption.

### E4. Escape routes to check before claiming the rescue (CPU, synthetic)

The identity in §1 has loopholes. We should test them before a referee raises them.

- **Interleaving independent jobs.** Run several jobs with different *fixed* batch
  sizes and step them in random order. Each run keeps a constant batch, so pays no
  learning-efficiency cost, but the combined cadence is aperiodic. The price is
  latency, not efficiency. This is the most serious loophole. Test whether the
  tracker sees through it, and how many jobs of what spread it takes.
- **Filling gaps with useful non-training work** (evals, checkpointing, serving). This
  is really dilution, and its cost is training throughput. The paper should say so.
- **Slowly ramping batch size.** This is a legitimate, widely used practice (GPT-3
  ramped its batch). It is cheap but slow, so it looks like drift, which the tracker
  survives. One sentence in the paper covers it.

## 4. Recommended order

1. **E0 first.** It is cheap, needs no GPU, and decides the framing.
2. **E1 with E2** on GPU, using the arms E0 picks.
3. **E4** in parallel with E1, since it is also CPU-only.
4. **E3** once E0 and E1 have results to anchor it.
