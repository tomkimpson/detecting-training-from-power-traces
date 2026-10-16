# How the work-variation attack works, and why it is "free"

**Date:** 2026-10-01
**Status:** pedagogical walkthrough (rationale record, not a tracker). It explains the
one frontier cell where the Viterbi tracker bends: work variation at σ_G = 0.7, which
hides 0.55 at a measured throughput cost of −0.65% (§6.3 of `paper/main.tex`). The
companion plan, on whether the attack has a hidden cost and how to measure it, is
[`../plans/work-variation-hidden-cost-experiments.md`](../plans/work-variation-hidden-cost-experiments.md).

## 1. What the tracker locks onto

Synchronous training repeats one two-phase cycle:

- **Up phase.** The GPU runs forward and backward passes, and power is high.
- **Down phase.** The GPU syncs gradients and takes the optimiser step, and power dips.

If every iteration takes the same time `T`, the meter sees a near-periodic square
wave. That wave puts a narrow spectral line at the cadence `f₀ = 1/T`, plus
harmonics. The Viterbi tracker follows that line as it wanders slowly. Inference
spreads its power across the band and has no such line. That difference is the whole
detection signal.

## 2. The key identity: the period is the batch size

An honest iteration accumulates gradients over `G = 8` micro-batches, then steps the
optimiser (`work_base_accum = 8` in `powerladder/config.py`, which matches the
hardware campaign's `TrainLoop`). When the GPU is saturated, every micro-batch takes
about the same time `t_micro`, so

```
T_up   = G · t_micro
T_iter = G · t_micro + T_down
```

> **At saturated compute, an iteration's length is set by how many tokens go into
> that optimiser step.**

To make the iteration lengths irregular without idling, the operator has to make the
number of tokens per optimiser step irregular. Everything below follows from this.

## 3. What the attacker does

Each iteration, the attacker draws a fresh micro-step count:

```
G_i = stochastic_round( 8 / max(1 + ξ_i, 0.05) ),   ξ_i ~ N(0, σ_G),   G_i ≥ 1
```

Where this lives in code:

- Hardware: `work_jitter_schedule` in the monorepo's `code/b2/workloads.py`.
- Synthetic: the `work_sigma` branch of `_periodic_F_meta` in `powerladder/ko_workload.py`.

The down phase is left unchanged, because the gradient sync and optimiser step don't
depend on how many micro-batches came before them. So the duty ratio moves with `G_i`.

Iteration lengths are now random, so the line's power spreads across a band instead of
sitting in one bin. At small σ_G the spread is narrow and the tracker still collects
it. At σ_G = 0.7 the spread is too wide for one track, and the tracker's detection
rate falls to 0.46 at false-alarm rate 10⁻².

## 4. What the schedule actually looks like

The `8/(1+ξ)` form is not symmetric. As `ξ → −1`, `G` blows up until it hits the cap
at `8/0.05 = 160`. Sampling the schedule (10⁶ draws; snippet below) gives:

| σ_G | mean G | median G | steps with G ≤ 4 | steps capped at G = 160 | share of tokens in G = 160 steps | optimiser steps per token, vs honest |
|---|---|---|---|---|---|---|
| 0.35 | 10.1 | 8 | 2% | 0.3% | 5% | 0.79 |
| 0.5 | 15.5 | 8 | 6% | 2.9% | 30% | 0.52 |
| **0.7** | **25.0** | 8 | 13% | 8.7% | **56%** | **0.32** |

The typical step looks honest, since the median is still 8. The tail does the hiding:
at σ_G = 0.7 about 1 step in 11 is 20 times the honest batch, and those steps hold
more than half of all tokens trained. The mean batch is about 3 times the honest one,
so the attacker takes only about a third as many optimiser steps per token.

```python
import numpy as np
rng = np.random.default_rng(0)
for s in [0.35, 0.5, 0.7]:
    xi = rng.normal(0, s, 10**6)
    g = 8 / np.maximum(1 + xi, 0.05)
    lo = np.floor(g)
    G = np.maximum(lo + (rng.random(g.size) < g - lo), 1)
    w = G / G.sum()
    print(s, G.mean(), np.median(G), (G <= 4).mean(), (G == 160).mean(),
          w[G == 160].sum(), 8 / G.mean())
```

## 5. Why our metric calls it free

The cost metric is `throughput_overhead` in the monorepo's `code/b2/analysis.py`. For
work variation it is **tokens per second against the honest loop's calibrated
tokens per second**. Compare the two ways to de-periodicise:

| | Idle-pad jitter | Work variation |
|---|---|---|
| How the period changes | sleep is inserted after the optimiser step | more or fewer real micro-batches per step |
| Is the GPU ever idle? | yes | no |
| Effect on tokens/s | falls | unchanged |
| Measured cost | 15–1027% | −0.10% to −0.65% |

Work variation never idles the GPU: every second is spent on real micro-batches.
Fewer optimiser steps means slightly less per-step overhead (optimiser update, sync,
host work), which is why the measured cost is slightly *negative*. Within noise it is
zero.

The underlying reason:

> **Tokens per second cannot see how tokens are grouped into optimiser steps, and
> that grouping is exactly what the attack changes.**

## 6. Two details of the hardware measurement

1. **The campaign trains on random tokens.** `TrainLoop._micro_step` draws
   `torch.randint(0, vocab, ...)` as input. There is nothing to learn, so the
   campaign could not have seen an effect on learning even if one existed. It
   measures systems throughput only, which is what it was built for.
2. **Big steps don't take bigger steps.** The loop divides each micro-batch loss by
   the current `G_i` (`(loss / self.grad_accum).backward()`), so each optimiser step
   sees the mean gradient. It then calls AdamW at a fixed learning rate (3e-4).
   Adam normalises the update size, so a 160-micro-batch step moves the weights about
   as far as an 8-micro-batch step. Its gradient is less noisy, but it costs 20 times
   the tokens.

## 7. Where this leaves the paper

The paper's statement is correct: in throughput, the far end of the frontier is free.
But throughput is the wrong currency for this attack, because the attack changes
*learning per token*, not *tokens per second*. Whether that is a real cost, and how
large, is the subject of the companion plan,
[`../plans/work-variation-hidden-cost-experiments.md`](../plans/work-variation-hidden-cost-experiments.md).
