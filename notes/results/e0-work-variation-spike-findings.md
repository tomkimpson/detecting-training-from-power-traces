# E0 spike: which micro-step distributions hide the line, and at what learning cost

**Date:** 2026-10-01
**Status:** findings from a throwaway spike (exploratory; not frozen; not in the
paper). This is experiment E0 of
[`../plans/work-variation-hidden-cost-experiments.md`](../plans/work-variation-hidden-cost-experiments.md).
The attack mechanism is explained in
[`../discussion/work-variation-attack-explained.md`](../discussion/work-variation-attack-explained.md).
Repo state when run: `2b1e9a1` on `feat/frontier-aggregate-explore`.

## Question

The work-variation attack in §6 (σ_G = 0.7) hides 0.55 from the Viterbi tracker at
zero throughput cost. Two questions:

1. Which part of its micro-step distribution does the hiding?
2. Can an attacker hide while keeping the **mean** micro-step count at the honest 8?
   That attacker takes as many optimiser steps per token as honest training, so it
   avoids the biggest learning cost of the shipped attack.

## Headline

- **The rescue only partly holds.** No attack hides for free in learning
  efficiency, but the attack in the paper is a poor one. A better-designed attacker
  hides more (0.73 against 0.53) at a much smaller predicted learning cost (8–25%
  against 53–184%).
- **The paper understates the threat.** A mean-8 lognormal attacker takes the
  tracker's detection rate down to about 0.26, against 0.46 in §6.
- **Hiding is never free.** With the mean held at 8, nothing below a lognormal
  spread of s ≈ 0.6 hides at all, and any spread costs learning efficiency under the
  noise-scale model. The cheapest attacker hiding ≥ 0.6 pays a predicted 4–16%.
- **All learning costs here are model estimates, not measurements.** E1 must
  measure them.

## Method

- **Patch.** The work-variation branch of
  `powerladder.ko_workload._periodic_F_meta` sets the per-iteration micro-step
  target to `g_real = 8 / max(1 + ξ, 0.05)`. The probe replaces that one line with a
  call to a custom sampler and leaves everything else alone: stochastic rounding to
  an integer `G ≥ 1`, the unchanged down phase, the aggregate background, and the
  observation channel. **Check:** with the shipped sampler, the patched generator's
  output is byte-identical to the unpatched one.
- **Scoring.** The Viterbi tracker only (`DETECTORS["viterbi"]`), on the §2
  aggregate (`ko_make_aggregate_trace`), against the aggregate null, at false-alarm
  rate 1e-2. 200 traces per class, with the meter as in `DEFAULT.st2`. Hiding is
  1 − detection rate; the noise from 200 traces is about ±0.035.
- **Learning cost.** For each arm, 2×10⁵ integer `G` draws made exactly as the
  generator makes them. The cost is the gradient-noise-scale penalty of the plan
  note, §2: `(h(8)/8) / (E[h(G)]/E[G]) − 1` with `h(B) = B/(B + B_noise)`, at
  B_noise = 8, 32 and 128 micro-batches. It assumes a learning rate tuned every step.

## Results

### 1. The giant batches drive the shipped attack's hiding

| Arm | Hiding | Mean G | Median G | CV of G | P(G ≥ 64) | Optimiser steps per token, vs honest | Cost at B_noise = 8 / 32 / 128 |
|---|---|---|---|---|---|---|---|
| shipped σ_G = 0.7 | 0.520, 0.545 | 25.0 | 8 | 1.77 | 0.106 | 0.32 | 184% / 120% / 53% |
| shipped σ_G = 0.5 | 0.040 | 15.5 | 8 | 1.79 | 0.040 | 0.52 | 82% / 56% / 28% |
| σ_G = 0.7, G capped at 64 | 0.185 | 16.0 | 8 | 1.16 | 0.106 | 0.50 | 84% / 49% / 19% |
| σ_G = 0.7, G capped at 32 | 0.160 | 12.2 | 8 | 0.78 | 0 | 0.66 | 43% / 23% / 8% |
| σ_G = 0.7, G capped at 16 | 0.030 | 9.4 | 8 | 0.49 | 0 | 0.85 | 16% / 8% / 3% |
| σ_G = 0.7, small-batch side only (G ≤ 8) | 0.000 | 5.5 | 5 | 0.25 | 0 | 1.46 | −14% / −6% / −2% |
| σ_G = 0.7, large-batch side only (G ≥ 8) | 0.245 | 44.5 | 15 | 1.26 | 0.211 | 0.18 | 298% / 163% / 64% |

The probe reproduces the frozen value: the frozen run in
`results/st2/aggregate/work_summary.json` hides 0.545. Hiding collapses once the
rare G = 160 steps go. Those are unbroken compute blocks about 20 times the honest
length. For this attack, then, the hiding comes from the same giant batches that
carry most of its predicted learning cost.

### 2. Mean held at 8: a lognormal attacker hides more, more cheaply

The micro-step target is `8 · exp(s·z − s²/2)` with z ~ N(0, 1), capped at 160, so
the mean stays at 8 (up to the cap).

| Lognormal spread s | Hiding | Median G | CV of G | Cost at B_noise = 8 / 32 / 128 |
|---|---|---|---|---|
| 0.3 | 0.000 | 8 | 0.31 | 2% / 2% / 0% |
| 0.5 | 0.010 | 7 | 0.54 | 6% / 4% / 2% |
| 0.6 | 0.125 | 7 | 0.66 | 9% / 6% / 2% |
| 0.7 | 0.375 | 6 | 0.80 | 12% / 8% / 3% |
| 0.8 | 0.610, 0.620 | 6 | 0.94 | 16% / 11% / 4% |
| 0.9 | 0.690 | 5 | 1.10 | 20% / 14% / 6% |
| **1.0** | **0.735, 0.725, 0.740** | 5 | 1.27 | **25% / 18% / 8%** |
| 1.1 | 0.660 | 4 | 1.44 | 30% / 22% / 9% |
| 1.2 | 0.610 | 4 | 1.61 | 35% / 26% / 11% |
| 1.5 | 0.280 | 3 | 2.08 | 50% / 38% / 17% |

Multiple hiding values are independent positive populations (see Reproduction).
They agree within the noise.

Other mean-8 distributions hide less:

| Arm (mean 8) | Hiding | Median G | CV of G | Cost at B_noise = 8 / 32 / 128 |
|---|---|---|---|---|
| shipped σ_G = 0.7 shape, rescaled to mean 8 | 0.425 | 3 | 1.77 | 55% / 41% / 16% |
| two-point, G ∈ {2, 62} | 0.000 | 2 | 2.25 | 86% / 68% / 26% |
| two-point, G ∈ {1, 64} | 0.025 | 1 | 2.48 | 153% / 98% / 34% |

## Interpretation

- **Hiding peaks at a moderate, smooth spread.** At mean 8 it rises steeply from
  s = 0.5 to s = 1.0, then falls. Distributions that put a lot of mass at small G
  (large-s lognormals, two-point) hide less. My guess is that runs of short, equal
  iterations create a new regular rhythm the tracker can find. **This is
  unverified.**
- **Spread in G, not the CV of the cadence, is what hiding depends on, and the
  relation isn't monotone.** This matches the frozen sweep, where idle-pad jitter
  0.7 has a larger cadence CV than work 0.7 yet is detected every time.
- **The two attacks pay for different things.** The shipped attack's predicted
  learning cost is mostly the inflated mean: 3× fewer optimiser steps per token. The
  rescaled-to-mean-8 arm costs 16–55% against 53–184%. The lognormal attacker pays
  only for spread.
- **Throughput.** The shipped attack's throughput cost is measured (−0.65%). The
  lognormal attacker's is not, but it takes the same number of optimiser steps as
  honest training and never idles, so ≈ 0% is expected.

## Caveats

- **The learning costs are a model.** They assume a learning rate tuned at every
  step, and B_noise is unknown. A real AdamW run at a fixed learning rate could pay
  more, for example on noisy small-batch steps, or less.
- **Single detector.** Only the Viterbi tracker was run. The fixed tests were not.
- **Small search.** The attacker's distribution was searched over one smooth family
  (lognormal) and a few hand-picked shapes. A better mean-8 distribution may exist,
  so 0.74 is a lower bound on what an attacker can hide.
- **One scenario.** The §2 aggregate at false-alarm rate 1e-2, 200 traces per class,
  two seeds.

## Implications for next steps

- **E1** should test the mean-8 lognormal at s = 0.8 and s = 1.0, alongside honest
  training and the shipped σ_G = 0.7. Those arms bracket the attacker's best hiding
  against its predicted cost.
- **§6** quotes the shipped attack as the far end of the frontier. If these results
  hold, that understates the threat. Any revision should wait until E0 is a proper,
  reproducible script and E1 has a measured cost.

## Reproduction

The probe is a throwaway script, `e0_probe.py`, kept in full below. It needs only the
repo's `powerladder` package; set the `sys.path` line to the repo root. Each run uses
one core and takes about 20 s per arm.

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
# Run A: the twelve original arms, seed 0
python e0_probe.py 200
# Run B: lognormal sweep, seed 0
python e0_probe.py 200 mean8_lognorm0.3 mean8_lognorm0.5 mean8_lognorm0.6 \
  mean8_lognorm0.7 mean8_lognorm0.8 mean8_lognorm0.9 mean8_lognorm1.0 \
  mean8_lognorm1.1 mean8_lognorm1.2 mean8_lognorm1.5
# Run C: replicate, seed 1
E0_SEED=1 python e0_probe.py 200 ship_0.7 mean8_lognorm0.8 mean8_lognorm1.0
```

The positive-population seed is `[SEED, 99, 1 + k]`, where k is the arm's position
in the command. The same arm in a different position therefore gets an independent
population. That is why s = 1.0 has three values: run A (0.735), run B (0.725) and
run C (0.740). The shipped σ_G = 0.7 arm has two: run A (0.520) and run C (0.545).

<details>
<summary><code>e0_probe.py</code></summary>

```python
"""THROWAWAY E0 spike: which part of the micro-step distribution hides the line?

Patches the work-variation branch of powerladder.ko_workload._periodic_F_meta so
the per-iteration micro-step target g_real comes from a custom sampler, then
scores the Viterbi tracker on the section-2 aggregate at FAR 1e-2, against the
same aggregate null as results/st2/aggregate. Not product code.

Usage: python e0_probe.py [n_each] [arm ...]
"""
from __future__ import annotations

import inspect
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, "/fred/oz022/tkimpson/detecting-training-from-power-traces")
import powerladder.ko_workload as kw  # noqa: E402
from powerladder.config import DEFAULT  # noqa: E402
from powerladder.typeb.gate import DETECTORS, score_population  # noqa: E402
from powerladder.typeb.ko_synth import ko_make_aggregate_trace  # noqa: E402
from powerladder.typeb.roc import auc, tpr_at_far  # noqa: E402
from powerladder.typeb.st2 import _effective_meter  # noqa: E402

# --- patch: g_real <- _G_SAMPLER(rng, xi_w) ---------------------------------
_OLD = "g_real = work_base_accum / (g_w * max(1.0 + xi_w, _MIN_JITTER_FACTOR))"
src = inspect.getsource(kw._periodic_F_meta)
assert src.count(_OLD) == 1
src = src.replace(_OLD, "g_real = _G_SAMPLER(rng, xi_w)")
kw._G_SAMPLER = None
exec(compile(src, kw.__file__, "exec"), kw.__dict__)   # rebinds module global

FLOOR = 0.05
SEED = int(os.environ.get("E0_SEED", 0))


def ko_form(cap=None, scale=1.0, tail=None):
    """The shipped 8/(1+xi) form, optionally capped, rescaled or one-tailed."""
    def f(rng, xi):
        if tail == "small":
            xi = abs(xi)        # 1+|xi| >= 1 -> G <= 8
        elif tail == "large":
            xi = -abs(xi)       # 1-|xi| <= 1 -> G >= 8
        g = scale * 8.0 / max(1.0 + xi, FLOOR)
        return min(g, cap) if cap is not None else g
    return f


def lognormal(s, cap=160.0):
    def f(rng, xi):
        return min(8.0 * np.exp(s * rng.standard_normal() - s * s / 2), cap)
    return f


def two_point(lo, hi):
    p_hi = (8.0 - lo) / (hi - lo)       # mean exactly 8
    def f(rng, xi):
        return float(hi if rng.random() < p_hi else lo)
    return f


# mean of the sigma=0.7 shipped form, to rescale it to mean 8
_rng = np.random.default_rng(1)
_m07 = np.mean(8.0 / np.maximum(1 + _rng.normal(0, 0.7, 10**6), FLOOR))

ARMS = {
    # name: (sampler, work_sigma fed to xi_w)
    "ship_0.7": (ko_form(), 0.7),
    "ship_0.5": (ko_form(), 0.5),
    "cap16": (ko_form(cap=16), 0.7),
    "cap32": (ko_form(cap=32), 0.7),
    "cap64": (ko_form(cap=64), 0.7),
    "small_tail": (ko_form(tail="small"), 0.7),
    "large_tail": (ko_form(tail="large"), 0.7),
    "mean8_rescaled": (ko_form(scale=8.0 / _m07), 0.7),
    "mean8_lognorm1.0": (lognormal(1.0), 0.7),
    "mean8_lognorm1.5": (lognormal(1.5), 0.7),
    "mean8_2pt_2_62": (two_point(2, 62), 0.7),
    "mean8_2pt_1_64": (two_point(1, 64), 0.7),
}
# follow-up: lognormal spread sweep at mean 8
for _s in (0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.1, 1.2):
    ARMS[f"mean8_lognorm{_s}"] = (lognormal(_s), 0.7)


def g_samples(sampler, sigma, n=200_000, seed=2):
    """Integer G draws exactly as the generator makes them (stoch. round, >=1)."""
    rng = np.random.default_rng(seed)
    out = np.empty(n)
    for i in range(n):
        xi = rng.normal(0.0, sigma)
        g = sampler(rng, xi)
        lo = np.floor(g)
        out[i] = max(int(lo) + (1 if rng.random() < g - lo else 0), 1)
    return out


def penalty(G, b_noise):
    """McCandlish extra-token fraction vs honest G=8 (see notes/plans)."""
    return (8 / (8 + b_noise)) / 8 / (np.mean(G / (G + b_noise)) / G.mean()) - 1


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    arms = sys.argv[2:] or list(ARMS)[:12]
    ko, glue = DEFAULT.ko, DEFAULT.ko_typeb
    meter = _effective_meter(DEFAULT.st2)
    det = DETECTORS["viterbi"]
    far = 0.01

    t0 = time.time()
    rng = np.random.default_rng([SEED, 99, 0])
    negs = [ko_make_aggregate_trace("infer", ko, glue, rng, meter=meter)
            for _ in range(n)]
    neg = score_population(negs, det, glue)
    print(f"null: {n} traces in {time.time() - t0:.0f}s", flush=True)

    rows = []
    for k, name in enumerate(arms):
        sampler, sigma = ARMS[name]
        kw._G_SAMPLER = sampler
        t0 = time.time()
        rng = np.random.default_rng([SEED, 99, 1 + k])
        pos_tr = [ko_make_aggregate_trace("train", ko, glue, rng, meter=meter,
                                          work_sigma=sigma) for _ in range(n)]
        pos = score_population(pos_tr, det, glue)
        G = g_samples(sampler, sigma)
        row = dict(
            arm=name, tpr=tpr_at_far(pos, neg, far),
            hiding=1 - tpr_at_far(pos, neg, far),
            auc=auc(pos, neg), mean_G=G.mean(), median_G=float(np.median(G)),
            cv_G=G.std() / G.mean(), p_G_ge_64=float((G >= 64).mean()),
            steps_per_token=8 / G.mean(),
            pen_b8=penalty(G, 8), pen_b32=penalty(G, 32),
            pen_b128=penalty(G, 128),
        )
        rows.append(row)
        print(json.dumps({k2: (round(v, 3) if isinstance(v, float) else v)
                          for k2, v in row.items()})
              + f"  [{time.time() - t0:.0f}s]", flush=True)
    json.dump(rows, open(f"e0_results_n{n}_seed{SEED}_{len(arms)}arms.json", "w"),
              indent=1)


if __name__ == "__main__":
    main()
```

</details>
