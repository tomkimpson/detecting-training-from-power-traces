# E1 pilot: the learning cost of work variation, measured

**Date:** 2026-10-01
**Status:** exploratory findings. **Outside the scope of `spec.md`**, which defers
measured learning-efficiency penalties. Not in the paper. This is E1 (with E2) of
[`../plans/work-variation-hidden-cost-experiments.md`](../plans/work-variation-hidden-cost-experiments.md);
it follows the E0 spike ([`e0-work-variation-spike-findings.md`](e0-work-variation-spike-findings.md)).
The harness is in `scripts/e1/` (see its README). Numbers come from
`results/e1/summary.json`; the figure is `figures/e1_loss_curves.{pdf,png}`.

## Question

The work-variation attack hides the training cadence at zero throughput cost. Does
it pay in learning efficiency instead, and how does that compare with the
noise-scale model's prediction?

## Headline

- **Every attack that hides costs a lot of learning efficiency, at least under a
  standard training recipe.** Honest training needs far fewer tokens to reach the
  loss each attack ends at:

  | Attack | Hiding (E0, synthetic) | Measured cost | Predicted cost |
  |---|---|---|---|
  | lognormal s = 0.8 | 0.61 | **61% ± 4** | 15% |
  | lognormal s = 1.0 | 0.73 | **93% ± 6** | 24% |
  | shipped σ_G = 0.7 (the §6 attack) | 0.53 | **267% ± 33** | 163% |

- **The model underestimates the cost by a factor of 1.6 to 4.** The prediction was
  committed before any arm finished (`8aa8576`).
- **The obvious best response made things worse.** Scaling the learning rate by
  √(G_i/8) raised the lognormal s = 1.0 cost from 93% to 116% ± 5.
- **What this does and doesn't establish.** The "free" attack is expensive for an
  operator using a standard recipe. It is **not yet** a lower bound on what a
  well-tuned attacker pays. See Caveats.

## Setup

- **Model:** nanoGPT, 29.9M parameters (6 layers, 384 wide, 6 heads), sequence
  length 512, bf16.
- **Optimiser:** AdamW with peak learning rate 1e-3, β = (0.9, 0.95), weight decay 0.1
  and gradient clipping at 1.0. The learning-rate schedule is indexed by tokens:
  2% warmup, then cosine to 10% of the peak.
- **Data:** FineWeb-edu `sample-10BT`, shard `000_00000`, revision `87f0914`, GPT-2
  tokenizer. 500M train tokens and 5M validation tokens from disjoint documents;
  see `results/e1/data_manifest.json`. Validation loss is measured every 5M tokens
  on a fixed 2M-token slice.
- **Budget:** D = 400M tokens per run, on one A100-SXM4-80GB (500 W), OzSTAR
  `milan-gpu`. About 14–17 minutes per run.
- **Batching:** micro-batch m = 16 sequences. Honest training takes G = 8
  micro-steps per optimiser step, a batch of 128 sequences (65.5k tokens).
- **Arms:** 5 arms × 3 seeds. Within a seed, every arm has the same initial weights
  and the same micro-batches in the same order. Arms differ only in how micro-batches
  are grouped into optimiser steps (the E0 distributions, `scripts/e1/schedules.py`)
  and, for one arm, in the per-step learning rate.

### Choosing the honest batch (E2)

The McCandlish simple noise scale B_simple was probed every 4M tokens in honest runs
at m = 8, 16 and 32, each over the full 400M-token budget. It was much the same
whichever micro-batch size measured it, and it rose through training as the
literature reports:

| Part of the run | 0–10% | 10–25% | 25–50% | 50–75% | 75–100% |
|---|---|---|---|---|---|
| B_simple (sequences), m = 16 | 4 | 57 | 103 | 163 | 310 |

The pre-set rule was the power-of-two m whose honest batch (8m) is closest to
B_simple at mid-run, about 160 sequences. It picked m = 16. That m also gave the best
final loss of the three (3.938, against 3.961 at m = 8 and 4.021 at m = 32).

Over the second half of training B_simple is 222 sequences (13.9 micro-batches).
Feeding that into the E0 penalty formula gives the predicted costs in the headline
table.

## Results

| Arm | Final validation loss (seeds 0, 1, 2) | Gap to honest | Measured cost | Predicted cost |
|---|---|---|---|---|
| honest | 3.939, 3.938, 3.946 | 0 | 0 | 0 |
| shipped σ_G = 0.7 | 4.364, 4.430, 4.375 | +0.449 ± 0.037 | 267% ± 33 | 163% |
| lognormal s = 0.8 | 4.074, 4.063, 4.064 | +0.126 ± 0.009 | 61% ± 4 | 15% |
| lognormal s = 1.0 | 4.140, 4.131, 4.128 | +0.192 ± 0.010 | 93% ± 6 | 24% |
| lognormal s = 1.0, √lr | 4.178, 4.172, 4.173 | +0.233 ± 0.006 | 116% ± 5 | 24% |

Spreads are the standard deviation over 3 seeds.

**The measured cost is D / (the tokens honest training needed to reach the arm's
final loss) − 1.** This metric was **added after the runs**. The pre-registered one
(tokens to reach the honest loss at 0.75·D) saturated: no attack arm reached that
target within D. So every arm's pre-registered cost is only the lower bound "> 33%",
which `summary.json` also records. The replacement metric is conservative in the
attacker's favour. An honest run at fewer tokens still has an undecayed learning
rate, so it reaches a given loss later than an honest run tuned to the shorter
budget would. That makes its token count too high and the attack's cost too low.

The gap is largest early in training: the shipped arm is about 1 nat behind at 40M
tokens. It then settles to a steady offset late in training rather than closing
(see the figure).

### Sanity checks

- **Honest seeds agree:** final losses within 0.008 of each other.
- **The logged schedules match E0:**

  | Arm | Mean G | Median G | Steps at G = 160 |
  |---|---|---|---|
  | shipped | 25.7 | 8 | 9.1% |
  | lognormal s = 0.8 | 7.9 | 6 | 0 |
  | lognormal s = 1.0 | 7.9 | 5 | 0 |

  The lognormal arms took about as many optimiser steps as honest (6,080–6,260
  against 6,104). The shipped arm took about a third as many (1,800–1,980).
- **Throughput can't be compared across arms here.** It ranged from 503k to 587k
  tokens/s. The variation is between nodes, not arms: honest runs sit at both ends.
  This pilot does not measure throughput cost; the hardware anchors do that.

## Interpretation

- **Batch-size spread costs far more than the noise-scale model says.** The model
  assumes SGD with the learning rate tuned at every step. Two features of a standard
  recipe could plausibly add cost on top. **Neither is tested yet:**
  - **Gradient clipping at a fixed threshold** clips the noisier small-batch steps
    harder.
  - **Adam's second-moment estimate** (β₂ = 0.95) averages over steps with very
    different gradient noise. That could mis-scale both the small-batch and the
    large-batch steps.
- **The √lr rule hurt.** It raises the learning rate on large-G steps: by 2.5× at
  the 99th percentile of G (G = 50) and by up to 4.5× at the cap (G = 160). The honest learning rate may already be near its stability
  limit.
- **For the rescue.** Against an operator who varies batch size with an otherwise
  standard recipe, hiding is expensive:
  - 0.61 hiding costs about 60% more compute;
  - 0.73 hiding costs about 90% more.

  That is the same order as idle-pad jitter, which buys 0.24 hiding for 159%
  throughput. The cheap, effective attack E0 found disappears, unless a tuned
  attacker can recover most of the gap.

## Caveats

1. **The attacker isn't tuned, so this is not a lower bound.** Only one best response
   was tried (√lr). An attacker could tune the peak learning rate, the clip
   threshold, β₂, or use a G-aware optimiser. The noise-scale model's 15–24% may be
   nearer what a well-tuned attacker pays. The rescue needs the cost a *tuned*
   attacker can't avoid.
2. **The honest run isn't tuned either.** The peak learning rate of 1e-3 was not
   swept. If honest training is sub-optimal, the measured costs are *understated*.
3. **Small scale.** 30M parameters, 400M tokens, one dataset. B_noise grows with
   training and scale, so the cost at frontier scale needs the scaling argument of
   the plan note (E2).
4. **The metric was chosen after the runs** (see Results), though it is conservative
   in the attacker's favour.
5. **Hiding and cost come from different systems.** The hiding numbers are E0's,
   from the synthetic generator. The costs are from real training. They are joined
   only through the shared G distribution.

## Next steps

- **E1b: a tuned attacker, and a tuned honest run.** Cheap, at about 15 minutes per
  run:
  - sweep the peak learning rate for honest and lognormal s = 1.0 (×0.5, ×1, ×2);
  - try lognormal s = 1.0 with clipping off (or the threshold scaled with G), and
    with β₂ = 0.99;
  - the minimum over these is the attacker's best measured cost.
- **Mechanism.** Log the clipping rate and per-step update norms by G, to test the
  two candidate mechanisms above.
- **The paper.** §6 is unchanged. If E1b holds, the frontier's cost axis needs a
  learning-efficiency column for work variation; that would also bring this work
  into scope, which needs a `spec.md` amendment.

## Reproduction

```bash
# runs (OzSTAR, one A100 each); see scripts/e1/README.md for data prep
for m in 8 16 32; do sbatch --array=0 --time=01:00:00 \
  --export=ALL,MICRO_BATCH=$m,EXTRA_ARGS="--probe-every 4e6 --tag bnoise" scripts/slurm/e1_arms.sbatch; done
sbatch --array=0-14 --time=01:00:00 --export=ALL,MICRO_BATCH=16 scripts/slurm/e1_arms.sbatch
# analysis
cd scripts/e1 && python analyse.py --micro-batch 16
```

Slurm jobs: noise scans 17840633–17840635, arms array 17841545. Run logs are in
`results/e1/<arm>_m16_s<seed>/log.jsonl` (tracked).
