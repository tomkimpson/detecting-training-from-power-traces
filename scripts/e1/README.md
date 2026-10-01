# E1: learning-efficiency cost of work variation (exploratory pilot)

**Status:** exploratory. This is **outside the scope of `spec.md`**, which defers
measured learning-efficiency penalties to future work. Nothing here feeds the paper
or the CPU pipeline. It is the only GPU code in the repo and has its own
dependencies (`requirements-gpu.txt`).

**Why:** the work-variation attack hides the cadence at zero throughput cost (§6).
The E0 spike found that a mean-8 lognormal micro-step schedule hides even more, at a
small *predicted* learning cost. E1 measures that cost by training a small GPT
under each schedule and comparing tokens to a target validation loss.

**Background:**
- The plan: [`notes/plans/work-variation-hidden-cost-experiments.md`](../../notes/plans/work-variation-hidden-cost-experiments.md)
- The E0 results: [`notes/results/e0-work-variation-spike-findings.md`](../../notes/results/e0-work-variation-spike-findings.md)

## Files

| File | Role |
|---|---|
| `schedules.py` | Micro-step count per optimiser step for each arm (same distributions as E0) |
| `noise_scale.py` | McCandlish et al. (2018) B_simple estimator |
| `train.py` | One training run; logs val loss against tokens, plus noise-scale probes |
| `analyse.py` | Tokens-to-target cost per arm, predicted against measured; summary and figure |
| `prep_data.py` | Tokenises a pinned FineWeb-edu shard to `uint16` `.bin` files |
| `model.py` | nanoGPT's `model.py`, vendored (MIT) at commit `3adf61e` |

## Reproduce (OzSTAR)

```bash
# Environment: the conda base has torch 2.5.1+cu124
/fred/oz022/tkimpson/miniconda3/bin/python -m venv --system-site-packages .venv-gpu
.venv-gpu/bin/pip install -r requirements-gpu.txt

# Data: download on a login node (compute nodes have no internet), tokenise on slurm
export HF_HOME=/fred/oz022/tkimpson/hf_cache
export TIKTOKEN_CACHE_DIR=/fred/oz022/tkimpson/hf_cache/tiktoken
.venv-gpu/bin/python scripts/e1/prep_data.py --download-only
sbatch scripts/slurm/e1_prep.sbatch          # -> results/e1/data_manifest.json

# Smoke, noise-scale scan, then the 15 runs (see the header of the sbatch script)
sbatch --array=0,9 --time=00:30:00 --export=ALL,MICRO_BATCH=16,EXTRA_ARGS=--smoke \
    scripts/slurm/e1_arms.sbatch
sbatch --array=0 --export=ALL,MICRO_BATCH=16,EXTRA_ARGS="--tokens 150e6 --probe-every 2e6 --tag bnoise" \
    scripts/slurm/e1_arms.sbatch
sbatch --array=0-14 --export=ALL,MICRO_BATCH=<m> scripts/slurm/e1_arms.sbatch

# Analysis (CPU)
python scripts/e1/analyse.py --micro-batch <m>   # -> results/e1/summary.json, figures/e1_loss_curves.*
```

The tests (`tests/test_e1_*.py`) run on CPU. The schedule, noise-scale and analysis
tests are numpy-only. The accumulation test needs torch, and is skipped without it.
