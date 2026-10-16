#!/bin/bash
# E1 learning-cost pilot (exploratory): one training run per array task on one
# A100. Task i runs arm ARMS[i / 3] with seed i % 3 (5 arms x 3 seeds = 0-14):
#     0-2 honest   3-5 shipped0.7   6-8 lognorm0.8   9-11 lognorm1.0
#     12-14 lognorm1.0_sqrtlr
# MICRO_BATCH (sequences per micro-step) is required; EXTRA_ARGS is passed through.
#     sbatch --array=0-14 --export=ALL,MICRO_BATCH=16 scripts/slurm/e1_arms.sbatch
# Smoke (2M tokens, arms honest and lognorm1.0):
#     sbatch --array=0,9 --time=00:30:00 \
#         --export=ALL,MICRO_BATCH=16,EXTRA_ARGS=--smoke scripts/slurm/e1_arms.sbatch
# Noise-scale scan (honest arm, probes on, one micro-batch size per job):
#     sbatch --array=0 --export=ALL,MICRO_BATCH=8,EXTRA_ARGS="--tokens 150e6 --probe-every 2e6 --tag bnoise" \
#         scripts/slurm/e1_arms.sbatch
#SBATCH --account=oz022
#SBATCH --job-name=e1arm
#SBATCH --partition=milan-gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=results/e1/slurm/e1arm_%A_%a.out

set -euo pipefail
REPO="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO"
: "${MICRO_BATCH:?set MICRO_BATCH}"
ARMS=(honest shipped0.7 lognorm0.8 lognorm1.0 lognorm1.0_sqrtlr)  # == schedules.ARMS
i=${SLURM_ARRAY_TASK_ID:-0}
arm=${ARMS[$((i / 3))]}
seed=$((i % 3))
nvidia-smi --query-gpu=name,power.limit --format=csv,noheader
# shellcheck disable=SC2086  # EXTRA_ARGS is deliberately word-split
.venv-gpu/bin/python scripts/e1/train.py --arm "$arm" --seed "$seed" \
    --micro-batch "$MICRO_BATCH" --data-dir /fred/oz022/tkimpson/e1_data ${EXTRA_ARGS:-}
