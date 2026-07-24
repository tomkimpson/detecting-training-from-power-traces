#!/bin/bash
# Rung-2 training-vs-inference classification sweep (Phase 2, plan §3.3) as a
# Slurm array — one cell per task. CPU-only synthetic; the `compute` partition
# runs the whole grid concurrently. Each task reads SLURM_ARRAY_TASK_ID and
# evaluates one cell (the stated population, a transfer shift, or a semantic
# control), persisting crash-safe to results/rung2/ (per-cell raw JSON +
# flock-merged rung2_summary.json).
#
# The array range must match the cell count. Print it first:
#     python scripts/rung2_eval.py --list         # -> "<N> cells"
# then submit with the range 0..N-1 (override the #SBATCH default if it drifts):
#     sbatch --array=0-13 scripts/slurm/rung2_eval.sbatch
#
#SBATCH --job-name=rung2
#SBATCH --partition=compute
#SBATCH --array=0-13
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=03:00:00
#SBATCH --output=results/rung2/slurm/rung2_%A_%a.out

set -euo pipefail

# Keep BLAS single-threaded (the physics feature extraction is numpy); the RF
# baseline's own joblib parallelism uses the 4 allocated CPUs.
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

REPO="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO"
mkdir -p results/rung2/slurm

# Activate the CPU venv created for this repo (see handoff / README).
source venv/bin/activate

echo "cell ${SLURM_ARRAY_TASK_ID} on $(hostname) at $(date)"
python scripts/rung2_eval.py --array-id "${SLURM_ARRAY_TASK_ID}"
