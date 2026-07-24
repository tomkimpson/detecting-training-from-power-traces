#!/bin/bash
# ST1 surrogate confirmation (number-freeze) as a Slurm array — one
# (stage x null x calib) cell per task. CPU-only synthetic; the `compute`
# partition (252 CPU / 458 GB) runs the whole grid concurrently. Each task reads
# SLURM_ARRAY_TASK_ID, scores one surrogate-calibrated cell, and persists
# crash-safe to results/st1/ (per-cell raw npz + flock-merged far_summary.json).
#
# Grid: --calibs surrogate over stages {1,3,4} x 8 nulls (white, ar1,
# ar2_resonant, ar1_t, lognormal, sq_gauss, tvar, controller) = 24 cells, at
# the full surrogate sizing M=2000 (n_null_surrogate_cells cap) x S=999.
# stage2 is intentionally excluded (surrogate under a known warp is unsupported).
#
# The array range must match the cell count. Print it first (SAME args):
#     python scripts/st1_far.py --calibs surrogate \
#         --stages stage1,stage3,stage4 \
#         --nulls white,ar1,ar2_resonant,ar1_t,lognormal,sq_gauss,tvar,controller \
#         --list                                   # -> "24 cells"
# then submit with the range 0..N-1 (override the #SBATCH default if it drifts):
#     sbatch --array=0-23 scripts/slurm/st1_far_surrogate.sbatch
# A single-cell timing test first is wise (surrogate cells are the expensive ones):
#     sbatch --array=0-0 scripts/slurm/st1_far_surrogate.sbatch
#
#SBATCH --job-name=st1surr
#SBATCH --partition=compute
#SBATCH --array=0-23
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --output=results/st1/slurm/st1surr_%A_%a.out

set -euo pipefail

# Keep BLAS single-threaded: one cell per task, no intra-cell thread contention.
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

REPO="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO"
mkdir -p results/st1/slurm

# Activate the CPU venv created for this repo (see handoff / README).
source .venv/bin/activate

echo "cell ${SLURM_ARRAY_TASK_ID} on $(hostname) at $(date)"
python scripts/st1_far.py \
    --calibs surrogate \
    --stages stage1,stage3,stage4 \
    --nulls white,ar1,ar2_resonant,ar1_t,lognormal,sq_gauss,tvar,controller \
    --array-id "${SLURM_ARRAY_TASK_ID}"
