#!/bin/bash
# Meter-requirement boundary sweep (Phase 2) as a Slurm array — one cell per
# task. CPU-only synthetic; the `compute` partition (252 CPU / 458 GB) runs the
# whole grid concurrently. Each task reads SLURM_ARRAY_TASK_ID and scores one
# MeterParams channel, persisting crash-safe to results/st2/ (per-cell raw JSON
# + flock-merged meter_boundary_summary.json).
#
# The array range must match the cell count. Print it first:
#     python scripts/st2_meter_boundary.py --list        # -> "<N> cells"
# then submit with the range 0..N-1 (override the #SBATCH default if it drifts):
#     sbatch --array=0-47 scripts/slurm/st2_meter_boundary.sbatch
#
#SBATCH --job-name=st2meter
#SBATCH --partition=compute
#SBATCH --array=0-47
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=results/st2/slurm/st2meter_%A_%a.out

set -euo pipefail

# Keep BLAS single-threaded: one cell per task, no intra-cell thread contention.
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

REPO="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO"
mkdir -p results/st2/slurm

# Activate the CPU venv created for this repo (see handoff / README).
source venv/bin/activate

echo "cell ${SLURM_ARRAY_TASK_ID} on $(hostname) at $(date)"
python scripts/st2_meter_boundary.py --array-id "${SLURM_ARRAY_TASK_ID}"
