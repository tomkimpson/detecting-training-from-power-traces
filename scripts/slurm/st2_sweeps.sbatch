#!/bin/bash
# ST2 per-family sweep re-verification (number-freeze) as a Slurm array — one
# attack family per task. CPU-only synthetic; the `compute` partition
# (252 CPU / 458 GB) runs all nine families concurrently. Each task reads
# SLURM_ARRAY_TASK_ID, sweeps one family of FAMILY_ORDER over the St2Params level
# grid, and writes results/st2/<family>_summary.json + figures/st2_<family>.*
# independently (no shared-file race).
#
# Regenerates the nine per-family summaries the frontier assembles; the `meter`
# family covers the integrating_1hz channel claim. Run the frontier step
# (scripts/slurm/st2_frontier.sbatch) after this array completes.
#
# The array range must match the family count. Print it first:
#     python scripts/plot_st2_sweeps.py --list        # -> "9 families"
# then submit with the range 0..N-1 (override the #SBATCH default if it drifts):
#     sbatch --array=0-8 scripts/slurm/st2_sweeps.sbatch
#
#SBATCH --job-name=st2sweep
#SBATCH --partition=compute
#SBATCH --array=0-8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=results/st2/slurm/st2sweep_%A_%a.out

set -euo pipefail

# Keep BLAS single-threaded: one family per task, no intra-cell thread contention.
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

REPO="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO"
mkdir -p results/st2/slurm

# Activate the CPU venv created for this repo (see handoff / README).
source .venv/bin/activate

echo "family ${SLURM_ARRAY_TASK_ID} on $(hostname) at $(date)"
python scripts/plot_st2_sweeps.py \
    --detectors full \
    --array-id "${SLURM_ARRAY_TASK_ID}"
