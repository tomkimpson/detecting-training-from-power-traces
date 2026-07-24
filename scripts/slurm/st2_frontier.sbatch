#!/bin/bash
# ST2 de-periodicisation frontier — per-family sweeps as a Slurm array, one
# family per task (Phase 2 full-resolution number-freeze). CPU-only synthetic;
# the `compute` partition runs all nine families concurrently. Each task reads
# SLURM_ARRAY_TASK_ID and sweeps one attack family over its St2Params level
# grid, writing a DISJOINT results/st2/<family>_summary.json (+ figure) — no
# cross-task race, so no flock needed. Assemble the frontier afterwards on the
# login node (cheap JSON aggregation, not a sweep):
#     python scripts/plot_st2_frontier.py   # -> results/st2/frontier_summary.json
#
# The array range must match the family count. Print it first:
#     python scripts/plot_st2_sweeps.py --list        # -> "9 families"
# then submit with the range 0..N-1 (override the #SBATCH default if it drifts):
#     sbatch --array=0-8 scripts/slurm/st2_frontier.sbatch
#
#SBATCH --job-name=st2front
#SBATCH --partition=compute
#SBATCH --array=0-8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=results/st2/slurm/st2frontier_%A_%a.out

set -euo pipefail

# Keep BLAS single-threaded: one family per task, no intra-task thread
# contention. Also the canonical build for the frozen numbers.
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

REPO="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO"
mkdir -p results/st2/slurm

# Activate the CPU venv created for this repo (see handoff / README).
source venv/bin/activate

echo "family ${SLURM_ARRAY_TASK_ID} on $(hostname) at $(date)"
python scripts/plot_st2_sweeps.py --array-id "${SLURM_ARRAY_TASK_ID}"
