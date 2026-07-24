#!/bin/bash
# NP-optimal LRT ceiling for the ST1 bake-off (Phase 2, optional number-freeze).
# CPU-only synthetic. A SINGLE task: the shared Whittle training template bank
# dominates the cost (~10-15 min single-threaded), so splitting by negative class
# would only rebuild it. The driver builds the bank, fits the three ceilings
# (inference / structural / controller_only), reproduces the exact bake-off eval
# populations, scores ceiling + all corpus-free detectors, checks TPR parity against
# results/st1/bakeoff_summary.json, and writes results/st1/np_ceiling_summary.json.
# Assemble the figure afterwards on the login node (cheap; not a sweep):
#     python scripts/plot_st1_np_ceiling.py   # -> figures/st1_np_ceiling.*
#
# Submit (repo policy: freezes on Slurm, never the dev node):
#     sbatch scripts/slurm/st1_np_ceiling.sbatch
#
#SBATCH --job-name=st1npceil
#SBATCH --partition=compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=results/st1/slurm/st1npceil_%j.out

set -euo pipefail

# Single-threaded BLAS: the canonical build for the frozen numbers (matches the
# ST2 frontier freeze), no intra-process thread contention.
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

REPO="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO"
mkdir -p results/st1/slurm

# Activate the CPU venv created for this repo (see handoff / README).
source venv/bin/activate

echo "st1 np-ceiling freeze on $(hostname) at $(date)"
python scripts/st1_np_ceiling.py
python scripts/plot_st1_np_ceiling.py
echo "done at $(date)"
