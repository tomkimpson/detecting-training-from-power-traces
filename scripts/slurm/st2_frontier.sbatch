#!/bin/bash
# ST2 frontier assembly (number-freeze) — single Slurm task. Assembles the nine
# per-family summaries onto the shared physical x-axis, attaches the measured
# cost anchors, applies the pre-registered GO/NO-GO criteria, and writes
# results/st2/frontier_summary.json + figures/st2_frontier.*. CPU-trivial
# aggregation, but run on slurm to honour the "no dev-node runs" decision.
#
# Run AFTER scripts/slurm/st2_sweeps.sbatch has produced all nine
# results/st2/<family>_summary.json. Chain it on the array job id:
#     sw=$(sbatch --parsable --array=0-8 scripts/slurm/st2_sweeps.sbatch)
#     sbatch --dependency=afterok:${sw} scripts/slurm/st2_frontier.sbatch
#
# NOTE: --b2-dir MUST point at data/measured_cost_anchors (the frontier default
# results/b2 does not exist in this repo, and would silently yield no anchors).
#
#SBATCH --job-name=st2front
#SBATCH --partition=compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=results/st2/slurm/st2front_%A.out

set -euo pipefail

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

REPO="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO"
mkdir -p results/st2/slurm

# Activate the CPU venv created for this repo (see handoff / README).
source .venv/bin/activate

echo "frontier assembly on $(hostname) at $(date)"
python scripts/plot_st2_frontier.py --b2-dir data/measured_cost_anchors
