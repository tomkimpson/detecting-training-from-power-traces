#!/bin/bash
# Section-5 headline on the section-2 aggregate: the ST1 bake-off and its Whittle
# NP ceiling, both with --scenario aggregate (dominant training over 4 small
# trainings + 4 fine-tunings, vs the inference-dominant aggregate null). CPU-only
# synthetic, a SINGLE task: the ceiling's parity guard reads the bake-off summary
# this job writes first. Outputs:
#     results/st1/bakeoff_aggregate_summary.json
#     results/st1/np_ceiling_aggregate_summary.json
#     figures/st1_bakeoff_aggregate.* , figures/st1_np_ceiling_aggregate.*
# The main-text figure is assembled afterwards (cheap, no sweep):
#     python scripts/plot_drift_ceiling.py      # -> figures/drift_ceiling.*
#
# Submit (repo policy: freezes on Slurm, never the dev node):
#     sbatch scripts/slurm/st1_aggregate_headline.sbatch
#
#SBATCH --job-name=st1agg
#SBATCH --partition=compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=03:00:00
#SBATCH --output=results/st1/slurm/st1agg_%j.out

set -euo pipefail

# Single-threaded BLAS: the canonical build for the frozen numbers.
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

REPO="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO"
mkdir -p results/st1/slurm

source .venv/bin/activate

echo "st1 aggregate headline on $(hostname) at $(date)"
python scripts/plot_st1_bakeoff.py --scenario aggregate
python scripts/st1_np_ceiling.py --scenario aggregate
python scripts/plot_st1_np_ceiling.py \
    --summary results/st1/np_ceiling_aggregate_summary.json \
    --stem st1_np_ceiling_aggregate
echo "done at $(date)"
