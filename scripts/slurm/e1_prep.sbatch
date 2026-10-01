#!/bin/bash
# E1 learning-cost pilot (exploratory): tokenise the pinned FineWeb-edu shard into
# /fred/oz022/tkimpson/e1_data/{val,train}.bin and write
# results/e1/data_manifest.json. Fetch the shard on a login node first (compute
# nodes may have no internet):
#     HF_HOME=/fred/oz022/tkimpson/hf_cache .venv-gpu/bin/python scripts/e1/prep_data.py --download-only
#     sbatch scripts/slurm/e1_prep.sbatch
#SBATCH --account=oz022
#SBATCH --job-name=e1prep
#SBATCH --partition=milan
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=results/e1/slurm/e1prep_%A.out

set -euo pipefail
REPO="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO"
export HF_HOME=/fred/oz022/tkimpson/hf_cache
export HF_HUB_OFFLINE=1
export TIKTOKEN_CACHE_DIR=/fred/oz022/tkimpson/hf_cache/tiktoken
.venv-gpu/bin/python scripts/e1/prep_data.py
