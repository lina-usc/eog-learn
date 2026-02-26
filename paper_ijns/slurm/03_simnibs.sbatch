#!/usr/bin/env bash
# =============================================================================
# 03_simnibs.sbatch — Biophysical forward model (single job, runs once)
#
# Runs 1.3_eog_gen_model_2025.py under the SimNIBS conda environment, which
# requires SimNIBS ≥ 4 and Python 3.9.  Produces leadfield_elect_only.npy.
#
# Required --export variables (set by submit_all.sh):
#   SIMNIBS_ENV  name of conda env with SimNIBS installed
#   MODULE       anaconda module name (needed to access conda)
#   SCRIPT_DIR   path to paper_ijns/
# =============================================================================
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=1:00:00
#SBATCH --output=slurm/logs/simnibs_%j.out
#SBATCH --error=slurm/logs/simnibs_%j.err

set -euo pipefail

# Load the module so that 'conda' is available on $PATH
module load "$MODULE"

# Activate the SimNIBS conda environment
# shellcheck disable=SC1090
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$SIMNIBS_ENV"

cd "$SCRIPT_DIR"
echo "[$(date '+%H:%M:%S')] Running biophysical forward model (SimNIBS env: $SIMNIBS_ENV)"

python 1.3_eog_gen_model_2025.py

echo "[$(date '+%H:%M:%S')] Done. leadfield_elect_only.npy written to $SCRIPT_DIR"
