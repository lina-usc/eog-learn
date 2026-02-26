#!/usr/bin/env bash
# =============================================================================
# 05_xarray.sbatch — Aggregate all processed EDFs into xarray .netcdf files
#
# Single serial job.  Depends on ALL array jobs from Steps 1.1/1.2/1.4 finishing
# successfully (dependency set by submit_all.sh).
#
# Required --export variables (set by submit_all.sh):
#   ROOT, MODULE, VENV, SCRIPT_DIR
# =============================================================================
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --output=slurm/logs/xarray_%j.out
#SBATCH --error=slurm/logs/xarray_%j.err

set -euo pipefail

module load "$MODULE"
source "$VENV/bin/activate"

cd "$SCRIPT_DIR"

export EOG_PROCESSED_PATH="$ROOT"
echo "[$(date '+%H:%M:%S')] Aggregating processed EDF files from: $ROOT"

python 2_LSTM_compute_xr.py

echo "[$(date '+%H:%M:%S')] Done. netcdf files written to $SCRIPT_DIR"
