#!/usr/bin/env bash
# =============================================================================
# 05_xarray.sbatch — Aggregate all processed EDFs into xarray .netcdf files
#
# Single serial job.  Depends on ALL array jobs from Steps 1.1/1.2/1.4 finishing
# in any state (dependency set by submit_all.sh via afterany).
#
# Required --export variables (set by submit_all.sh):
#   ROOT, MODULE, VENV, SCRIPT_DIR
# =============================================================================
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --output=slurm/logs/xarray_%j.out

set -euo pipefail

module load "$MODULE"
source "$VENV/bin/activate"

cd "$SCRIPT_DIR"

echo "[$(date '+%H:%M:%S')] Aggregating processed EDF files from: $ROOT"

python 2_LSTM_compute_xr.py --root "$ROOT"

echo "[$(date '+%H:%M:%S')] Done. netcdf files written to $SCRIPT_DIR"
