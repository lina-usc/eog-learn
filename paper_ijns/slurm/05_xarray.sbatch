#!/usr/bin/env bash
# =============================================================================
# 05_xarray.sbatch — Aggregate all processed EDFs into xarray .netcdf files
#
# Array job: 4 tasks run in parallel, one per compute type:
#   0 → et        (eye-tracking SNR/topo, stimulus-locked)
#   1 → erp       (EEG signals/topo, stimulus-locked)
#   2 → et_diff   (eye-tracking SNR/topo, saccade-direction-locked)
#   3 → erp_diff  (EEG signals/topo, saccade-direction-locked)
#
# Depends on ALL array jobs from Steps 1.1/1.2/1.4 finishing in any state
# (dependency set by submit_all.sh via afterany).
#
# Required --export variables (set by submit_all.sh):
#   ROOT, MODULE, VENV, SCRIPT_DIR
# =============================================================================
#SBATCH --array=0-3
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=6:00:00
#SBATCH --output=slurm/logs/xarray_%A_%a.out

set -euo pipefail

module load "$MODULE"
source "$VENV/bin/activate"

cd "$SCRIPT_DIR"

TASKS=(et erp et_diff erp_diff)
TASK=${TASKS[$SLURM_ARRAY_TASK_ID]}

echo "[$(date '+%H:%M:%S')] Task $TASK — reading EDF files from: $ROOT"

python 2_LSTM_compute_xr.py --root "$ROOT" --task "$TASK" --condition "$CONDITION"

echo "[$(date '+%H:%M:%S')] Task $TASK done. netcdf written to $SCRIPT_DIR"
