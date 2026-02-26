#!/usr/bin/env bash
# =============================================================================
# 02_ica.sbatch — ICA + ICLabel cleaning (array job, one task per subject)
#
# Required --export variables (set by submit_all.sh):
#   ROOT, RECOMPUTE_FLAG, MODULE, VENV, SCRIPT_DIR
# =============================================================================
#SBATCH --cpus-per-task=5
#SBATCH --mem=12G
#SBATCH --time=8:00:00
#SBATCH --output=slurm/logs/ica_%A_%a.out
#SBATCH --error=slurm/logs/ica_%A_%a.err

set -euo pipefail

module load "$MODULE"
source "$VENV/bin/activate"

cd "$SCRIPT_DIR"

SUBJECT=$(awk "NR==$((SLURM_ARRAY_TASK_ID + 1))" slurm/subjects.txt)
echo "[$(date '+%H:%M:%S')] Task $SLURM_ARRAY_TASK_ID → subject=$SUBJECT"

python 1.2_run_eog_lstm_ica_mp.py \
    --subjects "$SUBJECT" \
    --root "$ROOT" \
    $RECOMPUTE_FLAG

echo "[$(date '+%H:%M:%S')] Done: subject=$SUBJECT"
