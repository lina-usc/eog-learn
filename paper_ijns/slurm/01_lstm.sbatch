#!/usr/bin/env bash
# =============================================================================
# 01_lstm.sbatch — LSTM regression cleaning (array job, one task per subject)
#
# Required --export variables (set by submit_all.sh):
#   CONDITION    perrecording | persubject | acrosssubject
#   ROOT         output directory
#   RECOMPUTE_FLAG  "" or "--no-recompute"
#   MODULE       anaconda module name
#   VENV         path to virtual environment
#   SCRIPT_DIR   path to paper_ijns/
#
# Resource defaults below are overridden per-condition by submit_all.sh:
#   perrecording:   --cpus-per-task=5 --mem=8G   --time=1:00:00
#   persubject:     --cpus-per-task=5 --mem=10G  --time=2:00:00
#   acrosssubject:  --cpus-per-task=2 --mem=20G  --time=6:00:00
# =============================================================================
#SBATCH --cpus-per-task=5
#SBATCH --mem=8G
#SBATCH --time=1:00:00
#SBATCH --output=slurm/logs/lstm_%A_%a.out
#SBATCH --error=slurm/logs/lstm_%A_%a.err

set -euo pipefail

module load "$MODULE"
source "$VENV/bin/activate"

cd "$SCRIPT_DIR"

SUBJECT=$(awk "NR==$((SLURM_ARRAY_TASK_ID + 1))" slurm/subjects.txt)
echo "[$(date '+%H:%M:%S')] Task $SLURM_ARRAY_TASK_ID → subject=$SUBJECT  condition=$CONDITION"

python 1.1_run_eog_lstm_regression_mp.py \
    --subjects "$SUBJECT" \
    --condition "$CONDITION" \
    --root "$ROOT" \
    $RECOMPUTE_FLAG

echo "[$(date '+%H:%M:%S')] Done: subject=$SUBJECT  condition=$CONDITION"
