#!/usr/bin/env bash
# =============================================================================
# 01_lstm.sbatch — LSTM regression cleaning (array job, one task per subject)
#
# Required --export variables (set by submit_all.sh):
#   CONDITION      perrecording | persubject | acrosssubject
#   ROOT           output directory
#   RECOMPUTE_FLAG "" or "--no-recompute"
#   NO_MP_FLAG     "" or "--no-multiprocessing" (lstm_as only)
#   MODULE         anaconda module name
#   VENV           path to virtual environment
#   SCRIPT_DIR     path to paper_ijns/
#
# Resource defaults below are overridden per-condition by submit_all.sh:
#   perrecording:   --cpus-per-task=5 --mem=8G   --time=1:00:00
#   persubject:     --cpus-per-task=5 --mem=10G  --time=2:00:00
#   acrosssubject:  --cpus-per-task=2 --mem=20G  --time=6:00:00
# =============================================================================
#SBATCH --cpus-per-task=5
#SBATCH --mem=8G
#SBATCH --time=4:00:00
#SBATCH --output=slurm/logs/lstm_%A_%a.out

set -euo pipefail

module load "$MODULE"
source "$VENV/bin/activate"

cd "$SCRIPT_DIR"

SUBJECT=$(awk "NR==$((SLURM_ARRAY_TASK_ID + 1))" slurm/subjects.txt)
echo "[$(date '+%H:%M:%S')] Task $SLURM_ARRAY_TASK_ID → subject=$SUBJECT  condition=$CONDITION"
ln -sf "lstm_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out" \
       "slurm/logs/lstm_${SUBJECT}_${CONDITION}.out" 2>/dev/null || true

python 1.1_run_eog_lstm_regression_mp.py \
    --subjects "$SUBJECT" \
    --condition "$CONDITION" \
    --root "$ROOT" \
    ${RECOMPUTE_FLAG:-} \
    ${NO_MP_FLAG:-}

echo "[$(date '+%H:%M:%S')] Done: subject=$SUBJECT  condition=$CONDITION"
