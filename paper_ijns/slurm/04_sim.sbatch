#!/usr/bin/env bash
# =============================================================================
# 04_sim.sbatch — Biophysical simulation cleaning (array job, one per subject)
#
# Requires leadfield_elect_only.npy to be present in SCRIPT_DIR (produced by
# 03_simnibs.sbatch).  submit_all.sh sets this job's dependency accordingly.
#
# Required --export variables (set by submit_all.sh):
#   ROOT, RECOMPUTE_FLAG, MODULE, VENV, SCRIPT_DIR
# =============================================================================
#SBATCH --cpus-per-task=5
#SBATCH --mem=8G
#SBATCH --time=4:00:00
#SBATCH --output=slurm/logs/sim_%A_%a.out

set -euo pipefail

module load "$MODULE"
source "$VENV/bin/activate"

cd "$SCRIPT_DIR"

SUBJECT=$(awk "NR==$((SLURM_ARRAY_TASK_ID + 1))" slurm/subjects.txt)
echo "[$(date '+%H:%M:%S')] Task $SLURM_ARRAY_TASK_ID → subject=$SUBJECT"

python 1.4_run_eog_lstm_sim_mp.py \
    --subjects "$SUBJECT" \
    --root "$ROOT" \
    $RECOMPUTE_FLAG

echo "[$(date '+%H:%M:%S')] Done: subject=$SUBJECT"
