#!/usr/bin/env bash
# =============================================================================
# 06_analysis.sbatch — Run analysis notebooks (Steps 6–9) sequentially
#
# Single serial job.  Depends on 05_xarray.sbatch completing successfully.
# Each analysis notebook is run as a plain Python script so that all
# computations execute and output files are written; figures are not displayed.
# Open notebooks interactively with `marimo edit` to explore results visually.
#
# Required --export variables (set by submit_all.sh):
#   ROOT, MODULE, VENV, SCRIPT_DIR
# =============================================================================
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=16:00:00
#SBATCH --output=slurm/logs/analysis_%j.out

set -euo pipefail

module load "$MODULE"
source "$VENV/bin/activate"

cd "$SCRIPT_DIR"

export EOG_PROCESSED_PATH="$ROOT"

echo "[$(date '+%H:%M:%S')] Step 6 — ERP analysis & figures"
python 3_Pytorch_EOG_LSTM_analysis.py

echo "[$(date '+%H:%M:%S')] Step 7 — Generalisation analysis"
python 4_generalization_analysis.py

echo "[$(date '+%H:%M:%S')] Step 8 — Power spectrum analysis"
python 5_power_spectrum_analysis.py

echo "[$(date '+%H:%M:%S')] Step 9 — Functional connectivity (ciPLV)"
python 6_functional_connectivity_analysis.py

echo "[$(date '+%H:%M:%S')] All analysis steps complete."
echo ""
echo "To explore results interactively:"
echo "  marimo edit paper_ijns/3_Pytorch_EOG_LSTM_analysis.py"
echo "  marimo edit paper_ijns/4_generalization_analysis.py"
echo "  marimo edit paper_ijns/5_power_spectrum_analysis.py"
echo "  marimo edit paper_ijns/6_functional_connectivity_analysis.py"
