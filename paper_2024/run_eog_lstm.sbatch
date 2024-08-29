#!/bin/bash
#SBATCH --array=0-176
#SBATCH --time=1:00:00
#SBATCH --mem-per-cpu=2G
#SBATCH -p AI_Center

module load python3/anaconda/2023.9
source /work/co20/eog_lstm/venv_lstm/bin/activate

./run_eog_lstm_regression.py $SLURM_ARRAY_TASK_ID