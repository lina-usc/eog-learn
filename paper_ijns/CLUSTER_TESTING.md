# Cluster Testing Guide — paper_ijns SLURM Pipeline

This guide covers the tests that must be run on the HPC cluster to validate
the SLURM pipeline before submitting a full run.

The shell syntax checks (listed in the Prerequisites section) are automated
and have already been verified to pass locally before this guide was written.

---

## Prerequisites (verify before copying to cluster)

All local syntax checks should pass:

```bash
bash -n paper_ijns/slurm/submit_all.sh
bash -n paper_ijns/slurm/01_lstm.sbatch
bash -n paper_ijns/slurm/02_ica.sbatch
bash -n paper_ijns/slurm/03_simnibs.sbatch
bash -n paper_ijns/slurm/04_sim.sbatch
bash -n paper_ijns/slurm/05_xarray.sbatch
bash -n paper_ijns/slurm/06_analysis.sbatch
python3 -c "import yaml; yaml.safe_load(open('paper_ijns/slurm/config.yml'))"
```

Expected: no output from the `bash -n` commands; no errors from the Python command.

---

## Step 1 — Edit `slurm/config.yml`

Update the settings for your cluster:

```yaml
partition: AI_Center          # your SLURM partition name
module: "python3/anaconda/2023.9"
venv: /work/co20/eog_lstm/venv_lstm   # path to venv with eog-learn installed
simnibs_env: simnibs4         # conda env with SimNIBS ≥ 4
root: processed/              # output directory
recompute: true
```

---

## Step 2 — Verify Python environment and subjects list

```bash
cd /path/to/eog-learn/paper_ijns

module load python3/anaconda/2023.9
source /work/co20/eog_lstm/venv_lstm/bin/activate

python3 -c "
import sys; sys.path.insert(0, '.')
import eoglearn
subjects = sorted(eoglearn.datasets.eegeyenet.get_subjects_runs().keys())
print(f'Found {len(subjects)} subjects:')
print(subjects)
"
```

**Expected output:** ~31 subject IDs printed, e.g.:
```
Found 31 subjects:
['AAO', 'EP10', 'EP12', 'EP14', 'EP18', 'EP23', ..., 'EP38']
```

---

## Step 3 — SLURM dry-run (validates without submitting)

```bash
sbatch --test-only \
    --array=0-0 \
    --export=ALL,CONDITION=perrecording,ROOT=processed_test/,RECOMPUTE_FLAG="",\
MODULE=python3/anaconda/2023.9,VENV=/work/co20/eog_lstm/venv_lstm,\
SCRIPT_DIR=$(pwd) \
    paper_ijns/slurm/01_lstm.sbatch
```

**Expected:** SLURM prints something like:
```
sbatch: Job submission to batch system would have id = 12345
```
Exit code 0. If you see errors about unknown partitions, invalid memory, or
missing directives, check your `config.yml` settings.

---

## Step 4 — Single-subject smoke test

This test runs the full pipeline for one subject to verify end-to-end correctness
before committing to the full 31-subject run.

**4a.** Temporarily edit `config.yml`:
```yaml
root: processed_test/
recompute: true
```

**4b.** Temporarily edit `submit_all.sh` — find the line that sets `ARRAY` and
override it for one subject (e.g. the first in `subjects.txt`):
```bash
# Change this line:
ARRAY="0-$((N - 1))"
# To:
ARRAY="0-0"
```

**4c.** Submit:
```bash
bash paper_ijns/slurm/submit_all.sh
```

**4d.** Monitor:
```bash
watch squeue -u $USER
# Or for more detail:
squeue -u $USER --format="%.10i %.20j %.8T %.10M %E"
```

**4e.** After all jobs finish, verify output files:
```bash
ls processed_test/ | sort
```

Expected files (one subject, one or more runs, e.g. `EP10`):
```
EP10_1_clean.edf              EP10_1_noise.edf
EP10_1_clean_acrosssubject.edf EP10_1_noise_acrosssubject.edf
EP10_1_clean_persubject.edf   EP10_1_noise_persubject.edf
EP10_1_ica.edf                EP10_1_noiseica.edf
EP10_1_original.edf
EP10_1_sim.edf                EP10_1_noisesim.edf
EP10_1_simlocal.edf           EP10_1_noisesimlocal.edf
```

Plus xarray outputs after Step 5 completes:
```
snr.netcdf            topo_erp.netcdf      et_signals.netcdf
eeg_signals.netcdf    topo_raw.netcdf
snr_diff.netcdf       topo_erp_diff.netcdf et_signals_diff.netcdf
eeg_signals_diff.netcdf topo_raw_diff.netcdf
```

**4f.** Restore `config.yml` and `submit_all.sh` after the smoke test.

---

## Step 5 — Inspect logs for errors

```bash
# Any log files containing Python errors?
grep -il "error\|traceback\|exception" paper_ijns/slurm/logs/*.out | head -20

# Tail a specific log:
tail -50 paper_ijns/slurm/logs/lstm_<JOB_ID>_0.out

# Check exit codes (all lines should end with "Done:")
grep "Done:" paper_ijns/slurm/logs/lstm_*.out | wc -l
```

Expected: no files from `grep -il`, all tasks show "Done:" in their logs.

---

## Step 6 — Verify SLURM dependency chain

Shortly after submitting, run:
```bash
squeue -u $USER --format="%.10i %.25j %.8T %E" | sort -k3
```

Expected output structure:
- Steps 1.1 × 3, 1.2, and 1.4 (if SimNIBS was needed): status `RUNNING` or `PENDING`
- Step 5 (xarray): status `DEPENDENCY` with all five job IDs listed
- Steps 6–9: status `DEPENDENCY` pending on Step 5

---

## Step 7 — Full pipeline submission

Once the smoke test passes, revert any overrides and submit the full pipeline:

```bash
# Ensure config.yml has the correct final settings:
#   root: processed/
#   recompute: true   (or false if restarting from partial results)

bash paper_ijns/slurm/submit_all.sh
```

---

## Resubmitting a single failed subject

If one array task fails, resubmit only that subject without re-running the rest:

```bash
# Find the failed subject (e.g. subject index 5 = 6th line of subjects.txt)
SUBJECT=$(awk "NR==6" paper_ijns/slurm/subjects.txt)

# Resubmit just that subject
sbatch --array=5 \
    --export=ALL,CONDITION=perrecording,ROOT=processed/,RECOMPUTE_FLAG="--no-recompute",\
MODULE=python3/anaconda/2023.9,VENV=/work/co20/eog_lstm/venv_lstm,\
SCRIPT_DIR=/path/to/eog-learn/paper_ijns \
    paper_ijns/slurm/01_lstm.sbatch
```

---

## Useful monitoring commands

```bash
# All your jobs, with state and dependency info
squeue -u $USER --format="%.10i %.25j %.8T %.10M %E"

# Watch queue (refreshes every 5 seconds)
watch -n 5 "squeue -u $USER"

# Show completed/failed jobs (last 24 hours)
sacct -u $USER --starttime=now-1day --format=JobID,JobName,State,ExitCode,Elapsed

# Tail a log file
tail -f paper_ijns/slurm/logs/lstm_<JOB_ID>_<ARRAY_ID>.out

# Count completed tasks for a step
grep -c "Done:" paper_ijns/slurm/logs/lstm_*.out
```
