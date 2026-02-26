# paper_ijns — IJNS Paper Analysis Pipeline

This folder contains the analysis code for the IJNS paper comparing three approaches
for removing ocular (EOG) artefacts from simultaneous EEG + eye-tracking recordings:

1. **LSTM Regression** — a PyTorch LSTM trained to predict EOG from eye-tracking signals
2. **ICA + ICLabel** — independent component analysis with automatic eye-blink labelling
3. **Biophysical simulation** — a finite-element forward model (SimNIBS) mapping 3D eye
   dipole moments to the EEG sensor space

All analyses use the [EEGEyeNet](https://osf.io/ktv7m/) DOTS task dataset (39 subjects,
6 runs each, 129-channel EEG + binocular eye-tracking at 500 Hz).

---

## Installation

```bash
# From the repo root
pip install -e ".[extras,paper]"

# Additional dependencies for the pipeline scripts:
pip install torch mne-icalabel

# SimNIBS (required for 1.3 only — biophysical model generation):
# See https://simnibs.github.io/simnibs/build/html/installation/setup.html
```

---

## Pipeline Steps

Run the steps **in order**. Steps 1–4 write processed EDF files to a `processed/`
subdirectory. Steps 5–9 aggregate and visualise results.

The simplest way to run the full pipeline is the orchestration script:

```bash
# Run everything with defaults (output → paper_ijns/processed/)
python paper_ijns/run_all.py

# Custom output directory, skip already-computed files
python paper_ijns/run_all.py --root /data/eog_study/processed/ --no-recompute

# Run only data-processing steps (skip analysis notebooks)
python paper_ijns/run_all.py --skip-analysis

# Run only per-recording LSTM condition (fastest)
python paper_ijns/run_all.py --lstm-conditions perrecording

# Abort on first failure
python paper_ijns/run_all.py --fail-fast
```

> **Note:** Step 3 (biophysical forward model) requires SimNIBS + FreeSurfer and is
> always skipped by `run_all.py` (local mode). Run it interactively with `marimo edit`
> beforehand, **or** use the SLURM pipeline (`slurm/submit_all.sh`), which submits
> Step 3 automatically if `leadfield_elect_only.npy` is not yet present.

### Step 1 — LSTM Regression cleaning
```bash
python 1.1_run_eog_lstm_regression_mp.py [--condition {perrecording,persubject,acrosssubject}] \
                                          [--root PATH] [--no-recompute]
```
Trains a 2-layer PyTorch LSTM (eye-tracking → EEG regression) on each subject/run
in parallel. The `--condition` flag controls the training/testing regime (see
[Step 7](#step-7--generalisation-analysis) for details). Default is `perrecording`.

Writes three EDF files per subject/run for `perrecording`:
- `{subject}_{run}_original.edf` — band-passed, resampled original EEG
- `{subject}_{run}_clean.edf` — LSTM-denoised EEG
- `{subject}_{run}_noise.edf` — predicted EOG component

Run all three conditions to enable the generalisation analysis:
```bash
python 1.1_run_eog_lstm_regression_mp.py --condition perrecording
python 1.1_run_eog_lstm_regression_mp.py --condition persubject
python 1.1_run_eog_lstm_regression_mp.py --condition acrosssubject
```

### Step 2 — ICA + ICLabel cleaning
```bash
python 1.2_run_eog_lstm_ica_mp.py [--root PATH] [--no-recompute]
```
Fits extended Infomax ICA, labels components with ICLabel, excludes eye-blink
components. Writes:
- `{subject}_{run}_ica.edf`
- `{subject}_{run}_noiseica.edf`

### Step 3 — Biophysical forward model (run once)
```bash
marimo edit 1.3_eog_gen_model_2025.py
```
Interactive marimo notebook (requires SimNIBS + FreeSurfer `ernie` subject).
Uses `electric_dipole()` simulation on the CHARM-segmented `ernie` head model
to generate a leadfield mapping 6 eye dipoles to 129 EEG electrodes. Outputs:
- `leadfield.npy` — full mesh leadfield `(6, n_nodes)`
- `leadfield_elect_only.npy` — electrode-only leadfield `(6, 129)`

> This step only needs to be run once. The leadfield files can be reused across all
> subjects.

### Step 4 — Biophysical simulation cleaning
```bash
python 1.4_run_eog_lstm_sim_mp.py [--root PATH] [--no-recompute]
```
Applies the precomputed leadfield to each subject's gaze trajectory to generate a
subject-specific EOG estimate. Scales to the data using either a global or
per-channel optimal scaling factor. Writes four EDF files per subject/run:
- `{subject}_{run}_sim.edf` / `_noisesim.edf` — global scaling
- `{subject}_{run}_simlocal.edf` / `_noisesimlocal.edf` — per-channel scaling

> **Note:** `leadfield_elect_only.npy` must be present in the `paper_ijns/`
> directory before running this step.

### Step 5 — Aggregate results into xarray datasets
```bash
marimo edit 2_LSTM_compute_xr.py
```
Loads all processed EDF files (steps 1–4) and computes event-related EEG/ET
signals, SNR metrics, and topographic noise maps. Saves to `.netcdf` files:
- `snr.netcdf`, `topo_erp.netcdf`, `et_signals.netcdf` — stimulus-locked
- `eeg_signals.netcdf`, `topo_raw.netcdf`
- `*_diff.netcdf` variants — saccade-direction-locked

Set the path to the `processed/` directory in the notebook UI before running.

### Step 6 — Analysis and figures
```bash
marimo edit 3_Pytorch_EOG_LSTM_analysis.py
```
Main analysis notebook. Generates all paper figures:
- ERP comparison figures (noise, cleaned signals, gaze, topomaps)
- Topomap comparison of noise percentage across methods
- Gain analysis (global vs. per-channel biophysical scaling)
- SNR box plots and statistical tests

Set the path to the `.netcdf` files in the notebook UI before running.

### Step 7 — Generalisation analysis
```bash
marimo edit 4_generalization_analysis.py
```
Compares LSTM cleaning performance across three training/testing regimes:

| Condition | Training data | Test data |
|-----------|--------------|-----------|
| **Per-recording** | Same recording (train = test) | Same recording |
| **Per-subject** | All other runs of same subject | Held-out run |
| **Across-subject** | All runs from all other subjects | Held-out subject |

**Prerequisites:** Step 1 must have been run for all three conditions (see above).

Produces:
- SNR box plots (pre/post saccade) across conditions
- Per-channel SNR topomaps for each condition
- Paired t-tests with effect sizes between all condition pairs
- Summary table of mean SNR and degradation relative to the per-recording baseline

Set the path to the `processed/` directory in the notebook UI before running.

### Step 8 — Power spectrum analysis
```bash
marimo edit 5_power_spectrum_analysis.py
```
Compares the power spectral density (PSD) across all cleaning approaches. Requires
Steps 1–4 to have been run (all four EDF sets must be present).

Produces:
- Mean PSD curves for original and cleaned signals (all methods overlaid)
- Mean PSD of the removed components (noise EDFs)
- Noise fraction per frequency: `PSD_noise / PSD_original` (%)
- PSD topomaps of noise fraction at δ, θ, α bands for each method
- Frontal vs. posterior PSD comparison
- Band-power summary table (mean ± SD across recordings, frontal channels)

Set the path to the `processed/` directory in the notebook UI before running.

### Step 9 — Functional connectivity analysis (ciPLV)
```bash
marimo edit 6_functional_connectivity_analysis.py
```
Compares **corrected imaginary Phase-Locking Value (ciPLV)** across all cleaning
approaches. ciPLV is insensitive to zero-phase-lag coupling, making it a sensitive
measure of whether methods remove artifactual connectivity (e.g., spurious frontal
synchrony due to shared EOG) without distorting genuine brain connectivity.

**Prerequisites:** Steps 1–4 must have been run. Also requires `mne-connectivity`
(included in `pip install -e ".[paper]"`).

Produces:
- Full channel × channel ciPLV matrices for each approach at δ, θ, α bands
- Difference matrices (original − cleaned) highlighting removed/added connectivity
- 4-ROI (Frontal / Central / Left-Posterior / Right-Posterior) summary heatmaps
- Frontal-connectivity topomaps: mean ciPLV between each channel and frontal channels
- Summary table of mean ciPLV per ROI pair, approach, and band
- Paired t-tests: frontal ciPLV original vs. cleaned per approach and band

Set the path to the `processed/` directory and the epoch length in the notebook UI
before running.

---

## Running on HPC (SLURM)

For full-dataset runs (31 subjects × ~5 runs each) the pipeline is too slow to run
locally. The `slurm/` directory contains everything needed to submit to a SLURM
cluster. All cluster-specific settings live in a single YAML file — no other files
need to be changed.

### Prerequisites

- The `eog-learn` repository is cloned on the cluster
- The main Python environment is set up:
  ```bash
  pip install -e ".[extras,paper]"
  pip install torch mne-icalabel
  ```
- EEGEyeNet data is in `~/mne_data/` (downloaded automatically on first task if not)
- For Step 3 (biophysical forward model): a separate conda environment with SimNIBS ≥ 4
  (see `simnibs_env` below)

### 1 — Edit `slurm/config.yml`

```yaml
partition: AI_Center                        # your SLURM partition
module: "python3/anaconda/2023.9"           # Anaconda module on your cluster
venv: /work/co20/eog_lstm/venv_lstm         # path to main Python environment
simnibs_env: simnibs4                       # conda env for Step 3 (SimNIBS)
root: processed/                            # output directory for EDF files
recompute: true                             # false → skip existing output files
```

This is the **only file you need to edit** before submitting.

### 2 — Submit

```bash
bash paper_ijns/slurm/submit_all.sh
```

The script will:
1. Generate `slurm/subjects.txt` from the dataset
2. Submit Steps 1.1 (× 3 conditions), 1.2, 1.3, and 1.4 concurrently
3. Submit Step 5 with a dependency on all preceding jobs
4. Chain Steps 6–9 sequentially after Step 5

> **Step 3 (biophysical forward model):** If `leadfield_elect_only.npy` already
> exists in `paper_ijns/`, Step 3 is skipped automatically. Otherwise it is
> submitted as a separate SLURM job under the `simnibs_env` environment, and
> Step 4 waits for it to complete.

### 3 — Skip already-computed files

Set `recompute: false` in `config.yml` to skip subject/run pairs whose output
files already exist (useful when restarting a partially-completed run).

### 4 — Monitor

```bash
squeue -u $USER                                   # show all your jobs
squeue -u $USER --format="%.10i %.25j %.8T %E"   # with state and dependency
tail -f paper_ijns/slurm/logs/lstm_<JOB>_<TASK>.out
```

### 5 — Resubmit a single failed subject

Find the subject's index in `slurm/subjects.txt` (0-based), then:

```bash
sbatch --array=<INDEX> \
    --export=ALL,CONDITION=perrecording,ROOT=processed/,RECOMPUTE_FLAG="--no-recompute",\
MODULE=<module>,VENV=<venv>,SCRIPT_DIR=$(pwd)/paper_ijns \
    paper_ijns/slurm/01_lstm.sbatch
```

See [CLUSTER_TESTING.md](CLUSTER_TESTING.md) for a step-by-step test procedure
(smoke test, dependency chain verification, log inspection).

---

## Running Marimo Notebooks

Marimo is a reactive notebook where cells automatically re-run when their inputs
change. To run a notebook interactively:

```bash
marimo edit paper_ijns/2_LSTM_compute_xr.py
```

To run as a script (non-interactive):
```bash
python paper_ijns/2_LSTM_compute_xr.py
```

---

## File Structure

```
paper_ijns/
├── README.md                           # This file
├── CLUSTER_TESTING.md                  # Step-by-step HPC testing guide
├── run_all.py                          # Orchestration script (runs all steps locally)
├── filter.py                           # Shared bandpass filter parameters
├── analyses.py                         # Paper-specific analysis utilities
├── mesh.py                             # SimNIBS mesh helpers (Plotly 3D)
├── plotting.py                         # plot_head() for 3D visualisation
├── 1.1_run_eog_lstm_regression_mp.py   # Step 1: LSTM regression (multiprocessing, 3 conditions)
├── 1.2_run_eog_lstm_ica_mp.py          # Step 2: ICA+ICLabel (multiprocessing)
├── 1.3_eog_gen_model_2025.py           # Step 3: Biophysical model (marimo, interactive)
├── 1.4_run_eog_lstm_sim_mp.py          # Step 4: Biophysical cleaning (multiprocessing)
├── 2_LSTM_compute_xr.py                # Step 5: Aggregate to xarray (marimo)
├── 3_Pytorch_EOG_LSTM_analysis.py      # Step 6: Figures & statistics (marimo)
├── 4_generalization_analysis.py        # Step 7: Generalisation analysis (marimo)
├── 5_power_spectrum_analysis.py        # Step 8: Power spectrum comparison (marimo)
├── 6_functional_connectivity_analysis.py  # Step 9: ciPLV connectivity (marimo)
└── slurm/
    ├── config.yml                      # ← Edit this before submitting to HPC
    ├── submit_all.sh                   # Master SLURM submission script
    ├── 01_lstm.sbatch                  # Array job: LSTM regression (per subject)
    ├── 02_ica.sbatch                   # Array job: ICA + ICLabel
    ├── 03_simnibs.sbatch               # Single job: biophysical forward model
    ├── 04_sim.sbatch                   # Array job: biophysical simulation
    ├── 05_xarray.sbatch                # Single job: xarray aggregation
    └── 06_analysis.sbatch              # Single job: analysis notebooks (Steps 6–9)
```

Reusable utilities (pixel-to-radian conversion, annotation helpers, viz functions,
optimal alpha scaling) have been extracted into the main `eoglearn` package and can
be imported from there:

```python
from eoglearn.io import pixels_to_radians, get_annotations_diff
from eoglearn.models.utils import optimal_alpha
from eoglearn.viz import plot_values_topomap, overlay_raws_stack, plot_dist_dot
```

---

## Data

EEGEyeNet data is downloaded automatically on first use via:

```python
import eoglearn
fpath = eoglearn.datasets.fetch_eegeyenet(subject="EP10", run=1)
```

Data is cached in the MNE data directory (`~/mne_data/` by default).
