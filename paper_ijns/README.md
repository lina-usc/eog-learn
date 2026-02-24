# paper_ijns — IJNS Paper Analysis Pipeline

This folder contains the analysis code for the IJNS paper comparing three approaches
for removing ocular (EOG) artefacts from simultaneous EEG + eye-tracking recordings:

1. **LSTM Regression** — a PyTorch LSTM trained to predict EOG from eye-tracking signals
2. **ICA + ICLabel** — independent component analysis with automatic eye-blink labelling
3. **Biophysical simulation** — a finite-element forward model (SimNIBS) mapping 3D eye
   dipole moments to the EEG sensor space

All analyses use the [EEGEyeNet](https://osf.io/ktv7m/) DOTS task dataset (39 subjects,
~2 runs each, 129-channel EEG + binocular eye-tracking at 500 Hz).

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
subdirectory. Steps 5–6 aggregate and visualise results.

### Step 1 — LSTM Regression cleaning
```bash
python 1.1_run_eog_lstm_regression_mp.py
```
Trains a 2-layer PyTorch LSTM (eye-tracking → EEG regression) on each subject/run
in parallel. Writes three EDF files per subject/run:
- `{subject}_{run}_original.edf` — band-passed, resampled original EEG
- `{subject}_{run}_clean.edf` — LSTM-denoised EEG
- `{subject}_{run}_noise.edf` — predicted EOG component

### Step 2 — ICA + ICLabel cleaning
```bash
python 1.2_run_eog_lstm_ica_mp.py
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
python 1.4_run_eog_lstm_sim_mp.py
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
├── filter.py                           # Shared bandpass filter parameters
├── analyses.py                         # Paper-specific analysis utilities
├── mesh.py                             # SimNIBS mesh helpers (Plotly 3D)
├── plotting.py                         # plot_head() for 3D visualisation
├── 1.1_run_eog_lstm_regression_mp.py   # Step 1: LSTM regression (multiprocessing)
├── 1.2_run_eog_lstm_ica_mp.py          # Step 2: ICA+ICLabel (multiprocessing)
├── 1.3_eog_gen_model_2025.py           # Step 3: Biophysical model (marimo)
├── 1.4_run_eog_lstm_sim_mp.py          # Step 4: Biophysical cleaning (multiprocessing)
├── 2_LSTM_compute_xr.py                # Step 5: Aggregate to xarray (marimo)
└── 3_Pytorch_EOG_LSTM_analysis.py      # Step 6: Figures & statistics (marimo)
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
