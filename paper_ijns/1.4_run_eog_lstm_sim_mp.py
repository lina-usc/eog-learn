import sys
import multiprocessing
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# Scientific Stack
import numpy as np

# File I/O, Signal Processing
import mne
import eoglearn  # This is my package for this project

from eoglearn.models.utils import optimal_alpha
from analyses import get_sim_eog


def process(*args):
    try:
        subject, run = args[0]
        if not "EP" in subject:
            return

        raw_sim, raw = get_sim_eog(subject, run, return_raw=True)

        # EEGEyeNet has been filtered at 0.5 and 40 Hz. Need to filter
        # simulated data the same way for the EOG to fit the data
        # (the low-pass is not that important but the high-pass has a lot of impact)
        raw_sim.filter(0.5, 40)  # Filter the simulated data as the original data have been filtered

        x_sim = raw_sim.copy().get_data(picks="eeg")
        x_raw = raw.copy().get_data(picks="eeg")

        x_sim_global = x_sim * optimal_alpha(x_sim, x_raw)
        x_sim_local = x_sim * optimal_alpha(x_sim, x_raw, axis=1)[:, None]

        raw_sim_noise = mne.io.RawArray(x_sim_global, raw.copy().pick("eeg").info)
        raw_sim = mne.io.RawArray(x_raw - x_sim_global, raw.copy().pick("eeg").info)

        raw_sim_noise.resample(100).export(root + f"{subject}_{run}_noisesim.edf", overwrite=True)
        raw_sim.resample(100).export(root + f"{subject}_{run}_sim.edf", overwrite=True)

        raw_sim_noise_local = mne.io.RawArray(x_sim_local, raw.copy().pick("eeg").info)
        raw_sim_local = mne.io.RawArray(x_raw - x_sim_local, raw.copy().pick("eeg").info)

        raw_sim_noise_local.resample(100).export(root + f"{subject}_{run}_noisesimlocal.edf", overwrite=True)
        raw_sim_local.resample(100).export(root + f"{subject}_{run}_simlocal.edf", overwrite=True)

    except:
        raise


root = "processed/"
root = "/Users/christian/Library/CloudStorage/OneDrive-UniversityofSouthCarolina/Data/eog_cleaning_study/processed/"

if __name__ == "__main__":

    recompute = True
    nb_processes = 5
    Path("processed").mkdir(exist_ok=True)

    runs_dict = eoglearn.datasets.eegeyenet.get_subjects_runs()
    subject_run = np.concatenate([[(subject, run)
                                   for run in runs_dict[subject]]
                                  for subject in runs_dict])
    subject_run = [(subject, run)
                   for subject, run in subject_run
                   if recompute or not Path(root + f"{subject}_{run}_noisesimlocal.edf").exists()]

    p = multiprocessing.Pool(nb_processes)
    p.map(process, subject_run)
