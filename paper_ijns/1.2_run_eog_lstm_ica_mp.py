import sys
import multiprocessing
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# Scientific Stack
import numpy as np

# File I/O, Signal Processing
import mne
import eoglearn  # This is my package for this project
import mne_icalabel

from filter import filter_kwargs


def process(*args, tmax=None):
    try:
        subject, run = args[0]

        fpath = eoglearn.datasets.fetch_eegeyenet(subject=subject, run=run)
        raw = eoglearn.io.read_raw_eegeyenet(fpath)

        raw.set_montage("GSN-HydroCel-129")
        raw.set_eeg_reference("average")
        raw.set_annotations(None)  # get rid of BAD_blinks annots

        # reload raw and bandpass 1-100 to be fair to ICLabel
        raw.pick("eeg").filter(1, 100)

        tmax = int(raw.times[-1])
        raw.crop(tmax=tmax, include_tmax=False)

        raw_ica = raw.copy()

        print("######################### Fitting ICA ######################")
        ica = mne.preprocessing.ICA(method="infomax", fit_params=dict(extended=True))
        ica.fit(raw_ica)

        print("######################### Labelling components ######################")
        component_dict = mne_icalabel.label_components(raw_ica, ica, "iclabel")

        exclude_idx = [idx for idx, label in enumerate(component_dict["labels"]) if label in ["eye blink"]]

        print("######################### Applying ICA ######################")
        ica.apply(raw_ica, exclude=exclude_idx)
        tmax = min(raw.times[-1], raw_ica.times[-1])

        raw_icanoise = (mne.io.RawArray(raw.get_data(picks="eeg", tmax=tmax) -
                        raw_ica.get_data(picks="eeg", tmax=tmax), raw_ica.info))

        print("######################### Saving ICA ######################")
        print(root + f"{subject}_{run}_noiseica.edf")
        print(root + f"{subject}_{run}_ica.edf")
        # Now apply the ICA to raw, lowpass to 30Hz to match our DL Raw, and plot.
        raw_icanoise.filter(**filter_kwargs).resample(100).export(root + f"{subject}_{run}_noiseica.edf", overwrite=True)
        raw_ica.filter(**filter_kwargs).resample(100).export(root + f"{subject}_{run}_ica.edf", overwrite=True)

        print("######################### Saving ICA DONE ######################")


    except:
        raise


root = "processed/"
#root = "/Users/christian/Library/CloudStorage/OneDrive-UniversityofSouthCarolina/Data/eog_cleaning_study/processed/"

# Requires base12 environment. base does not work because of incompatibility of ICALabel

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
                   if recompute or not Path(root + f"{subject}_{run}_ica.edf").exists()]

    p = multiprocessing.Pool(nb_processes)
    p.map(process, subject_run)
