import multiprocessing
from pathlib import Path

# Scientific Stack
import numpy as np

# File I/O, Signal Processing
import mne
import eoglearn  # This is my package for this project
import mne_icalabel


def process(*args, tmax=None):
    try:
        subject, run = args[0]

        # reload raw and bandpass 1-100 to be fair to ICLabel
        fpath = eoglearn.datasets.fetch_eegeyenet(subject=subject, run=run)
        raw_ica = eoglearn.io.read_raw_eegeyenet(fpath)

        raw_ica.set_montage("GSN-HydroCel-129")
        raw_ica.set_eeg_reference("average")
        raw_ica.set_annotations(None) # get rid of BAD_blinks annots
        raw_ica.pick("eeg").filter(1, 100)

        tmax = int(raw_ica.times[-1])
        raw_ica.crop(tmax=tmax, include_tmax=False)

        ica = mne.preprocessing.ICA(method="infomax", fit_params=dict(extended=True))
        ica.fit(raw_ica)
        component_dict = mne_icalabel.label_components(raw_ica, ica, "iclabel")

        exclude_idx = [idx for idx, label in enumerate(component_dict["labels"]) if label in ["eye blink"]]

        # Now apply the ICA to raw, lowpass to 30Hz to match our DL Raw, and plot.
        ica.apply(raw_ica, exclude=exclude_idx)
        raw_ica.filter(1, 30).resample(100)
        raw_ica.export(root + f"{subject}_{run}_ica.edf")
    except:
        pass


root = "processed/"
root = ""

if __name__ == "__main__":

    nb_processes = 5
    Path("processed").mkdir(exist_ok=True)

    runs_dict = eoglearn.datasets.eegeyenet.get_subjects_runs()
    subject_run = np.concatenate([[(subject, run) 
                                   for run in runs_dict[subject]]
                                  for subject in runs_dict])
    subject_run = [(subject, run) 
                   for subject, run in subject_run 
                   if not Path(root + f"{subject}_{run}_ica.edf").exists()]

    p = multiprocessing.Pool(nb_processes)
    p.map(process, subject_run)
