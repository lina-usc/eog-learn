import sys
import os
import time
import errno
import traceback
import multiprocessing
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# Scientific Stack
import numpy as np

# File I/O, Signal Processing
import mne
import eoglearn  # This is my package for this project
import mne_icalabel

from tqdm import tqdm

from filter import filter_kwargs

mne.set_log_level("WARNING")


# ── Timing helpers ────────────────────────────────────────────────────────────

def _init_timing_csv(path: Path) -> None:
    """Atomically create timings.csv with header (safe for concurrent processes)."""
    try:
        fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, b"subject,run,step,condition,duration_s,status\n")
        os.close(fd)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def _log_timing(root: str, subject: str, run, step: str, condition: str,
                duration: float, status: str) -> None:
    """Append one timing row to timings.csv (safe for concurrent O_APPEND writes)."""
    row = f"{subject},{run},{step},{condition},{duration:.2f},{status}\n"
    try:
        with open(Path(root) / "timings.csv", "a") as f:
            f.write(row)
    except OSError:
        pass  # timing failure is non-critical


def process(subject_run, root, tmax=None):
    subject, run = subject_run
    t0 = time.perf_counter()
    status = "failed"
    try:
        print(f"  [{subject} run {run}] Fitting ICA...", flush=True)

        fpath = eoglearn.datasets.fetch_eegeyenet(subject=subject, run=run)
        raw = eoglearn.io.read_raw_eegeyenet(fpath)

        raw.set_montage("GSN-HydroCel-129", verbose=False)
        raw.set_eeg_reference("average", verbose=False)
        raw.set_annotations(None)  # get rid of BAD_blinks annots

        # reload raw and bandpass 1-100 to be fair to ICLabel
        raw.pick("eeg").filter(1, 100, verbose=False)

        tmax = int(raw.times[-1])
        raw.crop(tmax=tmax, include_tmax=False)

        raw_ica = raw.copy()

        ica = mne.preprocessing.ICA(
            method="infomax", fit_params=dict(extended=True), verbose=False)
        ica.fit(raw_ica, verbose=False)

        component_dict = mne_icalabel.label_components(raw_ica, ica, "iclabel")

        exclude_idx = [idx for idx, label in enumerate(component_dict["labels"])
                       if label in ["eye blink"]]

        ica.apply(raw_ica, exclude=exclude_idx, verbose=False)
        tmax = min(raw.times[-1], raw_ica.times[-1])

        raw_icanoise = mne.io.RawArray(
            raw.get_data(picks="eeg", tmax=tmax) -
            raw_ica.get_data(picks="eeg", tmax=tmax),
            raw_ica.info, verbose=False)

        # Apply the filter to match LSTM raw, then export
        raw_icanoise.filter(verbose=False, **filter_kwargs).resample(
            100, verbose=False).export(
            str(Path(root) / f"{subject}_{run}_noiseica.edf"), overwrite=True, verbose=False)
        raw_ica.filter(verbose=False, **filter_kwargs).resample(
            100, verbose=False).export(
            str(Path(root) / f"{subject}_{run}_ica.edf"), overwrite=True, verbose=False)

        print(f"  [{subject} run {run}] Done "
              f"({len(exclude_idx)} eye-blink components removed).", flush=True)
        status = "ok"
        return True

    except Exception:
        traceback.print_exc()
        return False
    finally:
        _log_timing(root, subject, run, "1.2", "",
                    time.perf_counter() - t0, status)


root = "processed/"

# Requires base12 environment. base does not work because of incompatibility of ICALabel

if __name__ == "__main__":
    import argparse
    from functools import partial

    parser = argparse.ArgumentParser(
        description="Run ICA + ICLabel EOG cleaning pipeline."
    )
    parser.add_argument(
        "--root",
        default=root,
        help="Output directory for processed EDF files (default: %(default)s)",
    )
    parser.add_argument(
        "--no-recompute",
        dest="recompute",
        action="store_false",
        default=True,
        help="Skip recordings whose output files already exist",
    )
    parser.add_argument(
        "--subjects",
        nargs="+",
        default=None,
        metavar="SUBJECT",
        help="Restrict processing to these subjects (e.g. EP10 EP11)",
    )
    args = parser.parse_args()

    root = args.root
    recompute = args.recompute

    nb_processes = 5
    Path(root).mkdir(parents=True, exist_ok=True)
    _init_timing_csv(Path(root) / "timings.csv")

    runs_dict = eoglearn.datasets.eegeyenet.get_subjects_runs()
    subjects = args.subjects if args.subjects else list(runs_dict.keys())
    subject_run = np.concatenate([[(subject, run)
                                   for run in runs_dict[subject]]
                                  for subject in subjects
                                  if subject in runs_dict])
    subject_run = [(subject, run)
                   for subject, run in subject_run
                   if recompute or not (Path(root) / f"{subject}_{run}_ica.edf").exists()]

    if not subject_run:
        print("WARNING: Nothing to process — all output files exist. "
              "Pass --recompute to force reprocessing.", flush=True)
    with multiprocessing.Pool(nb_processes) as p:
        results = list(tqdm(p.imap(partial(process, root=root), subject_run),
                            total=len(subject_run), desc="Recordings",
                            position=0, leave=True))
    if not all(results):
        sys.exit(1)
