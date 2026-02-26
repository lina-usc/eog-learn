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

from tqdm import tqdm

from eoglearn.models.utils import optimal_alpha
from analyses import get_sim_eog

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


def process(subject_run, root):
    subject, run = subject_run
    t0 = time.perf_counter()
    status = "failed"
    try:
        if "EP" not in subject:
            status = "n/a"
            return True

        print(f"  [{subject} run {run}] Applying biophysical simulation...", flush=True)
        raw_sim, raw = get_sim_eog(subject, run, return_raw=True)

        # EEGEyeNet has been filtered at 0.5 and 40 Hz. Need to filter
        # simulated data the same way for the EOG to fit the data
        # (the low-pass is not that important but the high-pass has a lot of impact)
        raw_sim.filter(0.5, 40, verbose=False)

        x_sim = raw_sim.copy().get_data(picks="eeg")
        x_raw = raw.copy().get_data(picks="eeg")

        x_sim_global = x_sim * optimal_alpha(x_sim, x_raw)
        x_sim_local = x_sim * optimal_alpha(x_sim, x_raw, axis=1)[:, None]

        raw_sim_noise = mne.io.RawArray(x_sim_global, raw.copy().pick("eeg").info, verbose=False)
        raw_sim = mne.io.RawArray(x_raw - x_sim_global, raw.copy().pick("eeg").info, verbose=False)

        raw_sim_noise.resample(100, verbose=False).export(
            str(Path(root) / f"{subject}_{run}_noisesim.edf"), overwrite=True, verbose=False)
        raw_sim.resample(100, verbose=False).export(
            str(Path(root) / f"{subject}_{run}_sim.edf"), overwrite=True, verbose=False)

        raw_sim_noise_local = mne.io.RawArray(
            x_sim_local, raw.copy().pick("eeg").info, verbose=False)
        raw_sim_local = mne.io.RawArray(
            x_raw - x_sim_local, raw.copy().pick("eeg").info, verbose=False)

        raw_sim_noise_local.resample(100, verbose=False).export(
            str(Path(root) / f"{subject}_{run}_noisesimlocal.edf"), overwrite=True, verbose=False)
        raw_sim_local.resample(100, verbose=False).export(
            str(Path(root) / f"{subject}_{run}_simlocal.edf"), overwrite=True, verbose=False)

        print(f"  [{subject} run {run}] Done.", flush=True)
        status = "ok"
        return True

    except Exception:
        traceback.print_exc()
        return False
    finally:
        _log_timing(root, subject, run, "1.4", "",
                    time.perf_counter() - t0, status)


root = "processed/"

if __name__ == "__main__":
    import argparse
    from functools import partial

    parser = argparse.ArgumentParser(
        description="Run biophysical simulation EOG cleaning pipeline."
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
                   if recompute or not (Path(root) / f"{subject}_{run}_noisesimlocal.edf").exists()]

    if not subject_run:
        print("WARNING: Nothing to process — all output files exist. "
              "Pass --recompute to force reprocessing.", flush=True)
    with multiprocessing.Pool(nb_processes) as p:
        results = list(tqdm(p.imap(partial(process, root=root), subject_run),
                            total=len(subject_run), desc="Recordings",
                            position=0, leave=True))
    if not all(results):
        sys.exit(1)
