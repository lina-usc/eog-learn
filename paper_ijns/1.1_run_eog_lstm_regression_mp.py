#!/work/co20/eog_lstm/venv_lstm/bin/python

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

# ML/DL Stack
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F # for easy use of relu


# File I/O, Signal Processing
import mne
import eoglearn  # This is my package for this project

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


class EOGRegressor(nn.Module):
    def __init__(self, n_input_features, n_output_features,
                 hidden_size=64, num_layers=1, dropout=0.5):
        super(EOGRegressor, self).__init__()
        self.input_size = n_input_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)

        self.rnn = nn.LSTM(n_input_features, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, n_output_features)

    def forward(self, input):
        # input shape: (batch_size, seq_len, input_size)
        batch_size = input.size(0)  # same as input.shape[0]

        # Initialize hidden state & cell states
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)

        # Forward propagate RNN
        out, (h0, c0) = self.rnn(input, (h0, c0))

        # Decode the hidden state of the last time step
        out = self.dropout(out)
        out = self.fc(out)

        return out


def train_the_model(X, Y, num_epochs=1000, hidden_size=64, num_layers=1, dropout=0.5):
    """ Train the Pytorch model."""

    # Instantiate the model
    if X.ndim == 3:
        assert Y.ndim == 3
        input_features = X.shape[2]  # Assuming (batch_size, seq_len, input_size)
        output_features = Y.shape[2]
    else:
        raise ValueError("Input data must have 3 dimensions: (batch_size, seq_len, input_size)")

    model = EOGRegressor(input_features, output_features, hidden_size=hidden_size,
                         num_layers=num_layers, dropout=dropout)

    # Loss function (Mean Squared Error)
    criterion = nn.MSELoss()

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Pick a tqdm position based on which pool worker we're in so that
    # concurrent training bars each get their own terminal line.
    identity = multiprocessing.current_process()._identity
    pos = identity[0] if identity else 0  # workers are 1-indexed; main has no identity

    losses = np.zeros(num_epochs)
    # Training loop
    model.train()
    for i in tqdm(range(num_epochs), desc="Training LSTM",
                  position=pos, leave=False):
        # Forward pass
        outputs = model(X)

        # Compute loss
        loss = criterion(outputs, Y)
        losses[i] = loss.detach().numpy()

        # Zero gradients, backward pass, and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Set model to eval mode to turn off dropout
    model.eval()
    return model, losses


def eval_model(model, X, Y):
    with torch.no_grad():
        predicted_noise = model(X)
        denoised_output = (Y - predicted_noise).numpy()

    return predicted_noise, denoised_output


def prep_data(subject="EP10", run=1):
    fpath = eoglearn.datasets.fetch_eegeyenet(subject=subject, run=run)
    raw = eoglearn.io.read_raw_eegeyenet(fpath)

    raw.set_montage("GSN-HydroCel-129", verbose=False)
    raw.filter(picks="eeg", verbose=False, **filter_kwargs).resample(100, verbose=False)
    raw.set_eeg_reference("average", verbose=False)
    return raw


def format_data_for_ml(raw, tmax, scaler_x=None, scaler_y=None):
    # normalize the dataset
    X = raw.get_data(picks=["eyetrack"]).T #[::5] # decimate the eyetracking data

    Y = raw.get_data(picks="eeg").T

    if scaler_x is None:
        scaler_x = StandardScaler().fit(X)
    X = scaler_x.transform(X)

    # For Y we need to split the fit and transform into 2 steps
    # Because we will need to inverse transform the model output later during evaluation
    if scaler_y is None:
        scaler_y = StandardScaler().fit(Y)
    Y = scaler_y.transform(Y)

    # 1s epochs
    X = X.reshape(tmax, int(raw.info["sfreq"]), 3)
    Y = Y.reshape(tmax, int(raw.info["sfreq"]), 129)

    # Convert data to tensors
    X_tensor = torch.from_numpy(X).float()
    Y_tensor = torch.from_numpy(Y).float()

    return X_tensor, Y_tensor, scaler_x, scaler_y


def fit_scalers(raws):
    """Fit StandardScalers on concatenated data from multiple raws."""
    X_all = np.vstack([raw.get_data(picks=["eyetrack"]).T for raw in raws])
    Y_all = np.vstack([raw.get_data(picks="eeg").T for raw in raws])
    return StandardScaler().fit(X_all), StandardScaler().fit(Y_all)


def concat_tensors(raws, scaler_x, scaler_y):
    """Format and concatenate multiple raws into training tensors."""
    X_list, Y_list = [], []
    for raw in raws:
        tmax = int(raw.times[-1])
        raw_crop = raw.copy().crop(tmax=tmax, include_tmax=False)
        X, Y, _, _ = format_data_for_ml(raw_crop, tmax, scaler_x, scaler_y)
        X_list.append(X)
        Y_list.append(Y)
    return torch.cat(X_list, dim=0), torch.cat(Y_list, dim=0)


def clean_data(subject, run, tmax=None):

    raw = prep_data(subject=subject, run=run)

    raw_train = raw.copy()
    if tmax is None:
        tmax = int(raw.times[-1])
    raw_train.crop(tmax=tmax, include_tmax=False)

    X_tensor, Y_tensor, _, scaler_y = format_data_for_ml(raw_train, tmax)
    model, losses = train_the_model(X_tensor, Y_tensor, dropout=.5, num_layers=2)

    tmax = int(raw.times[-1])
    raw.crop(tmax=tmax, include_tmax=False)
    X_tensor, Y_tensor, _, scaler_y = format_data_for_ml(raw, tmax)
    predicted_noise, denoised_output = eval_model(model, X_tensor, Y_tensor)

    # Reshape back to 2D and inverse transform to original units (Volts)
    sfreq = int(raw.info['sfreq'])
    predicted_noise = scaler_y.inverse_transform(
        predicted_noise.reshape(tmax * sfreq, 129)).T
    denoised_output = scaler_y.inverse_transform(
        denoised_output.reshape(tmax * sfreq, 129)).T

    raw_clean = mne.io.RawArray(denoised_output, raw.copy().pick("eeg").info, verbose=False)
    raw_noise = mne.io.RawArray(predicted_noise, raw.copy().pick("eeg").info, verbose=False)
    return raw, raw_clean, raw_noise


def clean_data_per_subject(subject, run):
    """Train on all other runs from the same subject; test on target run.

    Returns None if the subject has only one run (no training data available).
    """
    runs_dict = eoglearn.datasets.eegeyenet.get_subjects_runs()
    all_runs = runs_dict[subject]
    train_runs = [r for r in all_runs if r != run]
    if not train_runs:
        return None

    train_raws = [prep_data(subject=subject, run=r) for r in train_runs]
    test_raw = prep_data(subject=subject, run=run)

    scaler_x, scaler_y = fit_scalers(train_raws)
    X_train, Y_train = concat_tensors(train_raws, scaler_x, scaler_y)
    model, _ = train_the_model(X_train, Y_train, dropout=.5, num_layers=2)

    tmax = int(test_raw.times[-1])
    test_raw.crop(tmax=tmax, include_tmax=False)
    X_test, Y_test, _, _ = format_data_for_ml(test_raw, tmax, scaler_x, scaler_y)
    predicted_noise, denoised_output = eval_model(model, X_test, Y_test)

    sfreq = int(test_raw.info['sfreq'])
    predicted_noise = scaler_y.inverse_transform(
        predicted_noise.reshape(tmax * sfreq, 129)).T
    denoised_output = scaler_y.inverse_transform(
        denoised_output.reshape(tmax * sfreq, 129)).T

    raw_clean = mne.io.RawArray(denoised_output, test_raw.copy().pick("eeg").info, verbose=False)
    raw_noise = mne.io.RawArray(predicted_noise, test_raw.copy().pick("eeg").info, verbose=False)
    return test_raw, raw_clean, raw_noise


def clean_data_across_subjects(subject, run):
    """Train on all runs from all other subjects; test on target subject/run."""
    runs_dict = eoglearn.datasets.eegeyenet.get_subjects_runs()

    train_raws = []
    for subj in runs_dict:
        if subj == subject or "EP" not in subj:
            continue
        for r in runs_dict[subj]:
            train_raws.append(prep_data(subject=subj, run=r))

    test_raw = prep_data(subject=subject, run=run)

    scaler_x, scaler_y = fit_scalers(train_raws)
    X_train, Y_train = concat_tensors(train_raws, scaler_x, scaler_y)
    model, _ = train_the_model(X_train, Y_train, dropout=.5, num_layers=2)

    tmax = int(test_raw.times[-1])
    test_raw.crop(tmax=tmax, include_tmax=False)
    X_test, Y_test, _, _ = format_data_for_ml(test_raw, tmax, scaler_x, scaler_y)
    predicted_noise, denoised_output = eval_model(model, X_test, Y_test)

    sfreq = int(test_raw.info['sfreq'])
    predicted_noise = scaler_y.inverse_transform(
        predicted_noise.reshape(tmax * sfreq, 129)).T
    denoised_output = scaler_y.inverse_transform(
        denoised_output.reshape(tmax * sfreq, 129)).T

    raw_clean = mne.io.RawArray(denoised_output, test_raw.copy().pick("eeg").info, verbose=False)
    raw_noise = mne.io.RawArray(predicted_noise, test_raw.copy().pick("eeg").info, verbose=False)
    return test_raw, raw_clean, raw_noise


def process(subject_run, root, tmax=None):
    subject, run = subject_run
    t0 = time.perf_counter()
    status = "failed"
    try:
        print(f"  [{subject} run {run}] Training LSTM...", flush=True)
        raw, raw_clean, raw_noise = clean_data(subject=subject, run=run, tmax=tmax)
        raw.export(str(Path(root) / f"{subject}_{run}_original.edf"), overwrite=True, verbose=False)
        raw_clean.export(str(Path(root) / f"{subject}_{run}_clean.edf"), overwrite=True, verbose=False)
        raw_noise.export(str(Path(root) / f"{subject}_{run}_noise.edf"), overwrite=True, verbose=False)
        print(f"  [{subject} run {run}] Done.", flush=True)
        status = "ok"
        return True
    except Exception:
        traceback.print_exc()
        return False
    finally:
        _log_timing(root, subject, run, "1.1", "perrecording",
                    time.perf_counter() - t0, status)


def process_persubject(subject_run, root):
    subject, run = subject_run
    t0 = time.perf_counter()
    status = "failed"
    try:
        if "EP" not in subject:
            status = "n/a"
            return True
        print(f"  [{subject} run {run}] Training LSTM (per-subject)...", flush=True)
        result = clean_data_per_subject(subject, run)
        if result is None:
            status = "n/a"
            return True
        raw, raw_clean, raw_noise = result
        raw_clean.export(str(Path(root) / f"{subject}_{run}_clean_persubject.edf"), overwrite=True, verbose=False)
        raw_noise.export(str(Path(root) / f"{subject}_{run}_noise_persubject.edf"), overwrite=True, verbose=False)
        print(f"  [{subject} run {run}] Done.", flush=True)
        status = "ok"
        return True
    except Exception:
        traceback.print_exc()
        return False
    finally:
        _log_timing(root, subject, run, "1.1", "persubject",
                    time.perf_counter() - t0, status)


def process_acrosssubject(subject_run, root):
    subject, run = subject_run
    t0 = time.perf_counter()
    status = "failed"
    try:
        if "EP" not in subject:
            status = "n/a"
            return True
        print(f"  [{subject} run {run}] Training LSTM (across-subject)...", flush=True)
        raw, raw_clean, raw_noise = clean_data_across_subjects(subject, run)
        raw_clean.export(
            str(Path(root) / f"{subject}_{run}_clean_acrosssubject.edf"), overwrite=True, verbose=False)
        raw_noise.export(
            str(Path(root) / f"{subject}_{run}_noise_acrosssubject.edf"), overwrite=True, verbose=False)
        print(f"  [{subject} run {run}] Done.", flush=True)
        status = "ok"
        return True
    except Exception:
        traceback.print_exc()
        return False
    finally:
        _log_timing(root, subject, run, "1.1", "acrosssubject",
                    time.perf_counter() - t0, status)


root = "processed/"
# root = "/Users/christian/Library/CloudStorage/OneDrive-UniversityofSouthCarolina/Data/eog_cleaning_study/processed/"


if __name__ == "__main__":
    import argparse
    from functools import partial

    parser = argparse.ArgumentParser(
        description="Run LSTM EOG regression cleaning pipeline."
    )
    parser.add_argument(
        "--condition",
        choices=["perrecording", "persubject", "acrosssubject"],
        default="perrecording",
        help="Training/testing regime (default: perrecording)",
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

    condition = args.condition
    root = args.root
    recompute = args.recompute

    # Use fewer processes for across-subject (memory-intensive)
    nb_processes = 2 if condition == "acrosssubject" else 5
    Path(root).mkdir(parents=True, exist_ok=True)
    _init_timing_csv(Path(root) / "timings.csv")

    runs_dict = eoglearn.datasets.eegeyenet.get_subjects_runs()
    subjects = args.subjects if args.subjects else list(runs_dict.keys())
    subject_run = np.concatenate([[(subject, run)
                                   for run in runs_dict[subject]]
                                  for subject in subjects
                                  if subject in runs_dict])

    if condition == "perrecording":
        subject_run = [(s, r) for s, r in subject_run
                       if recompute or not (Path(root) / f"{s}_{r}_noise.edf").exists()]
        if not subject_run:
            print("WARNING: Nothing to process — all output files exist. "
                  "Pass --recompute to force reprocessing.", flush=True)
        with multiprocessing.Pool(nb_processes) as p:
            results = list(tqdm(p.imap(partial(process, root=root), subject_run),
                                total=len(subject_run), desc="Recordings",
                                position=0, leave=True))
        if not all(results):
            sys.exit(1)
    elif condition == "persubject":
        subject_run = [(s, r) for s, r in subject_run
                       if recompute or not (
                           Path(root) / f"{s}_{r}_noise_persubject.edf").exists()]
        if not subject_run:
            print("WARNING: Nothing to process — all output files exist. "
                  "Pass --recompute to force reprocessing.", flush=True)
        with multiprocessing.Pool(nb_processes) as p:
            results = list(tqdm(p.imap(partial(process_persubject, root=root), subject_run),
                                total=len(subject_run), desc="Recordings"))
        if not all(results):
            sys.exit(1)
    elif condition == "acrosssubject":
        subject_run = [(s, r) for s, r in subject_run
                       if recompute or not (
                           Path(root) / f"{s}_{r}_noise_acrosssubject.edf").exists()]
        if not subject_run:
            print("WARNING: Nothing to process — all output files exist. "
                  "Pass --recompute to force reprocessing.", flush=True)
        with multiprocessing.Pool(nb_processes) as p:
            results = list(tqdm(p.imap(partial(process_acrosssubject, root=root), subject_run),
                                total=len(subject_run), desc="Recordings"))
        if not all(results):
            sys.exit(1)
