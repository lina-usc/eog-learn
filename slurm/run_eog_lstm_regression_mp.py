#!/work/co20/eog_lstm/venv_lstm/bin/python

import sys
import multiprocessing

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


class EOGRegressor(nn.Module):
    def __init__(self, n_input_features, n_output_features, hidden_size=64, num_layers=1, dropout=0.5):
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

    model = EOGRegressor(input_features, output_features, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)

    # Loss function (Mean Squared Error)
    criterion = nn.MSELoss()

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    losses = np.zeros(num_epochs)
    # Training loop
    model.train()
    for i, epoch in enumerate(range(num_epochs)):
        # Forward pass
        outputs = model(X)

        # Compute loss
        loss = criterion(outputs, Y)
        losses[i] = loss.detach().numpy()

        # Zero gradients, backward pass, and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print the loss every 100 iterations
        if i % 100 == 0:
            print(f'Epoch: {epoch} Loss: {loss.item():.4f}')

    # Set model to eval mode to turn off dropout
    model.eval()
    with torch.no_grad():
        predicted_noise = model(X)
        denoised_output = (Y - predicted_noise).numpy()

    return losses, predicted_noise, denoised_output


def prep_data(subject="EP10", run=1):
  fpath = eoglearn.datasets.fetch_eegeyenet(subject=subject, run=run)
  raw = eoglearn.io.read_raw_eegeyenet(fpath)

  raw.set_montage("GSN-HydroCel-129")
  raw.filter(1, 30, picks="eeg").resample(100)  # DO NOT filter eyetrack channels
  raw.set_eeg_reference("average")
  return raw


def format_data_for_ml(raw, tmax):
  # normalize the dataset
  X = raw.get_data(picks=["eyetrack"]).T #[::5] # decimate the eyetracking data

  Y = raw.get_data(picks="eeg").T

  scaler = StandardScaler()
  X = scaler.fit_transform(X)
  # For Y we need to split the fit and transform into 2 steps
  # Because we will need to inverse transform the model output later during evaluation
  scaler = StandardScaler()
  scaler_y = scaler.fit(Y)
  Y = scaler_y.transform(Y)

  # 1s epochs
  X = X.reshape(tmax, int(raw.info["sfreq"]), 3)
  Y = Y.reshape(tmax, int(raw.info["sfreq"]), 129)

  # Convert data to tensors
  X_tensor = torch.from_numpy(X).float()
  Y_tensor = torch.from_numpy(Y).float()

  return X_tensor, Y_tensor, scaler_y


def clean_data(subject, run, tmax=None):

  raw = prep_data(subject="EP10", run=1)

  if tmax is None:
    tmax = int(raw.times[-1])
  raw.crop(tmax=tmax, include_tmax=False)

  X_tensor, Y_tensor, scaler_y = format_data_for_ml(raw, tmax)
  losses, predicted_noise, denoised_output = train_the_model(X_tensor, Y_tensor, dropout=.5, num_layers=2)

  # Reshape back to 2D and inverse transform to original units (Volts)
  predicted_noise = scaler_y.inverse_transform(predicted_noise.reshape(tmax*int(raw.info['sfreq']), 129)).T
  denoised_output = scaler_y.inverse_transform(denoised_output.reshape(tmax*int(raw.info['sfreq']), 129)).T

  raw_clean = mne.io.RawArray(denoised_output, raw.copy().pick("eeg").info)
  raw_noise = mne.io.RawArray(predicted_noise, raw.copy().pick("eeg").info)
  return raw, raw_clean, raw_noise


if __name__ == "__main__":

    runs_dict = eoglearn.datasets.eegeyenet.get_subjects_runs()
    subject_run = np.concatenate([[(subject, run) for run in runs_dict[subject]]
                                for subject in runs_dict])

    tmax = None
    def process(subject, run):
      raw, raw_clean, raw_noise = clean_data(subject=subject, run=run, tmax=tmax)
      raw.export(f"{subject}_{run}_original.edf")
      raw_clean.export(f"{subject}_{run}_clean.edf")
      raw_noise.export(f"{subject}_{run}_noise.edf")
      
    p = multiprocessing.Pool(85)
    p.map(process, subject_run)
