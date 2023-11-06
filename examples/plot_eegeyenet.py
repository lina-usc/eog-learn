"""

Load an example file from EEGEyeNet
===================================

This example shows how to load an example file from EEGEyeNet, an open-access
dataset of EEG and eyetracking data. We'll load one file from the "dots" task,
which presents a subject with a series of dots at fixed locations on the screen.
"""

# %%
# Import the necessary packages
# ----------------------------
from eoglearn.datasets import fetch_eegeyenet
from eoglearn.io import read_raw_eegeyenet
from eoglearn.models import EOGDenoiser
import mne


fpath = fetch_eegeyenet()
fname = fpath / "EP10_DOTS1_EEG.mat"
raw = read_raw_eegeyenet(fname)
raw

# %%
# Plot the raw data
# -----------------
raw.plot()

# %%
# Create and fit model
# --------------------
eog_denoiser = EOGDenoiser(raw, downsample=5)
eog_denoiser.fit_model(epochs=10)  # limit to 10 epochs for speed

# %%
# Plot A topomap of the EOG activity
# ----------------------------------
montage = mne.channels.make_standard_montage("GSN-HydroCel-129")
eog_denoiser.plot_eog_topo(montage=montage)
