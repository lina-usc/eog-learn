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
import matplotlib.pyplot as plt

import eoglearn
import mne

from mne.viz.eyetracking import plot_gaze


fpath = eoglearn.datasets.fetch_eegeyenet()
fname = fpath / "EP10_DOTS1_EEG.mat"
raw = eoglearn.io.read_raw_eegeyenet(fname)
raw

# %%
# Plot the raw data
# -----------------
raw.plot()

# %%
# Create and fit model
# --------------------
eog_denoiser = eoglearn.models.EOGDenoiser(raw, downsample=5)
eog_denoiser.fit_model(epochs=10)  # limit to 10 epochs for speed

# %%
# Plot A topomap of the EOG activity
# ----------------------------------
montage = mne.channels.make_standard_montage("GSN-HydroCel-129")
eog_denoiser.plot_eog_topo(montage=montage)

# %%
# Understanding the task structure
# --------------------------------
#
# In the dots task, a subject is presented with a series of dots at fixed
# locations on the screen. In the data, the dot onsets are marked with
# an integer event trigger.

# %%
target_positions = eoglearn.io.eegeyenet.get_dot_positions()

fig, ax = plt.subplots()
for trigger, position in target_positions.items():
    ax.scatter(*position)
    xy = (
        position - 10
        if trigger == "1"
        else position + 10
        if trigger == "27"
        else position
    )
    ax.text(*xy, trigger)
ax.invert_yaxis()
ax.set_title("Dot positions")
fig.show()

# %%
# Plot a gaze heatmap
# -------------------
# Let's plot a heatmap of the subject's gaze over the course of the task,
# to see if they were looking at the dots.

# %%
mne.preprocessing.eyetracking.interpolate_blinks(raw, interpolate_gaze=True)
# Events from annotations
events, _ = mne.events_from_annotations(raw, regexp="^[0-9]*$")

# Epoch data from events 1-second
epochs = mne.Epochs(raw, events, tmin=0, tmax=1, baseline=None)

# Get data and pick our eyetrack channels
data = epochs.get_data(picks=["L-GAZE-X", "L-GAZE-Y"])

# Plot heatmap
plot_gaze(epochs, width=800, height=600)
