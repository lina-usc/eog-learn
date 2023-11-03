"""

Create and fit Model for denoising EOG artifact
===============================================

This example shows how to load create a :class:`~eoglearn.models.model.EOGDenoiser`
instance and fit it using a :class:`~mne.io.BaseRaw` instance with EEG and eyetracking
channels..
"""

# %%
# Import the necessary packages
import mne
from eoglearn.datasets import read_mne_eyetracking_raw
from eoglearn.models import EOGDenoiser

# %%
# Load the data
raw = read_mne_eyetracking_raw()

# %%
raw

# %%
# Plot the data
raw.plot()

# %%
# .. warning::
#     Currently, the :class:`~eoglearn.models.model.EOGDenoiser` expects the
#     :class:`~mne.io.BaseRaw` instance to be bandpass filtered between 1 and 30 Hz.
#
# %%
# Create the model
eog_denoiser = EOGDenoiser(raw=raw, downsample=10)
eog_denoiser

# %%
# Fit the model
# We will only use 10 epochs to speed up the example
eog_denoiser.fit_model(epochs=10)
history = eog_denoiser.model.history

# %%
# display the training history
print(history.history["loss"])
print(history.history["val_loss"])

# %%
# Plot a topomap of the predicted EOG artifact.
# ---------------------------------------------
# The plot below displays the predicted amount of EOG artifact for each EEG sensor.
# The output is as we would expect, with frontal sensors containing the most EOG
# artifact.
montage = mne.channels.make_standard_montage("GSN-HydroCel-129")
eog_denoiser.plot_eog_topo(montage=montage)

# %%
# .. todo::
#    Add a plot of the predicted EOG artifact for each EEG sensor over time.
#    Add plots of the denoised EEG data.
