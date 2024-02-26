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
import matplotlib.pyplot as plt
from eoglearn.datasets import read_mne_eyetracking_raw
from eoglearn.models import EOGDenoiser

# %%
# Load the data
raw = read_mne_eyetracking_raw()

# %%
raw

# %%
# .. note::
#     If you want eye tracking data in head-referenced-eye-angle (HREF) units, you can
#     pass ``eyetrack_unit="href"`` to
#     :func:`~eoglearn.datasets.read_mne_eyetracking_raw`.

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

# %%
eog_denoiser.fit_model(epochs=10)
history = eog_denoiser.model.history

# %%
# display the training history
print(history.history["loss"])
print(history.history["val_loss"])
eog_denoiser.plot_loss()

# %%
# Plot a topomap of the predicted EOG artifact.
# ---------------------------------------------
# The plot below displays the predicted amount of EOG artifact for each EEG sensor.
# The output is as we would expect, with frontal sensors containing the most EOG
# artifact.

# %%
montage = mne.channels.make_standard_montage("GSN-HydroCel-129")
eog_denoiser.plot_eog_topo(montage=montage)

# %%
# .. todo::
#    Add a plot of the predicted EOG artifact for each EEG sensor over time.
#    Add plots of the denoised EEG data.

# %%
# Compare ERP between the original and "EOG-denoised" signals
# -----------------------------------------------------------
#
# Let's create an averaged evoked response to the flash stimuli for both the original
# data and the "EOG-denoised" data. We'll focus on the frontal EEG channels, since it is
# these will contain the most EOG in the original signal.

# %%
pred_raw = eog_denoiser.get_denoised_neural_raw()
events, event_id = mne.events_from_annotations(pred_raw, regexp="Flash")
pred_epochs = mne.Epochs(
    pred_raw, events=events, event_id=event_id, tmin=-0.3, tmax=3, preload=True
)

events, event_id = mne.events_from_annotations(eog_denoiser.raw, regexp="Flash")
original_epochs = mne.Epochs(
    eog_denoiser.raw, events=events, event_id=event_id, tmin=-0.3, tmax=3, preload=True
)

frontal = ["E19", "E11", "E4", "E12", "E5"]
pred_avg_frontal = pred_epochs.average().get_data(picks=frontal).mean(0)
original_avg_frontal = original_epochs.average().get_data(picks=frontal).mean(0)

ax = plt.subplot()
ax.plot(pred_epochs.times, pred_avg_frontal, label="predicted")
ax.plot(original_epochs.times, original_avg_frontal, label="original")
ax.set_xlim(-0.3, 1)
ax.legend()
