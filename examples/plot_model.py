"""

Create and fit Model for denoising EOG artifact
===============================================

This example shows how to load create a :class:`~eoglearn.models.model.EOGDenoiser`
instance and fit it using a :class:`~mne.io.BaseRaw` instance with EEG and eyetracking
channels..
"""

# %%
# Import the necessary packages
from eoglearn.datasets import read_mne_eyetracking_raw
from eoglearn.models import EOGDenoiser

# %%
# Load the data
raw = read_mne_eyetracking_raw()
raw.load_data().filter(1, 30)

# %%
raw

# %%
# Plot the data
raw.plot()


# %%
# Create the model
eog_denoiser = EOGDenoiser(raw=raw, downsample=10)
eog_denoiser

# %%
# Fit the model
# We will only use 10 epochs to speed up the example
fitting_kwargs = dict(epochs=10, validation_split=0.2, batch_size=1, verbose=2)
eog_denoiser.fit_model(fitting_kwargs=fitting_kwargs)
history = eog_denoiser.model.history

# %%
# display the training history
history.history
