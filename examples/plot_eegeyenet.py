"""

Load an example file from EEGEyeNet
===================================

This example shows how to load an example file from EEGEyeNet, an open-access
dataset of EEG and eyetracking data. We'll load one file from the "dots" task,
which presents a subject with a series of dots at fixed locations on the screen.
"""

# %%
# Import the necessary packages
from eoglearn.datasets import fetch_eeyeenet
from eoglearn.io import read_raw_eegeyenet

fpath = fetch_eeyeenet()
fname = fpath / "EP10_DOTS1_EEG.mat"
raw = read_raw_eegeyenet(fname)
raw.plot()
