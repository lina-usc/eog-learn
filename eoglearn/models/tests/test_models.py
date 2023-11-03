# Author: Scott Huberty <seh33@uw.edu>
#
# License: BSD-3-Clause

from eoglearn.models import EOGDenoiser
import mne


def test_build_model(mne_fixture):
    """Test the build_model function."""
    from mne.io import BaseRaw
    from tensorflow.keras.layers import LSTM

    # set up model
    raw = mne_fixture.raw
    eog_denoiser = EOGDenoiser(raw=raw, downsample=10)
    assert 1 == 0

    # test model attributes
    assert eog_denoiser.downsample == 10
    assert isinstance(eog_denoiser.raw, BaseRaw)
    assert eog_denoiser.X.shape == (14290, 3)
    assert eog_denoiser.Y.shape == (14290, 129)

    # test model architecture
    assert len(eog_denoiser.model.layers) == 2
    assert isinstance(eog_denoiser.model.layers[0], LSTM)
    assert isinstance(eog_denoiser.model.layers[1], LSTM)
    assert eog_denoiser.model.layers[0].get_config()["units"] == 50
    assert eog_denoiser.model.layers[1].get_config()["units"] == 129
    assert eog_denoiser.model.layers[0].input_shape == (None, 100, 3)
    assert eog_denoiser.model.layers[1].input_shape == (None, 100, 50)

    # test model training
    eog_denoiser.fit_model(epochs=3)
    history = eog_denoiser.model.history
    # For now, just check that the loss isn't any higher than what we've seen so far.
    assert history.history["loss"][-1] < 0.05
    assert history.history["val_loss"][-1] < 0.07

    # test viz
    montage = mne.channels.make_standard_montage("GSN-HydroCel-129")
    fig = eog_denoiser.plot_eog_topo(montage=montage, show=False)
    del fig
