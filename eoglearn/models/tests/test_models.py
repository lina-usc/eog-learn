# Author: Scott Huberty <seh33@uw.edu>
#
# License: BSD-3-Clause

from eoglearn.models import EOGDenoiser


def test_build_model(mne_fixture):
    """Test the build_model function."""
    from mne.io import BaseRaw
    from tensorflow.keras.layers import LSTM

    # set up model
    raw = mne_fixture.raw
    eog_denoiser = EOGDenoiser(raw=raw, downsample=10)

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
    fitting_kwargs = dict(epochs=1, validation_split=0.2, batch_size=1, verbose=2)
    eog_denoiser.fit_model(fitting_kwargs=fitting_kwargs)
    history = eog_denoiser.model.history
    # For now, just check that the final loss is somewhat reasonable
    history.history["loss"][-1] < 0.05
