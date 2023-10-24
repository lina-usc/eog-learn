import numpy as np


def test_read_mne_eyetracking_raw(mne_fixture):
    """Test the read_mne_eyetracking_raw function."""
    ch_types = mne_fixture.raw.get_channel_types()
    assert ch_types.count("eyegaze") == 2
    assert ch_types.count("pupil") == 1
    assert ch_types.count("misc") == 3
    assert ch_types.count("eeg") == 129
    events_dict = mne_fixture.events_dict
    np.testing.assert_array_equal(events_dict["eeg"][:, -1], np.repeat(2, 16))
    np.testing.assert_array_equal(events_dict["eyetrack"][:, -1], np.repeat(2, 16))
