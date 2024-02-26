import pytest

from eoglearn.datasets import fetch_eegeyenet, read_mne_eyetracking_raw


@pytest.mark.parametrize("unit", ["px", "href"])
def test_read_mne_eyetracking_raw(unit):
    """Test the read_mne_eyetracking_raw function."""
    raw, events = read_mne_eyetracking_raw(eyetrack_unit=unit, return_events=True)
    ch_types = raw.get_channel_types()
    assert ch_types.count("eyegaze") == 2
    assert ch_types.count("pupil") == 1
    assert ch_types.count("misc") == 3
    assert ch_types.count("eeg") == 129
    assert len(events["eeg"][:, -1]) == 16
    assert len(events["eyetrack"][:, -1]) == 16


def test_fetch_eegeyenet():
    """Test downloading eegeyenet data."""
    fetch_dataset_kwargs = dict(force_update=True)
    fpath = fetch_eegeyenet(fetch_dataset_kwargs=fetch_dataset_kwargs)
    assert fpath.exists()
    fname = fpath / "EP10_DOTS1_EEG.mat"
    assert fname.exists()
