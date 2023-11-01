from eoglearn.datasets import fetch_eegeyenet


def test_read_mne_eyetracking_raw(mne_fixture):
    """Test the read_mne_eyetracking_raw function."""
    ch_types = mne_fixture.raw.get_channel_types()
    assert ch_types.count("eyegaze") == 2
    assert ch_types.count("pupil") == 1
    assert ch_types.count("misc") == 3
    assert ch_types.count("eeg") == 129
    events_dict = mne_fixture.events_dict
    assert len(events_dict["eeg"][:, -1]) == 16
    assert len(events_dict["eyetrack"][:, -1]) == 16


def test_fetch_eegeyenet():
    """Test downloading eegeyenet data."""
    fetch_dataset_kwargs = dict(force_update=True)
    fpath = fetch_eegeyenet(fetch_dataset_kwargs=fetch_dataset_kwargs)
    assert fpath.exists()
    fname = fpath / "EP10_DOTS1_EEG.mat"
    assert fname.exists()
