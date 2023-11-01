from eoglearn.datasets import fetch_eegeyenet
from eoglearn.io import read_raw_eegeyenet


def test_read_raw_eegeyenet():
    """Test reading a file from the EEG EyeNet dataset."""
    fpath = fetch_eegeyenet(fetch_dataset_kwargs=dict(force_update=True))
    fname = fpath / "EP10_DOTS1_EEG.mat"
    raw = read_raw_eegeyenet(fname)
    assert raw.info["sfreq"] == 500
    assert len(raw.info["ch_names"]) == 133
    assert raw.get_channel_types().count("eeg") == 129
    assert raw.get_channel_types().count("eyegaze") == 2
    assert raw.get_channel_types().count("pupil") == 1
    assert raw.get_channel_types().count("misc") == 1
    assert raw.info["chs"][-2]["loc"][3] == -1  # left eye
    assert raw.info["chs"][-2]["loc"][4] == 1  # ypos
    assert raw.info["chs"][-3]["loc"][4] == -1  # xpos
