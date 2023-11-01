from importlib import import_module

import mne


def _check_pymatreader_installed():
    try:
        mod = import_module("pymatreader")
        return mod
    except (ImportError, ModuleNotFoundError):
        raise ImportError(
            "The 'pymatreader' package is required to read the EEGEYENET dataset. "
            "Please install it first via pip or conda."
        )


def read_raw_eegeyenet(fname):
    """Load a sample file from the EEG Eyenet dataset.

    Parameters
    ----------
    fname : str | pathlib.Path
        Path to the EEGEyeNet file to load. EEGEyeNet files can be downloaded with
        :func:`fetch_eegeyenet`.

    Returns
    -------
    mne.io.Raw
        Raw object containing the EEG data.
    """
    pm = _check_pymatreader_installed()

    info = dict(names=[], types=[])

    mat_content = pm.read_mat(fname)
    mat_content = mat_content["sEEG"]
    ch_names = mat_content["chanlocs"]["labels"]
    ch_types = mat_content["chanlocs"]["type"]
    info["sfreq"] = mat_content["srate"]

    # Fix channel names and types to be compliant with MNE
    for ch_name, ch_type in zip(ch_names, ch_types):
        info["names"].append(ch_name)
        if ch_name.startswith("E") or ch_name.upper() == "CZ":
            assert not len(ch_type)  # chanlocs['labels'] empty for eeg in this dataset
            info["types"].append("eeg")
        elif ch_type.upper() == "EYE":
            this_type = (
                "eyegaze"
                if "GAZE" in ch_name
                else "pupil"
                if "AREA" in ch_name
                else "misc"
            )
            info["types"].append(this_type)
        else:
            raise ValueError(f"Unknown channel type: {ch_type} for {ch_name}")
    info = mne.create_info(
        ch_names=info["names"], sfreq=info["sfreq"], ch_types=info["types"]
    )
    assert len(info["ch_names"]) == 133
    raw = mne.io.RawArray(mat_content["data"], info)
    # TODO: remove sanity after development
    assert raw.get_channel_types().count("eeg") == 129
    assert raw.get_channel_types().count("eyegaze") == 2
    assert raw.get_channel_types().count("pupil") == 1
    assert raw.get_channel_types().count("misc") == 1
    # change units to SI. microvolts to volts
    raw.apply_function(lambda x: x * 1e-6, picks="eeg")

    # set channel locations for eyetracking channels
    for ch_dict in raw.info["chs"]:
        if ch_dict["ch_name"].upper().startswith("L"):
            ch_dict["loc"][3] = -1
        elif ch_dict["ch_name"].upper().startswith("R"):
            ch_dict["loc"][3] = 1
        if ch_dict["ch_name"].upper().endswith("X"):
            ch_dict["loc"][4] = -1
        elif ch_dict["ch_name"].upper().endswith("Y"):
            ch_dict["loc"][4] = 1
    return raw
