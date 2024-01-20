from importlib import import_module

import numpy as np

import mne
from mne._fiff.constants import FIFF


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

    # read the events (dot onsets, blinks etc)
    eye_chs = [
        name
        for name, ch in zip(raw.ch_names, raw.get_channel_types())
        if ch in ["eyegaze", "pupil"]
    ]
    for onset, duration, description in zip(
        mat_content["event"]["latency"],
        mat_content["event"]["duration"],
        mat_content["event"]["type"],
    ):
        onset /= mat_content["srate"]
        duration /= mat_content["srate"]
        description = description.strip()  # remove trailing whitespace from numbers

        if description in ["55", "56"]:
            continue  # We dont know what these are. probably some start cue
        if description.startswith(("L_", "R_")):
            # remove the L_ or R_ prefix, will put eye info in ch_names key of annot
            eye = [eye_chs]
            description = description.lstrip("LR_")
            if "blink" in description.lower():
                description = "BAD_" + description
        else:  # cue or end_cue
            eye = None
            if description == "41":
                description = "end_cue"
            # cue 1 is the same position as cue 101, so make them the same name
            elif int(description) >= 100:
                description = str(int(description) - 100)
        raw.annotations.append(onset, duration, description, ch_names=eye)
    return raw


def get_dot_positions():
    """Get the x/y pixel coordinates of the dots in the EEGEyeNet dots dataset.

    Returns
    -------
    tar_pos_dict : dict
        dictionary where the keys are the event triggers
        (``'1'``, ``'2'``, ..., ``'27'``), and the values are arrays of shape ``(2,)``,
        indicating the x/y positions of the dot shown on the screen.
    """
    CUE_TRIGGER = list(map(str, range(1, 28)))
    TAR_POS = np.array(
        [
            [400, 300],
            [650, 500],
            [400, 100],
            [100, 450],
            [700, 450],
            [100, 500],
            [200, 350],
            [300, 400],
            [100, 150],
            [150, 500],
            [150, 100],
            [700, 100],
            [300, 200],
            [100, 100],
            [700, 500],
            [500, 400],
            [600, 250],
            [650, 100],
            [400, 300],
            [200, 250],
            [400, 500],
            [700, 150],
            [500, 200],
            [100, 300],
            [700, 300],
            [600, 350],
            [400, 300],
        ]
    )
    return dict(zip(CUE_TRIGGER, TAR_POS))
