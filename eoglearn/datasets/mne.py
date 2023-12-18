# Author: Scott Huberty <seh33@uw.edu>
#
# License: BSD-3-Clause

import warnings

import mne
from mne.utils import logger


def read_mne_eyetracking_raw(return_events=False, bandpass=True):
    """Return an MNE Raw object containing the EyeLink dataset.

    Parameters
    ----------
    return_events : bool
        If ``True``, return the events for the eyetracking and EEG data.
    bandpass: bool
        If ``True``, applied a [1, 30]Hz bandpass.

    Returns
    -------
    raw : mne.io.Raw
        A MNE Raw object containing the EyeLink dataset.
    events : dict
        A dictionary where the values for "eyetrack" and "EEG" keys are the
        events arrays for the eyetracking and EEG data, respectively.

    Notes
    -----
    See MNE-Python's
    `tutorial <https://mne.tools/dev/auto_tutorials/preprocessing/\
        90_eyetracking_data.html>`_
    for more information on this dataset.
    """
    data_path = mne.datasets.eyelink.data_path()
    if mne.utils.check_version("mne", "1.6"):
        data_path = data_path / "eeg-et"
    et_fpath = data_path / "sub-01_task-plr_eyetrack.asc"
    eeg_fpath = data_path / "sub-01_task-plr_eeg.mff"

    logger.debug(f"## EOGLEARN: Reading data from {et_fpath} and {eeg_fpath}")
    raw_et = mne.io.read_raw_eyelink(et_fpath, create_annotations=["blinks"])
    raw_eeg = mne.io.read_raw_egi(eeg_fpath, preload=True)
    if bandpass:
        raw_eeg.filter(1, 30)

    logger.debug("## EOGLEARN: Finding events from the raw objects")
    # due to a rogue one-shot event, find_events emits a warning
    # that we can safely ignore here
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="invalid value encountered")
        et_events = mne.find_events(
            raw_et, min_duration=0.01, shortest_event=1, uint_cast=True
        )
    eeg_events = mne.find_events(raw_eeg, stim_channel="DIN3")

    # Convert event onsets from samples to seconds
    logger.debug("## EOGLEARN: Converting event onsets from samples to seconds")
    et_din_times = et_events[:, 0] / raw_et.info["sfreq"]
    eeg_din_times = eeg_events[:, 0] / raw_eeg.info["sfreq"]

    # Align the data
    logger.debug("## EOGLEARN: Aligning the EEG and Eyetracking data.")
    mne.preprocessing.realign_raw(
        raw_et, raw_eeg, et_din_times, eeg_din_times, verbose="error"
    )
    # Add EEG channels to the eye-tracking raw object
    raw_et.add_channels([raw_eeg], force_update_info=True)

    annots = mne.annotations_from_events(eeg_events, raw_et.info["sfreq"],
                                         event_desc={2: "Flash"},
                                         orig_time=raw_et.info["meas_date"])
    raw_et.set_annotations(raw_et.annotations + annots)

    if return_events:
        return raw_et, dict(eyetrack=et_events, eeg=eeg_events)
    return raw_et
