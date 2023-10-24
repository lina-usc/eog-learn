# Author: Scott Huberty <seh33@uw.edu>
#
# License: BSD-3-Clause

from collections import namedtuple

import pytest

from eoglearn.datasets import read_mne_eyetracking_raw

MNEData = namedtuple("MNEData", ["raw", "events_dict"])


@pytest.fixture(scope="session")
def mne_fixture():
    """Return a namedTuple containing MNE eyetracking raw data and events."""
    raw, events_dict = read_mne_eyetracking_raw(return_events=True)
    yield MNEData(raw=raw, events_dict=events_dict)
