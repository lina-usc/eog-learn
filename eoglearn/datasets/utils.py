import mne


def _fetch_dataset(fetch_dataset_kwargs):
    return mne.datasets.fetch_dataset(**fetch_dataset_kwargs)
