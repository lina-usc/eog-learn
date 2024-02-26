from .utils import _fetch_dataset

PARAMS = {
    "EP10_DOTS": list(
        [
            dict(
                url="https://osf.io/download/u2hct/",
                archive_name="EP10_DOTS1_EEG.mat",
                folder_name="EEGEYENET-Data/dots/EP10",
                hash="md5:3e05afdf9c873ac34cb2ffc1141256a3",
                dataset_name="EEGEYENET",
            )
        ]
    ),
}

DOTS = {"EP10": PARAMS["EP10_DOTS"]}


def fetch_eegeyenet(subject="EP10", run=0, fetch_dataset_kwargs=None):
    """Fetch a sample file from the EEG Eyenet dataset.

    Parameters
    ----------
    subject : str
        Subject identifier. Defaults to ``'EP10'``.
    run : int | str
        Which run to download. Most Participants completed 6 runs of the task, saved
        to 6 different files. integers are treated as indices, i.e. ``0`` for subject
        ``"EP10"``corresponds to ``"EP10_DOTS1_EEG.mat"``. If ``str``, it must be the
        exact filename to download for the ``subject``, ie. ``"EP10_DOTS1_EEG.mat"``.
        Defaults to ``0``.
    fetch_dataset_kwargs : dict | None
        Keyword arguments to pass to :func:`~mne.datasets.fetch_dataset`.
        if ``None``, no keyword arguments are passed. Defaults to ``None``.

    Returns
    -------
    pathlib.Path
        Path to the downloaded file.
    """
    if not fetch_dataset_kwargs:
        fetch_dataset_kwargs = dict()
    if isinstance(run, int):
        dataset_params = DOTS[subject][run]
    elif isinstance(run, str):
        run = [run for run in DOTS[subject] if run["archive_name"] == run][0]
        dataset_params = DOTS[subject][run]
    else:
        raise ValueError("run must be an integer or string, not {}".format(type(run)))
    fetch_dataset_kwargs["dataset_params"] = dataset_params
    fpath = _fetch_dataset(fetch_dataset_kwargs=fetch_dataset_kwargs)
    return fpath
