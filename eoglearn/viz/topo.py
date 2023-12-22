import warnings

import matplotlib.pyplot as plt
import mne
import numpy as np
from mne.viz.utils import plt_show


def plot_values_topomap(
    value_dict,
    montage,
    *,
    axes=None,
    colorbar=True,
    cmap="RdBu_r",
    vmin=None,
    vmax=None,
    names=None,
    image_interp="bilinear",
    sensors=True,
    show=True,
    **kwargs
):
    """Plot a 2D topographic map of EEG data.

    Parameters
    ----------
    value_dict : dict
        a dict containing EEG sensor names as keys, and a scalar value as values.
        The value is subject to what is to be plotted. (For example, it can be an
        EEG power value for each sensor, or an ICA activation for each sensor).
    montage : mne.channels.DigMontage | str
        Montage for digitized electrode and headshape position data.
        See mne.channels.make_standard_montage(), and
        mne.channels.get_builtin_montages() for more information
        on making montage objects in MNE.
    axes : matplotlib.axes
        The axes object to plot on
    colorbar | bool
        if True, show the corresponding colorbar for z values
    cmap : matplotlib colormap | str | None
        The matplotlib colormap to use. Defaults to 'RdBu_r'
    vmin : float | None
        The minimum value for the colormap. If ``None``, the minimum value is
        set to the minimum value of the heatmap. Default is ``None``.
    vmax : float | None
        The maximum value for the colormap. If ``None``, the maximum value is
        set to the maximum value of the heatmap. Default is ``None``.
    image_interp : str
        The interpolation method to use with matplotlib.imshow. Defaults to
        'bilinear'
    sensors : bool
        Whether to plot black dots on the topoplot, representing the EEG sensor
        positions. Defaults to True.
    show : bool
        Whether to show the plot or not. Defaults to True.
    kwargs : dict
        Valid keyword arguments for mne.viz.plot_topomap

    Returns
    -------
    fig : instance of matplotlib.figure.Figure
      The resulting figure object for the heatmap plot
    """
    if names is None:
        names = [ch for ch in montage.ch_names if ch in value_dict]

    info = mne.create_info(names, sfreq=256, ch_types="eeg")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mne.io.RawArray(
            np.zeros((len(names), 1)), info, copy=None, verbose=False
        ).set_montage(montage)

    if axes:
        ax = axes
        fig = ax.figure
    else:
        fig, ax = plt.subplots(constrained_layout=True)
    im = mne.viz.plot_topomap(
        [value_dict[ch] for ch in names],
        pos=info,
        show=False,
        image_interp=image_interp,
        sensors=sensors,
        res=64,
        axes=ax,
        names=names,
        vlim=[vmin, vmax],
        cmap=cmap,
        **kwargs
    )

    if colorbar:
        fig.colorbar(im[0], ax=axes, shrink=0.6, label="Percentage of EOG in signal")
    plt_show(show, fig)
    return fig
