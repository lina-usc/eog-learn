import numpy as np
import warnings

import mne
import matplotlib.pyplot as plt


def plot_values_topomap(
    value_dict,
    montage,
    axes,
    colorbar=True,
    cmap="RdBu_r",
    vmin=None,
    vmax=None,
    names=None,
    image_interp="bilinear",
    side_cb="right",
    sensors=True,
    show_names=True,
    **kwargs
):
    """Plot a 2D topographic map of EEG data.

    Parameters
    ---------
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
        if True, show the corresponding colorbar for z value
    side_cb: str
        The side of the plot to display the colorbar. must be one fo "left",
        "right". Defaults to "Right".
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
    show_names : bool
        Whether to plot the EEG sensor names over their corresponding location
        on the topoplot. Defaults to True.
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

    im = mne.viz.plot_topomap(
        [value_dict[ch] for ch in names],
        pos=info,
        show=False,
        image_interp=image_interp,
        sensors=sensors,
        res=64,
        axes=axes,
        names=names,
        vlim=[vmin, vmax],
        cmap=cmap,
        **kwargs
    )

    if colorbar:
        try:
            cbar, cax = mne.viz.topomap._add_colorbar(
                axes, im[0], cmap, pad=0.05, format="%3.2f", side=side_cb
            )
            axes.cbar = cbar
            cbar.ax.tick_params(labelsize=12)

        except TypeError:
            pass

    return im
