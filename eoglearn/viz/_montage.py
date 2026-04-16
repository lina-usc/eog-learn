import numpy as np
import mne


def plot_montage_topo(ax, picks, montage, scale=1, show=False, show_names=False):
    """Plot a topographic map highlighting selected electrode positions.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to plot on.
    picks : list of str
        Channel names to highlight on the topomap.
    montage : mne.channels.DigMontage
        Montage containing the full electrode layout.
    scale : float
        Scale factor for the marker size. Defaults to 1.
    show : bool
        Whether to call ``plt.show()``. Defaults to ``False``.
    show_names : bool
        Whether to annotate each electrode with its name. Defaults to
        ``False``.
    """
    info = mne.create_info(montage.ch_names, sfreq=256, ch_types="eeg")
    raw = mne.io.RawArray(
        np.zeros((len(montage.ch_names), 1)), info, copy=None, verbose=False
    ).set_montage(montage)
    raw.pick(picks).get_montage().plot(
        axes=ax, show_names=show_names, show=show, scale=scale
    )
