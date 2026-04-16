import numpy as np
import matplotlib.pyplot as plt


def overlay_raws_stack(
    raws,
    picks=None,
    start=0.0,
    duration=10,
    resample=True,
    scale_to_uV=False,
    offset_factor=1.2,
    alpha=0.9,
    linewidth=0.8,
    figsize=(12, 8),
    nb_sig_display=None,
    labels=None,
    annotations=None,
    show_annot_labels=True,
    title=None,
    ax=None,
):
    """Overlay multiple MNE Raw objects as a stacked channel plot.

    Each channel is plotted on the same axes with a vertical offset,
    in the style of ``Raw.plot``. The first raw sets the reference
    sampling rate; subsequent raws are resampled to match if needed.

    Parameters
    ----------
    raws : list of mne.io.Raw
        Raw objects to overlay. All must share the requested channels.
    picks : list of str | None
        Channel names to plot. If ``None``, the first 20 common channels
        are used.
    start : float
        Start time in seconds. Defaults to 0.
    duration : float | None
        Duration in seconds. Default to 10.
    resample : bool
        If ``True``, resample raws[1:] to match raws[0] sample rate.
        Defaults to ``True``.
    scale_to_uV : bool
        If ``True``, scale data to µV for display. Defaults to ``False``.
    offset_factor : float
        Spacing between channels as a multiple of median peak-to-peak
        amplitude. Defaults to 1.2.
    alpha : float
        Line transparency. Defaults to 0.9.
    linewidth : float
        Line width. Defaults to 0.8.
    figsize : tuple
        Figure size ``(width, height)`` in inches. Defaults to (12, 8).
    nb_sig_display : int | None
        Maximum number of channels to display. If ``None``, all picked
        channels are shown (up to 20 when ``picks`` is also ``None``).
    labels : list of str | None
        Legend labels for each raw. If ``None``, no legend is shown.
    annotations : list of str | None
        Annotation descriptions to draw as vertical lines.
    show_annot_labels : bool
        Whether to label annotation lines. Defaults to ``True``.
    title : str | None
        Plot title.
    ax : matplotlib.axes.Axes | None
        Axes to plot on. If ``None``, a new figure is created.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """
    # Determine common channels
    if picks is None:
        common = raws[0].ch_names
    else:
        common = picks

    for raw in raws:
        common = [ch for ch in common if ch in raw.ch_names]

    if nb_sig_display is None:
        if picks is None:
            common = common[:20]
    else:
        common = common[:nb_sig_display]

    if len(common) == 0:
        raise ValueError(
            "No common channels found between raws for the requested picks."
        )
    picks = common

    sfreq = raws[0].info["sfreq"]
    if resample:
        for no, raw in enumerate(raws[1:]):
            if sfreq != raw.info["sfreq"]:
                raws[no + 1] = raw.resample(sfreq, npad="auto")

    times = raws[0].copy().crop(tmin=start, tmax=start+duration,
                                include_tmax=False).times + start
    data = np.stack(
        [
            raw.get_data(picks=picks, tmin=start, tmax=start+duration)
            for raw in raws
        ]
    )

    if scale_to_uV:
        data = data * 1e6
        ylab = "Amplitude (µV)"
    else:
        ylab = "Amplitude (native units)"

    ptp = np.ptp(data.reshape(len(raws), len(picks), -1), axis=2)
    ch_ptp = np.max(ptp, axis=0)

    eps = 1e-12
    ch_ptp = np.where(ch_ptp < eps, np.max(ch_ptp) * 0.01 + eps, ch_ptp)
    spacing = np.median(ch_ptp) * offset_factor
    if spacing == 0:
        spacing = np.mean(ch_ptp) * offset_factor + eps

    n = len(picks)
    offsets = np.arange(n, 0, -1) * spacing
    centers = offsets.copy()

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.figure

    for i, ch in enumerate(picks):
        y_off = offsets[i]
        for no, (dat, color) in enumerate(zip(data, colors)):
            ax.plot(
                times,
                dat[i] + y_off,
                color=color,
                linewidth=linewidth,
                alpha=alpha,
                label=labels[no] if labels and i == 0 else None,
            )

    ymin = offsets[-1] - spacing * 0.3
    ymax = offsets[0] + spacing * 0.3
    ax.set_ylim(ymin, ymax + 0.1 * (ymax - ymin))
    ax.set_yticks(centers)
    ax.set_yticklabels(picks)
    ax.set_ylabel(ylab)

    ax.set_xlabel("Time (s)")
    ax.set_xlim(times[0], times[-1])

    ax.set_title(title)

    if annotations:
        display_coords = fig.transFigure.transform((0, 0.885))
        data_coords = ax.transData.inverted().transform(display_coords)

        annot_df = raws[0].annotations.to_data_frame("ms")
        annot_df.onset /= 1000
        annot_df = annot_df[np.in1d(annot_df.description, annotations)]
        annot_df = annot_df[
            (annot_df.onset >= start)
            & (annot_df.onset <= start + duration)
        ]
        for onset, description in zip(annot_df.onset, annot_df.description):
            ax.axvline(x=onset, linestyle="dashed")
            if show_annot_labels:
                ax.text(onset, data_coords[1], description, ha="center")

    if labels:
        ax.legend(loc="upper center", ncol=2, frameon=True)

    ax.grid(True, linestyle=":", linewidth=0.4)
    fig.tight_layout()
    return fig, ax
