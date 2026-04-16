import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import eoglearn.io


def plot_dot_fig(ax, triggers=None, **kwargs):
    """Plot EEGEyeNet dot positions on a screen-space axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to plot on.
    triggers : str | list of str | None
        If provided, only plot positions for the specified trigger labels.
        If ``None`` (default), all 27 dot positions are plotted.
    **kwargs
        Additional keyword arguments forwarded to ``ax.scatter``.
    """
    if isinstance(triggers, str):
        triggers = [triggers]
    target_positions = eoglearn.io.get_dot_positions()
    for trigger, position in target_positions.items():
        if triggers is not None and trigger not in triggers:
            continue
        ax.scatter(*position, **kwargs)


def plot_dist_dot(ax, data_xr, triggers=None):
    """Plot gaze distribution KDE overlaid with dot positions.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to plot on.
    data_xr : xarray.DataArray
        DataArray with dimensions ``times``, ``ch_name`` (containing
        ``"eye-x"`` and ``"eye-y"``), ``event_id``, ``subject``, and
        ``run``. Typically the ``"amp"`` variable from the
        ``et_signals.netcdf`` file produced by ``compute_et_xarrays``.
    triggers : str | list of str | None
        If provided, restrict the dot overlay to those trigger labels.
        If ``None`` (default), all dot positions are shown.
    """
    times = data_xr.times[data_xr.times > 0.3]
    tmp_x = data_xr.sel(times=times, ch_name="eye-x").median("times")
    tmp_x = tmp_x.to_dataframe().rename(columns={"amp": "x"})
    tmp_y = data_xr.sel(times=times, ch_name="eye-y").median("times")
    tmp_y = tmp_y.to_dataframe().rename(columns={"amp": "y"})
    tmp = pd.concat([tmp_x, tmp_y[["y"]]], axis=1)
    tmp[["x", "y"]] *= 1e6

    sns.kdeplot(
        data=tmp, x="x", y="y", hue="event_id",
        fill=True, legend=False, ax=ax, thresh=0.1, alpha=0.1,
    )
    sns.kdeplot(
        data=tmp, x="x", y="y", hue="event_id",
        fill=False, legend=False, ax=ax, thresh=0.1,
    )

    target_positions = eoglearn.io.get_dot_positions()
    for event_id in data_xr.event_id.values:
        x_et = (
            data_xr.sel(event_id=event_id, times=times, ch_name="eye-x")
            .mean("times").values.ravel()
        )
        x_et = x_et[~np.isnan(x_et)]
        y_et = (
            data_xr.sel(event_id=event_id, times=times, ch_name="eye-y")
            .mean("times").values.ravel()
        )
        y_et = y_et[~np.isnan(y_et)]

        x, y = target_positions[event_id]
        if event_id == "1":
            ax.text(y + 50, y + 10, event_id, color="k", weight="bold",
                    zorder=110, fontsize=14)
        elif event_id == "19":
            ax.text(x - 10, y + 30, event_id, color="k", weight="bold",
                    zorder=11, fontsize=14)
        else:
            ax.text(x + 10, y - 10, event_id, color="k", weight="bold",
                    zorder=110, fontsize=14)

    ax.invert_yaxis()
    plot_dot_fig(ax, triggers=triggers, s=100, marker="o", color="r", zorder=100)
    plot_dot_fig(ax, triggers=triggers, s=100, marker=".", color="b", zorder=100)

    ax.xaxis.set_tick_params(labelsize=14)
    ax.yaxis.set_tick_params(labelsize=14)
    ax.set_ylim(0, 600)
    ax.set_xlim(0, 800)
    ax.invert_yaxis()
    ax.set_xlabel("Pixels")
    ax.set_ylabel("Pixels")
