import marimo

__generated_with = "0.20.3"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import xarray as xr
    import mne
    import seaborn as sns
    from tqdm.notebook import tqdm
    from scipy import stats
    from mne.stats import permutation_cluster_test
    import eoglearn
    import matplotlib as mpl
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    from eoglearn.viz import (
        plot_values_topomap,
        plot_dot_fig,
        overlay_raws_stack,
        plot_montage_topo,
        plot_dist_dot,
    )
    from analyses import get_insts

    return (
        Path,
        eoglearn,
        get_insts,
        inset_axes,
        mne,
        mpl,
        np,
        overlay_raws_stack,
        pd,
        plot_dist_dot,
        plot_montage_topo,
        plot_values_topomap,
        plt,
        sns,
        stats,
        xr,
    )


@app.cell
def _(mo):
    mo.md("""
    # PyTorch EOG LSTM — Analysis and Figures

    Main analysis notebook comparing LSTM regression / ICA+ICLabel /
    biophysical simulation for EOG artefact removal on the EEGEyeNet dataset.

    **Prerequisites:** `2_LSTM_compute_xr.py` must have been run to produce
    the `.netcdf` xarray files.
    """)
    return


@app.cell
def _(Path):
    xr_path = Path("/Volumes/SSD/eoglearn_results")  # Path to directory containing .netcdf xarray files
    processed_path = Path("/Volumes/SSD/eoglearn_results")  # Path to processed EDF files (for raw signal examples)
    condition = "perrecording"  # LSTM condition. Options: "perrecording", "persubject", "acrosssubject"
    return condition, xr_path


@app.cell
def _(Path, condition, xr, xr_path):
    def load_xarrays(postfix="", path=None):
        if path is None:
            path = Path("./")
        else:
            path = Path(path)

        subsets = {
            "et_signals": "amp",
            "snr": None,
            "topo_erp": "percent",
            "eeg_signals": "amp",
            "topo_raw": "percent",
        }
        xarrays = {}
        for label in subsets:
            xarrays[label] = xr.open_dataset(
                path / f"{label}{postfix}.netcdf", engine="h5netcdf")
            if subsets[label]:
                if label == "eeg_signals":
                    xarrays["eeg_nave"] = xarrays[label]["nave"]
                xarrays[label] = xarrays[label][subsets[label]]

        # Outlier detection on eye-tracking signals (IQR-based threshold = 6)
        err = (
            (xarrays["et_signals"] - xarrays["et_signals"].mean(["subject", "run"])) ** 2
        ).mean("times")
        q1, q2, q3 = err.quantile([0.25, 0.5, 0.75]).values
        outlier_mask = ((err - q2) / (q3 - q1)).mean(["event_id", "ch_name"]) < 6

        for label in list(xarrays.keys()):
            #try:
            xarrays[label] = xarrays[label].where(outlier_mask)
            #except Exception:
            #    pass

        return xarrays

    xarrays = load_xarrays(postfix=f"_{condition}", path=xr_path)
    xarrays_diff = load_xarrays(postfix=f"_{condition}_diff", path=xr_path)

    xarrays_diff_across = load_xarrays(postfix=f"_acrosssubject_diff", path=xr_path)
    return xarrays, xarrays_diff, xarrays_diff_across


@app.cell
def _(mo):
    mo.md("""
    ## Gaze distribution — dot positions
    """)
    return


@app.cell
def _(Path, plot_dist_dot, plt, xarrays):
    Path("images").mkdir(exist_ok=True)
    fig_dots, ax_dots = plt.subplots(1, 1, figsize=(6, 5))
    plot_dist_dot(ax_dots, xarrays["et_signals"])
    fig_dots.savefig("images/dot_positions.png", dpi=300)
    fig_dots
    return


@app.cell
def _(mo):
    mo.md("""
    ## Raw signal overlay — one subject example
    """)
    return


@app.cell
def _():
    subject = "EP10"
    run = "1"
    return run, subject


@app.cell
def _(
    get_insts,
    inset_axes,
    mne,
    overlay_raws_stack,
    plot_montage_topo,
    plt,
    run,
    subject,
    xr_path,
):
    montage = mne.channels.make_standard_montage('GSN-HydroCel-129')
    EOG = ["E17",  "E25",  "E16", "E8"] #"E15", "E21", "E14", "E127", "E126", "E22", "E9",

    fpath = xr_path / f"{subject}_{run}_clean.edf"
    raw = get_insts(fpath)[0]

    tmax = min(raw["original"].times[-1], raw["ica"].times[-1])
    raw["noiseica"] = mne.io.RawArray(raw["original"].get_data(picks="eeg", tmax=tmax) - raw["ica"].get_data(picks="eeg", tmax=tmax), raw["ica"].info)

    fig, axes = plt.subplots(3, 1, figsize=(3.5, 6), height_ratios=[7, 7, 1], sharex=True)

    _axins = inset_axes(axes[0], width="10%", height="10%", loc="upper right")
    plot_montage_topo(_axins, EOG, montage, scale=0.1)

    start=2
    duration=8

    signals = ["original", "noise", "noiseica", "noisesim", "noisesimlocal"]
    labels = ["Original", "ML", "ICA", "Biophysical", "Regression"]
    overlay_raws_stack([raw[kind] for kind in signals], start=start, duration=duration,
                       picks=EOG, labels=labels, annotations=[str(i+1) for i in range(27)],
                       ax=axes[0], linewidth=0.8)


    handles, labels = axes[0].get_legend_handles_labels()
    order = [0, 3, 2, 1, 4]

    reordered_handles = [handles[i] for i in order]
    reordered_labels = [labels[i] for i in order]
    print(reordered_labels)

    axes[0].legend(reordered_handles, reordered_labels,
                   loc="lower center", ncol=3, frameon=False,
                   bbox_to_anchor=(0.5, 1.07), columnspacing=0.2,
                   borderpad=0, labelspacing=0.05, handletextpad=0.1, borderaxespad=0)

    signals = ["original", "clean", "ica", "sim", "simlocal"]
    overlay_raws_stack([raw[kind] for kind in signals], start=start, duration=duration,
                       picks=EOG, annotations=[str(i+1) for i in range(27)],
                       ax=axes[1], show_annot_labels=False, linewidth=0.8)

    et_data, t = raw["original"].copy().get_data(picks=['L-GAZE-X', 'L-GAZE-Y'], tmin=start, 
                                                 tmax=start+duration, return_times=True)
    axes[2].plot(t, et_data[0], linewidth=0.8, label="x")
    axes[2].plot(t, et_data[1], linewidth=0.8, label="y")

    axes[0].set_xlabel("")
    axes[1].set_xlabel("")
    axes[2].set_xlabel("Time (s)")
    axes[2].legend(ncols=2, frameon=False, loc="upper center",
                   bbox_to_anchor=(0.42, 1.), columnspacing=0.2,
                   borderpad=0, labelspacing=0.05, handletextpad=0.1,
                   borderaxespad=0)

    axes[0].set_ylabel("Noise fitting")
    axes[1].set_ylabel("Cleaned signals")
    axes[2].set_ylabel("Eye tracking")
    axes[2].set_yticklabels([])

    fig.tight_layout(pad=0, rect=(0, 0, 1, 0.98))
    fig.align_ylabels()

    fig.savefig("raw_fitting_narrow.png", dpi=300)
    fig
    return montage, raw


@app.cell
def _(mo):
    mo.md("""
    ## ERP comparison figures

    Multi-panel figure: noise lineplot / cleaned lineplot / gaze / topomaps
    comparing Original / ICA / Biophysical / Regression approaches.
    """)
    return


@app.cell
def _(
    inset_axes,
    montage,
    plot_dist_dot,
    plot_montage_topo,
    plot_values_topomap,
    plt,
    sns,
    xarrays_diff,
    xr,
):
    _channels = [
        "E3", "E8", "E9", "E10", "E15",
        "E16", "E18", "E22", "E23", "E25",
    ]
    _hue_order = ["Original", "ICA", "Biophysical", "Regression", "ML"]

    def _remap_kinds(df, kind_map):
        for old, new in kind_map.items():
            df.loc[df.kind == old, "kind"] = new

    def _plot_noise_ax(ax, arrays, event_id, xlim, ylim, n_boot, ylabel):
        _tmp = (
            arrays["eeg_signals"]
            .sel(event_id=event_id, ch_name=_channels)
            .to_dataframe()
            .reset_index()
        )
        _remap_kinds(_tmp, {
            "noise": "ML", "noiseica": "ICA",
            "noisesim": "Biophysical", "original": "Original",
            "noisesimlocal": "Regression",
        })
        sns.lineplot(
            data=_tmp, x="times", hue="kind", y="amp", ax=ax,
            hue_order=_hue_order, n_boot=n_boot, legend=False,
        )
        ax.set_ylabel(ylabel)
        ax.axvline(x=0, linestyle="dashed", color="r")
        ax.set_ylim(*ylim)
        ax.set_xlabel("")
        ax.set_xlim(*xlim)
        plt.setp(ax.get_xticklabels(), visible=False)

    def _plot_clean_ax(ax, arrays, event_id, xlim, ylim, n_boot, ylabel, legend_kw):
        _tmp = (
            arrays["eeg_signals"]
            .sel(event_id=event_id, ch_name=_channels)
            .to_dataframe()
            .reset_index()
        )
        _remap_kinds(_tmp, {
            "clean": "ML", "ica": "ICA",
            "sim": "Biophysical", "original": "Original",
            "simlocal": "Regression",
        })
        _g = sns.lineplot(
            data=_tmp, x="times", hue="kind", y="amp", ax=ax,
            hue_order=_hue_order, n_boot=n_boot,
        )
        _g.legend_.set_title(None)
        _g.legend(title=None, **legend_kw)
        ax.set_ylabel(ylabel)
        ax.axvline(x=0, linestyle="dashed", color="r")
        ax.set_ylim(*ylim)
        ax.set_xlabel("")
        ax.set_xlim(*xlim)
        plt.setp(ax.get_xticklabels(), visible=False)

    def _plot_gaze_ax(ax, arrays, event_id, xlim, n_boot):
        _tmp = arrays["et_signals"].sel(event_id=event_id).to_dataframe()
        _tmp["amp"] *= 1e6
        _g = sns.lineplot(
            data=_tmp, x="times", hue="ch_name", y="amp", ax=ax, n_boot=n_boot,
        )
        _g.legend_.set_title(None)
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Gaze Position (Pixel)")
        ax.axvline(x=0, linestyle="dashed", color="r")
        ax.set_ylim(50, 750)
        ax.set_xlim(*xlim)

    def _plot_topomap_grid(fig, gs2, arrays, event_id, vmax_func, cbar_label):
        for _j, _condition in enumerate(["pre", "post"]):
            _vmax = vmax_func(arrays, _condition, event_id)
            for _i, _kind in enumerate(["clean", "ica", "sim", "simlocal"]):
                _ax = fig.add_subplot(gs2[_i, _j])
                _data_dict = (
                    arrays["topo_erp"]
                    .sel(kind=_kind, condition=_condition, event_id=event_id)
                    .mean(["subject", "run"])
                    .to_dataframe()
                    .drop(columns="kind")["percent"]
                )
                plot_values_topomap(
                    _data_dict, montage, axes=_ax,
                    colorbar=True, cmap="RdBu_r",
                    vmin=-_vmax, vmax=_vmax,
                    image_interp="linear", sensors=True,
                    cbar_label=cbar_label if _j == 1 else "",
                )

    def _add_topo_labels(fig):
        for _x, _y, _txt in [
            (0.47, 0.84, "ML"), (0.47, 0.60, "ICA"),
            (0.47, 0.35, "Biophysical"), (0.47, 0.12, "Regression"),
        ]:
            fig.text(_x, _y, _txt, fontsize=12, va="center", rotation=90)
        fig.text(0.59, 0.95, "Pre-RT", fontsize=12, ha="center")
        fig.text(0.835, 0.95, "Post-RT", fontsize=12, ha="center")

    def plot_ERP_figure(xarrays, event_id, diff=False, n_boot=100):
        fig = plt.figure(constrained_layout=False, figsize=(10, 9))

        # Montage inset
        gs3 = fig.add_gridspec(nrows=1, ncols=1, left=0.0, right=0.2, top=0.22, bottom=0.05)
        plot_montage_topo(fig.add_subplot(gs3[0]), _channels, montage)

        # Dot / gaze distribution inset (stimulus-locked only)
        if not diff:
            gs4 = fig.add_gridspec(
                nrows=1, ncols=1, left=0.24, right=0.45, top=0.22, bottom=0.05
            )
            _ax_dot = fig.add_subplot(gs4[0])
            plot_dist_dot(
                _ax_dot,
                xarrays["et_signals"].sel(event_id=[event_id]),
                triggers=event_id,
            )
            _ax_dot.xaxis.set_tick_params(labelsize=10)
            _ax_dot.yaxis.set_tick_params(labelsize=10)
            _ax_dot.set_xlabel("")
            _ax_dot.set_ylabel("")

        gs1 = fig.add_gridspec(
            nrows=3, ncols=1,
            left=0.06, right=0.45, top=0.92, bottom=0.33,
            wspace=0.05, hspace=0.2,
        )
        _xlim = [xarrays["eeg_signals"].times.min().values, 0.4]

        _ax_noise = fig.add_subplot(gs1[0])
        _ax_clean = fig.add_subplot(gs1[1])
        _ax_gaze = fig.add_subplot(gs1[2], sharex=_ax_clean)

        _plot_noise_ax(_ax_noise, xarrays, event_id, _xlim,
                       ylim=(-16, 31), n_boot=n_boot, ylabel="Amplitude Noise (µV)")
        _plot_clean_ax(_ax_clean, xarrays, event_id, _xlim,
                       ylim=(-16, 31), n_boot=n_boot, ylabel="Amplitude Cleaned (µV)",
                       legend_kw={"ncol": 3, "frameon": False})
        _plot_gaze_ax(_ax_gaze, xarrays, event_id, _xlim, n_boot)

        gs2 = fig.add_gridspec(
            nrows=4, ncols=2,
            left=0.5, right=0.98, bottom=0.0, top=0.95,
            hspace=0.0, wspace=0.1,
        )
        _plot_topomap_grid(
            fig, gs2, xarrays, event_id,
            vmax_func=lambda arrs, cond, eid: float(
                xr.apply_ufunc(abs, arrs["topo_erp"]
                               .sel(condition=cond, event_id=eid)
                               .mean(["subject", "run"])).max()
            ),
            cbar_label="% of EOG",
        )
        _add_topo_labels(fig)
        for _letter, _x, _y in [
            ("A", 0.0, 0.97), ("B", 0.0, 0.75), ("C", 0.0, 0.50),
            ("D", 0.0, 0.24), ("E", 0.2, 0.24), ("F", 0.48, 0.97),
        ]:
            fig.text(_x, _y, _letter, fontsize=14, fontweight="bold")

        _suffix = "_diff" if diff else ""
        fig.savefig(f"images/erp_and_topo_{event_id}{_suffix}.png", dpi=300)
        return fig

    def plot_ERP_figure_4_panels(arrays, event_id, n_boot=100):
        _fig = plt.figure(constrained_layout=False, figsize=(10, 9))
        gs1 = _fig.add_gridspec(
            nrows=3, ncols=1, left=0.06, right=0.45, top=0.96, bottom=0.05,
            wspace=0.05, hspace=0.05,
        )
        _xlim = [arrays["eeg_signals"].times.min().values, 0.3]

        _ax_noise = _fig.add_subplot(gs1[0])
        _ax_clean = _fig.add_subplot(gs1[1])
        _ax_gaze = _fig.add_subplot(gs1[2], sharex=_ax_clean)

        _plot_noise_ax(_ax_noise, arrays, event_id, _xlim,
                       ylim=(-10, 44), n_boot=n_boot, ylabel="Ampltitude Noise (µV)")
        _axins = inset_axes(_ax_noise, width="30%", height="50%", loc="upper left")
        plot_montage_topo(_axins, _channels, montage, scale=0.5)
        _plot_clean_ax(_ax_clean, arrays, event_id, _xlim,
                       ylim=(-16, 44), n_boot=n_boot, ylabel="Ampltitude Cleaned (µV)",
                       legend_kw={"ncol": 1, "frameon": False, "loc": "upper left"})
        _plot_gaze_ax(_ax_gaze, arrays, event_id, _xlim, n_boot)

        gs2 = _fig.add_gridspec(
            nrows=4, ncols=2, left=0.5, right=0.98, bottom=0.0, top=0.95,
            hspace=0.0, wspace=0.1,
        )

        def _vmax_func(arrs, cond, eid):
            _vmax = xr.ufuncs.absolute(
                arrs["topo_erp"].sel(condition=cond, event_id=eid).mean(["subject", "run"])
            ).max().values
            if cond == "post":
                _vmax = 100
            return _vmax

        _plot_topomap_grid(
            _fig, gs2, arrays, event_id,
            vmax_func=_vmax_func,
            cbar_label="% of EOG in signal",
        )
        _add_topo_labels(_fig)
        _fig.text(0.0, 0.97, "A", fontsize=14, fontweight="bold")
        _fig.text(0.0, 0.65, "B", fontsize=14, fontweight="bold")
        _fig.text(0.0, 0.35, "C", fontsize=14, fontweight="bold")
        _fig.text(0.48, 0.97, "D", fontsize=14, fontweight="bold")
        _fig.savefig(f"images/erp_and_topo_{event_id}_paper.png", dpi=300)
        return _fig

    plot_ERP_figure_4_panels(xarrays_diff, "0.3_0.3")
    return plot_ERP_figure, plot_ERP_figure_4_panels


@app.cell
def _(xarrays_diff_across):
    xarrays_diff_across
    return


@app.cell
def _(xarrays_diff_across):
    kind_map = {"clean_acrosssubject": "clean", "noise_acrosssubject": "noise"}
    arr = xarrays_diff_across["eeg_signals"]
    xarrays_diff_across["eeg_signals"] = arr.assign_coords(kind=[kind_map.get(k, k) for k in arr.kind.values])
    return


@app.cell
def _(xarrays_diff_across):
    _kind_map = {"clean_acrosssubject": "clean"}
    _arr = xarrays_diff_across["topo_erp"]
    xarrays_diff_across["topo_erp"] = _arr.assign_coords(kind=[_kind_map.get(k, k) for k in _arr.kind.values])
    return


@app.cell
def _(plot_ERP_figure_4_panels, xarrays_diff_across):
    plot_ERP_figure_4_panels(xarrays_diff_across, "0.3_0.3")
    return


@app.cell
def _(Path, plot_ERP_figure, xarrays):
    Path("images").mkdir(exist_ok=True)
    fig_erp = plot_ERP_figure(xarrays, "15", diff=False)
    fig_erp
    return


@app.cell
def _(plot_ERP_figure, xarrays_diff):
    fig_erp_diff = plot_ERP_figure(xarrays_diff, "0.3_0.3", diff=True)
    fig_erp_diff
    return


@app.cell
def _(mo):
    mo.md("""
    ## Topomap — noise percentage (raw RMS-based)

    Comparison of percent EOG across all four cleaning methods.
    """)
    return


@app.cell
def _(mne, np, plot_values_topomap, plt, xarrays):
    _montage = mne.channels.make_standard_montage("GSN-HydroCel-129")
    _kinds = ["clean", "ica", "sim", "simlocal"]
    _titles = ["ML Regression", "ICA+ICLabel", "Biophysical (global)", "Biophysical (local)"]

    fig_topo, axes_topo = plt.subplots(1, 4, figsize=(16, 4))
    for _ax, _kind, _title in zip(axes_topo, _kinds, _titles):
        _data = (
            xarrays["topo_raw"]
            .sel(kind=_kind)
            .mean(["subject", "run"])
            .to_dataframe()["percent"]
        )
        _vmax = float(np.abs(_data.values).max())
        plot_values_topomap(
            _data, _montage, axes=_ax,
            colorbar=True, cmap="RdBu_r",
            vmin=-_vmax, vmax=_vmax,
            image_interp="linear", sensors=True,
            cbar_label="% noise removed",
        )
        _ax.set_title(_title)

    fig_topo.suptitle("Noise percentage by method (raw RMS)")
    fig_topo.tight_layout()
    fig_topo.savefig("images/topo_noise_comparison.png", dpi=300)
    fig_topo
    return


@app.cell
def _(mo):
    mo.md("""
    ## Gain analysis

    Loads per-subject scaling factors (`gains.csv`) produced by the
    `1.3_eog_gen_model_2025.py` notebook and compares global vs.
    per-channel (individual) gains using permutation cluster tests.
    """)
    return


@app.cell
def _():
    return


@app.cell
def _(
    inset_axes,
    mne,
    montage,
    mpl,
    np,
    pd,
    plot_values_topomap,
    plt,
    raw,
    stats,
):
    from functools import partial

    _montage = mne.channels.make_standard_montage("GSN-HydroCel-129")

    try:
        gains_df = pd.read_csv("gains.csv").set_index("ch_names").drop(index="Cz")

        # Global vs individual gain ratio
        _ratio = gains_df.ind_gains / gains_df.gains
        _ratio_mean = _ratio.groupby(level=0).mean()

        # One-sample t-test (ratio ≠ 0)
        _t_results = _ratio.groupby(level=0).apply(
            partial(stats.ttest_1samp, popmean=0)
        )

        info = raw["original"].copy().set_montage(montage).drop_channels("Cz").info
        adjacency, ch_names = mne.channels.find_ch_adjacency(info, ch_type='eeg')

        x = gains_df.pivot_table(index="ch_names", columns=['run', "subject"], values="gains")
        y = gains_df.pivot_table(index="ch_names", columns=['run', "subject"], values="ind_gains")
        F_obs, clusters, cluster_pv, H0 = mne.stats.permutation_cluster_test([x.loc[ch_names].T,
                                                                              y.loc[ch_names].T],
                                                                             adjacency=adjacency,
                                                                             seed=3245)

        print(cluster_pv, len(clusters[0][0]))


        # Topomap of mean ratio
        fig_gains, _axes = plt.subplots(1, 3, figsize=(6, 2.8), gridspec_kw={'wspace': 0.02})
        fig_gains.subplots_adjust(left=0.0, right=1.0, bottom=0.02, top=0.85) 


        for _ax, _col, _title in zip(
            _axes,
            [None, "ind_gains", "gains"],
            ["Ratio (ind/global)", "Individual gain",  "Global gain"],
        ):
            if _col is not None:
                _data = gains_df[_col].groupby(level=0).mean()
            else:
                _data = _ratio_mean
            _vmax = float(np.abs(_data.values).max())

            if _col is None:
                mask = clusters[0][0]
            else:
                mask = None

            plot_values_topomap(
                _data, _montage, axes=_ax,
                cmap="RdBu_r",
                colorbar=False, 
                vmin=-_vmax, vmax=_vmax, #names=list(_data.index.values),
                image_interp="linear", sensors=True,
                cbar_label="", mask=mask
            )
            _ax.set_title(_title, pad=32)   # extra padding so title clears the colorbar

            # colorbar above this axes
            _norm = mpl.colors.Normalize(vmin=-_vmax, vmax=_vmax)
            _sm = mpl.cm.ScalarMappable(cmap="RdBu_r", norm=_norm)
            _sm.set_array([])
            _cbar_ax = inset_axes(
                _ax, width="85%", height="5%", loc="lower center",
                bbox_to_anchor=(0, 1.02, 1, 1), bbox_transform=_ax.transAxes,
                borderpad=0,
            )
            _cbar_ax.tick_params(labelsize=10)
            cbar = fig_gains.colorbar(_sm, cax=_cbar_ax, orientation="horizontal")
            _cbar_ax.xaxis.set_ticks_position("top")
            cbar.locator = mpl.ticker.MaxNLocator(nbins=1)
            cbar.update_ticks()


        fig_gains.tight_layout()

        fig_gains.text(0.15, 0.01, "a", fontweight='bold', fontsize=14)
        fig_gains.text(0.48, 0.01, "b", fontweight='bold', fontsize=14)
        fig_gains.text(0.83, 0.01, "c", fontweight='bold', fontsize=14)

        fig_gains.savefig("images/gain_ratios.png", dpi=300)

    except FileNotFoundError:
        fig_gains, _ax = plt.subplots()
        _ax.text(0.5, 0.5, "gains.csv not found.\nRun 1.3 notebook first.",
                 ha="center", va="center")

    fig_gains
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Number of trials per recording after binning
    """)
    return


@app.cell
def _(np, plt, sns, xarrays_diff):
    tmp = xarrays_diff["eeg_nave"].sel(subject="EP12", run="2").astype(int).to_dataframe().reset_index()
    tmp["theta"], tmp["phi"] = np.stack(tmp.event_id.str.split("_").values).astype(float).T
    _g = sns.heatmap(tmp.pivot_table(index="theta", columns="phi", values="nave").T.sort_index(ascending=False), annot=True, fmt=".0f", cbar=False)
    plt.tight_layout()
    plt.savefig("images/nb_samples.png", dpi=300)
    _g
    return


@app.cell
def _(mo):
    mo.md("""
    ## SNR analysis

    Pre- vs. post-RT SNR (dB) per approach across all subjects/runs and events.
    """)
    return


@app.cell
def _(mne, plt, sns, xarrays):
    _montage = mne.channels.make_standard_montage("GSN-HydroCel-129")

    # Reshape SNR data for plotting
    try:
        _snr = xarrays["snr"]
        _snr_df = (
            _snr.to_dataframe()
            .reset_index()
            .dropna(subset=["snr"])
        )

        # Rename approaches for display
        _rename = {
            "clean": "ML Regression",
            "ica": "ICA+ICLabel",
            "sim": "Biophysical (global)",
            "simlocal": "Biophysical (local)",
        }
        _snr_df["Approach"] = _snr_df["approach"].map(_rename).fillna(_snr_df["approach"])

        fig_snr, axes_snr = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
        for _ax, _cond in zip(axes_snr, ["pre", "post"]):
            _tmp = _snr_df[_snr_df.condition == _cond]
            sns.boxplot(
                data=_tmp, x="Approach", y="snr", ax=_ax,
                order=list(_rename.values()),
            )
            _ax.set_title(f"{_cond.capitalize()}-RT SNR (dB)")
            _ax.set_xlabel("")
            _ax.tick_params(axis="x", rotation=20)
        axes_snr[0].set_ylabel("SNR (dB)")
        fig_snr.suptitle("Signal-to-noise ratio by method and time window")
        fig_snr.tight_layout()
        fig_snr.savefig("images/snr_comparison.png", dpi=300)
    except Exception as _e:
        fig_snr, _ax = plt.subplots()
        _ax.text(0.5, 0.5, f"SNR plot error:\n{_e}", ha="center", va="center")

    fig_snr
    return


@app.cell
def _(mo):
    mo.md("""
    ## Statistical tests

    Permutation cluster tests comparing approaches across subjects/runs.
    One-sample t-test on SNR gain vs. zero (i.e., vs. no improvement).
    """)
    return


@app.cell
def _(np, pd, stats, xarrays):
    try:
        _snr = xarrays["snr"]
        _kinds = ["clean", "ica", "sim", "simlocal"]
        _stat_rows = []

        for _kind in _kinds:
            for _cond in ["pre", "post"]:
                _vals = (
                    _snr.sel(approach=_kind, condition=_cond)
                    .mean(["event_id", "ch_name"])
                    .to_dataframe()["snr"]
                    .dropna()
                    .values
                )
                if len(_vals) > 2:
                    _t, _p = stats.ttest_1samp(_vals, popmean=0)
                    _stat_rows.append({
                        "approach": _kind,
                        "condition": _cond,
                        "mean_snr": float(np.nanmean(_vals)),
                        "t": float(_t),
                        "p": float(_p),
                        "significant": _p < 0.05,
                    })

        stats_df = pd.DataFrame(_stat_rows)
        stats_df
    except Exception as _e:
        stats_df = pd.DataFrame({"error": [str(_e)]})

    stats_df
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Spectral comparison
    """)
    return


@app.cell
def _(eoglearn, get_insts, np, plt, run, subject, xr_path):
    _fpath = xr_path / f"{subject}_{run}_clean.edf"
    _insts = get_insts(_fpath, diff="True", adjust_for_RT=True)

    for _kind, label in zip(["original", "clean", "ica", "sim", "simlocal"],
                           ["Original", "ML", "ICA", "Biophysical", "Regression"]):
        _raw = _insts[0][_kind]
        sfreq = _raw.info['sfreq']
        win_s = 20.0                     # desired window length in seconds
        n_per_seg = int(win_s * sfreq)  # samples per window
        n_fft = 2**int(np.log2(n_per_seg*4))

        print(n_fft, n_per_seg, int(1.0 * sfreq))

        psd = _raw.compute_psd(picks="eeg",
                              reject_by_annotation=False,
                            method='welch',
                            n_fft=n_fft,                 # length of FFT (>= n_per_seg). choose power of two if you want.
                            n_per_seg=n_per_seg,        # window length in samples (controls the Welch window)
                            n_overlap=int(0.0 * sfreq),  # overlap in samples (here 1 second overlap)
                            fmax=2
        )

        psds, freqs = psd.get_data(picks="eeg", return_freqs=True)
        _y = psds.mean(0) # Average across channels
        _y -= _y.min()
        _y /= _y.max()
        plt.plot(freqs, _y, label=label)



    _fpath = eoglearn.datasets.fetch_eegeyenet(subject="EP10", run="1")
    _raw = eoglearn.io.read_raw_eegeyenet(_fpath)
    _raw.set_montage("GSN-HydroCel-129")

    sfreq = _raw.info['sfreq']
    win_s = 20.0                     # desired window length in seconds
    n_per_seg = int(win_s * sfreq)  # samples per window
    n_fft = 2**int(np.log2(n_per_seg*4))

    print(n_fft, n_per_seg, int(1.0 * sfreq))

    psd = _raw.compute_psd(picks="eeg",
                          reject_by_annotation=False,
                        method='welch',
                        n_fft=n_fft,                 # length of FFT (>= n_per_seg). choose power of two if you want.
                        n_per_seg=n_per_seg,        # window length in samples (controls the Welch window)
                        n_overlap=int(0.0 * sfreq),  # overlap in samples (here 1 second overlap)
                        fmax=2
    )

    psds, freqs = psd.get_data(picks="eeg", return_freqs=True)
    _y = psds.mean(0) # Average across channels
    _y -= _y.min()
    _y /= _y.max()
    plt.plot(freqs, _y, label=label)



    plt.legend()
    plt.ylim(0, 1.02)
    plt.xlim(0, 2)
    plt.xlabel("Frequencies (Hz)")
    plt.ylabel("Normalized spectrum")
    plt.tight_layout()
    plt.savefig("spectrum.png", dpi=300)
    plt.gcf()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Signal decomposition
    """)
    return


@app.cell
def _(mne, np, pd, plot_montage_topo, plt, sns, xarrays_diff):
    from itertools import product


    def plot_signal_decomposition(arrays):
        _fig = plt.figure(constrained_layout=False, figsize=(5, 7))

        nrows = 4
        ncols = 2

        left_border = 0.025
        right_border = 0.03
        bottom_border = 0.01
        _gs1 = _fig.add_gridspec(nrows=nrows, ncols=ncols, left=0.11+left_border, 
                                 right=0.95+right_border, top=0.99, bottom=0.08+bottom_border,
                                 wspace=0.01, hspace=0.01)

        _axes = np.empty((nrows, ncols), dtype="object")
        for _i, _j in product(range(nrows), range(ncols)):
            sharex = _axes[0, 0] if _i or _j else None
            sharey = _axes[_i, 0] if _j else None
            _axes[_i, _j] = _fig.add_subplot(_gs1[_i, _j], sharex=sharex, sharey=sharey)
            if _i < nrows-1:
                _axes[_i, _j].axes.get_xaxis().set_visible(False)
            if _j:
                _axes[_i, _j].axes.get_yaxis().set_visible(False)


        _montage = mne.channels.make_standard_montage('GSN-HydroCel-129')
        diff = True

        channel_lists = [["E3", "E8", "E9", "E10", "E15", "E16", "E18", "E22", "E23", "E25"], 
                         ["E35", "E36", "E37", "E29", "E30", "E41", "E42"],
                         ["E105", "E104", "E103", "E111", "E110", "E87", "E93"],
                         ["E55", "E54", "E79", "E78", "E62", "E61"],
                         #["E75", "E74", "E83", "E82", "E70"]
                        ]
        event_ids = ["0.3_0.3", "0.5_0.0"]  # "-0.5_0.0"]

        for _i, channels in enumerate(channel_lists):
            for _j, event_id in enumerate(event_ids):
                _ax = _axes[_i, _j]
                noisesim = arrays["eeg_signals"].sel(event_id=event_id, ch_name=channels, kind="noisesim")
                original = arrays["eeg_signals"].sel(event_id=event_id, ch_name=channels, kind="original")
                noise_ml = arrays["eeg_signals"].sel(event_id=event_id, ch_name=channels, kind="noise")

                no_correl = original - noise_ml

                df_orig = original.mean("ch_name").mean("run").to_dataframe().reset_index()
                df_orig["kind"] = "Original"

                df_art = noisesim.mean("ch_name").mean("run").to_dataframe().reset_index()
                df_art["kind"] = "EOG"

                df_eeg_mvt = (noise_ml - noisesim).mean("ch_name").mean("run").to_dataframe().reset_index()
                df_eeg_mvt["kind"] = "EEG movement"

                df_eeg_no_mvt = no_correl.mean("ch_name").mean("run").to_dataframe().reset_index()
                df_eeg_no_mvt["kind"] = "EEG no movement"

                tmp_df = pd.concat([df_eeg_mvt, df_eeg_no_mvt, df_art, df_orig])

                if _i == 0 and _j == 1:
                    _g = sns.lineplot(data=tmp_df, hue="kind", x="times", y="amp", ax=_ax)#, errorbar=None)
                    sns.move_legend(
                        _ax, "upper center",
                        bbox_to_anchor=(0.5, 0.95), ncol=1, title=None, frameon=False,
                        columnspacing=1,
                        borderpad=0, labelspacing=0.2, handletextpad=0.1,
                        borderaxespad=0
                    )
                else:
                    _g = sns.lineplot(data=tmp_df, hue="kind", x="times", y="amp", ax=_ax, legend=False)#, errorbar=None)

                _ax.set_xlim(-0.2, 0.5)

                if _i == nrows-1:
                    _ax.set_xlabel("Time (s) \n" + r"($\theta$, $\phi$)=({}, {})".format(*event_id.split("_")))

                if _j == 0:
                    _ax.set_ylabel("Amplitude (uV)")

                if _j == 0:
                    off_y = 0.0
                    off_x = 0.12
                    alpha_y = 0.228
                    size = 0.1
                    _gs3 = _fig.add_gridspec(nrows=1, ncols=1, 
                                           left=off_x+left_border, right=off_x + size, 
                                           top=1 - _i*alpha_y - off_y, 
                                           bottom=1-_i*alpha_y - off_y - size)
                    plot_montage_topo(_fig.add_subplot(_gs3[0]), channels, _montage, scale=0.02)

        _fig.align_ylabels()
        _fig.savefig("decomposition_many.png", dpi=300)
        return _fig

    plot_signal_decomposition(xarrays_diff)
    return


@app.cell
def _(ca):
    ca
    return


if __name__ == "__main__":
    app.run()
