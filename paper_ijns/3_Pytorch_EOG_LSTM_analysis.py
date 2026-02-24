import marimo

__generated_with = "0.10.0"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo
    return (mo,)


@app.cell
def __():
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

    from eoglearn.viz import (
        plot_values_topomap,
        plot_dot_fig,
        overlay_raws_stack,
        plot_montage_topo,
        plot_dist_dot,
    )
    from analyses import get_insts
    return (
        Path, eoglearn, mne, np, overlay_raws_stack, pd, permutation_cluster_test,
        plot_dist_dot, plot_dot_fig, plot_montage_topo, plot_values_topomap,
        plt, sns, stats, sys, tqdm, xr, get_insts,
    )


@app.cell
def __(mo):
    mo.md(
        """
        # PyTorch EOG LSTM — Analysis and Figures

        Main analysis notebook comparing LSTM regression / ICA+ICLabel /
        biophysical simulation for EOG artefact removal on the EEGEyeNet dataset.

        **Prerequisites:** `2_LSTM_compute_xr.py` must have been run to produce
        the `.netcdf` xarray files.
        """
    )
    return ()


@app.cell
def __(mo):
    xr_path_input = mo.ui.text(
        value=".",
        label="Path to directory containing .netcdf xarray files",
        full_width=True,
    )
    processed_path_input = mo.ui.text(
        value="processed/",
        label="Path to processed EDF files (for raw signal examples)",
        full_width=True,
    )
    mo.vstack([xr_path_input, processed_path_input])
    return processed_path_input, xr_path_input


@app.cell
def __(Path, np, xr, xr_path_input):
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
            xarrays[label] = xr.open_dataset(path / f"{label}{postfix}.netcdf")
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
            try:
                xarrays[label] = xarrays[label].where(outlier_mask)
            except Exception:
                pass

        return xarrays

    xarrays = load_xarrays(postfix="", path=xr_path_input.value)
    xarrays_diff = load_xarrays(postfix="_diff", path=xr_path_input.value)
    return load_xarrays, xarrays, xarrays_diff


@app.cell
def __(mo):
    mo.md("## Gaze distribution — dot positions")
    return ()


@app.cell
def __(plot_dist_dot, plt, xarrays):
    fig_dots, ax_dots = plt.subplots(1, 1, figsize=(6, 5))
    plot_dist_dot(ax_dots, xarrays["et_signals"])
    fig_dots.savefig("images/dot_positions.png", dpi=300)
    fig_dots
    return ax_dots, fig_dots


@app.cell
def __(mo):
    mo.md("## Raw signal overlay — one subject example")
    return ()


@app.cell
def __(mo):
    subject_input = mo.ui.text(value="EP10", label="Subject")
    run_input = mo.ui.text(value="1", label="Run")
    mo.hstack([subject_input, run_input])
    return run_input, subject_input


@app.cell
def __(get_insts, mne, overlay_raws_stack, processed_path_input,
        run_input, subject_input, Path):
    _EOG_CH = [
        "E127", "E126", "E17", "E21", "E14",
        "E25", "E22", "E15", "E16", "E9", "E8",
    ]
    _path = Path(processed_path_input.value)
    _fname = _path / f"{subject_input.value}_{run_input.value}_clean.edf"
    if _fname.exists():
        _raw_dict = get_insts(_fname)[0]
        fig_overlay, _ = overlay_raws_stack(
            [_raw_dict["original"], _raw_dict["noisesim"]],
            picks=_EOG_CH,
            start=100, duration=10,
            labels=["Original", "Biophysical noise"],
            annotations=[str(i + 1) for i in range(27)],
            title=f"Subject {subject_input.value} run {run_input.value}",
        )
        fig_overlay.savefig("images/raw_fitting.png", dpi=300)
    else:
        import matplotlib.pyplot as _plt
        fig_overlay, _ = _plt.subplots()
        fig_overlay.text(
            0.5, 0.5, f"File not found:\n{_fname}",
            ha="center", va="center",
        )
    fig_overlay
    return (fig_overlay,)


@app.cell
def __(mo):
    mo.md(
        """
        ## ERP comparison figures

        Multi-panel figure: noise lineplot / cleaned lineplot / gaze / topomaps
        comparing Original / ICA / Biophysical / Regression approaches.
        """
    )
    return ()


@app.cell
def __(mne, plot_dist_dot, plot_montage_topo, plot_values_topomap,
        plt, sns, xr):
    import matplotlib.pyplot as _plt

    _channels = [
        "E3", "E8", "E9", "E10", "E15",
        "E16", "E18", "E22", "E23", "E25",
    ]
    _montage = mne.channels.make_standard_montage("GSN-HydroCel-129")
    _hue_order = ["Original", "ICA", "Biophysical", "Regression", "ML"]

    def plot_ERP_figure(xarrays, event_id, diff=False, n_boot=100):
        fig = _plt.figure(constrained_layout=False, figsize=(10, 9))

        # Montage inset
        gs3 = fig.add_gridspec(nrows=1, ncols=1, left=0.0, right=0.2, top=0.22, bottom=0.05)
        plot_montage_topo(fig.add_subplot(gs3[0]), _channels, _montage)

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

        # Noise lineplot
        _ax_noise = fig.add_subplot(gs1[0])
        _tmp = (
            xarrays["eeg_signals"]
            .sel(event_id=event_id, ch_name=_channels)
            .to_dataframe()
            .reset_index()
        )
        _kind_map = {
            "noise": "ML", "noiseica": "ICA",
            "noisesim": "Biophysical", "original": "Original",
            "noisesimlocal": "Regression",
        }
        for _k, _v in _kind_map.items():
            _tmp.loc[_tmp.kind == _k, "kind"] = _v
        sns.lineplot(
            data=_tmp, x="times", hue="kind", y="amp",
            ax=_ax_noise, hue_order=_hue_order, n_boot=n_boot, legend=False,
        )
        _ax_noise.set_ylabel("Amplitude Noise (µV)")
        _ax_noise.axvline(x=0, linestyle="dashed", color="r")
        _ax_noise.set_ylim(-16, 31)
        _ax_noise.set_xlabel("")
        _ax_noise.set_xlim(*_xlim)
        plt.setp(_ax_noise.get_xticklabels(), visible=False)

        # Cleaned lineplot
        _ax_clean = fig.add_subplot(gs1[1])
        _tmp2 = (
            xarrays["eeg_signals"]
            .sel(event_id=event_id, ch_name=_channels)
            .to_dataframe()
            .reset_index()
        )
        _kind_map2 = {
            "clean": "ML", "ica": "ICA",
            "sim": "Biophysical", "original": "Original",
            "simlocal": "Regression",
        }
        for _k, _v in _kind_map2.items():
            _tmp2.loc[_tmp2.kind == _k, "kind"] = _v
        _g = sns.lineplot(
            data=_tmp2, x="times", hue="kind", y="amp",
            ax=_ax_clean, hue_order=_hue_order, n_boot=n_boot,
        )
        _g.legend_.set_title(None)
        _g.legend(title=None, ncol=3, frameon=False)
        _ax_clean.set_ylabel("Amplitude Cleaned (µV)")
        _ax_clean.axvline(x=0, linestyle="dashed", color="r")
        _ax_clean.set_ylim(-16, 31)
        _ax_clean.set_xlabel("")
        _ax_clean.set_xlim(*_xlim)
        plt.setp(_ax_clean.get_xticklabels(), visible=False)

        # Eye-tracking lineplot
        _ax_gaze = fig.add_subplot(gs1[2], sharex=_ax_clean)
        _tmp3 = xarrays["et_signals"].sel(event_id=event_id).to_dataframe()
        _tmp3["amp"] *= 1e6
        _gg = sns.lineplot(
            data=_tmp3, x="times", hue="ch_name", y="amp",
            ax=_ax_gaze, n_boot=n_boot,
        )
        _gg.legend_.set_title(None)
        _ax_gaze.set_xlabel("Time (seconds)")
        _ax_gaze.set_ylabel("Gaze Position (Pixel)")
        _ax_gaze.axvline(x=0, linestyle="dashed", color="r")
        _ax_gaze.set_ylim(50, 750)
        _ax_gaze.set_xlim(*_xlim)

        # 4×2 topomaps (4 methods × pre/post RT)
        gs2 = fig.add_gridspec(
            nrows=4, ncols=2,
            left=0.5, right=0.98, bottom=0.0, top=0.95,
            hspace=0.0, wspace=0.1,
        )
        for _j, _condition in enumerate(["pre", "post"]):
            _vmax = float(
                xr.apply_ufunc(abs, xarrays["topo_erp"]
                               .sel(condition=_condition, event_id=event_id)
                               .mean(["subject", "run"])).max()
            )
            for _i, _kind in enumerate(["clean", "ica", "sim", "simlocal"]):
                _ax_topo = fig.add_subplot(gs2[_i, _j])
                _data_dict = (
                    xarrays["topo_erp"]
                    .sel(kind=_kind, condition=_condition, event_id=event_id)
                    .mean(["subject", "run"])
                    .to_dataframe()
                    .drop(columns="kind")["percent"]
                )
                plot_values_topomap(
                    _data_dict, _montage, _ax_topo,
                    colorbar=True, cmap="RdBu_r",
                    vmin=-_vmax, vmax=_vmax,
                    image_interp="linear", sensors=True,
                    cbar_label="% of EOG" if _j == 1 else "",
                )

        # Panel labels
        _labels = [
            (0.47, 0.84, "ML"), (0.47, 0.60, "ICA"),
            (0.47, 0.35, "Biophysical"), (0.47, 0.12, "Regression"),
        ]
        for _x, _y, _txt in _labels:
            fig.text(_x, _y, _txt, fontsize=12, va="center", rotation=90)
        fig.text(0.59, 0.95, "Pre-RT", fontsize=12, ha="center")
        fig.text(0.835, 0.95, "Post-RT", fontsize=12, ha="center")
        for _letter, _x, _y in [
            ("A", 0.0, 0.97), ("B", 0.0, 0.75), ("C", 0.0, 0.50),
            ("D", 0.0, 0.24), ("E", 0.2, 0.24), ("F", 0.48, 0.97),
        ]:
            fig.text(_x, _y, _letter, fontsize=14, fontweight="bold")

        _suffix = "_diff" if diff else ""
        fig.savefig(f"images/erp_and_topo_{event_id}{_suffix}.png", dpi=300)
        return fig

    return (plot_ERP_figure,)


@app.cell
def __(mo):
    event_id_input = mo.ui.text(value="15", label="Event ID for ERP figure")
    event_id_input
    return (event_id_input,)


@app.cell
def __(event_id_input, plot_ERP_figure, xarrays):
    import os as _os
    _os.makedirs("images", exist_ok=True)
    fig_erp = plot_ERP_figure(xarrays, event_id_input.value, diff=False)
    fig_erp
    return (fig_erp,)


@app.cell
def __(event_id_input, plot_ERP_figure, xarrays_diff):
    fig_erp_diff = plot_ERP_figure(xarrays_diff, event_id_input.value, diff=True)
    fig_erp_diff
    return (fig_erp_diff,)


@app.cell
def __(mo):
    mo.md(
        """
        ## Topomap — noise percentage (raw RMS-based)

        Comparison of percent EOG across all four cleaning methods.
        """
    )
    return ()


@app.cell
def __(mne, np, plot_values_topomap, plt, xarrays):
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
            _data, _montage, _ax,
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
    return axes_topo, fig_topo


@app.cell
def __(mo):
    mo.md(
        """
        ## Gain analysis

        Loads per-subject scaling factors (`gains.csv`) produced by the
        `1.3_eog_gen_model_2025.py` notebook and compares global vs.
        per-channel (individual) gains using permutation cluster tests.
        """
    )
    return ()


@app.cell
def __(mne, np, pd, permutation_cluster_test, plot_values_topomap,
        plt, stats, xarrays):
    from functools import partial

    _montage = mne.channels.make_standard_montage("GSN-HydroCel-129")

    try:
        gains_df = pd.read_csv("gains.csv").set_index("ch_names")

        # Global vs individual gain ratio
        _ratio = gains_df.ind_gains / gains_df.gains
        _ratio_mean = _ratio.groupby(level=0).mean()

        # One-sample t-test (ratio ≠ 0)
        _t_results = _ratio.groupby(level=0).apply(
            partial(stats.ttest_1samp, popmean=0)
        )

        # Topomap of mean ratio
        fig_gains, _axes = plt.subplots(1, 3, figsize=(15, 4))

        for _ax, _col, _title in zip(
            _axes,
            ["gains", "ind_gains", None],
            ["Global gain", "Individual gain", "Ratio (ind/global)"],
        ):
            if _col is not None:
                _data = gains_df[_col].groupby(level=0).mean()
            else:
                _data = _ratio_mean
            _vmax = float(np.abs(_data.values).max())
            plot_values_topomap(
                _data, _montage, _ax,
                colorbar=True, cmap="RdBu_r",
                vmin=-_vmax, vmax=_vmax,
                image_interp="linear", sensors=True,
                cbar_label=_title,
            )
            _ax.set_title(_title)

        fig_gains.tight_layout()
        fig_gains.savefig("images/gain_ratios.png", dpi=300)
    except FileNotFoundError:
        fig_gains, _ax = plt.subplots()
        _ax.text(0.5, 0.5, "gains.csv not found.\nRun 1.3 notebook first.",
                 ha="center", va="center")

    fig_gains
    return fig_gains, gains_df, partial


@app.cell
def __(mo):
    mo.md(
        """
        ## SNR analysis

        Pre- vs. post-RT SNR (dB) per approach across all subjects/runs and events.
        """
    )
    return ()


@app.cell
def __(mne, np, pd, permutation_cluster_test, plot_values_topomap,
        plt, sns, stats, xarrays):
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
    return axes_snr, fig_snr


@app.cell
def __(mo):
    mo.md(
        """
        ## Statistical tests

        Permutation cluster tests comparing approaches across subjects/runs.
        One-sample t-test on SNR gain vs. zero (i.e., vs. no improvement).
        """
    )
    return ()


@app.cell
def __(np, pd, permutation_cluster_test, stats, xarrays):
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
    return (stats_df,)


if __name__ == "__main__":
    app.run()
