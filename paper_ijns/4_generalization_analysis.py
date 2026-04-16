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
    import matplotlib as mpl
    import seaborn as sns
    import mne
    from tqdm.notebook import tqdm
    from scipy import stats

    import eoglearn
    from analyses import (get_epochs, get_evoked, rms,
                          adjust_events_for_RT, get_snrs)
    from eoglearn.io.eegeyenet import get_annotations_diff
    from eoglearn.viz import plot_values_topomap

    return (
        Path,
        adjust_events_for_RT,
        get_evoked,
        get_snrs,
        mne,
        mpl,
        np,
        pd,
        plot_values_topomap,
        plt,
        sns,
        stats,
        tqdm,
    )


@app.cell
def _(mo):
    mo.md("""
    # LSTM Generalisation Analysis

    Compares LSTM EOG-cleaning performance across three training/testing regimes:

    | Condition | Training data | Test data |
    |-----------|--------------|-----------|
    | **Per-recording** | Same recording (train = test) | Same recording |
    | **Per-subject** | All other runs of same subject | Held-out run |
    | **Across-subject** | All runs from all other subjects | Held-out subject |

    **Prerequisites:** run `1.1_run_eog_lstm_regression_mp.py` for each condition
    (`condition = "perrecording"`, `"persubject"`, `"acrosssubject"`) to produce the
    corresponding `*_clean_<condition>.edf` / `*_noise_<condition>.edf` files.
    The per-recording results use the standard `*_clean.edf` / `*_noise.edf` names.
    """)
    return


@app.cell
def _(Path, adjust_events_for_RT, get_evoked, get_snrs, mne, pd, tqdm):
    path_input = "/Volumes/SSD/eoglearn_results"  # "Path to processed data directory"

    # Maps condition label → suffix used in EDF filenames
    CONDITION_SUFFIXES = {
        "per-recording": "clean",
        "per-subject": "clean_persubject",
        "across-subject": "clean_acrosssubject",
    }

    ET_MAPPING = {"L-GAZE-X": ('eyegaze', 'px', 'left', 'x'),
                  "L-GAZE-Y": ("eyegaze", "px", "left", "y"),
                  "L-AREA": ("pupil", "au", "left")}

    def _load_raw(fname):
        raw = mne.io.read_raw_edf(fname, verbose=False)
        if "TIME" in raw.ch_names:
            raw.drop_channels(["TIME"])
        ch_names = [c for c in raw.ch_names if c.startswith("E")]
        raw.set_channel_types(dict(zip(ch_names, ["eeg"] * len(ch_names))))
        return raw

    def load_generalization_data(path):
        """Walk processed directory and compute SNR for all three conditions.

        Returns a tidy DataFrame with columns:
        subject, run, event_id, time_window, approach, ch_name, snr
        """
        files = sorted(Path(path).glob("*_original.edf"))
        dfs = []
        dfs_pct = []    

        for fname_orig in tqdm(files, desc="Loading recordings"):
            stem = fname_orig.name.replace("_original.edf", "")
            parts = stem.split("_")
            if len(parts) < 2:
                continue
            subject, run = parts[0], parts[1]

            # Check all condition files exist
            cond_fnames = {
                cond: fname_orig.parent / f"{stem}_{suffix}.edf"
                for cond, suffix in CONDITION_SUFFIXES.items()
            }
            if not all(f.exists() for f in cond_fnames.values()):
                continue

            try:
                raw_orig = _load_raw(fname_orig)
                mne.preprocessing.eyetracking.set_channel_types_eyetrack(
                    raw_orig, ET_MAPPING)

                annotations = (raw_orig.copy()
                               .drop_channels(
                                   ['L-GAZE-X', 'L-GAZE-Y', 'L-AREA'],
                                   on_missing="ignore")
                               .annotations)
                events, event_id = mne.events_from_annotations(
                    raw_orig.copy().set_annotations(annotations),
                    regexp="^[0-9]*$", verbose=False)
                events = adjust_events_for_RT(events, event_id, raw_orig)

                evoked = {"original": get_evoked(
                    raw_orig, "eeg", events, event_id, event_list=None)}

                for cond, fname in cond_fnames.items():
                    raw_cond = _load_raw(fname)
                    evoked[cond] = get_evoked(
                        raw_cond, "eeg", events, event_id, event_list=None)

                # Remap evoked keys to match get_snrs expectation
                evoked_mapped = {"original": evoked["original"]}
                for cond in CONDITION_SUFFIXES:
                    evoked_mapped[cond] = evoked[cond]

                for ev_id in event_id:
                    snr_df, topo_df = get_snrs(
                        evoked_mapped, ev_id,
                        kinds=tuple(CONDITION_SUFFIXES.keys()))
                    snr_df["subject"] = subject
                    snr_df["run"] = run
                    dfs.append(snr_df)

                    for (cond, kind), pct_df in topo_df.items():
                        pct_df = pct_df.copy()
                        pct_df["approach"] = kind
                        pct_df["condition"] = cond
                        pct_df["subject"] = subject
                        pct_df["run"] = run
                        dfs_pct.append(pct_df)


            except Exception as e:
                print(f"Skipping {stem}: {e}")
                continue

        if not dfs:
            return pd.DataFrame(), pd.DataFrame() 
        return pd.concat(dfs, ignore_index=True), pd.concat(dfs_pct, ignore_index=True)

    gen_cache = Path(path_input) / "gen_df_cache.csv"
    pct_cache = Path(path_input) / "pct_df_cache.csv"
    if gen_cache.exists():
        gen_df = pd.read_csv(gen_cache)
        pct_df = pd.read_csv(pct_cache)
        print(f"Loaded from cache: {gen_cache}, {pct_cache}")
    else:
        gen_df, pct_df = load_generalization_data(path_input)
        gen_df.to_csv(gen_cache, index=False)
        pct_df.to_csv(pct_cache, index=False)
        print(f"Saved cache to {gen_cache} and  {pct_cache}")

    n_rec = gen_df[['subject', 'run']].drop_duplicates().shape[0] if not gen_df.empty else 0
    print(f"Loaded {len(gen_df)} rows from {n_rec} recordings")
    return gen_df, pct_df


@app.cell
def _(mo):
    mo.md("""
    ## SNR comparison across generalisation conditions
    """)
    return


@app.cell
def _(gen_df, pd, plt, sns):
    if gen_df.empty:
        fig_snr, ax_snr = plt.subplots()
        ax_snr.text(0.5, 0.5, "No data loaded", ha="center", va="center")
    else:
        # Average SNR across channels per recording/condition
        snr_avg = (gen_df.drop(columns=["Cz"])
                   .set_index(["subject", "run", "approach", "condition",
                              "event_id"]).mean(axis=1)
                   .reset_index()).rename(columns={0: "snr"})

        CONDITION_ORDER = ["per-recording", "per-subject", "across-subject"]
        snr_avg["approach"] = pd.Categorical(
            snr_avg["approach"], categories=CONDITION_ORDER, ordered=True)

        fig_snr, axes_snr = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
        for _ax, _time_win in zip(axes_snr, ["pre", "post"]):
            _dat = snr_avg[snr_avg["condition"] == _time_win]
            sns.boxplot(data=_dat, x="approach", y="snr", ax=_ax,
                        order=CONDITION_ORDER, palette="Set2")
            _ax.set_title(f"SNR — {_time_win}-saccade window")
            _ax.set_xlabel("Training condition")
            _ax.set_ylabel("SNR (dB)")
            _ax.tick_params(axis="x", rotation=20)
        fig_snr.suptitle(
            "LSTM cleaning performance by generalisation condition", y=1.02)
        fig_snr.tight_layout()
    fig_snr
    return (snr_avg,)


@app.cell
def _(mo):
    mo.md("""
    ## Statistical tests (paired t-tests across conditions)
    """)
    return


@app.cell
def _(gen_df, np, pd, snr_avg, stats):
    if gen_df.empty:
        stat_df = pd.DataFrame()
    else:
        snr_subj = snr_avg.groupby(["subject", "run", "approach", "condition"])["snr"].mean().reset_index()

        rows = []
        pairs = [
            ("per-recording", "per-subject"),
            ("per-recording", "across-subject"),
            ("per-subject", "across-subject"),
        ]
        for _time_win in ["pre", "post"]:
            for cond_a, cond_b in pairs:
                _dat = snr_subj[snr_subj["condition"] == _time_win]
                a = _dat[_dat["approach"] == cond_a]["snr"].values
                b = _dat[_dat["approach"] == cond_b]["snr"].values
                # Align lengths (same recordings should appear in both)
                n = min(len(a), len(b))
                t, p = stats.ttest_rel(a[:n], b[:n])
                d = np.mean(a[:n] - b[:n]) / np.std(a[:n] - b[:n])
                rows.append({
                    "time_window": _time_win,
                    "condition_A": cond_a,
                    "condition_B": cond_b,
                    "t": round(t, 3),
                    "p": round(p, 4),
                    "cohen_d": round(d, 3),
                    "mean_diff_dB": round(np.mean(a[:n] - b[:n]), 3),
                })
        stat_df = pd.DataFrame(rows)

    stat_df
    return


@app.cell
def _(mo):
    mo.md("""
    ## Topomap — noise reduction per condition
    """)
    return


@app.cell
def _(pct_df):
    #long_gen_df = gen_df.melt(id_vars=['approach', 'condition', 'event_id', 'subject', 'run'], 
    #                          var_name="ch_name", value_name="snr")

    long_pct_df = pct_df.melt(
        id_vars=["approach", "condition", "event_id", "subject", "run"],
        var_name="ch_name", value_name="percent")
    return (long_pct_df,)


@app.cell
def _(gen_df, long_pct_df, mne, mpl, plot_values_topomap, plt):
    if gen_df.empty:
        fig_topo = plt.figure()
    else:
        montage = mne.channels.make_standard_montage("GSN-HydroCel-129")

        CONDITION_ORDER_TOPO = ["per-recording", "per-subject", "across-subject"]

        fig_topo, axes_topo = plt.subplots(
            2, len(CONDITION_ORDER_TOPO), figsize=(6, 3.4))
        fig_topo.subplots_adjust(bottom=0, top=0.93, left=0.05, right=0.84, hspace=0.05, wspace=0.05) 

        for rt_cond, ax_row, y_offset in zip(["pre", "post"], axes_topo, [0.47, 0]):
            # Compute per-channel mean percentage noise reduction
            pct_ch = (long_pct_df[long_pct_df["condition"] == rt_cond]
                      .groupby(["approach", "ch_name"])["percent"]
                      .mean()
                      .reset_index())

            vmax = pct_ch["percent"].abs().max()
            vmin = -vmax

            for _ax, _cond in zip(ax_row, CONDITION_ORDER_TOPO):
                _dat = pct_ch[pct_ch["approach"] == _cond]
                ch_names = _dat["ch_name"].tolist()
                values = _dat["percent"].values
                plot_values_topomap(
                    dict(zip(ch_names, values)), montage, axes=_ax,
                    vmin=vmin, vmax=vmax,
                    colorbar=False)
                if rt_cond == "pre":
                    _ax.set_title(_cond.replace("subject", "participant"))

            # Single shared colorbar on the left
            norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
            sm = mpl.cm.ScalarMappable(cmap="RdBu_r", norm=norm)
            sm.set_array([])
            cbar_ax = fig_topo.add_axes([0.85, 0.08+y_offset, 0.02, 0.3])
            cbar = fig_topo.colorbar(sm, cax=cbar_ax, label="Noise removal (%)")
            cbar.ax.yaxis.set_label_position("right")
            cbar.ax.yaxis.tick_right()

        fig_topo.text(0.02, 0.71, "Pre-RT", rotation=90,
                horizontalalignment='left', verticalalignment='center')
        fig_topo.text(0.02, 0.29, "Post-RT", rotation=90,
                horizontalalignment='left', verticalalignment='center')


        #fig_topo.tight_layout()
        #fig_topo.tight_layout(pad=0, rect=(0.07, 0, 1.3, 0.98))
        fig_topo.savefig("generalization.png", dpi=300)
    fig_topo
    return


@app.cell
def _(mo):
    mo.md("""
    ## Summary table — mean SNR (dB) and degradation vs. per-recording
    """)
    return


@app.cell
def _(long_gen_df, np, pd):
    if long_gen_df.empty:
        summary_df = pd.DataFrame()
    else:
        summary = (long_gen_df
                   .groupby(["approach", "condition"])["snr"]
                   .agg(mean=np.mean, std=np.std, median=np.median)
                   .reset_index())

        # Add degradation column vs per-recording baseline
        baseline = summary[summary["approach"] == "per-recording"].set_index(
            "condition")["mean"]
        summary["delta_vs_perrecording"] = summary.apply(
            lambda r: round(r["mean"] - baseline[r["condition"]], 3), axis=1)
        summary["mean"] = summary["mean"].round(3)
        summary["std"] = summary["std"].round(3)
        summary["median"] = summary["median"].round(3)
        summary_df = summary.sort_values(["condition", "approach"])

    summary_df
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
