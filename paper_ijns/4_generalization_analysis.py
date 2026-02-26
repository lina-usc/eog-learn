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
        Path, adjust_events_for_RT, eoglearn, get_annotations_diff,
        get_epochs, get_evoked, get_snrs, mne, np, pd, plt,
        plot_values_topomap, rms, sns, stats, sys, tqdm,
    )


@app.cell
def __(mo):
    mo.md(
        """
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
        """
    )
    return ()


@app.cell
def __(mo):
    import os as _os
    path_input = mo.ui.text(
        value=_os.environ.get("EOG_PROCESSED_PATH", "processed"),
        label="Path to processed data directory",
        full_width=True,
    )
    path_input
    return (path_input,)


@app.cell
def __(Path, adjust_events_for_RT, get_annotations_diff, get_epochs,
       get_evoked, get_snrs, mne, np, pd, path_input, tqdm):
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
                    snr_df, _ = get_snrs(
                        evoked_mapped, ev_id,
                        kinds=tuple(CONDITION_SUFFIXES.keys()))
                    snr_df["subject"] = subject
                    snr_df["run"] = run
                    dfs.append(snr_df)

            except Exception as e:
                print(f"Skipping {stem}: {e}")
                continue

        if not dfs:
            return pd.DataFrame()
        return pd.concat(dfs, ignore_index=True)

    gen_df = load_generalization_data(path_input.value)
    n_rec = gen_df[['subject', 'run']].drop_duplicates().shape[0] if not gen_df.empty else 0
    print(f"Loaded {len(gen_df)} rows from {n_rec} recordings")
    return (CONDITION_SUFFIXES, ET_MAPPING, gen_df, load_generalization_data)


@app.cell
def __(mo):
    mo.md("## SNR comparison across generalisation conditions")
    return ()


@app.cell
def __(gen_df, pd, plt, sns):
    if gen_df.empty:
        fig_snr, ax_snr = plt.subplots()
        ax_snr.text(0.5, 0.5, "No data loaded", ha="center", va="center")
    else:
        # Average SNR across channels per recording/condition
        snr_avg = (gen_df
                   .groupby(["subject", "run", "approach", "condition",
                              "event_id"])["snr"]
                   .mean()
                   .reset_index())

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
    return axes_snr, fig_snr, snr_avg


@app.cell
def __(mo):
    mo.md("## Statistical tests (paired t-tests across conditions)")
    return ()


@app.cell
def __(gen_df, np, pd, stats):
    if gen_df.empty:
        stat_df = pd.DataFrame()
    else:
        snr_subj = (gen_df
                    .groupby(["subject", "run", "approach", "condition"])["snr"]
                    .mean()
                    .reset_index())

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
    return pairs, rows, snr_subj, stat_df


@app.cell
def __(mo):
    mo.md("## Topomap — noise reduction per condition")
    return ()


@app.cell
def __(gen_df, mne, np, plt, plot_values_topomap):
    if gen_df.empty:
        fig_topo = plt.figure()
    else:
        montage = mne.channels.make_standard_montage("GSN-HydroCel-129")

        CONDITION_ORDER_TOPO = ["per-recording", "per-subject", "across-subject"]

        # Compute per-channel mean SNR across subjects (post-saccade window)
        snr_ch = (gen_df[gen_df["condition"] == "post"]
                  .groupby(["approach", "ch_name"])["snr"]
                  .mean()
                  .reset_index())

        vmin = snr_ch["snr"].min()
        vmax = snr_ch["snr"].max()

        fig_topo, axes_topo = plt.subplots(
            1, len(CONDITION_ORDER_TOPO), figsize=(14, 4))

        for _ax, _cond in zip(axes_topo, CONDITION_ORDER_TOPO):
            _dat = snr_ch[snr_ch["approach"] == _cond]
            ch_names = _dat["ch_name"].tolist()
            values = _dat["snr"].values
            plot_values_topomap(
                _ax, values, ch_names, montage,
                vmin=vmin, vmax=vmax,
                cbar_label="SNR (dB)")
            _ax.set_title(_cond)

        fig_topo.suptitle(
            "Post-saccade SNR topomap per generalisation condition")
        fig_topo.tight_layout()
    fig_topo
    return (
        CONDITION_ORDER_TOPO, axes_topo, fig_topo,
        montage, snr_ch, vmax, vmin,
    )


@app.cell
def __(mo):
    mo.md("## Summary table — mean SNR (dB) and degradation vs. per-recording")
    return ()


@app.cell
def __(gen_df, np, pd):
    if gen_df.empty:
        summary_df = pd.DataFrame()
    else:
        summary = (gen_df
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
    return baseline, summary, summary_df


if __name__ == "__main__":
    app.run()
