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
    import matplotlib.colors as mcolors
    import seaborn as sns
    import mne
    from mne_connectivity import spectral_connectivity_epochs
    from tqdm.notebook import tqdm
    from scipy import stats

    import eoglearn
    from eoglearn.viz import plot_values_topomap
    return (
        Path, eoglearn, mne, mcolors, np, pd, plt,
        plot_values_topomap, sns, spectral_connectivity_epochs,
        stats, sys, tqdm,
    )


@app.cell
def __(mo):
    mo.md(
        """
        # Functional Connectivity Analysis — ciPLV

        Compares **corrected imaginary Phase-Locking Value (ciPLV)** across EOG-cleaning
        approaches. ciPLV measures phase synchrony while being insensitive to
        zero-phase-lag connectivity introduced by volume conduction and (analogously)
        by common EOG artefacts.

        The key question is: does each cleaning method remove artifactual connectivity
        (especially in low-frequency bands at frontal channels) without altering genuine
        brain connectivity elsewhere?

        **Frequency bands analysed:**
        - δ (1–4 Hz) — dominated by slow EOG drift
        - θ (4–8 Hz) — saccade-related transients
        - α (8–13 Hz) — posterior alpha rhythm

        **Prerequisites:** Steps 1–4 of the pipeline must have been run.
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
    epoch_len_input = mo.ui.slider(
        start=1, stop=10, step=1, value=4,
        label="Epoch length for connectivity estimation (seconds)",
    )
    mo.vstack([path_input, epoch_len_input])
    return epoch_len_input, path_input


@app.cell
def __(mo):
    mo.md(
        """
        ## Configuration — channel ROIs

        Channels are grouped into four anatomical regions based on their index in the
        GSN-HydroCel-129 montage.  These ROIs are used for the region-level connectivity
        summary plots.
        """
    )
    return ()


@app.cell
def __(mne):
    ROIS = {
        "Frontal":    list(range(1,  33)),   # E1–E32
        "Central":    list(range(33, 65)),   # E33–E64
        "Left-Post":  list(range(65, 97)),   # E65–E96
        "Right-Post": list(range(97, 129)),  # E97–E128
    }
    ROI_COLORS = {
        "Frontal": "#e41a1c",
        "Central": "#ff7f00",
        "Left-Post": "#377eb8",
        "Right-Post": "#984ea3",
    }

    CLEANED_KINDS = {
        "original":              "original",
        "LSTM":                  "clean",
        "ICA":                   "ica",
        "Biophysical (global)":  "sim",
        "Biophysical (local)":   "simlocal",
    }
    NOISE_KINDS = {
        "LSTM":                  "noise",
        "ICA":                   "noiseica",
        "Biophysical (global)":  "noisesim",
        "Biophysical (local)":   "noisesimlocal",
    }
    ALL_SUFFIXES = list(CLEANED_KINDS.values()) + list(NOISE_KINDS.values())

    BANDS = {
        "δ (1–4 Hz)":   (1.0,  4.0),
        "θ (4–8 Hz)":   (4.0,  8.0),
        "α (8–13 Hz)":  (8.0, 13.0),
    }

    montage = mne.channels.make_standard_montage("GSN-HydroCel-129")
    return (
        ALL_SUFFIXES, BANDS, CLEANED_KINDS, NOISE_KINDS,
        ROI_COLORS, ROIS, montage,
    )


@app.cell
def __(
    ALL_SUFFIXES, BANDS, Path, mne, np,
    path_input, epoch_len_input, spectral_connectivity_epochs, tqdm,
):
    def _fix_raw(raw):
        if "TIME" in raw.ch_names:
            raw.drop_channels(["TIME"])
        eeg_ch = [c for c in raw.ch_names if c.startswith("E") or c == "Cz"]
        misc_ch = [c for c in raw.ch_names if c not in eeg_ch]
        raw.set_channel_types(dict(zip(eeg_ch, ["eeg"] * len(eeg_ch))))
        raw.set_channel_types(dict(zip(misc_ch, ["misc"] * len(misc_ch))))
        return raw

    def _make_fixed_epochs(raw, epoch_len_s):
        """Segment raw into non-overlapping fixed-length epochs."""
        sfreq = raw.info["sfreq"]
        n_samples = int(epoch_len_s * sfreq)
        n_epochs = int(raw.n_times // n_samples)
        data = raw.get_data(picks="eeg")[:, : n_epochs * n_samples]
        data = data.reshape(data.shape[0], n_epochs, n_samples)
        # (n_epochs, n_channels, n_times)
        data = data.transpose(1, 0, 2)
        info = raw.copy().pick("eeg").info
        return mne.EpochsArray(data, info, verbose=False)

    def load_ciplv(path, epoch_len_s=4):
        """Compute ciPLV matrices for every recording and approach.

        Returns
        -------
        ciplv : dict
            {suffix: np.ndarray (n_recordings, n_channels, n_channels)}
        ch_names : list[str]
        freqs_dict : dict {band_label: np.ndarray}  (representative freqs used)
        recording_ids : list[tuple[str, str]]
        """
        files = sorted(Path(path).glob("*_clean.edf"))

        # We accumulate per-recording matrices, one per band per suffix
        acc = {(s, b): [] for s in ALL_SUFFIXES for b in BANDS}
        recording_ids = []
        ch_names = None

        for fname_clean in tqdm(files, desc="Computing ciPLV"):
            stem = fname_clean.name.replace("_clean.edf", "")
            parts = stem.split("_")
            if len(parts) < 2:
                continue
            subject, run = parts[0], parts[1]

            fnames = {
                s: fname_clean.parent / f"{stem}_{s}.edf"
                for s in ALL_SUFFIXES
            }
            if not all(f.exists() for f in fnames.values()):
                continue

            try:
                rec_ciplv = {}
                for suffix, fpath in fnames.items():
                    raw = _fix_raw(mne.io.read_raw_edf(fpath, verbose=False))
                    epochs = _make_fixed_epochs(raw, epoch_len_s)
                    if ch_names is None:
                        ch_names = epochs.ch_names

                    # spectral_connectivity_epochs returns one matrix per freq
                    # Compute once across all bands in a single call
                    fmin_all = [v[0] for v in BANDS.values()]
                    fmax_all = [v[1] for v in BANDS.values()]

                    con = spectral_connectivity_epochs(
                        epochs,
                        method="ciplv",
                        mode="multitaper",
                        sfreq=raw.info["sfreq"],
                        fmin=fmin_all,
                        fmax=fmax_all,
                        faverage=True,   # one value per band
                        verbose=False,
                    )
                    n_ch = len(ch_names)
                    data_flat = con.get_data()  # (n_conn, n_bands)
                    n_conn = data_flat.shape[0]

                    rec_ciplv[suffix] = {}
                    for b_idx, band_label in enumerate(BANDS):
                        if n_conn == n_ch ** 2:
                            # Full n×n matrix returned
                            mat = data_flat[:, b_idx].reshape(n_ch, n_ch)
                        else:
                            # Upper triangle only
                            mat = np.zeros((n_ch, n_ch))
                            idx = np.triu_indices(n_ch, k=1)
                            mat[idx] = data_flat[:, b_idx]
                            mat = mat + mat.T   # symmetrize
                        rec_ciplv[suffix][band_label] = mat

                for suffix in ALL_SUFFIXES:
                    for band_label in BANDS:
                        acc[(suffix, band_label)].append(
                            rec_ciplv[suffix][band_label])
                recording_ids.append((subject, run))

            except Exception as e:
                print(f"Skipping {stem}: {e}")
                continue

        if not recording_ids:
            raise RuntimeError(
                "No complete recordings found in the processed directory. "
                "Ensure all Steps 1–4 have been run successfully."
            )
        # Stack into arrays (n_recordings, n_ch, n_ch)
        ciplv = {
            (s, b): np.stack(acc[(s, b)], axis=0)
            for s, b in acc
            if acc[(s, b)]
        }
        print(f"Loaded {len(recording_ids)} recordings, "
              f"{len(ch_names)} channels")
        return ciplv, ch_names, recording_ids

    ciplv, ch_names, recording_ids = load_ciplv(
        path_input.value, epoch_len_s=epoch_len_input.value)
    return (
        ch_names, ciplv, epoch_len_input, load_ciplv, recording_ids,
    )


@app.cell
def __(mo):
    mo.md(
        """
        ## ciPLV matrices — mean across recordings

        Full channel × channel ciPLV matrix for each cleaning approach, averaged
        across recordings. Channels are ordered by index (E1 → E129, roughly
        anterior to posterior).
        """
    )
    return ()


@app.cell
def __(BANDS, CLEANED_KINDS, ch_names, ciplv, np, plt):
    n_methods = len(CLEANED_KINDS)
    n_bands = len(BANDS)

    # Shared color scale per band (vmax = 95th percentile of 'original')
    vmaxes = {}
    for _band_label in BANDS:
        _orig_mean = ciplv[("original", _band_label)].mean(axis=0)
        vmaxes[_band_label] = np.percentile(_orig_mean, 95)

    fig_mat, axes_mat = plt.subplots(
        n_bands, n_methods,
        figsize=(3.2 * n_methods, 3.2 * n_bands),
        squeeze=False,
    )

    for _row, _band_label in enumerate(BANDS):
        for _col, (_method_label, _suffix) in enumerate(CLEANED_KINDS.items()):
            _ax = axes_mat[_row, _col]
            _mat = ciplv[(_suffix, _band_label)].mean(axis=0)
            _im = _ax.imshow(_mat, vmin=0, vmax=vmaxes[_band_label],
                             cmap="viridis", aspect="auto",
                             interpolation="nearest")
            if _row == 0:
                _ax.set_title(_method_label, fontsize=9)
            if _col == 0:
                _ax.set_ylabel(_band_label, fontsize=9)
            _ax.set_xticks([])
            _ax.set_yticks([])
            if _col == n_methods - 1:
                plt.colorbar(_im, ax=_ax, fraction=0.046, label="ciPLV")

    fig_mat.suptitle("Mean ciPLV matrices per approach and frequency band",
                     y=1.01)
    fig_mat.tight_layout()
    fig_mat
    return axes_mat, fig_mat, n_bands, n_methods, vmaxes


@app.cell
def __(mo):
    mo.md(
        """
        ## Difference matrices — original minus cleaned

        `ciPLV_original - ciPLV_cleaned` per channel pair.
        Positive values (warm colours) indicate connectivity that the method removed;
        negative values (cool colours) indicate connectivity the method *added*
        (over-correction).
        """
    )
    return ()


@app.cell
def __(BANDS, CLEANED_KINDS, ciplv, np, plt):
    cleaning_methods = {k: v for k, v in CLEANED_KINDS.items()
                        if k != "original"}
    n_methods_c = len(cleaning_methods)
    n_bands_c = len(BANDS)

    fig_diff, axes_diff = plt.subplots(
        n_bands_c, n_methods_c,
        figsize=(3.2 * n_methods_c, 3.2 * n_bands_c),
        squeeze=False,
    )

    for _row, _band_label in enumerate(BANDS):
        _orig_mean = ciplv[("original", _band_label)].mean(axis=0)
        _diff_abs_max = max(
            np.abs(_orig_mean - ciplv[(s, _band_label)].mean(axis=0)).max()
            for s in cleaning_methods.values()
        )
        for _col, (_method_label, _suffix) in enumerate(cleaning_methods.items()):
            _ax = axes_diff[_row, _col]
            _diff = _orig_mean - ciplv[(_suffix, _band_label)].mean(axis=0)
            _im = _ax.imshow(_diff, vmin=-_diff_abs_max, vmax=_diff_abs_max,
                             cmap="RdBu_r", aspect="auto",
                             interpolation="nearest")
            if _row == 0:
                _ax.set_title(_method_label, fontsize=9)
            if _col == 0:
                _ax.set_ylabel(_band_label, fontsize=9)
            _ax.set_xticks([])
            _ax.set_yticks([])
            if _col == n_methods_c - 1:
                plt.colorbar(_im, ax=_ax, fraction=0.046,
                             label="Δ ciPLV (orig − clean)")

    fig_diff.suptitle(
        "ciPLV change: original − cleaned  (red = removed, blue = added)",
        y=1.01)
    fig_diff.tight_layout()
    fig_diff
    return axes_diff, cleaning_methods, fig_diff, n_bands_c, n_methods_c


@app.cell
def __(mo):
    mo.md(
        """
        ## Region-level connectivity — 4-ROI summary

        Mean ciPLV within and between four anatomical regions.  The frontal region
        is most affected by EOG artefacts.
        """
    )
    return ()


@app.cell
def __(BANDS, CLEANED_KINDS, ROI_COLORS, ROIS, ch_names, ciplv, np, plt):
    roi_labels = list(ROIS.keys())
    n_roi = len(roi_labels)

    # Map channel name → ROI index
    def _ch_roi_idx(ch_names, rois):
        idx = {}
        for roi_name, numbers in rois.items():
            ch_set = {f"E{n}" for n in numbers}
            idx[roi_name] = [i for i, c in enumerate(ch_names) if c in ch_set]
        return idx

    roi_idx = _ch_roi_idx(ch_names, ROIS)

    def roi_matrix_fn(mat, roi_idx, roi_labels):
        """Average ciPLV within/between ROIs."""
        n = len(roi_labels)
        roi_mat = np.zeros((n, n))
        for i, ri in enumerate(roi_labels):
            for j, rj in enumerate(roi_labels):
                if i == j:
                    idx_i = roi_idx[ri]
                    pairs = np.ix_(idx_i, idx_i)
                    sub = mat[pairs]
                    # exclude diagonal
                    np.fill_diagonal(sub, np.nan)
                    roi_mat[i, j] = np.nanmean(sub)
                else:
                    idx_i, idx_j = roi_idx[ri], roi_idx[rj]
                    roi_mat[i, j] = mat[np.ix_(idx_i, idx_j)].mean()
        return roi_mat

    n_methods_r = len(CLEANED_KINDS)
    n_bands_r = len(BANDS)

    fig_roi, axes_roi = plt.subplots(
        n_bands_r, n_methods_r,
        figsize=(2.8 * n_methods_r, 2.8 * n_bands_r),
        squeeze=False,
    )

    vmax_roi = {}
    for _band_label in BANDS:
        _orig_roi = roi_matrix_fn(
            ciplv[("original", _band_label)].mean(axis=0), roi_idx, roi_labels)
        vmax_roi[_band_label] = np.nanpercentile(_orig_roi, 95)

    for _row, _band_label in enumerate(BANDS):
        for _col, (_method_label, _suffix) in enumerate(CLEANED_KINDS.items()):
            _ax = axes_roi[_row, _col]
            _mean_mat = ciplv[(_suffix, _band_label)].mean(axis=0)
            _roi_mat = roi_matrix_fn(_mean_mat, roi_idx, roi_labels)
            _im = _ax.imshow(_roi_mat, vmin=0, vmax=vmax_roi[_band_label],
                             cmap="viridis", aspect="auto")
            _ax.set_xticks(range(n_roi))
            _ax.set_yticks(range(n_roi))
            _ax.set_xticklabels(roi_labels, rotation=45, ha="right", fontsize=7)
            _ax.set_yticklabels(roi_labels, fontsize=7)
            if _row == 0:
                _ax.set_title(_method_label, fontsize=9)
            if _col == 0:
                _ax.set_ylabel(_band_label, fontsize=9)
            if _col == n_methods_r - 1:
                plt.colorbar(_im, ax=_ax, fraction=0.046, label="ciPLV")

    fig_roi.suptitle("ROI-level ciPLV per approach and band", y=1.01)
    fig_roi.tight_layout()
    fig_roi
    return (
        _ch_roi_idx, roi_matrix_fn, axes_roi,
        fig_roi, n_bands_r, n_methods_r,
        n_roi, roi_idx, roi_labels, vmax_roi,
    )


@app.cell
def __(mo):
    mo.md(
        """
        ## Frontal connectivity topomap

        For each cleaning approach, the mean ciPLV between each channel and all
        *frontal* channels (E1–E32) is projected onto the scalp.  High values
        indicate strong coupling with the frontal region, which is often artefactual
        in the low-frequency bands.
        """
    )
    return ()


@app.cell
def __(BANDS, CLEANED_KINDS, ch_names, ciplv, mne, np, plt, plot_values_topomap, ROIS):
    frontal_idx = [i for i, c in enumerate(ch_names)
                   if c in {f"E{n}" for n in ROIS["Frontal"]}]

    montage_topo = mne.channels.make_standard_montage("GSN-HydroCel-129")

    n_b = len(BANDS)
    n_m = len(CLEANED_KINDS)

    # Shared vmax per band (95th pct of original)
    vmax_fc = {}
    for b_label in BANDS:
        orig_m = ciplv[("original", b_label)].mean(axis=0)
        fc = orig_m[:, frontal_idx].mean(axis=1)
        vmax_fc[b_label] = np.percentile(fc, 95)

    fig_fc, axes_fc = plt.subplots(
        n_b, n_m,
        figsize=(2.8 * n_m, 2.8 * n_b),
        squeeze=False,
    )

    for _row, b_label in enumerate(BANDS):
        for _col, (_method_label, _suffix) in enumerate(CLEANED_KINDS.items()):
            _mean_m = ciplv[(_suffix, b_label)].mean(axis=0)
            _fc_vec = _mean_m[:, frontal_idx].mean(axis=1)
            _ax = axes_fc[_row, _col]
            plot_values_topomap(
                dict(zip(ch_names, _fc_vec)),
                montage_topo,
                axes=_ax,
                vmin=0, vmax=vmax_fc[b_label],
                colorbar=(_col == n_m - 1),
                cbar_label="ciPLV",
                show=False,
            )
            if _row == 0:
                _ax.set_title(_method_label, fontsize=9)
            if _col == 0:
                _ax.set_ylabel(b_label, fontsize=9)

    fig_fc.suptitle(
        "Mean ciPLV with frontal channels (E1–E32) per approach and band",
        y=1.01)
    fig_fc.tight_layout()
    fig_fc
    return (
        axes_fc, b_label, fig_fc, frontal_idx,
        montage_topo, n_b, n_m, vmax_fc,
    )


@app.cell
def __(mo):
    mo.md("## Summary — mean ciPLV per ROI pair, method, and band")
    return ()


@app.cell
def __(
    BANDS, CLEANED_KINDS, roi_matrix_fn, ch_names, ciplv,
    np, pd, roi_idx, roi_labels,
):
    rows_stat = []
    for band_label_s, _ in BANDS.items():
        for method_label_s, suffix_s in CLEANED_KINDS.items():
            mean_mat_s = ciplv[(suffix_s, band_label_s)].mean(axis=0)
            std_mat_s = ciplv[(suffix_s, band_label_s)].std(axis=0)
            roi_mean = roi_matrix_fn(mean_mat_s, roi_idx, roi_labels)
            for i, ri in enumerate(roi_labels):
                for j, rj in enumerate(roi_labels):
                    if j < i:
                        continue
                    rows_stat.append({
                        "band": band_label_s,
                        "approach": method_label_s,
                        "ROI_A": ri,
                        "ROI_B": rj,
                        "mean_ciPLV": round(roi_mean[i, j], 4),
                    })

    summary_conn = (pd.DataFrame(rows_stat)
                    .sort_values(["band", "ROI_A", "ROI_B", "approach"])
                    .reset_index(drop=True))
    summary_conn
    return (
        band_label_s, i, j, method_label_s, ri, rj, roi_mean, rows_stat,
        std_mat_s, suffix_s, summary_conn,
    )


@app.cell
def __(mo):
    mo.md(
        """
        ## Statistical test — frontal ciPLV change (original vs. cleaned)

        Paired t-test across recordings: is the mean frontal ciPLV significantly
        reduced after cleaning?
        """
    )
    return ()


@app.cell
def __(BANDS, ch_names, ciplv, np, pd, stats, ROIS):
    frontal_i = [i for i, c in enumerate(ch_names)
                 if c in {f"E{n}" for n in ROIS["Frontal"]}]
    cleaning_map = {
        "LSTM":                 "clean",
        "ICA":                  "ica",
        "Biophysical (global)": "sim",
        "Biophysical (local)":  "simlocal",
    }

    stat_rows = []
    for b_lbl, _ in BANDS.items():
        # Per-recording mean frontal-frontal ciPLV
        orig_vals = (ciplv[("original", b_lbl)]
                     [:, frontal_i, :]
                     [:, :, frontal_i].mean(axis=(1, 2)))
        for method, _suffix in cleaning_map.items():
            clean_vals = (ciplv[(_suffix, b_lbl)]
                          [:, frontal_i, :]
                          [:, :, frontal_i].mean(axis=(1, 2)))
            n = min(len(orig_vals), len(clean_vals))
            t, p = stats.ttest_rel(orig_vals[:n], clean_vals[:n])
            d = np.mean(orig_vals[:n] - clean_vals[:n]) / np.std(
                orig_vals[:n] - clean_vals[:n])
            stat_rows.append({
                "band": b_lbl,
                "approach": method,
                "mean_orig": round(orig_vals.mean(), 4),
                "mean_clean": round(clean_vals.mean(), 4),
                "mean_diff": round((orig_vals[:n] - clean_vals[:n]).mean(), 4),
                "t": round(t, 3),
                "p": round(p, 4),
                "cohen_d": round(d, 3),
            })

    stat_conn_df = pd.DataFrame(stat_rows).sort_values(["band", "approach"])
    stat_conn_df
    return (
        b_lbl, clean_vals, cleaning_map, d, frontal_i,
        method, n, orig_vals, p, stat_conn_df, stat_rows, t,
    )


if __name__ == "__main__":
    app.run()
