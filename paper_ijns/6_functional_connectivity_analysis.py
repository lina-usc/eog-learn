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
    import matplotlib.colors as mcolors
    import seaborn as sns
    import mne
    from mne_connectivity import spectral_connectivity_epochs
    from tqdm.notebook import tqdm
    from scipy import stats

    import eoglearn
    from eoglearn.viz import plot_values_topomap

    return (
        Path,
        mne,
        np,
        pd,
        plot_values_topomap,
        plt,
        spectral_connectivity_epochs,
        stats,
        tqdm,
    )


@app.cell
def _(mo):
    mo.md("""
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
    """)
    return


@app.cell
def _():
    import os as _os
    path_input = "/Volumes/SSD/eoglearn_results"
    epoch_len = 5.0 #5s; necessary for 5 cycles of at 1 Hz
    return epoch_len, path_input


@app.cell
def _(mo):
    mo.md("""
    ## Configuration — channel ROIs

    Channels are grouped into four anatomical regions based on their index in the
    GSN-HydroCel-129 montage.  These ROIs are used for the region-level connectivity
    summary plots.
    """)
    return


@app.cell
def _(mne):
    ROIS = {
        "Frontal": ["E3", "E8", "E9", "E10", "E15", "E16", "E18", "E22", "E23", "E25"], 
        "Left": ["E35", "E36", "E37", "E29", "E30", "E41", "E42"],
        "Right": ["E105", "E104", "E103", "E111", "E110", "E87", "E93"],
        "Pariental": ["E55", "E54", "E79", "E78", "E62", "E61"],
        "Occipital": ["E75", "E74", "E83", "E82", "E70"]
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
    return ALL_SUFFIXES, BANDS, CLEANED_KINDS, ROIS


@app.cell
def _(ALL_SUFFIXES, BANDS, Path, mne, np, spectral_connectivity_epochs, tqdm):
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

        _ALL_KEYS = ALL_SUFFIXES + ["eeg_movement"]

        def _ciplv_from_epochs(epochs, sfreq):
            fmin_all = [v[0] for v in BANDS.values()]
            fmax_all = [v[1] for v in BANDS.values()]
            con = spectral_connectivity_epochs(
                epochs, method="ciplv", mode="multitaper",
                sfreq=sfreq, fmin=fmin_all, fmax=fmax_all,
                faverage=True, verbose=False,
            )
            n_ch = len(epochs.ch_names)
            data_flat = con.get_data()
            n_conn = data_flat.shape[0]
            result = {}
            for b_idx, band_label in enumerate(BANDS):
                if n_conn == n_ch ** 2:
                    mat = data_flat[:, b_idx].reshape(n_ch, n_ch)
                else:
                    mat = np.zeros((n_ch, n_ch))
                    idx = np.triu_indices(n_ch, k=1)
                    mat[idx] = data_flat[:, b_idx]
                    mat = mat + mat.T
                result[band_label] = mat
            return result

        # We accumulate per-recording matrices, one per band per suffix
        acc = {(s, b): [] for s in _ALL_KEYS for b in BANDS}
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
                _raws_deriv = {}  # keep noise + noisesim for derived signal

                for suffix, fpath in fnames.items():
                    raw = _fix_raw(mne.io.read_raw_edf(fpath, verbose=False))
                    if suffix in ("noise", "noisesim"):
                        _raws_deriv[suffix] = raw
                    epochs = _make_fixed_epochs(raw, epoch_len_s)
                    if ch_names is None:
                        ch_names = epochs.ch_names
                    rec_ciplv[suffix] = _ciplv_from_epochs(epochs, raw.info["sfreq"])

                # Derived: EEG movement = noise − noisesim
                _mvt_info = _raws_deriv["noise"].copy().pick("eeg").info
                _mvt_raw = mne.io.RawArray(
                    _raws_deriv["noise"].get_data(picks="eeg")
                    - _raws_deriv["noisesim"].get_data(picks="eeg"),
                    _mvt_info, verbose=False,
                )
                rec_ciplv["eeg_movement"] = _ciplv_from_epochs(
                    _make_fixed_epochs(_mvt_raw, epoch_len_s), _mvt_info["sfreq"])

                for suffix in _ALL_KEYS:
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

    return (load_ciplv,)


@app.cell
def _(Path, epoch_len, load_ciplv, np, path_input):
    import ast
    _cache = Path(path_input) / "con_cache.npz"

    if _cache.exists():
        _f = np.load(_cache, allow_pickle=True)
        ch_names = list(_f["ch_names"])
        ciplv = _f["ciplv"].item()
        recording_ids = [tuple(r) for r in _f["recording_ids"]]
    else:
        ciplv, ch_names, recording_ids = load_ciplv(path_input,
                                                    epoch_len_s=epoch_len)
        np.savez(_cache, ciplv=ciplv, ch_names=ch_names,
                 recording_ids=recording_ids)
        print(f"Saved cache to {_cache}")
    return ch_names, ciplv


@app.cell
def _(mo):
    mo.md("""
    ## ciPLV matrices — mean across recordings

    Full channel × channel ciPLV matrix for each cleaning approach, averaged
    across recordings. Channels are ordered by index (E1 → E129, roughly
    anterior to posterior).
    """)
    return


@app.cell
def _(BANDS, CLEANED_KINDS, ciplv, np, plt):
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
    return


@app.cell
def _(mo):
    mo.md("""
    ## Difference matrices — original minus cleaned

    `ciPLV_original - ciPLV_cleaned` per channel pair.
    Positive values (warm colours) indicate connectivity that the method removed;
    negative values (cool colours) indicate connectivity the method *added*
    (over-correction).
    """)
    return


@app.cell
def _(BANDS, CLEANED_KINDS, ciplv, np, plt):
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
    return


@app.cell
def _(mo):
    mo.md("""
    ## Region-level connectivity — ROI summary

    Mean ciPLV within and between four anatomical regions.  The frontal region
    is most affected by EOG artefacts.
    """)
    return


@app.cell
def _(BANDS, CLEANED_KINDS, ROIS, ch_names, ciplv, np, plt):
    roi_labels = list(ROIS.keys())
    n_roi = len(roi_labels)

    # Map channel name → ROI index
    def _ch_roi_idx(ch_names, rois):
        idx = {}
        for roi_name, ch_set in rois.items():
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
    return roi_idx, roi_labels, roi_matrix_fn


@app.cell
def _(mo):
    mo.md("""
    ## Frontal connectivity topomap

    For each cleaning approach, the mean ciPLV between each channel and all
    *frontal* channels (E1–E32) is projected onto the scalp.  High values
    indicate strong coupling with the frontal region, which is often artefactual
    in the low-frequency bands.
    """)
    return


@app.cell
def _(
    BANDS,
    CLEANED_KINDS,
    ROIS,
    ch_names,
    ciplv,
    mne,
    np,
    plot_values_topomap,
    plt,
):
    frontal_idx = [i for i, c in enumerate(ch_names)
                   if c in ROIS["Frontal"]]

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
                cmap="Reds"
            )
            if _row == 0:
                _ax.set_title(_method_label, fontsize=9)
            if _col == 0:
                _ax.set_ylabel(b_label, fontsize=9)

    fig_fc.suptitle(
        "Mean ciPLV with frontal channels per approach and band",
        y=1.01)
    fig_fc.tight_layout()
    fig_fc
    return


@app.cell
def _():
    """_ALPHA = "α (8–13 Hz)"
    _METHODS = {
        "Original":             "original",
        "LSTM":                 "clean",
        "ICA":                  "ica",
        "Biophysical (global)": "sim",
    }

    _frontal_idx = [i for i, c in enumerate(ch_names) if c in ROIS["Frontal"]]
    _montage = mne.channels.make_standard_montage("GSN-HydroCel-129")

    _orig_fc = ciplv[("original", _ALPHA)].mean(axis=0)[:, _frontal_idx].mean(axis=1)
    _vmax = np.percentile(_orig_fc, 95)

    fig_alpha, axes_alpha = plt.subplots(2, 2, figsize=(5.6, 5.6))

    for (_method_label, _suffix), _ax in zip(_METHODS.items(), axes_alpha.flat):
        _fc_vec = ciplv[(_suffix, _ALPHA)].mean(axis=0)[:, _frontal_idx].mean(axis=1)
        plot_values_topomap(
            dict(zip(ch_names, _fc_vec)),
            _montage,
            axes=_ax,
            vmin=0, vmax=_vmax,
            colorbar=False,
            cmap="Reds",
            show=False,
        )
        _ax.set_title(_method_label, fontsize=10)

    # Single shared colorbar
    import matplotlib as _mpl
    _sm = _mpl.cm.ScalarMappable(cmap="Reds",
                                  norm=_mpl.colors.Normalize(vmin=0, vmax=_vmax))
    _sm.set_array([])
    fig_alpha.colorbar(_sm, ax=axes_alpha, shrink=0.6, label="ciPLV with frontal channels")

    fig_alpha.suptitle(f"Frontal ciPLV — {_ALPHA}", y=1.01)
    #fig_alpha.tight_layout()
    fig_alpha""";
    return


@app.cell
def _(ROIS, ch_names, ciplv, mne, np, plot_values_topomap, plt):
    _ALPHA = "α (8–13 Hz)"
    _CLEANING_METHODS = {
        "LSTM":                 "clean",
        "ICA":                  "ica",
        "Biophysical (global)": "sim",
    }

    _frontal_idx = [i for i, c in enumerate(ch_names) if c in ROIS["Frontal"]]
    _montage = mne.channels.make_standard_montage("GSN-HydroCel-129")

    _orig_fc = ciplv[("original", _ALPHA)].mean(axis=0)[:, _frontal_idx].mean(axis=1)
    _vmax = np.percentile(_orig_fc, 95)

    # Precompute diffs and their shared scale
    _diffs = {
        label: _orig_fc - ciplv[(suffix, _ALPHA)].mean(axis=0)[:, _frontal_idx].mean(axis=1)
        for label, suffix in _CLEANING_METHODS.items()
    }
    _diff_vmax = max(np.abs(d).max() for d in _diffs.values())

    import matplotlib as _mpl

    fig_alpha, axes_alpha = plt.subplots(2, 2, figsize=(5.6, 5.6))
    _ax_orig, *_axes_diff = axes_alpha.flat

    # Original — first panel
    plot_values_topomap(
        dict(zip(ch_names, _orig_fc)), _montage,
        axes=_ax_orig, vmin=0, vmax=_vmax,
        colorbar=False, cmap="Reds", show=False,
    )
    _ax_orig.set_title("Original", fontsize=10)

    # Difference panels
    for (_label, _diff), _ax in zip(_diffs.items(), _axes_diff):
        plot_values_topomap(
            dict(zip(ch_names, _diff)), _montage,
            axes=_ax, vmin=-_diff_vmax, vmax=_diff_vmax,
            colorbar=False, cmap="RdBu_r", show=False,
        )
        _ax.set_title(f"Orig − {_label}", fontsize=10)

    # Colorbar for original (top-left only)
    _sm_orig = _mpl.cm.ScalarMappable(
        cmap="Reds", norm=_mpl.colors.Normalize(vmin=0, vmax=_vmax))
    _sm_orig.set_array([])
    fig_alpha.colorbar(_sm_orig, ax=_ax_orig, shrink=0.8,
                       label="ciPLV", fraction=0.046, pad=0.04)

    # Shared colorbar for the three difference panels
    _sm_diff = _mpl.cm.ScalarMappable(
        cmap="RdBu_r", norm=_mpl.colors.Normalize(vmin=-_diff_vmax, vmax=_diff_vmax))
    _sm_diff.set_array([])
    fig_alpha.colorbar(_sm_diff, ax=list(_axes_diff), shrink=0.8,
                       label="Δ ciPLV (orig − cleaned)")

    fig_alpha.suptitle(f"Frontal ciPLV — {_ALPHA}", y=1.01)
    fig_alpha
    return


@app.cell
def _(ROIS, ch_names, ciplv, mne, np, plot_values_topomap, plt, stats):
    import matplotlib as _mpl
    import matplotlib.gridspec as _gs
    from itertools import combinations as _combinations

    _ALPHA = "α (8–13 Hz)"
    _METHODS = {
        "Original":              "original",
        "LSTM":                  "clean",
        "ICA":                   "ica",
        "Biophysical\n(global)": "sim",
    }
    _METHOD_LABELS = list(_METHODS.keys())

    _frontal_idx = [i for i, c in enumerate(ch_names) if c in ROIS["Frontal"]]
    _montage = mne.channels.make_standard_montage("GSN-HydroCel-129")

    _vmax = np.percentile(
        ciplv[("original", _ALPHA)].mean(axis=0)[:, _frontal_idx].mean(axis=1), 95)

    # Per-recording scalar: mean ciPLV of every channel with the frontal region
    _vals = [
        ciplv[(suffix, _ALPHA)][:, :, _frontal_idx].mean(axis=(1, 2))
        for suffix in _METHODS.values()
    ]

    # ── Layout ────────────────────────────────────────────────────────────────
    fig_alpha2 = plt.figure(figsize=(11, 5))
    _outer = _gs.GridSpec(1, 2, figure=fig_alpha2,
                          width_ratios=[1.1, 1], wspace=0.35)
    _left_gs = _gs.GridSpecFromSubplotSpec(
        2, 2, subplot_spec=_outer[0], wspace=0.05, hspace=0.1)
    _ax_vio = fig_alpha2.add_subplot(_outer[1])

    # ── Topomaps ──────────────────────────────────────────────────────────────
    _topo_axes = []
    for _idx, (_label, _suffix) in enumerate(_METHODS.items()):
        _r, _c = divmod(_idx, 2)
        _ax = fig_alpha2.add_subplot(_left_gs[_r, _c])
        _topo_axes.append(_ax)
        _fc_vec = ciplv[(_suffix, _ALPHA)].mean(axis=0)[:, _frontal_idx].mean(axis=1)
        plot_values_topomap(
            dict(zip(ch_names, _fc_vec)), _montage,
            axes=_ax, vmin=0, vmax=_vmax,
            colorbar=False, cmap="Reds", show=False,
        )
        _ax.set_title(_label, fontsize=9)

    _sm = _mpl.cm.ScalarMappable(
        cmap="Reds", norm=_mpl.colors.Normalize(vmin=0, vmax=_vmax))
    _sm.set_array([])
    fig_alpha2.colorbar(_sm, ax=_topo_axes, shrink=0.55, label="ciPLV")

    # ── Violin plot ───────────────────────────────────────────────────────────
    _vp = _ax_vio.violinplot(_vals, positions=range(len(_METHOD_LABELS)),
                              showmedians=True, showextrema=False)
    for _pc in _vp["bodies"]:
        _pc.set_alpha(0.7)

    _rng = np.random.default_rng(42)
    for _i, _v in enumerate(_vals):
        _ax_vio.scatter(
            _rng.uniform(-0.06, 0.06, len(_v)) + _i, _v,
            s=6, color="k", alpha=0.35, zorder=3)

    _ax_vio.set_xticks(range(len(_METHOD_LABELS)))
    _ax_vio.set_xticklabels(_METHOD_LABELS, fontsize=8)
    _ax_vio.set_ylabel("Mean ciPLV with frontal channels")
    _ax_vio.set_title("Distribution across recordings")

    # ── Significance brackets ─────────────────────────────────────────────────
    # Sort by span (adjacent pairs first → lowest bars, avoids crossings)
    _sig_pairs = []
    for _i, _j in sorted(_combinations(range(len(_METHOD_LABELS)), 2),
                          key=lambda x: x[1] - x[0]):
        _n = min(len(_vals[_i]), len(_vals[_j]))
        _, _p = stats.ttest_rel(_vals[_i][:_n], _vals[_j][:_n])
        _stars = "***" if _p < 0.001 else "**" if _p < 0.01 else "*" if _p < 0.05 else None
        if _stars:
            _sig_pairs.append((_i, _j, _stars))

    _y0   = max(v.max() for v in _vals)
    _h    = (_y0 - _ax_vio.get_ylim()[0]) * 0.06
    _y_cur = _y0 + _h
    for _i, _j, _stars in _sig_pairs:
        _ax_vio.plot([_i, _i, _j, _j],
                     [_y_cur, _y_cur + _h * 0.4, _y_cur + _h * 0.4, _y_cur],
                     lw=1, c="k")
        _ax_vio.text((_i + _j) / 2, _y_cur + _h * 0.45, _stars,
                     ha="center", va="bottom", fontsize=9)
        _y_cur += _h * 0.9

    _ax_vio.set_ylim(top=_y_cur + _h)

    fig_alpha2.suptitle(f"Frontal ciPLV — {_ALPHA}")
    fig_alpha2.tight_layout()
    fig_alpha2
    return


@app.cell
def _(mo):
    mo.md("""
    ## Signal decomposition — frontal ciPLV (α band)

    Frontal ciPLV for the three LSTM signal components (EOG artifact, EEG movement,
    EEG without movement) alongside the original signal, in the alpha band.
    """)
    return


@app.cell
def _(ciplv):
    type(ciplv[("noisesim", "α (8–13 Hz)")])
    return


@app.cell
def _(ROIS, ch_names, ciplv, mne, np, plot_values_topomap, plt):
    import matplotlib as _mpl_d

    _ALPHA_D = "α (8–13 Hz)"
    _DECOMP = {
        "Original":        "original",
        "EOG":             "noisesim",
        "EEG movement":    "eeg_movement",
        "EEG no movement": "clean",
    }

    _frontal_idx_d = [i for i, c in enumerate(ch_names) if c in ROIS["Frontal"]]
    _montage_d = mne.channels.make_standard_montage("GSN-HydroCel-129")

    _orig_fc_d = ciplv[("original", _ALPHA_D)].mean(axis=0)[:, _frontal_idx_d].mean(axis=1)
    _vmax_d = np.percentile(_orig_fc_d, 95)

    fig_decomp, axes_decomp = plt.subplots(2, 2, figsize=(5.6, 5.6))

    for (_label_d, _suffix_d), _ax_d in zip(_DECOMP.items(), axes_decomp.flat):
        _fc_vec_d = np.nanmean(ciplv[(_suffix_d, _ALPHA_D)], axis=0)[:, _frontal_idx_d].mean(axis=1)
        if _label_d != "Original":
            _fc_vec_d -= _orig_fc_d

        _vmax_d = np.percentile(np.abs(_fc_vec_d), 95)
        plot_values_topomap(
            dict(zip(ch_names, _fc_vec_d)), _montage_d,
            axes=_ax_d, vmin=-_vmax_d, vmax=_vmax_d,
            colorbar=True, show=False,
        )
        _ax_d.set_title(_label_d, fontsize=10)

    #_sm_d = _mpl_d.cm.ScalarMappable(
    #    cmap="Reds", norm=_mpl_d.colors.Normalize(vmin=0, vmax=_vmax_d))
    #_sm_d.set_array([])
    #fig_decomp.colorbar(_sm_d, ax=axes_decomp, shrink=0.6,
    #                    label="ciPLV with frontal channels")
    fig_decomp.suptitle(f"Signal decomposition — frontal ciPLV ({_ALPHA_D})", y=1.01)
    fig_decomp.tight_layout()
    fig_decomp
    return


@app.cell
def _(mo):
    mo.md("""
    ## Summary — mean ciPLV per ROI pair, method, and band
    """)
    return


@app.cell
def _(BANDS, CLEANED_KINDS, ciplv, pd, roi_idx, roi_labels, roi_matrix_fn):
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
    return


@app.cell
def _(mo):
    mo.md("""
    ## Statistical test — frontal ciPLV change (original vs. cleaned)

    Paired t-test across recordings: is the mean frontal ciPLV significantly
    reduced after cleaning?
    """)
    return


@app.cell
def _(BANDS, ROIS, ch_names, ciplv, np, pd, stats):
    frontal_i = [i for i, c in enumerate(ch_names)
                 if c in ROIS["Frontal"]]
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
    return


if __name__ == "__main__":
    app.run()
