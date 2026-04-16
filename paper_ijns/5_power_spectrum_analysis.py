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
    import matplotlib.ticker as mticker
    import seaborn as sns
    import mne
    from tqdm.notebook import tqdm
    from scipy import stats
    import matplotlib as mpl
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    import eoglearn
    from eoglearn.viz import plot_values_topomap

    return Path, inset_axes, mne, mpl, np, pd, plot_values_topomap, plt, tqdm


@app.cell
def _(mo):
    mo.md("""
    # Power Spectrum Analysis

    Compares the power spectral density (PSD) across EOG-cleaning approaches:

    - **Original** — band-passed, average-referenced EEG before cleaning
    - **LSTM** — LSTM regression cleaning (`_clean.edf`)
    - **ICA** — ICA + ICLabel cleaning (`_ica.edf`)
    - **Biophysical (global)** — leadfield-based simulation, global scaling (`_sim.edf`)
    - **Biophysical (local)** — leadfield-based simulation, per-channel scaling (`_simlocal.edf`)

    Also shows the **removed components** (noise EDFs) to characterise what each
    method discards.

    **Prerequisites:** Steps 1–4 of the pipeline must have been run.
    """)
    return


@app.cell
def _():
    path_input = "/Volumes/SSD/eoglearn_results"  # "Path to processed data directory"
    return (path_input,)


@app.cell
def _(Path, mne, np, tqdm):
    # Cleaning approaches and their EDF suffixes
    CLEANED_KINDS = {
        "original": "original",
        "ML": "clean",
        "ICA": "ica",
        "Biophysical": "sim",
        "Regression": "simlocal",
    }
    NOISE_KINDS = {
        "ML": "noise",
        "ICA": "noiseica",
        "Biophysical": "noisesim",
        "Regression": "noisesimlocal",
    }
    ALL_SUFFIXES = list(CLEANED_KINDS.values()) + list(NOISE_KINDS.values())

    def _fix_raw(raw):
        if "TIME" in raw.ch_names:
            raw.drop_channels(["TIME"])
        eeg_ch = [c for c in raw.ch_names if c.startswith("E") or c == "Cz"]
        misc_ch = [c for c in raw.ch_names if c not in eeg_ch]
        raw.set_channel_types(dict(zip(eeg_ch, ["eeg"] * len(eeg_ch))))
        raw.set_channel_types(dict(zip(misc_ch, ["misc"] * len(misc_ch))))
        return raw

    def load_psds(path, fmin=0.5, fmax=40.0):
        """Load all processed recordings and compute PSD per approach.

        Returns
        -------
        psds : dict
            {suffix: np.ndarray of shape (n_recordings, n_channels, n_freqs)}
        freqs : np.ndarray  (n_freqs,)
        ch_names : list[str]
        recording_ids : list[tuple[str, str]]  (subject, run)
        """
        files = sorted(Path(path).glob("*_clean.edf"))

        psds = {s: [] for s in ALL_SUFFIXES}
        recording_ids = []
        freqs = None
        ch_names = None

        for fname_clean in tqdm(files, desc="Computing PSDs"):
            stem = fname_clean.name.replace("_clean.edf", "")
            parts = stem.split("_")
            if len(parts) < 2:
                continue
            subject, run = parts[0], parts[1]

            # Require all files to be present
            fnames = {
                s: fname_clean.parent / f"{stem}_{s}.edf"
                for s in ALL_SUFFIXES
            }
            if not all(f.exists() for f in fnames.values()):
                continue

            rec_psds = {}
            try:
                for suffix, fpath in fnames.items():
                    raw = _fix_raw(mne.io.read_raw_edf(fpath, verbose=False))
                    spectrum = raw.compute_psd(
                        method="welch", fmin=fmin, fmax=fmax,
                        picks="eeg", verbose=False)
                    rec_psds[suffix] = spectrum.get_data()  # (n_ch, n_freqs)
                    if freqs is None:
                        freqs = spectrum.freqs
                    if ch_names is None:
                        ch_names = spectrum.ch_names
            except Exception as e:
                print(f"Skipping {stem}: {e}")
                continue

            for suffix in ALL_SUFFIXES:
                psds[suffix].append(rec_psds[suffix])
            recording_ids.append((subject, run))

        if not recording_ids:
            raise RuntimeError(
                "No complete recordings found in the processed directory. "
                "Ensure all Steps 1–4 have been run successfully."
            )
        psds = {s: np.stack(psds[s], axis=0) for s in ALL_SUFFIXES}
        print(f"Loaded {len(recording_ids)} recordings, "
              f"{len(ch_names)} channels, {len(freqs)} frequency bins")
        return psds, freqs, ch_names, recording_ids

    return CLEANED_KINDS, NOISE_KINDS, load_psds


@app.cell
def _(Path, load_psds, np, path_input):
    _cache = Path(path_input) / "psds_cache.npz"

    if _cache.exists():
        _f = np.load(_cache, allow_pickle=True)
        psds = {k: _f[k] for k in _f.files
                if k not in ("freqs", "ch_names", "recording_ids")}
        freqs = _f["freqs"]
        ch_names = list(_f["ch_names"])
        recording_ids = [tuple(r) for r in _f["recording_ids"]]
        print(f"Loaded from cache: {_cache}")
    else:
        psds, freqs, ch_names, recording_ids = load_psds(path_input)
        np.savez(_cache, freqs=freqs, ch_names=ch_names,
                 recording_ids=recording_ids, **psds)
        print(f"Saved cache to {_cache}")
    return ch_names, freqs, psds


@app.cell
def _(mo):
    mo.md("""
    ## Mean PSD — cleaned signals vs. original
    """)
    return


@app.cell
def _(CLEANED_KINDS, freqs, np, plt, psds):
    COLORS = {
        "original": "black",
        "LSTM": "#e41a1c",
        "ICA": "#377eb8",
        "Biophysical (global)": "#4daf4a",
        "Biophysical (local)": "#984ea3",
    }

    # Average across recordings and channels → (n_freqs,)
    mean_psd = {
        label: 10 * np.log10(
            psds[suffix].mean(axis=(0, 1))
        )
        for label, suffix in CLEANED_KINDS.items()
    }

    fig_psd, ax_psd = plt.subplots(figsize=(9, 4))
    for _label, _psd_db in mean_psd.items():
        _lw = 2.0 if _label == "original" else 1.5
        _ls = "-" if _label == "original" else "--"
        ax_psd.plot(freqs, _psd_db, label=_label,
                    color=COLORS[_label], lw=_lw, ls=_ls)

    ax_psd.set_xlabel("Frequency (Hz)")
    ax_psd.set_ylabel("Power (dB)")
    ax_psd.set_title("Mean PSD — original vs. cleaned signals\n"
                     "(averaged across recordings and channels)")
    ax_psd.legend(fontsize=9)
    ax_psd.set_xlim(freqs[0], freqs[-1])
    fig_psd.tight_layout()
    fig_psd
    return (COLORS,)


@app.cell
def _(mo):
    mo.md("""
    ## Mean PSD — removed components (noise)
    """)
    return


@app.cell
def _(COLORS, NOISE_KINDS, freqs, np, plt, psds):
    mean_noise_psd = {
        label: 10 * np.log10(
            psds[suffix].mean(axis=(0, 1))
        )
        for label, suffix in NOISE_KINDS.items()
    }

    # Include original for reference
    from analyses import rms as _rms  # noqa: F401 (just for reference line)

    fig_noise, ax_noise = plt.subplots(figsize=(9, 4))
    for _label, _psd_db in mean_noise_psd.items():
        ax_noise.plot(freqs, _psd_db, label=_label,
                      color=COLORS[_label], lw=1.5)

    ax_noise.set_xlabel("Frequency (Hz)")
    ax_noise.set_ylabel("Power (dB)")
    ax_noise.set_title("Mean PSD — removed EOG components\n"
                       "(averaged across recordings and channels)")
    ax_noise.legend(fontsize=9)
    ax_noise.set_xlim(freqs[0], freqs[-1])
    fig_noise.tight_layout()
    fig_noise
    return


@app.cell
def _(mo):
    mo.md("""
    ## Noise fraction per frequency

    Proportion of original power that each method removes:
    `noise_fraction(f) = PSD_noise(f) / PSD_original(f)`

    Values near 1 indicate the approach removes most of the power at that frequency;
    values near 0 indicate little removal.
    """)
    return


@app.cell
def _(CLEANED_KINDS, COLORS, NOISE_KINDS, freqs, np, plt, psds):
    # noise / original (linear ratio, per recording then averaged)
    orig_suffix = CLEANED_KINDS["original"]
    noise_frac = {}
    for _label, _noise_suffix in NOISE_KINDS.items():
        ratio = (psds[_noise_suffix] /
                 np.maximum(psds[orig_suffix], 1e-30))  # avoid divide-by-zero
        noise_frac[_label] = ratio.mean(axis=(0, 1))  # mean over recordings & ch

    fig_frac, ax_frac = plt.subplots(figsize=(9, 4))
    for _label, frac in noise_frac.items():
        ax_frac.plot(freqs, frac * 100, label=_label,
                     color=COLORS[_label], lw=1.5)

    ax_frac.axhline(0, color="grey", lw=0.8, ls=":")
    ax_frac.set_xlabel("Frequency (Hz)")
    ax_frac.set_ylabel("Noise fraction (%)")
    ax_frac.set_title("Fraction of original power removed per frequency\n"
                      "(averaged across recordings and channels)")
    ax_frac.legend(fontsize=9)
    ax_frac.set_xlim(freqs[0], freqs[-1])
    ax_frac.set_ylim(bottom=0)
    fig_frac.tight_layout()
    fig_frac
    return


@app.cell
def _(mo):
    mo.md("""
    ## PSD topomaps — noise fraction by frequency band

    Spatial distribution of the noise fraction across EEG channels at frequency
    bands where EOG artefacts dominate.
    """)
    return


@app.cell
def _(
    NOISE_KINDS,
    ch_names,
    freqs,
    inset_axes,
    mne,
    mpl,
    np,
    plot_values_topomap,
    plt,
    psds,
):
    BANDS = {
        "δ (0.5–4 Hz)": (0.5, 4.0),
        "θ (4–8 Hz)": (4.0, 8.0),
        "α (8–13 Hz)": (8.0, 13.0),
    }
    montage = mne.channels.make_standard_montage("GSN-HydroCel-129")
    orig_suf = "original"


    n_bands = len(BANDS)
    n_methods = len(NOISE_KINDS)

    fig_topo, axes_topo = plt.subplots(
        n_methods, n_bands,
        figsize=(2 * n_bands, 2 * n_methods),
        squeeze=False,
    )
    fig_topo.subplots_adjust(bottom=0, top=0.93, left=0.05, right=0.99, hspace=0.05, wspace=0.05) 

    for col, (_band_label, (_fmin_b, _fmax_b)) in enumerate(BANDS.items()):
        _band_mask = (freqs >= _fmin_b) & (freqs < _fmax_b)

        for row, (_method_label, _noise_suffix) in enumerate(NOISE_KINDS.items()):
            # Per-channel noise fraction in band: mean over freq bins then recordings
            _orig_band = psds[orig_suf][:, :, _band_mask].mean(axis=2)  # (rec, ch)
            _noise_band = psds[_noise_suffix][:, :, _band_mask].mean(axis=2)
            frac_ch = (_noise_band / np.maximum(_orig_band, 1e-30)).mean(axis=0) * 100

            _ax = axes_topo[row, col]
            plot_values_topomap(
                dict(zip(ch_names, frac_ch)),
                montage,
                axes=_ax,
                vmin=0, vmax=100,
                colorbar=False, #(col == n_methods - 1),
                cbar_label="Noise fraction (%)",
                show=False,
            )

            if row == 0:
                _ax.set_title(_band_label, pad=28)   # extra padding so title clears the colorbar
                # colorbar above this axes
                _norm = mpl.colors.Normalize(vmin=0, vmax=100)
                _sm = mpl.cm.ScalarMappable(cmap="RdBu_r", norm=_norm)
                _sm.set_array([])
                _cbar_ax = inset_axes(
                    _ax, width="85%", height="5%", loc="lower center",
                    bbox_to_anchor=(0, 1.02, 1, 1), bbox_transform=_ax.transAxes,
                    borderpad=0,
                )
                _cbar_ax.tick_params(labelsize=10)
                cbar = fig_topo.colorbar(_sm, cax=_cbar_ax, orientation="horizontal")
                _cbar_ax.xaxis.set_ticks_position("top")
                cbar.locator = mpl.ticker.MaxNLocator(nbins=1)
                cbar.update_ticks()
        
            if col == 0:
                _ax.set_ylabel(_method_label)


    fig_topo.savefig("effect_spectrum.png", dpi=300)
    fig_topo.tight_layout()

    fig_topo
    return (BANDS,)


@app.cell
def _(mo):
    mo.md("""
    ## Frontal vs. posterior PSD comparison

    EOG artefacts are largest at frontal electrodes. Comparing how cleaning
    affects PSD at frontal vs. posterior sites shows whether methods
    selectively target the artefact topography.
    """)
    return


@app.cell
def _(CLEANED_KINDS, COLORS, ch_names, freqs, np, plt, psds):
    # Frontal channels: E1–E32 area (anterior); posterior: E65–E128
    frontal = [c for c in ch_names
               if c.startswith("E") and 1 <= int(c[1:]) <= 32]
    posterior = [c for c in ch_names
                 if c.startswith("E") and 65 <= int(c[1:]) <= 128]
    frontal_idx = [ch_names.index(c) for c in frontal]
    posterior_idx = [ch_names.index(c) for c in posterior]

    fig_fp, axes_fp = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax_fp, (region, idx) in zip(
            axes_fp, [("Frontal (E1–E32)", frontal_idx),
                      ("Posterior (E65–E128)", posterior_idx)]):
        for _label, _suffix in CLEANED_KINDS.items():
            psd_reg = 10 * np.log10(psds[_suffix][:, idx, :].mean(axis=(0, 1)))
            _lw = 2.0 if _label == "original" else 1.5
            _ls = "-" if _label == "original" else "--"
            ax_fp.plot(freqs, psd_reg, label=_label,
                       color=COLORS[_label], lw=_lw, ls=_ls)
        ax_fp.set_title(region)
        ax_fp.set_xlabel("Frequency (Hz)")
        ax_fp.set_xlim(freqs[0], freqs[-1])

    axes_fp[0].set_ylabel("Power (dB)")
    axes_fp[1].legend(fontsize=8, loc="upper right")
    fig_fp.suptitle("PSD by scalp region — original vs. cleaned signals")
    fig_fp.tight_layout()
    fig_fp
    return


@app.cell
def _(mo):
    mo.md("""
    ## Band-power summary — mean ± SD across recordings
    """)
    return


@app.cell
def _(BANDS, CLEANED_KINDS, ch_names, freqs, np, pd, psds):
    rows = []
    frontal_ch = [c for c in ch_names
                  if c.startswith("E") and 1 <= int(c[1:]) <= 32]
    frontal_i = [ch_names.index(c) for c in frontal_ch]

    for _band_label, (_fmin_b, _fmax_b) in BANDS.items():
        _mask = (freqs >= _fmin_b) & (freqs < _fmax_b)
        for _method_label, _suffix in CLEANED_KINDS.items():
            # Mean over frontal channels and freq bins, per recording
            bp = psds[_suffix][:, frontal_i, :][:, :, _mask].mean(axis=(1, 2))
            bp_db = 10 * np.log10(bp)
            rows.append({
                "band": _band_label,
                "approach": _method_label,
                "mean_dB": round(bp_db.mean(), 2),
                "std_dB": round(bp_db.std(), 2),
                "median_dB": round(np.median(bp_db), 2),
            })

    band_summary = pd.DataFrame(rows).sort_values(["band", "approach"])
    band_summary
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
