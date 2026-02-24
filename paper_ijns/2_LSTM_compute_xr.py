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
    from analyses import compute_et_xarrays, compute_erp_xarrays
    return Path, compute_erp_xarrays, compute_et_xarrays, sys


@app.cell
def __(mo):
    mo.md(
        """
        # LSTM Compute xarray Datasets

        Aggregates all processed EDF files (produced by the pipeline scripts) into
        xarray `.netcdf` files for downstream analysis.

        **Prerequisites:** the following scripts must have been run first:
        - `1.1_run_eog_lstm_regression_mp.py`
        - `1.2_run_eog_lstm_ica_mp.py`
        - `1.4_run_eog_lstm_sim_mp.py`
        """
    )
    return ()


@app.cell
def __(mo):
    path_input = mo.ui.text(
        value="processed",
        label="Path to processed data directory (containing *_clean.edf files)",
        full_width=True,
    )
    path_input
    return (path_input,)


@app.cell
def __(mo):
    mo.md("## Standard epoching (stimulus-locked)")
    return ()


@app.cell
def __(compute_et_xarrays, path_input):
    snr_xr, topo_ev_xr, et_signals_xr = compute_et_xarrays(
        path_input.value, diff=False
    )
    return et_signals_xr, snr_xr, topo_ev_xr


@app.cell
def __(compute_erp_xarrays, path_input):
    eeg_signals_xr, topo_xr = compute_erp_xarrays(path_input.value, diff=False)
    return eeg_signals_xr, topo_xr


@app.cell
def __(mo):
    mo.md("## Differential epoching (saccade-direction-locked)")
    return ()


@app.cell
def __(compute_et_xarrays, path_input):
    snr_xr_diff, topo_ev_xr_diff, et_signals_xr_diff = compute_et_xarrays(
        path_input.value, diff=True
    )
    return et_signals_xr_diff, snr_xr_diff, topo_ev_xr_diff


@app.cell
def __(compute_erp_xarrays, path_input):
    eeg_signals_xr_diff, topo_xr_diff = compute_erp_xarrays(
        path_input.value, diff=True
    )
    return eeg_signals_xr_diff, topo_xr_diff


if __name__ == "__main__":
    app.run()
