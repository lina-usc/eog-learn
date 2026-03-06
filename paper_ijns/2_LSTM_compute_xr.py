#!/usr/bin/env python3
"""2_LSTM_compute_xr.py — Aggregate processed EDF files into xarray .netcdf files.

Reads all *_clean.edf / *_noise.edf files produced by the pipeline scripts and
writes four .netcdf files to the same directory:

    snr.netcdf, topo_erp.netcdf, et_signals.netcdf
    eeg_signals.netcdf, topo_raw.netcdf
    (and *_diff.netcdf counterparts for saccade-direction-locked analysis)

Prerequisites: the following scripts must have been run first:
    1.1_run_eog_lstm_regression_mp.py
    1.2_run_eog_lstm_ica_mp.py
    1.4_run_eog_lstm_sim_mp.py

Usage
-----
    python paper_ijns/2_LSTM_compute_xr.py --root /path/to/processed
    EOG_PROCESSED_PATH=/path/to/processed python paper_ijns/2_LSTM_compute_xr.py
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from analyses import compute_et_xarrays, compute_erp_xarrays


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate processed EDF files into xarray .netcdf files."
    )
    parser.add_argument(
        "--root",
        default=os.environ.get("EOG_PROCESSED_PATH", "processed"),
        help="Path to processed data directory containing *_clean.edf files "
             "(default: $EOG_PROCESSED_PATH or 'processed')",
    )
    args = parser.parse_args()
    root = args.root

    print(f"Input directory: {root}", flush=True)

    print("Computing standard (stimulus-locked) xarrays ...", flush=True)
    compute_et_xarrays(root, diff=False)
    compute_erp_xarrays(root, diff=False)

    print("Computing differential (saccade-direction-locked) xarrays ...", flush=True)
    compute_et_xarrays(root, diff=True)
    compute_erp_xarrays(root, diff=True)

    print("Done.", flush=True)


if __name__ == "__main__":
    main()
