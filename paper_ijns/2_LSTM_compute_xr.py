#!/usr/bin/env python3
"""2_LSTM_compute_xr.py — Aggregate processed EDF files into xarray .netcdf files.

Reads all *_clean.edf / *_noise.edf files produced by the pipeline scripts and
writes .netcdf files to the current directory (paper_ijns/ when run via SLURM).

The four tasks are independent and can be submitted as separate SLURM jobs:
    et        — eye-tracking SNR/topo (stimulus-locked)
    et_diff   — eye-tracking SNR/topo (saccade-direction-locked)
    erp       — EEG signals/topo (stimulus-locked)
    erp_diff  — EEG signals/topo (saccade-direction-locked)

Prerequisites: the following scripts must have been run first:
    1.1_run_eog_lstm_regression_mp.py  (all three conditions)
    1.2_run_eog_lstm_ica_mp.py
    1.4_run_eog_lstm_sim_mp.py

Usage
-----
    python paper_ijns/2_LSTM_compute_xr.py --root /path/to/processed
    python paper_ijns/2_LSTM_compute_xr.py --root /path/to/processed --task et
    EOG_PROCESSED_PATH=/path/to/processed python paper_ijns/2_LSTM_compute_xr.py
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from analyses import compute_et_xarrays, compute_erp_xarrays

TASKS: dict[str, tuple] = {
    "et":       (compute_et_xarrays,  {"diff": False}),
    "et_diff":  (compute_et_xarrays,  {"diff": True}),
    "erp":      (compute_erp_xarrays, {"diff": False}),
    "erp_diff": (compute_erp_xarrays, {"diff": True}),
}


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
    parser.add_argument(
        "--task",
        choices=list(TASKS),
        default=None,
        metavar="TASK",
        help="Which xarray to compute: et, et_diff, erp, erp_diff. "
             "If omitted, all four are run sequentially.",
    )
    parser.add_argument(
        "--condition",
        choices=["perrecording", "persubject", "acrosssubject"],
        default="perrecording",
        help="Which LSTM condition to aggregate (default: perrecording).",
    )
    args = parser.parse_args()
    root = args.root

    print(f"Input directory: {root}  condition: {args.condition}", flush=True)

    to_run = [args.task] if args.task else list(TASKS)
    for task in to_run:
        fn, kwargs = TASKS[task]
        print(f"[{task}] Computing ...", flush=True)
        fn(root, lstm_condition=args.condition, **kwargs)
        print(f"[{task}] Done.", flush=True)


if __name__ == "__main__":
    main()
