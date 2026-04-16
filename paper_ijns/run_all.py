#!/usr/bin/env python
"""run_all.py — Orchestrate the full paper_ijns analysis pipeline.

Runs all automated steps in order.  Step 3 (biophysical forward model) requires
SimNIBS + FreeSurfer; it runs via the ``--env-simnibs`` conda environment
(default: ``base``) which is expected to have SimNIBS ≥ 4 installed.
With ``--no-recompute`` it is skipped if ``leadfield_elect_only.npy`` already exists.
Analysis notebooks (Steps 6–9) are run as plain Python scripts so that all
computations execute and netcdf artefacts are written, but figures are not
displayed — open them interactively with ``marimo edit <notebook>`` to explore
results visually.

Usage
-----
# Run everything with defaults (output → paper_ijns/processed/)
python paper_ijns/run_all.py

# Custom output directory, skip already-computed files
python paper_ijns/run_all.py --root /data/eog_study/processed/ --no-recompute

# Run only data-processing steps (skip analysis notebooks)
python paper_ijns/run_all.py --skip-analysis

# Run only LSTM per-recording and ICA (skip generalisation conditions)
python paper_ijns/run_all.py --lstm-conditions perrecording
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

# All paths are resolved relative to this script's directory
SCRIPT_DIR = Path(__file__).parent.resolve()

# ANSI colours (disabled automatically on non-TTY)
_USE_COLOR = sys.stdout.isatty()
GREEN  = "\033[32m" if _USE_COLOR else ""
RED    = "\033[31m" if _USE_COLOR else ""
YELLOW = "\033[33m" if _USE_COLOR else ""
BOLD   = "\033[1m"  if _USE_COLOR else ""
RESET  = "\033[0m"  if _USE_COLOR else ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _header(title: str) -> None:
    width = 68
    print(f"\n{BOLD}{'─' * width}{RESET}")
    print(f"{BOLD}  {title}{RESET}")
    print(f"{BOLD}{'─' * width}{RESET}")


def run_step(
    title: str,
    cmd: list[str],
    *,
    skip: bool = False,
    cwd: Path | None = None,
    env: dict | None = None,
) -> bool:
    """Run a subprocess step, print timing and pass/fail status.

    Returns True on success, False on failure (non-zero exit code or exception).
    """
    _header(title)

    if skip:
        print(f"  {YELLOW}SKIPPED{RESET}")
        return True

    print(f"  cmd: {' '.join(cmd)}")
    print(f"  cwd: {cwd or SCRIPT_DIR}\n")

    t0 = time.monotonic()
    try:
        result = subprocess.run(
            cmd,
            cwd=str(cwd or SCRIPT_DIR),
            env=env,
        )
        ok = result.returncode == 0
    except Exception as exc:
        print(f"  {RED}ERROR: {exc}{RESET}")
        ok = False

    elapsed = time.monotonic() - t0
    mins, secs = divmod(int(elapsed), 60)
    elapsed_str = f"{mins}m {secs}s" if mins else f"{secs}s"

    if ok:
        print(f"\n  {GREEN}DONE{RESET}  ({elapsed_str})")
    else:
        print(f"\n  {RED}FAILED{RESET}  ({elapsed_str})")
    return ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the full paper_ijns analysis pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--root",
        default=str(SCRIPT_DIR / "processed"),
        help="Output directory for processed EDF files",
    )
    parser.add_argument(
        "--no-recompute",
        dest="recompute",
        action="store_false",
        default=True,
        help="Skip recordings whose output files already exist",
    )
    parser.add_argument(
        "--lstm-conditions",
        nargs="+",
        choices=["perrecording", "persubject", "acrosssubject"],
        default=["perrecording", "persubject", "acrosssubject"],
        metavar="COND",
        help="LSTM training/testing conditions to run (Step 1)",
    )
    parser.add_argument(
        "--skip-analysis",
        action="store_true",
        default=False,
        help="Skip analysis notebooks (Steps 5–9); run data-processing only",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        default=False,
        help="Abort the pipeline on the first failed step",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        default=False,
        help=(
            "Smoke-test mode: process only subject EP10 with the perrecording "
            "condition. Useful for verifying the full pipeline is functional."
        ),
    )
    parser.add_argument(
        "--quick-subjects",
        nargs="+",
        default=["EP10"],
        metavar="SUBJECT",
        help="Subjects to use in --quick mode (default: EP10)",
    )
    parser.add_argument(
        "--env-simnibs",
        default="base",
        metavar="ENV",
        help=(
            "Conda environment that has SimNIBS installed, used for Step 3 "
            "(default: base, which uses Python 3.9 required by SimNIBS on macOS Intel)"
        ),
    )
    args = parser.parse_args()

    py = sys.executable
    # Step 3 must run under a Python 3.9 conda env (SimNIBS macOS Intel constraint)
    py_simnibs = ["conda", "run", "-n", args.env_simnibs, "python"]
    # Resolve root to absolute so sub-scripts get a valid path regardless of cwd
    root = str(Path(args.root).resolve()) + "/"
    recompute_flag = [] if args.recompute else ["--no-recompute"]

    # --quick overrides: single subject, perrecording only
    if args.quick:
        subjects_flag = ["--subjects"] + args.quick_subjects
        lstm_conditions = ["perrecording"]
        print(
            f"\n{YELLOW}{BOLD}Quick mode:{RESET} processing subjects "
            f"{args.quick_subjects} with perrecording condition only.\n"
        )
    else:
        subjects_flag = []
        lstm_conditions = args.lstm_conditions

    results: dict[str, bool] = {}

    # ------------------------------------------------------------------
    # Step 1 — LSTM regression (one pass per requested condition)
    # ------------------------------------------------------------------
    for cond in lstm_conditions:
        key = f"Step 1 — LSTM ({cond})"
        ok = run_step(
            key,
            [py, "1.1_run_eog_lstm_regression_mp.py",
             "--condition", cond,
             "--root", root] + recompute_flag + subjects_flag,
        )
        results[key] = ok
        if not ok and args.fail_fast:
            _abort(results)

    # ------------------------------------------------------------------
    # Step 2 — ICA + ICLabel
    # ------------------------------------------------------------------
    ok = run_step(
        "Step 2 — ICA + ICLabel",
        [py, "1.2_run_eog_lstm_ica_mp.py",
         "--root", root] + recompute_flag + subjects_flag,
    )
    results["Step 2 — ICA + ICLabel"] = ok
    if not ok and args.fail_fast:
        _abort(results)

    # ------------------------------------------------------------------
    # Step 3 — Biophysical forward model (runs under --env-simnibs env)
    # ------------------------------------------------------------------
    leadfield_file = SCRIPT_DIR / "leadfield_elect_only.npy"
    skip_step3 = (not args.recompute) and leadfield_file.exists()
    ok = run_step(
        "Step 3 — Biophysical forward model",
        py_simnibs + ["1.3_eog_gen_model_2025.py"],
        skip=skip_step3,
    )
    step3_result = None if skip_step3 else ok
    results["Step 3 — Biophysical forward model"] = step3_result  # type: ignore
    if not skip_step3 and not ok and args.fail_fast:
        _abort(results)

    # ------------------------------------------------------------------
    # Step 4 — Biophysical simulation cleaning
    # ------------------------------------------------------------------
    ok = run_step(
        "Step 4 — Biophysical simulation cleaning",
        [py, "1.4_run_eog_lstm_sim_mp.py",
         "--root", root] + recompute_flag + subjects_flag,
    )
    results["Step 4 — Biophysical simulation"] = ok
    if not ok and args.fail_fast:
        _abort(results)

    # ------------------------------------------------------------------
    # Steps 5–9 — Analysis notebooks (run as scripts)
    # ------------------------------------------------------------------
    analysis_steps = [
        ("Step 5 — Aggregate to xarray",        "2_LSTM_compute_xr.py"),
        ("Step 6 — ERP analysis & figures",      "3_Pytorch_EOG_LSTM_analysis.py"),
        ("Step 7 — Generalisation analysis",     "4_generalization_analysis.py"),
        ("Step 8 — Power spectrum analysis",     "5_power_spectrum_analysis.py"),
        ("Step 9 — Functional connectivity",     "6_functional_connectivity_analysis.py"),
    ]

    # Pass root path so analysis notebooks use the correct processed directory
    analysis_env = {**os.environ, "EOG_PROCESSED_PATH": root}

    for title, notebook in analysis_steps:
        ok = run_step(
            title,
            [py, notebook],
            skip=args.skip_analysis,
            env=analysis_env,
        )
        results[title] = ok
        if not ok and args.fail_fast:
            _abort(results)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    _header("Pipeline summary")
    all_ok = True
    for step, ok in results.items():
        if ok is None:
            status = f"{YELLOW}SKIPPED{RESET}"
        elif ok:
            status = f"{GREEN}OK{RESET}"
        else:
            status = f"{RED}FAILED{RESET}"
            all_ok = False
        print(f"  {status}  {step}")

    print()
    if all_ok:
        print(f"{GREEN}{BOLD}All steps completed successfully.{RESET}")
    else:
        print(f"{RED}{BOLD}One or more steps failed. "
              f"Re-run failed steps individually for details.{RESET}")
        sys.exit(1)

    print(
        f"\n{YELLOW}Note:{RESET} Analysis notebooks (Steps 6–9) were run as scripts "
        "and computed all results, but figures were not displayed.\n"
        "Open them interactively for visual exploration:\n"
        "  marimo edit paper_ijns/3_Pytorch_EOG_LSTM_analysis.py\n"
        "  marimo edit paper_ijns/4_generalization_analysis.py\n"
        "  marimo edit paper_ijns/5_power_spectrum_analysis.py\n"
        "  marimo edit paper_ijns/6_functional_connectivity_analysis.py"
    )


def _abort(results: dict) -> None:
    _header("Pipeline aborted (--fail-fast)")
    for step, ok in results.items():
        if ok is None:
            status = f"{YELLOW}SKIPPED{RESET}"
        elif ok:
            status = f"{GREEN}OK{RESET}"
        else:
            status = f"{RED}FAILED{RESET}"
        print(f"  {status}  {step}")
    sys.exit(1)


if __name__ == "__main__":
    main()
