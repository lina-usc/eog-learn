#!/usr/bin/env python3
"""report.py — Pipeline execution report.

Checks which subjects/runs completed each processing step by looking for
expected output files, and scans SLURM log files for tracebacks.

Usage
-----
    python paper_ijns/slurm/report.py
    python paper_ijns/slurm/report.py --root /path/to/processed/ --logs /path/to/logs/
"""
from __future__ import annotations

import re
import sys
import argparse
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent.parent   # paper_ijns/
sys.path.insert(0, str(SCRIPT_DIR))

import eoglearn.datasets.eegeyenet as _eyenet  # noqa: E402

# ── Step definitions ──────────────────────────────────────────────────────────
# (step_id, display_label, output_file_templates, ep_only)
# Templates are formatted with s=subject, r=run.
# ep_only=True → subjects without "EP" in their name are not expected to have
# output for this step (the processing scripts skip them by design).
STEPS: list[tuple[str, str, list[str], bool]] = [
    (
        "1.1 perrecording", "LSTM (per-recording)",
        ["{s}_{r}_original.edf", "{s}_{r}_clean.edf", "{s}_{r}_noise.edf"],
        False,
    ),
    (
        "1.1 persubject", "LSTM (per-subject)",
        ["{s}_{r}_clean_persubject.edf", "{s}_{r}_noise_persubject.edf"],
        True,
    ),
    (
        "1.1 acrosssubject", "LSTM (across-subject)",
        ["{s}_{r}_clean_acrosssubject.edf", "{s}_{r}_noise_acrosssubject.edf"],
        True,
    ),
    (
        "1.2", "ICA + ICLabel",
        ["{s}_{r}_ica.edf", "{s}_{r}_noiseica.edf"],
        False,
    ),
    (
        "1.4", "Biophysical simulation",
        [
            "{s}_{r}_sim.edf", "{s}_{r}_noisesim.edf",
            "{s}_{r}_simlocal.edf", "{s}_{r}_noisesimlocal.edf",
        ],
        True,
    ),
]


def _all_files_exist(templates: list[str], subject: str, run: int, root: Path) -> bool:
    return all((root / t.format(s=subject, r=run)).exists() for t in templates)


def scan_logs(log_dir: Path) -> list[tuple[str, str]]:
    """Return (filename, context_snippet) for every traceback found in log files."""
    hits = []
    log_files = sorted(log_dir.glob("*.out")) + sorted(log_dir.glob("*.err"))
    for log_file in log_files:
        try:
            text = log_file.read_text(errors="replace")
        except OSError:
            continue
        if "Traceback" not in text:
            continue
        for match in re.finditer(r"Traceback \(most recent call last\):", text):
            # Grab the last non-empty line before the traceback for context
            before = text[max(0, match.start() - 400) : match.start()]
            lines = [ln.strip() for ln in before.splitlines() if ln.strip()]
            ctx = lines[-1][:120] if lines else "(no context)"
            hits.append((log_file.name, ctx))
    return hits


def print_report(root: Path, log_dir: Path) -> None:
    runs_dict = _eyenet.get_subjects_runs()
    subjects = sorted(runs_dict.keys())

    W = 72
    any_failures = False

    print("=" * W)
    print("  PIPELINE EXECUTION REPORT")
    print(f"  Output root : {root}")
    print(f"  Log dir     : {log_dir}")
    print("=" * W)

    # ── Per-step file check ───────────────────────────────────────────────────
    for step_id, label, templates, ep_only in STEPS:
        ok: list[str] = []
        missing: list[tuple[str, int]] = []
        na_count = 0

        for subject in subjects:
            for run in runs_dict[subject]:
                if ep_only and "EP" not in subject:
                    na_count += 1
                    continue
                if _all_files_exist(templates, subject, run, root):
                    ok.append(f"{subject} r{run}")
                else:
                    missing.append((subject, run))

        total = len(ok) + len(missing)
        if missing:
            any_failures = True
            status = f"INCOMPLETE  ({len(missing)}/{total} recordings missing output)"
        else:
            status = f"OK  ({total}/{total} recordings complete)"

        print(f"\n{'─' * W}")
        print(f"  Step {step_id}  [{label}]")
        print(f"  Status : {status}")
        if na_count:
            print(f"  N/A    : {na_count} recordings (non-EP subjects, skipped by design)")
        if missing:
            # Group runs by subject for compact display
            by_subj: dict[str, list[str]] = {}
            for s, r in missing:
                by_subj.setdefault(s, []).append(str(r))
            print("  Missing recordings:")
            for s in sorted(by_subj):
                print(f"    {s}  runs: {', '.join(by_subj[s])}")

    # ── Log scan ──────────────────────────────────────────────────────────────
    print(f"\n{'=' * W}")
    print("  LOG ERRORS")
    print(f"{'─' * W}")
    if not log_dir.exists():
        print(f"  Log directory not found: {log_dir}")
        print(f"  (Run submit_all.sh first, or pass --logs PATH)")
    else:
        hits = scan_logs(log_dir)
        if hits:
            any_failures = True
            print(f"  {len(hits)} traceback(s) found across {len(set(f for f, _ in hits))} log file(s):\n")
            for fname, ctx in hits:
                print(f"  [{fname}]")
                print(f"    context : {ctx}")
        else:
            print("  No tracebacks found in log files.")

    # ── Overall verdict ───────────────────────────────────────────────────────
    print(f"\n{'=' * W}")
    if any_failures:
        print("  OVERALL: FAILURES DETECTED — review missing recordings and log errors above")
    else:
        print("  OVERALL: ALL STEPS COMPLETE — no missing output files or log errors")
    print(f"{'=' * W}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pipeline execution report: check output files and scan logs for errors."
    )
    parser.add_argument(
        "--root",
        default=str(SCRIPT_DIR / "processed"),
        help="Path to processed output directory (default: %(default)s)",
    )
    parser.add_argument(
        "--logs",
        default=str(SCRIPT_DIR / "slurm" / "logs"),
        help="Path to SLURM log directory (default: %(default)s)",
    )
    args = parser.parse_args()
    print_report(Path(args.root), Path(args.logs))
