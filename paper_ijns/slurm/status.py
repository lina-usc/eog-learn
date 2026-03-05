#!/usr/bin/env python3
"""status.py — Live pipeline status monitor.

Queries SLURM for the current status of pipeline jobs submitted by submit_all.sh
and shows a per-step summary: how many array tasks are pending, running,
completed, or failed, along with which subjects are still running or failed.

Usage
-----
    python paper_ijns/slurm/status.py                # reads submitted_jobs.json
    python paper_ijns/slurm/status.py --watch        # refresh every 30 s
    python paper_ijns/slurm/status.py --watch 60     # refresh every 60 s
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
import argparse
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent        # paper_ijns/slurm/
JOBS_FILE = SCRIPT_DIR / "submitted_jobs.json"
SUBJECTS_FILE = SCRIPT_DIR / "subjects.txt"

# Map job key → human label
JOB_LABELS: dict[str, str] = {
    "lstm_pr":  "LSTM perrecording  (Step 1.1)",
    "lstm_ps":  "LSTM persubject    (Step 1.1)",
    "lstm_as":  "LSTM acrosssubject (Step 1.1)",
    "ica":      "ICA + ICLabel      (Step 1.2)",
    "simnibs":  "SimNIBS fwd model  (Step 1.3)",
    "sim":      "Biophysical sim    (Step 1.4)",
    "xarray":   "Xarray aggregation (Step 5)",
    "analysis": "Analysis (Steps 6–9)",
}

# Which jobs are array jobs (have individual tasks per subject)
ARRAY_JOBS = {"lstm_pr", "lstm_ps", "lstm_as", "ica", "sim"}


def load_jobs(path: Path) -> dict[str, str]:
    if not path.exists():
        print(f"ERROR: {path} not found.")
        print("Run submit_all.sh first, or check the path.")
        sys.exit(1)
    with open(path) as f:
        return json.load(f)


def load_subjects(path: Path) -> list[str]:
    if not path.exists():
        return []
    return path.read_text().splitlines()


def run_squeue(job_ids: list[str]) -> list[dict]:
    """Query squeue for a list of job IDs (array notation, e.g. 12345)."""
    if not job_ids:
        return []
    cmd = [
        "squeue",
        "--jobs=" + ",".join(job_ids),
        "--noheader",
        "--format=%i %j %T %r",   # jobid, name, state, reason
    ]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True)
    except subprocess.CalledProcessError:
        return []  # jobs may have aged out of the queue
    rows = []
    for line in out.splitlines():
        parts = line.split(None, 3)
        if len(parts) >= 3:
            rows.append({"jobid": parts[0], "name": parts[1],
                         "state": parts[2], "reason": parts[3] if len(parts) > 3 else ""})
    return rows


def run_sacct(job_ids: list[str]) -> list[dict]:
    """Query sacct for completed/failed tasks (not in squeue anymore)."""
    if not job_ids:
        return []
    cmd = [
        "sacct",
        "--jobs=" + ",".join(job_ids),
        "--noheader",
        "--parsable2",
        "--format=JobID,JobName,State,ExitCode",
    ]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []
    rows = []
    for line in out.splitlines():
        parts = line.split("|")
        if len(parts) >= 3 and "_" in parts[0]:  # array task rows have jobid_taskid
            rows.append({"jobid": parts[0], "name": parts[1],
                         "state": parts[2].split()[0], "exitcode": parts[3] if len(parts) > 3 else ""})
    return sacct_rows_to_unique(rows)


def sacct_rows_to_unique(rows: list[dict]) -> list[dict]:
    """Keep one row per task — prefer terminal states over intermediate."""
    order = {"COMPLETED": 0, "FAILED": 1, "CANCELLED": 2, "TIMEOUT": 3,
             "RUNNING": 4, "PENDING": 5}
    best: dict[str, dict] = {}
    for row in rows:
        jid = row["jobid"]
        if jid not in best or order.get(row["state"], 9) < order.get(best[jid]["state"], 9):
            best[jid] = row
    return list(best.values())


def task_id_from_jobid(jobid: str) -> int | None:
    """Extract array task index from '12345_3' → 3."""
    if "_" in jobid:
        try:
            return int(jobid.rsplit("_", 1)[1])
        except ValueError:
            pass
    return None


def summarise_job(master_id: str, subjects: list[str],
                  squeue_rows: list[dict], sacct_rows: list[dict]) -> dict:
    """
    Return counts {pending, running, completed, failed, cancelled, unknown}
    and lists of subjects per non-ok state.
    """
    counts: dict[str, int] = {s: 0 for s in
                               ("pending", "running", "completed", "failed",
                                "cancelled", "timeout", "unknown")}
    running_subjects: list[str] = []
    failed_subjects: list[str] = []

    # Collect all task rows from both sources, keyed by task index
    task_states: dict[int, str] = {}

    for row in squeue_rows:
        jid = row["jobid"]
        if not jid.startswith(master_id):
            continue
        tid = task_id_from_jobid(jid)
        if tid is not None:
            task_states[tid] = row["state"]
        elif jid == master_id:
            # whole-array row (e.g. all pending): fill from squeue reason
            pass

    for row in sacct_rows:
        jid = row["jobid"]
        if not jid.startswith(master_id):
            continue
        tid = task_id_from_jobid(jid)
        if tid is not None and tid not in task_states:
            task_states[tid] = row["state"]

    # If no per-task rows found, try the whole-job rows from squeue.
    # SLURM may return a single compressed-range row (e.g. "12345_[0-30]")
    # rather than individual per-task rows when all tasks share the same state.
    if not task_states:
        whole = [r for r in squeue_rows
                 if r["jobid"].startswith(master_id)
                 and task_id_from_jobid(r["jobid"]) is None]
        if whole:
            state = whole[0]["state"]
            # Map to a per-task count (approximate — we don't know n)
            n = len(subjects) if subjects else 1
            for _ in range(n):
                state_key = state.lower() if state.lower() in counts else "unknown"
                counts[state_key] += 1
        return {"counts": counts, "running": [], "failed": []}

    for tid, state in task_states.items():
        subj = subjects[tid] if tid < len(subjects) else f"task_{tid}"
        state_lower = state.lower()
        if state_lower in ("completed",):
            counts["completed"] += 1
        elif state_lower in ("running",):
            counts["running"] += 1
            running_subjects.append(subj)
        elif state_lower in ("pending", "configuring", "requeued"):
            counts["pending"] += 1
        elif state_lower in ("failed", "node_fail", "out_of_memory"):
            counts["failed"] += 1
            failed_subjects.append(subj)
        elif state_lower in ("cancelled", "revoked"):
            counts["cancelled"] += 1
            failed_subjects.append(f"{subj} (cancelled)")
        elif state_lower in ("timeout",):
            counts["timeout"] += 1
            failed_subjects.append(f"{subj} (timeout)")
        else:
            counts["unknown"] += 1

    return {"counts": counts, "running": running_subjects, "failed": failed_subjects}


def print_status(jobs: dict[str, str], subjects: list[str]) -> None:
    all_ids = list(jobs.values())
    squeue_rows = run_squeue(all_ids)
    sacct_rows = run_sacct(all_ids)

    W = 72
    print("=" * W)
    print("  PIPELINE STATUS")
    print("=" * W)

    for key, label in JOB_LABELS.items():
        master_id = jobs.get(key)
        if not master_id:
            continue

        is_array = key in ARRAY_JOBS
        n_total = len(subjects) if (is_array and subjects) else 1

        if is_array:
            summary = summarise_job(master_id, subjects, squeue_rows, sacct_rows)
            c = summary["counts"]
            done = c["completed"]
            run = c["running"]
            pend = c["pending"]
            fail = c["failed"] + c["cancelled"] + c["timeout"]
            total_seen = done + run + pend + fail + c["unknown"]

            # Derive a status indicator.
            # Mirror the Python script's 80 % threshold: a step is considered
            # done when ≥80 % of subjects completed (≤20 % may have failed).
            if run > 0 or pend > 0:
                indicator = "⋯ RUNNING"
            elif total_seen == 0:
                indicator = "? NOT STARTED"
            elif n_total > 0 and fail >= 0.2 * n_total:
                indicator = "✗ FAILURES"
            elif n_total > 0 and done >= 0.8 * n_total:
                indicator = "✓ DONE"
            else:
                indicator = "– PARTIAL"

            print(f"\n  {label}  [{master_id}]")
            print(f"  {indicator}  |  "
                  f"done={done}/{n_total}  running={run}  "
                  f"pending={pend}  failed={fail}")
            if summary["running"]:
                print(f"  Running : {', '.join(summary['running'][:8])}"
                      + (" ..." if len(summary["running"]) > 8 else ""))
            if summary["failed"]:
                print(f"  Failed  : {', '.join(summary['failed'][:8])}"
                      + (" ..." if len(summary["failed"]) > 8 else ""))
        else:
            # Single job
            sq = [r for r in squeue_rows if r["jobid"] == master_id]
            sa = [r for r in sacct_rows if r["jobid"].split("_")[0] == master_id]
            state = "NOT STARTED"
            if sq:
                state = sq[0]["state"]
            elif sa:
                state = sa[0]["state"]
            print(f"\n  {label}  [{master_id}]")
            print(f"  State: {state}")

    print(f"\n{'=' * W}")
    print("  Tip: run 'python paper_ijns/slurm/report.py' for a full output-file check")
    print(f"{'=' * W}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Live pipeline status: query SLURM for job progress."
    )
    parser.add_argument(
        "--jobs",
        default=str(JOBS_FILE),
        help="Path to submitted_jobs.json (default: %(default)s)",
    )
    parser.add_argument(
        "--subjects",
        default=str(SUBJECTS_FILE),
        help="Path to subjects.txt (default: %(default)s)",
    )
    parser.add_argument(
        "--watch",
        nargs="?",
        const=30,
        type=int,
        metavar="SECONDS",
        help="Refresh every N seconds (default: 30 when flag is given)",
    )
    args = parser.parse_args()

    jobs = load_jobs(Path(args.jobs))
    subjects = load_subjects(Path(args.subjects))

    if args.watch is not None:
        interval = args.watch
        try:
            while True:
                # Clear screen
                print("\033[2J\033[H", end="")
                print(f"  (refreshing every {interval}s — Ctrl+C to stop)\n")
                print_status(jobs, subjects)
                time.sleep(interval)
        except KeyboardInterrupt:
            print("\nStopped.")
    else:
        print_status(jobs, subjects)


if __name__ == "__main__":
    main()
