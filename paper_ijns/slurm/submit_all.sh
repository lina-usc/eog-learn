#!/usr/bin/env bash
# =============================================================================
# submit_all.sh — Submit the full paper_ijns analysis pipeline to SLURM
#
# Usage
# -----
#   bash paper_ijns/slurm/submit_all.sh                          # submit everything
#   bash paper_ijns/slurm/submit_all.sh --only lstm_as           # one step only
#   bash paper_ijns/slurm/submit_all.sh --only lstm_pr lstm_ps   # multiple steps
#
# Valid --only keys:
#   lstm_pr   LSTM perrecording   (Step 1.1)
#   lstm_ps   LSTM persubject     (Step 1.1)
#   lstm_as   LSTM acrosssubject  (Step 1.1)
#   ica       ICA + ICLabel       (Step 1.2)
#   simnibs   SimNIBS fwd model   (Step 1.3)
#   sim       Biophysical sim     (Step 1.4)
#   xarray    Xarray aggregation  (Step 5)
#   analysis  Analysis notebooks  (Steps 6–9)
#
# All cluster-specific settings (partition, environment, output path, etc.) are
# read from slurm/config.yml — edit that file before submitting.
#
# Dependency graph
# ----------------
#   Steps 1.1 (perrecording)  ─┐
#   Steps 1.1 (persubject)    ─┤
#   Steps 1.1 (acrosssubject) ─┤── no inter-step deps; run concurrently
#   Steps 1.2 (ICA)           ─┤
#   Step  3   (SimNIBS)  ──────┘──→ Step 1.4 (Simulation)
#             │                           │
#             └────── (all five) ─────────┘
#                           ↓  afterany all five (runs even if some subjects failed)
#                    Step 5 (xarray aggregation)
#                           ↓  afterok
#                    Steps 6–9 (analysis notebooks, chained)
# =============================================================================
set -euo pipefail

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
ONLY=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --only)
            shift
            while [[ $# -gt 0 && "$1" != --* ]]; do
                ONLY+=("$1")
                shift
            done
            ;;
        *)
            echo "Unknown argument: $1" >&2
            echo "Usage: $0 [--only KEY [KEY ...]]" >&2
            echo "Keys: lstm_pr lstm_ps lstm_as ica simnibs sim xarray analysis" >&2
            exit 1
            ;;
    esac
done

# Returns 0 (true) if ONLY is empty (run all) or contains the given key
_should_run() {
    [[ ${#ONLY[@]} -eq 0 ]] && return 0
    local key="$1" k
    for k in "${ONLY[@]}"; do
        [[ "$k" == "$key" ]] && return 0
    done
    return 1
}

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)   # paper_ijns/
SLURM_DIR="$SCRIPT_DIR/slurm"
mkdir -p "$SLURM_DIR/logs"

# ---------------------------------------------------------------------------
# Read all settings from config.yml (parsed via Python; no yq required)
# ---------------------------------------------------------------------------
_cfg() {
    python3 -c "
import yaml, sys
c = yaml.safe_load(open('$SLURM_DIR/config.yml'))
val = c.get('$1', '')
print(str(val))
"
}

PARTITION=$(_cfg partition)
MODULE=$(_cfg module)
VENV=$(_cfg venv)
SIMNIBS_ENV=$(_cfg simnibs_env)
ROOT=$(_cfg root)
[[ "$(_cfg recompute)" == "False" ]] && RECOMPUTE_FLAG="--no-recompute" || RECOMPUTE_FLAG=""

# Build --partition flag only when a partition is configured
[[ -n "$PARTITION" ]] && _PARTITION_ARG="--partition=$PARTITION" || _PARTITION_ARG=""

echo "============================================================"
echo "  paper_ijns SLURM pipeline submission"
echo "============================================================"
echo "  Partition  : ${PARTITION:-<default>}"
echo "  Module     : $MODULE"
echo "  Venv       : $VENV"
echo "  Root       : $ROOT"
echo "  Recompute  : $([[ -z "$RECOMPUTE_FLAG" ]] && echo yes || echo no)"
[[ ${#ONLY[@]} -gt 0 ]] && echo "  Only       : ${ONLY[*]}"
echo "============================================================"

# ---------------------------------------------------------------------------
# Generate subjects.txt (one subject ID per line)
# ---------------------------------------------------------------------------
echo ""
echo "Generating subjects.txt ..."
module load "$MODULE"
source "$VENV/bin/activate"
python3 - <<PYEOF > "$SLURM_DIR/subjects.txt"
import sys
sys.path.insert(0, "$SCRIPT_DIR")
import eoglearn
subjects = sorted(s for s in eoglearn.datasets.eegeyenet.get_subjects_runs().keys()
                  if s != "AA0")
print('\n'.join(subjects))
PYEOF
deactivate 2>/dev/null || true

N=$(wc -l < "$SLURM_DIR/subjects.txt")
ARRAY="0-$((N - 1))"
echo "  Found $N subjects → array range $ARRAY"
echo ""

# Common --export base (each sbatch gets extra vars appended)
_COMMON="MODULE=$MODULE,VENV=$VENV,SCRIPT_DIR=$SCRIPT_DIR"

# Initialise all job-ID variables so they're always defined
JID_PR="" JID_PS="" JID_AS="" JID_ICA="" JID_SIMNIBS="" JID_SIM="" JID_XR="" JID_ANALYSIS=""

# ---------------------------------------------------------------------------
# Steps 1.1 × 3  +  1.2 — submitted with no inter-step dependencies
# ---------------------------------------------------------------------------
if _should_run lstm_pr; then
    JID_PR=$(sbatch --parsable \
        --array="$ARRAY" \
        --job-name="eog_lstm_pr" \
        --export=ALL,CONDITION=perrecording,ROOT="$ROOT",RECOMPUTE_FLAG="$RECOMPUTE_FLAG",$_COMMON \
        ${_PARTITION_ARG:+"$_PARTITION_ARG"} \
        --cpus-per-task=5 --mem=8G --time=4:00:00 \
        "$SLURM_DIR/01_lstm.sbatch")
fi

if _should_run lstm_ps; then
    JID_PS=$(sbatch --parsable \
        --array="$ARRAY" \
        --job-name="eog_lstm_ps" \
        --export=ALL,CONDITION=persubject,ROOT="$ROOT",RECOMPUTE_FLAG="$RECOMPUTE_FLAG",$_COMMON \
        ${_PARTITION_ARG:+"$_PARTITION_ARG"} \
        --cpus-per-task=2 --mem=16G --time=16:00:00 \
        "$SLURM_DIR/01_lstm.sbatch")
fi

if _should_run lstm_as; then
    JID_AS=$(sbatch --parsable \
        --array="$ARRAY" \
        --job-name="eog_lstm_as" \
        --export=ALL,CONDITION=acrosssubject,ROOT="$ROOT",RECOMPUTE_FLAG="$RECOMPUTE_FLAG",$_COMMON \
        ${_PARTITION_ARG:+"$_PARTITION_ARG"} \
        --cpus-per-task=1 --mem=24G --time=48:00:00 \
        "$SLURM_DIR/01_lstm.sbatch")
fi

if _should_run ica; then
    JID_ICA=$(sbatch --parsable \
        --array="$ARRAY" \
        --job-name="eog_ica" \
        --export=ALL,ROOT="$ROOT",RECOMPUTE_FLAG="$RECOMPUTE_FLAG",$_COMMON \
        ${_PARTITION_ARG:+"$_PARTITION_ARG"} \
        --cpus-per-task=5 --mem=12G --time=8:00:00 \
        "$SLURM_DIR/02_ica.sbatch")
fi

# ---------------------------------------------------------------------------
# Step 3 — Biophysical forward model (one-time; skip if leadfield exists)
# ---------------------------------------------------------------------------
SIM_DEP=""
if _should_run simnibs; then
    LEADFIELD="$SCRIPT_DIR/leadfield_elect_only.npy"
    if [[ -f "$LEADFIELD" && -z "$RECOMPUTE_FLAG" ]]; then
        echo "Skipping Step 3: $LEADFIELD already exists."
    else
        JID_SIMNIBS=$(sbatch --parsable \
            --job-name="eog_simnibs" \
            --export=ALL,SIMNIBS_ENV="$SIMNIBS_ENV",$_COMMON \
            ${_PARTITION_ARG:+"$_PARTITION_ARG"} \
            --cpus-per-task=1 --mem=4G --time=4:00:00 \
            "$SLURM_DIR/03_simnibs.sbatch")
        SIM_DEP="afterok:$JID_SIMNIBS"
        printf "  Step 3 — SimNIBS fwd model : %s\n" "$JID_SIMNIBS"
    fi
fi

# ---------------------------------------------------------------------------
# Step 4 — Biophysical simulation (depends on Step 3 if submitted)
# ---------------------------------------------------------------------------
if _should_run sim; then
    SIM_ARGS=(--parsable
        --array="$ARRAY"
        --job-name="eog_sim"
        --export=ALL,ROOT="$ROOT",RECOMPUTE_FLAG="$RECOMPUTE_FLAG",$_COMMON
        ${_PARTITION_ARG:+"$_PARTITION_ARG"}
        --cpus-per-task=2 --mem=16G --time=8:00:00)
    [[ -n "$SIM_DEP" ]] && SIM_ARGS+=(--dependency="$SIM_DEP")
    JID_SIM=$(sbatch "${SIM_ARGS[@]}" "$SLURM_DIR/04_sim.sbatch")
fi

# ---------------------------------------------------------------------------
# Step 5 — Aggregate to xarray (waits for all submitted preceding jobs)
# ---------------------------------------------------------------------------
if _should_run xarray; then
    XARRAY_DEP_JIDS=()
    for _jid in "$JID_PR" "$JID_PS" "$JID_AS" "$JID_ICA" "$JID_SIM"; do
        [[ -n "$_jid" ]] && XARRAY_DEP_JIDS+=("$_jid")
    done
    XARRAY_ARGS=(--parsable
        --job-name="eog_xarray"
        --export=ALL,ROOT="$ROOT",$_COMMON
        ${_PARTITION_ARG:+"$_PARTITION_ARG"})
    if [[ ${#XARRAY_DEP_JIDS[@]} -gt 0 ]]; then
        XARRAY_ARGS+=(--dependency="afterany:$(IFS=:; echo "${XARRAY_DEP_JIDS[*]}")")
    fi
    JID_XR=$(sbatch "${XARRAY_ARGS[@]}" "$SLURM_DIR/05_xarray.sbatch")
fi

# ---------------------------------------------------------------------------
# Steps 6–9 — Analysis notebooks (sequential chain)
# ---------------------------------------------------------------------------
if _should_run analysis; then
    ANALYSIS_ARGS=(--parsable
        --job-name="eog_analysis"
        --export=ALL,ROOT="$ROOT",$_COMMON
        ${_PARTITION_ARG:+"$_PARTITION_ARG"})
    # Depend on xarray only if it was submitted in this run
    [[ -n "$JID_XR" ]] && ANALYSIS_ARGS+=(--dependency="afterok:$JID_XR")
    JID_ANALYSIS=$(sbatch "${ANALYSIS_ARGS[@]}" "$SLURM_DIR/06_analysis.sbatch")
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "Submitted pipeline:"
[[ -n "$JID_PR"       ]] && printf "  %-30s %s\n" "LSTM perrecording:"    "$JID_PR"
[[ -n "$JID_PS"       ]] && printf "  %-30s %s\n" "LSTM persubject:"      "$JID_PS"
[[ -n "$JID_AS"       ]] && printf "  %-30s %s\n" "LSTM acrosssubject:"   "$JID_AS"
[[ -n "$JID_ICA"      ]] && printf "  %-30s %s\n" "ICA + ICLabel:"        "$JID_ICA"
[[ -n "$JID_SIMNIBS"  ]] && printf "  %-30s %s\n" "SimNIBS fwd model:"    "$JID_SIMNIBS"
[[ -n "$JID_SIM"      ]] && printf "  %-30s %s\n" "Biophysical sim:"      "$JID_SIM"
[[ -n "$JID_XR"       ]] && printf "  %-30s %s\n" "Xarray aggregation:"   "$JID_XR"
[[ -n "$JID_ANALYSIS" ]] && printf "  %-30s %s\n" "Analysis (Steps 6–9):" "$JID_ANALYSIS"
echo ""
echo "Monitor with:  squeue -u \$USER"
echo "Logs in:       $SLURM_DIR/logs/"
echo ""

# Save/merge job IDs so that status.py can query them without needing manual input.
# When --only is used, existing entries for other jobs are preserved.
python3 -c "
import json
from pathlib import Path

path = Path('$SLURM_DIR/submitted_jobs.json')
existing = json.loads(path.read_text()) if path.exists() else {}

new_ids = {k: v for k, v in {
    'lstm_pr':  '${JID_PR}',
    'lstm_ps':  '${JID_PS}',
    'lstm_as':  '${JID_AS}',
    'ica':      '${JID_ICA}',
    'sim':      '${JID_SIM}',
    'simnibs':  '${JID_SIMNIBS}',
    'xarray':   '${JID_XR}',
    'analysis': '${JID_ANALYSIS}',
}.items() if v}

existing.update(new_ids)
path.write_text(json.dumps(existing, indent=2))
print('  Job IDs saved to $SLURM_DIR/submitted_jobs.json')
print('  Live status:  python paper_ijns/slurm/status.py')
"
