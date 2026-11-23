#!/usr/bin/env bash
# Run the same cross-slot FSI model at four remeshing-Jacobian thresholds.
#
# Local, sequential:
#   bash run_remeshing_thresholds.sh /path/to/cross_slot_remesh.py
#
# Slurm, one independent job per threshold:
#   sbatch --array=0-3 --time=04:00:00 --mem=16G run_remeshing_thresholds.sh /path/to/cross_slot_remesh.py
# Adapt --time/--mem to your cluster.  The TU Delft Umbrella Python
# environment and ~/myenv are activated automatically for actual runs.
#
# Smoke test (generate source snapshots, do NOT run FSI):
#   PREPARE_ONLY=1 bash run_remeshing_thresholds.sh /path/to/cross_slot_remesh.py
#
# The original Python file is never modified. Each run has an independent
# directory and a separate CSV, plus a copy with its threshold in the name.

#SBATCH --job-name=remeshJ
#SBATCH --output=remeshJ_%A_%a.out

set -Eeuo pipefail

THRESHOLDS=(0.50 0.60 0.70 0.80)
PREPARE_ONLY="${PREPARE_ONLY:-0}"

# Use the SAME Python environment as the successful cross-slot FSI jobs.
# Skip cluster modules for PREPARE_ONLY, so source generation can be tested
# without Nutils or the Slurm/Umbrella environment.
setup_python() {
    if [[ "$PREPARE_ONLY" == "1" ]]; then
        PYTHON_EXE="${PYTHON_EXE:-python}"
        return
    fi

    if ! type module >/dev/null 2>&1; then
        echo "ERROR: 'module' command is unavailable. Run this on the TU Delft cluster." >&2
        exit 1
    fi

    module purge
    module load Umbrella/2024
    module load Python/3.10.13-GCCcore-11.3.0
    module load imkl/2025.1.0
    module load libGLU/9.0.2-GCCcore-11.3.0

    [[ -f "$HOME/myenv/bin/activate" ]] || {
        echo "ERROR: Missing $HOME/myenv/bin/activate" >&2
        exit 1
    }
    # shellcheck disable=SC1091
    source "$HOME/myenv/bin/activate"

    # Respect an explicit override, otherwise lock to the activated venv.
    PYTHON_EXE="${PYTHON_EXE:-$(command -v python)}"
    echo "Using Python: $PYTHON_EXE"
    "$PYTHON_EXE" -c 'from nutils.mesh import gmsh; import nutils; print("Nutils:", nutils.__version__)' || {
        echo "ERROR: Nutils import failed. Check ~/myenv or PYTHON_EXE." >&2
        exit 1
    }
}

setup_python

# Source file defaults to the cross_slot_remesh.py beside this shell script.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MODEL="${1:-${SCRIPT_DIR}/cross_slot_remesh.py}"
[[ -f "$MODEL" ]] || { echo "ERROR: Python model not found: $MODEL" >&2; exit 1; }
MODEL_DIR="$(cd -- "$(dirname -- "$MODEL")" && pwd)"
MODEL="${MODEL_DIR}/$(basename -- "$MODEL")"
GEO="${MODEL_DIR}/crosschannel_2.geo"
[[ -f "$GEO" ]] || { echo "ERROR: Required geometry file not found: $GEO" >&2; exit 1; }

# Consistent output folder across both local runs and Slurm array jobs.
ROOT="${RUNS_ROOT:-${MODEL_DIR}/remeshing_threshold_sweep}"
mkdir -p -- "$ROOT/results"

run_case() {
    local threshold="$1"
    local tag="${threshold/./p}"
    local run_dir="${ROOT}/J${tag}"
    local model_copy="${run_dir}/cross_slot_remesh.py"
    local copied_geo="${run_dir}/crosschannel_2.geo"
    local status_file="${run_dir}/COMPLETED"
    mkdir -p -- "$run_dir"

    # Don't erase a completed run on an accidental repeated submission.
    if [[ -e "$status_file" && "$PREPARE_ONLY" != 1 ]]; then
        echo "[J=${threshold}] Already completed; skipping $run_dir"
        return 0
    fi

    # Patch a run-specific copy of the Python model; original is unchanged.
    "$PYTHON_EXE" - "$MODEL" "$model_copy" "$threshold" <<'PY'
from pathlib import Path
import re
import sys
src, dst, threshold = sys.argv[1:]
code = Path(src).read_text(encoding='utf-8')
for name, expected in [('predictor_J_safety', '0.4'), ('remesh_interval_s', 'None')]:
    check = rf'(?m)^[ \t]*{name}[ \t]*=[ \t]*{re.escape(expected)}(?:[ \t]|#|$)'
    if not re.search(check, code):
        raise SystemExit(f'ERROR: Expected {name} = {expected} in {src}. Check the model.')
pattern = r'(?m)^([ \t]*)remesh_J_threshold[ \t]*=[ \t]*[^\n#]*(?:[ \t]*#.*)?$'
code, count = re.subn(pattern,
    lambda m: f'{m.group(1)}remesh_J_threshold = {threshold}',
    code)
if count != 1:
    raise SystemExit(f'ERROR: Expected 1 remesh_J_threshold assignment; found {count}')
Path(dst).write_text(code, encoding='utf-8')
print(f'Prepared: {dst}, remesh_J_threshold = {threshold}')
PY

    # The model resolves crosschannel_2.geo relative to __file__.
    cp -- "$GEO" "$copied_geo"

    if [[ "$PREPARE_ONLY" == 1 ]]; then
        echo "[J=${threshold}] Preparation only, no FSI execution"
        return 0
    fi

    echo "[J=${threshold}] Starting model in ${run_dir}"
    (
        cd -- "$run_dir"
        # Keep the log from each run; failures propagate through the pipe.
        # Nutils uses cli.choose(main), so explicitly select the 'main' entry.
        # Slurm srun matches the existing successful FSI submission pattern.
        if [[ -n "${SLURM_JOB_ID:-}" ]]; then
            srun "$PYTHON_EXE" -u "$model_copy" main 2>&1 | tee simulation.log
        else
            "$PYTHON_EXE" -u "$model_copy" main 2>&1 | tee simulation.log
        fi
    )

    # CSV filenames are unmodified inside Python but unique by run directory.
    local csv_files=("$run_dir"/D__h*um_c*.csv)
    if [[ ${#csv_files[@]} -ne 1 || ! -f "${csv_files[0]}" ]]; then
        echo "ERROR: Expected exactly one particle CSV in $run_dir; found ${#csv_files[@]}" >&2
        exit 1
    fi
    local csv_name="$(basename -- "${csv_files[0]}")"
    local named_csv="${csv_name%.csv}_J${tag}.csv"
    cp -- "${csv_files[0]}" "$ROOT/results/$named_csv"
    printf 'threshold=%s\nparticle_csv=%s\n' "$threshold" "$named_csv" > "$status_file"
    echo "[J=${threshold}] COMPLETE. CSV: ${ROOT}/results/${named_csv}"
}

shopt -s nullglob
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    idx="$SLURM_ARRAY_TASK_ID"
    if ! [[ "$idx" =~ ^[0-3]$ ]]; then
        echo "ERROR: SLURM_ARRAY_TASK_ID must be 0, 1, 2, or 3; got $idx" >&2
        exit 1
    fi
    run_case "${THRESHOLDS[$idx]}"
else
    for threshold in "${THRESHOLDS[@]}"; do
        run_case "$threshold"
    done
fi

echo "Finished requested threshold cases. Output: $ROOT"
