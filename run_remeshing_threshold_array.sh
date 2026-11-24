#!/bin/bash
#SBATCH --job-name=remeshJ
#SBATCH --array=0-3%4
#SBATCH --output=output_%A_%a.txt
#SBATCH --error=error_%A_%a.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=48:00:00
#SBATCH --mem=32G
#SBATCH --partition=tue.default.q
#SBATCH --no-requeue
#SBATCH --open-mode=append

set -euo pipefail

# ======================================================
# 1. LOAD MODULES AND PYTHON ENVIRONMENT
# ======================================================

module purge
module load Umbrella/2024
module load Python/3.10.13-GCCcore-11.3.0
module load imkl/2025.1.0
module load libGLU/9.0.2-GCCcore-11.3.0

source ~/myenv/bin/activate

# ======================================================
# 2. PROJECT DIRECTORY
# ======================================================

PROJECT_DIR="/scratch-shared/20236379/Remeshing"
MODEL_SOURCE="$PROJECT_DIR/cross_slot_remesh.py"
GEO_SOURCE="$PROJECT_DIR/crosschannel_2.geo"

[[ -f "$MODEL_SOURCE" ]] || { echo "Missing $MODEL_SOURCE" >&2; exit 1; }
[[ -f "$GEO_SOURCE" ]] || { echo "Missing $GEO_SOURCE" >&2; exit 1; }

# ======================================================
# 3. REMESHING THRESHOLD STUDY PARAMETERS
# ======================================================

# All other model parameters, including mesh size, timestep,
# predictor_J_safety and remesh_interval_s, are unchanged.
THRESHOLDS=(
    0.60
)

LABELS=(
    J0p60
)

IDX="${SLURM_ARRAY_TASK_ID:?Submit this script using sbatch, not bash}"
THRESHOLD="${THRESHOLDS[$IDX]}"
LABEL="${LABELS[$IDX]}"

# ======================================================
# 4. CREATE SEPARATE OUTPUT DIRECTORY
# ======================================================

SWEEP_DIR="$PROJECT_DIR/remeshing_threshold_runs/$SLURM_ARRAY_JOB_ID"
RUN_DIR="$SWEEP_DIR/$LABEL"
RESULTS_DIR="$SWEEP_DIR/results"

mkdir -p "$RUN_DIR" "$RESULTS_DIR"
cd "$RUN_DIR"

printf '%s\n' \
    '=======================================' \
    'Remeshing threshold independence study' \
    "Threshold: $THRESHOLD" \
    "Output directory: $RUN_DIR" \
    "Slurm job: $SLURM_JOB_ID" \
    '======================================='

# ======================================================
# 5. CHECK NUTILS
# ======================================================

python -c 'import nutils; print("Nutils version:", nutils.__version__)'

# ======================================================
# 6. MATRIX SOLVER AND THREAD SETTINGS
# ======================================================

export NUTILS_MATRIX=mkl
export NUTILS_NPROCS=8

export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# ======================================================
# 7. PREPARE CASE-SPECIFIC MODEL
# ======================================================

# cross_slot_remesh.py currently defines remesh_J_threshold inside main.
# Patch only this assignment in a copy, leaving the original untouched.
cp "$MODEL_SOURCE" "$RUN_DIR/cross_slot_remesh.py"
cp "$GEO_SOURCE" "$RUN_DIR/crosschannel_2.geo"

python - "$RUN_DIR/cross_slot_remesh.py" "$THRESHOLD" <<'PY'
from pathlib import Path
import re
import sys

path = Path(sys.argv[1])
threshold = sys.argv[2]
source = path.read_text(encoding='utf-8')

pattern = r'(?m)^([ \t]*remesh_J_threshold[ \t]*=[ \t]*)([^\n#]*)([ \t]*(?:\#.*)?)$'
updated, count = re.subn(
    pattern,
    lambda m: m.group(1) + threshold + m.group(3),
    source,
)
if count != 1:
    raise SystemExit(
        f'Expected exactly one remesh_J_threshold assignment, found {count}. '
        'Check the Python model.'
    )

path.write_text(updated, encoding='utf-8')
print(f'Prepared {path.name} with remesh_J_threshold = {threshold}')
PY

# ======================================================
# 8. RUN SIMULATION
# ======================================================

srun python -u "$RUN_DIR/cross_slot_remesh.py" main 2>&1 | tee "$RUN_DIR/simulation.log"

# ======================================================
# 9. SAVE CSV WITH UNIQUE THRESHOLD LABEL
# ======================================================

shopt -s nullglob
csv_files=("$RUN_DIR"/D__h*um_c*.csv)
if [[ ${#csv_files[@]} -ne 1 ]]; then
    echo "Expected one D__h*um_c*.csv in $RUN_DIR; found ${#csv_files[@]}" >&2
    exit 1
fi

csv_file="${csv_files[0]}"
csv_basename="$(basename "$csv_file" .csv)"
cp "$csv_file" "$RESULTS_DIR/${csv_basename}_${LABEL}.csv"

echo "Finished $LABEL"
echo "Consolidated CSV: $RESULTS_DIR/${csv_basename}_${LABEL}.csv"
