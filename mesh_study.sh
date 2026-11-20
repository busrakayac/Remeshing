
#!/bin/bash
#SBATCH --job-name=mesh_study
#SBATCH --array=0-3%2
#SBATCH --output=mesh_%A_%a.out
#SBATCH --error=mesh_%A_%a.err

# Keep your existing Slurm resource settings here:
# partition, time limit, memory, CPUs, etc.

set -euo pipefail

REPO="$SLURM_SUBMIT_DIR"

# Mesh sizes corresponding to array indices 0, 1, 2, 3
MESH_SIZES=(0.000012m 0.000014m 0.000016m 0.000018m)
LABELS=(12um 14um 16um 18um)

export MESH_SIZE="${MESH_SIZES[$SLURM_ARRAY_TASK_ID]}"
LABEL="${LABELS[$SLURM_ARRAY_TASK_ID]}"

# Activate your Python environment
source /home/20236379/myenv/bin/activate

# Separate results for each mesh
RUN_DIR="$REPO/mesh_runs/$LABEL"
mkdir -p "$RUN_DIR"
cd "$RUN_DIR"

echo "Running mesh size: $MESH_SIZE"
echo "Working directory: $RUN_DIR"

# Adjust the filename to match your Python script
python "$REPO/cross_slot_remesh.py" main
