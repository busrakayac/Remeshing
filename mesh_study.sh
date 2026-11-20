#!/bin/bash
#SBATCH --job-name=mesh_study
#SBATCH --array=0-3%2
#SBATCH --output=output_%A_%a.txt
#SBATCH --error=error_%A_%a.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
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

# ======================================================
# 3. MESH-INDEPENDENCE PARAMETERS
# ======================================================

MESH_SIZES=(0.000012m 0.000014m 0.000016m 0.000018m)
LABELS=(12um 14um 16um 18um)

export MESH_SIZE="${MESH_SIZES[$SLURM_ARRAY_TASK_ID]}"
LABEL="${LABELS[$SLURM_ARRAY_TASK_ID]}"

# ======================================================
# 4. CREATE SEPARATE OUTPUT DIRECTORY
# ======================================================

RUN_DIR="$PROJECT_DIR/mesh_runs/${SLURM_ARRAY_JOB_ID}/$LABEL"

mkdir -p "$RUN_DIR"
cd "$RUN_DIR"

echo "Mesh size: $MESH_SIZE"
echo "Output directory: $RUN_DIR"
echo "Slurm job: $SLURM_JOB_ID"

# ======================================================
# 5. CHECK NUTILS
# ======================================================

python -c "import nutils; print(nutils.__version__)"

# ======================================================
# 6. MATRIX SOLVER AND THREAD SETTINGS
# ======================================================

export NUTILS_MATRIX=mkl

export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

export NUTILS_NPROCS=1

# ======================================================
# 7. RUN SIMULATION
# ======================================================

srun python -u "$PROJECT_DIR/cross_slot_remesh.py" main
