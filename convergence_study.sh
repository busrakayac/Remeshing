#!/bin/bash
#SBATCH --job-name=convergence
#SBATCH --array=0-1%2
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
# 3. CONVERGENCE STUDY PARAMETERS
# ======================================================

# Each index corresponds to one simulation.

MESH_SIZES=(
    0.000010m
    0.000012m
    0.000012m
)

TIME_STEPS=(
    0.01ms
    0.005ms
    0.0025ms
)

LABELS=(
    h10um_dt0p01ms
    h12um_dt0p005ms
    h12um_dt0p0025ms
)

IDX=$SLURM_ARRAY_TASK_ID

export MESH_SIZE="${MESH_SIZES[$IDX]}"
export TIME_STEP="${TIME_STEPS[$IDX]}"

LABEL="${LABELS[$IDX]}"

# ======================================================
# 4. CREATE SEPARATE OUTPUT DIRECTORY
# ======================================================

RUN_DIR="$PROJECT_DIR/convergence_runs/${SLURM_ARRAY_JOB_ID}/$LABEL"

mkdir -p "$RUN_DIR"
cd "$RUN_DIR"

echo "======================================="
echo "Convergence study"
echo "======================================="
echo "Mesh size: $MESH_SIZE"
echo "Timestep: $TIME_STEP"
echo "Output directory: $RUN_DIR"
echo "Slurm job: $SLURM_JOB_ID"
echo "======================================="

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
