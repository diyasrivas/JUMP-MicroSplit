#!/bin/bash
# Build cpg0029-chroma-pilot training dataset (Step 1).
#
# Creates a single dataset from 7 fixed plates (U2OS cells).
#
# Usage:
#   ./datasets.sh                    # run interactively
#   ./datasets.sh --slurm            # submit as a SLURM job
#   ./datasets.sh --samples 3500 --slurm
#
# Options:
#   --data_dir DIR     Root of chroma-pilot data (default: /project/cell_paint_mono/cpg0029-chroma-pilot/images)
#   --output_dir DIR   Where to write the dataset (default: /project/cell_paint_mono/training_datasets/training_dataset_cpg0029)
#   --samples N        Target training FOVs (default: 3500)
#   --holdout_frac F   Fraction of wells held out per plate (default: 0.3)
#   --seed N           Random seed (default: 42)
#   --slurm            Submit as a SLURM job instead of running interactively
#   --partition NAME   SLURM partition (default: cpu)
#   --time HH:MM:SS    SLURM walltime (default: 8:00:00)
#   --mem XG           SLURM memory (default: 32G)

set -euo pipefail

SUBMIT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PYTHON_SCRIPT="${SUBMIT_DIR}/1_datasets.py"

DATA_DIR="/project/cell_paint_mono/cpg0029-chroma-pilot/images"
OUTPUT_DIR="/project/cell_paint_mono/training_datasets/training_dataset_cpg0029"
SAMPLES=3500
HOLDOUT_FRAC=0.3
SEED=42
SLURM_MODE=false

SLURM_PARTITION="cpu"
SLURM_TIME="8:00:00"
SLURM_MEM="32G"
SLURM_CPUS=4

while [[ $# -gt 0 ]]; do
    case $1 in
        --data_dir)     DATA_DIR="$2";    shift 2 ;;
        --output_dir)   OUTPUT_DIR="$2";  shift 2 ;;
        --samples)      SAMPLES="$2";     shift 2 ;;
        --holdout_frac) HOLDOUT_FRAC="$2"; shift 2 ;;
        --seed)         SEED="$2";        shift 2 ;;
        --slurm)        SLURM_MODE=true;  shift ;;
        --partition)    SLURM_PARTITION="$2"; shift 2 ;;
        --time)         SLURM_TIME="$2";  shift 2 ;;
        --mem)          SLURM_MEM="$2";   shift 2 ;;
        --help)
            sed -n '2,22p' "$0" | sed 's/^# \{0,1\}//'
            exit 0 ;;
        *) echo "ERROR: Unknown option: $1" >&2; exit 1 ;;
    esac
done

_activate_conda() {
    for CONDA_PATH in \
        "/localscratch/mambaforge/etc/profile.d/conda.sh" \
        "${HOME}/mambaforge/etc/profile.d/conda.sh" \
        "${HOME}/miniforge3/etc/profile.d/conda.sh" \
        "${HOME}/miniconda3/etc/profile.d/conda.sh" \
        "${HOME}/anaconda3/etc/profile.d/conda.sh"; do
        if [ -f "${CONDA_PATH}" ]; then
            source "${CONDA_PATH}"
            conda activate microsplit_jobs || exit 1
            export PYTHONPATH="/home/diya.srivastava/Desktop/repos/JUMP-MicroSplit/src:${PYTHONPATH:-}"
            return
        fi
    done
    echo "ERROR: Conda not found" >&2; exit 1
}

echo "================================================================================" >&2
echo "cpg0029 datasets | samples=${SAMPLES} | holdout=${HOLDOUT_FRAC}" >&2
echo "  data_dir:   ${DATA_DIR}" >&2
echo "  output_dir: ${OUTPUT_DIR}" >&2
echo "================================================================================" >&2

if [[ "$SLURM_MODE" == false ]]; then
    _activate_conda
    python "${PYTHON_SCRIPT}" \
        --data_dir      "${DATA_DIR}" \
        --output_dir    "${OUTPUT_DIR}" \
        --samples       "${SAMPLES}" \
        --holdout_frac  "${HOLDOUT_FRAC}" \
        --seed          "${SEED}"
    exit 0
fi

# ---- SLURM submission ----
LOG_DIR="${OUTPUT_DIR}/logs"
mkdir -p "${LOG_DIR}"

JOB_ID=$(sbatch --parsable <<EOF
#!/bin/bash
#SBATCH --job-name=cpg0029_datasets
#SBATCH --output=${LOG_DIR}/datasets_%j.log
#SBATCH --error=${LOG_DIR}/datasets_%j.log
#SBATCH --partition=${SLURM_PARTITION}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${SLURM_CPUS}
#SBATCH --mem=${SLURM_MEM}
#SBATCH --time=${SLURM_TIME}

echo "Job \${SLURM_JOB_ID}: cpg0029 datasets | Node: \${SLURMD_NODENAME} | Start: \$(date)"

$(declare -f _activate_conda)
_activate_conda

python "${PYTHON_SCRIPT}" \
    --data_dir      "${DATA_DIR}" \
    --output_dir    "${OUTPUT_DIR}" \
    --samples       "${SAMPLES}" \
    --holdout_frac  "${HOLDOUT_FRAC}" \
    --seed          "${SEED}"

EXIT_CODE=\$?
echo "End: \$(date) | Exit: \$EXIT_CODE"
exit \$EXIT_CODE
EOF
)

echo "Submitted job ${JOB_ID}: cpg0029 datasets" >&2
echo "${JOB_ID}"
