#!/bin/bash
# Build cpg0000-jump-pilot training dataset (Step 1).
#
# Creates one dataset per cell line (A549, U2OS) by stratified sampling.
#
# Usage:
#   ./datasets.sh                          # run interactively (both cell lines)
#   ./datasets.sh --cell_lines "A549"      # single cell line
#   ./datasets.sh --slurm                  # submit as SLURM jobs
#   ./datasets.sh --samples 3500 --slurm
#
# Options:
#   --cell_lines "A549 U2OS"  Space-separated list (default: A549 U2OS)
#   --samples N               Target FOVs per cell line (default: 3500)
#   --data_dir DIR            Root of pilot data (default: /project/cell_paint_mono/cpg0000-jump-pilot)
#   --output_base DIR         Where to write datasets (default: /project/cell_paint_mono/training_datasets)
#   --metadata_file FILE      experiment-metadata.tsv (default: experiment-metadata.tsv in this dir)
#   --seed N                  Random seed (default: 42)
#   --slurm                   Submit as SLURM job(s) instead of running interactively
#   --partition NAME          SLURM partition (default: cpu)
#   --time HH:MM:SS           SLURM walltime (default: 8:00:00)
#   --mem XG                  SLURM memory (default: 32G)

set -euo pipefail

SUBMIT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PYTHON_SCRIPT="${SUBMIT_DIR}/1_datasets.py"
METADATA_FILE="${SUBMIT_DIR}/experiment-metadata.tsv"

CELL_LINES=("A549" "U2OS")
SAMPLES=3500
DATA_DIR="/project/cell_paint_mono/cpg0000-jump-pilot"
OUTPUT_BASE="/project/cell_paint_mono/training_datasets"
SEED=42
SLURM_MODE=false

SLURM_PARTITION="cpu"
SLURM_TIME="8:00:00"
SLURM_MEM="32G"
SLURM_CPUS=4

while [[ $# -gt 0 ]]; do
    case $1 in
        --cell_lines)    read -ra CELL_LINES <<< "$2"; shift 2 ;;
        --samples)       SAMPLES="$2";        shift 2 ;;
        --data_dir)      DATA_DIR="$2";       shift 2 ;;
        --output_base)   OUTPUT_BASE="$2";    shift 2 ;;
        --metadata_file) METADATA_FILE="$2";  shift 2 ;;
        --seed)          SEED="$2";           shift 2 ;;
        --slurm)         SLURM_MODE=true;     shift ;;
        --partition)     SLURM_PARTITION="$2"; shift 2 ;;
        --time)          SLURM_TIME="$2";     shift 2 ;;
        --mem)           SLURM_MEM="$2";      shift 2 ;;
        --help)
            sed -n '2,22p' "$0" | sed 's/^# \{0,1\}//'
            exit 0 ;;
        *) echo "ERROR: Unknown option: $1" >&2; exit 1 ;;
    esac
done

_run_python() {
    local CELL_LINE="$1"
    local OUT_DIR="${OUTPUT_BASE}/training_dataset_${CELL_LINE}"
    python "${PYTHON_SCRIPT}" \
        --cell_line "${CELL_LINE}" \
        --samples   "${SAMPLES}" \
        --data_dir  "${DATA_DIR}" \
        --output_dir "${OUT_DIR}" \
        --metadata_file "${METADATA_FILE}" \
        --seed "${SEED}"
}

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
echo "cpg0000 datasets | cell_lines=${CELL_LINES[*]} | samples=${SAMPLES}" >&2
echo "================================================================================" >&2

if [[ "$SLURM_MODE" == false ]]; then
    _activate_conda
    for CL in "${CELL_LINES[@]}"; do
        echo "Building dataset for ${CL} ..." >&2
        _run_python "${CL}"
    done
    exit 0
fi

# ---- SLURM submission ----
SUBMITTED=()
for CL in "${CELL_LINES[@]}"; do
    OUT_DIR="${OUTPUT_BASE}/training_dataset_${CL}"
    LOG_DIR="${OUT_DIR}/logs"
    mkdir -p "${LOG_DIR}"

    JOB_ID=$(sbatch --parsable <<EOF
#!/bin/bash
#SBATCH --job-name=cpg0000_datasets_${CL}
#SBATCH --output=${LOG_DIR}/datasets_${CL}_%j.log
#SBATCH --error=${LOG_DIR}/datasets_${CL}_%j.log
#SBATCH --partition=${SLURM_PARTITION}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${SLURM_CPUS}
#SBATCH --mem=${SLURM_MEM}
#SBATCH --time=${SLURM_TIME}

echo "Job \${SLURM_JOB_ID}: ${CL} datasets | Node: \${SLURMD_NODENAME} | Start: \$(date)"

$(declare -f _activate_conda)
_activate_conda

python "${PYTHON_SCRIPT}" \
    --cell_line "${CL}" \
    --samples   "${SAMPLES}" \
    --data_dir  "${DATA_DIR}" \
    --output_dir "${OUT_DIR}" \
    --metadata_file "${METADATA_FILE}" \
    --seed "${SEED}"

EXIT_CODE=\$?
echo "End: \$(date) | Exit: \$EXIT_CODE"
exit \$EXIT_CODE
EOF
    )
    SUBMITTED+=("${JOB_ID}")
    echo "  Submitted job ${JOB_ID} for ${CL}" >&2
done

echo "Submitted: ${#SUBMITTED[@]}/${#CELL_LINES[@]} jobs" >&2
echo "${SUBMITTED[*]//\ /:}"
