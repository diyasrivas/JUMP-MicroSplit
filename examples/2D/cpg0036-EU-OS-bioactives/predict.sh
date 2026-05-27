#!/bin/bash
# MicroSplit plate-level prediction for cpg0036-EU-OS-bioactives (Step 4).
#
# Submits one SLURM job per plate.  Always use the training_dir for the
# same cell line as the plates being predicted.
#
# Usage:
#   ./predict.sh --plate_dir /path/to/plate --training_dir .../training_dataset_cpg0036_HepG2
#   ./predict.sh --batch 2021_XX_XX_Batch1 --training_dir .../training_dataset_cpg0036_HepG2 --slurm
#
# Options:
#   --batch NAME             Batch name for plate discovery
#   --plate_dir DIR          Process a single plate directory
#   --data_dir DIR           Root data dir (default: /project/cell_paint_mono/cpg0036-EU-OS-bioactives)
#   --training_dir DIR       Training dataset dir for the same cell line (required)
#   --output_dir DIR         Output root (default: /project/cell_paint_mono/predictions/cpg0036-EU-OS-bioactives)
#   --mmse_count N           MMSE samples (default: 50)
#   --num_posterior N        Posterior samples to save (default: 3)
#   --grid_size N            Prediction grid size (default: 32)
#   --max_fovs N             Limit FOVs per plate, 0=all (default: 0)
#   --slurm                  Submit as SLURM jobs
#   --partition NAME         (default: dgx)
#   --time HH:MM:SS          (default: 48:00:00)
#   --mem XG                 (default: 64G)
#   --dependency JOBS        Colon-separated job dependencies

set -euo pipefail

SUBMIT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PYTHON_SCRIPT="${SUBMIT_DIR}/4_predict.py"

BATCH=""
PLATE_DIR=""
DATA_DIR="/project/cell_paint_mono/cpg0036-EU-OS-bioactives"
TRAINING_DIR=""
OUTPUT_DIR="/project/cell_paint_mono/predictions/cpg0036-EU-OS-bioactives"
MMSE_COUNT=50
NUM_POSTERIOR=3
GRID_SIZE=32
MAX_FOVS=0
SLURM_MODE=false
DEPENDENCY_JOBS=""

SLURM_PARTITION="dgx"
SLURM_TIME="48:00:00"
SLURM_MEM="64G"
SLURM_CPUS=8
SLURM_GRES="gpu:71gb:1"

while [[ $# -gt 0 ]]; do
    case $1 in
        --batch)        BATCH="$2";           shift 2 ;;
        --plate_dir)    PLATE_DIR="$2";       shift 2 ;;
        --data_dir)     DATA_DIR="$2";        shift 2 ;;
        --training_dir) TRAINING_DIR="$2";    shift 2 ;;
        --output_dir)   OUTPUT_DIR="$2";      shift 2 ;;
        --mmse_count)   MMSE_COUNT="$2";      shift 2 ;;
        --num_posterior) NUM_POSTERIOR="$2";  shift 2 ;;
        --grid_size)    GRID_SIZE="$2";       shift 2 ;;
        --max_fovs)     MAX_FOVS="$2";        shift 2 ;;
        --slurm)        SLURM_MODE=true;      shift ;;
        --partition)    SLURM_PARTITION="$2"; shift 2 ;;
        --time)         SLURM_TIME="$2";      shift 2 ;;
        --mem)          SLURM_MEM="$2";       shift 2 ;;
        --dependency)   DEPENDENCY_JOBS="$2"; shift 2 ;;
        --help)
            sed -n '2,28p' "$0" | sed 's/^# \{0,1\}//'
            exit 0 ;;
        *) echo "ERROR: Unknown option: $1" >&2; exit 1 ;;
    esac
done

if [[ -z "$TRAINING_DIR" ]]; then
    echo "ERROR: --training_dir is required (use the model for the same cell line)" >&2
    exit 1
fi

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

# Collect plate directories to process
PLATE_DIRS=()
if [[ -n "$PLATE_DIR" ]]; then
    PLATE_DIRS=("$PLATE_DIR")
elif [[ -n "$BATCH" ]]; then
    BATCH_IMAGES="${DATA_DIR}/${BATCH}/images"
    if [[ ! -d "$BATCH_IMAGES" ]]; then
        echo "ERROR: ${BATCH_IMAGES} not found" >&2; exit 1
    fi
    while IFS= read -r -d '' d; do
        PLATE_DIRS+=("$d")
    done < <(find "${BATCH_IMAGES}" -mindepth 1 -maxdepth 1 -type d -print0 | sort -z)
else
    echo "ERROR: Specify --plate_dir or --batch" >&2; exit 1
fi

echo "================================================================================" >&2
echo "cpg0036 predict | plates=${#PLATE_DIRS[@]}" >&2
echo "  training_dir: ${TRAINING_DIR}" >&2
echo "  output_dir:   ${OUTPUT_DIR}" >&2
echo "================================================================================" >&2

DEPENDENCY_DIRECTIVE=""
[[ -n "$DEPENDENCY_JOBS" ]] && DEPENDENCY_DIRECTIVE="#SBATCH --dependency=afterok:${DEPENDENCY_JOBS}"

SUBMITTED=()
for PD in "${PLATE_DIRS[@]}"; do
    PNAME=$(basename "$PD")
    LOG_DIR="${OUTPUT_DIR}/logs/${PNAME}"
    mkdir -p "${LOG_DIR}"

    if [[ "$SLURM_MODE" == false ]]; then
        _activate_conda
        python "${PYTHON_SCRIPT}" \
            --plate_dir    "${PD}" \
            --training_dir "${TRAINING_DIR}" \
            --output_dir   "${OUTPUT_DIR}" \
            --mmse_count   "${MMSE_COUNT}" \
            --num_posterior_samples "${NUM_POSTERIOR}" \
            --grid_size    "${GRID_SIZE}" \
            --max_fovs     "${MAX_FOVS}"
        continue
    fi

    JOB_ID=$(sbatch --parsable <<EOF
#!/bin/bash
#SBATCH --job-name=pred_cpg0036_${PNAME:0:20}
#SBATCH --output=${LOG_DIR}/predict_%j.log
#SBATCH --error=${LOG_DIR}/predict_%j.log
#SBATCH --partition=${SLURM_PARTITION}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${SLURM_CPUS}
#SBATCH --mem=${SLURM_MEM}
#SBATCH --gres=${SLURM_GRES}
#SBATCH --time=${SLURM_TIME}
${DEPENDENCY_DIRECTIVE}

echo "Job \${SLURM_JOB_ID}: predict cpg0036 ${PNAME} | Node: \${SLURMD_NODENAME} | Start: \$(date)"
$(declare -f _activate_conda)
_activate_conda

export TORCHINDUCTOR_DISABLE_CUDAGRAPHS=1

python "${PYTHON_SCRIPT}" \
    --plate_dir    "${PD}" \
    --training_dir "${TRAINING_DIR}" \
    --output_dir   "${OUTPUT_DIR}" \
    --mmse_count   "${MMSE_COUNT}" \
    --num_posterior_samples "${NUM_POSTERIOR}" \
    --grid_size    "${GRID_SIZE}" \
    --max_fovs     "${MAX_FOVS}"

EXIT_CODE=\$?
echo "End: \$(date) | Exit: \$EXIT_CODE"
exit \$EXIT_CODE
EOF
    )
    SUBMITTED+=("${JOB_ID}")
    echo "  Submitted job ${JOB_ID}: ${PNAME}" >&2
done

if [[ "$SLURM_MODE" == true ]]; then
    echo "Submitted: ${#SUBMITTED[@]} jobs" >&2
    echo "${SUBMITTED[*]//\ /:}"
fi
