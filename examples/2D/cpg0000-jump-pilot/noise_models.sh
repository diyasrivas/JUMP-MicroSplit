#!/bin/bash
# Train per-channel noise models for cpg0000-jump-pilot (Step 2).
#
# Submits 5 parallel SLURM jobs (one per channel) or runs sequentially.
#
# Usage:
#   ./noise_models.sh --cell_line A549
#   ./noise_models.sh --cell_line A549 --slurm
#   ./noise_models.sh --cell_lines "A549 U2OS" --slurm
#
# Options:
#   --cell_lines "A549 U2OS"   Space-separated cell lines (default: A549 U2OS)
#   --dataset_base DIR         Base dir for training datasets
#                              (default: /project/cell_paint_mono/training_datasets)
#   --slurm                    Submit as SLURM jobs
#   --partition NAME           SLURM GPU partition (default: dgx)
#   --time HH:MM:SS            SLURM walltime (default: 4:00:00)
#   --mem XG                   SLURM memory (default: 32G)
#   --dependency JOBS          Colon-separated job IDs to depend on

set -euo pipefail

SUBMIT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PYTHON_SCRIPT="${SUBMIT_DIR}/2_noisemodels.py"

CELL_LINES=("A549" "U2OS")
CHANNELS=("DNA" "RNA" "ER" "AGP" "Mito")
DATASET_BASE="/project/cell_paint_mono/training_datasets"
SLURM_MODE=false
DEPENDENCY_JOBS=""

SLURM_PARTITION="dgx"
SLURM_TIME="4:00:00"
SLURM_MEM="32G"
SLURM_CPUS=4
SLURM_GRES="gpu:71gb:1"

while [[ $# -gt 0 ]]; do
    case $1 in
        --cell_lines)    read -ra CELL_LINES <<< "$2"; shift 2 ;;
        --dataset_base)  DATASET_BASE="$2"; shift 2 ;;
        --slurm)         SLURM_MODE=true;   shift ;;
        --partition)     SLURM_PARTITION="$2"; shift 2 ;;
        --time)          SLURM_TIME="$2";   shift 2 ;;
        --mem)           SLURM_MEM="$2";    shift 2 ;;
        --dependency)    DEPENDENCY_JOBS="$2"; shift 2 ;;
        --help)
            sed -n '2,18p' "$0" | sed 's/^# \{0,1\}//'
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
echo "cpg0000 noise models | cell_lines=${CELL_LINES[*]}" >&2
echo "================================================================================" >&2

SUBMITTED=()
DEPENDENCY_DIRECTIVE=""
[[ -n "$DEPENDENCY_JOBS" ]] && DEPENDENCY_DIRECTIVE="#SBATCH --dependency=afterok:${DEPENDENCY_JOBS}"

for CL in "${CELL_LINES[@]}"; do
    DATASET_DIR="${DATASET_BASE}/training_dataset_${CL}"
    NM_DIR="${DATASET_DIR}/noise_models"
    mkdir -p "${NM_DIR}"

    for CH in "${CHANNELS[@]}"; do
        if [[ "$SLURM_MODE" == false ]]; then
            _activate_conda
            python "${PYTHON_SCRIPT}" \
                --dataset_dir "${DATASET_DIR}" \
                --output_dir  "${NM_DIR}" \
                --channel     "${CH}"
            continue
        fi

        LOG_DIR="${DATASET_DIR}/logs/noise_models"
        mkdir -p "${LOG_DIR}"
        JOB_ID=$(sbatch --parsable <<EOF
#!/bin/bash
#SBATCH --job-name=nm_${CL}_${CH}
#SBATCH --output=${LOG_DIR}/nm_${CL}_${CH}_%j.log
#SBATCH --error=${LOG_DIR}/nm_${CL}_${CH}_%j.log
#SBATCH --partition=${SLURM_PARTITION}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${SLURM_CPUS}
#SBATCH --mem=${SLURM_MEM}
#SBATCH --gres=${SLURM_GRES}
#SBATCH --time=${SLURM_TIME}
${DEPENDENCY_DIRECTIVE}

echo "Job \${SLURM_JOB_ID}: NM ${CL} ${CH} | Node: \${SLURMD_NODENAME} | Start: \$(date)"

$(declare -f _activate_conda)
_activate_conda

python "${PYTHON_SCRIPT}" \
    --dataset_dir "${DATASET_DIR}" \
    --output_dir  "${NM_DIR}" \
    --channel     "${CH}"

EXIT_CODE=\$?
echo "End: \$(date) | Exit: \$EXIT_CODE"
exit \$EXIT_CODE
EOF
        )
        SUBMITTED+=("${JOB_ID}")
        echo "  Submitted NM job ${JOB_ID}: ${CL}/${CH}" >&2
    done
done

if [[ "$SLURM_MODE" == true ]]; then
    echo "Submitted ${#SUBMITTED[@]} jobs" >&2
    echo "${SUBMITTED[*]//\ /:}"
fi
