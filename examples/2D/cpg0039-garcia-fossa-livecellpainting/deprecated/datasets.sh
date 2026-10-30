#!/bin/bash
# Build cpg0039-garcia-fossa-livecellpainting training dataset (Step 1).
#
# Run once per cell line (Huh_7, MCF_7, PNT1A, PC_3).
# Each cell line must be trained separately.
#
# Usage:
#   ./datasets.sh --cell_line Huh_7
#   ./datasets.sh --cell_line MCF_7 --slurm
#   ./datasets.sh --cell_line PNT1A --samples 2500 --slurm
#
# Options:
#   --cell_line NAME   Cell line (Huh_7 | MCF_7 | PNT1A | PC_3) (required)
#   --data_dir DIR     Root of cpg0039 data for this cell line
#                      (default: /project/cell_paint_mono/cpg0039-garcia-fossa-livecellpainting)
#   --output_dir DIR   Output directory
#                      (default: /project/cell_paint_mono/training_datasets/training_dataset_cpg0039_{cell_line})
#   --samples N        Target training FOVs (default: 2500)
#   --seed N           Random seed (default: 42)
#   --slurm            Submit as a SLURM job instead of running interactively
#   --partition NAME   SLURM partition (default: cpu)
#   --time HH:MM:SS    SLURM walltime (default: 6:00:00)
#   --mem XG           SLURM memory (default: 16G)

set -euo pipefail

SUBMIT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PYTHON_SCRIPT="${SUBMIT_DIR}/1_datasets.py"

CELL_LINE=""
DATA_DIR="/project/cell_paint_mono/cpg0039-garcia-fossa-livecellpainting"
OUTPUT_DIR=""
SAMPLES=2500
SEED=42
SLURM_MODE=false

SLURM_PARTITION="cpu"
SLURM_TIME="6:00:00"
SLURM_MEM="16G"
SLURM_CPUS=4

while [[ $# -gt 0 ]]; do
    case $1 in
        --cell_line)  CELL_LINE="$2";    shift 2 ;;
        --data_dir)   DATA_DIR="$2";     shift 2 ;;
        --output_dir) OUTPUT_DIR="$2";   shift 2 ;;
        --samples)    SAMPLES="$2";      shift 2 ;;
        --seed)       SEED="$2";         shift 2 ;;
        --slurm)      SLURM_MODE=true;   shift ;;
        --partition)  SLURM_PARTITION="$2"; shift 2 ;;
        --time)       SLURM_TIME="$2";   shift 2 ;;
        --mem)        SLURM_MEM="$2";    shift 2 ;;
        --help)
            sed -n '2,21p' "$0" | sed 's/^# \{0,1\}//'
            exit 0 ;;
        *) echo "ERROR: Unknown option: $1" >&2; exit 1 ;;
    esac
done

if [[ -z "$CELL_LINE" ]]; then
    echo "ERROR: --cell_line is required (Huh_7 | MCF_7 | PNT1A | PC_3)" >&2
    exit 1
fi

[[ -z "$OUTPUT_DIR" ]] && OUTPUT_DIR="/project/cell_paint_mono/training_datasets/training_dataset_cpg0039_${CELL_LINE}"

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
echo "cpg0039 datasets | cell_line=${CELL_LINE} | samples=${SAMPLES}" >&2
echo "  data_dir:   ${DATA_DIR}" >&2
echo "  output_dir: ${OUTPUT_DIR}" >&2
echo "================================================================================" >&2

if [[ "$SLURM_MODE" == false ]]; then
    _activate_conda
    python "${PYTHON_SCRIPT}" \
        --cell_line   "${CELL_LINE}" \
        --data_dir    "${DATA_DIR}" \
        --output_dir  "${OUTPUT_DIR}" \
        --samples     "${SAMPLES}" \
        --seed        "${SEED}"
    exit 0
fi

LOG_DIR="${OUTPUT_DIR}/logs"
mkdir -p "${LOG_DIR}"

JOB_ID=$(sbatch --parsable <<EOF
#!/bin/bash
#SBATCH --job-name=cpg0039_ds_${CELL_LINE}
#SBATCH --output=${LOG_DIR}/datasets_${CELL_LINE}_%j.log
#SBATCH --error=${LOG_DIR}/datasets_${CELL_LINE}_%j.log
#SBATCH --partition=${SLURM_PARTITION}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${SLURM_CPUS}
#SBATCH --mem=${SLURM_MEM}
#SBATCH --time=${SLURM_TIME}

echo "Job \${SLURM_JOB_ID}: cpg0039 datasets ${CELL_LINE} | Node: \${SLURMD_NODENAME} | Start: \$(date)"

$(declare -f _activate_conda)
_activate_conda

python "${PYTHON_SCRIPT}" \
    --cell_line   "${CELL_LINE}" \
    --data_dir    "${DATA_DIR}" \
    --output_dir  "${OUTPUT_DIR}" \
    --samples     "${SAMPLES}" \
    --seed        "${SEED}"

EXIT_CODE=\$?
echo "End: \$(date) | Exit: \$EXIT_CODE"
exit \$EXIT_CODE
EOF
)

echo "Submitted job ${JOB_ID}: cpg0039 datasets ${CELL_LINE}" >&2
echo "${JOB_ID}"
