#!/bin/bash
# Build cpg0036-EU-OS-bioactives training dataset (Step 1).
#
# Run once per cell line (HepG2, U2OS). Both cell lines must be trained
# separately; they produce separate dataset directories.
#
# Usage:
#   ./datasets.sh --cell_line HepG2
#   ./datasets.sh --cell_line U2OS --slurm
#   ./datasets.sh --cell_line HepG2 --samples 3500 --slurm
#
# Options:
#   --cell_line NAME   Cell line to build dataset for: HepG2 or U2OS (required)
#   --data_dir DIR     Root of cpg0036 data for this cell line
#                      (default: /project/cell_paint_mono/cpg0036-EU-OS-bioactives)
#   --output_dir DIR   Where to write the dataset
#                      (default: /project/cell_paint_mono/training_datasets/training_dataset_cpg0036_{cell_line})
#   --samples N        Target training FOVs (default: 3500)
#   --seed N           Random seed (default: 42)
#   --plate_filter STR Optional plate-name substring to filter (repeat for multiple)
#   --slurm            Submit as a SLURM job instead of running interactively
#   --partition NAME   SLURM partition (default: cpu)
#   --time HH:MM:SS    SLURM walltime (default: 8:00:00)
#   --mem XG           SLURM memory (default: 32G)

set -euo pipefail

SUBMIT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PYTHON_SCRIPT="${SUBMIT_DIR}/1_datasets.py"

CELL_LINE=""
DATA_DIR="/project/cell_paint_mono/cpg0036-EU-OS-bioactives"
OUTPUT_DIR=""
SAMPLES=3500
SEED=42
PLATE_FILTER_ARGS=()
SLURM_MODE=false

SLURM_PARTITION="cpu"
SLURM_TIME="8:00:00"
SLURM_MEM="32G"
SLURM_CPUS=4

while [[ $# -gt 0 ]]; do
    case $1 in
        --cell_line)    CELL_LINE="$2";    shift 2 ;;
        --data_dir)     DATA_DIR="$2";     shift 2 ;;
        --output_dir)   OUTPUT_DIR="$2";   shift 2 ;;
        --samples)      SAMPLES="$2";      shift 2 ;;
        --seed)         SEED="$2";         shift 2 ;;
        --plate_filter) PLATE_FILTER_ARGS+=("$2"); shift 2 ;;
        --slurm)        SLURM_MODE=true;   shift ;;
        --partition)    SLURM_PARTITION="$2"; shift 2 ;;
        --time)         SLURM_TIME="$2";   shift 2 ;;
        --mem)          SLURM_MEM="$2";    shift 2 ;;
        --help)
            sed -n '2,22p' "$0" | sed 's/^# \{0,1\}//'
            exit 0 ;;
        *) echo "ERROR: Unknown option: $1" >&2; exit 1 ;;
    esac
done

if [[ -z "$CELL_LINE" ]]; then
    echo "ERROR: --cell_line is required (HepG2 or U2OS)" >&2
    exit 1
fi

[[ -z "$OUTPUT_DIR" ]] && OUTPUT_DIR="/project/cell_paint_mono/training_datasets/training_dataset_cpg0036_${CELL_LINE}"

# Build plate_filter argument if any
PLATE_FILTER_EXTRA=""
for PF in "${PLATE_FILTER_ARGS[@]+"${PLATE_FILTER_ARGS[@]}"}"; do
    PLATE_FILTER_EXTRA+=" --plate_filter ${PF}"
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
echo "cpg0036 datasets | cell_line=${CELL_LINE} | samples=${SAMPLES}" >&2
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
        --seed        "${SEED}" \
        ${PLATE_FILTER_EXTRA}
    exit 0
fi

LOG_DIR="${OUTPUT_DIR}/logs"
mkdir -p "${LOG_DIR}"

JOB_ID=$(sbatch --parsable <<EOF
#!/bin/bash
#SBATCH --job-name=cpg0036_ds_${CELL_LINE}
#SBATCH --output=${LOG_DIR}/datasets_${CELL_LINE}_%j.log
#SBATCH --error=${LOG_DIR}/datasets_${CELL_LINE}_%j.log
#SBATCH --partition=${SLURM_PARTITION}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${SLURM_CPUS}
#SBATCH --mem=${SLURM_MEM}
#SBATCH --time=${SLURM_TIME}

echo "Job \${SLURM_JOB_ID}: cpg0036 datasets ${CELL_LINE} | Node: \${SLURMD_NODENAME} | Start: \$(date)"

$(declare -f _activate_conda)
_activate_conda

python "${PYTHON_SCRIPT}" \
    --cell_line   "${CELL_LINE}" \
    --data_dir    "${DATA_DIR}" \
    --output_dir  "${OUTPUT_DIR}" \
    --samples     "${SAMPLES}" \
    --seed        "${SEED}" \
    ${PLATE_FILTER_EXTRA}

EXIT_CODE=\$?
echo "End: \$(date) | Exit: \$EXIT_CODE"
exit \$EXIT_CODE
EOF
)

echo "Submitted job ${JOB_ID}: cpg0036 datasets ${CELL_LINE}" >&2
echo "${JOB_ID}"
