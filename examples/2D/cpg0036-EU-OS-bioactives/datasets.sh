#!/bin/bash
# Steps 0+1 for cpg0036-EU-OS-bioactives: build the split manifests, then the training datasets.
#
# Usage:
#   ./datasets.sh                              # both cell lines, interactive (CPU)
#   ./datasets.sh --cell_lines "HepG2" --slurm # one cell line as a SLURM job
#   ./datasets.sh --skip_split                 # reuse existing manifests
#
# Options:
#   --cell_lines "HepG2 U2OS"     Space-separated (default: HepG2 U2OS)
#   --n_train N                   Training FOVs per cell line (default: 300)
#   --seed N                      (default: 42)
#   --data_dir DIR                Root of raw data (default: /project/cell_paint_mono/cpg0036-EU-OS-bioactives)
#   --splits_dir DIR              Manifest root (default: /project/cell_paint_mono/splits/cpg0036-EU-OS-bioactives)
#   --output_base DIR             Dataset root (default: /project/cell_paint_mono/training_datasets)
#   --skip_split                  Do not (re)run 0_split.py
#   --slurm                       Submit as SLURM job(s)
#   --partition NAME              (default: cpu)
#   --time HH:MM:SS               (default: 4:00:00)
#   --mem XG                      (default: 16G)

set -euo pipefail

SUBMIT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
SPLIT_SCRIPT="${SUBMIT_DIR}/0_split.py"
DATASET_SCRIPT="${SUBMIT_DIR}/1_datasets.py"

CELL_LINES=("HepG2" "U2OS")
N_TRAIN=300
SEED=42
DATA_DIR="/project/cell_paint_mono/cpg0036-EU-OS-bioactives"
SPLITS_DIR="/project/cell_paint_mono/splits/cpg0036-EU-OS-bioactives"
OUTPUT_BASE="/project/cell_paint_mono/training_datasets"
SKIP_SPLIT=false
SLURM_MODE=false
SLURM_PARTITION="cpu"
SLURM_TIME="4:00:00"
SLURM_MEM="16G"
SLURM_CPUS=4

while [[ $# -gt 0 ]]; do
    case $1 in
        --cell_lines)   read -ra CELL_LINES <<< "$2"; shift 2 ;;
        --n_train)      N_TRAIN="$2";         shift 2 ;;
        --seed)         SEED="$2";            shift 2 ;;
        --data_dir)     DATA_DIR="$2";        shift 2 ;;
        --splits_dir)   SPLITS_DIR="$2";      shift 2 ;;
        --output_base)  OUTPUT_BASE="$2";     shift 2 ;;
        --skip_split)   SKIP_SPLIT=true;      shift ;;
        --slurm)        SLURM_MODE=true;      shift ;;
        --partition)    SLURM_PARTITION="$2"; shift 2 ;;
        --time)         SLURM_TIME="$2";      shift 2 ;;
        --mem)          SLURM_MEM="$2";       shift 2 ;;
        --help) sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
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

_commands() {
    local CL="$1"
    if [[ "$SKIP_SPLIT" == false ]]; then
        echo "python ${SPLIT_SCRIPT} --cell_lines ${CL} --n_train ${N_TRAIN} --seed ${SEED} --data_dir ${DATA_DIR} --out_dir ${SPLITS_DIR}"
    fi
    echo "python ${DATASET_SCRIPT} --cell_line ${CL} --splits_dir ${SPLITS_DIR} --out_base ${OUTPUT_BASE}"
}

echo "================================================================================" >&2
echo "cpg0036 split+datasets | cell_lines=${CELL_LINES[*]} | n_train=${N_TRAIN}" >&2
echo "================================================================================" >&2

if [[ "$SLURM_MODE" == false ]]; then
    _activate_conda
    for CL in "${CELL_LINES[@]}"; do
        while IFS= read -r CMD; do eval "${CMD}"; done < <(_commands "${CL}")
    done
    exit 0
fi

SUBMITTED=()
for CL in "${CELL_LINES[@]}"; do
    LOG_DIR="${OUTPUT_BASE}/training_dataset_cpg0036_allsites_${CL}/logs"; mkdir -p "${LOG_DIR}"
    JOB_ID=$(sbatch --parsable <<EOF
#!/bin/bash
#SBATCH --job-name=cpg0036_ds_${CL}
#SBATCH --output=${LOG_DIR}/datasets_${CL}_%j.log
#SBATCH --error=${LOG_DIR}/datasets_${CL}_%j.log
#SBATCH --partition=${SLURM_PARTITION}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${SLURM_CPUS}
#SBATCH --mem=${SLURM_MEM}
#SBATCH --time=${SLURM_TIME}

echo "Job \${SLURM_JOB_ID}: cpg0036 ${CL} split+dataset | Node: \${SLURMD_NODENAME} | Start: \$(date)"
$(declare -f _activate_conda)
_activate_conda
set -e
$(_commands "${CL}")
echo "End: \$(date)"
EOF
    )
    SUBMITTED+=("${JOB_ID}")
    echo "  Submitted job ${JOB_ID} for ${CL}" >&2
done
echo "${SUBMITTED[*]//\ /:}"
