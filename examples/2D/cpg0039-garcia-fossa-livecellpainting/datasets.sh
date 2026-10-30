#!/bin/bash
# Steps 0+1 for cpg0039-garcia-fossa-livecellpainting: build the split manifest, then the training dataset.
#
# Usage:
#   ./datasets.sh                              # interactive (CPU, minutes)
#   ./datasets.sh --slurm                      # as a SLURM job
#   ./datasets.sh --skip_split                 # reuse the existing manifest
#
# Options:
#   --n_train N                   Training FOVs (default: 300)
#   --dili_fields "1 9 17 25"     DILI fields predicted per test well (default: 1 9 17 25)
#   --seed N                      (default: 42)
#   --data_dir DIR                Root of raw data (default: /project/cell_paint_mono/cpg0039-garcia-fossa-livecellpainting/unicamp)
#   --splits_dir DIR              Manifest root (default: /project/cell_paint_mono/splits/cpg0039-garcia-fossa-livecellpainting)
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

N_TRAIN=300
DILI_FIELDS="1 9 17 25"
SEED=42
DATA_DIR="/project/cell_paint_mono/cpg0039-garcia-fossa-livecellpainting/unicamp"
SPLITS_DIR="/project/cell_paint_mono/splits/cpg0039-garcia-fossa-livecellpainting"
OUTPUT_BASE="/project/cell_paint_mono/training_datasets"
SKIP_SPLIT=false
SLURM_MODE=false
SLURM_PARTITION="cpu"
SLURM_TIME="4:00:00"
SLURM_MEM="16G"
SLURM_CPUS=4

while [[ $# -gt 0 ]]; do
    case $1 in
        --n_train)      N_TRAIN="$2";         shift 2 ;;
        --dili_fields)  DILI_FIELDS="$2";     shift 2 ;;
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
    if [[ "$SKIP_SPLIT" == false ]]; then
        echo "python ${SPLIT_SCRIPT} --n_train ${N_TRAIN} --dili_fields ${DILI_FIELDS} --seed ${SEED} --data_dir ${DATA_DIR} --out_dir ${SPLITS_DIR}"
    fi
    echo "python ${DATASET_SCRIPT} --splits_dir ${SPLITS_DIR} --out_base ${OUTPUT_BASE}"
}

echo "================================================================================" >&2
echo "cpg0039 split+dataset | n_train=${N_TRAIN} | dili_fields=${DILI_FIELDS}" >&2
echo "================================================================================" >&2

if [[ "$SLURM_MODE" == false ]]; then
    _activate_conda
    while IFS= read -r CMD; do eval "${CMD}"; done < <(_commands)
    exit 0
fi

LOG_DIR="${OUTPUT_BASE}/training_dataset_cpg0039_Huh7/logs"; mkdir -p "${LOG_DIR}"
JOB_ID=$(sbatch --parsable <<EOF
#!/bin/bash
#SBATCH --job-name=cpg0039_ds
#SBATCH --output=${LOG_DIR}/datasets_%j.log
#SBATCH --error=${LOG_DIR}/datasets_%j.log
#SBATCH --partition=${SLURM_PARTITION}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${SLURM_CPUS}
#SBATCH --mem=${SLURM_MEM}
#SBATCH --time=${SLURM_TIME}

echo "Job \${SLURM_JOB_ID}: cpg0039 split+dataset | Node: \${SLURMD_NODENAME} | Start: \$(date)"
$(declare -f _activate_conda)
_activate_conda
set -e
$(_commands)
echo "End: \$(date)"
EOF
)
echo "Submitted job ${JOB_ID}" >&2
echo "${JOB_ID}"
