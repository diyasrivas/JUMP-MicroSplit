#!/bin/bash
# Steps 0+1 for cpg0000-jump-pilot: build the split manifest, then the training dataset.
#
# Usage:
#   ./datasets.sh                              # both cell lines, interactive (CPU, minutes)
#   ./datasets.sh --cell_lines "A549" --slurm  # one cell line as a SLURM job
#   ./datasets.sh --skip_split                 # reuse existing manifests
#
# Options:
#   --cell_lines "A549 U2OS"      Space-separated (default: A549 U2OS)
#   --n_genes N                   Test genes (default: 10)
#   --seed N                      (default: 42)
#   --train_fovs_per_plate N      Training FOVs sampled per benchmark plate (default: 15 → 300 per cell line)
#   --test_sites "2 4 6 8"        Sites predicted per test well (default: 2 4 6 8)
#   --data_dir DIR                Root of raw data (default: /project/cell_paint_mono)
#   --paper_dir DIR               Cloned paper repo (default: /project/cell_paint_mono/2024_Chandrasekaran_NatureMethods)
#   --splits_dir DIR              Manifest root (default: /project/cell_paint_mono/splits/cpg0000-jump-pilot)
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
METADATA_FILE="${SUBMIT_DIR}/experiment-metadata.tsv"

CELL_LINES=("A549" "U2OS")
N_GENES=10
SEED=42
TRAIN_FOVS_PER_PLATE=15
TEST_SITES="2 4 6 8"
DATA_DIR="/project/cell_paint_mono"
PAPER_DIR="/project/cell_paint_mono/2024_Chandrasekaran_NatureMethods"
SPLITS_DIR="/project/cell_paint_mono/splits/cpg0000-jump-pilot"
OUTPUT_BASE="/project/cell_paint_mono/training_datasets"
SKIP_SPLIT=false
SLURM_MODE=false
SLURM_PARTITION="cpu"
SLURM_TIME="4:00:00"
SLURM_MEM="16G"
SLURM_CPUS=4

while [[ $# -gt 0 ]]; do
    case $1 in
        --cell_lines)           read -ra CELL_LINES <<< "$2"; shift 2 ;;
        --n_genes)              N_GENES="$2";              shift 2 ;;
        --seed)                 SEED="$2";                 shift 2 ;;
        --train_fovs_per_plate) TRAIN_FOVS_PER_PLATE="$2"; shift 2 ;;
        --test_sites)           TEST_SITES="$2";           shift 2 ;;
        --data_dir)             DATA_DIR="$2";             shift 2 ;;
        --paper_dir)            PAPER_DIR="$2";            shift 2 ;;
        --splits_dir)           SPLITS_DIR="$2";           shift 2 ;;
        --output_base)          OUTPUT_BASE="$2";          shift 2 ;;
        --skip_split)           SKIP_SPLIT=true;           shift ;;
        --slurm)                SLURM_MODE=true;           shift ;;
        --partition)            SLURM_PARTITION="$2";      shift 2 ;;
        --time)                 SLURM_TIME="$2";           shift 2 ;;
        --mem)                  SLURM_MEM="$2";            shift 2 ;;
        --help) sed -n '2,24p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
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

_run() {
    local CL="$1"
    local OUT_DIR="${OUTPUT_BASE}/training_dataset_cpg0000_${CL}"
    if [[ "$SKIP_SPLIT" == false ]]; then
        python "${SPLIT_SCRIPT}" --cell_line "${CL}" --n_genes "${N_GENES}" --seed "${SEED}" \
            --train_fovs_per_plate "${TRAIN_FOVS_PER_PLATE}" --test_sites ${TEST_SITES} \
            --data_dir "${DATA_DIR}" --paper_dir "${PAPER_DIR}" --metadata_file "${METADATA_FILE}" \
            --out_dir "${SPLITS_DIR}"
    fi
    python "${DATASET_SCRIPT}" --cell_line "${CL}" --splits_dir "${SPLITS_DIR}" \
        --data_dir "${DATA_DIR}" --output_dir "${OUT_DIR}"
}

echo "================================================================================" >&2
echo "cpg0000 split+datasets | cell_lines=${CELL_LINES[*]} | genes=${N_GENES} | train/plate=${TRAIN_FOVS_PER_PLATE}" >&2
echo "================================================================================" >&2

if [[ "$SLURM_MODE" == false ]]; then
    _activate_conda
    for CL in "${CELL_LINES[@]}"; do _run "${CL}"; done
    exit 0
fi

SUBMITTED=()
for CL in "${CELL_LINES[@]}"; do
    OUT_DIR="${OUTPUT_BASE}/training_dataset_cpg0000_${CL}"
    LOG_DIR="${OUT_DIR}/logs"; mkdir -p "${LOG_DIR}"
    SPLIT_CMD=""
    [[ "$SKIP_SPLIT" == false ]] && SPLIT_CMD="python ${SPLIT_SCRIPT} --cell_line ${CL} --n_genes ${N_GENES} --seed ${SEED} --train_fovs_per_plate ${TRAIN_FOVS_PER_PLATE} --test_sites ${TEST_SITES} --data_dir ${DATA_DIR} --paper_dir ${PAPER_DIR} --metadata_file ${METADATA_FILE} --out_dir ${SPLITS_DIR}"
    JOB_ID=$(sbatch --parsable <<EOF
#!/bin/bash
#SBATCH --job-name=cpg0000_ds_${CL}
#SBATCH --output=${LOG_DIR}/datasets_${CL}_%j.log
#SBATCH --error=${LOG_DIR}/datasets_${CL}_%j.log
#SBATCH --partition=${SLURM_PARTITION}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${SLURM_CPUS}
#SBATCH --mem=${SLURM_MEM}
#SBATCH --time=${SLURM_TIME}

echo "Job \${SLURM_JOB_ID}: ${CL} split+dataset | Node: \${SLURMD_NODENAME} | Start: \$(date)"
$(declare -f _activate_conda)
_activate_conda
set -e
${SPLIT_CMD}
python ${DATASET_SCRIPT} --cell_line ${CL} --splits_dir ${SPLITS_DIR} --data_dir ${DATA_DIR} --output_dir ${OUT_DIR}
echo "End: \$(date)"
EOF
    )
    SUBMITTED+=("${JOB_ID}")
    echo "  Submitted job ${JOB_ID} for ${CL}" >&2
done
echo "${SUBMITTED[*]//\ /:}"
