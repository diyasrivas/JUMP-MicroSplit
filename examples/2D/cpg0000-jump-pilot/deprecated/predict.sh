#!/bin/bash
# Step 4 for cpg0000-jump-pilot: one SLURM job per benchmark plate, predicting only the
# test FOVs listed in the split manifest. MMSE-50, no posterior samples.
#
# Usage:
#   ./predict.sh --cell_line A549 --slurm                 # all 20 benchmark plates of A549
#   ./predict.sh --cell_line A549 --plate_dir /path/to/plate --slurm
#   ./predict.sh --cell_line A549 --benchmark             # 1 FOV timing test, interactive
#
# Options:
#   --cell_line NAME         A549 or U2OS (required)
#   --manifest FILE          (default: /project/cell_paint_mono/splits/cpg0000-jump-pilot/{cell_line}/split_manifest.csv)
#   --plate_dir DIR          Process a single plate directory (default: every plate in the manifest)
#   --data_dir DIR           (default: /project/cell_paint_mono)
#   --dataset_base DIR       (default: /project/cell_paint_mono/training_datasets)
#   --output_dir DIR         (default: /project/cell_paint_mono/predictions/cpg0000-jump-pilot)
#   --mmse_count N           (default: 50)
#   --num_posterior N        Posterior samples to save (default: 0)
#   --batch_size N           Tiles per forward pass (default: 256)
#   --no_autocast            Disable bfloat16 autocast
#   --max_fovs N             Limit FOVs per plate, 0 = all (default: 0)
#   --benchmark              Run 1 FOV of the first plate interactively and report s/FOV
#   --slurm                  Submit as SLURM jobs
#   --partition NAME         (default: dgx)
#   --gres GRES              (default: gpu:71gb:1)
#   --time HH:MM:SS          (default: 12:00:00)
#   --mem XG                 (default: 64G)
#   --dependency JOBS        Colon-separated job dependencies

set -euo pipefail

SUBMIT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PYTHON_SCRIPT="${SUBMIT_DIR}/4_predict.py"
METADATA_FILE="${SUBMIT_DIR}/experiment-metadata.tsv"

CELL_LINE=""
MANIFEST=""
PLATE_DIR=""
DATA_DIR="/project/cell_paint_mono"
DATASET_BASE="/project/cell_paint_mono/training_datasets"
OUTPUT_DIR="/project/cell_paint_mono/predictions/cpg0000-jump-pilot"
MMSE_COUNT=50
NUM_POSTERIOR=0
BATCH_SIZE=256
AUTOCAST_FLAG="--autocast"
MAX_FOVS=0
BENCHMARK=false
SLURM_MODE=false
DEPENDENCY_JOBS=""
SLURM_PARTITION="dgx"
SLURM_GRES="gpu:71gb:1"
SLURM_TIME="12:00:00"
SLURM_MEM="64G"
SLURM_CPUS=8

while [[ $# -gt 0 ]]; do
    case $1 in
        --cell_line)     CELL_LINE="$2";       shift 2 ;;
        --manifest)      MANIFEST="$2";        shift 2 ;;
        --plate_dir)     PLATE_DIR="$2";       shift 2 ;;
        --data_dir)      DATA_DIR="$2";        shift 2 ;;
        --dataset_base)  DATASET_BASE="$2";    shift 2 ;;
        --output_dir)    OUTPUT_DIR="$2";      shift 2 ;;
        --mmse_count)    MMSE_COUNT="$2";      shift 2 ;;
        --num_posterior) NUM_POSTERIOR="$2";   shift 2 ;;
        --batch_size)    BATCH_SIZE="$2";      shift 2 ;;
        --no_autocast)   AUTOCAST_FLAG="";     shift ;;
        --max_fovs)      MAX_FOVS="$2";        shift 2 ;;
        --benchmark)     BENCHMARK=true;       shift ;;
        --slurm)         SLURM_MODE=true;      shift ;;
        --partition)     SLURM_PARTITION="$2"; shift 2 ;;
        --gres)          SLURM_GRES="$2";      shift 2 ;;
        --time)          SLURM_TIME="$2";      shift 2 ;;
        --mem)           SLURM_MEM="$2";       shift 2 ;;
        --dependency)    DEPENDENCY_JOBS="$2"; shift 2 ;;
        --help) sed -n '2,28p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "ERROR: Unknown option: $1" >&2; exit 1 ;;
    esac
done

[[ -z "$CELL_LINE" ]] && { echo "ERROR: --cell_line is required" >&2; exit 1; }
[[ -z "$MANIFEST" ]] && MANIFEST="/project/cell_paint_mono/splits/cpg0000-jump-pilot/${CELL_LINE}/split_manifest.csv"
[[ -f "$MANIFEST" ]] || { echo "ERROR: manifest not found: ${MANIFEST} (run datasets.sh first)" >&2; exit 1; }
TRAINING_DIR="${DATASET_BASE}/training_dataset_cpg0000_${CELL_LINE}"

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

# Plates: from --plate_dir, else every test plate in the manifest (column plate_dir, batch)
PLATE_DIRS=()
if [[ -n "$PLATE_DIR" ]]; then
    PLATE_DIRS=("$PLATE_DIR")
else
    while IFS= read -r line; do
        PLATE_DIRS+=("$line")
    done < <(awk -F',' 'NR==1{for(i=1;i<=NF;i++){h[$i]=i}} NR>1 && $h["role"]!="train" {print $h["batch"]"/images/"$h["plate_dir"]}' "$MANIFEST" | sort -u | sed "s#^#${DATA_DIR}/#")
fi
[[ ${#PLATE_DIRS[@]} -eq 0 ]] && { echo "ERROR: no plates found" >&2; exit 1; }

if [[ "$BENCHMARK" == true ]]; then
    _activate_conda
    echo "Timing benchmark: 1 FOV of $(basename "${PLATE_DIRS[0]}") (batch ${BATCH_SIZE}, autocast=${AUTOCAST_FLAG:-off})" >&2
    python "${PYTHON_SCRIPT}" --plate_dir "${PLATE_DIRS[0]}" --training_dir "${TRAINING_DIR}" \
        --output_dir "${OUTPUT_DIR}/_benchmark" --manifest "${MANIFEST}" --cell_line "${CELL_LINE}" \
        --mmse_count "${MMSE_COUNT}" --num_posterior_samples 0 --batch_size "${BATCH_SIZE}" ${AUTOCAST_FLAG} \
        --max_fovs 1 --no_resume --data_dir "${DATA_DIR}"
    exit 0
fi

echo "================================================================================" >&2
echo "cpg0000 predict | ${CELL_LINE} | ${#PLATE_DIRS[@]} plates | mmse=${MMSE_COUNT} posterior=${NUM_POSTERIOR} batch=${BATCH_SIZE}" >&2
echo "  manifest:     ${MANIFEST}" >&2
echo "  training_dir: ${TRAINING_DIR}" >&2
echo "  output_dir:   ${OUTPUT_DIR}/${CELL_LINE}" >&2
echo "================================================================================" >&2

DEPENDENCY_DIRECTIVE=""
[[ -n "$DEPENDENCY_JOBS" ]] && DEPENDENCY_DIRECTIVE="#SBATCH --dependency=afterok:${DEPENDENCY_JOBS}"

SUBMITTED=()
for PD in "${PLATE_DIRS[@]}"; do
    PNAME=$(basename "$PD")
    LOG_DIR="${OUTPUT_DIR}/logs/${CELL_LINE}"; mkdir -p "${LOG_DIR}"
    CMD="python ${PYTHON_SCRIPT} --plate_dir ${PD} --training_dir ${TRAINING_DIR} --output_dir ${OUTPUT_DIR} --manifest ${MANIFEST} --cell_line ${CELL_LINE} --mmse_count ${MMSE_COUNT} --num_posterior_samples ${NUM_POSTERIOR} --batch_size ${BATCH_SIZE} ${AUTOCAST_FLAG} --max_fovs ${MAX_FOVS} --metadata_file ${METADATA_FILE} --data_dir ${DATA_DIR}"
    if [[ "$SLURM_MODE" == false ]]; then
        _activate_conda
        eval "${CMD}"
        continue
    fi
    JOB_ID=$(sbatch --parsable <<EOF
#!/bin/bash
#SBATCH --job-name=pred_${CELL_LINE}_${PNAME:0:10}
#SBATCH --output=${LOG_DIR}/predict_${PNAME:0:10}_%j.log
#SBATCH --error=${LOG_DIR}/predict_${PNAME:0:10}_%j.log
#SBATCH --partition=${SLURM_PARTITION}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${SLURM_CPUS}
#SBATCH --mem=${SLURM_MEM}
#SBATCH --gres=${SLURM_GRES}
#SBATCH --time=${SLURM_TIME}
${DEPENDENCY_DIRECTIVE}

echo "Job \${SLURM_JOB_ID}: predict ${CELL_LINE} ${PNAME} | Node: \${SLURMD_NODENAME} | Start: \$(date)"
$(declare -f _activate_conda)
_activate_conda
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
export TORCHINDUCTOR_DISABLE_CUDAGRAPHS=1
${CMD}
EXIT_CODE=\$?
echo "End: \$(date) | Exit: \$EXIT_CODE"
exit \$EXIT_CODE
EOF
    )
    SUBMITTED+=("${JOB_ID}")
    echo "  Submitted job ${JOB_ID}: ${PNAME}" >&2
done
[[ "$SLURM_MODE" == true ]] && { echo "Submitted: ${#SUBMITTED[@]} jobs" >&2; echo "${SUBMITTED[*]//\ /:}"; }
