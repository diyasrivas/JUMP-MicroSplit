#!/bin/bash
# MicroSplit training for cpg0029-chroma-pilot (Step 3).
#
# Usage:
#   ./train.sh
#   ./train.sh --test          # 2-epoch sanity check
#   ./train.sh --dependency "12345:12346"   # after noise model jobs
#
# Options:
#   --dataset_dir DIR            (default: /project/cell_paint_mono/training_datasets/training_dataset_cpg0029)
#   --epochs N                   Training epochs (default: 100)
#   --batch_size N               (default: 64)
#   --num_workers N              DataLoader workers (default: 8)
#   --check_val_every N          (default: 5)
#   --early_stopping_patience N  (default: 50)
#   --train_grid_size N          (default: 64)
#   --no-compile                 Disable torch.compile
#   --test                       2 epochs on 2 frames, 1-hour walltime
#   --time HH:MM:SS              SLURM walltime override (default: 24:00:00)
#   --mem XG                     (default: 32G)
#   --partition NAME             (default: dgx)
#   --gres GRES                  (default: gpu:71gb:1)
#   --dependency "j1:j2"         Colon-separated SLURM job dependencies

set -euo pipefail

SUBMIT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PYTHON_SCRIPT="${SUBMIT_DIR}/3_train.py"
DATASET_DIR="/project/cell_paint_mono/training_datasets/training_dataset_cpg0029"

NUM_EPOCHS=100
BATCH_SIZE=64
NUM_WORKERS=8
CHECK_VAL_EVERY=5
EARLY_STOPPING_PATIENCE=50
COMPILE_FLAG="--compile"
TRAIN_GRID_SIZE=64
TEST_MODE=false
DEPENDENCY_JOBS=""
WANDB_ENTITY="juglab"
WANDB_PROJECT="JUMP-MicroSplit"
WANDB_FLAG=""

SLURM_PARTITION="dgx"
SLURM_TIME="24:00:00"
SLURM_MEM="32G"
SLURM_CPUS=8
SLURM_GRES="gpu:71gb:1"

while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset_dir)             DATASET_DIR="$2";               shift 2 ;;
        --epochs)                  NUM_EPOCHS="$2";                 shift 2 ;;
        --batch_size)              BATCH_SIZE="$2";                 shift 2 ;;
        --num_workers)             NUM_WORKERS="$2";                shift 2 ;;
        --check_val_every)         CHECK_VAL_EVERY="$2";            shift 2 ;;
        --early_stopping_patience) EARLY_STOPPING_PATIENCE="$2";   shift 2 ;;
        --train_grid_size)         TRAIN_GRID_SIZE="$2";            shift 2 ;;
        --wandb_entity)            WANDB_ENTITY="$2";               shift 2 ;;
        --wandb_project)           WANDB_PROJECT="$2";              shift 2 ;;
        --no_wandb)                WANDB_FLAG="--no_wandb";         shift ;;
        --no-compile)              COMPILE_FLAG="";                 shift ;;
        --test)
            TEST_MODE=true; NUM_EPOCHS=2; CHECK_VAL_EVERY=1
            SLURM_TIME="1:00:00"; shift ;;
        --time)      SLURM_TIME="$2";      shift 2 ;;
        --mem)       SLURM_MEM="$2";       shift 2 ;;
        --partition) SLURM_PARTITION="$2"; shift 2 ;;
        --gres)      SLURM_GRES="$2";      shift 2 ;;
        --dependency) DEPENDENCY_JOBS="$2"; shift 2 ;;
        --help)
            sed -n '2,26p' "$0" | sed 's/^# \{0,1\}//'
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

MODE_NAME=$([[ "$TEST_MODE" = true ]] && echo "TEST" || echo "PROD")
DEPENDENCY_DIRECTIVE=""
[[ -n "$DEPENDENCY_JOBS" ]] && DEPENDENCY_DIRECTIVE="#SBATCH --dependency=afterok:${DEPENDENCY_JOBS}"
TEST_FLAG=$([[ "$TEST_MODE" = true ]] && echo "--test_mode" || echo "")

echo "================================================================================" >&2
echo "${MODE_NAME}: cpg0029 | epochs=${NUM_EPOCHS} batch=${BATCH_SIZE}" >&2
echo "  dataset_dir: ${DATASET_DIR}" >&2
echo "================================================================================" >&2

LOG_DIR="${DATASET_DIR}/logs/train"
mkdir -p "${LOG_DIR}"

JOB_ID=$(sbatch --parsable <<EOF
#!/bin/bash
#SBATCH --job-name=${MODE_NAME}_train_cpg0029
#SBATCH --output=${LOG_DIR}/${MODE_NAME,,}_cpg0029_%j.log
#SBATCH --error=${LOG_DIR}/${MODE_NAME,,}_cpg0029_%j.log
#SBATCH --partition=${SLURM_PARTITION}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${SLURM_CPUS}
#SBATCH --mem=${SLURM_MEM}
#SBATCH --gres=${SLURM_GRES}
#SBATCH --time=${SLURM_TIME}
${DEPENDENCY_DIRECTIVE}

echo "Job \${SLURM_JOB_ID}: ${MODE_NAME} train cpg0029 | Node: \${SLURMD_NODENAME} | Start: \$(date)"
$(declare -f _activate_conda)
_activate_conda

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
export TORCHINDUCTOR_DISABLE_CUDAGRAPHS=1

python "${PYTHON_SCRIPT}" \
    --dataset_dir              "${DATASET_DIR}" \
    --num_epochs               ${NUM_EPOCHS} \
    --batch_size               ${BATCH_SIZE} \
    --num_workers              ${NUM_WORKERS} \
    --check_val_every_n_epoch  ${CHECK_VAL_EVERY} \
    --early_stopping_patience  ${EARLY_STOPPING_PATIENCE} \
    --train_grid_size          ${TRAIN_GRID_SIZE} \
    --wandb_entity             "${WANDB_ENTITY}" \
    --wandb_project            "${WANDB_PROJECT}" \
    ${TEST_FLAG} ${COMPILE_FLAG} ${WANDB_FLAG}

EXIT_CODE=\$?
echo "End: \$(date) | Exit: \$EXIT_CODE"
exit \$EXIT_CODE
EOF
)

echo "Submitted job ${JOB_ID}: ${MODE_NAME} train cpg0029" >&2
echo "${JOB_ID}"
