#!/bin/bash
# Steps 0+1 for one cpg0029 dye set: split manifest, then training dataset.
# Run from a dye-set folder (./datasets.sh) or pass --dye_set.
#
# Usage:
#   ./datasets.sh                 # interactive (CPU, minutes)
#   ./datasets.sh --slurm         # as a SLURM job
#   ./datasets.sh --skip_split    # reuse the existing manifest
#
# Options:
#   --dye_set NAME     standard_cp | mitobrilliant | phalloidin_400ls | post_chromalive_cp | chromalive
#   --n_train N        training FOVs (default: 300)
#   --seed N           (default: 42)
#   --skip_split       do not (re)run 0_split.py
#   --slurm            submit as a SLURM job
#   --partition NAME   (default: cpu)
#   --time HH:MM:SS    (default: 4:00:00)
#   --mem XG           (default: 32G)
set -euo pipefail
source "$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )/common.sh"

DYE_SET=""; N_TRAIN=300; SEED=42; SKIP_SPLIT=false; SLURM_MODE=false
PARTITION="cpu"; TIME="4:00:00"; MEM="32G"
while [[ $# -gt 0 ]]; do
    case $1 in
        --dye_set)    DYE_SET="$2";   shift 2 ;;
        --n_train)    N_TRAIN="$2";   shift 2 ;;
        --seed)       SEED="$2";      shift 2 ;;
        --skip_split) SKIP_SPLIT=true; shift ;;
        --slurm)      SLURM_MODE=true; shift ;;
        --partition)  PARTITION="$2"; shift 2 ;;
        --time)       TIME="$2";      shift 2 ;;
        --mem)        MEM="$2";       shift 2 ;;
        --help) sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "ERROR: Unknown option: $1" >&2; exit 1 ;;
    esac
done
_check_dye_set "$DYE_SET"
DIR="${EXP_DIR}/${DYE_SET}"
TRAINING_DIR=$(_dye_set_attr "$DYE_SET" training_dir)

CMD=""
[[ "$SKIP_SPLIT" == false ]] && CMD="python ${DIR}/0_split.py --n_train ${N_TRAIN} --seed ${SEED}"$'\n'
CMD+="python ${DIR}/1_datasets.py"

echo "cpg0029 ${DYE_SET}: split + dataset -> ${TRAINING_DIR}" >&2
if [[ "$SLURM_MODE" == false ]]; then
    _activate_conda
    while IFS= read -r line; do eval "$line"; done <<< "$CMD"
    exit 0
fi
_submit "cpg0029_ds_${DYE_SET}" "${TRAINING_DIR}/logs/datasets_%j.log" \
    "$PARTITION" 4 "$MEM" "$TIME" "" "" "$CMD"
