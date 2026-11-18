#!/bin/bash
# Step 3 for one cpg0029 dye set: MicroSplit training (logged to W&B).
#
# Options:
#   --dye_set NAME         standard_cp | mitobrilliant | phalloidin_400ls | post_chromalive_cp | chromalive
#   --test                 2 epochs on 2 frames, 1-hour walltime
#   --dependency "j1:j2"   SLURM afterok dependency (e.g. the noise model jobs)
#   --no_wandb             disable W&B
#   --no-compile           disable torch.compile
#   --partition NAME       (default: dgx)
#   --gres GRES            (default: gpu:71gb:1)
#   --time HH:MM:SS        (default: 48:00:00)
#   --mem XG               (default: 64G)
#   Any other option is passed to 3_train.py (e.g. --num_epochs 50 --batch_size 32).
set -euo pipefail
source "$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )/common.sh"

DYE_SET=""; DEP=""; EXTRA="--compile"; MODE="prod"
PARTITION="dgx"; GRES="gpu:71gb:1"; TIME="48:00:00"; MEM="64G"
PASS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --dye_set)    DYE_SET="$2";   shift 2 ;;
        --test)       MODE="test"; PASS+=(--test_mode); TIME="1:00:00"; shift ;;
        --dependency) DEP="$2";       shift 2 ;;
        --no-compile) EXTRA="";       shift ;;
        --partition)  PARTITION="$2"; shift 2 ;;
        --gres)       GRES="$2";      shift 2 ;;
        --time)       TIME="$2";      shift 2 ;;
        --mem)        MEM="$2";       shift 2 ;;
        --help) sed -n '2,16p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *)            PASS+=("$1");   shift ;;
    esac
done
_check_dye_set "$DYE_SET"
TRAINING_DIR=$(_dye_set_attr "$DYE_SET" training_dir)

ID=$(_submit "${MODE}_cpg0029_${DYE_SET}" "${TRAINING_DIR}/logs/train/${MODE}_%j.log" \
    "$PARTITION" 8 "$MEM" "$TIME" "$GRES" "$DEP" \
    "python ${EXP_DIR}/${DYE_SET}/3_train.py ${EXTRA} ${PASS[*]:-}")
echo "Submitted ${ID}: ${MODE} training ${DYE_SET}" >&2
echo "$ID"
