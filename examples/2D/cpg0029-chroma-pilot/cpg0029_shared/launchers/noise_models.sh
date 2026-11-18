#!/bin/bash
# Step 2 for one cpg0029 dye set: one SLURM job per unmixed channel.
# Prints the job ids joined by ':' (pass to train.sh --dependency).
#
# Options:
#   --dye_set NAME        standard_cp | mitobrilliant | phalloidin_400ls | post_chromalive_cp | chromalive
#   --channels "A B"      subset of channels (default: all unmixed channels of the dye set)
#   --dependency "j1:j2"  SLURM afterok dependency
#   --partition NAME      (default: dgx)
#   --gres GRES           (default: gpu:71gb:1)
#   --time HH:MM:SS       (default: 2:00:00)
#   --mem XG              (default: 16G)
set -euo pipefail
source "$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )/common.sh"

DYE_SET=""; CHANNELS=""; DEP=""
PARTITION="dgx"; GRES="gpu:71gb:1"; TIME="2:00:00"; MEM="16G"
while [[ $# -gt 0 ]]; do
    case $1 in
        --dye_set)    DYE_SET="$2";   shift 2 ;;
        --channels)   CHANNELS="$2";  shift 2 ;;
        --dependency) DEP="$2";       shift 2 ;;
        --partition)  PARTITION="$2"; shift 2 ;;
        --gres)       GRES="$2";      shift 2 ;;
        --time)       TIME="$2";      shift 2 ;;
        --mem)        MEM="$2";       shift 2 ;;
        --help) sed -n '2,14p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "ERROR: Unknown option: $1" >&2; exit 1 ;;
    esac
done
_check_dye_set "$DYE_SET"
TRAINING_DIR=$(_dye_set_attr "$DYE_SET" training_dir)
CHANNELS=${CHANNELS:-$(_dye_set_attr "$DYE_SET" channels)}

IDS=()
for CH in $CHANNELS; do
    ID=$(_submit "nm_cpg0029_${DYE_SET}_${CH}" "${TRAINING_DIR}/logs/noise_models/nm_${CH}_%j.log" \
        "$PARTITION" 4 "$MEM" "$TIME" "$GRES" "$DEP" \
        "python ${EXP_DIR}/${DYE_SET}/2_noisemodels.py --channel ${CH}")
    echo "Submitted ${ID}: noise model ${DYE_SET} ${CH}" >&2
    IDS+=("$ID")
done
(IFS=:; echo "${IDS[*]}")
