#!/bin/bash
# Step 4 for one cpg0029 dye set: predict the test FOVs (MMSE 50) and write the
# CellProfiler load_data CSVs. Resumes where a previous job stopped.
#
# Options:
#   --dye_set NAME         standard_cp | mitobrilliant | phalloidin_400ls | post_chromalive_cp | chromalive
#   --benchmark            predict 1 FOV only, to time it (written to a separate folder)
#   --dependency "j1"      SLURM afterok dependency (e.g. the training job)
#   --no_autocast          disable bfloat16 autocast
#   --partition NAME       (default: dgx)
#   --gres GRES            (default: gpu:71gb:1)
#   --time HH:MM:SS        (default: 24:00:00)
#   --mem XG               (default: 64G)
#   Any other option is passed to 4_predict.py (e.g. --batch_size 512).
set -euo pipefail
source "$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )/common.sh"

DYE_SET=""; DEP=""; AUTOCAST="--autocast"; MODE="predict"
PARTITION="dgx"; GRES="gpu:71gb:1"; TIME="24:00:00"; MEM="64G"
PASS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --dye_set)     DYE_SET="$2";   shift 2 ;;
        --benchmark)   MODE="benchmark"; TIME="1:00:00"; shift ;;
        --dependency)  DEP="$2";       shift 2 ;;
        --no_autocast) AUTOCAST="";    shift ;;
        --partition)   PARTITION="$2"; shift 2 ;;
        --gres)        GRES="$2";      shift 2 ;;
        --time)        TIME="$2";      shift 2 ;;
        --mem)         MEM="$2";       shift 2 ;;
        --help) sed -n '2,16p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *)             PASS+=("$1");   shift ;;
    esac
done
_check_dye_set "$DYE_SET"
OUT=$(_dye_set_attr "$DYE_SET" prediction_dir)
[[ "$MODE" == benchmark ]] && { OUT="${OUT}_benchmark"; PASS+=(--max_fovs 1 --no_resume); }

ID=$(_submit "${MODE}_cpg0029_${DYE_SET}" "${OUT}/logs/${MODE}_%j.log" \
    "$PARTITION" 8 "$MEM" "$TIME" "$GRES" "$DEP" \
    "python ${EXP_DIR}/${DYE_SET}/4_predict.py --output_dir ${OUT} --mmse_count 50 ${AUTOCAST} ${PASS[*]:-}")
echo "Submitted ${ID}: ${MODE} ${DYE_SET} -> ${OUT}" >&2
echo "$ID"
