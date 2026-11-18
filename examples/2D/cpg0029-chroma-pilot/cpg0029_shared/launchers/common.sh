#!/bin/bash
# Helpers sourced by the cpg0029 launchers.

EXP_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../.." && pwd )"
REPO_SRC="$( cd "${EXP_DIR}/../../../src" && pwd )"
DYE_SETS="standard_cp mitobrilliant phalloidin_400ls post_chromalive_cp chromalive"

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
            # an active virtualenv (e.g. CellPaintMONO/.venv) would otherwise shadow conda's python
            unset VIRTUAL_ENV
            export PATH="${CONDA_PREFIX}/bin:${PATH}"
            export PYTHONPATH="${REPO_SRC}:${PYTHONPATH:-}"
            return
        fi
    done
    echo "ERROR: Conda not found" >&2; exit 1
}

_check_dye_set() {
    [[ " ${DYE_SETS} " == *" $1 "* ]] || { echo "ERROR: --dye_set must be one of: ${DYE_SETS}" >&2; exit 1; }
}

# _dye_set_attr NAME ATTR: prints one attribute of the dye set (stdlib only, no conda needed)
_dye_set_attr() {
    python3 -c "
import sys; sys.path.insert(0, '${EXP_DIR}')
from cpg0029_shared.dye_sets import get_dye_set
v = getattr(get_dye_set('$1'), '$2')
print(' '.join(v) if isinstance(v, list) else v)"
}

# _submit NAME LOG PARTITION CPUS MEM TIME GRES DEPENDENCY COMMAND
# Submits COMMAND as a SLURM job and prints its job id.
_submit() {
    local name=$1 log=$2 partition=$3 cpus=$4 mem=$5 time=$6 gres=$7 dep=$8 cmd=$9
    local gres_line="" dep_line=""
    [[ -n "$gres" ]] && gres_line="#SBATCH --gres=${gres}"
    [[ -n "$dep" ]] && dep_line="#SBATCH --dependency=afterok:${dep}"
    mkdir -p "$(dirname "$log")"
    sbatch --parsable <<SBATCH_EOF
#!/bin/bash
#SBATCH --job-name=${name}
#SBATCH --output=${log}
#SBATCH --error=${log}
#SBATCH --partition=${partition}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${cpus}
#SBATCH --mem=${mem}
#SBATCH --time=${time}
${gres_line}
${dep_line}

echo "Job \${SLURM_JOB_ID}: ${name} | Node: \${SLURMD_NODENAME} | Start: \$(date)"
$(declare -f _activate_conda)
REPO_SRC="${REPO_SRC}"
_activate_conda
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
export TORCHINDUCTOR_DISABLE_CUDAGRAPHS=1
set -e
${cmd}
echo "End: \$(date)"
SBATCH_EOF
}
