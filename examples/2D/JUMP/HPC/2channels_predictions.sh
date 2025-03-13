#!/bin/bash
#SBATCH --partition=gpuq
#SBATCH --job-name=microsplit_predict
#SBATCH --output=logs/microsplit_predict_%A_%a.out
#SBATCH --error=logs/microsplit_predict_%A_%a.err
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB
#SBATCH --time=04:00:00
#SBATCH --array=0-9  # For 10 2-channel combinations

# Set the repository path
REPO_PATH="/home/diya.srivastava/Desktop/repos/JUMP-MicroSplit"

# Configure environment
export PYTHONPATH="${REPO_PATH}/src:$PYTHONPATH"
export PATH=~/miniforge3/bin:$PATH
source ~/.bashrc
conda activate microsplit_jobs

# Create logs directory
mkdir -p logs

# Get dataset names
DATASETS=(
    "dna_rna"
    "dna_er"
    "dna_agp"
    "dna_mito"
    "rna_er"
    "rna_agp"
    "rna_mito"
    "er_agp"
    "er_mito"
    "agp_mito"
)

# Get the dataset for this array job
DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID]}
echo "Processing dataset: $DATASET"

# Run the script with the full repository path
cd ${REPO_PATH}/examples/2D/JUMP
python 2channels_predictions.py --repo-dir "${REPO_PATH}" --num-channels 2 --dataset "$DATASET" 
