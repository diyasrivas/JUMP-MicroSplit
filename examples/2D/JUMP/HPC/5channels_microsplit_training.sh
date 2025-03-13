#!/bin/bash
#SBATCH --partition=gpuq
#SBATCH --job-name=microsplit_5channels
#SBATCH --output=logs/microsplit_5ch_%A_%a.out
#SBATCH --error=logs/microsplit_5ch_%A_%a.err
#SBATCH --array=0-9
#SBATCH --gres=gpu:1
#SBATCH --mem=128GB
#SBATCH --time=06:00:00

export PYTHONPATH=/home/diya.srivastava/Desktop/repos/MicroSplit-reproducibility/src:$PYTHONPATH
export PATH=~/miniforge3/bin:$PATH
source ~/.bashrc
mamba activate microsplit_jobs
cd /home/diya.srivastava/Desktop/repos/JUMP-MicroSplit/examples/2D/JUMP

# Create logs directory
mkdir -p logs

# Define all 5-channel combinations
COMBINATIONS=(
  "DNA, RNA, ER, AGP, Mito"
)

# Get the current combination based on array index
COMBO=${COMBINATIONS[$SLURM_ARRAY_TASK_ID]}

# Format channels for directory name
DIR_NAME=$(echo $COMBO | tr ',' '_' | tr '[:upper:]' '[:lower:]')
DATASET_DIR="experiments/5_channels/${DIR_NAME}"
OUTPUT_DIR="training_results/5_channels/${DIR_NAME}"

# Create output directory
mkdir -p $OUTPUT_DIR

# Print details for logging
echo "Running training for channels: $COMBO"
echo "Dataset directory: $DATASET_DIR"
echo "Output directory: $OUTPUT_DIR"

# Run the training script with the selected channels
python 02JUMP_train.py \
  --channels "$COMBO" \
  --dataset_dir "$DATASET_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --task_id "$SLURM_ARRAY_TASK_ID" \
  --epochs 10
