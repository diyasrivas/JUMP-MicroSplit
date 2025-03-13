#!/bin/bash
#SBATCH --partition=gpuq
#SBATCH --job-name=microsplit_4ch
#SBATCH --output=logs/microsplit_4ch_%A_%a.out
#SBATCH --error=logs/microsplit_4ch_%A_%a.err
#SBATCH --gres=gpu:1
#SBATCH --mem=128GB
#SBATCH --time=04:00:00
#SBATCH --array=0-4

export PYTHONPATH=/home/diya.srivastava/Desktop/repos/MicroSplit-reproducibility/src:$PYTHONPATH
export PATH=~/miniforge3/bin:$PATH
source ~/.bashrc
mamba activate microsplit_jobs
cd /home/diya.srivastava/Desktop/repos/JUMP-MicroSplit/examples/2D/JUMP

mkdir -p logs

# Print combination info
echo "Processing 4-channel combination index: ${SLURM_ARRAY_TASK_ID}"

# Run the training script with the combination index from the array job
python 4channels_train.py --index ${SLURM_ARRAY_TASK_ID} --epochs 10 --batch-size 16
