#!/bin/bash
#SBATCH --partition=gpuq
#SBATCH --job-name=microsplit_5channels
#SBATCH --output=logs/microsplit_5ch_%A_%a.out
#SBATCH --error=logs/microsplit_5ch_%A_%a.err
#SBATCH --gres=gpu:1
#SBATCH --mem=128GB
#SBATCH --time=06:00:00

export PYTHONPATH=/home/diya.srivastava/Desktop/repos/MicroSplit-reproducibility/src:$PYTHONPATH
export PATH=~/miniforge3/bin:$PATH
source ~/.bashrc
mamba activate microsplit_jobs
cd /home/diya.srivastava/Desktop/repos/JUMP-MicroSplit/examples/2D/JUMP

mkdir -p logs

# Run the training script - channels are now hardcoded in the script
python 5channels_train.py
