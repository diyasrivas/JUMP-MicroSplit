#!/bin/bash
#SBATCH --partition=gpuq
#SBATCH --job-name=rand-train
#SBATCH --output=logs/rand-train6.out
#SBATCH --error=logs/rand-train6.err
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --time=30:00:00

export PYTHONPATH=/home/diya.srivastava/Desktop/repos/MicroSplit-reproducibility/src:$PYTHONPATH
export PATH=~/miniforge3/bin:$PATH
source ~/.bashrc
mamba activate microsplit_jobs
cd /home/diya.srivastava/Desktop/repos/JUMP-MicroSplit/examples/2D/JUMP_rep_ORF

mkdir -p logs
python 02_train_rand6.py