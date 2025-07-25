#!/bin/bash
#SBATCH --partition=gpuq
#SBATCH --job-name=microsplit_pilot_expt3_2
#SBATCH --output=logs/microsplit_pilot_expt3_2.out
#SBATCH --error=logs/microsplit_pilot_expt3_2.err
#SBATCH --gres=gpu:1
#SBATCH --mem=256GB
#SBATCH --time=90:00:00

export PYTHONPATH=/home/diya.srivastava/Desktop/repos/MicroSplit-reproducibility/src:$PYTHONPATH
export PATH=~/miniforge3/bin:$PATH
source ~/.bashrc
mamba activate microsplit_jobs
cd /home/diya.srivastava/Desktop/repos/JUMP-MicroSplit/examples/2D/JUMP

mkdir -p logs
python 5channels_train_pilot_2.py