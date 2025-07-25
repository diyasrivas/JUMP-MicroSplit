#!/bin/bash
#SBATCH --partition=gpuq
#SBATCH --job-name=microsplit_pilot_expt3
#SBATCH --output=logs/microsplit_pilot_expt3.out
#SBATCH --error=logs/microsplit_pilot_expt3.err
#SBATCH --gres=gpu:1
#SBATCH --mem=128GB
#SBATCH --time=08:00:00

export PYTHONPATH=/home/diya.srivastava/Desktop/repos/MicroSplit-reproducibility/src:$PYTHONPATH
export PATH=~/miniforge3/bin:$PATH
source ~/.bashrc
mamba activate microsplit_jobs
cd /home/diya.srivastava/Desktop/repos/JUMP-MicroSplit/examples/2D/JUMP

mkdir -p logs
python 5channels_train_pilot.py