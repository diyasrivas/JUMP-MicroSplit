#!/bin/bash
#SBATCH --partition=gpuq
#SBATCH --job-name=rand_prediction
#SBATCH --output=logs/rand3_%j.out
#SBATCH --error=logs/rand3_%j.err
#SBATCH --gres=gpu:1
#SBATCH --mem=128GB            
#SBATCH --cpus-per-task=8       
#SBATCH --time=1:00:00

# Create logs directory 
mkdir -p logs

# Load only CUDA module
module load cuda/11.8

# Activate conda environment
export PYTHONPATH=/home/diya.srivastava/Desktop/repos/MicroSplit-reproducibility/src:$PYTHONPATH
export PATH=~/miniforge3/bin:$PATH
source ~/.bashrc
mamba activate microsplit_jobs

# Navigate to correct directory
cd /home/diya.srivastava/Desktop/repos/JUMP-MicroSplit/examples/2D/JUMP_rep_datasets

# Set memory-related environment variables
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Enable CUDA memory optimization
export CUDA_LAUNCH_BLOCKING=0

# Run prediction script
python rand3_predict.py --test-data /home/diya.srivastava/Desktop/repos/JUMP-MicroSplit/examples/2D/JUMP_rep_datasets/test_data/5_channels/dna_rna_er_agp_mito