#!/bin/bash
#SBATCH --partition=gpuq
#SBATCH --job-name=microsplit_1_4channel
#SBATCH --output=logs/microsplit_dna_rna_er_agp_%A.out
#SBATCH --error=logs/microsplit_dna_rna_er_agp_%A.err
#SBATCH --gres=gpu:1
#SBATCH --mem=128GB
#SBATCH --time=04:00:00

# Set up environment
export PYTHONPATH=/home/diya.srivastava/Desktop/repos/MicroSplit-reproducibility/src:$PYTHONPATH
export PATH=~/miniforge3/bin:$PATH
source ~/.bashrc
mamba activate microsplit_jobs

# Move to the correct directory
cd /home/diya.srivastava/Desktop/repos/JUMP-MicroSplit/examples/2D/JUMP

# Create logs directory if it doesn't exist
mkdir -p logs

# Print info about which dataset we're processing
echo "Processing 4-channel combination: DNA, RNA, ER, AGP"

# Run the training script
python remaining_4channeltrain.py
