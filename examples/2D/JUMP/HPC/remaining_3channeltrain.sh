#!/bin/bash
#SBATCH --partition=gpuq
#SBATCH --job-name=microsplit_dna_agp_mito
#SBATCH --output=logs/microsplit_dna_agp_mito_%A.out
#SBATCH --error=logs/microsplit_dna_agp_mito_%A.err
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
echo "Processing 3-channel combination: DNA, AGP, Mito"

# Run the training script
python remaining_3channeltrain.py
