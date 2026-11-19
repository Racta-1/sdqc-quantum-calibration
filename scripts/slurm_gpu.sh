#!/bin/bash
# Example SLURM job for a GPU cluster (e.g., UA HPC). Adjust partition/account/module names for your cluster.
#SBATCH --job-name=sdqc
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/sdqc-%j.out

mkdir -p logs
# module load cuda            # if your cluster requires it
source ~/miniconda3/etc/profile.d/conda.sh
conda activate sdqc

python scripts/check_gpu.py
python scripts/run_experiment.py \
    --feature-device cuda \
    --sim-device cpu \
    --jobs 6 \
    --seeds 0 1 2 3 4 5 6 7 8 9 \
    --out results/v5_1_10seeds
