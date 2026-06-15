#!/bin/bash
#SBATCH --job-name=compar
#SBATCH -p short
#SBATCH -N 1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --output=results/logs/comparison_%j.out
#SBATCH --error=results/logs/comparison_%j.err

source activate envname
python model_comparison.py
