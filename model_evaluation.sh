#!/bin/bash
#SBATCH --job-name=evaluate
#SBATCH -p short
#SBATCH -N 1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --output=results/logs/evaluation_%j.out
#SBATCH --error=results/logs/evaluation_%j.err

source activate envname
python model_evaluation.py
