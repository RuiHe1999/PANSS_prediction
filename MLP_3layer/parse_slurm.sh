#!/bin/bash
#SBATCH --job-name=slurm
#SBATCH -p short
#SBATCH -N 1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16G
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

source activate envname
python parse_slurm.py
