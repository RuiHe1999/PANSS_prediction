#!/bin/bash
#SBATCH --job-name=run
#SBATCH -p medium
#SBATCH -N 1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

source activate envname

python MLP.py --item "P3" --feat "AcouPros" --dim 35 --dropout 0.7 --lr 0.001 --decay 0 


