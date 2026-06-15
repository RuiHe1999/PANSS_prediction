#!/bin/bash
#SBATCH --job-name=MLSel
#SBATCH -p medium
#SBATCH -N 1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --output=slurm/%x_%A_%a.out
#SBATCH --error=slurm/%x_%A_%a.err
#SBATCH --array=0-335 # total combinations = 14*3*8=336

mkdir -p logs
export OMP_NUM_THREADS=1

source activate envname

models=("OLS" "Ridge" "BayesRidge" "Lasso" "Elastic" "Huber" "SVR" "GPR" "KNN" "RF" "ET" "GBR" "ABR" "XGB")
features=("AcouPros" "m-HuBERT" "Concat")
symptoms=("P1" "P2" "P3" "N1" "N4" "N6" "G5" "G9")

num_m=${#models[@]}   # 14
num_f=${#features[@]} # 3
num_s=${#symptoms[@]} # 8

i=${SLURM_ARRAY_TASK_ID}

s_index=$(( i % num_s ))
j=$(( i / num_s ))
f_index=$(( j % num_f ))
m_index=$(( j / num_f ))

model=${models[$m_index]}
feature=${features[$f_index]}
symptom=${symptoms[$s_index]}

echo "Task ${SLURM_ARRAY_TASK_ID}: model=${model}, feature=${feature}, symptom=${symptom}"

python model_selection.py --model "$model" --feature "$feature" --symptom "$symptom"
