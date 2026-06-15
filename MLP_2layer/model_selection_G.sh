#!/bin/bash
#SBATCH --job-name=mlp2G
#SBATCH -p short
#SBATCH -N 1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --output=slurm/%x_%A_%a.out
#SBATCH --error=slurm/%x_%A_%a.err
#SBATCH --array=0-215   # total combinations = 2*3*3*4*3=216

export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

source activate envname

# Define hyperparameters
features=("AcouPros" "m-HuBERT" "Concat")
symptoms=("G5" "G9")
lr_list=(5e-4 3e-4 1e-3 3e-3)
decay_list=(0 1e-3 3e-3)
dropouts=(0.2 0.3 0.4)

# Create combination array
combs=()
for s in "${symptoms[@]}"; do
  for f in "${features[@]}"; do
    for lr in "${lr_list[@]}"; do
      for wd in "${decay_list[@]}"; do
        for d in "${dropouts[@]}"; do
          combs+=("$s|$f|$lr|$wd|$d")
        done
      done
    done
  done
done

# Get the combination for this array task
idx=$SLURM_ARRAY_TASK_ID
if [ $idx -ge ${#combs[@]} ]; then
    echo "Task index $idx exceeds total combinations. Skipping."
    exit 0
fi

IFS="|" read -r symptom feature lr decay dropout <<< "${combs[$idx]}"

# Loop over dim
total_dim=49  # dim from 2 to 50
count=0

for dim in $(seq 2 50); do
    count=$((count + 1))
    echo "[${count}/${total_dim}] [RUNNING] ${symptom} ${feature} | Dim: ${dim}, Dropout: ${dropout}, LR: ${lr}, L2: ${decay}"
    
    python MLP_2layer.py \
        --item "${symptom}" \
        --feat "${feature}" \
        --dim "${dim}" \
        --dropout "${dropout}" \
        --lr "${lr}" \
        --decay "${decay}" \
        --save
done
