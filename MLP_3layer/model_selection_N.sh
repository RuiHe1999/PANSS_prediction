#!/bin/bash
#SBATCH --job-name=mlp3N
#SBATCH -p medium
#SBATCH -N 1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --output=slurm/%x_%A_%a.out
#SBATCH --error=slurm/%x_%A_%a.err
#SBATCH --array=0-323   # total combinations = 3*3*3*4*3=324

export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

source activate envname

# Define hyperparameters
features=("AcouPros" "m-HuBERT" "Concat")
symptoms=("N1" "N4" "N6")
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
count=0
total_pairs=625

for dim1 in $(seq 2 50); do
    max_dim2=$((dim1 / 2))
    for (( dim2=1; dim2<=max_dim2; dim2++ )); do
        count=$((count + 1))
        echo "[${count}/${total_pairs}] [RUNNING] ${symptom} ${feature} | Dim1: ${dim1}, Dim2: ${dim2}, Dropout: ${dropout}, LR: ${lr}, L2: ${decay}"
        
        python MLP_3layer.py \
            --item "${symptom}" \
            --feat "${feature}" \
            --dim_1 "${dim1}" \
            --dim_2 "${dim2}" \
            --dropout "${dropout}" \
            --lr "${lr}" \
            --decay "${decay}" \
            --save
    done
done