#!/bin/bash

# Define array of lasso_penalty values to sweep over
# lasso_values=(0.0001 0.0005 0.001 0.005 0.01 0.05 0.1)
lasso_values=(0.0001 0.0005)

# Loop over dataset indices 1 through 5
for ds_idx in {1..5}; do
  data_path="./data/superconduct_${ds_idx}"

  # Sweep over each lasso_penalty value
  for lasso_penalty in "${lasso_values[@]}"; do
    group_penalty=$(echo "$lasso_penalty / 10" | bc -l)

    echo "Dataset: superconduct_${ds_idx} | lasso_penalty=$lasso_penalty | group_penalty=$group_penalty"

    python train_ar.py \
      --task "reg" \
      --data_path "$data_path" \
      --data_name "synthetic" \
      --batch_size 16384 \
      --input_dim 81 \
      --output_dim 1 \
      --num_epochs 800 \
      --learning_rate 5e-3 \
      --lasso_penalty "$lasso_penalty" \
      --group_penalty "$group_penalty" \
      --lasso_norm 1 \
      --group_norm 1 \
      --noise_injection \
      --expr_name "ar_reg_superconduct_${ds_idx}_lp_${lasso_penalty}"
  done
done