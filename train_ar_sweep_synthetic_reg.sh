#!/bin/bash

data_size=(4000)
seeds=(1 2 3 4 5)

# Sweep over each lasso_penalty value
for ds in "${data_size[@]}"; do
  # set num_epochs based on data size
  if [ "$ds" -eq 400 ]; then
    num_epochs=4000
  else
    num_epochs=800
  fi

  for s in "${seeds[@]}"; do
    echo "Dataset: synthetic_setting1_n${ds}_s${s}"

    python train_ar.py \
      --task "reg" \
      --data_path "./data/synthetic_reg/set1_n${ds}_p5_s${s}" \
      --data_name "synthetic" \
      --batch_size 32768 \
      --input_dim 5 \
      --output_dim 1 \
      --num_epochs "$num_epochs" \
      --learning_rate 5e-3 \
      --lasso_penalty 0.05 \
      --group_penalty 0.005 \
      --lasso_norm 1 \
      --group_norm 1 \
      --noise_injection \
      --expr_name "ar_reg_synthetic_setting1_n${ds}_s${s}"
  done
done
