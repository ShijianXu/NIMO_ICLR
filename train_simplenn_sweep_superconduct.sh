#!/bin/bash

# Loop over dataset indices 1 through 5
for ds_idx in {1..5}; do
  data_path="./data/superconduct_${ds_idx}"

    echo "Training SimpleNN for superconduct_${ds_idx}"

    python train_simple_nn.py \
        --task "reg" \
        --data_path "$data_path" \
        --data_name "synthetic" \
        --batch_size 16384 \
        --input_dim 81 \
        --output_dim 1 \
        --lr 5e-3 \
        --dropout 0.1 \
        --num_epochs 1500 \
        --expr_name "simplenn_reg_superconduct_${ds_idx}"
done