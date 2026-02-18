#!/bin/bash

for ds_idx in {1..9}; do
    data_path="./data/boston_${ds_idx}"
    
    echo "Training SimpleNN for boston_${ds_idx}"

    python train_simple_nn.py \
        --task "reg" \
        --data_path "$data_path" \
        --data_name "synthetic" \
        --batch_size 1024 \
        --input_dim 13 \
        --output_dim 1 \
        --lr 5e-3 \
        --dropout 0.6 \
        --num_epochs 3000 \
        --expr_name "simplenn_reg_boston_${ds_idx}"
done