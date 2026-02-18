#!/bin/bash

# Training script for NIMO on PMLB datasets

# Dataset names and their corresponding feature dimensions
datasets=("197_cpu_act" "201_pol" "225_puma8NH" "227_cpu_small" "294_satellite_image" "4544_GeographicalOriginalofMusic" "503_wind" "529_pollen" "562_cpu_small" "573_cpu_act")
feat_nums=(21 48 8 12 36 117 14 4 12 21)

# datasets=("225_puma8NH_poly")
# feat_nums=(44)

# Training hyperparameters (adjust as needed)
LEARNING_RATE=5e-3
NUM_EPOCHS=1000
BATCH_SIZE=16384

# Dropout values to test
DROPOUT_VALUES=(0.5 0.3 0.1)

# Loop through all datasets
for i in "${!datasets[@]}"; do
    dataset="${datasets[$i]}"
    n_features="${feat_nums[$i]}"

    # Loop through dropout values
    for dropout in "${DROPOUT_VALUES[@]}"; do
        echo "=========================================="
        echo "Training NN on dataset: ${dataset}"
        echo "Number of features: ${n_features}"
        echo "Dropout: ${dropout}"
        echo "=========================================="

        python train_simple_nn.py \
            --task "reg" \
            --data_path "./data/pmlb/${dataset}" \
            --data_name "synthetic" \
            --batch_size ${BATCH_SIZE} \
            --input_dim ${n_features} \
            --output_dim 1 \
            --num_epochs ${NUM_EPOCHS} \
            --lr ${LEARNING_RATE} \
            --dropout ${dropout} \
            --expr_name "nimo_pmlb_nn_${dataset}_dropout${dropout}"

        echo ""
        echo "Finished training on ${dataset} with dropout ${dropout}"
        echo ""
    done
done

echo "=========================================="
echo "All training jobs completed!"
echo "=========================================="
