#!/bin/bash

# Training script for NIMO on PMLB datasets

# Dataset names and their corresponding feature dimensions
datasets=("197_cpu_act" "201_pol" "225_puma8NH" "227_cpu_small" "294_satellite_image" "4544_GeographicalOriginalofMusic" "503_wind" "529_pollen" "562_cpu_small" "573_cpu_act")
feat_nums=(21 48 8 12 36 117 14 4 12 21)


# datasets=("225_puma8NH_poly")
# feat_nums=(44)

# Training hyperparameters (adjust as needed)
LEARNING_RATE=5e-3
LASSO_NORM=1
GROUP_NORM=1
NUM_EPOCHS=1000
BATCH_SIZE=16384

# Penalty values to test
LASSO_PENALTIES=(0.005 0.01 0.05 0.1)
GROUP_PENALTIES=(0.005 0.01 0.05 0.1)

# Loop through all datasets
for i in "${!datasets[@]}"; do
    dataset="${datasets[$i]}"
    n_features="${feat_nums[$i]}"

    # Loop through lasso penalty values
    for lasso_penalty in "${LASSO_PENALTIES[@]}"; do
        # Loop through group penalty values
        for group_penalty in "${GROUP_PENALTIES[@]}"; do
            echo "=========================================="
            echo "Training NIMO on dataset: ${dataset}"
            echo "Number of features: ${n_features}"
            echo "Lasso penalty: ${lasso_penalty}"
            echo "Group penalty: ${group_penalty}"
            echo "=========================================="

            python train_ar.py \
                --task "reg" \
                --data_path "./data/pmlb/${dataset}" \
                --data_name "synthetic" \
                --batch_size ${BATCH_SIZE} \
                --input_dim ${n_features} \
                --output_dim 1 \
                --num_epochs ${NUM_EPOCHS} \
                --learning_rate ${LEARNING_RATE} \
                --lasso_penalty ${lasso_penalty} \
                --group_penalty ${group_penalty} \
                --lasso_norm ${LASSO_NORM} \
                --group_norm ${GROUP_NORM} \
                --noise_injection \
                --fit_intercept \
                --expr_name "nimo_pmlb_${dataset}_lasso${lasso_penalty}_group${group_penalty}"

            echo ""
            echo "Finished training on ${dataset} with lasso=${lasso_penalty}, group=${group_penalty}"
            echo ""
        done
    done
done

echo "=========================================="
echo "All training jobs completed!"
echo "=========================================="
