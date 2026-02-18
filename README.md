# NIMO: Nonlinear Interpretable MOdel

## Pre-requirement
Create the environment by running:
```shell
conda env create -f environment.yml
```

## Create datasets
To create the synthetic datasets we used in the paper, please follow the procedure in notebooks `synthetic_reg.ipynb` and `synthetic_cls.ipynb`.

## Reproduce the experiment results
**Step 1: Split the dataset**. Run the following command to split the dataset with different random seeds (from 1 to 9) and save the data:
```shell
python split_diabetes.py
```
The same data split procedure for **Boston housing dataset**. Running the following command to split and save the data:
```shell
python split_boston.py
```

**Step 2: Train the models**. 

1. Train naive neural network by running:
```shell
./train_simple_nn.sh
```
Please modify it according to the task and datasets.


2. Train NIMO by running:
```shell
./train_ar.sh
```
Please modify it according to the task and datasets.

**Steo 3: Reproduce the plots**

For reproducing the plots in our paper, please go through the jupyter notebooks.