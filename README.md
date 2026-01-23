# Data augmented nonparamteric noise contrastive estimation

This repository contains the code used for the paper

> Chenghao Li and Yuanyuan Lin. A data-augmented contrastive learning approach to nonparametric density estimation. Journal of Machine Learning Research. 2026, accepted.

## Requirements

The main requirements:
- Python>=3.11.3
- Pytorch>=2.0.1

The details can be found in `environment.yml` and `requirements.txt`.
You can create an virtual environment using:
```shell
$ conda env create -f environment.yml
```

Code has been tested on [CUHK High-Performance Computing Cluster](https://www.cuhk.edu.hk/itsc/hpc/overview.html) with NVIDIA Tesla V100(32GB)/Tesla P100(16GB) GPUs.

## Reproduction
### 2-D experiments

The main python script `main_est.py` is used for implementing data augmented nonparamteric noise contrastive estimation in 2-D cases.
The neural network architeture, distribution models, and basic functions are defined in `setup.py`.

The following three models of data ditributions has been implemented in the code:
1) Indepedent Gaussian mixture mdoel; 
2) Eight-octagon Gaussian mixture model;
3) Involute.

Before running `main_est.py`, you may specfiy 
- `[rho]`: The parameter of data augmentation, $\rho$;
- `[model_type]`: the distribution model type, one of 'indep_GMM', 'octagon_GMM', or 'involute'. 
The configuration files of all parameters related to network training are in the `configs` folder.

### Higher dimensional experiments

The script `main_est_multi_dim.py` is used for implementing data augmented nonparamteric noise contrastive estimation in higher dimensional cases.
Before running `main_est_multi_dim.py`, you may specfiy
- `[d]`: the dimension;
- `[no_augmentation]`: whether to use data augmentaion, that is, whether to set the value of $\rho$ to be 0.

### Experiments with varying $\rho$

The scripts `main_est_varrho.py` and `main_est_varrho_md.py` are used for comparing the performance of data augmented nonparamteric noise contrastive estimation under different $\rho$, in 2-D cases and higher dimensional cases, respectively.
Before running the scripts, you may specfiy 
- `[d]`: the dimension, exclusive for `main_est_varrho_md.py`;
- `[rho_list]`: the sequence of rho used for comparison.

### Real data analysis on Shuttle and Mammography datasets

The scripts `main_est_anomaly_detection.py` is used for applying data augmented nonparamteric noise contrastive estimation to two anomaly/outlier detection datasets: Shuttle and Mammography. 
Both datasets are availiable from the [ODDS Database](https://shebuti.com/outlier-detection-datasets-odds/).
Before running the scripts, you may specfiy 
- `[dataset_name]`: the name of datset, one of 'Shuttle' and 'Mammography'.

### Visualization

The scripts `visualization_2d.py` and `visualization_line_chart` are used for visualizing true and estimated densities in 2-D cases, and generating the line chart for comparing the performance under different $\rho$, respectively.
Before running the scripts, please make sure you have obtained corresponding data in `[data_folder]`.
