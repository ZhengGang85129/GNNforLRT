# Hyperparameter optimization Details

This folder contains scripts and configuration files for manually tuning hyperparameters for modules at each stage of the pipeline.

⚠️ Notes:
- These scripts are intended for **local experimentation only**
- These are **not** part of core production pipeline or the official training workflow.
- Hyperparameter searches are conducted manually and may not follow systematic optimization.

## Structure
|File| Description| 
|:---|:-----------|
|train.py| A main scheduler script for preparing configuration files for assigned stages.|
|train-Embedding_template.yaml| A template configuration file for embedding stage.|
|train-Filter_template.yaml| A template configuration file for filter stage.|
|train-GNN_template.yaml| A template configuration file for gnn stage.|
|config_hyperparam.py| A helper for preparing specific stage configuration file.|
|prepare_train_config.py| A helper for organizing the training pipeline.|
|random_search.py| A helper for randomly sampling the hyperparameter values|
|run_model.py| |
|to_table.py| A tool to aggregate the results of optimization searching.|
