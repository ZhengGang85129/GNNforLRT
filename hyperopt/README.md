# Hyperparameter optimization Details

This folder contains scripts and configuration files for manually tuning hyperparameters for modules at each stage of the pipeline.

⚠️ Notes:
- These scripts are intended for **local experimentation only**
- These are **not** part of core production pipeline or the official training workflow.
- Hyperparameter searches are conducted manually and may not follow systematic optimization.

## Structure
|File| Description| 
|:---|:-----------|
|train.py||
|train-Embedding_template.yaml||
|train-Filter_template.yaml||
|train-GNN_template.yaml||
|config_hyperparam.py||
|prepare_train_config.py||
|random_search.py||
|run_model.py||
|to_table.py||
