# Hyperparameter optimization Details

This folder contains scripts and configuration files for manually tuning hyperparameters for modules at each stage of the pipeline.

⚠️ Notes:
- These scripts are intended for **local experimentation only**
- These are **not** part of core production pipeline or the official training workflow.
- Hyperparameter searches are conducted manually and may not follow systematic optimization.

## Structure
|File| Description| 
|:---|:-----------|
|scheduler_train.py| A main scheduler script for preparing and dispatching configuration files for different stages.|
|train-Embedding_template.yaml| Template configuration file for the embedding stage.|
|train-Filter_template.yaml| Template configuration file for the filter stage.|
|train-GNN_template.yaml| Template configuration file for the gnn stage.|
|config_generate_hyperparam.py| Helper script for preparing customized configuration files for a specific stage.|
|config_prepare_pipeline.py| Utility script for organizing the overall training pipeline configurations.|
|search_random_hyperparam.py| Tool for randomly sampling of hyperparameter values during optimization. This will produces the configuration files|
|runner_submit_job.py| Helper for preparing the slurm scripts.|
|report_aggregate_results.py| A tool to aggregate and summarize hyperparameter optimization results into tables.|

## Usage

```bash
python3 ./hyperopt/search_random_hyperparam.py
sh ./slurm_script/hyperopt_grid_search.sh 
```
