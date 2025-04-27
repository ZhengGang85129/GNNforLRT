# Configs Details

This folder contains configuration files for setting up the training pipeline, managing GPU/CPU resource allocation, and defining checkpoint storage paths.  
Users should adjust these files according to their specific environment and training requirements.



|Files| Description|
|:----|:-----------|
|batch_gpu_default.yaml| Specifies GPU resource requirements and runtime settings for submitting jobs to a Slurm workload system|
|batch_cpu_default.yaml| Specifies CPU resource requirements and runtime settings for submitting jobs to a Slurm workload system|
|pipeline.yaml| Configuration file defining the end-to-end training pipeline, including stage specifications and workflow parameters.|
|project_config.yaml| Defines the environment setup procedures and resource locations required before running the training or evaluation pipeline.|





