# Lightning Modules

This folder contains the core PyTorch Lightning modules used in the training pipeline. Each module is responsible for specific stages in the particle track reconstruction process.

The modules are designed to be loaded and managed automatically through the main training script and configuration files.


## Structure

The modular design based on the following structure:

| Folder | Description |
|:-------|:------------|
| `Embedding` | Modules for initial graph construction from raw hit data (coordinate embedding, graph building).|
| `Filter`    | Lightweight classifiers for edge pruning to reduce fake edges and save run time for downstream track clustering. |
| `GNN`       | Graph Neural Networks for predicting refined edge scores and enhancing track reconstruction efficiency. |



## Usage 

**You are not expected to run individual modules directly.**
Instead, training and evaluation are launched by:
```bash
traintrack ./configs/**.yaml 
traintrack ./configs/**.yaml --inference
```
