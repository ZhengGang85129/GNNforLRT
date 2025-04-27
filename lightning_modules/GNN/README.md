# GNN Details

## GNNBase Summary


|Module|Description|
|:-----|:----------|
|`gnn_base.py`| `GNNBase` is a PyTorch Lightning module that implements the training and evaluation flow for Graph Neural Network (GNN) models, designed for particle track reconstruction tasks.

## GNNBase Summary

It supports multiple training regimes and loss computation strategies, providing a modular, flexible backbone for GNN-based learning on graph-structured particle detector data.

## Key Features

- **Flexible Input Handling**:
  - Supports raw spatial features or combined features (e.g., spatial + cell information).
  - Handles both truth labels (`batch.y`) and particle ID matching (`pid`) as supervision targets.

- **Training Regimes**:
  - Supports balanced or weighted loss computation.
  - Optional use of edge attributes (e.g., manual weights).

- **Loss Functions**:
  - Binary cross-entropy loss with optional class/edge reweighting.
  - Dynamic positive/negative sample balancing during training.

- **Evaluation Metrics**:
  - Computes efficiency (true positive recall) and purity (precision) based on edge classification.
  - Monitors validation loss, efficiency, purity, and learning rate during training.

- **Scheduler and Optimization**:
  - AdamW optimizer with StepLR or ReduceLROnPlateau scheduler.
  - Supports learning rate warm-up during early training stages.

- **Dataset Management**:
  - Flexible setup for loading train/validation/test splits.
  - Uses `torch_geometric`'s `DataLoader` for efficient mini-batching.

## Typical Workflow

- **Training**:
  - Forward pass through the GNN to predict edge scores.
  - Apply binary cross-entropy loss based on predicted scores vs. ground truth.
  - Optional weighting based on sample difficulty or class imbalance.

- **Validation / Testing**:
  - Predict edges and apply a configurable score threshold (`edge_cut`).
  - Evaluate reconstruction performance in terms of efficiency and purity.

- **Logging**:
  - Training and validation losses, efficiency, and purity are logged per epoch.
  - TensorBoard summaries are automatically created for visualization.

## Design Philosophy

The `GNNBase` class abstracts:
- Data loading and preprocessing.
- GNN model forward pass (to be implemented in subclasses).
- Training logic, loss calculation, and metric evaluation.

This promotes modular reuse and easy extension for different GNN architectures (e.g., GCN, EdgeConv, GAT, GravNet).

Users are expected to subclass `GNNBase` and implement their specific GNN layers and forward behavior.
