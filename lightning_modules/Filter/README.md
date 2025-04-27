# Filter Details

|Module|Description|
|:-----|:----------|
|`filter_base.py`|`FilterBase` is a PyTorch Lightning module that provides a flexible framework for training lightweight edge classifiers in the graph-based track reconstruction pipeline.|

## FilterBase Summary

It supports different training regimes, including:
- **Standard Binary Classification**: Classify each edge as true or false.
- **Subset-based Training**: Optionally sample a balanced subset of edges to handle class imbalance.
- **Weighting Regimes**: Apply manual weighting on classes or edges based on the training setting.


Key functionalities include:
- **Flexible Sampling**: Supports random sampling, hard negative mining, and class balancing strategies.
- **Loss Function**: Default binary cross-entropy loss with support for positive class weighting.
- **Data Handling**: Includes dynamic batch splitting, edge sampling, and optional use of auxiliary features (e.g., cell information).
- **Evaluation Metrics**: Calculates standard track reconstruction metrics such as efficiency (recall) and purity (precision) during validation and testing.
- **Learning Rate Warm-up**: Optional learning rate warm-up scheduling during early training.

Training and evaluation are based on dynamically selecting and filtering edges based on model predictions and pre-defined thresholds.

## Design Philosophy

The `FilterBase` class separates data loading, forward pass, loss computation, and metric evaluation, enabling:
- Easy experimentation with different sampling or weighting strategies.
- Modular extension for more complex filtering regimes (e.g., filter with PID information).

The `FilterBaseBalanced` class extends `FilterBase` by:
- Introducing a more aggressive hard negative mining strategy.
- Explicitly randomizing and balancing true and false edges before training steps.

## Typical Workflow

- **Training**:
  - Select a subset of edges based on the regime.
  - Apply forward pass and calculate binary cross-entropy loss.
  - Optionally apply dynamic positive weight scaling.
- **Validation**:
  - Evaluate model performance on full batches, calculating efficiency and purity.
- **Testing**:
  - Same as validation, without logging to TensorBoard.