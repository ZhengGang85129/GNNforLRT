# Embedding Details

| Module | Description |
|:-------|:------------|
|`embedding_base.py`| `EmbeddingBase` is a PyTorch Lightning module that implements a flexible and modular framework for training embedding models for graph-based track reconstruction tasks. | 
|`utils.py`| It provides utilities for using FRNN to construct the initial graph, dataset splitting, batch construction and input pre-processing.|
|`Models/layerless_embedding.py`| `LayerlessEmbedding` is a module built upon EmbeddingBase.|
|`Models/inference.py`|Contains callback classes: `EmbeddingPurEff`, `EmbeddingTelemetry`, `EmbeddingBuilder`, which are used to calculate purity and efficiency, stadandard tests of performance of model, and saving initial constructed graph.|

## EmbeddingBase Summary
It supports multiple training regimes, including:
- **Random Pairing (rp)**: Randomly pairly hits to create negative samples
- **Hard Negative Mining (hnm)**: Supports hinge embedding loss by default, with extendibility towards cross-entropy or triplet-based losses.
- **Data Handling**: Includes utilities for dataset splitting, batch construction, and input pre-processing.
- **Evaluation Metrics**: Provides standard track reconstructio metrics such as efficeincy, purity.

The key components include:
- **Graph Construction**: Dynamically builds candidate edges from query points using KNN search or radius-based selection.
- **Training Loss**: Supports hinge embedding loss by default, with extendibility towards cross-entropy or triplet-based losses.
- **Data Handling**: Includes utilities for dataset splitting, batch construction, and input pre-processing.
- **Evaluation Metrics**: Provides standard track reconstruction metrics such as efficiency and purity.

### Design Philosophy
The `EmbeddingBase` class separates data handling, model forward pass, and loss computation into modular functions, enabling:
- Flexible experimentation with different graph construction and loss strategies.
- Easy adaptation for different physics datasets or detector conditions.
- Integration into a larger modular pipeline based on the xo4n4 architecture.

### Typical Workflow
- **Train**: Select query points, dynamically build candidate graphs, perform forward pass, calculate contrastive loss (hinge or other).
- **Validate/Test**: Construct global graphs and evaluate performance using standard reconstruction metrics.
- **Optimization**: Includes support for learning rate scheduling and warm-up strategies.

---

> Note: 
> This class is designed for internal usage as a base module. Specific embedding model architectures should inherit from `EmbeddingBase` and implement their own forward pass.
