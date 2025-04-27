# GNNforLRT
## Overview
The High-Luminosity Large Hadron Collider(HL-LHC) and Long-Lived Particles(LLP) pose a challenge for particle trajectory reconstruction in high-energy physics(HEP) experiments, with high-noise conditions and displaced vertices. This project implements a Graph Neural Network(GNN)- based pipeline, Exatrkx, to reconstruct particle trajectories. This method achieves a 90% reconstruction efficiency for muons from TTBar dileptonic decays and a 75% reconstruction efficiency from Heavy Neutral Lepton decays.  
This project demonstrates how GNNs can effectively capture complex spatial dependencies and improve upon traditional track reconstruction methods.


## Motivation
In HEP experiments, accurately reconstructing particle tracks from few ten-thousands of noisy hits is a fundamental yet challenging task. Traditional track reconstruction algorithms often struggle with noise and combinatorial complexity. Leveraging GNN provides: robustness to noise, scalability to large datasets, and superior accuracy compared to classical algorithms.

## Key features
* Graph Construction: Represent raw detector hits as nodes and connections as edges.
* Edge Classification: Predict the likelihood of true physical connections between nodes.
* Track Building: Use the filtered edges to cluster hits into individual particle tracks.
* High Efficiency: Achieved 90% reconstruction efficiency in synthetic datasets simulating high-noise environments.

## Methods
* Graph Neural Network: Implemented using PyTorch Lightning, with customized message-passing layers to fit the physics constraints.
* Edge Filtering: Applied a simple classifier to reduce fake edges before final track building.
* Clustering: Implemented a DFS-based algorithm, Wrangler, based on the filtered graph. We used DBSCAN as our baseline.

## Tools and Libraries
* Python
* PyTorch Lightning
* scikit-learn
* NetworkX
* Matplotlib
* Numpy
* Pandas

## Results


## Folder Structure

## Future Work
* Apply on real collision data
* Explore advanced GNN architecture, like DGCNN, transformer-based model.
* Implement end-to-end track reconstruction pipelines

## Contact
For any questions or collaboration inquiries, please contact:
Zheng-Gang Chen

## Paper (In Progress)
This project is part of an ongoing research work. A related paper is currently under preparation and will be made available soon.
Tentative title:
_Large Radius Tracking Reconstruction_

1. Please see [this](https://quiet-magnesium-057.notion.site/Pipeline-to-install-TrackML-and-training-the-model-for-GNNforLRT-project-c5de6509e5ef409bb968a9d5f3969306?pvs=4) for training instruction.
2. To monitor the loss curve and performance during training, you can see [this](https://quiet-magnesium-057.notion.site/Pipeline-to-plot-the-training-validation-curve-for-each-stage-650886b4a9bf46fbb30ce536a966c347).
3. To visualize the dataset, you can see [this](https://quiet-magnesium-057.notion.site/Event-Display-29d2bb171d3c47eba6d3ab1a68cfe06a?pvs=4) (efficiency & purity vs detector)
4. To display the performance in terms of efficiency & purity, please follow [this](https://quiet-magnesium-057.notion.site/Stage-performance-11f67c6786f6808d9fc9fc78d93a8573?pvs=4)
5. To plot efficiency over npileup etc, see [this](https://quiet-magnesium-057.notion.site/Evaluate-the-model-11f67c6786f68022abebd33843e4608b?pvs=4)
6. Model inference [this](https://quiet-magnesium-057.notion.site/Inference-11f67c6786f6803f93c1d256dc30bee1?pvs=4)
   
