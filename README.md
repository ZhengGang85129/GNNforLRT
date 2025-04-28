# Particle Trajectory Reconstruction with Graph Neural Networks
## Overview
The High-Luminosity Large Hadron Collider(HL-LHC) and Long-Lived Particles(LLP) pose a significant challenge for particle trajectory reconstruction in high-energy physics(HEP) experiments, due to high-noise conditions and displaced vertices. This project implements a Graph Neural Network(GNN)-based pipeline, Exa.TrkX, to reconstruct particle trajectories. This method achieves a 90% reconstruction efficiency for muons from TTBar dileptonic decays and a 75% reconstruction efficiency from Heavy Neutral Lepton decays.  
This project demonstrates how GNNs can effectively capture complex spatial dependencies and improve upon traditional track reconstruction methods.

## Motivation
In HEP experiments, accurately reconstructing particle tracks from a few tens of thousands of noisy hits is a fundamental yet challenging task. Traditional track reconstruction algorithms often struggle with noise and combinatorial complexity. Leveraging GNN provides robustness to noise, scalability to large datasets, and superior accuracy compared to classical algorithms.

## Methods
* Graph Neural Network: Implemented using PyTorch Lightning, with customized message-passing layers to fit the physics constraints.
* Edge Filtering: Applied a simple classifier to reduce fake edges before final track building.
* Clustering: Implemented a DFS-based algorithm, Wrangler, based on the filtered graph. 

## Installation


```bash
mkdir WORKSPACE
cd WORKSPACE #Your WORKSPACE
WORKSPACE=$(pwd)
git clone git@github.com:ZhengGang85129/GNNforLRT.git
git clone git@github.com:HSF-reco-and-software-triggers/Tracking-ML-Exa.TrkX.git ;
git clone git@gitlab.com:gnnparticletracking/largeradiustracking/analysis.git;
```

If you don't have conda environment, you can build it from scratch
```bash
#choose one
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh # for Linux
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh # for Mac
#choose one
sh Miniconda3-latest-Linux-x86_64.sh # for Linux
sh Miniconda3-latest-MacOSX-arm64.sh # for Mac
```

Now, switch to `Tracking-ML-Exa.TrkX` folder, and install the necessary libraries:

```bash
source ${HOME}/miniconda3/etc/profile.d/conda.sh;
conda create --name exatrkx-gpu python=3.9;
conda activate exatrkx-gpu;
pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
cd ${WORKSPACE}/Tracking-ML-Exa.Trkx;
pip3 install -e .;
pip3 install traintrack;
pip3 install wandb;
pip3 install tensorboard;
pip3 install faiss-gpu
pip3 install pytorch_lightning==1.9.2
pip install "ruamel.yaml<0.18.0"
```

## Usage

1. [Training Instruction](https://quiet-magnesium-057.notion.site/Pipeline-to-install-TrackML-and-training-the-model-for-GNNforLRT-project-c5de6509e5ef409bb968a9d5f3969306?pvs=4) - How to prepare configuration file for training pipeline and launch training.
```bash
conda activate exatrkx-gpu
traintrack ./configs/pipeline.yaml
```
2. [Model Inference](https://quiet-magnesium-057.notion.site/Inference-11f67c6786f6803f93c1d256dc30bee1?pvs=4) - How to do inference
```bash
conda activate exatrkx-gpu
traintrack ./configs/pipeline.yaml
```



## Tools and Libraries
* Python >= 3.9
* PyTorch Lightning == 1.9.2
* scikit-learn
* NetworkX
* Matplotlib
* Numpy
* Pandas
* CUDA Toolkit 12.1 (for GPU acceleration)

## Results

| Dataset       | Reconstruction Efficiency (PU = 200) |
|:--------------|:--------------------------|
| TTBar dileptonic muon | 90.3 %            |
| Heavy Neutral Lepton decays | 75.1 %      |


## Acknowledgement
This project builds upon the [Tracking-ML-Exa.TrkX](https://github.com/HSF-reco-and-software-triggers/Tracking-ML-Exa.TrkX) framework developed by the HEP software Foundation(HSF). We thank the original authors for providing an excellent foundation for particle tracking research.

If you use this codebase, please cite both this work and the following:
> X. Ju et al., "Performance of a Geometric Deep Learning Pipeline for HL-LHC Particle Tracking, " Eur. Phys. J. C, 2021. [arXiv:2103.06995](https://arxiv.org/pdf/2103.06995)

## Future Work
* Apply on real collision data
* Explore advanced GNN architecture, like DGCNN, transformer-based model.
* Implement end-to-end track reconstruction pipelines

## Contact
For any questions, feedback, or collaboration inquiries, please feel free to contact:
- **Zheng-Gang Chen**
   - [zheng-gang.chen@cern.ch](mailto:zheng-gang.chen@cern.ch) 
   - [GitHub](https//github.com/ZhengGang85129)
- **Yuan-Tang Chou**
   - [yuan-tang.chou@cern.ch](mailto:yuan-tang.chou@cern.ch)
- **Wei Fang**
   - [daniel.bb0321@gmail.com](mailto:daniel.bb0321@gmail.com) 
- **You-Ying Li**
   - [you-ying.li@cern.ch](mailto:you-ying.li@cern.ch)


## Paper (In Progress)
This project is part of an ongoing research work. A related paper is currently under preparation and will be made available soon.
Tentative title:
_Large Radius Tracking Reconstruction_


   
