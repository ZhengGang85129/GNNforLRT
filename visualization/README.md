# Visualization Details


This folder provides tools for visualizing dataset distributions and model performance at different stages of the pipeline.

|File|Description|
|:---|:----------|
|plt_performance.py|Visualize GNN evaluation metrics such as AUC and ROC curve|
|stage_performance.py| Visualize efficiency and purity for each stage.|
|visualize_gnn_efficiency_purity_detector.py| Visualizes GNN efficiency and purity over the detector's r-z profile.|
|visualize_reconstructed_tracks.py| Visualizes reconstructed tracks and ground truth matching, annotated with edge scores.|
|visualize_seed_quality.py|Visualizes seed quality metrics for initial track candidates. |
|plot_configuration.py| Defines plotting configuration settings for visualizations.|
|evaluate_tracking_observables.py | Plots tracking efficiency as a function of collider observables, such as $p_T$, $\eta$, $d_0$, $z_0$, and pileup.|
|visualize_edge_filtering_ablation.py| Visualizes true and fake edge connections before and after the edge filtering stage.|
|analyze_raw_hit_distribution.py|This script analyzes and visualizes edge-related statistics from graph-based tracking outputs over different spatial regions of the detector.|


⚠️ Notes:
* These scripts are intended for local use only
* These are **not part of core production pipeline**.
* User should prepare their own configuration file (A template can be found in the `configs` directory).

## Usage
### Stage-wise performance
<img width="615" alt="image" src="https://github.com/user-attachments/assets/1344082f-bc95-4a77-9e9d-b5abd9c9f6e7" />

Visualize efficiency and purity for each training stage:

```bash
python3 ./stage_performance.py ./ZZZ.yaml  # ./ZZZ.yaml refer to the yaml file generated during training procedure, which contains the efficiency and purity for each stage.
```
* XXX.yaml: YAML file generated during training, containing recorded efficiency and purity metrics for each stage.
### Performance (AUC/Edge score distribution)
<img width="477" alt="image" src="https://github.com/user-attachments/assets/80b5e188-f6cc-4d75-a374-6afdd49b315c" />

Visualize model evaluation metrics such as AUC and edge score distribution:
```bash
python3 ./plt_performance.py GNN_output/XXXX # GNN_output/XXXX refer to the torch file generated by GNN stage.
```
* GNN_output/XXXX: Torch file generated during the GNN stage containing model outputs.

### Efficiency & Precision 

<img width="579" alt="image" src="https://github.com/user-attachments/assets/ba49c297-d4b5-413a-be53-778eb1cbbd31" />

```bash
python3 evaluate_tracking_observables.py --config configs/XXX.yaml --output YYY --mode [extract:evaluate] --lepton [prompt:displaced] --fname ZZZ 
```
Explanation for each argument:
- —config: Path to the configuration file specifying the source objects (particles, hits, edges) and their relationships.(Template available in configs/Template.yaml)`
- —output(-d): Output folder for saving the extracted metrics (default: metrics/final). (Usually, you don't need to manually set this.)
- —mode(-m):
  *  `extract`: Extract information from generated, reconstructed, and matched particles.
  *  `evaluate`: Generate evaluation plots based on the extracted data.
- —fname: Specify a filename for both extraction and evaluation outputs. (Make sure the same name is used for both steps.)


### Efficiency and Purity vs detector 
<img width="262" alt="image" src="https://github.com/user-attachments/assets/bff28497-4630-43e4-b2a5-9719862a8aad" />

Visualize the spatial distribution of raw detector hits:

```bash
python3 visualize_gnn_efficiency_purity_detector.py configs/XXX.yaml 
```
* XXX.yaml: Configuration file describing the dataset layout.

### Visualize edge connection before and after filtering stage
<img width="505" alt="image" src="https://github.com/user-attachments/assets/b93cb0f2-4793-4862-a2a5-d6395d0b58ad" />

```bash
python3 visualize_edge_filtering_ablation.py  # You have to adjust the configs(line:14) to your own configs file
```
### Visualize Seed quality

<img width="276" alt="image" src="https://github.com/user-attachments/assets/55fba4c3-eae1-42b0-8bf3-6534855ad2e6" />


```bash
python3 ./visualize_seed_quality configs/XXX.yaml
```

### Visualize constructed tracks
<img width="398" alt="image" src="https://github.com/user-attachments/assets/be8ba8ec-3c4e-41f1-9aad-6ed8c5c55478" />

```bash
python3 visualize_reconstructed_tracks.py <algorithm: DBSCAB orWrangler> <configs file> <lepton type: displaced/prompt/all/HSS>
```
