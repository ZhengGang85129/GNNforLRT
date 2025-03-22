import yaml
import os
import argparse


def make_training_config():
    # make embedding config
    embedding_output_dir = f"/global/cfs/cdirs/m3443/data/GNNforLRT/results_final/Inference-Sample{sample_label}-Model{model_label}-Embed"
    log_dir = f"logging/{tag}"
    performance_path = f"{tag}.yaml"

    with open(f"/global/cfs/cdirs/m3443/data/GNNforLRT/bestconfig/Embed_best_PU200.yaml", 'r') as f:
        config = yaml.safe_load(f)

    config['input_dir'] = sample_dir
    config['output_dir'] = embedding_output_dir
    config['checkpoint_path'] = f"{checkpoint_dir}/Embed.ckpt"
    config['TAG'] = tag
    config['log_dir'] = log_dir
    config['performance_path'] = performance_path
    config['train_split'] = [[0, 0, test_split]]

    new_config_path = f"LightningModules/Embedding/inference_{tag}.yaml"
    with open(new_config_path, 'w') as f:
        yaml.dump(config, f)
    
    print(f"Embedding config is made at {new_config_path}")

    # make filter config
    filter_output_dir = f"/global/cfs/cdirs/m3443/data/GNNforLRT/results_final/Inference-Sample{sample_label}-Model{model_label}-Filter"

    with open(f"/global/cfs/cdirs/m3443/data/GNNforLRT/bestconfig/Filter_best_PU200.yaml", 'r') as f:
        config = yaml.safe_load(f)

    config['input_dir'] = embedding_output_dir
    config['output_dir'] = filter_output_dir
    config['checkpoint_path'] = f"{checkpoint_dir}/Filter.ckpt"
    config['TAG'] = tag
    config['log_dir'] = log_dir
    config['performance_path'] = performance_path
    config['train_split'] = [[0, 0, test_split]]
    new_config_path = f"LightningModules/Filter/inference_{tag}.yaml"
    with open(new_config_path, 'w') as f:
        yaml.dump(config, f)

    print(f"Filter config is made at {new_config_path}")

    # make gnn config
    gnn_output_dir = f"/global/cfs/cdirs/m3443/data/GNNforLRT/results_final/Inference-Sample{sample_label}-Model{model_label}-GNN"

    with open(f"/global/cfs/cdirs/m3443/data/GNNforLRT/bestconfig/GNN_best_PU200.yaml", 'r') as f:
        config = yaml.safe_load(f)

    config['input_dir'] = filter_output_dir
    config['output_dir'] = gnn_output_dir
    config['checkpoint_path'] = f"{checkpoint_dir}/GNN.ckpt"
    config['TAG'] = tag
    config['log_dir'] = log_dir
    config['performance_path'] = performance_path
    config['train_split'] = [[0, 0, test_split]]

    new_config_path = f"LightningModules/GNN/inference_{tag}.yaml"
    with open(new_config_path, 'w') as f:
        yaml.dump(config, f)
    
    print(f"GNN config is made at {new_config_path}")

def make_pipeline_config():
    reference_config = "configs/pipeline_inference_HNL_PU200-flatpu.yaml"
    with open(reference_config, 'r') as f:
        config = yaml.safe_load(f)

    config['stage_list'][0]['config'] = f"inference_{tag}.yaml"
    config['stage_list'][1]['config'] = f"inference_{tag}.yaml"
    config['stage_list'][2]['config'] = f"inference_{tag}.yaml"

    new_config_path = f"configs/pipeline_inference_{tag}.yaml"
    with open(new_config_path, 'w') as f:
        yaml.dump(config, f)

    print(f"Pipeline config is made at {new_config_path}")

def make_evaluation_config():
    reference_config = "evaluation/tracks/DBSCAN_config/HNL_PU200-flatpu.yaml"
    with open(reference_config, 'r') as f:
        config = yaml.safe_load(f)

    config['event']['files']['gnn_processed']['file'] = "/global/cfs/cdirs/m3443/data/GNNforLRT/results_final/Inference-Sample" + f"{sample_label}-Model{model_label}-GNN/test/" + "{evtid:04}"
    config['event']['files']['particles']['file'] = sample_particle_path
    config['event']['files']['hits']['file'] = "/global/cfs/cdirs/m3443/data/GNNforLRT/results_final/Inference-Sample" + f"{sample_label}-Model{model_label}-GNN/test/" + "{evtid:04}"
    config['event']['files']['edges']['file'] = "/global/cfs/cdirs/m3443/data/GNNforLRT/results_final/Inference-Sample" + f"{sample_label}-Model{model_label}-GNN/test/" + "{evtid:04}"

    new_config_path = f"evaluation/tracks/DBSCAN_config/{tag}.yaml"
    with open(new_config_path, 'w') as f:
        yaml.dump(config, f)

    print(f"Evaluation config is made at {new_config_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--sample_label", type=str, default="HSSPU200")
    parser.add_argument("-d", "--sample_hits_dir", type=str, default="/global/cfs/cdirs/m3443/data/GNNforLRT/HSS_pgy_PU200")
    parser.add_argument("-p", "--sample_particle_dir", type=str, default="/global/cfs/cdirs/m3443/data/GNNforLRT/Hss_Pt1GeV_PU200_RAW/HSS_output_PU200")
    parser.add_argument("-m", "--model", type=str, default="MIXED", choices=["MIXED", "HNLPU200", "TTBarPU200"])
    parser.add_argument("-t", "--test_split", type=int, default=1500)
    parser.add_argument("-c", "--checkpoint_dir", type=str, default="")
    args = parser.parse_args()

    model_attribute = {
        "MIXED": {
            "label": "MIXED",
            "checkpoint_dir": "/global/cfs/cdirs/m3443/data/GNNforLRT/bestckpt/forAll",
        },
        "HNLPU200": {
            "checkpoint_dir": "/global/cfs/cdirs/m3443/data/GNNforLRT/bestckpt/forHNL",
        },  
        "TTBarPU200": {
            "checkpoint_dir": "/global/cfs/cdirs/m3443/data/GNNforLRT/bestckpt/forTTBar",
        },
    }

    global sample_label
    global sample_dir
    global sample_particle_path
    global model_label
    global checkpoint_dir
    global test_split
    global tag

    sample_label = args.sample_label
    sample_dir = args.sample_hits_dir
    sample_particle_path = args.sample_particle_dir + "/event{evtid:09}-particles.csv"
    model_label = args.model
    checkpoint_dir = args.checkpoint_dir if args.checkpoint_dir else model_attribute[args.model]["checkpoint_dir"]
    test_split = args.test_split
    tag = f"{model_label}-{sample_label}"

    make_training_config()
    make_pipeline_config()
    make_evaluation_config()