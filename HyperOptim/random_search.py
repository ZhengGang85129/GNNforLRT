import yaml
from typing import List, Dict
import random
from collections import OrderedDict

def represent_dictionary_order(self, dict_data):
    return self.represent_mapping('tag:yaml.org,2002:map', dict_data.items())
def setup_yaml():
    yaml.add_representer(OrderedDict, represent_dictionary_order)

def produce_filter_hyperparam(trials: int =10, InputDir: str = None)->None:
    with open('./LightningModules/Filter/train-noPU_default.yaml', 'r') as file:
        cfg = yaml.safe_load(file) 
        cfg['checkpoint_path'] = '/global/cfs/cdirs/m3443/data/GNNforLRT/lightning_checkpoints'
    cfg_to_be_updated = OrderedDict(cfg)
    cfg_to_be_updated['datatype_split'] = [[1000, 500, 1000]]
    cfg_to_be_updated['input_dir'] = InputDir
    cfg_to_be_updated['output_dir'] = '/global/cfs/cdirs/m3443/data/GNNforLRT/results/TTbar_noPU_Filtering_output'
    trials = 100
    for trial in range(trials):
        cfg_to_be_updated.update(hyperparam_randon_choice_filter())
        with open(f'./HyperOptim/Filter/filter-grid_search_trial-set1-{trial:04d}.yaml', 'w') as file:
            for k, v in cfg_to_be_updated.items():
                yaml.safe_dump({k: v}, file, default_flow_style=False)
            yaml.safe_dump({'TAG': f'filter-grid_search_trial-set1-{trial:04d}'}, file, default_flow_style=False)

def produce_embed_hyperparam(trials: int = 10)->None:
    with open('./LightningModules/Embedding/train-noPU_default.yaml', 'r') as file:
        cfg = yaml.safe_load(file) 
        cfg['checkpoint_path'] = '/global/cfs/cdirs/m3443/data/GNNforLRT/lightning_checkpoints'
    cfg_to_be_updated = OrderedDict(cfg)
    cfg_to_be_updated['train_split'] = [[1000, 500, 1000]]
    cfg_to_be_updated['emb_hidden'] = [[256, 256, 256, 256]]
    for trial in range(trials):
        cfg_to_be_updated.update(hyperparam_randon_choice_embed())
        with open(f'./HyperOptim/Embedding/embed-grid_search_trial-{trial:04d}.yaml', 'w') as file:
            for k, v in cfg_to_be_updated.items():
                yaml.safe_dump({k: v}, file, default_flow_style=False)
            yaml.safe_dump({'TAG': f'embed-grid_search_trial-{trial:04d}'}, file, default_flow_style=False)
def main():
    setup_yaml()
    seed = 12345
    random.seed(seed)  
    
    #produce_filter_hyperparam(trials = 100, InputDir = '/global/cfs/cdirs/m3443/data/GNNforLRT/results/TTbar_noPU_Embedding_output_grid_search_trial-0001') 
    #produce_filter_hyperparam(trials = 100, ) 
    produce_gnn_hyperparam(trials = 100, InputDir = '/global/cfs/cdirs/m3443/data/GNNforLRT/results/TTbar_noPU_Filtering_best1/') 

def hyperparam_randon_choice_filter() -> Dict:
    choices = {
        'filter_cut': [0.001 * i for i in range(300)],
        'hidden': [32, 64, 128, 256, 512, 1024],
        'nb_layer': [2, 3, 4, 5, 6],
        'weight': [0.01 * i for i in range(300)],
        
    }
    
    result = dict()
    
    for hyperparam_name in choices.keys():
        result[hyperparam_name] = random.choice(choices.get(hyperparam_name))
        
     
    
    return result 

 
def hyperparam_randon_choice_embed() -> Dict: 

    choices = {
        'emb_dim': [4, 8, 16, 32, 64],
        'weight': [1.5 + i*0.1  for i in range(10)],
        'r': [0.1 + i * 0.01 for i in range(40)],
        'knn': [4, 8, 16, 32, 64],
        'margin': [0.1 + i * 0.01 for i in range(40)]
    }
    
    result = dict()
    
    for hyperparam_name in choices.keys():
        if hyperparam_name == 'r':
            result['r_test'] = random.choice(choices.get(hyperparam_name))
            result['r_train'] = result['r_val'] = result['r_test']
            continue
             
        result[hyperparam_name] = random.choice(choices.get(hyperparam_name))
        
     
    
    return result 
        
def produce_gnn_hyperparam(trials: int =10, InputDir: str = None)->None:
    with open('./LightningModules/GNN/train-noPU_default.yaml', 'r') as file:
        cfg = yaml.safe_load(file) 
        cfg['checkpoint_path'] = '/global/cfs/cdirs/m3443/data/GNNforLRT/lightning_checkpoints'
    cfg_to_be_updated = OrderedDict(cfg)
    cfg_to_be_updated['datatype_split'] = [[1000, 500, 1000]]
    cfg_to_be_updated['input_dir'] = InputDir
    cfg_to_be_updated['output_dir'] = '/global/cfs/cdirs/m3443/data/GNNforLRT/results/TTbar_noPU_GNN_search'
    trials = 100
    for trial in range(trials):
        cfg_to_be_updated.update(hyperparam_randon_choice_gnn())
        with open(f'./HyperOptim/GNN/filter-grid_search_trial-set1-{trial:04d}.yaml', 'w') as file:
            for k, v in cfg_to_be_updated.items():
                yaml.safe_dump({k: v}, file, default_flow_style=False)
            yaml.safe_dump({'TAG': f'gnn-grid_search_trial-set1-{trial:04d}'}, file, default_flow_style=False)
     
def hyperparam_randon_choice_gnn() -> Dict:
    choices = {
        'edgecut_cut': [0.001 * i for i in range(500)],
        'hidden': [32, 64, 128, 256, 512, 1024],
        'n_graph_iters': [2, 4, 6, 8, 10],
        'nb_node_layer': [2, 4, 6, 8],
        'nb_edge_layer': [2, 4, 6, 8],
        'weight': [0.01 * i for i in range(300)],
        'delta_eta': [0.01 * i for i in range(300)]
        
    }
    
    result = dict()
    
    for hyperparam_name in choices.keys():
        result[hyperparam_name] = random.choice(choices.get(hyperparam_name))
        
     
    
    return result 
    
if __name__ == '__main__':
    main()  
    
    
    
    