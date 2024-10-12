import yaml
from typing import List, Dict
import random
from collections import OrderedDict

def represent_dictionary_order(self, dict_data):
    return self.represent_mapping('tag:yaml.org,2002:map', dict_data.items())
def setup_yaml():
    yaml.add_representer(OrderedDict, represent_dictionary_order)

def produce_hyperparam(trials: int = 10, InputDir: str = None, stage: str = 'GNN', sample_type: str = 'PU200') -> None:
    with open(f'LightningModules/{stage}/train-{stage}_template.yaml', 'r') as file:
        cfg = yaml.safe_load(file)
    
    cfg_to_be_updated = OrderedDict(cfg)
    if stage == 'Embedding':
        cfg_to_be_updated['train_split'] = [[1000, 250, 250]]
    else:
        cfg_to_be_updated['datatype_split'] = [[1000, 250, 250]]
    cfg_to_be_updated['input_dir'] = InputDir
    cfg_to_be_updated['output_dir'] = f'/global/cfs/cdirs/m3443/data/GNNforLRT/results/TTbar_{sample_type}_{stage}_output-smeared'
    cfg_to_be_updated['stage_dir'] = 'HyperOptim/results/GNN_optim_for_MIX' #FIXME
    trials = 100
    for trial in range(trials):
        hyperparam_random_choice = eval(f'hyperparam_random_choice_{stage.lower()}')
        cfg_to_be_updated.update(hyperparam_random_choice())
        with open(f'./LightningModules/{stage}/optim_{stage}-{trial:04d}.yaml', 'w') as file:
            for k, v in cfg_to_be_updated.items():
                yaml.safe_dump({k: v}, file, default_flow_style=False)
            yaml.safe_dump({'TAG': f'{stage}-{trial:04d}'}, file, default_flow_style=False)


def main(stage: str = 'GNN'):
    setup_yaml()
    seed = 429
    random.seed(seed)  
    
    produce_hyperparam(trials = 100, InputDir = f'/global/cfs/cdirs/m3443/data/GNNforLRT/results/TTbar_MIX_Filter_output-smeared', stage = 'GNN', sample_type = 'PU200')#FIXME 

def hyperparam_random_choice_filter() -> Dict:
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

 
def hyperparam_random_choice_embedding() -> Dict: 

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
        
     
def hyperparam_random_choice_gnn() -> Dict:
    choices = {
        'edge_cut': [0.001 * i for i in range(500)],
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
    
    
    
    