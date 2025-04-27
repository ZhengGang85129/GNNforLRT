import yaml
from typing import Dict
from collections import OrderedDict

import argparse
r'''
python3 ./config_hyperparam.py --embedding ./HyperOptim/Embedding/embed-best.yaml --filter ./HyperOptim/Filter/filter-best.yaml --gnn ./HyperOptim/GNN/gnn-best.yaml --task_name best --sample PU200 
'''

def myparser():

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--embedding', type = str, default = None)
    parser.add_argument('--filter', type = str, default = None)
    parser.add_argument('--gnn', type = str, default = None)
    parser.add_argument('--task_name', type = str, required = True)
    parser.add_argument('--sample', type = str, required = True, choices = ['noPU', 'PU40', 'PU200', 'MIX']) 
    parser.add_argument('--only', default = 'Embedding:Filter:GNN', choices = ['Embedding', 'Filter', 'GNN', 'Embedding:Filter', 'Embedding:GNN', 'Filter:GNN', 'Embedding:Filter:GNN'])
    args = parser.parse_args()
    return args 

def main():
    setup_yaml()
    global args
    args = myparser()
    for stage in ['Embedding', 'Filter', 'GNN']:
        if getattr(args, stage.lower()) is None:
            config_overwrite = None
        else:
            with open(getattr(args, stage.lower()), 'r') as file:
                 config_overwrite = yaml.safe_load(file)
        
        #if stage in args.only:
        #    Config_Write(stage = stage, config_overwrite = config_overwrite)
        #    print(f'Check -> ./LightningModules/{stage}/optim-{args.sample}-{args.task_name}.yaml')


def represent_dictionary_order(self, dict_data):
    return self.represent_mapping('tag:yaml.org,2002:map', dict_data.items())

def setup_yaml():
    yaml.add_representer(OrderedDict, represent_dictionary_order)

def Config_Write(stage: str, config_overwrite: Dict = None) -> None:
    
    
    with open(f'./LightningModules/{stage}/train-{stage}_template.yaml', 'r') as file:
    #with open('test-123.yaml', 'r')  as file:
        default_config = yaml.safe_load(file)
    
    updated_config = OrderedDict(default_config)
    if config_overwrite is not None:
        updated_config.update(config_overwrite)
        
    with open(f'./LightningModules/{stage}/train-{args.sample}-{args.task_name}.yaml', 'w')  as file:
        for k, v in updated_config.items():
            #yaml.dump({k: v}, file)
            yaml.safe_dump({k: v}, file, default_flow_style=False)
        yaml.safe_dump({'TAG': args.task_name}, file, default_flow_style=False)
    
    
    
if __name__ == '__main__':
    
    main()


