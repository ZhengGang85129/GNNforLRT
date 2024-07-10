from ExaTrkXDataIO import DataReader
import sys
import os
sys.path.append(os.getcwd())
from log.log_lib import plot_train_log
from functools import partial

if __name__ == '__main__':
    
    if len(sys.argv) != 2: raise RuntimeError('usage: python3 ./log/logs.py <stage: embedding/filter/gnn>')    
    
    for log in DataReader(
        config_path=f'log/{sys.argv[1]}.yaml',
        base_dir=''
    ).read():
        print(f"======{log.version}======")
        plot_train_log(
            log[f'{sys.argv[1]}_train'],
            log[f'{sys.argv[1]}_val'],
            f'metrics/{sys.argv[1]}'
        )
