import os
from tracks.track_reconstruction.algorithm.Wrangler import reconstruct_and_match_tracks as Wrangler
import multiprocessing
from pathlib import Path
from functools import partial
import numpy as np
import pandas as pd
from ExaTrkXDataIO import DataReader
import time
from tracks.plot_configurations import (
    particle_filters,
)
import sys
from typing import List, Tuple, Union

def aggregate_results(results: List[Union[pd.DataFrame, float]]) -> Tuple[pd.DataFrame, float, float]:
    
    dataframes = []
    ghost_rates = []
    mean_track_lengths = []
    
    for df, statistics in results:
        dataframes.append(df)
        ghost_rates.append(statistics['ghost_rate'])
        mean_track_lengths.append(statistics['mean_track_length'])
    dataframes = pd.concat(dataframes, ignore_index=True)
    ghost_rates = np.array(ghost_rates).mean()
    mean_track_lengths = np.array(mean_track_lengths).mean()

    return dataframes, ghost_rates, mean_track_lengths
def search_parameter_space(filter_cut: float, walk_min: float, walk_max: float):
    FNAME = fname.replace('PARAM1', f'{filter_cut:.2f}'.replace('.', 'p'))
    FNAME = FNAME.replace('PARAM2', f'{walk_min:.2f}'.replace('.', 'p'))
    FNAME = FNAME.replace('PARAM3', f'{walk_max:.2f}'.replace('.', 'p'))
    with multiprocessing.Pool(processes = 8) as pool:
        reader = DataReader(
            config_path = configs,
            base_dir = base_dir
        )
        algo = partial(Wrangler, filter_cut = filter_cut, walk_min = walk_min, walk_max = walk_max)
        results = pool.map(algo, reader.read())
            
    
    particles, ghost_rate, mean_track_length = aggregate_results(results)       
    particles = particles[particle_filters[lepton_type](particles)]
    particles.to_csv(os.path.join(OUT_FOLDER, FNAME+f'_gen-{lepton_type}.csv'), index = False)
    particles[particles.is_trackable].to_csv(os.path.join(OUT_FOLDER, FNAME+f'_reco-{lepton_type}.csv'), index = False)
    particles[particles.is_trackable & particles.is_matched].to_csv(os.path.join(OUT_FOLDER, FNAME+f'_match-{lepton_type}.csv'), index = False)


from argparse import ArgumentParser

def my_parser():
    parser = ArgumentParser()
    parser.add_argument("--filter_cut", help="positional argument 1", type = float, required = True)
    parser.add_argument("--walk_min", help="optional argument", type = float, required=True)
    parser.add_argument("--walk_max", help="optional argument", type = float, required = True)
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    global configs, base_dir, lepton_type, fname, parser
    parser = my_parser() 
    
    
    fname = 'filtercutPARAM1_wrangercut-minPARAM2_wrangercut-maxPARAM3' 
    OUT_FOLDER = 'metrics/final/experiments_on_filter_cut'
    if not os.path.isdir(OUT_FOLDER):
        os.mkdir(OUT_FOLDER)
    configs = Path('./tracks/DBSCAN_config/HNL_FlatPU-ModelHNL_PU200.yaml') #!CHANGE ME!
    
    base_dir = Path(os.path.basename(configs))
    lepton_type = 'displaced' #!CHANGE ME! option:  displaced/prompt/HSS/prompt
    start = time.time()
    print(f'grid search starting... parameters: (filter_cut= {parser.filter_cut}, walk_min = {parser.walk_min}, walk_max = {parser.walk_max})')
    search_parameter_space(filter_cut = parser.filter_cut, walk_min = parser.walk_min, walk_max = parser.walk_max)
    print(time.time() - start, ' sec')
    print('Completed.')