import os
    
from track_reconstruction.algorithm.Wrangler_track_reco import reconstruct_and_match_tracks as Wrangler

from scipy.spatial import cKDTree
import multiprocessing
from pathlib import Path
from functools import partial
import numpy as np
import pandas as pd
from ExaTrkXDataIO import DataReader
import time
from plot_configurations import (
    particle_filters,
)
import sys
from typing import List, Tuple, Union, Dict
import matplotlib.pyplot as plt
import sys
import torch
import matplotlib.pyplot as plt
import random
import seaborn as sns

if len(sys.argv) != 2:
    raise ValueError('python3 ./visualize_seed_quality configs/XXX.yaml')


configs = Path(sys.argv[1]) 


colors = plt.cm.tab10.colors
colors = list(colors)  
random.shuffle(colors)  


def aggregate_results(results: List[Union[pd.DataFrame, float]]) -> Tuple[pd.DataFrame,Dict[str, pd.DataFrame]]:
    dataframes = []
    aggregate_stats = {}
    
    for df, statistics in results:
        dataframes.append(df)
        for key, value in statistics.items():
            if key not in aggregate_stats:
               aggregate_stats[key] = []
            aggregate_stats[key].append(value)  
         
    dataframes = pd.concat(dataframes, ignore_index=True)
    
    for key in aggregate_stats:
        if isinstance(aggregate_stats[key][0], pd.DataFrame):
            aggregate_stats[key] = pd.concat(aggregate_stats[key], ignore_index = True)
    
     
    return dataframes, aggregate_stats 

lepton_type = 'displaced'
base_dir = Path(os.path.basename(configs))



with multiprocessing.Pool(processes = 8) as pool:
    reader = DataReader(
    config_path = configs,
    base_dir = base_dir
    )
    algo = partial(Wrangler, filter_cut = 0.1, walk_min = 0.4, walk_max = 0.9, return_statistics = True)
    results = pool.map(algo, reader.read())

    particles, aggregate_stats= aggregate_results(results)
    hit_ids = aggregate_stats['hits']['hit_id'].reset_index(drop=True)
    hits_df = aggregate_stats['hits']
    edges_df = aggregate_stats['edges']
    matched_track_df = aggregate_stats['matched_track']
    constructed_tracks_df = aggregate_stats['constructed_tracks']
    edges_df['sender_hit_id'] = edges_df['sender'].map(hit_ids)
    edges_df['receiver_hit_id'] = edges_df['receiver'].map(hit_ids)

    particles = particles[particle_filters[lepton_type](particles)][particles.is_matched]
    #print()
    matched_track_df = matched_track_df[matched_track_df.major_particle_id.isin(particles.particle_id)]
    matched_track_df['purity'] = matched_track_df['major_nhits']/matched_track_df['nhits']
    constructed_tracks_df = constructed_tracks_df[constructed_tracks_df.track_id.isin(matched_track_df.track_id)]
    hits_df = hits_df[hits_df['hit_id'].isin(constructed_tracks_df.seed_id)]



plt.figure(figsize=(8, 6))
sns.histplot(data = particles, x = 'z0', y = 'vr', bins = 60, color = 'red', label = 'particles[signal]', alpha = 0.5, cbar = True)
plt.title('Signal decay origin (r vs z)')
plt.xlabel('z [m]')
plt.ylabel('r [m]')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('seed_hits_spatial_distribution.png')   