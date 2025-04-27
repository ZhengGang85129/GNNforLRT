import os
from tracks.track_reconstruction.algorithm.Wrangler import reconstruct_and_match_tracks as Wrangler
from scipy.spatial import cKDTree
from tracks.workflows import reconstruct_and_match_tracks as DBSCAN
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
import matplotlib.pyplot as plt
import sys
import torch
algorithm = sys.argv[1]
import matplotlib.pyplot as plt
import random
import seaborn as sns

colors = plt.cm.tab10.colors  # 這會給你一個長度為 10 的 tuple list
colors = list(colors)  
random.shuffle(colors)  # 隨機打亂顏色順序
def aggregate_results(results: List[Union[pd.DataFrame, float]]) -> Tuple[pd.DataFrame, float, float]:
    
    dataframes = []
    hits = []
    constructed_tracks = []
    edges = []
    matched_tracks = [] 
    for df, statistics in results:
        dataframes.append(df)
        hits.append(statistics['hits'])
        constructed_tracks.append(statistics['constructed_tracks'])
        if statistics.get('edges', None) is not None:
            edges.append(statistics['edges'])
        if statistics.get('matched_track', None) is not None:
            matched_tracks.append(statistics['matched_track'])
    dataframes = pd.concat(dataframes, ignore_index=True)
    hits = pd.concat(hits, ignore_index=True)
    constructed_tracks = pd.concat(constructed_tracks, ignore_index=True)
    if edges != []:
        edges = pd.concat(edges, ignore_index=True)
    if matched_tracks != []:
        matched_tracks = pd.concat(matched_tracks, ignore_index=True)
    return dataframes, hits, constructed_tracks, edges, matched_tracks

lepton_type = 'displaced'
configs = Path('./tracks/DBSCAN_config/HNL_PU200-ModelHNL_PU200.yaml') 
base_dir = Path(os.path.basename(configs))



with multiprocessing.Pool(processes = 8) as pool:
    reader = DataReader(
    config_path = configs,
    base_dir = base_dir
    )
    if algorithm == 'DBSCAN':
        algo = partial(eval(algorithm), epsilon = 0.20, statistics = True)
    else:
        algo = partial(eval(algorithm), filter_cut = 0.1, walk_min = 0.4, walk_max = 0.9, statistics = True)
    results = pool.map(algo, reader.read())

    particles, hits_df, constructed_tracks_df, edges_df, matched_track_df = aggregate_results(results)
#edges_df = edges_df.drop(colums)
    hit_ids = hits_df['hit_id'].reset_index(drop=True)
    edges_df['sender_hit_id'] = edges_df['sender'].map(hit_ids)
    edges_df['receiver_hit_id'] = edges_df['receiver'].map(hit_ids)

    particles = particles[particle_filters[lepton_type](particles)][particles.is_matched]
    #print()
    matched_track_df = matched_track_df[matched_track_df.major_particle_id.isin(particles.particle_id)]
    matched_track_df['purity'] = matched_track_df['major_nhits']/matched_track_df['nhits']
    constructed_tracks_df = constructed_tracks_df[constructed_tracks_df.track_id.isin(matched_track_df.track_id)]
    hits_df = hits_df[hits_df['hit_id'].isin(constructed_tracks_df.seed_id)]
plot = False
if plot:
    plt.figure(figsize=(8, 6))
    sns.histplot(data = particles, x = 'z0', y = 'vr', bins = 60, color = 'red', label = 'particles[signal]', alpha = 0.5, cbar = True)
    plt.title('Signal decay origin (r vs z)')
    plt.xlabel('z [m]')
    plt.ylabel('r [m]')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('seed_hits_spatial_distribution.png')   