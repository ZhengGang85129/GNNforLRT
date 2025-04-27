import os
from tracks.track_reconstruction.algorithm.Wrangler import reconstruct_and_match_tracks_visualization as Wrangler
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
colors = plt.cm.tab10.colors  # 這會給你一個長度為 10 的 tuple list
colors = list(colors)  
random.shuffle(colors)  # 隨機打亂顏色順序
def aggregate_results(results: List[Union[pd.DataFrame, float]]) -> Tuple[pd.DataFrame, float, float]:
    
    dataframes = []
    hits = []
    constructed_tracks = []
    matched_tracks = []
    edges = [] 
    for df, statistics in results:
        dataframes.append(df)
        hits.append(statistics['hits'])
        constructed_tracks.append(statistics['constructed_tracks'])
        if statistics['edges'] is not None:
            edges.append(statistics['edges'])
        matched_tracks.append(statistics['matched_track'])
    dataframes = pd.concat(dataframes, ignore_index=True)
    hits = pd.concat(hits, ignore_index=True)
    constructed_tracks = pd.concat(constructed_tracks, ignore_index=True)
    if edges != []:
        edges = pd.concat(edges, ignore_index=True)
    matched_tracks = pd.concat(matched_tracks, ignore_index = True)
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
        algo = partial(eval(algorithm), filter_cut = 0.1, walk_min = 0.05, walk_max = 0.95, statistics = True)
    results = pool.map(algo, reader.read())

particles, hits_df, constructed_tracks_df, edges_df, matched_tracks_df = aggregate_results(results)
#edges_df = edges_df.drop(colums)
hit_ids = hits_df['hit_id'].reset_index(drop=True)
edges_df['sender_hit_id'] = edges_df['sender'].map(hit_ids)
edges_df['receiver_hit_id'] = edges_df['receiver'].map(hit_ids)

particles = particles[particle_filters[lepton_type](particles)]

hits_from_particles = pd.merge(hits_df, particles, on = 'particle_id', how = 'inner')
hits_from_tracks = pd.merge(hits_df, constructed_tracks_df, on = 'hit_id', how = 'inner')
hits_tracks_particles = pd.merge(hits_from_tracks, particles, on = 'particle_id', how = 'inner')


plt.scatter(hits_df['z']/1000., hits_df['r']/1000, s = 1, marker = '.', color = 'red', alpha = 0.05)

count = 0
score_threshold = 0.1
true_edges_labeled = False
fake_edges_labeled = False
true_hits_labeled = False
show_score = 0.5
track_edges_color = ['blue', 'green', 'orange', 'purple', 'brown', 'pink']
n_tracks = 0
'''
for _, particle in particles.iterrows():
    if count == 1:
        count += 1
        continue 
    print(particle.particle_id)
    mask = hits_from_particles['particle_id'] == particle.particle_id
    
    z_hits_p = hits_from_particles[hits_from_particles['particle_id'] == particle.particle_id]['z']/1000. 
    
    r_hits_p = hits_from_particles[hits_from_particles['particle_id'] == particle.particle_id]['r']/1000. 
    
    
    valid_track_id = set(hits_from_tracks[hits_from_tracks.hit_id.isin(hits_df[hits_df['particle_id'] == particle.particle_id].hit_id)].track_id.unique())
    
    for track_id in valid_track_id:
        hits_in_gentracks = set(hits_tracks_particles[hits_tracks_particles['track_id'] == track_id][hits_tracks_particles.particle_id == particle.particle_id].hit_id.unique())
        for sender, receiver, score, truth in zip(edges_df['sender_hit_id'].values, edges_df['receiver_hit_id'].values, edges_df['score'].values, edges_df['truth'].values): 
            if (sender in hits_in_gentracks or receiver in hits_in_gentracks) and score > score_threshold:
                if not truth:
                    z1, r1 = hits_df[hits_df['hit_id'] == sender]['z'].values[0]/1000., hits_df[hits_df['hit_id'] == sender]['r'].values[0]/1000.
                    z2, r2 = hits_df[hits_df['hit_id'] == receiver]['z'].values[0]/1000., hits_df[hits_df['hit_id'] == receiver]['r'].values[0]/1000.
                    plt.plot([z1, z2], [r1, r2],'k--.' , alpha = score, label = 'edges[fake]' if not fake_edges_labeled else '') 
                    fake_edges_labeled = True 
                    xm, ym = (z1 + z2) / 2, (r1 + r2) / 2
                    #if score > show_score:
                    #    plt.text(xm - 0.2, ym , str(round(score, 2)), fontsize=8, ha='center', va='center', color = 'grey') 
                else:
                    z1, r1 = hits_df[hits_df['hit_id'] == sender]['z'].values[0]/1000., hits_df[hits_df['hit_id'] == sender]['r'].values[0]/1000.
                    z2, r2 = hits_df[hits_df['hit_id'] == receiver]['z'].values[0]/1000., hits_df[hits_df['hit_id'] == receiver]['r'].values[0]/1000.
                    plt.plot([z1, z2], [r1, r2],'-' , alpha = score, label = 'edges[true]' if not true_edges_labeled else '', color = colors[n_tracks % len(colors)]) 
                    true_edges_labeled = True
                    xm, ym = (z1 + z2) / 2, (r1 + r2) / 2
                    if score > show_score:
                        plt.text(xm, ym, str(round(score, 2)), fontsize=8, ha='center', va='center') 
        n_tracks += 1 
        
    plt.scatter(z_hits_p, r_hits_p, s = 80, marker = '*', label = 'truth[signal]' if not true_hits_labeled else '')
    true_hits_labeled = True
    break 
    

    count += 1     
'''
plt.legend(loc = 'best')
plt.title(f'Track Visualization [{algorithm}]')
plt.xlabel('z [m]')
plt.ylabel('r [m]')
plt.tight_layout()
plt.savefig('track_visualization.png') 