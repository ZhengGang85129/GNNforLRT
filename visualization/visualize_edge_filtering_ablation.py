from pathlib import Path
import pandas as pd
from ExaTrkXDataIO import DataReader
from plot_configurations import (
    particle_filters,
)
import matplotlib.pyplot as plt
import sys
import matplotlib.pyplot as plt
import multiprocessing
import os


configs = './hits_config.yaml'
base_dir = Path(os.path.basename(configs))
with multiprocessing.Pool(processes = 8) as pool:
    reader = DataReader(
    config_path = configs,
    base_dir = base_dir
    )

    data = reader.read_one(evtid = 298)
    
    
    hits = data['hits']
    edges = data['edges']
    particles = data['particles']
    particles = particles[particle_filters['displaced'](particles)]
    print(particles)
    particles = particles[particles.particle_id == 4523393084817408]
    hits_p = pd.merge(hits, particles, how = 'inner', on = 'particle_id')
    print(hits_p)
    hits_p = hits_p[hits_p.particle_id.isin(particles.particle_id)]
    #edges.columns = ['hit_id', 'receiver', 'truth', 'score']
    plt.scatter(hits.z/1000, hits.r/1000, label = 'Noise', color = 'red', alpha = 0.01, marker='.') 
    plt.scatter(hits_p.z/1000, hits_p.r/1000, label ='Signal')
    hit_ids = hits['hit_id'].reset_index(drop=True) 
    hits_p = pd.merge(hits, particles, how = 'inner', on = 'particle_id')
    print(hits_p)
    hits_p = hits_p[hits_p.particle_id.isin(particles.particle_id)]
    edges['sender_hit_id'] = edges['sender'].map(hit_ids)
    edges['receiver_hit_id'] = edges['receiver'].map(hit_ids) 
    
    count = 0 
    for sender, receiver in zip(edges['sender_hit_id'].values, edges['receiver_hit_id'].values):
        if sender in set(hits_p.hit_id.unique()) or receiver in set(hits_p.hit_id.unique()):
            z1, r1 = hits[hits['hit_id'] == sender]['z'].values[0]/1000., hits[hits['hit_id'] == sender]['r'].values[0]/1000.
            z2, r2 = hits[hits['hit_id'] == receiver]['z'].values[0]/1000., hits[hits['hit_id'] == receiver]['r'].values[0]/1000.
            plt.plot([z1, z2], [r1, r2],'k--.' , alpha = 0.1, label = 'edges' if count == 0 else '') 
            count += 1
    plt.xlim([-1, 0])                 
    plt.title('After Edge Filtering')
    
    
    plt.xlabel('z [m]')
    plt.xlabel('r [m]')
    plt.legend(loc = 'lower left')
    plt.savefig('embed.png') 