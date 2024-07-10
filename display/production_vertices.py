import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from ExaTrkXDataIO import DataReader

from pathlib import Path

if __name__ == '__main__':
    all_particles = []
    data_dir  = Path("")
    reader = DataReader(
        config_path=data_dir/'production_vertices.yaml',
        base_dir=data_dir
    )
    event_ids = range(1000, 4000)

    plt.rcParams.update({'font.size': 16})
    cnt = 0
    for event_id in event_ids:
        data = reader.read_one(evtid=event_id)
        if data is None: continue
        hit_data = data['hits']
        particle_data = data['particles']
        if particle_data is None: continue
        pair_data = pd.DataFrame(data={
            'hit_id_1': hit_data.iloc[data['edges']['sender']]['hit_id'].to_numpy(),
            'hit_id_2': hit_data.iloc[data['edges']['receiver']]['hit_id'].to_numpy()
        })

        pid = pd.DataFrame(
            data={'pid': data['particles']['particle_id']}
        )
        pid = pid.merge(particle_data, left_on='pid', right_on='particle_id', how='inner')
        pid = pid.fillna(0)

        all_particles.append(particle_data)
        hits = data['hits']['r']
        cnt += 1
        if cnt == 100:break  

    nhit = 0
    nparticles = 0


    all_particles = pd.concat(all_particles)

    fig, ax = plt.subplots(1, 1, figsize=(12, 4), tight_layout=True)

    ax.set_title('Production Vertex Position')

    h = ax.hist2d(
        x=all_particles.vz,
        y=np.sqrt(all_particles['vx']**2 + all_particles['vy']**2),
        bins=(
            np.arange(-10, 10, 0.1),
            np.arange(0, 10, 0.1)
        ),
        # cmap='Reds',
        norm=LogNorm()
    )
    ax.set_xlabel('z [mm]')
    ax.set_ylabel('r [mm]')

    fig.colorbar(h[3], ax=ax, label='#Particles')

    fig.savefig('results/production_vertices.pdf')
