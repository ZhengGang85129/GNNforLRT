#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

import pandas as pd

from ExaTrkXDataIO import DataReader

import matplotlib.pyplot as plt
from ExaTrkXPlotting import Plotter, PlotConfig
from ExaTrkXPlots import hits, pairs, particles
def make_detector_plot(hits):

    fig, ax = plt.subplots()
    # draw less dots to reduce file size
    hits = hits.sample(frac=1).reset_index(drop=True)
    r = hits[0:hits.shape[0]//2].r
    z = hits[0:hits.shape[0]//2].z
    #x_to_draw = x_to_draw[:len(event_file.x)//10]
    ax.scatter(z/1000, r/1000, s=1, color='lightgrey')
    ax.set_xlabel("z [m]")
    ax.set_ylabel("r [m]")
    fig.savefig('123.png')
    return fig, ax
if __name__ == '__main__':
    data_dir = Path("/global/homes/z/zhenggan/workspace/Project/display")
    reader = DataReader(
        config_path=data_dir/'hit.yaml',
        base_dir=data_dir
    )

    event_ids = range(0, 10000)

    plt.rcParams.update({'font.size': 16})

    for event_id in event_ids:
        data = reader.read_one(evtid=event_id)
        if data is None: continue
        save = Path(f'results/event{event_id}')
        save.mkdir(exist_ok=True, parents=True)
        print(data)
        hit_data = data['hits']
        particle_data = data['particles']
        pair_data = pd.DataFrame(data={
            'hit_id_1': hit_data.iloc[data['edges']['sender']]['hit_id'].to_numpy(),
            'hit_id_2': hit_data.iloc[data['edges']['receiver']]['hit_id'].to_numpy()
        })
        make_detector_plot(hit_data)
        fig, ax = plt.subplots(figsize=(8, 8), tight_layout=True)

        Plotter(fig, {
            ax: [
                PlotConfig(
                    plot=hits.hit_plot,
                    data={
                        'hits': hit_data
                    }
                ),
                #PlotConfig(
                #    plot=pairs.hit_pair_plot,
                #    data={
                #        'hits': hit_data,
                #        'pairs': pair_data
                #    }
                #)
            ]
        }).plot(save=save/'hits.png')

        hit_with_particles = pd.merge(
            hit_data,
            particle_data[['particle_id', 'particle_type']],
            on='particle_id'
        )
        hit_filtered = hit_with_particles[hit_with_particles['particle_type'].isin([13, -13, 11, -11])]
        pair_filtered = pair_data[
            pair_data['hit_id_1'].isin(hit_data['hit_id']) &
            pair_data['hit_id_2'].isin(hit_data['hit_id'])
        ]

        fig, ax = plt.subplots(figsize=(8, 8), tight_layout=True)

        Plotter(fig, {
            ax: [
                PlotConfig(
                    plot=hits.hit_plot,
                    data={
                        'hits': hit_data
                    }
                ),
                PlotConfig(
                    plot=particles.particle_track_with_production_vertex,
                    data={
                        'hits': hit_filtered,
                        'pairs': pair_filtered,
                        'particles': particle_data,
                        'truth': data['truth']
                    },
                    args={
                        'line_width': 1.0
                    }
                ),
                PlotConfig(
                    plot=particles.particle_types,
                    data={
                        'hits': hit_filtered,
                        'pairs': pair_filtered,
                        'particles': particle_data,
                        'truth': data['truth']
                    }
                )

            ]
        }).plot(save=save/'particles.png')
        #break
