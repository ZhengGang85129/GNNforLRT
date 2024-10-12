#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import multiprocessing
from pathlib import Path

import numpy as np
import pandas as pd

from ExaTrkXDataIO import DataReader

import matplotlib.pyplot as plt
from ExaTrkXPlotting import Plotter, PlotConfig
from ExaTrkXPlots.tracks import (
    tracks,
    tracking_efficiency
)

from workflows import reconstruct_and_match_tracks
from plot_configurations import (
    particle_filters,
)

import sys, os
def plot_tracks(particles, save):

    # Setup data.
    generated = particles
    reconstructable = particles[particles.is_trackable]
    matched = particles[particles.is_trackable & particles.is_matched]

    print(f'#generated: {len(generated)}')
    print(f'#reconstructable: {len(reconstructable)}')
    print(f'#matched: {len(matched)}')
    print(generated)
    
    args = []
    
    args.append(
        {
            'var_col': 'pt',
            'var_name': '$p_T$ [GeV]',
            'bins': 10 ** np.arange(0, 2.1, 0.1),
            'ax_opts': {
                'xscale': 'log'
            }
        }
    ) 
    args.append(
        {
            'var_col': 'eta',
            'var_name': r'$\eta$',
            'bins': np.arange(-4.0, 4.1, 0.4)
        }
    )
     
    
    args.append(
        {
            'var_col': 'd0',
            'var_name': r'$d0$',
            'bins': np.arange(0, 0.05, step=0.005)
            #'bins': np.arange(0, 800, )
        }
    ) 
    
    
    args.append(
        {
            'var_col': 'z0',
            'var_name': r'$z0$',
            'bins': np.arange(-200, 201, step=20)
            #'bins': np.arange(-4.0, 4.1, 0.4)
        }
        
    )
    plots = [tracking_efficiency, tracking_efficiency,tracking_efficiency, tracking_efficiency,]
    
    for index, (plot, arg) in enumerate(zip(plots, args)):
        fig, ax = plt.subplots(1, 1, figsize=(8, 8), tight_layout=True)
        plotter = Plotter(fig)
        plotter[ax] = PlotConfig(
            args = arg,
            plot = plot
        )
        plotter.data = {
            'generated': generated,
            'reconstructable': reconstructable,
            'matched': matched
        }

        plotter.plot(save= Path(f'{index}-noPU.png'))


if __name__ == '__main__':
    
    if len(sys.argv) != 2:
        raise RuntimeError('usage: python3 ./track_reconstruction_DBSCAN.py <DBSCAN configuration file>')   
    
    path = Path(f'{sys.argv[1]}')
    base_dir = Path(f'{os.path.basename(sys.argv[1])}')
    save = Path('.')
   
    save.mkdir(parents=True, exist_ok=True)

    with multiprocessing.Pool(processes=8) as pool:

        reader = DataReader(
            config_path=path,
            base_dir=base_dir
        )
        particles = pd.concat(
            pool.map(reconstruct_and_match_tracks, reader.read())
        )

    # All.
    #plot_tracks(
    #    particles,
    #    save=save / 'all.pdf'
    #)

    # Displaced.
    #plot_tracks(
    #    particles[
    #        particle_filters['displaced'](particles)
    #    ],
    #    save=save / 'displaced.pdf'
    #)

    # Prompt.
    #plot_tracks(
    #    particles[
    #        particle_filters['prompt'](particles)
    #    ],
    #    save=save / 'prompt.pdf'
    #)
    #plot_tracks(
    #    particles,
    #    save=save / 'all.png'
    #)

    # Displaced.
    #plot_tracks(
    #    particles[
    #        particle_filters['displaced'](particles)
    #    ],
    #    save=save / 'displaced.png'
    #)

    # Prompt.
    plot_tracks(
        particles[
            particle_filters['prompt'](particles)
        ],
        save=save / 'prompt.png'
    )
