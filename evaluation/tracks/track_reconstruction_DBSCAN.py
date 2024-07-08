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
    fig, ax = plt.subplots(2, 2, figsize=(8, 8), tight_layout=True)
    plotter = Plotter(fig)

    # Setup data.
    generated = particles
    reconstructable = particles[particles.is_trackable]
    matched = particles[particles.is_trackable & particles.is_matched]

    print(f'#generated: {len(generated)}')
    print(f'#reconstructable: {len(reconstructable)}')
    print(f'#matched: {len(matched)}')

    plotter.data = {
        'generated': generated,
        'reconstructable': reconstructable,
        'matched': matched
    }

    plotter[ax[0, 0]] = PlotConfig(
        plot=tracks,
        args={
            'var_col': 'pt',
            'var_name': '$p_T$ [GeV]',
            'bins': 10 ** np.arange(0, 2.1, 0.1),
            'ax_opts': {
                'xscale': 'log'
            }
        }
    )

    plotter[ax[0, 1]] = PlotConfig(
        plot=tracking_efficiency,
        args={
            'var_col': 'pt',
            'var_name': '$p_T$ [GeV]',
            'bins': 10 ** np.arange(0, 2.1, 0.1),
            'ax_opts': {
                'xscale': 'log'
            }
        }
    )

    plotter[ax[1, 0]] = PlotConfig(
        plot=tracks,
        args={
            'var_col': 'eta',
            'var_name': r'$\eta$',
            'bins': np.arange(-4.0, 4.1, 0.4)
        }
    )

    plotter[ax[1, 1]] = PlotConfig(
        plot=tracking_efficiency,
        args={
            'var_col': 'eta',
            'var_name': r'$\eta$',
            'bins': np.arange(-4.0, 4.1, 0.4)
        }
    )

    plotter.plot(save=save)


if __name__ == '__main__':
    
    if len(sys.argv) != 2:
        raise RuntimeError('usage: python3 ./track_reconstruction_DBSCAN.py <DBSCAN configuration file>')   
    
    path = Path(f'{sys.argv[1]}')
    base_dir = Path(f'{os.path.basename(sys.argv[1])}')
    save = Path('../metrics')
   
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
    plot_tracks(
        particles,
        save=save / 'all.pdf'
    )

    # Displaced.
    plot_tracks(
        particles[
            particle_filters['displaced'](particles)
        ],
        save=save / 'displaced.pdf'
    )

    # Prompt.
    plot_tracks(
        particles[
            particle_filters['prompt'](particles)
        ],
        save=save / 'prompt.pdf'
    )
    plot_tracks(
        particles,
        save=save / 'all.png'
    )

    # Displaced.
    plot_tracks(
        particles[
            particle_filters['displaced'](particles)
        ],
        save=save / 'displaced.png'
    )

    # Prompt.
    plot_tracks(
        particles[
            particle_filters['prompt'](particles)
        ],
        save=save / 'prompt.png'
    )
