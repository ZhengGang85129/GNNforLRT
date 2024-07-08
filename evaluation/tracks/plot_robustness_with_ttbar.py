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
    tracking_efficiency,
    tracking_efficiency_physical,
    tracking_efficiency_techical
)

from workflows import reconstruct_and_match_tracks

plot_config = {
    'pt': {
        'args': {
            'var_col': 'pt',
            'var_name': 'Truth Transverse Momentum $p_T$ [GeV]',
            'bins': np.arange(1.0, 30.5, 3.0)
        }
    },
    'd0': {
        'args': {
            'var_col': 'd0',
            'var_name': r'Truth Transverse Impact Parameter $|d_0|$[mm]',
            'bins': np.arange(0, 200, step=10)
        }
    },
    'z0': {
        'args': {
            'var_col': 'z0',
            'var_name': r'Truth Longitudinal Impact Parameter $z_0$[mm]',
            'bins': np.arange(-500, 501, step=50)
        }
    },
    'vr': {
        'args': {
            'var_col': 'vr',
            'var_name': 'Production vertex radius [mm]',
            'bins': np.arange(0.0, 301, 25)
        }
    }
}


def plot_tracks_technical(particles, save):
    # Setup data categorized with track type.
    data = {
        'generated': particles,
        'reconstructable': particles[
            particles.is_trackable
        ],
        'matched': particles[
            particles.is_trackable & particles.is_matched
        ]
    }

    # Plot tracks.
    for name, config in plot_config.items():
        plt.rcParams.update({'font.size': 16})

        fig, ax = plt.subplots(1, 1, figsize=(8, 8), tight_layout=True)
        plotter = Plotter(fig)

        plotter[ax] = PlotConfig(
            plot=tracking_efficiency_techical,
            config=config,
            args={'label': r'$t\bar{t}$'},
            data=data
        )

        plotter.plot(Path(save) / f'{name}.pdf')


if __name__ == '__main__':
    # Read and reconstruct particles.
    with multiprocessing.Pool(processes=8) as pool:
        reader = DataReader(
            config_path='../configs/reading/v4/processed/gnn.yaml',
            base_dir='../../data'
        )
        reader.variables['HNL'] = ['ttbar']
        reader.variables['evtid'] = list(range(0, 10000))

        particles = pd.concat(
            pool.map(reconstruct_and_match_tracks, reader.read())
        )

    save = Path('../../output/plots/tracks/robustness/ttbar')
    save.mkdir(parents=True, exist_ok=True)

    plot_tracks_technical(
        particles,
        save=save
    )
