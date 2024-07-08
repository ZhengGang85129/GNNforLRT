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
from plot_configurations import (
    track_types,
    particle_parameters,
    particle_filters,
    plot_configs
)

# Which HNL property configuration will be used.
HNL_configs = [
    '10GeV100mm',
    '15GeV100mm',
    '15GeV200mm'
    #'ttbar'
]
# Event range.
evtids = {
    '10GeV100mm': range(10000),
    '15GeV100mm': range(5000),
    '15GeV200mm': range(10000),
    'ttbar': range(10000)
}
# Labels show on plot for each data.
labels = {
    '10GeV100mm': r'$mass=10GeV, c\tau = 100mm$',
    '15GeV100mm': r'$mass=15GeV, c\tau = 100mm$ (Train Dataset)',
    '15GeV200mm': r'$mass=15GeV, c\tau = 200mm$',
    'ttbar': r'$t\bar{t}$'
}


def plot_tracks_technical(particles, save):
    # Setup data categorized with track type.
    data = {}
    for HNL_config, particles_for_config in particles.items():
        data[HNL_config] = {}
        for track_type, particle_filter in particle_filters.items():
            if particle_filter is not None:
                filter_result = particle_filter(particles_for_config)
            else:
                filter_result = np.ones_like(particles_for_config)
            data[HNL_config][track_type] = {
                'generated': particles_for_config[
                    filter_result
                ],
                'reconstructable': particles_for_config[
                    filter_result & particles_for_config.is_trackable
                ],
                'matched': particles_for_config[
                    filter_result & particles_for_config.is_trackable & particles_for_config.is_matched
                ]
            }

    # Plot tracks.
    for track_type in track_types:
        for name, config in plot_configs[track_type].items():
            plt.rcParams.update({'font.size': 18})

            fig, ax = plt.subplots(1, 1, figsize=(8, 8), tight_layout=True)
            plotter = Plotter(fig)

            plotter[ax] = [
                PlotConfig(
                    plot=tracking_efficiency_techical,
                    config=config,
                    args={
                        'errbar_opts': {
                            'label': labels[HNL_config]
                        }
                    },
                    data=data[HNL_config][track_type]
                )
                for HNL_config in particles.keys()
            ]

            plotter.plot(Path(save) / f'{track_type}_{name}.pdf')


if __name__ == '__main__':
    # Read and reconstruct particles.
    particles = {}
    for HNL_config in HNL_configs:
        with multiprocessing.Pool(processes=8) as pool:
            reader = DataReader(
                config_path='../configs/reading/v4/processed/gnn.yaml',
                base_dir='../../data'
            )
            reader.variables['HNL'] = [HNL_config]
            reader.variables['evtid'] = evtids[HNL_config]

            particles[HNL_config] = pd.concat(
                pool.map(reconstruct_and_match_tracks, reader.read())
            )

    save = Path('../../output/plots/tracks/robustness-poster')
    save.mkdir(parents=True, exist_ok=True)

    plot_tracks_technical(
        particles,
        save=save
    )
