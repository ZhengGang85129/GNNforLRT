#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import multiprocessing
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from datetime import datetime

from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV


from ExaTrkXDataIO import DataReader
from ExaTrkXPlotting import Plotter, PlotConfig
from ExaTrkXPlots.tracks import (
    tracks,
    tracking_efficiency
)

from workflows import reconstruct_and_match_tracks
from plot_configurations import (
    particle_filters,
)


def plot_tracks(particles, save):
    fig, ax = plt.subplots(3, 2, figsize=(8, 8), tight_layout=True) # TODO: change it back
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

    plotter[ax[2, 0]] = PlotConfig(
        plot=tracking_efficiency,
        args={
            'var_col': 'd0',
            'var_name': '$d0$',
            'bins': np.arange(0, 500, 25),
            'ax_opts': {
                'xscale': 'linear'
            }
        }
    )

    plotter[ax[2, 1]] = PlotConfig(
        plot=tracking_efficiency,
        args={
            'var_col': 'z0',
            'var_name': '$z0$',
            'bins': np.arange(0, 500, 25),
            'ax_opts': {
                'xscale': 'linear'
            }
        }
    )

    plotter.plot(save=save)


if __name__ == '__main__':

    config_name = f"HNL_PU200.yaml"
    config_path = Path('tracks/DBSCAN_config')

    # root dir should be "analysis HSF"
    save = Path('metrics/final/')
    save.mkdir(parents=True, exist_ok=True)

    # reader
    reader = DataReader(
        # config_path='../configs/reading/processed/gnn.yaml',
        # config_path='../configs/reading/v4/processed/gnn.yaml',
        config_path=config_path/config_name,
        base_dir="."
    )

    # epsilons = np.linspace(0.01, 1, 20)
    epsilons = [0.16631578947368422]
    best_score = 0

    for epsilon in epsilons:
        def _reconstruct_and_match_tracks(data):
            return reconstruct_and_match_tracks(data=data, epsilon=epsilon)

        with multiprocessing.Pool(processes=8) as pool:

            particles = pd.concat(
                pool.map(_reconstruct_and_match_tracks,
                         reader.read(silent_skip=True))
            )

        matched = len(
            particles[particles.is_trackable & particles.is_matched])
        print("matched particles", matched)

        if matched >= best_score:
            best_score = matched
            best_eps = epsilon
            best_particle = particles

    print("The best epsilon is ", best_eps)
    # All.
    plot_tracks(
        best_particle,
        save=save /
        f'dbscan_{config_name}.pdf' # TODO: change it back
        # f'dbscan_z0.png' # TODO: change it back
    )

    # FIXME: The plot below requres parent type data, which is used to seperate displaced and prompt data
    # Displaced.
    # plot_tracks(
    #     particles[
    #         particle_filters['displaced'](particles)
    #     ],
    #     save=save / 'displaced.pdf'
    # )

    # # Prompt.
    # plot_tracks(
    #     particles[
    #         particle_filters['prompt'](particles)
    #     ],
    #     save=save / 'prompt.pdf'
    # )
