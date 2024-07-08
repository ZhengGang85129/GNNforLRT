#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import multiprocessing
import math
from pathlib import Path

import numpy as np
import pandas as pd

from ExaTrkXDataIO import DataReader

import matplotlib.pyplot as plt
from ExaTrkXPlotting import Plotter, PlotConfig
from ExaTrkXPlots.tracks import (
    tracks,
    tracking_efficiency,
    tracking_efficiency_techical,
    tracking_efficiency_physical
)

from workflows import reconstruct_and_match_tracks

ptype_filters = {
    # W bosons.
    # r"$W^+$/$W^-$": lambda particles: (
    #    particles['particle_type'].abs() == 24
    # ),
    # Electrons.
    r"$e^+$/$e^-$": lambda particles: (
        particles['particle_type'].abs() == 11
    ),
    # Muons.
    r"$\mu^+$/$\mu^-$": lambda particles: (
        particles['particle_type'].abs() == 13
    ),
    # Pions.
    r"$\pi^+$/$\pi^-$": lambda particles: (
        particles['particle_type'].abs() == 211
    ),
    # Proton.
    r"$p^+$/$p^-$": lambda particles: (
        particles['particle_type'].abs() == 2212
    ),
    # Kion
    r"$K^+$/$K^-$": lambda particles: (
        particles['particle_type'].abs() == 321
    ),
}

plot_configs = {
    'pt': {
        'plot': tracking_efficiency_techical,
        'args': {
            'x_variable': 'pt',
            'x_label': 'Truth Transverse Momentum $p_T$ [GeV]',
            'bins': 10 ** np.arange(0, 2.1, 0.1),
            'ax_opts': {
                'xscale': 'log'
            }
        }
    },
    'd0': {
        'plot': tracking_efficiency_techical,
        'args': {
            'x_variable': 'd0',
            'x_label': r'Truth Transverse Impact Parameter $|d_0|$[mm]',
            'bins': np.arange(0, 800, step=50)
        }
    },
    'z0': {
        'plot': tracking_efficiency_techical,
        'args': {
            'x_variable': 'z0',
            'x_label': r'Truth Longitudinal Impact Parameter $z_0$[mm]',
            'bins': np.arange(-2000, 2001, step=200)
        }
    },
    'vr': {
        'plot': tracking_efficiency_techical,
        'args': {
            'x_variable': 'vr',
            'x_label': 'Production vertex radius [mm]',
            'bins': np.arange(0.0, 301, 25)
        }
    }
}


def plot_tracks_technical(particles, save):
    plt.rcParams.update({'font.size': 12})

    generated = particles
    reconstructable = particles[particles.is_trackable]
    matched = particles[
        particles.is_trackable & particles.is_matched
    ]
    eff = len(matched) / len(reconstructable) if len(reconstructable) > 0 else 0

    print((
        f'#tracks(All):\n'
        f' generated:       {len(generated)}\n'
        f' reconstructable: {len(reconstructable)}\n'
        f' matched:         {len(matched)}'
    ))
    print(f'Technical Efficiency: {eff}')

    print(pd.unique(particles['particle_type']))

    for plot_type, plot_config in plot_configs.items():
        fig, ax = plt.subplots(1, 1, figsize=(8, 8), tight_layout=True)
        plotter = Plotter(fig)

        for label, ptype_filter in ptype_filters.items():
            is_ptype = ptype_filter(particles)

            generated = particles[is_ptype]
            reconstructable = particles[is_ptype & particles.is_trackable]
            matched = particles[
                is_ptype & particles.is_trackable & particles.is_matched
            ]

            eff = len(matched)/len(reconstructable) if len(reconstructable) > 0 else 0

            fraction = len(generated)/len(particles)

            print((
                f'#tracks({label}):\n'
                f' generated:       {len(generated)}({100*fraction:.2f}%)\n'
                f' reconstructable: {len(reconstructable)}\n'
                f' matched:         {len(matched)}'
            ))
            print(f'Technical Efficiency: {eff}\n')
            print(f'Parents:', pd.unique(generated["parent_ptype"]))

            plotter[ax].append(PlotConfig(
                config=plot_config,
                args={
                    'label': f'{label}, eff={eff*100:.2f}%'
                },
                data={
                    'generated': generated,
                    'reconstructable': reconstructable,
                    'matched': matched
                }
            ))

        plotter.plot(Path(save) / f'{plot_type}.pdf')


if __name__ == '__main__':
    with multiprocessing.Pool(processes=8) as pool:
        reader = DataReader(
            config_path='../configs/reading/processed/gnn.yaml',
            # config_path='../configs/reading/v4/processed/gnn.yaml',
            base_dir='../../data'
        )
        particles = pd.concat(
            pool.map(reconstruct_and_match_tracks, reader.read())
        )

    save = Path('../../output/plots/tracks/ptypes')
    save.mkdir(parents=True, exist_ok=True)

    plot_tracks_technical(
        particles,
        save=save
    )
