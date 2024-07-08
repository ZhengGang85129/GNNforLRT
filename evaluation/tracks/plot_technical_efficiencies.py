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
    tracking_efficiency_techical
)

from workflows import reconstruct_and_match_tracks
from plot_configurations import (
    particle_filters
)


def plot_tracks_technical(particles, save):
    save = Path(save)
    save.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({'font.size': 24})

    is_displaced = particle_filters['displaced'](particles)
    is_prompt = particle_filters['prompt'](particles)

    # Show statistics
    eff_all = len(particles[
        particles.is_trackable & particles.is_matched
    ]) / len(particles[
        particles.is_trackable
    ])
    print(f"all: {eff_all}")

    eff_hard = len(particles[
        (is_prompt | is_displaced) & particles.is_trackable & particles.is_matched
    ]) / len(particles[
        (is_prompt | is_displaced) & particles.is_trackable
    ])
    print(f"hard process: {eff_hard}")

    eff_prompt = len(particles[
        is_prompt & particles.is_trackable & particles.is_matched
    ]) / len(particles[
        is_prompt & particles.is_trackable
    ])
    print(f"prompt: {eff_prompt}")

    eff_displaced = len(particles[
        is_displaced & particles.is_trackable & particles.is_matched
    ]) / len(particles[
        is_displaced & particles.is_trackable
    ])
    print(f"displaced: {eff_displaced}")

    print(f"#prompt+#displaced: {len(particles[is_prompt | is_displaced])}")
    print(f"#prompt: {len(particles[is_prompt])}")
    print(f"#displaced: {len(particles[is_displaced])}")

    # Define plotting configuration.
    configs = {
        'pt': {
            'plot': tracking_efficiency_techical,
            'args': {
                'var_col': 'pt',
                'var_name': 'Truth Transverse Momentum $p_T$ [GeV]',
                'bins': 10 ** np.arange(0, 2.1, 0.1),
                'ax_opts': {
                    'xscale': 'log'
                }
            }
        },
        'd0': {
            'plot': tracking_efficiency_techical,
            'args': {
                'var_col': 'd0',
                'var_name': r'Truth Transverse Impact Parameter $|d_0|$[mm]',
                'bins': np.arange(0, 800, step=50)
            }
        },
        'z0': {
            'plot': tracking_efficiency_techical,
            'args': {
                'var_col': 'z0',
                'var_name': r'Truth Longitudinal Impact Parameter $z_0$[mm]',
                'bins': np.arange(-2000, 2001, step=200)
            }
        },
        'vr': {
            'plot': tracking_efficiency_techical,
            'args': {
                'var_col': 'vr',
                'var_name': 'Production vertex radius [mm]',
                'bins': np.arange(0.0, 301, 10)
            }
        }
    }

    for name, config in configs.items():
        plt.rcParams.update({'font.size': 20})
        fig, ax = plt.subplots(1, 1, figsize=(8, 8), tight_layout=True)

        plotter = Plotter(fig)
        plotter[ax].append(PlotConfig(
            config=config,
            args={
                'errbar_opts': {
                    'label': 'Prompt'
                }
            },
            data={
                'generated': particles[
                    is_prompt
                ],
                'reconstructable': particles[
                    is_prompt & particles.is_trackable
                ],
                'matched': particles[
                    is_prompt & particles.is_trackable & particles.is_matched
                ]
            }
        ))

        plotter[ax].append(PlotConfig(
            config=config,
            args={
                'errbar_opts': {
                    'label': 'Displaced'
                }
            },
            data={
                'generated': particles[
                    is_displaced
                ],
                'reconstructable': particles[
                    is_displaced & particles.is_trackable
                ],
                'matched': particles[
                    is_displaced & particles.is_trackable & particles.is_matched
                ]
            }
        ))

        plotter.plot(save=Path(save)/f'{name}.pdf')


if __name__ == '__main__':
    path = Path('../../data/v5')
    save = Path('../../output/plots/v5/tracks')
    save.mkdir(exist_ok=True, parents=True)

    with multiprocessing.Pool(processes=8) as pool:
        reader = DataReader(
            # config_path='../configs/reading/processed/gnn.yaml',
            # config_path='../configs/reading/v4/processed/gnn.yaml',
            config_path=path / 'gnn.yaml',
            base_dir=path
        )
        particles = pd.concat(
            pool.map(reconstruct_and_match_tracks, reader.read())
        )

    plot_tracks_technical(
        particles,
        save=save
    )
