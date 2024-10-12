#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import multiprocessing

import numpy as np
import pandas as pd

from ExaTrkXDataIO import DataReader

import matplotlib.pyplot as plt
from ExaTrkXPlotting import Plotter, PlotConfig
from ExaTrkXPlots import tracks

from track_reconstruction import reconstruct_tracks
from track_reconstruction.algorithm import DBSCANTrackReco
from track_matching import match_tracks

from workflows import reconstruct_and_match_tracks

import sys, os 

def plot_tracks(particles, save):
    track_filters = {
        'all': None,
        'dilepton': (
            particles.parent_ptype.abs() == 50
        ),
        'prompt': (
            particles.parent_ptype.abs() == 24
        )
    }

    bins = {
        'all': {
            'pt': np.arange(0, 10.5, 0.5),
            'eta': np.arange(-4.0, 4.4, 0.4),
            'd0': np.arange(0, 2500, step=50),
            'z0': np.arange(-5000, 5000, step=200)
        },
        'dilepton': {
            'pt': np.arange(0, 10.5, 0.5),
            'eta': np.arange(-4.0, 4.4, 0.4),
            'd0': np.arange(0, 2500, step=50),
            'z0': np.arange(-5000, 5000, step=200)
        },
        'prompt': {
            'pt': np.arange(0, 100, step=2.0),
            'eta': np.arange(-4.0, 4.4, 0.4),
            'd0': np.arange(0, 0.1, step=0.005),
            'z0': np.arange(-150.0, 150.0, step=10.0)
        }
    }

    for name, track_filter in track_filters.items():
        path = save / name
        path.mkdir(exist_ok=True, parents=True)

        if track_filter is not None:
            generated = particles[track_filter]
            reconstructable = particles[track_filter & particles.is_trackable]
            matched = particles[track_filter & particles.is_trackable & particles.is_matched]
        else:
            generated = particles
            reconstructable = particles[particles.is_trackable]
            matched = particles[particles.is_trackable & particles.is_matched]

        n_true = generated.shape[0]
        n_match = matched.shape[0]

        with open(f'{path}/summary.txt', 'w') as fp:
            fp.write(
                f'Truth tracks{n_true}\n'
                f'Match tracks{n_match}\n'
                f'Tracking Eff.: {n_match / max(n_true, 1e-20):>20.4f}\n'
            )

        # pt and eta.
        fig, ax = plt.subplots(2, 2, figsize=(8, 8), tight_layout=True)
        plotter = Plotter(fig)

        plotter.data = {
            'generated': generated,
            'reconstructable': reconstructable,
            'matched': matched,
        }

        plotter[ax[0, 0]] = PlotConfig(
            plot='exatrkx.tracks.distribution',
            args={
                'var_col': 'pt',
                'var_name': '$p_T$ [GeV]',
                'bins': bins[name]['pt']
            }
        )
        plotter[ax[0, 1]] = PlotConfig(
            plot='exatrkx.tracks.efficiency',
            args={
                'var_col': 'pt',
                'var_name': '$p_T$ [GeV]',
                'bins': bins[name]['pt']
            }
        )
        plotter[ax[1, 0]] = PlotConfig(
            plot='exatrkx.tracks.distribution',
            args={
                'var_col': 'eta',
                'var_name': r'$\eta$',
                'bins': bins[name]['eta']
            }
        )
        plotter[ax[1, 1]] = PlotConfig(
            plot='exatrkx.tracks.efficiency',
            args={
                'var_col': 'eta',
                'var_name': r'$\eta$',
                'bins': bins[name]['eta']
            }
        )

        # d0 and z0
        fig, ax = plt.subplots(2, 2, figsize=(8, 8), tight_layout=True)
        plotter.data = {
            'generated': generated,
            'reconstructable': reconstructable,
            'matched': matched,
        }

        plotter[ax[0, 0]] = PlotConfig(
            plot='exatrkx.tracks.distribution',
            args={
                'var_col': 'd0',
                'var_name': '$d_0$ [mm]',
                'bins': bins[name]['d0']
            }
        )
        plotter[ax[0, 1]] = PlotConfig(
            plot='exatrkx.tracks.efficiency',
            args={
                'var_col': 'd0',
                'var_name': '$d_0$ [mm]',
                'bins': bins[name]['d0']
            }
        )
        plotter[ax[1, 0]] = PlotConfig(
            plot='exatrkx.tracks.distribution',
            args={
                'var_col': 'z0',
                'var_name': r'$z_0$ [mm]',
                'bins': bins[name]['z0']
            }
        )
        plotter[ax[1, 1]] = PlotConfig(
            plot='exatrkx.tracks.efficiency',
            args={
                'var_col': 'z0',
                'var_name': r'$z_0$ [mm]',
                'bins': bins[name]['z0']
            }
        )

        plotter.plot(save=path / 'd0_z0.png')


def plot_pur_eff(
    epsilon_sample_points,
    efficiencies,
    purities,
    save=None
):
    plt.rcParams.update({'font.size': 12})

    # Request 1x2 figure for purity and efficiency plot.
    fig, ax = plt.subplots(
        1, 2, figsize=(16, 8), tight_layout=True
    )
    print(efficiencies)
    ax[0].plot(epsilon_sample_points, efficiencies)
    ax[0].set_xlabel(r'DBSCAN $\epsilon$')
    ax[0].set_ylabel('Efficiency')

    ax[1].plot(epsilon_sample_points, purities)
    ax[1].set_xlabel(r'DBSCAN $\epsilon$')
    ax[1].set_ylabel('Purity')

    if save is not None:
        plt.savefig(save/'DBSCAN_eff_pur.pdf')
    else:
        plt.show()


if __name__ == '__main__':
    
    if len(sys.argv) != 3:
        raise RuntimeError('usage: python3 ./track_reconstruction_DBSCAN_optimize_search.py <DBSCAN configuration file> <epsilon_point>')
    
    
    output_path = Path('../metrics-PU200/epsilon_scan')
    output_path.mkdir(exist_ok=True, parents=True)

    #epsilon_sample_points = [
    #    0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9
    #]
    epsilon = float(sys.argv[2].replace('p', '.'))

    efficiencies = []
    purities = []
    #for epsilon in epsilon_sample_points:
    print(f'================= epsilon: {epsilon} ===================')
    all_particles = []

    reader = DataReader(
        config_path=sys.argv[1],
        base_dir=os.path.basename(sys.argv[1])
    )

    with multiprocessing.Pool(processes=8) as pool:
        data = reader.read_all()
        results = pool.starmap(
            reconstruct_and_match_tracks,
            zip(
                data,
                [epsilon]*len(data),
                [True]*len(data)
            )
        )

        all_particles.append(pd.concat([
            r[0] for r in results
        ]))

        n_true = sum([
            r[1]['n_true'] for r in results
        ])

        n_reco = sum([
            r[1]['n_reco'] for r in results
        ])

        n_match = sum([
            r[1]['n_match'] for r in results
        ])

    efficiency = n_match / n_true
    purity = n_match / n_reco

    # Create output directory.
    path = output_path / f'epsilon_{epsilon:.2f}'
    path.mkdir(parents=True, exist_ok=True)

    # Output summary.
    with open(path / 'summary.txt', 'w') as fp:
        fp.write(
            f'===epsilon_{epsilon}===\n'
            f'Truth tracks: {n_true:>20}\n'
            f'Reco. tracks: {n_reco:>20}\n'
            f'Reco. tracks Matched: {n_match:>20}\n'
            f'Tracking Eff.: {efficiency:>20.4f}\n'
            f'Tracking Pur.: {purity:>20.4f}\n'
        )

    plot_tracks(pd.concat(all_particles), path)

        #purities.append(purity)
        #efficiencies.append(efficiency)

    #plot_pur_eff(
    #    epsilon_sample_points,
    #    efficiencies,
    #    purities,
    #    output_path
    #)
