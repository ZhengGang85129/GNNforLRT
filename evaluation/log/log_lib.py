#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

import numpy as np
import pandas as pd

# Data source.
from ExaTrkXDataIO import DataReader

# Plotting tools.
import matplotlib.pyplot as plt
from ExaTrkXPlotting import Plotter, PlotConfig
from ExaTrkXPlots import hits, particles, pairs, train_logs


def plot_train_log(train_history, val_history, save=None):
    save = Path(save)

    save.mkdir(
        exist_ok=True, parents=True
    )

    print(
        'Purity', val_history['purity'].values[-1],
    )

    print(
        'Efficiency', val_history['efficiency'].values[-1],
    )

    # Request 1x2 figure for purity and efficiency plot.
    fig, ax = plt.subplots(
        1, 2, figsize=(16, 8), tight_layout=True
    )
    Plotter(
        fig, {
            ax[0]: PlotConfig(
                config='purity'
            ),
            ax[1]: PlotConfig(
                config='efficiency'
            )
        },
        data={
            'history': val_history
        },
        config='./log/plot.yaml',
    ).plot(save=save / 'pur_eff.pdf')

    Plotter(
        fig, {
            ax[0]: PlotConfig(
                config='purity'
            ),
            ax[1]: PlotConfig(
                config='efficiency'
            )
        },
        data={
            'history': val_history
        },
        config='./log/plot.yaml',
    ).plot(save=save / 'pur_eff.png')

    steps_per_epoch = int(
        len(train_history['loss']) / (len(val_history['loss'])-1)
    )

    fig, ax = plt.subplots(
        1, 1, figsize=(8, 8), tight_layout=True
    )
    Plotter(
        fig, {
            ax: [
                PlotConfig(
                    config='train_loss',
                    data={
                        'history': train_history
                    },
                    args={
                        'steps_per_epoch': steps_per_epoch
                    }
                ),
                PlotConfig(
                    config='val_loss',
                    data={
                        'history': val_history
                    }
                )
            ]
        },
        config='./log/plot.yaml'
    ).plot(save=save / 'loss.pdf')

    Plotter(
        fig, {
            ax: [
                PlotConfig(
                    config='train_loss',
                    data={
                        'history': train_history
                    },
                    args={
                        'steps_per_epoch': steps_per_epoch
                    }
                ),
                PlotConfig(
                    config='val_loss',
                    data={
                        'history': val_history
                    }
                )
            ]
        },
        config='./log/plot.yaml'
    ).plot(save=save / 'loss.png')

def plot_gnn_train_log(train_history, val_history, sig_history, save=None):
    save = Path(save)

    save.mkdir(
        exist_ok=True,
        parents=True
    )

    print(
        'Purity', sig_history['purity'].values[-1],
    )

    print(
        'Efficiency', sig_history['efficiency'].values[-1],
    )

    steps_per_epoch = int(
        len(train_history['loss']) / (len(val_history['loss']) - 1)
    )

    fig, ax = plt.subplots(
        1, 1, figsize=(8, 8), tight_layout=True
    )
    Plotter(
        fig, {
            ax: [
                PlotConfig(
                    config='train_loss',
                    data={
                        'history': train_history
                    },
                    args={
                        'steps_per_epoch': steps_per_epoch
                    }
                ),
                PlotConfig(
                    config='val_loss',
                    data={
                        'history': val_history
                    }
                )
            ]
        },
        config='./log/plot.yaml'
    ).plot(save=save / 'loss.pdf')

    # Request 1x2 figure for purity and efficiency plot.
    fig, ax = plt.subplots(
        1, 2, figsize=(16, 8), tight_layout=True
    )
    Plotter(
        fig, {
            ax[0]: PlotConfig(
                config='purity'
            ),
            ax[1]: PlotConfig(
                config='efficiency'
            )
        },
        data={
            'history': sig_history
        },
        config='./log/plot.yaml',
    ).plot(save=save / 'pur_eff.pdf')

    fig, ax = plt.subplots(
        1, 1, figsize=(8, 8), tight_layout=True
    )
    Plotter(
        fig, {
            ax: [
                PlotConfig(
                    plot='exatrkx.train_log',
                    data={
                        'history': sig_history
                    },
                    args={
                        'tag': 'auc',
                        'y_label': 'AUC',
                    }
                )
            ]
        },
        config='./log/plot.yaml'
    ).plot(save=save / 'auc.pdf')


if __name__ == '__main__':
    # Draw one log only.
    model = Path('../../data/models/v5/full')
    output = Path('../../output/plots/v5/full')
    log = DataReader(
        config_path=model/'log.yaml',
        base_dir=model
    ).read_all()[0]

    # Embedding.
    plot_train_log(
        log['embedding_train'],
        log['embedding_val'],
        save=output/'embedding'
    )

    # Filter.
    plot_train_log(
        log['filter_train'],
        log['filter_val'],
        save=output/'filter'
    )

    # GNN.
    plot_gnn_train_log(
        log['gnn_train'],
        log['gnn_val'],
        log['gnn_sig'],
        save=output/'gnn'/'0.3'
    )

    plot_gnn_train_log(
        log['gnn_train'],
        log['gnn_val'],
        log['gnn_tot'],
        '../../output/plots/v3/pileup40/GNN/tot'
    )
