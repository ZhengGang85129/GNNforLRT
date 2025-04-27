# -*- coding: utf-8 -*-

import numpy as np
import torch
from datetime import datetime
from matplotlib import pyplot as plt

# Plotter.
from ExaTrkXPlotting import Plotter, PlotConfig

# Include performance plots.
import ExaTrkXPlots.performance
import sys,os
if __name__ == '__main__':
    fig, ax = plt.subplots(2, 2, figsize=(8, 8), tight_layout=True)
    
    if len(sys.argv) != 2:
        raise RuntimeError('usage: python3 ./plt_performance.py <GNN_output_file>')
    
    gnn_output = sys.argv[1]
    if not os.path.isfile(gnn_output):
        raise RuntimeError(f'Check whether "{gnn_output}" is filename')
    data = torch.load(gnn_output)
    truth, score = data['truth'], data['score']
    print("truth: ", truth.shape)
    print("score: ", score.shape)

    truth = truth.cpu().numpy()
    score = score.cpu().numpy()
    
    # You can also precompute values and pass to plotter in data
    # to avoid multiple computation in each plot if many plots share same data.
    """
    import sklearn.metrics
    
    false_positive_rate, true_positive_rate, _ = sklearn.metrics.roc_curve(
        truth, 
        score
    )
    precision, recall, thresholds = sklearn.metrics.precision_recall_curve(
        truth,
        score
    )
    """

    Plotter(
        fig = fig, 
        plots={
            ax[0, 0]: PlotConfig(
                plot='exatrkx.performance.score_distribution'
            ),
            ax[0, 1]: PlotConfig(
                plot='exatrkx.performance.roc_curve'
            ),
            ax[1, 0]: PlotConfig(
                plot='exatrkx.performance.precision_recall_with_threshold'
            ),
            ax[1, 1]: PlotConfig(
                plot='exatrkx.performance.precision_recall'
            )
        },
        data={
            'truth': truth,
            'score': score
        }
    ).plot(save=f"metrics/TTbar_noPU_performance_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf")
